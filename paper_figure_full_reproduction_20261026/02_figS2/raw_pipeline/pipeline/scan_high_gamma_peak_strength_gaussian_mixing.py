from __future__ import annotations

"""Scan high-gamma input peak strength under frozen Gaussian covariance mixing."""

import json
import multiprocessing as mp
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


WORKSPACE = Path(__file__).resolve().parent
PROJECT_CODE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project/experiments/pap_theory_validation/pap_validation"
for source_path in [WORKSPACE, PROJECT_CODE]:
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

import redraw_fooof_irasa_split_3x3 as shared  # noqa: E402


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_high_gamma_strength_scan_2026-09-13"
)
DISPLAY_BAND = "High gamma"
BAND = "high gamma"
N_REPEATS = 6
N_TRAIN = shared.frozen.N_TRAIN
N_HELDOUT = shared.frozen.N_HELDOUT
N_MC = shared.frozen.N_MC
BASE_SEED = shared.frozen.BASE_SEED + 4_000_000
HIGH_GAMMA_INDEX = list(shared.researcher.BANDS).index(DISPLAY_BAND)
STRENGTH_RANGES = [
    (2.8, 3.8),
    (3.4, 4.4),
    (4.0, 5.0),
    (4.6, 5.6),
    (5.2, 6.2),
    (6.0, 7.0),
    (7.0, 8.0),
]


def high_gamma_batch(
    rng: np.random.Generator,
    n_trials: int,
    strength_low: float,
    strength_high: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate the frozen common s(t), changing only peak-strength sampling."""
    assert shared.fixed.FRACTAL_RESIDUALS is not None
    low, high, _ = shared.researcher.BANDS[DISPLAY_BAND]
    centers = rng.uniform(low, high, n_trials)
    widths = rng.uniform(1.5, 6.0, n_trials)
    strengths = rng.uniform(strength_low, strength_high, n_trials)
    knees = shared.frozen.sample_empirical_knees_independent_of_centers(
        rng, centers
    )

    # Preserve the frozen random stream exactly apart from the values drawn
    # from the explicitly varied uniform peak-strength interval.
    rng.integers(0, len(shared.fixed.FRACTAL_RESIDUALS), n_trials)
    signals: list[np.ndarray] = []
    original: list[np.ndarray] = []
    for center, width, strength, knee in zip(
        centers, widths, strengths, knees
    ):
        composite, frequency, log_psd = shared.composite_trial(
            rng,
            float(center),
            float(width),
            float(strength),
            float(knee),
        )
        signals.append(composite)
        original.append(
            np.interp(shared.frozen.base.FREQ, frequency, log_psd)
        )
    irasa_frequency, irasa_aperiodic = shared.irasa_aperiodic_batch(
        np.asarray(signals)
    )
    irasa_log = np.log10(
        np.maximum(irasa_aperiodic, np.finfo(float).tiny)
    )
    fractal = np.asarray(
        [
            np.interp(shared.frozen.base.FREQ, irasa_frequency, row)
            for row in irasa_log
        ]
    )
    return np.asarray(original), fractal, centers, knees, strengths


def valid_corr(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if good.sum() < 8:
        return np.nan
    return float(stats.spearmanr(values[good, 0], values[good, 1]).statistic)


def worker(task: tuple[int, int]) -> dict[str, float | int]:
    range_index, repeat = task
    strength_low, strength_high = STRENGTH_RANGES[range_index]
    seed = BASE_SEED + repeat * 100 + HIGH_GAMMA_INDEX * 10
    train = high_gamma_batch(
        np.random.default_rng(seed), N_TRAIN, strength_low, strength_high
    )
    held = high_gamma_batch(
        np.random.default_rng(seed + 1),
        N_HELDOUT,
        strength_low,
        strength_high,
    )
    train_o, train_f, train_c, train_k, train_s = train
    held_o, held_f, held_c, held_k, held_s = held
    mc_o, mc_f = shared.frozen.covariance_mc_spectra(
        np.random.default_rng(seed + 2), train_o, train_f, n_mc=N_MC
    )
    fooof_mc = shared.frozen.official_fit(mc_o, DISPLAY_BAND)
    split_mc = shared.six.fit_split_batch(mc_o, mc_f, BAND)
    fooof_held = shared.frozen.official_fit(held_o, DISPLAY_BAND)
    split_held = shared.six.fit_split_batch(held_o, held_f, BAND)
    fooof_mc_good = np.isfinite(fooof_mc).all(axis=1)
    split_mc_good = np.isfinite(split_mc).all(axis=1)
    fooof_held_good = np.isfinite(fooof_held).all(axis=1)
    split_held_good = np.isfinite(split_held).all(axis=1)
    return {
        "range_index": range_index,
        "repeat": repeat,
        "strength_low": strength_low,
        "strength_high": strength_high,
        "strength_mean": float(np.mean(train_s)),
        "fooof_pred": valid_corr(fooof_mc),
        "split_pred": valid_corr(split_mc),
        "fooof_emp": valid_corr(fooof_held),
        "split_emp": valid_corr(split_held),
        "fooof_mc_valid_n": int(fooof_mc_good.sum()),
        "split_mc_valid_n": int(split_mc_good.sum()),
        "fooof_held_valid_n": int(fooof_held_good.sum()),
        "split_held_valid_n": int(split_held_good.sum()),
        "fooof_mc_valid_rate": float(fooof_mc_good.mean()),
        "split_mc_valid_rate": float(split_mc_good.mean()),
        "fooof_held_valid_rate": float(fooof_held_good.mean()),
        "split_held_valid_rate": float(split_held_good.mean()),
        "fooof_abs_error": float(abs(valid_corr(fooof_mc) - valid_corr(fooof_held))),
        "split_abs_error": float(abs(valid_corr(split_mc) - valid_corr(split_held))),
        "empirical_method_separation": float(
            abs(valid_corr(fooof_held) - valid_corr(split_held))
        ),
        "true_center_knee_rho_train": float(
            stats.spearmanr(train_c, train_k).statistic
        ),
        "true_center_knee_rho_held": float(
            stats.spearmanr(held_c, held_k).statistic
        ),
    }


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (low, high), group in frame.groupby(
        ["strength_low", "strength_high"], sort=True
    ):
        rows.append(
            {
                "strength_low": low,
                "strength_high": high,
                "strength_midpoint": (low + high) / 2.0,
                "n_repeats": len(group),
                "fooof_mc_detection": group.fooof_mc_valid_rate.mean(),
                "split_mc_detection": group.split_mc_valid_rate.mean(),
                "fooof_held_detection": group.fooof_held_valid_rate.mean(),
                "split_held_detection": group.split_held_valid_rate.mean(),
                "fooof_theory_rho": group.fooof_pred.mean(),
                "fooof_held_rho": group.fooof_emp.mean(),
                "split_theory_rho": group.split_pred.mean(),
                "split_held_rho": group.split_emp.mean(),
                "fooof_mean_abs_error": group.fooof_abs_error.mean(),
                "split_mean_abs_error": group.split_abs_error.mean(),
                "empirical_method_separation": group.empirical_method_separation.mean(),
                "finite_split_theory_groups": int(group.split_pred.notna().sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shared.fixed.load_sources()
    shared.matched.install_variant()
    shared.frozen.official._fooof_group_class()
    shared.frozen.load_empirical_knees()
    tasks = [
        (range_index, repeat)
        for range_index in range(len(STRENGTH_RANGES))
        for repeat in range(N_REPEATS)
    ]
    rows: list[dict[str, float | int]] = []
    with mp.get_context("fork").Pool(4) as pool:
        for completed, row in enumerate(
            pool.imap_unordered(worker, tasks), start=1
        ):
            rows.append(row)
            pd.DataFrame(rows).sort_values(
                ["range_index", "repeat"]
            ).to_csv(OUTPUT / "scan_checkpoint.csv", index=False)
            print(f"completed scan tasks: {completed}/{len(tasks)}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["range_index", "repeat"])
    summary = summarize(frame)
    frame.to_csv(OUTPUT / "scan_all_repeats.csv", index=False)
    summary.to_csv(OUTPUT / "scan_summary.csv", index=False)
    manifest = {
        "band": DISPLAY_BAND,
        "peak_strength_definition": (
            "target oscillatory-to-local-OU-background power ratio at the "
            "sampled center frequency"
        ),
        "ranges": STRENGTH_RANGES,
        "n_repeats": N_REPEATS,
        "n_train": N_TRAIN,
        "n_heldout": N_HELDOUT,
        "n_gaussian_covariance_mc": N_MC,
        "all_other_sampling_and_estimator_settings": "frozen",
    }
    (OUTPUT / "scan_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
