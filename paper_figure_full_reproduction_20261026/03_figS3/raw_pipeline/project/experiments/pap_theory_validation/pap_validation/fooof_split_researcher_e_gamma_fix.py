from __future__ import annotations

"""Correct panel E using the other researcher's OU/Welch simulation regime."""

import json
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

import fooof_split_attached_code_theory as base
import fooof_split_researcher_pipeline_matched as matched
import reproduce_gamma_artifact as researcher


ROOT = Path(__file__).resolve().parents[3]
DATA = Path("/PORTABLE_HOME/Desktop/data")
OLD = ROOT / "output" / "fooof_split_researcher_pipeline_matched_2026-09-01"
OUT = ROOT / "output" / "fooof_split_researcher_pipeline_gamma_corrected_2026-09-01"
SEED = 20260901 + 8800
N_GROUPS = 20
N_TRAIN = 120
N_HELDOUT = 120
FRACTAL_NOISE_SCALE = 0.10

BASE_SIGNAL: np.ndarray | None = None
BASE_FREQ: np.ndarray | None = None
BASE_PSD: np.ndarray | None = None
BASE_LOG_PSD: np.ndarray | None = None
FRACTAL_RESIDUALS: np.ndarray | None = None


def load_sources() -> None:
    global BASE_SIGNAL, BASE_FREQ, BASE_PSD, BASE_LOG_PSD, FRACTAL_RESIDUALS
    with h5py.File(DATA / "simulate_data" / "x_with_oscillation.mat", "r") as handle:
        BASE_SIGNAL = np.asarray(handle["x_base"]).squeeze()
    BASE_FREQ, BASE_PSD = signal.welch(
        BASE_SIGNAL,
        fs=researcher.FS,
        window="hamming",
        nperseg=len(BASE_SIGNAL),
        noverlap=len(BASE_SIGNAL) // 2,
        nfft=32768,
        detrend="constant",
    )
    BASE_LOG_PSD = np.interp(base.FREQ, BASE_FREQ, np.log10(BASE_PSD))
    with h5py.File(DATA / "simulate_data" / "irasa_data.mat", "r") as handle:
        frequency = np.asarray(handle["f"]).squeeze()
        fractal = np.log10(np.asarray(handle["fractal_powspectra"]).T)
    mask = (frequency >= 0.2) & (frequency <= 200.0)
    frequency = frequency[mask]
    fractal = fractal[:, mask]
    interpolated = np.asarray(
        [np.interp(base.FREQ, frequency, row) for row in fractal]
    )
    FRACTAL_RESIDUALS = interpolated - interpolated.mean(axis=0, keepdims=True)


def generate_batch(
    rng: np.random.Generator, band: str, n_trials: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assert BASE_SIGNAL is not None and BASE_FREQ is not None and BASE_PSD is not None
    assert BASE_LOG_PSD is not None and FRACTAL_RESIDUALS is not None
    lo, hi, _ = researcher.BANDS[band]
    centers = rng.uniform(lo, hi, n_trials)
    widths = rng.uniform(1.5, 6.0, n_trials)
    fractions = rng.uniform(2.8, 3.8, n_trials)
    original = []
    for center, width, fraction in zip(centers, widths, fractions):
        frequency, log_psd = researcher.generate_trial_psd(
            rng,
            BASE_SIGNAL,
            BASE_FREQ,
            BASE_PSD,
            float(center),
            float(width),
            float(fraction),
        )
        original.append(np.interp(base.FREQ, frequency, log_psd))
    residual_index = rng.integers(0, len(FRACTAL_RESIDUALS), n_trials)
    fractal = (
        BASE_LOG_PSD[None, :]
        + FRACTAL_NOISE_SCALE * FRACTAL_RESIDUALS[residual_index]
    )
    return np.asarray(original), np.asarray(fractal), centers


def sequential_influence(y: np.ndarray, peak_range: tuple[float, float]) -> np.ndarray:
    """Active-set Jacobian [peak center, final exponent] wrt log PSD."""
    stages = matched.sequential_stages(y, peak_range)
    n = len(base.FREQ)
    selection = np.flatnonzero(stages["baseline"])
    robust_design = matched._ap_design(base.FREQ[selection], stages["robust"])
    robust_map = np.zeros((3, n))
    robust_map[:, selection] = np.linalg.pinv(robust_design)
    robust_full = matched._ap_design(base.FREQ, stages["robust"])
    peak_design = matched._peak_design(base.FREQ, stages["peak"])
    peak_inverse = np.linalg.pinv(peak_design)
    peak_map = peak_inverse - (peak_inverse @ robust_full) @ robust_map
    final_design = matched._ap_design(base.FREQ, stages["final"])
    final_inverse = np.linalg.pinv(final_design)
    final_map = final_inverse - (final_inverse @ peak_design) @ peak_map
    return np.vstack([peak_map[0], final_map[1]])


def split_projected_errors(
    original: np.ndarray, fractal: np.ndarray, band_key: str
) -> np.ndarray:
    mask = base.band_mask(band_key)
    fit_frequency = base.FREQ[mask]
    mean_residual = (original - fractal).mean(axis=0)[mask]
    center0 = float(fit_frequency[np.argmax(mean_residual)])
    x0 = np.array([center0, max(float(mean_residual.max()), 0.60), 1.5])
    peak = optimize.least_squares(
        lambda x: mean_residual - base.gaussian(fit_frequency, *x),
        x0,
        bounds=([fit_frequency[0], 0.0, 0.75], [fit_frequency[-1], 3.0, 3.0]),
        max_nfev=250,
    ).x
    center, height, sigma = peak
    g = np.exp(-0.5 * ((fit_frequency - center) / sigma) ** 2)
    peak_design = np.column_stack(
        [
            height * g * (fit_frequency - center) / sigma**2,
            g,
            height * g * (fit_frequency - center) ** 2 / sigma**3,
        ]
    )
    center_weight = np.linalg.pinv(peak_design)[0]
    ap_design = np.column_stack([np.ones(len(base.FREQ)), -np.log10(base.FREQ)])
    exponent_weight = np.linalg.pinv(ap_design)[1]
    residual_delta = (original - fractal) - (original - fractal).mean(axis=0)
    fractal_delta = fractal - fractal.mean(axis=0)
    return np.column_stack(
        [residual_delta[:, mask] @ center_weight, fractal_delta @ exponent_weight]
    )


def correlation(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if good.sum() < 8:
        return np.nan
    return float(stats.spearmanr(values[good, 0], values[good, 1]).statistic)


def group_worker(repeat: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for band_index, band in enumerate(researcher.BANDS):
        band_key = band.lower()
        peak_range = researcher.BANDS[band][2]
        train_rng = np.random.default_rng(SEED + repeat * 100 + band_index * 10)
        held_rng = np.random.default_rng(SEED + repeat * 100 + band_index * 10 + 1)
        train_o, train_f, _ = generate_batch(train_rng, band, N_TRAIN)
        held_o, held_f, _ = generate_batch(held_rng, band, N_HELDOUT)

        influence = sequential_influence(train_o.mean(axis=0), peak_range)
        projected = (train_o - train_o.mean(axis=0)) @ influence.T
        fooof_theory = correlation(projected)
        fooof = np.asarray(
            [
                researcher.fit_sequential_knee_fooof(row, base.FREQ, peak_range)
                for row in held_o
            ]
        )
        split_theory = correlation(split_projected_errors(train_o, train_f, band_key))
        split = np.asarray(
            [
                base.fit_split_local(row_o, row_f, base.band_mask(band_key))
                for row_o, row_f in zip(held_o, held_f)
            ]
        )
        rows.append(
            {
                "repeat": repeat,
                "band": band_key,
                "fooof_pred": fooof_theory,
                "fooof_emp": correlation(fooof),
                "split_pred": split_theory,
                "split_emp": correlation(split),
                "fooof_valid": float(np.isfinite(fooof).all(axis=1).mean()),
                "split_valid": float(np.isfinite(split).all(axis=1).mean()),
                "n_train": N_TRAIN,
                "n_heldout": N_HELDOUT,
            }
        )
    return rows


def aggregate_groups(groups: pd.DataFrame) -> pd.DataFrame:
    centers = {"alpha": 10.0, "beta": 21.5, "low gamma": 45.0, "high gamma": 105.0}
    rows = []
    for band in ["alpha", "beta", "low gamma", "high gamma"]:
        data = groups[groups.band == band]
        rows.append(
            {
                "center": centers[band],
                "band": band,
                "fooof_pred": float(data.fooof_pred.mean()),
                "fooof_emp": float(data.fooof_emp.mean()),
                "fooof_lo": float(data.fooof_emp.quantile(0.025)),
                "fooof_hi": float(data.fooof_emp.quantile(0.975)),
                "split_pred": float(data.split_pred.mean()),
                "split_emp": float(data.split_emp.mean()),
                "split_lo": float(data.split_emp.quantile(0.025)),
                "split_hi": float(data.split_emp.quantile(0.975)),
                "n_fooof": int(round(N_HELDOUT * data.fooof_valid.mean())),
                "n_split": int(round(N_HELDOUT * data.split_valid.mean())),
                "n_groups": N_GROUPS,
                "n_trials": N_HELDOUT,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    load_sources()
    matched.install_variant()
    base.OUT = OUT
    base.style()
    OUT.mkdir(parents=True, exist_ok=True)

    with mp.get_context("fork").Pool(4) as pool:
        nested = pool.map(group_worker, range(N_GROUPS))
    groups = pd.DataFrame([row for part in nested for row in part])
    sweep = aggregate_groups(groups)

    cond = base.conditional_experiment(np.random.default_rng(matched.SEED), n=1400)
    calibration = pd.read_csv(OLD / "uniform_theory_calibration_conditions.csv")
    null = pd.read_csv(OLD / "null_dataset_results.csv")
    base.plot_six_panel(cond, calibration, sweep, null, pd.DataFrame())
    explicit = OUT / "figure2_theory_validation_nine_panels_researcher_fooof_gamma_corrected.svg"
    (OUT / "figure2_theory_validation_nine_panels.svg").replace(explicit)
    preview = OUT / "figure2_theory_validation_nine_panels_researcher_fooof_gamma_corrected_preview.png"
    (OUT / "figure2_theory_validation_nine_panels_preview.png").replace(preview)

    groups.to_csv(OUT / "panel_e_ou_welch_group_results.csv", index=False)
    sweep.to_csv(OUT / "panel_e_ou_welch_summary.csv", index=False)
    metrics = {
        "figure": str(explicit),
        "diagnosis": "Previous E fixed the true center and used a smooth no-knee log-PSD noise model.",
        "correction": "30 s OU/Welch base with fitted knee near 5.6 Hz; center, bandwidth and power vary within each band.",
        "groups": N_GROUPS,
        "training_trials_per_group": N_TRAIN,
        "heldout_trials_per_group": N_HELDOUT,
        "panel_e": sweep.to_dict(orient="records"),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
