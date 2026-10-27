from __future__ import annotations

"""Re-run the other researcher's FOOOF 1.1.1 settings in four bands.

The spectra use the provided 30-s OU signal and the time-domain oscillation /
Hamming-Welch generator.  Only the oscillation center range changes by band;
bandwidth and peak/base power ratio remain those in a2_add_oscillation.m.
"""

import json
import multiprocessing as mp
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import io, stats

import fooof_split_attached_code_theory as base
import fooof_split_researcher_e_gamma_fix as fixed
import fooof_split_researcher_pipeline_matched as matched
import reproduce_gamma_artifact as researcher


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "researcher_official_fooof_band_rerun_2026-09-01"
FOOOF_PACKAGE = Path("/private/tmp/fooof_pkg")
N_GROUPS = 20
N_TRIALS = 120
SEED = fixed.SEED + 24000


def _fooof_group_class():
    if str(FOOOF_PACKAGE) not in sys.path:
        sys.path.insert(0, str(FOOOF_PACKAGE))
    warnings.filterwarnings("ignore", message=".*fooof.*deprecated.*")
    from fooof import FOOOFGroup

    return FOOOFGroup


def correlation(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 8:
        return np.nan, np.nan, int(good.sum())
    test = stats.spearmanr(x[good], y[good])
    return float(test.statistic), float(test.pvalue), int(good.sum())


def official_fooof(
    original_log10: np.ndarray, display_band: str, max_n_peaks: int = 3
) -> np.ndarray:
    """Exact public API and options in human_data/a_fooof_analysis.py."""
    FOOOFGroup = _fooof_group_class()
    fit = (base.FREQ >= 0.2) & (base.FREQ <= 197.0)
    fg = FOOOFGroup(
        aperiodic_mode="knee",
        max_n_peaks=max_n_peaks,
        peak_width_limits=[1.5, 6.0],
        min_peak_height=0.6,
        peak_threshold=2.0,
        verbose=False,
    )
    fg.fit(base.FREQ[fit], 10.0 ** original_log10[:, fit], [0.2, 197.0])
    exponents = np.asarray(fg.get_params("aperiodic_params", "exponent"), dtype=float)
    lo, hi, _ = researcher.BANDS[display_band]
    centers = np.full(len(original_log10), np.nan)
    for index in range(len(original_log10)):
        peaks = np.asarray(fg.get_fooof(index, regenerate=True).peak_params_, dtype=float)
        if peaks.size == 0:
            continue
        peaks = np.atleast_2d(peaks)
        candidates = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)]
        if len(candidates):
            centers[index] = candidates[np.argmax(candidates[:, 1]), 0]
    return np.column_stack([centers, exponents])


def worker(group: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for band_index, display_band in enumerate(researcher.BANDS):
        band = display_band.lower()
        rng = np.random.default_rng(SEED + group * 100 + band_index * 10)
        original, fractal, true_centers = fixed.generate_batch(rng, display_band, N_TRIALS)
        fooof = official_fooof(original, display_band, max_n_peaks=3)
        fooof_one_peak = official_fooof(original, display_band, max_n_peaks=1)
        split = np.asarray(
            [
                base.fit_split_local(row_o, row_f, base.band_mask(band))
                for row_o, row_f in zip(original, fractal)
            ]
        )
        fr, fp, fn = correlation(fooof[:, 0], fooof[:, 1])
        f1r, f1p, f1n = correlation(fooof_one_peak[:, 0], fooof_one_peak[:, 1])
        sr, sp, sn = correlation(split[:, 0], split[:, 1])
        true_fr, true_fp, _ = correlation(true_centers, fooof[:, 1])
        rows.append(
            {
                "group": group,
                "band": band,
                "fooof_rho": fr,
                "fooof_p": fp,
                "fooof_n": fn,
                "fooof_one_peak_rho": f1r,
                "fooof_one_peak_p": f1p,
                "fooof_one_peak_n": f1n,
                "split_rho": sr,
                "split_p": sp,
                "split_n": sn,
                "true_center_vs_fooof_exponent_rho": true_fr,
                "true_center_vs_fooof_exponent_p": true_fp,
                "n_trials": N_TRIALS,
            }
        )
    return rows


def saved_researcher_result() -> dict[str, float | int]:
    root = Path("/PORTABLE_HOME/Desktop/data/simulate_data")
    fooof = io.loadmat(root / "fooof_result.mat", squeeze_me=True)
    fr, fp, fn = correlation(fooof["peak_cf"], fooof["aperiodic_exponent"])
    split_ap = io.loadmat(root / "split_exponent.mat", squeeze_me=True, struct_as_record=False)
    split_p = io.loadmat(root / "split_oscillation.mat", squeeze_me=True, struct_as_record=False)

    def scalar(value: object) -> float:
        array = np.asarray(value).squeeze()
        return float(array) if array.size == 1 else np.nan

    def peak(value: object) -> float:
        array = np.asarray(value).squeeze()
        return float(array.flat[0]) if array.size and np.any(array != 0) else np.nan

    exponents = np.asarray([scalar(value) for value in np.atleast_1d(split_ap["exponent_sum"])])
    centers = np.asarray([peak(value) for value in np.atleast_1d(split_p["peak_sum"])])
    sr, sp, sn = correlation(centers, exponents)
    return {
        "center_range_hz": [30, 150],
        "fooof_rho": fr,
        "fooof_p": fp,
        "fooof_n": fn,
        "split_rho": sr,
        "split_p": sp,
        "split_n": sn,
    }


def summarise(groups: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for band in ["alpha", "beta", "low gamma", "high gamma"]:
        data = groups[groups.band == band]
        rows.append(
            {
                "band": band,
                "fooof_mean_rho": float(data.fooof_rho.mean()),
                "fooof_sd_rho": float(data.fooof_rho.std(ddof=1)),
                "fooof_ci_low": float(data.fooof_rho.quantile(0.025)),
                "fooof_ci_high": float(data.fooof_rho.quantile(0.975)),
                "fooof_significant_groups": int((data.fooof_p < 0.05).sum()),
                "fooof_one_peak_mean_rho": float(data.fooof_one_peak_rho.mean()),
                "fooof_one_peak_sd_rho": float(data.fooof_one_peak_rho.std(ddof=1)),
                "fooof_one_peak_significant_groups": int(
                    (data.fooof_one_peak_p < 0.05).sum()
                ),
                "split_mean_rho": float(data.split_rho.mean()),
                "split_sd_rho": float(data.split_rho.std(ddof=1)),
                "split_ci_low": float(data.split_rho.quantile(0.025)),
                "split_ci_high": float(data.split_rho.quantile(0.975)),
                "split_significant_groups": int((data.split_p < 0.05).sum()),
                "true_center_vs_fooof_exponent_mean_rho": float(
                    data.true_center_vs_fooof_exponent_rho.mean()
                ),
                "n_groups": N_GROUPS,
                "trials_per_group": N_TRIALS,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed.load_sources()
    matched.install_variant()
    _fooof_group_class()
    with mp.get_context("fork").Pool(4) as pool:
        nested = pool.map(worker, range(N_GROUPS))
    groups = pd.DataFrame([row for part in nested for row in part])
    summary = summarise(groups)
    saved = saved_researcher_result()
    groups.to_csv(OUT / "official_fooof_four_band_groups.csv", index=False)
    summary.to_csv(OUT / "official_fooof_four_band_summary.csv", index=False)
    result = {
        "method": "FOOOF 1.1.1; knee; max_n_peaks=3; width=1.5-6 Hz; min_height=0.6; threshold=2; fit=0.2-197 Hz",
        "spectra": "provided 30-s OU base; time-domain bandpass oscillation; Hamming-Welch; center/width/power vary",
        "saved_original_30_150_hz": saved,
        "four_band_summary": summary.to_dict(orient="records"),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
