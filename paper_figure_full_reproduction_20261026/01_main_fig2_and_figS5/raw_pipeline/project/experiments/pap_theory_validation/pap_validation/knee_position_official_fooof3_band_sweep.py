from __future__ import annotations

"""Test knee-position effects on p--ap correlation with official FOOOF max=3.

Across knee conditions, each repeat uses common random innovations, identical
oscillation centers / widths / power fractions, and identical band-pass noise.
This paired design isolates the aperiodic knee position while retaining the
other researcher's 30-s time-domain oscillation and Hamming-Welch workflow.
"""

import json
import multiprocessing as mp
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats

import fooof_split_attached_code_theory as base
import reproduce_gamma_artifact as researcher


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "knee_position_official_fooof3_band_sweep_2026-09-01"
FOOOF_PACKAGE = Path("/private/tmp/fooof_pkg")
FS = 1000
T = 30.0
N = int(FS * T)
KNEES = np.asarray([2.0, 5.0, 10.0, 20.0, 40.0, 80.0])
N_REPEATS = 12
N_TRIALS = 100
SEED = 20260901 + 55000

BAND_ORDER = ["Alpha", "Beta", "Low gamma", "High gamma"]
BAND_COLORS = {
    "alpha": "#E9A03B",
    "beta": "#E76F51",
    "low gamma": "#4C78A8",
    "high gamma": "#343A69",
}


def fooof_group_class():
    if str(FOOOF_PACKAGE) not in sys.path:
        sys.path.insert(0, str(FOOOF_PACKAGE))
    warnings.filterwarnings("ignore", message=".*fooof.*deprecated.*")
    from fooof import FOOOFGroup

    return FOOOFGroup


def ou_from_innovations(innovations: np.ndarray, knee_hz: float) -> np.ndarray:
    """Euler OU update used in a1_simulate_onetau.m, implemented as an AR(1)."""
    dt = 1.0 / FS
    tau = 1.0 / (2.0 * np.pi * knee_hz)
    coefficient = 1.0 - dt / tau
    # MATLAB update: x[k+1] = coefficient*x[k] + sqrt(dt)*z[k].
    tail = signal.lfilter([np.sqrt(dt)], [1.0, -coefficient], innovations[:-1])
    return np.r_[0.0, tail]


def base_welch(base_signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return signal.welch(
        base_signal,
        fs=FS,
        window="hamming",
        nperseg=len(base_signal),
        noverlap=len(base_signal) // 2,
        nfft=32768,
        detrend="constant",
    )


def generate_spectra(
    base_signal: np.ndarray,
    base_freq: np.ndarray,
    base_psd: np.ndarray,
    display_band: str,
    repeat: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate matched oscillations; the RNG excludes knee by design."""
    band_index = BAND_ORDER.index(display_band)
    rng = np.random.default_rng(SEED + repeat * 100 + band_index * 10)
    lo, hi, _ = researcher.BANDS[display_band]
    centers = rng.uniform(lo, hi, N_TRIALS)
    widths = rng.uniform(1.5, 6.0, N_TRIALS)
    fractions = rng.uniform(2.8, 3.8, N_TRIALS)
    rows = []
    for center, width, fraction in zip(centers, widths, fractions):
        frequency, log_psd = researcher.generate_trial_psd(
            rng,
            base_signal,
            base_freq,
            base_psd,
            float(center),
            float(width),
            float(fraction),
        )
        rows.append(np.interp(base.FREQ, frequency, log_psd))
    return np.asarray(rows), centers


def fit_official_fooof3(
    spectra_log10: np.ndarray, display_band: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    FOOOFGroup = fooof_group_class()
    fit_mask = (base.FREQ >= 0.2) & (base.FREQ <= 197.0)
    fg = FOOOFGroup(
        aperiodic_mode="knee",
        max_n_peaks=3,
        peak_width_limits=[1.5, 6.0],
        min_peak_height=0.6,
        peak_threshold=2.0,
        verbose=False,
    )
    fg.fit(base.FREQ[fit_mask], 10.0 ** spectra_log10[:, fit_mask], [0.2, 197.0])
    ap = np.asarray(fg.get_params("aperiodic_params"), dtype=float)
    exponents = ap[:, 2]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        fitted_knee_hz = np.where(ap[:, 1] > 0, ap[:, 1] ** (1.0 / exponents), np.nan)
    lo, hi, _ = researcher.BANDS[display_band]
    centers = np.full(len(spectra_log10), np.nan)
    for index in range(len(spectra_log10)):
        peaks = np.asarray(fg.get_fooof(index, regenerate=True).peak_params_, dtype=float)
        if peaks.size == 0:
            continue
        peaks = np.atleast_2d(peaks)
        candidates = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)]
        if len(candidates):
            centers[index] = candidates[np.argmax(candidates[:, 1]), 0]
    return centers, exponents, fitted_knee_hz


def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 8:
        return np.nan, np.nan, int(good.sum())
    test = stats.spearmanr(x[good], y[good])
    return float(test.statistic), float(test.pvalue), int(good.sum())


def repeat_worker(repeat: int) -> list[dict[str, float | int | str]]:
    innovation_rng = np.random.default_rng(SEED + repeat * 10000 + 777)
    innovations = innovation_rng.normal(size=N)
    rows: list[dict[str, float | int | str]] = []
    for knee_hz in KNEES:
        base_signal = ou_from_innovations(innovations, float(knee_hz))
        base_freq, base_psd = base_welch(base_signal)
        for display_band in BAND_ORDER:
            spectra, true_centers = generate_spectra(
                base_signal, base_freq, base_psd, display_band, repeat
            )
            estimated_centers, exponents, fitted_knees = fit_official_fooof3(
                spectra, display_band
            )
            rho, pvalue, valid = spearman(estimated_centers, exponents)
            true_rho, true_p, _ = spearman(true_centers, exponents)
            knee_rho, knee_p, _ = spearman(estimated_centers, fitted_knees)
            good_knee = np.isfinite(fitted_knees)
            rows.append(
                {
                    "repeat": repeat,
                    "true_knee_hz": float(knee_hz),
                    "band": display_band.lower(),
                    "fooof_rho": rho,
                    "fooof_p": pvalue,
                    "valid_n": valid,
                    "true_center_vs_exponent_rho": true_rho,
                    "true_center_vs_exponent_p": true_p,
                    "estimated_center_vs_fitted_knee_rho": knee_rho,
                    "estimated_center_vs_fitted_knee_p": knee_p,
                    "median_fitted_knee_hz": float(np.nanmedian(fitted_knees)),
                    "fitted_knee_valid_fraction": float(good_knee.mean()),
                    "n_trials": N_TRIALS,
                }
            )
    return rows


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def statistics_tables(groups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    omnibus_rows = []
    pairwise_rows = []
    for band in [item.lower() for item in BAND_ORDER]:
        data = groups[groups.band == band]
        arrays = [
            data[data.true_knee_hz == knee].sort_values("repeat").fooof_rho.to_numpy()
            for knee in KNEES
        ]
        test = stats.friedmanchisquare(*arrays)
        kendall_w = float(test.statistic / (N_REPEATS * (len(KNEES) - 1)))
        omnibus_rows.append(
            {
                "band": band,
                "friedman_chi2": float(test.statistic),
                "df": len(KNEES) - 1,
                "pvalue": float(test.pvalue),
                "kendall_w": kendall_w,
            }
        )
        baseline = arrays[list(KNEES).index(5.0)]
        for knee, values in zip(KNEES, arrays):
            if knee == 5.0:
                continue
            test_pair = stats.wilcoxon(values, baseline, alternative="two-sided")
            pairwise_rows.append(
                {
                    "band": band,
                    "comparison": f"{knee:g} vs 5 Hz",
                    "knee_hz": float(knee),
                    "median_difference": float(np.median(values - baseline)),
                    "wilcoxon_statistic": float(test_pair.statistic),
                    "pvalue": float(test_pair.pvalue),
                }
            )
    omnibus = pd.DataFrame(omnibus_rows)
    omnibus["qvalue_bh"] = bh_adjust(omnibus.pvalue.to_numpy())
    pairwise = pd.DataFrame(pairwise_rows)
    pairwise["qvalue_bh"] = bh_adjust(pairwise.pvalue.to_numpy())
    return omnibus, pairwise


def summarise(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (band, knee), data in groups.groupby(["band", "true_knee_hz"], sort=False):
        values = data.fooof_rho.dropna().to_numpy()
        rows.append(
            {
                "band": band,
                "true_knee_hz": knee,
                "mean_rho": float(np.mean(values)),
                "sd_rho": float(np.std(values, ddof=1)),
                "ci_low": float(np.quantile(values, 0.025)),
                "ci_high": float(np.quantile(values, 0.975)),
                "significant_repeats": int((data.fooof_p < 0.05).sum()),
                "mean_valid_n": float(data.valid_n.mean()),
                "median_fitted_knee_hz": float(data.median_fitted_knee_hz.median()),
                "n_repeats": N_REPEATS,
                "trials_per_repeat": N_TRIALS,
            }
        )
    return pd.DataFrame(rows)


def plot(groups: pd.DataFrame, summary: pd.DataFrame, omnibus: pd.DataFrame) -> Path:
    base.style()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.6), constrained_layout=True)
    for ax, display_band in zip(axes.ravel(), BAND_ORDER):
        band = display_band.lower()
        color = BAND_COLORS[band]
        raw = groups[groups.band == band]
        agg = summary[summary.band == band].sort_values("true_knee_hz")
        for repeat in range(N_REPEATS):
            row = raw[raw.repeat == repeat].sort_values("true_knee_hz")
            ax.plot(row.true_knee_hz, row.fooof_rho, color=color, alpha=0.13, lw=1.1)
            ax.scatter(row.true_knee_hz, row.fooof_rho, color=color, alpha=0.22, s=16, lw=0)
        ax.fill_between(
            agg.true_knee_hz,
            agg.ci_low,
            agg.ci_high,
            color=color,
            alpha=0.17,
            lw=0,
        )
        ax.plot(agg.true_knee_hz, agg.mean_rho, "o-", color=color, lw=3.0, ms=7)
        ax.axhline(0, color="0.35", lw=1.4)
        test = omnibus[omnibus.band == band].iloc[0]
        ax.text(
            0.04,
            0.06,
            rf"Friedman $q$ = {test.qvalue_bh:.2g}; $W$ = {test.kendall_w:.2f}",
            transform=ax.transAxes,
            fontsize=12.5,
            fontweight="bold",
        )
        ax.set_xscale("log")
        ax.set_xticks(KNEES)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(display_band, fontweight="bold")
        ax.set_xlabel("True knee frequency (Hz)")
        ax.set_ylabel(r"FOOOF p–ap Spearman $\rho$")
        for spine in ax.spines.values():
            spine.set_linewidth(2.2)
    fig.suptitle("Knee position changes FOOOF p–ap error correlation", fontsize=20, fontweight="bold")
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "knee_position_by_band_official_fooof3.svg"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "knee_position_by_band_official_fooof3_preview.png", dpi=190,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fooof_group_class()
    with mp.get_context("fork").Pool(4) as pool:
        nested = pool.map(repeat_worker, range(N_REPEATS))
    groups = pd.DataFrame([row for part in nested for row in part])
    summary = summarise(groups)
    omnibus, pairwise = statistics_tables(groups)
    figure = plot(groups, summary, omnibus)
    groups.to_csv(OUT / "group_results.csv", index=False)
    summary.to_csv(OUT / "summary.csv", index=False)
    omnibus.to_csv(OUT / "friedman_tests.csv", index=False)
    pairwise.to_csv(OUT / "pairwise_vs_5hz.csv", index=False)
    metrics = {
        "figure": str(figure),
        "design": {
            "true_knees_hz": KNEES.tolist(),
            "bands": BAND_ORDER,
            "repeats": N_REPEATS,
            "trials_per_repeat": N_TRIALS,
            "fooof": "1.1.1 knee mode, max_n_peaks=3, width 1.5-6 Hz, min height 0.6, threshold 2, fit 0.2-197 Hz",
            "paired_common_random_numbers": True,
        },
        "omnibus": omnibus.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
