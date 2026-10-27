from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, stats
from scipy.special import polygamma

from .plot_formatting import format_correlation


ROOT = Path(__file__).resolve().parents[3]
VENDOR = ROOT / "experiments" / "pap_theory_validation" / "vendor"
if str(VENDOR) not in sys.path:
    sys.path.append(str(VENDOR))

from specparam import SpectralModel  # noqa: E402


OUTPUT = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "fooof_split_surrogate_100_1000hz"
SEED = 20260816
FREQUENCIES = np.arange(20.0, 1001.0, 1.0)
APERIODIC_BAND = (100.0, 1000.0)
TRUE_CENTER = 100.0
TRUE_EXPONENT = 1.5
PEAK_AMPLITUDE = 0.5
PEAK_SIGMA = 30.0
GAMMA_SHAPE = 4.0
CONDITIONAL_REALIZATIONS = 4000
NULL_DATASETS = 100
TRIALS_PER_DATASET = 256
CENTER_SD = 4.0
EXPONENT_SD = 0.08


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 16,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.linewidth": 2.3,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "legend.fontsize": 14,
            "lines.linewidth": 3.3,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _mean_psd(center: float, exponent: float) -> np.ndarray:
    log_power = 1.0 - exponent * np.log10(FREQUENCIES / 10.0)
    log_power += PEAK_AMPLITUDE * np.exp(
        -0.5 * ((FREQUENCIES - center) / PEAK_SIGMA) ** 2
    )
    return 10.0**log_power


def _noisy_psd(mean_psd: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return mean_psd * rng.gamma(GAMMA_SHAPE, 1.0 / GAMMA_SHAPE, size=len(mean_psd))


def _model() -> SpectralModel:
    return SpectralModel(
        aperiodic_mode="fixed",
        verbose=False,
        max_n_peaks=1,
        peak_width_limits=[10.0, 80.0],
        min_peak_height=0.02,
        peak_threshold=0.5,
    )


def _fooof_feature(model: SpectralModel, psd: np.ndarray) -> tuple[float, float]:
    # Estimate P from the same realization used for A. The high-frequency A
    # estimate is explicit so both methods use the requested 100-1000 Hz band.
    model.fit(FREQUENCIES, psd, [20.0, 300.0])
    periodic = model.results.params.periodic.asdict()["peak_fit"]
    if len(periodic) == 0:
        return np.nan, np.nan
    slope_band = (FREQUENCIES >= APERIODIC_BAND[0]) & (
        FREQUENCIES <= APERIODIC_BAND[1]
    )
    fit = stats.linregress(
        np.log10(FREQUENCIES[slope_band]), np.log10(psd[slope_band])
    )
    return float(periodic[0, 0]), float(-fit.slope)


def _split_feature(peak_psd: np.ndarray, slope_psd: np.ndarray) -> tuple[float, float]:
    # Peak and slope use disjoint bins; slope_psd is also an independent epoch.
    side = ((FREQUENCIES >= 20.0) & (FREQUENCIES <= 45.0)) | (
        (FREQUENCIES >= 165.0) & (FREQUENCIES <= 180.0)
    )
    background = np.polyfit(
        np.log10(FREQUENCIES[side]), np.log10(peak_psd[side]), 1
    )
    residual = np.log10(peak_psd) - np.polyval(
        background, np.log10(FREQUENCIES)
    )
    residual = ndimage.gaussian_filter1d(residual, 3.0)
    peak_band = (FREQUENCIES >= 50.0) & (FREQUENCIES <= 150.0)
    peak = float(FREQUENCIES[peak_band][np.argmax(residual[peak_band])])
    slope_band = (FREQUENCIES >= APERIODIC_BAND[0]) & (
        FREQUENCIES <= APERIODIC_BAND[1]
    )
    fit = stats.linregress(
        np.log10(FREQUENCIES[slope_band]), np.log10(slope_psd[slope_band])
    )
    return peak, float(-fit.slope)


def _fooof_tangent_covariance() -> np.ndarray:
    f = FREQUENCIES
    peak_band = (f >= 20.0) & (f <= 300.0)
    fp = f[peak_band]
    gaussian = np.exp(-0.5 * ((fp - TRUE_CENTER) / PEAK_SIGMA) ** 2)
    peak_design = np.column_stack(
        [
            np.ones(len(fp)),
            -np.log10(fp / 10.0),
            gaussian,
            PEAK_AMPLITUDE * gaussian * (fp - TRUE_CENTER) / PEAK_SIGMA**2,
            PEAK_AMPLITUDE * gaussian * (fp - TRUE_CENTER) ** 2 / PEAK_SIGMA**3,
        ]
    )
    slope_band = (f >= APERIODIC_BAND[0]) & (f <= APERIODIC_BAND[1])
    fa = f[slope_band]
    slope_design = np.column_stack([np.ones(len(fa)), -np.log10(fa / 10.0)])
    center_weights = np.zeros(len(f))
    exponent_weights = np.zeros(len(f))
    center_weights[peak_band] = np.linalg.pinv(peak_design)[3]
    exponent_weights[slope_band] = np.linalg.pinv(slope_design)[1]
    log_noise_variance = float(polygamma(1, GAMMA_SHAPE) / np.log(10.0) ** 2)
    weights = np.vstack([center_weights, exponent_weights])
    return log_noise_variance * weights @ weights.T


def _conditional_experiment(force: bool = False) -> pd.DataFrame:
    path = OUTPUT / "conditional_fixed_spectrum_estimates.csv.gz"
    if path.exists() and not force:
        return pd.read_csv(path)
    rng = np.random.default_rng(SEED)
    mean_psd = _mean_psd(TRUE_CENTER, TRUE_EXPONENT)
    model = _model()
    rows = []
    for realization in range(CONDITIONAL_REALIZATIONS):
        peak_epoch = _noisy_psd(mean_psd, rng)
        slope_epoch = _noisy_psd(mean_psd, rng)
        fooof_peak, fooof_exponent = _fooof_feature(model, peak_epoch)
        split_peak, split_exponent = _split_feature(peak_epoch, slope_epoch)
        rows.append(
            {
                "realization": realization,
                "fooof_peak": fooof_peak,
                "fooof_exponent": fooof_exponent,
                "split_peak": split_peak,
                "split_exponent": split_exponent,
            }
        )
        if (realization + 1) % 1000 == 0:
            print(f"conditional FOOOF/split {realization + 1}/{CONDITIONAL_REALIZATIONS}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False, compression="gzip")
    return result


def _null_experiment(force: bool = False) -> pd.DataFrame:
    path = OUTPUT / "latent_null_dataset_correlations.csv"
    if path.exists() and not force:
        return pd.read_csv(path)
    rng = np.random.default_rng(SEED + 100000)
    model = _model()
    rows = []
    for dataset in range(NULL_DATASETS):
        center = TRUE_CENTER + CENTER_SD * rng.normal(size=TRIALS_PER_DATASET)
        exponent = TRUE_EXPONENT + EXPONENT_SD * rng.normal(size=TRIALS_PER_DATASET)
        fooof = np.full((TRIALS_PER_DATASET, 2), np.nan)
        split = np.full((TRIALS_PER_DATASET, 2), np.nan)
        for trial in range(TRIALS_PER_DATASET):
            mean_psd = _mean_psd(center[trial], exponent[trial])
            peak_epoch = _noisy_psd(mean_psd, rng)
            slope_epoch = _noisy_psd(mean_psd, rng)
            fooof[trial] = _fooof_feature(model, peak_epoch)
            split[trial] = _split_feature(peak_epoch, slope_epoch)
        valid = np.isfinite(fooof).all(axis=1) & np.isfinite(split).all(axis=1)
        latent_test = stats.pearsonr(center[valid], exponent[valid])
        fooof_test = stats.pearsonr(fooof[valid, 0], fooof[valid, 1])
        split_test = stats.pearsonr(split[valid, 0], split[valid, 1])
        rows.append(
            {
                "dataset": dataset,
                "n_valid": int(valid.sum()),
                "latent_r": float(latent_test.statistic),
                "fooof_r": float(fooof_test.statistic),
                "fooof_p": float(fooof_test.pvalue),
                "split_r": float(split_test.statistic),
                "split_p": float(split_test.pvalue),
            }
        )
        if (dataset + 1) % 10 == 0:
            print(f"latent-null datasets {dataset + 1}/{NULL_DATASETS}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def _metrics(conditional: pd.DataFrame, null: pd.DataFrame) -> dict:
    conditional = conditional.dropna()
    fooof_empirical = float(
        stats.pearsonr(conditional.fooof_peak, conditional.fooof_exponent).statistic
    )
    split_empirical = float(
        stats.pearsonr(conditional.split_peak, conditional.split_exponent).statistic
    )
    fooof_cov = _fooof_tangent_covariance()
    fooof_theory = float(fooof_cov[0, 1] / np.sqrt(fooof_cov[0, 0] * fooof_cov[1, 1]))
    latent_cov = np.diag([CENTER_SD**2, EXPONENT_SD**2])
    total = latent_cov + fooof_cov
    null_fooof_theory = float(total[0, 1] / np.sqrt(total[0, 0] * total[1, 1]))
    return {
        "software": "specparam 2.0.0rc7 SpectralModel (FOOOF successor)",
        "conditional": {
            "n_realizations": int(len(conditional)),
            "fooof_empirical_error_r": fooof_empirical,
            "fooof_tangent_theory_error_r_eq25_32": fooof_theory,
            "split_empirical_error_r": split_empirical,
            "split_theory_error_r": 0.0,
        },
        "latent_null": {
            "n_datasets": int(len(null)),
            "trials_per_dataset": TRIALS_PER_DATASET,
            "fooof_mean_r": float(null.fooof_r.mean()),
            "fooof_theory_r": null_fooof_theory,
            "fooof_false_positive_rate": float((null.fooof_p < 0.05).mean()),
            "split_mean_r": float(null.split_r.mean()),
            "split_theory_r": 0.0,
            "split_false_positive_rate": float((null.split_p < 0.05).mean()),
            "latent_mean_r": float(null.latent_r.mean()),
            "fooof_abs_bias": float(abs(null.fooof_r.mean())),
            "split_abs_bias": float(abs(null.split_r.mean())),
        },
    }


def _plot(conditional: pd.DataFrame, null: pd.DataFrame, metrics: dict) -> None:
    rng = np.random.default_rng(SEED + 900000)
    mean_psd = _mean_psd(TRUE_CENTER, TRUE_EXPONENT)
    representative = _noisy_psd(mean_psd, rng)
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 9.1), constrained_layout=True)

    ax = axes[0, 0]
    ax.loglog(FREQUENCIES, representative, color="0.55", lw=2.2, alpha=0.8, label="Surrogate PSD")
    ax.loglog(FREQUENCIES, mean_psd, color="#202020", lw=4.0, label="True mean PSD")
    ax.axvspan(50, 150, color="#247BA0", alpha=0.18, label="Split P bins")
    ax.axvspan(100, 1000, color="#D1495B", alpha=0.14, label="A: 100-1000 Hz")
    ax.set(xlabel="Frequency (Hz)", ylabel="Power")
    ax.set_xticks([20, 100, 1000])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, loc="lower left")
    ax.text(-0.14, 1.07, "A", transform=ax.transAxes, fontsize=24, fontweight="bold")

    ax = axes[0, 1]
    methods = ["FOOOF", "Split"]
    empirical = [
        metrics["conditional"]["fooof_empirical_error_r"],
        metrics["conditional"]["split_empirical_error_r"],
    ]
    theory = [
        metrics["conditional"]["fooof_tangent_theory_error_r_eq25_32"],
        0.0,
    ]
    xpos = np.arange(2)
    width = 0.34
    empirical_bars = ax.bar(xpos - width / 2, empirical, width, color="#247BA0", label="Surrogate")
    theory_bars = ax.bar(xpos + width / 2, theory, width, color="#D1495B", label="Theory")
    ax.axhline(0, color="0.25", lw=2.0)
    ax.set(xticks=xpos, xticklabels=methods, ylabel="Conditional error correlation")
    ax.set_ylim(-0.12, 0.68)
    ax.legend(frameon=False, loc="upper right")
    for bars, values in ((empirical_bars, empirical), (theory_bars, theory)):
        for bar, value in zip(bars, values):
            offset = 0.025 if value >= 0 else -0.045
            ax.text(
                bar.get_x() + bar.get_width() / 2, value + offset,
                format_correlation(value), ha="center", va="center",
                fontsize=11.5, fontweight="bold",
            )
    ax.text(-0.14, 1.07, "B", transform=ax.transAxes, fontsize=24, fontweight="bold")

    ax = axes[1, 0]
    violin = ax.violinplot(
        [null.fooof_r.to_numpy(), null.split_r.to_numpy()],
        positions=[0, 1],
        widths=0.72,
        showmeans=True,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], ["#247BA0", "#2A9D8F"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.82)
    violin["cmeans"].set_color("black")
    violin["cmeans"].set_linewidth(3.0)
    ax.scatter(np.zeros(len(null)) - 0.08, null.fooof_r, s=14, color="#247BA0", alpha=0.32)
    ax.scatter(np.ones(len(null)) + 0.08, null.split_r, s=14, color="#2A9D8F", alpha=0.32)
    ax.plot([-0.36, 0.36], [metrics["latent_null"]["fooof_theory_r"]] * 2, "--", color="#D1495B", lw=3.2)
    ax.plot([0.64, 1.36], [0, 0], "--", color="#D1495B", lw=3.2)
    ax.axhline(0, color="0.25", lw=1.8)
    ax.set(xticks=[0, 1], xticklabels=methods, ylabel="Estimated P-A correlation")
    for xpos_value, value in enumerate((null.fooof_r.mean(), null.split_r.mean())):
        ax.text(
            xpos_value, ax.get_ylim()[1] * 0.92, format_correlation(value),
            ha="center", va="top", fontsize=11.5, fontweight="bold",
        )
    ax.text(-0.14, 1.07, "C", transform=ax.transAxes, fontsize=24, fontweight="bold")

    ax = axes[1, 1]
    false_positive = [
        metrics["latent_null"]["fooof_false_positive_rate"],
        metrics["latent_null"]["split_false_positive_rate"],
    ]
    bars = ax.bar(methods, false_positive, color=["#247BA0", "#2A9D8F"], width=0.62)
    ax.axhline(0.05, color="#D1495B", ls="--", lw=3.0, label="Nominal 0.05")
    ax.set(ylabel="False-positive rate", ylim=(0, max(0.25, max(false_positive) * 1.22)))
    for bar, value in zip(bars, false_positive):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontweight="bold")
    ax.legend(frameon=False, loc="upper right")
    ax.text(-0.14, 1.07, "D", transform=ax.transAxes, fontsize=24, fontweight="bold")

    for ax in axes.ravel():
        ax.tick_params(length=6, width=2.0)
        for spine in ax.spines.values():
            spine.set_linewidth(2.3)
    _save(fig, OUTPUT / "fooof_vs_split_surrogate_validation")


def main(force: bool = False) -> None:
    _style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    conditional = _conditional_experiment(force=force)
    null = _null_experiment(force=force)
    metrics = _metrics(conditional, null)
    _plot(conditional, null, metrics)
    (OUTPUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    setup = {
        "frequency_grid_hz": [20.0, 1000.0, 1.0],
        "mean_log10_psd": "1 - chi*log10(f/10) + 0.5*exp(-(f-P)^2/(2*30^2))",
        "periodogram_noise": "Gamma(shape=4, scale=1/4), independently by frequency and epoch",
        "true_center_hz": TRUE_CENTER,
        "true_exponent": TRUE_EXPONENT,
        "conditional_realizations": CONDITIONAL_REALIZATIONS,
        "null_datasets": NULL_DATASETS,
        "trials_per_dataset": TRIALS_PER_DATASET,
        "null_center_distribution": f"Normal({TRUE_CENTER}, {CENTER_SD}^2)",
        "null_exponent_distribution": f"Normal({TRUE_EXPONENT}, {EXPONENT_SD}^2), independent of center",
        "fooof": "specparam one-Gaussian peak over 20-300 Hz; same-realization log-log A over 100-1000 Hz",
        "split_peak": "peak epoch, residualized and smoothed 50-150 Hz maximum",
        "split_slope": "independent epoch, log-log OLS over 100-1000 Hz",
        "theory": "FOOOF sandwich/tangent covariance Eq. 25/32; split cross-error covariance=0",
    }
    (OUTPUT / "setup.json").write_text(json.dumps(setup, indent=2))
    print(json.dumps({"setup": setup, "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
