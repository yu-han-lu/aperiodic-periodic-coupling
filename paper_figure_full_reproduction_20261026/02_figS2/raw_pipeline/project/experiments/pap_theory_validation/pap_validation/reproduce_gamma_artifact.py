from __future__ import annotations

import json
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, optimize, signal, stats


DATA = Path("/PORTABLE_HOME/Desktop/data")
ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "fooof_gamma_discrepancy_2026-09-01"
SEED = 20260901
FS = 1000
BANDS = {
    "Alpha": (8.0, 12.0, (6.0, 15.0)),
    "Beta": (13.0, 30.0, (12.0, 35.0)),
    "Low gamma": (30.0, 60.0, (25.0, 70.0)),
    "High gamma": (60.0, 150.0, (50.0, 170.0)),
}
COLORS = {
    "Alpha": "#E9A03B",
    "Beta": "#E76F51",
    "Low gamma": "#4C78A8",
    "High gamma": "#343A69",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 15,
            "axes.labelsize": 16,
            "axes.titlesize": 16,
            "axes.linewidth": 2.1,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 13,
            "xtick.major.width": 2.0,
            "ytick.major.width": 2.0,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "svg.fonttype": "none",
        }
    )


def gaussian(freq: np.ndarray, center: float, height: float, sigma: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((freq - center) / sigma) ** 2)


def ap_knee(freq: np.ndarray, pars: np.ndarray) -> np.ndarray:
    offset, exponent, log10_knee = pars
    return offset - np.log10(10.0**log10_knee + freq**exponent)


def ap_fixed(freq: np.ndarray, pars: np.ndarray) -> np.ndarray:
    return pars[0] - pars[1] * np.log10(freq)


def initial_ap(y: np.ndarray, freq: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(freq)), -np.log10(freq)])
    offset, exponent = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(offset), float(exponent)


def fit_ap_knee(
    y: np.ndarray, freq: np.ndarray, x0: np.ndarray | None = None
) -> np.ndarray:
    if x0 is None:
        offset, exponent = initial_ap(y, freq)
        exponent = float(np.clip(exponent, 0.10, 4.0))
        x0 = np.array([np.clip(offset, -8, 8), exponent, exponent * np.log10(5.0)])
    return optimize.least_squares(
        lambda x: y - ap_knee(freq, x),
        x0,
        bounds=([-8.0, 0.05, -5.0], [8.0, 5.0, 6.0]),
        max_nfev=220,
    ).x


def fit_joint_fixed_local(
    y: np.ndarray, freq: np.ndarray, peak_range: tuple[float, float]
) -> tuple[float, float]:
    mask = (freq >= peak_range[0]) & (freq <= peak_range[1])
    use_f, use_y = freq[mask], y[mask]
    offset, exponent = initial_ap(use_y, use_f)
    flat = use_y - ap_fixed(use_f, np.array([offset, exponent]))
    center = float(use_f[np.argmax(flat)])
    x0 = np.array(
        [
            np.clip(offset, -8, 8),
            np.clip(exponent, 0.05, 4.5),
            center,
            np.clip(max(float(flat.max()), 0.60), 0.01, 2.95),
            1.5,
        ]
    )
    lo = np.array([-8.0, 0.0, peak_range[0], 0.0, 0.75])
    hi = np.array([8.0, 5.0, peak_range[1], 3.0, 3.0])

    def residual(x: np.ndarray) -> np.ndarray:
        return use_y - (
            ap_fixed(use_f, x[:2]) + gaussian(use_f, x[2], x[3], x[4])
        )

    fitted = optimize.least_squares(
        residual, x0, bounds=(lo, hi), max_nfev=180
    ).x
    return float(fitted[2]), float(fitted[1])


def fit_sequential_knee_fooof(
    y: np.ndarray, freq: np.ndarray, peak_range: tuple[float, float]
) -> tuple[float, float]:
    """FOOOF's robust-AP -> peak -> AP-refit sequence, one Gaussian peak."""
    first = fit_ap_knee(y, freq)
    flat = np.maximum(y - ap_knee(freq, first), 0.0)
    baseline = flat <= np.percentile(flat, 2.5)
    if baseline.sum() < 8:
        order = np.argsort(flat)
        baseline = np.zeros(len(flat), dtype=bool)
        baseline[order[: max(8, int(0.025 * len(flat)))]] = True
    robust = fit_ap_knee(y[baseline], freq[baseline], first)
    flattened = y - ap_knee(freq, robust)
    peak_mask = (freq >= peak_range[0]) & (freq <= peak_range[1])
    center0 = float(freq[peak_mask][np.argmax(flattened[peak_mask])])
    height0 = float(np.clip(max(flattened[peak_mask].max(), 0.60), 0.01, 2.95))
    peak = optimize.least_squares(
        lambda x: flattened - gaussian(freq, x[0], x[1], x[2]),
        np.array([center0, height0, 1.5]),
        bounds=(
            np.array([peak_range[0], 0.0, 0.75]),
            np.array([peak_range[1], 3.0, 3.0]),
        ),
        max_nfev=180,
    ).x
    final_ap = fit_ap_knee(y - gaussian(freq, *peak), freq, robust)
    return float(peak[0]), float(final_ap[1])


def fit_joint_knee_global(
    y: np.ndarray, freq: np.ndarray, peak_range: tuple[float, float]
) -> tuple[float, float]:
    ap0 = fit_ap_knee(y, freq)
    flat = y - ap_knee(freq, ap0)
    peak_mask = (freq >= peak_range[0]) & (freq <= peak_range[1])
    center0 = float(freq[peak_mask][np.argmax(flat[peak_mask])])
    x0 = np.array(
        [
            *ap0,
            center0,
            np.clip(max(flat[peak_mask].max(), 0.60), 0.01, 2.95),
            1.5,
        ]
    )
    fitted = optimize.least_squares(
        lambda x: y - (ap_knee(freq, x[:3]) + gaussian(freq, x[3], x[4], x[5])),
        x0,
        bounds=(
            np.array([-8.0, 0.05, -5.0, peak_range[0], 0.0, 0.75]),
            np.array([8.0, 5.0, 6.0, peak_range[1], 3.0, 3.0]),
        ),
        max_nfev=220,
    ).x
    return float(fitted[3]), float(fitted[1])


def scalar_cells(values: np.ndarray) -> np.ndarray:
    result = []
    for value in values:
        array = np.asarray(value).squeeze()
        result.append(float(array) if array.size == 1 else np.nan)
    return np.asarray(result)


def peak_cells(values: np.ndarray) -> np.ndarray:
    result = []
    for value in values:
        array = np.asarray(value).squeeze()
        result.append(
            float(array.flat[0]) if array.size and np.any(array != 0) else np.nan
        )
    return np.asarray(result)


def reproduce_saved_simulation() -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    saved = io.loadmat(
        DATA / "simulate_data" / "fooof_result.mat", squeeze_me=True,
        struct_as_record=False,
    )
    split_ap = io.loadmat(
        DATA / "simulate_data" / "split_exponent.mat", squeeze_me=True,
        struct_as_record=False,
    )
    split_p = io.loadmat(
        DATA / "simulate_data" / "split_oscillation.mat", squeeze_me=True,
        struct_as_record=False,
    )
    fooof = np.column_stack([saved["peak_cf"], saved["aperiodic_exponent"]])
    split = np.column_stack(
        [peak_cells(split_p["peak_sum"]), scalar_cells(split_ap["exponent_sum"])]
    )
    rows = []
    for method, values in [("FOOOF", fooof), ("SPLIT", split)]:
        good = np.isfinite(values).all(axis=1)
        test = stats.spearmanr(values[good, 0], values[good, 1])
        rows.append(
            {
                "dataset": "provided simulation",
                "method": method,
                "n": int(good.sum()),
                "spearman_rho": float(test.statistic),
                "pvalue": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows), {"FOOOF": fooof, "SPLIT": split}


def refit_provided_spectra() -> pd.DataFrame:
    with h5py.File(DATA / "simulate_data" / "irasa_data.mat", "r") as handle:
        freq = np.asarray(handle["f"]).squeeze()
        original = np.asarray(handle["original_powspectra"])
    mask = (freq >= 0.2) & (freq <= 200.0)
    freq = freq[mask]
    spectra = np.log10(original[mask, :].T)
    fitted = {
        "Simultaneous joint knee fit": np.asarray(
            [fit_joint_knee_global(row, freq, (30.0, 150.0)) for row in spectra]
        ),
        "Sequential robust FOOOF": np.asarray(
            [fit_sequential_knee_fooof(row, freq, (30.0, 150.0)) for row in spectra]
        ),
    }
    saved = io.loadmat(DATA / "simulate_data" / "fooof_result.mat", squeeze_me=True)
    fitted["Saved official FOOOF"] = np.column_stack(
        [saved["peak_cf"], saved["aperiodic_exponent"]]
    )
    rows = []
    for method, values in fitted.items():
        good = np.isfinite(values).all(axis=1)
        test = stats.spearmanr(values[good, 0], values[good, 1])
        rows.append(
            {
                "dataset": "same provided PSDs",
                "method": method,
                "n": int(good.sum()),
                "spearman_rho": float(test.statistic),
                "pvalue": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows)


def rows_from_peak_cell(value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.size == 0:
        return np.empty((0, 3))
    return array.reshape(1, -1) if array.ndim == 1 else array


def reproduce_human_high_gamma() -> pd.DataFrame:
    data = io.loadmat(
        DATA / "human_data" / "sub-74-38-result.mat", squeeze_me=True,
        struct_as_record=False,
    )
    fooof_center, fooof_exp = [], []
    split_center, split_exp = [], []
    for fp, fe, sp, se, sk in zip(
        data["fooof_peak"], data["fooof_exponent"], data["split_peak"],
        data["split_exponent"], data["split_knee"],
    ):
        for peak in rows_from_peak_cell(fp):
            if 60.0 <= peak[0] <= 150.0:
                fooof_center.append(float(peak[0]))
                fooof_exp.append(float(fe))
        exponents = np.atleast_1d(se).astype(float)
        knees = np.atleast_1d(sk).astype(float)
        boundaries = np.r_[-np.inf, knees, np.inf]
        for peak in rows_from_peak_cell(sp):
            center = float(peak[0])
            if 60.0 <= center <= 150.0:
                segment = int(np.searchsorted(boundaries, center, side="right") - 1)
                if segment < len(exponents):
                    split_center.append(center)
                    split_exp.append(float(exponents[segment]))
    rows = []
    for method, x, y in [
        ("FOOOF", fooof_center, fooof_exp),
        ("SPLIT", split_center, split_exp),
    ]:
        test = stats.spearmanr(x, y)
        rows.append(
            {
                "dataset": "human sub-74 channel-38, pooled 60-150 Hz peaks",
                "method": method,
                "n": len(x),
                "spearman_rho": float(test.statistic),
                "pvalue": float(test.pvalue),
            }
        )
    return pd.DataFrame(rows)


def generate_trial_psd(
    rng: np.random.Generator,
    base_signal: np.ndarray,
    base_freq: np.ndarray,
    base_psd: np.ndarray,
    center: float,
    bandwidth: float,
    power_fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    low = max(center - bandwidth / 2.0, 0.1)
    high = min(center + bandwidth / 2.0, FS / 2.0 - 0.1)
    sos = signal.butter(2, [low, high], btype="bandpass", fs=FS, output="sos")
    oscillation = signal.sosfiltfilt(sos, rng.normal(size=len(base_signal)))
    oscillation /= np.sqrt(np.mean(oscillation**2))
    _, unit_psd = signal.welch(
        base_signal + oscillation, fs=FS, window="hamming",
        nperseg=len(base_signal), noverlap=len(base_signal) // 2,
        nfft=32768, detrend="constant",
    )
    local_base = float(np.interp(center, base_freq, base_psd))
    unit_bump = float(np.interp(center, base_freq, unit_psd) - local_base)
    amplitude = np.sqrt(max(power_fraction * local_base, 1e-30) / max(unit_bump, 1e-30))
    freq, psd = signal.welch(
        base_signal + amplitude * oscillation, fs=FS, window="hamming",
        nperseg=len(base_signal), noverlap=len(base_signal) // 2,
        nfft=32768, detrend="constant",
    )
    mask = (freq >= 0.2) & (freq <= 200.0)
    return freq[mask], np.log10(psd[mask])


def controlled_before_after(
    n_repeats: int = 10, n_trials_per_band: int = 70
) -> pd.DataFrame:
    with h5py.File(DATA / "simulate_data" / "x_with_oscillation.mat", "r") as handle:
        base_signal = np.asarray(handle["x_base"]).squeeze()
    base_freq, base_psd = signal.welch(
        base_signal, fs=FS, window="hamming", nperseg=len(base_signal),
        noverlap=len(base_signal) // 2, nfft=32768, detrend="constant",
    )
    rows: list[dict[str, float | int | str]] = []
    for repeat in range(n_repeats):
        rng = np.random.default_rng(SEED + repeat)
        for band, (center_lo, center_hi, fit_range) in BANDS.items():
            true_centers = rng.uniform(center_lo, center_hi, n_trials_per_band)
            widths = rng.uniform(1.5, 6.0, n_trials_per_band)
            fractions = rng.uniform(2.8, 3.8, n_trials_per_band)
            spectra = [
                generate_trial_psd(
                    rng, base_signal, base_freq, base_psd, center, width, fraction
                )
                for center, width, fraction in zip(true_centers, widths, fractions)
            ]
            freq = spectra[0][0]
            log_psd = [item[1] for item in spectra]
            before = np.asarray(
                [fit_joint_fixed_local(row, freq, fit_range) for row in log_psd]
            )
            after = np.asarray(
                [fit_sequential_knee_fooof(row, freq, fit_range) for row in log_psd]
            )
            for pipeline, values in [
                ("Our approximation: local simultaneous fixed", before),
                ("Researcher pipeline: global sequential knee", after),
            ]:
                estimate_test = stats.spearmanr(values[:, 0], values[:, 1])
                bias_test = stats.spearmanr(true_centers, values[:, 1])
                rows.append(
                    {
                        "repeat": repeat,
                        "band": band,
                        "pipeline": pipeline,
                        "n_trials": n_trials_per_band,
                        "estimated_p_ap_rho": float(estimate_test.statistic),
                        "estimated_p_ap_p": float(estimate_test.pvalue),
                        "true_center_to_estimated_exponent_rho": float(bias_test.statistic),
                    }
                )
    return pd.DataFrame(rows)


def plot_results(
    saved_stats: pd.DataFrame,
    saved_values: dict[str, np.ndarray],
    refit: pd.DataFrame,
    controlled: pd.DataFrame,
) -> Path:
    style()
    fig, axes = plt.subplots(2, 3, figsize=(16.5, 10.5), constrained_layout=True)
    for ax, method, color, label in [
        (axes[0, 0], "FOOOF", "#247BA0", "A"),
        (axes[0, 1], "SPLIT", "#2A9D8F", "B"),
    ]:
        values = saved_values[method]
        good = np.isfinite(values).all(axis=1)
        values = values[good]
        result = saved_stats[saved_stats.method == method].iloc[0]
        ax.scatter(values[:, 0], values[:, 1], s=34, alpha=0.72,
                   color=color, edgecolor="white", linewidth=0.35)
        line = stats.linregress(values[:, 0], values[:, 1])
        xx = np.linspace(values[:, 0].min(), values[:, 0].max(), 100)
        ax.plot(xx, line.intercept + line.slope * xx, color="#222222", lw=2.0)
        ax.set(xlabel="Estimated peak center (Hz)", ylabel="Estimated exponent",
               title=f"Provided simulation · {method}")
        ax.text(0.04, 0.94,
                rf"Spearman $\rho$ = {result.spearman_rho:.2f}" + "\n" +
                rf"$p$ = {result.pvalue:.2g}",
                transform=ax.transAxes, ha="left", va="top", fontweight="bold")
        ax.text(-0.14, 1.08, label, transform=ax.transAxes,
                fontsize=23, fontweight="bold")

    ax = axes[0, 2]
    order = ["Simultaneous joint knee fit", "Sequential robust FOOOF", "Saved official FOOOF"]
    subset = refit.set_index("method").loc[order].reset_index()
    x = np.arange(len(order))
    bars = ax.bar(x, subset.spearman_rho, width=0.58,
                  color=["#7A7A7A", "#D1495B", "#247BA0"])
    ax.axhline(0, color="#333333", lw=1.5)
    ax.set(xticks=x,
           xticklabels=["Joint knee", "Sequential\nknee", "Official\nFOOOF"],
           ylabel="Spearman rho", title="Same 100 PSDs · fitting order only",
           ylim=(0.0, 1.12))
    for bar, value in zip(bars, subset.spearman_rho):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.025,
                f"{value:.2f}", ha="center", va="bottom", fontweight="bold")
    ax.text(-0.14, 1.08, "C", transform=ax.transAxes,
            fontsize=23, fontweight="bold")

    summary = (
        controlled.groupby(["pipeline", "band"])["estimated_p_ap_rho"]
        .agg(["mean", "std"]).reset_index()
    )
    pipelines = [
        "Our approximation: local simultaneous fixed",
        "Researcher pipeline: global sequential knee",
    ]
    for ax, pipeline, label, title in [
        (axes[1, 0], pipelines[0], "D", "Before: our analysis configuration"),
        (axes[1, 1], pipelines[1], "E", "After: researcher configuration"),
    ]:
        part = summary[summary.pipeline == pipeline].set_index("band").loc[list(BANDS)]
        x = np.arange(len(BANDS))
        ax.errorbar(x, part["mean"], yerr=part["std"], fmt="none",
                    ecolor="#222222", elinewidth=2.0, capsize=5)
        ax.scatter(x, part["mean"], s=115,
                   color=[COLORS[name] for name in BANDS],
                   edgecolor="#222222", linewidth=0.8, zorder=3)
        ax.axhline(0, color="#555555", lw=1.4)
        ax.set(xticks=x, xticklabels=["Alpha", "Beta", "Low\ngamma", "High\ngamma"],
               ylabel="Estimated p–ap Spearman rho", title=title, ylim=(-1.08, 1.08))
        ax.text(-0.14, 1.08, label, transform=ax.transAxes,
                fontsize=23, fontweight="bold")

    ax = axes[1, 2]
    pivot = summary.pivot(index="band", columns="pipeline", values="mean").loc[list(BANDS)]
    change = pivot[pipelines[1]] - pivot[pipelines[0]]
    x = np.arange(len(BANDS))
    ax.bar(x, change, width=0.62, color=[COLORS[name] for name in BANDS], alpha=0.88)
    ax.axhline(0, color="#333333", lw=1.5)
    ax.set(xticks=x, xticklabels=["Alpha", "Beta", "Low\ngamma", "High\ngamma"],
           ylabel=r"Change in $\rho$ (after − before)",
           title="The apparent artifact moves to gamma")
    ax.text(-0.14, 1.08, "F", transform=ax.transAxes,
            fontsize=23, fontweight="bold")

    for ax in axes.ravel():
        ax.tick_params(width=2.0, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(2.1)
    path = OUT / "fooof_gamma_discrepancy_reproduction.svg"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fooof_gamma_discrepancy_reproduction_preview.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    saved_stats, saved_values = reproduce_saved_simulation()
    refit = refit_provided_spectra()
    human = reproduce_human_high_gamma()
    controlled_path = OUT / "controlled_before_after_repeats.csv"
    controlled = (
        pd.read_csv(controlled_path)
        if controlled_path.exists()
        else controlled_before_after()
    )
    figure = plot_results(saved_stats, saved_values, refit, controlled)
    all_reproduction = pd.concat([saved_stats, refit, human], ignore_index=True)
    all_reproduction.to_csv(OUT / "reproduced_correlations.csv", index=False)
    controlled.to_csv(controlled_path, index=False)
    controlled_summary = (
        controlled.groupby(["pipeline", "band"])[
            ["estimated_p_ap_rho", "true_center_to_estimated_exponent_rho"]
        ].agg(["mean", "std"]).reset_index()
    )
    controlled_summary.to_csv(OUT / "controlled_before_after_summary.csv", index=False)
    summary = {
        "figure": str(figure),
        "provided_simulation": saved_stats.to_dict(orient="records"),
        "same_psd_refit": refit.to_dict(orient="records"),
        "human_high_gamma": human.to_dict(orient="records"),
        "controlled_repeats": int(controlled.repeat.nunique()),
        "controlled_trials_per_band_per_repeat": int(controlled.n_trials.iloc[0]),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
