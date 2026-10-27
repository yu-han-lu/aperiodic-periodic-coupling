from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, stats

import reproduce_gamma_artifact as rep


ROOT = Path(__file__).resolve().parents[3]
DATA = Path("/PORTABLE_HOME/Desktop/data")
OUT = ROOT / "output" / "fooof_split_sequential_official_2026-09-01"
BLUE = "#247BA0"
TEAL = "#2A9D8F"
RED = "#D1495B"
PURPLE = "#6A4C93"


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.15, 1.08, label, transform=ax.transAxes,
            fontsize=24, fontweight="bold", ha="left", va="bottom")


def thicken(ax: plt.Axes) -> None:
    ax.tick_params(width=2.3, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)


def sequential_stages(
    y: np.ndarray, freq: np.ndarray, peak_range: tuple[float, float]
) -> dict[str, np.ndarray]:
    first = rep.fit_ap_knee(y, freq)
    flat_positive = np.maximum(y - rep.ap_knee(freq, first), 0.0)
    baseline = flat_positive <= np.percentile(flat_positive, 2.5)
    robust = rep.fit_ap_knee(y[baseline], freq[baseline], first)
    flattened = y - rep.ap_knee(freq, robust)
    peak_mask = (freq >= peak_range[0]) & (freq <= peak_range[1])
    center0 = float(freq[peak_mask][np.argmax(flattened[peak_mask])])
    height0 = float(np.clip(max(flattened[peak_mask].max(), 0.60), 0.01, 2.95))
    peak = rep.optimize.least_squares(
        lambda x: flattened - rep.gaussian(freq, x[0], x[1], x[2]),
        np.array([center0, height0, 1.5]),
        bounds=([peak_range[0], 0.0, 0.75], [peak_range[1], 3.0, 3.0]),
        max_nfev=180,
    ).x
    peak_curve = rep.gaussian(freq, *peak)
    final_ap = rep.fit_ap_knee(y - peak_curve, freq, robust)
    return {
        "first": first,
        "robust": robust,
        "flattened": flattened,
        "peak": peak,
        "peak_curve": peak_curve,
        "final": final_ap,
        "cleaned": y - peak_curve,
        "baseline": baseline,
    }


def load_simulation() -> dict[str, np.ndarray]:
    with h5py.File(DATA / "simulate_data" / "irasa_data.mat", "r") as handle:
        freq = np.asarray(handle["f"]).squeeze()
        original = np.asarray(handle["original_powspectra"])
        fractal = np.asarray(handle["fractal_powspectra"])
    mask = (freq >= 0.2) & (freq <= 200.0)
    saved = io.loadmat(DATA / "simulate_data" / "fooof_result.mat", squeeze_me=True)
    split_ap = io.loadmat(DATA / "simulate_data" / "split_exponent.mat",
                          squeeze_me=True, struct_as_record=False)
    split_peak = io.loadmat(DATA / "simulate_data" / "split_oscillation.mat",
                            squeeze_me=True, struct_as_record=False)
    return {
        "freq": freq[mask],
        "original": np.log10(original[mask, :].T),
        "fractal": np.log10(fractal[mask, :].T),
        "fooof": np.column_stack([saved["peak_cf"], saved["aperiodic_exponent"]]),
        "split": np.column_stack(
            [rep.peak_cells(split_peak["peak_sum"]),
             rep.scalar_cells(split_ap["exponent_sum"])]
        ),
    }


def refit_sequential(values: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(
        [rep.fit_sequential_knee_fooof(row, values["freq"], (30.0, 150.0))
         for row in values["original"]]
    )


def bootstrap_validation(
    official: np.ndarray, sequential: np.ndarray, n_boot: int = 500
) -> pd.DataFrame:
    good = np.isfinite(official).all(axis=1) & np.isfinite(sequential).all(axis=1)
    official, sequential = official[good], sequential[good]
    rng = np.random.default_rng(20260901)
    rows = []
    for boot in range(n_boot):
        idx = rng.choice(len(official), size=60, replace=True)
        rows.append(
            {
                "bootstrap": boot,
                "official_rho": float(stats.spearmanr(
                    official[idx, 0], official[idx, 1]).statistic),
                "sequential_rho": float(stats.spearmanr(
                    sequential[idx, 0], sequential[idx, 1]).statistic),
            }
        )
    return pd.DataFrame(rows)


def draw_mechanism_panel(
    ax: plt.Axes, freq: np.ndarray, y: np.ndarray, stages: dict[str, np.ndarray]
) -> None:
    ax.set_axis_off()
    inset = [
        ax.inset_axes([0.05, 0.57, 0.42, 0.34]),
        ax.inset_axes([0.55, 0.57, 0.42, 0.34]),
        ax.inset_axes([0.05, 0.08, 0.42, 0.34]),
        ax.inset_axes([0.55, 0.08, 0.42, 0.34]),
    ]
    first_curve = rep.ap_knee(freq, stages["first"])
    robust_curve = rep.ap_knee(freq, stages["robust"])
    final_curve = rep.ap_knee(freq, stages["final"])

    sub = inset[0]
    sub.plot(freq, y, color="#444444", lw=1.5, label="PSD")
    sub.plot(freq, first_curve, color=RED, lw=2.2, label="Initial AP")
    sub.plot(freq, robust_curve, color=PURPLE, lw=2.2, label="Robust AP")
    sub.set(xscale="log", xlim=(0.2, 200), title="B1  Robust AP stage")
    sub.legend(frameon=False, fontsize=8.5, loc="lower left")

    sub = inset[1]
    sub.plot(freq, stages["flattened"], color="#777777", lw=1.4,
             label="Flattened PSD")
    sub.plot(freq, stages["peak_curve"], color=BLUE, lw=2.3,
             label="Gaussian peak")
    sub.axhline(0, color="0.45", lw=1.0)
    sub.set(xlim=(30, 150), title="B2  Peak stage")
    sub.legend(frameon=False, fontsize=8.5, loc="upper left")

    sub = inset[2]
    sub.plot(freq, stages["cleaned"], color="#555555", lw=1.4,
             label="PSD − peak")
    sub.plot(freq, final_curve, color=RED, lw=2.3, label="Final AP")
    sub.set(xscale="log", xlim=(0.2, 200), title="B3  AP refit stage")
    sub.legend(frameon=False, fontsize=8.5, loc="lower left")

    sub = inset[3]
    delta = final_curve - robust_curve
    sub.fill_between(freq, 0, delta, color=RED, alpha=0.35)
    sub.plot(freq, delta, color=RED, lw=2.0)
    sub.axhline(0, color="0.35", lw=1.0)
    sub.set(xscale="log", xlim=(0.2, 200), title="B4  Peak error → AP shift")

    for index, sub in enumerate(inset):
        if index == 1:
            sub.set_xticks([30, 60, 100, 150])
        else:
            sub.set_xticks([0.2, 1, 10, 100, 200])
            sub.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        sub.tick_params(labelsize=8.5, width=1.4, length=3.5)
        for spine in sub.spines.values():
            spine.set_linewidth(1.5)


def main() -> None:
    rep.style()
    OUT.mkdir(parents=True, exist_ok=True)
    values = load_simulation()
    sequential = refit_sequential(values)
    boot = bootstrap_validation(values["fooof"], sequential)
    controlled = pd.read_csv(
        ROOT / "output/fooof_gamma_discrepancy_2026-09-01/controlled_before_after_repeats.csv"
    )
    pipeline = "Researcher pipeline: global sequential knee"
    controlled = controlled[controlled.pipeline == pipeline]

    target = int(np.nanargmin(np.abs(values["fooof"][:, 0] - 110.0)))
    stages = sequential_stages(values["original"][target], values["freq"], (30.0, 150.0))

    fig, axes = plt.subplots(2, 3, figsize=(18.5, 10.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(values["freq"], values["original"][target], color="#333333",
            lw=1.8, label="Original log PSD")
    ax.plot(values["freq"], values["fractal"][target], color=RED,
            lw=2.6, label="IRASA fractal")
    ax.plot(values["freq"], values["original"][target] - values["fractal"][target],
            color=BLUE, lw=2.3, label="Oscillatory residual")
    ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel="log10 power")
    ax.set_xticks([0.2, 1, 10, 30, 100, 200])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, loc="lower left")
    panel_label(ax, "A")

    draw_mechanism_panel(axes[0, 1], values["freq"],
                         values["original"][target], stages)
    panel_label(axes[0, 1], "B")

    ax = axes[0, 2]
    for method, array, color in [
        ("FOOOF", values["fooof"], BLUE),
        ("SPLIT", values["split"], TEAL),
    ]:
        good = np.isfinite(array).all(axis=1)
        test = stats.spearmanr(array[good, 0], array[good, 1])
        ax.scatter(array[good, 0], array[good, 1], s=32, alpha=0.68,
                   color=color, edgecolor="white", linewidth=0.35,
                   label=f"{method}: ρ = {test.statistic:.2f}")
    ax.set(xlabel="Estimated peak center (Hz)", ylabel="Estimated exponent")
    ax.legend(frameon=False, loc="best")
    panel_label(ax, "C")

    ax = axes[1, 0]
    ax.scatter(boot.official_rho, boot.sequential_rho, s=24, alpha=0.28,
               color=BLUE, edgecolor="none")
    limits = np.r_[boot.official_rho, boot.sequential_rho]
    lo, hi = float(limits.min() - 0.03), float(limits.max() + 0.03)
    ax.plot([lo, hi], [lo, hi], color="#333333", ls="--", lw=2.0)
    fit = stats.linregress(boot.official_rho, boot.sequential_rho)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Official FOOOF bootstrap ρ",
           ylabel="Sequential reproduction ρ")
    ax.text(0.04, 0.95,
            f"Pearson r = {fit.rvalue:.2f}\nOLS slope = {fit.slope:.2f}; $R^2$ = {fit.rvalue**2:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontweight="bold")
    panel_label(ax, "D")

    band_order = list(rep.BANDS)
    summary = controlled.groupby("band").agg(
        rho_mean=("estimated_p_ap_rho", "mean"),
        rho_std=("estimated_p_ap_rho", "std"),
        bias_mean=("true_center_to_estimated_exponent_rho", "mean"),
        bias_std=("true_center_to_estimated_exponent_rho", "std"),
    ).loc[band_order]
    x = np.arange(len(band_order))
    colors = [rep.COLORS[name] for name in band_order]

    ax = axes[1, 1]
    ax.errorbar(x, summary.rho_mean, yerr=summary.rho_std, fmt="none",
                ecolor="#222222", lw=2.2, capsize=5)
    ax.scatter(x, summary.rho_mean, s=120, color=colors,
               edgecolor="#222222", linewidth=0.9, zorder=3)
    ax.axhline(0, color="0.35", lw=1.5)
    ax.set(xticks=x, xticklabels=["Alpha", "Beta", "Low\ngamma", "High\ngamma"],
           ylabel="Estimated p–ap Spearman rho", ylim=(-1.08, 1.08))
    panel_label(ax, "E")

    ax = axes[1, 2]
    ax.errorbar(x, summary.bias_mean, yerr=summary.bias_std, fmt="none",
                ecolor="#222222", lw=2.2, capsize=5)
    ax.scatter(x, summary.bias_mean, s=120, color=colors,
               edgecolor="#222222", linewidth=0.9, zorder=3)
    ax.axhline(0, color="0.35", lw=1.5)
    ax.set(xticks=x, xticklabels=["Alpha", "Beta", "Low\ngamma", "High\ngamma"],
           ylabel="True center–estimated exponent rho", ylim=(-1.08, 1.08))
    panel_label(ax, "F")

    for ax in axes.ravel():
        if ax.axison:
            thicken(ax)
    path = OUT / "fooof_split_sequential_knee_six_panel.svg"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fooof_split_sequential_knee_six_panel_preview.png",
                dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    boot.to_csv(OUT / "official_vs_sequential_bootstrap.csv", index=False)
    print(path)


if __name__ == "__main__":
    main()
