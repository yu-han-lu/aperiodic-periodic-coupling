from __future__ import annotations

"""Plot the audited knee=2 correction and a replacement cross-band panel E."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import fooof_split_attached_code_theory as base
import reproduce_gamma_artifact as researcher


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "knee2_alpha_correction_2026-09-01"
SAME = (
    ROOT
    / "output"
    / "knee2_same_data_fooof_peak_count_check_2026-09-01"
    / "same_data_group_results.csv"
)
INDEPENDENT = (
    ROOT
    / "output"
    / "fooof_split_knee2_corrected_official3_six_panel_2026-09-01"
    / "all_group_results.csv"
)
FOOOF = "#247BA0"
SPLIT = "#2A9D8F"
ONE = "#D1495B"
DARK = "#333333"
BANDS = ["alpha", "beta", "low gamma", "high gamma"]
CENTERS = np.asarray([10.0, 21.5, 45.0, 105.0])


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(labelsize=16, width=2.3, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)


def interval(data: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = data.groupby("band")[column]
    mean = np.asarray([grouped.get_group(band).mean() for band in BANDS])
    low = np.asarray([grouped.get_group(band).quantile(0.025) for band in BANDS])
    high = np.asarray([grouped.get_group(band).quantile(0.975) for band in BANDS])
    return mean, low, high


def plot_replacement_e(independent: pd.DataFrame) -> Path:
    base.style()
    fig, ax = plt.subplots(figsize=(7.5, 6.0), constrained_layout=True)
    for display_band, (lo, hi, _) in researcher.BANDS.items():
        ax.axvspan(lo, hi, color=base.BAND_COLORS[display_band.lower()], alpha=0.075, lw=0)
    fm, flo, fhi = interval(independent, "fooof_emp")
    sm, slo, shi = interval(independent, "split_emp")
    ax.fill_between(CENTERS, flo, fhi, color=FOOOF, alpha=0.16, lw=0)
    ax.fill_between(CENTERS, slo, shi, color=SPLIT, alpha=0.16, lw=0)
    ax.plot(CENTERS, fm, "o-", color=FOOOF, lw=3.2, ms=9,
            label="FOOOF (official, max peaks = 3)")
    ax.plot(CENTERS, sm, "o-", color=SPLIT, lw=3.2, ms=9, label="SPLIT")
    ax.axhline(0, color="0.35", lw=2.0)
    ax.set_xscale("log")
    ax.set_xlabel("Oscillation band center (Hz)", fontsize=18, fontweight="bold")
    ax.set_ylabel(r"Estimated p–ap Spearman $\rho$", fontsize=18, fontweight="bold")
    ax.set_xticks([8, 12, 30, 60, 100, 150])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(-1.08, 1.08)
    ax.legend(frameon=False, fontsize=13.5, loc="lower right")
    ax.text(-0.14, 1.04, "E", transform=ax.transAxes, fontsize=24, fontweight="bold")
    style_axis(ax)
    path = OUT / "panel_E_knee2_corrected_official3_independent_OU.svg"
    preview = OUT / "panel_E_knee2_corrected_official3_independent_OU_preview.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(preview, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_audit(same: pd.DataFrame, independent: pd.DataFrame) -> Path:
    base.style()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
    held = same[same["sample"] == "heldout"]
    x = np.arange(len(BANDS))
    width = 0.24
    for offset, column, label, color in [
        (-width, "custom_sequential_onepeak_rho", "Sequential one-peak", ONE),
        (0.0, "official_max1_rho", "Official FOOOF, max=1", "#F4A261"),
        (width, "official_max3_rho", "Official FOOOF, max=3", FOOOF),
    ]:
        mean, low, high = interval(held, column)
        axes[0].bar(x + offset, mean, width=width * 0.92, color=color,
                    edgecolor="black", linewidth=1.6, label=label)
        axes[0].errorbar(x + offset, mean, yerr=[mean - low, high - mean], fmt="none",
                         color="black", lw=1.8, capsize=4, capthick=1.8)
    axes[0].axhline(0, color="0.35", lw=1.8)
    axes[0].set_xticks(x, ["Alpha", "Beta", "Low\ngamma", "High\ngamma"])
    axes[0].set_ylabel(r"p–ap Spearman $\rho$", fontsize=18, fontweight="bold")
    axes[0].set_title("Same spectra: peak-model check", fontsize=17, fontweight="bold")
    axes[0].legend(frameon=False, fontsize=11.5, loc="lower right")
    axes[0].set_ylim(-1.05, 1.08)

    fixed3 = interval(held, "official_max3_rho")
    independent3 = interval(independent, "fooof_emp")
    axes[1].bar(x - width / 2, fixed3[0], width=width * 1.5, color="#7A9CC6",
                edgecolor="black", linewidth=1.6, label="One fixed 30-s OU record")
    axes[1].errorbar(x - width / 2, fixed3[0],
                     yerr=[fixed3[0] - fixed3[1], fixed3[2] - fixed3[0]], fmt="none",
                     color="black", lw=1.8, capsize=4, capthick=1.8)
    axes[1].bar(x + width / 2, independent3[0], width=width * 1.5, color=FOOOF,
                edgecolor="black", linewidth=1.6, label="Independent OU record/group")
    axes[1].errorbar(x + width / 2, independent3[0],
                     yerr=[independent3[0] - independent3[1], independent3[2] - independent3[0]],
                     fmt="none", color="black", lw=1.8, capsize=4, capthick=1.8)
    axes[1].axhline(0, color="0.35", lw=1.8)
    axes[1].set_xticks(x, ["Alpha", "Beta", "Low\ngamma", "High\ngamma"])
    axes[1].set_ylabel(r"Official FOOOF p–ap $\rho$", fontsize=18, fontweight="bold")
    axes[1].set_title("Same FOOOF: OU-realisation check", fontsize=17, fontweight="bold")
    axes[1].legend(frameon=False, fontsize=11.5, loc="lower right")
    axes[1].set_ylim(-1.05, 1.08)
    for ax, label in zip(axes, ["A", "B"]):
        ax.text(-0.15, 1.04, label, transform=ax.transAxes, fontsize=22, fontweight="bold")
        style_axis(ax)
    path = OUT / "knee2_alpha_discrepancy_audit.svg"
    preview = OUT / "knee2_alpha_discrepancy_audit_preview.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(preview, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    same = pd.read_csv(SAME)
    independent = pd.read_csv(INDEPENDENT)
    e_path = plot_replacement_e(independent)
    audit_path = plot_audit(same, independent)
    alpha = independent[independent.band == "alpha"].fooof_emp.dropna().to_numpy()
    alpha_test = stats.ttest_1samp(alpha, 0.0)
    metrics = {
        "replacement_panel_e": str(e_path),
        "audit_figure": str(audit_path),
        "alpha_official3_independent_ou_mean_rho": float(alpha.mean()),
        "alpha_95pct_group_interval": [float(np.quantile(alpha, 0.025)), float(np.quantile(alpha, 0.975))],
        "alpha_one_sample_t": float(alpha_test.statistic),
        "alpha_one_sample_p": float(alpha_test.pvalue),
        "fixed_base_fitted_knee_hz": 3.6858803817041212,
        "independent_base_median_fitted_knee_hz": float(independent.fitted_base_knee_hz.median()),
        "interpretation": "The strong alpha negative value was caused by a one-peak estimator plus one atypical finite OU realization, not knee=2 itself.",
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
