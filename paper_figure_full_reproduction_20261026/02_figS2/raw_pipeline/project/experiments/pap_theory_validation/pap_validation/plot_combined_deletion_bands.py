from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "experiments/pap_theory_validation/outputs/deletion_frequency_band_analysis"
STATS = ROOT / "output/wholebrain_extended_analyses_2026-08-29/deletion_band_direct_vs_nondirect_statistics.csv"
OUT = ROOT / "output/wholebrain_extended_analyses_2026-08-29"

BANDS = ["Alpha", "Beta", "Low gamma", "High gamma"]
SOURCES = ["Theory", "Simulation"]
RED = "#D1495B"
BLUE = "#247BA0"
ORANGE = "#F28E2B"
TEAL = "#2A9D8F"
BLACK = "#222222"


def significance_stars(pvalue: float) -> str:
    if pvalue < 1e-4:
        return "****"
    if pvalue < 1e-3:
        return "***"
    if pvalue < 1e-2:
        return "**"
    if pvalue < 0.05:
        return "*"
    return "ns"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 22,
            "axes.labelsize": 24,
            "axes.titlesize": 24,
            "axes.linewidth": 4.2,
            "xtick.labelsize": 22,
            "ytick.labelsize": 22,
            "legend.fontsize": 21,
            "xtick.major.width": 4.0,
            "ytick.major.width": 4.0,
            "xtick.major.size": 10,
            "ytick.major.size": 10,
            "svg.fonttype": "none",
        }
    )


def main() -> None:
    style()
    units = pd.read_csv(DATA / "band_output_area_units.csv.gz")
    stats = pd.read_csv(STATS)
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.10, top=0.96)
    # Directness and source are both visible in colour: theory uses the
    # red/blue pair, simulation the orange/teal pair.
    colors = {
        ("Theory", True): RED,
        ("Theory", False): BLUE,
        ("Simulation", True): ORANGE,
        ("Simulation", False): TEAL,
    }
    offsets = {
        ("Theory", True): -0.30,
        ("Theory", False): -0.13,
        ("Simulation", True): 0.13,
        ("Simulation", False): 0.30,
    }
    hatches = {"Theory": "", "Simulation": "///"}
    annotation_tops: list[float] = []

    for band_position, band in enumerate(BANDS):
        for source in SOURCES:
            subset = units[(units.band == band) & (units.source == source)]
            for direct in (True, False):
                values = subset.loc[subset.direct == direct, "value"].to_numpy(float)
                values = values[np.isfinite(values) & (values > 0)]
                xpos = band_position + offsets[(source, direct)]
                box = ax.boxplot(
                    [values],
                    positions=[xpos],
                    widths=0.12,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": BLACK, "linewidth": 4.5},
                    whiskerprops={"linewidth": 3.6, "color": BLACK},
                    capprops={"linewidth": 3.6, "color": BLACK},
                )
                box["boxes"][0].set_facecolor(colors[(source, direct)])
                box["boxes"][0].set_alpha(0.46)
                box["boxes"][0].set_edgecolor(BLACK)
                box["boxes"][0].set_linewidth(4.0)
                box["boxes"][0].set_hatch(hatches[source])

            qvalue = float(
                stats.loc[(stats.band == band) & (stats.source == source), "sign_flip_fdr_q"].iloc[0]
            )
            source_rank = SOURCES.index(source)
            # Place annotations above the largest upper whisker, not relative
            # to a high raw quantile that can coincide with a box outline.
            upper_whiskers = []
            for direct in (True, False):
                group = subset.loc[(subset.direct == direct) & (subset.value > 0), "value"].to_numpy(float)
                q1, q3 = np.quantile(group, [0.25, 0.75])
                upper_whiskers.append(float(np.max(group[group <= q3 + 1.5 * (q3 - q1)])))
            top = max(upper_whiskers) * (1.55 if source_rank == 0 else 2.65)
            left = band_position + offsets[(source, True)]
            right = band_position + offsets[(source, False)]
            ax.plot(
                [left, left, right, right],
                [top / 1.16, top, top, top / 1.16],
                color=BLACK,
                lw=3.8,
                clip_on=False,
            )
            star_top = top * 1.11
            annotation_tops.append(star_top)
            ax.text((left + right) / 2, star_top, significance_stars(qvalue),
                    ha="center", va="bottom", fontsize=24, fontweight="bold")

    ax.set_yscale("log")
    ax.set_xticks([])
    ax.set_xlim(-0.55, len(BANDS) - 0.45)
    positive_all = units.loc[units.value > 0, "value"].to_numpy(float)
    # Extra logarithmic headroom keeps the tallest star glyph well below the
    # upper axis frame (text ascenders extend above their anchor coordinate).
    ax.set_ylim(float(positive_all.min() * 0.72), max(annotation_tops) * 4.00)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(width=4.0, length=10)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(4.2)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "deletion_four_bands_direct_vs_nondirect_single_axis_4x5_source_colors_no_band_labels_no_mean_diamonds_no_legend.svg",
                facecolor="white")
    fig.savefig(OUT / "deletion_four_bands_direct_vs_nondirect_single_axis_4x5_source_colors_no_band_labels_no_mean_diamonds_no_legend_preview.png",
                dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    main()
