from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "wholebrain_extended_analyses_2026-08-29"

RED = "#D1495B"
BLUE = "#247BA0"
ORANGE = "#F28E2B"
TEAL = "#2A9D8F"
BLACK = "#222222"


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 21,
            "legend.fontsize": 21,
            "svg.fonttype": "none",
        }
    )
    handles = [
        Patch(facecolor=RED, edgecolor=BLACK, linewidth=3.0, alpha=0.52,
              label="Theory · Direct"),
        Patch(facecolor=BLUE, edgecolor=BLACK, linewidth=3.0, alpha=0.52,
              label="Theory · Non-direct"),
        Patch(facecolor=ORANGE, edgecolor=BLACK, hatch="///", alpha=0.52,
              linewidth=3.0, label="Simulation · Direct"),
        Patch(facecolor=TEAL, edgecolor=BLACK, hatch="///", alpha=0.52,
              linewidth=3.0, label="Simulation · Non-direct"),
    ]
    fig = plt.figure(figsize=(19.0, 1.25), facecolor="white")
    fig.legend(
        handles=handles,
        frameon=False,
        loc="center",
        ncol=4,
        columnspacing=1.15,
        handlelength=1.6,
        handleheight=1.15,
        prop={"family": "Arial", "size": 21, "weight": "bold"},
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / "deletion_four_bands_legend_only_single_row_bold.svg",
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.10,
    )
    fig.savefig(
        OUT / "deletion_four_bands_legend_only_single_row_bold_preview.png",
        dpi=200,
        facecolor="white",
        bbox_inches="tight",
        pad_inches=0.10,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
