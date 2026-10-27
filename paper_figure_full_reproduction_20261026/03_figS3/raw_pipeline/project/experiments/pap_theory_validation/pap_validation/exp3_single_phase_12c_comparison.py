from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import save_figure, save_json
from .pairwise_nonlinear_window_validation import _annotated_scatter


BASE = Path(__file__).resolve().parents[1]
WINDOW_PATHS = {
    "short": BASE / "outputs" / "exp3_single_phase_12c_short_rerun",
    "long": BASE / "outputs" / "exp3_single_phase_12c_long_rerun",
}


def run() -> Path:
    out = BASE / "outputs" / "exp3_single_phase_12c_comparison"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    data = {window: pd.read_csv(path / "delta_correlation_results.csv") for window, path in WINDOW_PATHS.items()}
    metrics = {window: json.loads((path / "metrics.json").read_text()) for window, path in WINDOW_PATHS.items()}
    manifests = {window: json.loads((path / "manifest.json").read_text()) for window, path in WINDOW_PATHS.items()}

    rows = []
    for window in ("short", "long"):
        for order in ("first", "second"):
            metric = metrics[window][f"delta_rho_{order}"]
            rows.append({
                "window": window,
                "order": order,
                "analysis_s": manifests[window]["analysis_s"],
                "n_input_trials_generated": manifests[window]["n_input_trials"],
                "n_common_valid_trials": metrics[window]["n_common_valid_trials"],
                "n_masks": manifests[window]["n_masks"],
                "identity_R2": metric["R2"],
                "ols_R2": metric["pearson_r"] ** 2,
                "slope": metric["slope"],
                "pearson_r": metric["pearson_r"],
                "regression_p_value": metric["regression_p_value"],
                "RMSE": metric["RMSE"],
            })
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "single_phase_identity_R2_summary.csv", index=False)
    save_json(out / "manifest.json", {
        "purpose": "exact Figure 12c single-phase accounting, expanded to first and second order",
        "input": "V1 E",
        "output": "PG E",
        "phase_seed_offset": 220000,
        "n_input_trials_generated": 64,
        "n_masks": 120,
        "common_random_number_inputs": True,
        "short_common_valid_trials": int(metrics["short"]["n_common_valid_trials"]),
        "long_common_valid_trials": int(metrics["long"]["n_common_valid_trials"]),
        "source_directories": {key: str(value) for key, value in WINDOW_PATHS.items()},
    })
    _plot(out, data, summary)
    return out


def _plot(out: Path, data: dict[str, pd.DataFrame], summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(13, 8.2), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=(1, 1, 0.78))
    colors = {"first": "#4C78A8", "second": "#E45756"}
    for row, window in enumerate(("short", "long")):
        for col, order in enumerate(("first", "second")):
            ax = fig.add_subplot(grid[row, col])
            frame = data[window]
            _annotated_scatter(
                ax,
                frame[f"delta_rho_{order}"].to_numpy(),
                frame.delta_rho_nonlinear.to_numpy(),
                f"{window.capitalize()} window, {order} order",
                colors[order],
            )
            n_valid = int(summary[(summary.window == window) & (summary.order == order)].n_common_valid_trials.iloc[0])
            ax.set(
                xlabel="Predicted Delta correlation",
                ylabel="Single-phase time-domain Delta correlation",
            )
            ax.text(0.96, 0.04, f"120 masks; {n_valid} input trials/r", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=7.5, color="0.3")
            ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)

    ax = fig.add_subplot(grid[:, 2])
    ordered = summary.set_index(["window", "order"]).loc[
        [("short", "first"), ("short", "second"), ("long", "first"), ("long", "second")]
    ].reset_index()
    labels = ["Short\nfirst", "Short\nsecond", "Long\nfirst", "Long\nsecond"]
    bar_colors = ["#9ECAE1", "#FCAE91", "#3182BD", "#CB181D"]
    bars = ax.bar(np.arange(4), ordered.identity_R2, color=bar_colors, width=0.68)
    ax.axhline(0, color="0.35", lw=1)
    ax.axhline(1, color="0.35", lw=1, ls="--", label="identity optimum")
    for bar, value in zip(bars, ordered.identity_R2):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.025 if value >= 0 else -0.025),
                f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.set_xticks(np.arange(4), labels)
    ax.set(ylabel="Identity $R^2$ across 120 masks", title="Absolute calibration summary")
    ax.set_ylim(-0.12, 1.08)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle("Experiment 3: exact Figure 12c single-phase accounting", fontsize=14)
    save_figure(fig, out / "figures" / "single_phase_short_long_first_second_identity_R2")
    plt.close(fig)


if __name__ == "__main__":
    print(run())
