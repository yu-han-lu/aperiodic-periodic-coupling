from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import save_figure
from .io import load_config, prepare_output
from .nonlinear_network_validation import run_exp1, run_exp3


COLORS = {
    "baseline_hann": "#4C78A8",
    "short_boxcar": "#E45756",
    "long_hann": "#54A24B",
}
LABELS = {
    "baseline_hann": "4.096 s, Hann-Welch 2.048 s",
    "short_boxcar": "4.096 s, full-record boxcar",
    "long_hann": "32.768 s, Hann-Welch 4.096 s",
}


def _root(cfg: dict) -> Path:
    return Path(__file__).resolve().parents[3] / cfg["output_dir"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _add_fit(ax, x: np.ndarray, y: np.ndarray, variant: str) -> dict[str, float]:
    fit = stats.linregress(x, y)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    p_text = f"{fit.pvalue:.1e}" if fit.pvalue < 0.001 else f"{fit.pvalue:.3f}"
    label = f"{LABELS[variant]}: b={fit.slope:.3g}, r={fit.rvalue:.3f}, p={p_text}"
    ax.scatter(x, y, s=27, alpha=0.68, color=COLORS[variant])
    ax.plot(xx, fit.intercept + fit.slope * xx, color=COLORS[variant], lw=2, label=label)
    identity_r2 = 1.0 - np.sum((x - y) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return {
        "slope": float(fit.slope),
        "pearson_r": float(fit.rvalue),
        "p_value": float(fit.pvalue),
        "identity_R2": float(identity_r2),
    }


def _metric_bars(ax, rows: pd.DataFrame, metric_names: list[str], title: str) -> None:
    x = np.arange(len(rows))
    width = 0.24
    palette = ["#4C78A8", "#F2CF5B", "#B279A2"]
    for offset, (metric, color) in enumerate(zip(metric_names, palette)):
        values = rows[metric].to_numpy(float)
        bars = ax.bar(x + (offset - 1) * width, values, width, label=metric, color=color, alpha=0.9)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(1.0, color="0.35", lw=1, ls="--")
    if np.nanmin(rows[metric_names].to_numpy(float)) < -3.0:
        ax.set_yscale("symlog", linthresh=0.5)
    ax.set_xticks(x, ["baseline", "boxcar", "long Hann"])
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)


def _figure_exp1(out: Path, root: Path, variants: list[str], summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.2))
    for variant in variants:
        data = pd.read_csv(root / _paths(variant)[0] / "condition_summary.csv")
        _add_fit(
            axes[0, 0], data.theory_linear_surrogate_r.to_numpy(), data.nonlinear_r.to_numpy(), variant
        )
        _add_fit(
            axes[1, 0], data.theory_C49.to_numpy(), data.nonlinear_covariance.to_numpy(), variant
        )
    for ax, title, xlabel, ylabel in [
        (axes[0, 0], "Correlation readout", "Linear-surrogate r", "Time-domain r"),
        (axes[1, 0], "Covariance readout", "Eq.49 covariance", "Time-domain covariance"),
    ]:
        lo, hi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        low, high = min(lo, ylo), max(hi, yhi)
        ax.plot([low, high], [low, high], "k--", lw=1, label="identity")
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        ax.legend(frameon=False, fontsize=7.5)
    exp1_rows = summary[(summary.experiment == "exp1") & (summary.target == "correlation")]
    _metric_bars(axes[0, 1], exp1_rows, ["slope", "pearson_r", "identity_R2"], "Correlation metrics")
    exp1_cov = summary[(summary.experiment == "exp1") & (summary.target == "covariance")]
    _metric_bars(axes[1, 1], exp1_cov, ["slope", "pearson_r", "identity_R2"], "Covariance metrics")
    fig.suptitle("Experiment 1: finite-record spectral-estimator controls", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out / "figures" / "finite_window_exp1")
    plt.close(fig)


def _figure_exp3(out: Path, root: Path, variants: list[str], summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for ax, variant in zip(axes[0], variants):
        data = pd.read_csv(root / _paths(variant)[1] / "delta_correlation_results.csv")
        fit = _add_fit(
            ax, data.delta_rho_second.to_numpy(), data.delta_rho_nonlinear.to_numpy(), variant
        )
        lo, hi = ax.get_xlim()
        ylo, yhi = ax.get_ylim()
        low, high = min(lo, ylo), max(hi, yhi)
        ax.plot([low, high], [low, high], "k--", lw=1)
        ax.set(
            xlabel="Second-order predicted Delta r",
            ylabel="Time-domain Delta r",
            title=LABELS[variant],
        )
        ax.text(
            0.04, 0.96,
            f"slope={fit['slope']:.3g}\nr={fit['pearson_r']:.3f}, p={fit['p_value']:.1e}\nidentity R2={fit['identity_R2']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8.5,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )

    exp3 = summary[(summary.experiment == "exp3") & (summary.target == "second_order")]
    x = np.arange(len(exp3))
    labels = ["baseline", "boxcar", "long Hann"]
    panels = [
        (axes[1, 0], "slope", "OLS slope", 1.0),
        (axes[1, 1], "identity_R2", "Identity R2", None),
        (axes[1, 2], "pearson_r", "Pearson r", 1.0),
    ]
    for ax, metric, title, reference in panels:
        values = exp3[metric].to_numpy(float)
        bars = ax.bar(x, values, color=[COLORS[v] for v in variants], alpha=0.9)
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8)
        if reference is not None:
            ax.axhline(reference, color="0.35", lw=1, ls="--")
        if metric == "slope":
            ax.set_yscale("symlog", linthresh=1.0)
        ax.set_xticks(x, labels)
        ax.set_title(title)
    fig.suptitle("Experiment 3: finite-record spectral-estimator controls", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out / "figures" / "finite_window_exp3")
    plt.close(fig)


def _paths(variant: str) -> tuple[str, str]:
    if variant == "baseline_hann":
        return "exp1_nonlinear_full55", "exp3_nonlinear_subnetwork"
    if variant == "short_boxcar":
        return "exp1_nonlinear_full55_boxcar", "exp3_nonlinear_subnetwork_boxcar"
    if variant == "long_hann":
        return "exp1_nonlinear_full55_long", "exp3_nonlinear_subnetwork_long"
    raise ValueError(variant)


def _collect(cfg: dict, variants: list[str]) -> pd.DataFrame:
    root = _root(cfg)
    rows = []
    for variant in variants:
        exp1_path, exp3_path = _paths(variant)
        m1 = _load_json(root / exp1_path / "metrics.json")
        m3 = _load_json(root / exp3_path / "metrics.json")
        for target, key in [
            ("covariance", "nonlinear_vs_C49_covariance"),
            ("correlation", "nonlinear_vs_linear_surrogate_correlation"),
        ]:
            metric = m1[key]
            rows.append({
                "experiment": "exp1", "variant": variant, "target": target,
                "analysis_s": 32.768 if variant == "long_hann" else 4.096,
                "n": metric["n"], "slope": metric["slope"], "pearson_r": metric["pearson_r"],
                "identity_R2": metric["R2"], "RMSE": metric["RMSE"],
                "sign_accuracy": metric["sign_accuracy"],
            })
        for target, key in [
            ("direct", "delta_rho_analytic_direct"),
            ("first_order", "delta_rho_first"),
            ("second_order", "delta_rho_second"),
        ]:
            metric = m3[key]
            rows.append({
                "experiment": "exp3", "variant": variant, "target": target,
                "analysis_s": 32.768 if variant == "long_hann" else 4.096,
                "n": metric["n"], "slope": metric["slope"], "pearson_r": metric["pearson_r"],
                "identity_R2": metric["R2"], "RMSE": metric["RMSE"],
                "sign_accuracy": metric["sign_accuracy"],
                "n_common_valid_trials": m3["n_common_valid_trials"],
            })
    return pd.DataFrame(rows)


def run(cfg: dict) -> Path:
    out = prepare_output(cfg, "finite_window_validation")
    fcfg = cfg.get("finite_window", {})
    short_s = float(fcfg.get("short_analysis_s", 4.096))
    long_s = float(fcfg.get("long_analysis_s", 32.768))
    long_segment_s = float(fcfg.get("long_segment_s", 4.096))
    smoothing_sigma_hz = float(fcfg.get("smoothing_sigma_hz", 0.9765625))

    short_cfg = copy.deepcopy(cfg)
    short_cfg["exp1_nonlinear"]["analysis_s"] = short_s
    short_cfg["exp3_nonlinear"]["analysis_s"] = short_s
    boxcar = {"mode": "boxcar_full", "smoothing_sigma_hz": smoothing_sigma_hz}
    run_exp1(short_cfg, "exp1_nonlinear_full55_boxcar", boxcar)
    run_exp3(short_cfg, "exp3_nonlinear_subnetwork_boxcar", boxcar)

    long_cfg = copy.deepcopy(cfg)
    long_cfg["exp1_nonlinear"]["analysis_s"] = long_s
    long_cfg["exp3_nonlinear"]["analysis_s"] = long_s
    long_hann = {
        "mode": "hann_welch", "segment_s": long_segment_s,
        "smoothing_sigma_hz": smoothing_sigma_hz,
    }
    run_exp1(long_cfg, "exp1_nonlinear_full55_long", long_hann)
    run_exp3(long_cfg, "exp3_nonlinear_subnetwork_long", long_hann)

    variants = ["baseline_hann", "short_boxcar", "long_hann"]
    summary = _collect(cfg, variants)
    summary.to_csv(out / "summary_metrics.csv", index=False)
    _figure_exp1(out, _root(cfg), variants, summary)
    _figure_exp3(out, _root(cfg), variants, summary)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
