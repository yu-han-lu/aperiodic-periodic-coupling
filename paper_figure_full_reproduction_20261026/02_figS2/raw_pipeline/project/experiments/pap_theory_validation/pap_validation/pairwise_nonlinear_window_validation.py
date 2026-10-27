from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    AREAS9,
    apply_fln_delta,
    finite_derivatives,
    frequency_grid,
    load_model,
    prediction_metrics,
    save_figure,
    save_json,
    transfer_power,
)
from .exp1 import Z0, ZHI, ZLO, ZSCALE, _draw_design, _feature_fun
from .exp3 import _input_trials, _linear_stats
from .io import load_config, prepare_output
from .nonlinear_network_validation import (
    _exp1_conditions,
    _fourier_inputs,
    _run_multioutput_batch,
    _timeseries_features,
)


WINDOWS = ("short", "long")
ORDERS = ("first", "second")
COLORS = {"first": "#4C78A8", "second": "#E45756"}
ROOT = Path(__file__).resolve().parents[3]


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def _window_specs(cfg: dict) -> dict[str, dict]:
    fcfg = cfg.get("finite_window", {})
    return {
        "short": {
            "analysis_s": float(fcfg.get("short_analysis_s", 4.096)),
            "spectral": {
                "mode": "hann_welch",
                "segment_s": 2.048,
                "smoothing_sigma_hz": float(fcfg.get("smoothing_sigma_hz", 0.9765625)),
            },
        },
        "long": {
            "analysis_s": float(fcfg.get("long_analysis_s", 32.768)),
            "spectral": {
                "mode": "hann_welch",
                "segment_s": float(fcfg.get("long_segment_s", 4.096)),
                "smoothing_sigma_hz": float(fcfg.get("smoothing_sigma_hz", 0.9765625)),
            },
        },
    }


def _feature_arrays(outputs: np.ndarray, dt_ms: float, specs: dict[str, dict]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    fs = 1000.0 / dt_ms
    result = {}
    for window, spec in specs.items():
        n_steps = int(round(spec["analysis_s"] * fs))
        values = np.empty((outputs.shape[1], outputs.shape[0], 2), dtype=float)
        valid = np.empty((outputs.shape[1], outputs.shape[0]), dtype=bool)
        for output_pos in range(outputs.shape[1]):
            frame, _ = _timeseries_features(outputs[:, output_pos, :n_steps], dt_ms, spec["spectral"])
            values[output_pos] = frame[["peak_omega", "exponent"]].to_numpy()
            valid[output_pos] = frame.peak_valid.to_numpy(bool) & np.isfinite(values[output_pos]).all(axis=1)
        result[window] = values, valid
    return result


def _taylor_values(fun, grad: np.ndarray, hess: np.ndarray, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    base = np.asarray(fun(np.zeros(u.shape[1])), float)
    first = base + u @ grad.T
    second = first.copy()
    for feature in range(second.shape[1]):
        second[:, feature] += 0.5 * np.einsum("ni,ij,nj->n", u, hess[feature], u)
    return first, second


def _cov_r(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 5 or np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0:
        return np.nan, np.nan
    return (
        float(np.cov(values.T, ddof=1)[0, 1]),
        float(stats.pearsonr(values[:, 0], values[:, 1]).statistic),
    )


def _fit_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(observed) & np.isfinite(predicted)
    observed = np.asarray(observed, float)[ok]
    predicted = np.asarray(predicted, float)[ok]
    result = prediction_metrics(observed, predicted)
    if len(observed) > 2 and np.std(predicted) > 0:
        fit = stats.linregress(predicted, observed)
        result.update({
            "ols_R2": float(fit.rvalue**2),
            "regression_p_value": float(fit.pvalue),
        })
    else:
        result.update({"ols_R2": np.nan, "regression_p_value": np.nan})
    return result


def _pair_metrics(rows: pd.DataFrame, observed: str, predicted: str) -> pd.DataFrame:
    metrics = []
    keys = ["window", "order", "input_area", "output_area"]
    for key, group in rows.groupby(keys, sort=False):
        metrics.append(dict(zip(keys, key)) | _fit_metrics(group[observed].to_numpy(), group[predicted].to_numpy()))
    return pd.DataFrame(metrics)


def run_exp1_pairwise(cfg: dict, n_trials: int | None = None) -> Path:
    started = time.time()
    out = prepare_output(cfg, "finite_window_exp1_pairwise_nonlinear")
    model = load_model(_rooted(cfg["model_dir"]))
    specs = _window_specs(cfg)
    ecfg = cfg["exp1_nonlinear"]
    pair_cfg = cfg.get("pairwise_window_nonlinear", {})
    n_trials = int(n_trials or pair_cfg.get("exp1_n_trials", ecfg["n_trials"]))
    dt = float(ecfg["dt_ms"])
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    long_steps = int(round(specs["long"]["analysis_s"] * 1000.0 / dt))
    output_indices = np.array([model.areas.index(area) for area in AREAS9], dtype=np.int64)
    freqs = frequency_grid()

    derivatives = {}
    for input_area in AREAS9:
        input_idx = model.areas.index(input_area)
        for output_area in AREAS9:
            output_idx = model.areas.index(output_area)
            transfer = transfer_power(model.J, input_idx, output_idx, freqs)
            fun = _feature_fun(transfer, freqs, tuple(cfg["peak_band_hz"]), tuple(cfg["slope_band_hz"]))
            grad, hess, _ = finite_derivatives(fun, np.zeros(4), h=2e-3)
            derivatives[(input_area, output_area)] = (fun, grad, hess)

    rows = []
    rectification = []
    for input_pos, input_area in enumerate(AREAS9):
        input_idx = model.areas.index(input_area)
        for condition_pos, (design, eps, rho) in enumerate(_exp1_conditions(cfg)):
            rng = np.random.default_rng(int(cfg["seed"]) + 1000 + condition_pos)
            u = _draw_design(design, n_trials, eps, rng, rho)
            raw = Z0 + u * ZSCALE
            z = np.clip(raw, ZLO, ZHI)
            clipped = np.any(raw != z, axis=1)
            inputs = _fourier_inputs(
                z, dt, long_steps, burn_steps, float(ecfg["input_sigma"]),
                int(cfg["seed"]) + 410000 + condition_pos,
            )
            outputs, rect, min_e, min_i, min_de, min_di = _run_multioutput_batch(
                model.params, inputs, dt, burn_steps, output_indices, input_idx=input_idx,
            )
            features = _feature_arrays(outputs, dt, specs)
            condition = f"{design}_eps{eps:g}_rho{rho:g}"
            rectification.append({
                "input_area": input_area, "condition": condition,
                "rectification_fraction": float(rect.sum() / (2 * len(model.areas) * inputs.shape[1] * n_trials)),
                "min_rate_E": float(min_e.min()), "min_rate_I": float(min_i.min()),
                "min_drive_E": float(min_de.min()), "min_drive_I": float(min_di.min()),
            })
            for output_pos, output_area in enumerate(AREAS9):
                fun, grad, hess = derivatives[(input_area, output_area)]
                for window in WINDOWS:
                    values, valid = features[window]
                    use = valid[output_pos] & ~clipped
                    accepted_u = (z[use] - Z0) / ZSCALE
                    actual_cov, actual_r = _cov_r(values[output_pos, use])
                    first, second = _taylor_values(fun, grad, hess, accepted_u)
                    for order, predicted_values in (("first", first), ("second", second)):
                        predicted_cov, predicted_r = _cov_r(predicted_values)
                        rows.append({
                            "input_area": input_area, "output_area": output_area,
                            "window": window, "order": order,
                            "condition": condition, "design": design, "epsilon": eps, "rho": rho,
                            "n_trials": n_trials, "n_valid": int(use.sum()),
                            "actual_covariance": actual_cov, "predicted_covariance": predicted_cov,
                            "actual_r": actual_r, "predicted_r": predicted_r,
                        })
            print(
                f"exp1 pairwise {input_pos + 1}/9 input={input_area}, "
                f"condition {condition_pos + 1}/{len(_exp1_conditions(cfg))}", flush=True,
            )

    results = pd.DataFrame(rows)
    covariance_metrics = _pair_metrics(results, "actual_covariance", "predicted_covariance")
    correlation_metrics = _pair_metrics(results, "actual_r", "predicted_r")
    results.to_csv(out / "condition_predictions.csv.gz", index=False, compression="gzip")
    covariance_metrics.to_csv(out / "covariance_R2_by_pair.csv", index=False)
    correlation_metrics.to_csv(out / "correlation_R2_by_pair.csv", index=False)
    pd.DataFrame(rectification).to_csv(out / "rectification_qc.csv", index=False)
    save_json(out / "manifest.json", {
        "experiment": "full55_threshold_nonlinear_9x9_input_output_finite_window",
        "actual_simulation": "explicit threshold-linear time-domain Euler integration; no resolvent values used as observations",
        "areas": AREAS9, "n_trials_per_condition": n_trials,
        "n_conditions": len(_exp1_conditions(cfg)), "dt_ms": dt,
        "burn_s": ecfg["burn_s"], "short_analysis_s": specs["short"]["analysis_s"],
        "long_analysis_s": specs["long"]["analysis_s"],
        "first_order": "linear Taylor feature map",
        "second_order": "complete quadratic Taylor feature map",
        "runtime_s": time.time() - started,
    })
    _plot_exp1(out, results, covariance_metrics, correlation_metrics)
    return out


def _delta_theory(model, sigma: np.ndarray, freqs: np.ndarray, inp: int, out: int, delta_j: np.ndarray) -> tuple[float, float]:
    base = _linear_stats(model.J, sigma, freqs, inp, out)
    h1 = 0.02
    first = (
        _linear_stats(model.J + h1 * delta_j, sigma, freqs, inp, out)
        - _linear_stats(model.J - h1 * delta_j, sigma, freqs, inp, out)
    ) / (2 * h1)
    h2 = 0.05
    second_derivative = (
        _linear_stats(model.J + h2 * delta_j, sigma, freqs, inp, out)
        - 2 * base
        + _linear_stats(model.J - h2 * delta_j, sigma, freqs, inp, out)
    ) / h2**2
    return float(first[1]), float(first[1] + 0.5 * second_derivative[1])


def run_exp3_pairwise(
    cfg: dict,
    n_trials: int | None = None,
    n_masks: int | None = None,
    output_name: str = "finite_window_exp3_pairwise_nonlinear",
    phase_seed_offset: int = 520000,
) -> Path:
    started = time.time()
    out = prepare_output(cfg, output_name)
    model = load_model(_rooted(cfg["model_dir"]), AREAS9)
    specs = _window_specs(cfg)
    ecfg = cfg["exp3_nonlinear"]
    pair_cfg = cfg.get("pairwise_window_nonlinear", {})
    n_trials = int(n_trials or pair_cfg.get("exp3_n_trials", 64))
    n_masks = int(n_masks or pair_cfg.get("exp3_n_masks", 40))
    dt = float(ecfg["dt_ms"])
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    long_steps = int(round(specs["long"]["analysis_s"] * 1000.0 / dt))
    output_indices = np.arange(len(AREAS9), dtype=np.int64)
    z, u = _input_trials(n_trials, int(cfg["seed"]) + 300)
    source_trials = pd.read_csv(_rooted(cfg["output_dir"]) / "exp3_structural_kernel" / "trials.csv.gz")
    selected = source_trials[(source_trials.kind == "random15_delete") & (source_trials.split == "heldout")].head(n_masks)
    masks = []
    for source_row in selected.itertuples(index=False):
        delta_w = np.zeros_like(model.fln)
        for source, target, weight in json.loads(source_row.edges_json):
            delta_w[model.areas.index(target), model.areas.index(source)] = -float(weight)
        masks.append((source_row.perturbation_id, delta_w, apply_fln_delta(model, delta_w) - model.J))

    # Arrays are [network (baseline + masks), output, trial, feature].
    all_features = {
        window: np.empty((n_masks + 1, len(AREAS9), n_trials, 2), dtype=float) for window in WINDOWS
    }
    all_valid = {
        window: np.empty((n_masks + 1, len(AREAS9), n_trials), dtype=bool) for window in WINDOWS
    }
    rows = []
    qc = []
    freqs = frequency_grid()
    for input_pos, input_area in enumerate(AREAS9):
        inputs = _fourier_inputs(
            z, dt, long_steps, burn_steps, float(ecfg["input_sigma"]),
            int(cfg["seed"]) + int(phase_seed_offset),
        )
        networks = [("baseline", model.params, None)]
        for perturbation_id, delta_w, delta_j in masks:
            params = copy.copy(model.params)
            params["fln_mat"] = model.fln + delta_w
            networks.append((perturbation_id, params, delta_j))

        for network_pos, (network_id, params, _) in enumerate(networks):
            outputs, rect, min_e, min_i, min_de, min_di = _run_multioutput_batch(
                params, inputs, dt, burn_steps, output_indices, input_idx=input_pos,
            )
            extracted = _feature_arrays(outputs, dt, specs)
            for window, (values, valid) in extracted.items():
                all_features[window][network_pos] = values
                all_valid[window][network_pos] = valid
            qc.append({
                "input_area": input_area, "network": network_id,
                "rectification_fraction": float(rect.sum() / (2 * len(model.areas) * inputs.shape[1] * n_trials)),
                "min_rate_E": float(min_e.min()), "min_rate_I": float(min_i.min()),
                "min_drive_E": float(min_de.min()), "min_drive_I": float(min_di.min()),
            })
            if network_pos % 10 == 0 or network_pos == n_masks:
                print(f"exp3 pairwise input={input_area}: network {network_pos}/{n_masks}", flush=True)

        for output_pos, output_area in enumerate(AREAS9):
            for window in WINDOWS:
                global_valid = np.all(all_valid[window][:, output_pos, :], axis=0)
                if global_valid.sum() < 8:
                    continue
                sigma = np.cov(u[global_valid], rowvar=False, ddof=1)
                baseline_values = all_features[window][0, output_pos, global_valid]
                _, baseline_r = _cov_r(baseline_values)
                for mask_pos, (perturbation_id, _, delta_j) in enumerate(masks, 1):
                    _, perturbed_r = _cov_r(all_features[window][mask_pos, output_pos, global_valid])
                    delta_actual = perturbed_r - baseline_r
                    delta_first, delta_second = _delta_theory(
                        model, sigma, freqs, input_pos, output_pos, delta_j,
                    )
                    rows.append({
                        "input_area": input_area, "output_area": output_area,
                        "window": window, "perturbation_id": perturbation_id,
                        "n_input_trials": n_trials, "n_valid_common": int(global_valid.sum()),
                        "baseline_actual_r": baseline_r, "perturbed_actual_r": perturbed_r,
                        "delta_rho_actual": delta_actual,
                        "delta_rho_first": delta_first, "delta_rho_second": delta_second,
                    })

    results = pd.DataFrame(rows)
    long_rows = []
    for order in ORDERS:
        frame = results.assign(order=order, predicted=results[f"delta_rho_{order}"])
        long_rows.append(frame)
    stacked = pd.concat(long_rows, ignore_index=True)
    metrics = _pair_metrics(stacked, "delta_rho_actual", "predicted")
    results.to_csv(out / "delta_correlation_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "delta_correlation_R2_by_pair.csv", index=False)
    pd.DataFrame(qc).to_csv(out / "rectification_qc.csv", index=False)
    save_json(out / "manifest.json", {
        "experiment": "nine_area_threshold_nonlinear_9x9_structural_finite_window",
        "actual_simulation": "explicit threshold-linear time-domain Euler integration; no resolvent values used as observations",
        "areas": AREAS9, "n_input_trials": n_trials, "n_masks": n_masks,
        "phase_seed_offset": int(phase_seed_offset),
        "mask_selection": "first n pre-existing held-out random15_delete masks",
        "common_random_inputs": True,
        "validity_rule": "per input-output-window intersection across baseline and all selected masks",
        "dt_ms": dt, "burn_s": ecfg["burn_s"],
        "short_analysis_s": specs["short"]["analysis_s"],
        "long_analysis_s": specs["long"]["analysis_s"],
        "first_order": "directional structural Taylor term",
        "second_order": "first-order term plus one-half directional Hessian term",
        "runtime_s": time.time() - started,
    })
    _plot_exp3(out, results, metrics)
    return out


def _annotated_scatter(ax, x: np.ndarray, y: np.ndarray, title: str, color: str) -> None:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[ok], np.asarray(y)[ok]
    ax.scatter(x, y, s=28, alpha=0.72, color=color)
    if len(x) > 2 and np.std(x) > 0:
        fit = stats.linregress(x, y)
        xx = np.linspace(x.min(), x.max(), 100)
        ax.plot(xx, fit.intercept + fit.slope * xx, color=color, lw=2)
        metric = prediction_metrics(y, x)
        p_text = f"{fit.pvalue:.1e}" if fit.pvalue < 0.001 else f"{fit.pvalue:.3f}"
        ax.text(
            0.04, 0.96,
            f"b={fit.slope:.3g}, p={p_text}\nOLS R2={fit.rvalue**2:.3f}\nidentity R2={metric['R2']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
    lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
    hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_title(title, fontsize=10)


def _heatmap_grid(metrics: pd.DataFrame, value: str, title: str, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    for ax, window, order in zip(axes.flat, ("short", "short", "long", "long"), ("first", "second", "first", "second")):
        data = metrics[(metrics.window == window) & (metrics.order == order)]
        matrix = data.pivot(index="output_area", columns="input_area", values=value).reindex(index=AREAS9, columns=AREAS9)
        values = matrix.to_numpy(float)
        if value == "ols_R2":
            im = ax.imshow(values, cmap="viridis", vmin=0, vmax=1)
        elif value == "R2":
            # Keep a common interpretable scale while preserving raw values in cell labels.
            im = ax.imshow(
                np.clip(values, -2.5, 1.0), cmap="RdYlBu",
                norm=TwoSlopeNorm(vmin=-2.5, vcenter=0.0, vmax=1.0),
            )
        else:
            finite = np.abs(values[np.isfinite(values)])
            vmax = max(1.0, float(np.percentile(finite, 95))) if len(finite) else 1.0
            im = ax.imshow(values, cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(9), AREAS9, rotation=45, ha="right")
        ax.set_yticks(range(9), AREAS9)
        ax.set(xlabel="input area", ylabel="output area", title=f"{window.capitalize()} window, {order} order")
        for row in range(9):
            for col in range(9):
                if np.isfinite(values[row, col]):
                    shown = f"{values[row, col]:.2f}" if values[row, col] > -10 else f"{values[row, col]:.0f}"
                    if value == "ols_R2":
                        text_color = "white" if values[row, col] < 0.45 else "black"
                    elif value == "R2":
                        text_color = "white" if values[row, col] < -0.9 else "black"
                    else:
                        text_color = "black"
                    ax.text(col, row, shown, ha="center", va="center", fontsize=6.3, color=text_color)
        color_label = "identity R2 (color clipped at -2.5)" if value == "R2" else value.replace("_", " ")
        fig.colorbar(im, ax=ax, shrink=0.76, label=color_label)
    fig.suptitle(title, fontsize=13)
    save_figure(fig, output_path)
    plt.close(fig)


def _plot_exp1(out: Path, results: pd.DataFrame, covariance_metrics: pd.DataFrame, correlation_metrics: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    chosen = results[(results.input_area == "V1") & (results.output_area == "PG")]
    for target, observed, predicted in (
        ("covariance", "actual_covariance", "predicted_covariance"),
        ("correlation", "actual_r", "predicted_r"),
    ):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
        for ax, window, order in zip(axes.flat, ("short", "short", "long", "long"), ("first", "second", "first", "second")):
            data = chosen[(chosen.window == window) & (chosen.order == order)]
            _annotated_scatter(ax, data[predicted].to_numpy(), data[observed].to_numpy(),
                               f"{window.capitalize()} window, {order} order", COLORS[order])
            ax.set(xlabel=f"Predicted {target}", ylabel=f"Actual nonlinear {target}")
        fig.suptitle(f"Experiment 1, V1 input -> PG output: {target}", fontsize=13)
        save_figure(fig, out / "figures" / f"exp1_V1_PG_{target}_short_long_orders")
        plt.close(fig)
    _heatmap_grid(covariance_metrics, "ols_R2", "Experiment 1: covariance OLS R2 across 9 inputs x 9 outputs", out / "figures" / "exp1_covariance_9x9_R2")
    _heatmap_grid(correlation_metrics, "ols_R2", "Experiment 1: correlation OLS R2 across 9 inputs x 9 outputs", out / "figures" / "exp1_correlation_9x9_R2")
    _heatmap_grid(covariance_metrics, "R2", "Experiment 1: covariance identity R2 across 9 inputs x 9 outputs", out / "figures" / "exp1_covariance_9x9_identity_R2")
    _heatmap_grid(correlation_metrics, "R2", "Experiment 1: correlation identity R2 across 9 inputs x 9 outputs", out / "figures" / "exp1_correlation_9x9_identity_R2")


def _plot_identity_comparison(out: Path, exp1_metrics: pd.DataFrame, exp3_metrics: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    rng = np.random.default_rng(20260802)
    all_identity = np.concatenate([
        exp1_metrics.R2.to_numpy(float), exp3_metrics.R2.to_numpy(float),
    ])
    finite_identity = all_identity[np.isfinite(all_identity)]
    lower = min(-300.0, float(finite_identity.min()) * 1.15) if len(finite_identity) else -300.0
    for ax, order in zip(axes, ORDERS):
        groups = []
        labels = []
        colors = []
        for experiment, data, color_pair in (
            ("Exp. 1 r", exp1_metrics, ("#9ECAE1", "#3182BD")),
            ("Exp. 3 Delta r", exp3_metrics, ("#FCAE91", "#CB181D")),
        ):
            for window, color in zip(WINDOWS, color_pair):
                values = data[(data.order == order) & (data.window == window)].R2.to_numpy(float)
                groups.append(values)
                labels.append(f"{experiment}\n{window}")
                colors.append(color)
        box = ax.boxplot(groups, patch_artist=True, widths=0.58, showfliers=False, medianprops={"color": "black", "lw": 1.4})
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.78)
        for pos, (values, color) in enumerate(zip(groups, colors), 1):
            jitter = rng.uniform(-0.16, 0.16, len(values))
            ax.scatter(pos + jitter, values, s=10, alpha=0.36, color=color, edgecolors="none")
            ax.text(pos, 1.12, f"median={np.median(values):.2f}", ha="center", va="bottom", fontsize=7.5)
        ax.axhline(1.0, color="0.25", ls="--", lw=1, label="identity optimum")
        ax.axhline(0.0, color="0.45", ls=":", lw=1, label="R2=0")
        ax.set_yscale("symlog", linthresh=0.5, linscale=0.8)
        ax.set_ylim(lower, 1.65)
        ax.set_xticks(range(1, 5), labels)
        ax.set(ylabel="Identity R2 across 81 input-output pairs", title=f"{order.capitalize()} order")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Short versus long windows: identity R2 in Experiments 1 and 3", fontsize=13)
    save_figure(fig, out / "figures" / "exp1_exp3_identity_R2_comparison")
    plt.close(fig)


def _plot_exp3(out: Path, results: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    detailed = {
        "short": pd.read_csv(
            ROOT / "experiments/pap_theory_validation/outputs/exp3_nonlinear_subnetwork/delta_correlation_results.csv"
        ),
        "long": pd.read_csv(
            ROOT / "experiments/pap_theory_validation/outputs/exp3_nonlinear_subnetwork_long/delta_correlation_results.csv"
        ),
    }
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, window, order in zip(axes.flat, ("short", "short", "long", "long"), ("first", "second", "first", "second")):
        data = detailed[window]
        _annotated_scatter(ax, data[f"delta_rho_{order}"].to_numpy(), data.delta_rho_nonlinear.to_numpy(),
                           f"{window.capitalize()} window, {order} order", COLORS[order])
        ax.set(xlabel="Predicted Delta correlation", ylabel="Actual nonlinear Delta correlation")
    fig.suptitle("Experiment 3, V1 input -> PG output: structural Delta correlation", fontsize=13)
    save_figure(fig, out / "figures" / "exp3_V1_PG_delta_correlation_short_long_orders")
    plt.close(fig)
    _heatmap_grid(metrics, "ols_R2", "Experiment 3: Delta-correlation OLS R2 across 9 inputs x 9 outputs", out / "figures" / "exp3_delta_correlation_9x9_R2")
    _heatmap_grid(metrics, "R2", "Experiment 3: Delta-correlation identity R2 across 9 inputs x 9 outputs", out / "figures" / "exp3_delta_correlation_9x9_identity_R2")
    exp1_metrics = pd.read_csv(
        ROOT / "experiments/pap_theory_validation/outputs/finite_window_exp1_pairwise_nonlinear/correlation_R2_by_pair.csv"
    )
    _plot_identity_comparison(out, exp1_metrics, metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", choices=("exp1", "exp3", "all"), default="all")
    parser.add_argument("--n-trials", type=int)
    parser.add_argument("--n-masks", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.experiment in ("exp1", "all"):
        run_exp1_pairwise(cfg, args.n_trials)
    if args.experiment in ("exp3", "all"):
        run_exp3_pairwise(cfg, args.n_trials, args.n_masks)


if __name__ == "__main__":
    main()
