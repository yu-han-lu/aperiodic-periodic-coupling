from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import feature_from_spectrum, frequency_grid, input_psd, load_model, prediction_metrics, save_figure, save_json
from .exp3 import Z0, ZS, _input_trials
from .exp3_dt_convergence import DT_VALUES
from .exp3_targeted_degree10 import GROUPS, N_MASKS, N_TRIALS, _targeted_masks, _transfer_power_eig
from .io import load_config, prepare_output
from .pairwise_nonlinear_window_validation import _rooted


OUTPUT_NAME = "exp3_absolute_correlation"
BOOTSTRAP = 1000
ORDERS = ("first", "second", "exact_resolvent")
H1 = 0.02
H2 = 0.05


def _feature_gradients(transfer: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    gradients = np.empty((len(transfer), 2, 4), dtype=float)
    h = 2e-3
    for output_pos, output_transfer in enumerate(transfer):
        for coordinate in range(4):
            perturbation = np.zeros(4)
            perturbation[coordinate] = h
            values = []
            for u in (perturbation, -perturbation):
                spectrum = output_transfer * input_psd(freqs, Z0 + ZS * u)
                feature = feature_from_spectrum(freqs, spectrum)
                values.append([feature["peak_omega"], feature["exponent"]])
            gradients[output_pos, :, coordinate] = (
                np.asarray(values[0]) - np.asarray(values[1])
            ) / (2 * h)
    return gradients


def _rho_from_gradients(gradients: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    gp = gradients[:, 0, :]
    ga = gradients[:, 1, :]
    covariance = np.einsum("ni,ij,nj->n", gp, sigma, ga)
    variance_p = np.einsum("ni,ij,nj->n", gp, sigma, gp)
    variance_a = np.einsum("ni,ij,nj->n", ga, sigma, ga)
    return covariance / np.sqrt(np.maximum(variance_p * variance_a, 1e-30))


def _correlations(features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    x = features[:, indices, 0]
    y = features[:, indices, 1]
    x = x - x.mean(axis=1, keepdims=True)
    y = y - y.mean(axis=1, keepdims=True)
    numerator = np.sum(x * y, axis=1)
    denominator = np.sqrt(np.sum(x * x, axis=1) * np.sum(y * y, axis=1))
    return numerator / np.maximum(denominator, 1e-30)


def _theory_cache(model, masks, input_idx: int, output_indices: list[int], freqs: np.ndarray):
    cache = {
        "base": _feature_gradients(
            _transfer_power_eig(model.J, input_idx, output_indices, freqs), freqs
        )
    }
    for mask_id, _, delta_j, _ in masks:
        cache[mask_id] = {
            "exact": _feature_gradients(
                _transfer_power_eig(model.J + delta_j, input_idx, output_indices, freqs), freqs
            ),
            "plus1": _feature_gradients(
                _transfer_power_eig(model.J + H1 * delta_j, input_idx, output_indices, freqs), freqs
            ),
            "minus1": _feature_gradients(
                _transfer_power_eig(model.J - H1 * delta_j, input_idx, output_indices, freqs), freqs
            ),
            "plus2": _feature_gradients(
                _transfer_power_eig(model.J + H2 * delta_j, input_idx, output_indices, freqs), freqs
            ),
            "minus2": _feature_gradients(
                _transfer_power_eig(model.J - H2 * delta_j, input_idx, output_indices, freqs), freqs
            ),
        }
    return cache


def _theory_values(cache, masks, sigma: np.ndarray, output_pos: int):
    base = float(_rho_from_gradients(cache["base"], sigma)[output_pos])
    values = {order: np.empty(N_MASKS, dtype=float) for order in ORDERS}
    for mask_pos, (mask_id, _, _, _) in enumerate(masks):
        item = cache[mask_id]
        exact = float(_rho_from_gradients(item["exact"], sigma)[output_pos])
        plus1 = float(_rho_from_gradients(item["plus1"], sigma)[output_pos])
        minus1 = float(_rho_from_gradients(item["minus1"], sigma)[output_pos])
        plus2 = float(_rho_from_gradients(item["plus2"], sigma)[output_pos])
        minus2 = float(_rho_from_gradients(item["minus2"], sigma)[output_pos])
        delta_first = (plus1 - minus1) / (2 * H1)
        delta_second = delta_first + 0.5 * (plus2 - 2 * base + minus2) / H2**2
        values["first"][mask_pos] = base + delta_first
        values["second"][mask_pos] = base + delta_second
        values["exact_resolvent"][mask_pos] = exact
    return base, values


def _metric_rows(observed_raw, observed_base, predicted_raw, predicted_base, metadata):
    rows = []
    for scale, observed, predicted in (
        ("raw_r", observed_raw, predicted_raw),
        ("delta_r", observed_raw - observed_base, predicted_raw - predicted_base),
    ):
        rows.append(metadata | {"scale": scale} | prediction_metrics(observed, predicted))
    return rows


def _load_features(source: Path, group_tag: str, dt_ms: float):
    tag = f"dt{dt_ms:g}".replace(".", "p")
    cached = np.load(source / f"features_{group_tag}_{tag}.npz")
    return cached["features"], cached["valid"]


def run(cfg: dict, bootstrap: int = BOOTSTRAP) -> Path:
    out = prepare_output(cfg, OUTPUT_NAME)
    source = _rooted(cfg["output_dir"]) / "exp3_targeted_degree10_dt_convergence"
    model = load_model(_rooted(cfg["model_dir"]))
    _, u = _input_trials(N_TRIALS, int(cfg["seed"]) + 300)
    freqs = frequency_grid()
    point_rows = []
    metric_rows = []
    bootstrap_rows = []
    noise_rows = []
    rng = np.random.default_rng(int(cfg["seed"]) + 1710000)

    for group_pos, group in enumerate(GROUPS):
        group_tag = f"{group['input_area']}_{group['lesion_area']}_{group['degree_type']}"
        masks, _ = _targeted_masks(
            model,
            group["lesion_area"],
            group["degree_type"],
            int(cfg["seed"]) + 1010000 + group_pos * 10000,
        )
        input_idx = model.areas.index(group["input_area"])
        output_areas = [group["significant_output"], group["null_output"]]
        output_indices = [model.areas.index(area) for area in output_areas]
        theory_cache = _theory_cache(model, masks, input_idx, output_indices, freqs)
        features_by_dt = {}
        valid_by_dt = {}
        for dt_ms in DT_VALUES:
            features_by_dt[dt_ms], valid_by_dt[dt_ms] = _load_features(source, group_tag, dt_ms)

        for output_pos, output_area in enumerate(output_areas):
            use = np.logical_and.reduce(
                [np.all(valid_by_dt[dt][:, output_pos, :], axis=0) for dt in DT_VALUES]
            )
            valid_indices = np.flatnonzero(use)
            sigma = np.cov(u[valid_indices], rowvar=False, ddof=1)
            theory_base, theory_raw = _theory_values(theory_cache, masks, sigma, output_pos)

            for dt_ms in DT_VALUES:
                feature = features_by_dt[dt_ms][:, output_pos, :, :]
                actual = _correlations(feature, valid_indices)
                actual_base = float(actual[0])
                actual_raw = actual[1:]
                for order in ORDERS:
                    predicted_raw = theory_raw[order]
                    metadata = {
                        "dt_ms": dt_ms,
                        "group": group_tag,
                        "input_area": group["input_area"],
                        "lesion_area": group["lesion_area"],
                        "degree_type": group["degree_type"],
                        "output_area": output_area,
                        "prior_class": "previously significant" if output_pos == 0 else "previously null",
                        "order": order,
                        "n_valid_common": len(valid_indices),
                    }
                    metric_rows.extend(
                        _metric_rows(actual_raw, actual_base, predicted_raw, theory_base, metadata)
                    )
                    for mask_pos, (mask_id, _, _, _) in enumerate(masks):
                        point_rows.append(
                            metadata
                            | {
                                "mask_id": mask_id,
                                "theory_baseline_r": theory_base,
                                "simulation_baseline_r": actual_base,
                                "theory_perturbed_r": predicted_raw[mask_pos],
                                "simulation_perturbed_r": actual_raw[mask_pos],
                                "theory_delta_r": predicted_raw[mask_pos] - theory_base,
                                "simulation_delta_r": actual_raw[mask_pos] - actual_base,
                            }
                        )

                boot_actual_raw = np.empty((bootstrap, N_MASKS), dtype=float)
                boot_actual_delta = np.empty((bootstrap, N_MASKS), dtype=float)
                boot_theory = {order: np.empty((bootstrap, N_MASKS), dtype=float) for order in ORDERS}
                boot_theory_delta = {order: np.empty((bootstrap, N_MASKS), dtype=float) for order in ORDERS}
                for boot in range(bootstrap):
                    selected = rng.choice(valid_indices, len(valid_indices), replace=True)
                    actual_b = _correlations(feature, selected)
                    sigma_b = np.cov(u[selected], rowvar=False, ddof=1)
                    theory_b, theory_raw_b = _theory_values(theory_cache, masks, sigma_b, output_pos)
                    boot_actual_raw[boot] = actual_b[1:]
                    boot_actual_delta[boot] = actual_b[1:] - actual_b[0]
                    for order in ORDERS:
                        boot_theory[order][boot] = theory_raw_b[order]
                        boot_theory_delta[order][boot] = theory_raw_b[order] - theory_b
                        for scale, observed, predicted in (
                            ("raw_r", boot_actual_raw[boot], boot_theory[order][boot]),
                            ("delta_r", boot_actual_delta[boot], boot_theory_delta[order][boot]),
                        ):
                            metrics = prediction_metrics(observed, predicted)
                            bootstrap_rows.append(
                                {
                                    "bootstrap": boot,
                                    "dt_ms": dt_ms,
                                    "group": group_tag,
                                    "input_area": group["input_area"],
                                    "output_area": output_area,
                                    "order": order,
                                    "scale": scale,
                                    "R2": metrics["R2"],
                                    "RMSE": metrics["RMSE"],
                                    "MAE": metrics["MAE"],
                                }
                            )
                raw_sd = np.std(boot_actual_raw, axis=0, ddof=1)
                delta_sd = np.std(boot_actual_delta, axis=0, ddof=1)
                raw_error_sd = np.std(boot_actual_raw - boot_theory["exact_resolvent"], axis=0, ddof=1)
                delta_error_sd = np.std(
                    boot_actual_delta - boot_theory_delta["exact_resolvent"], axis=0, ddof=1
                )
                noise_rows.append(
                    {
                        "dt_ms": dt_ms,
                        "group": group_tag,
                        "input_area": group["input_area"],
                        "output_area": output_area,
                        "n_valid_common": len(valid_indices),
                        "median_raw_r_bootstrap_sd": float(np.median(raw_sd)),
                        "median_delta_r_bootstrap_sd": float(np.median(delta_sd)),
                        "median_delta_to_raw_sd_ratio": float(np.median(delta_sd / np.maximum(raw_sd, 1e-15))),
                        "median_raw_error_sd": float(np.median(raw_error_sd)),
                        "median_delta_error_sd": float(np.median(delta_error_sd)),
                        "median_delta_to_raw_error_sd_ratio": float(
                            np.median(delta_error_sd / np.maximum(raw_error_sd, 1e-15))
                        ),
                    }
                )
            print(f"absolute-r bootstrap complete: {group_tag} -> {output_area}", flush=True)

    points = pd.DataFrame(point_rows)
    metrics = pd.DataFrame(metric_rows)
    bootstraps = pd.DataFrame(bootstrap_rows)
    noise = pd.DataFrame(noise_rows)
    points.to_csv(out / "absolute_correlation_points.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "absolute_correlation_metrics.csv", index=False)
    bootstraps.to_csv(out / "absolute_correlation_bootstrap.csv.gz", index=False, compression="gzip")
    noise.to_csv(out / "absolute_correlation_noise.csv", index=False)
    _plot(out, points, metrics, bootstraps, noise)
    save_json(
        out / "manifest.json",
        {
            "experiment": "absolute versus differenced Experiment-1 correlation for every S13 mask",
            "source_features": str(source),
            "orders": ORDERS,
            "dt_ms": DT_VALUES,
            "n_masks_per_combination": N_MASKS,
            "n_input_trials": N_TRIALS,
            "bootstrap_resamples": bootstrap,
            "common_valid_across_all_networks_and_both_dt": True,
        },
    )
    return out


def _ordered(points: pd.DataFrame) -> pd.DataFrame:
    combinations = points[
        ["group", "input_area", "lesion_area", "degree_type", "output_area", "prior_class"]
    ].drop_duplicates()
    return pd.concat(
        [
            combinations[combinations.prior_class == "previously significant"],
            combinations[combinations.prior_class == "previously null"],
        ],
        ignore_index=True,
    )


def _plot(out: Path, points: pd.DataFrame, metrics: pd.DataFrame, bootstraps: pd.DataFrame, noise: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    ordered = _ordered(points)
    labels = [
        f"{r.input_area}/{r.lesion_area}-{r.degree_type}/{r.output_area}"
        for r in ordered.itertuples(index=False)
    ]
    colors = {0.25: "#4C78A8", 0.05: "#E45756"}
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True, constrained_layout=True)
    for row_pos, scale in enumerate(("raw_r", "delta_r")):
        for col_pos, order in enumerate(ORDERS):
            ax = axes[row_pos, col_pos]
            for dt_ms, offset in ((0.25, -0.09), (0.05, 0.09)):
                medians, lows, highs, observed = [], [], [], []
                for item in ordered.itertuples(index=False):
                    subset = bootstraps[
                        (bootstraps.dt_ms == dt_ms)
                        & (bootstraps.group == item.group)
                        & (bootstraps.output_area == item.output_area)
                        & (bootstraps.order == order)
                        & (bootstraps.scale == scale)
                    ].R2
                    medians.append(subset.median())
                    lows.append(subset.quantile(0.025))
                    highs.append(subset.quantile(0.975))
                    observed.append(
                        metrics[
                            (metrics.dt_ms == dt_ms)
                            & (metrics.group == item.group)
                            & (metrics.output_area == item.output_area)
                            & (metrics.order == order)
                            & (metrics.scale == scale)
                        ].R2.iloc[0]
                    )
                medians = np.asarray(medians)
                ax.errorbar(
                    x + offset,
                    medians,
                    yerr=[medians - np.asarray(lows), np.asarray(highs) - medians],
                    fmt="o",
                    capsize=2,
                    color=colors[dt_ms],
                    alpha=0.8,
                    label=f"dt={dt_ms:g} ms bootstrap 95% CI",
                )
                ax.scatter(x + offset, observed, marker="x", s=42, color="black", zorder=4)
            ax.axhline(1, color="0.25", ls="--", lw=1)
            ax.axhline(0, color="0.5", ls=":", lw=1)
            ax.set_title(f"{scale.replace('_', ' ')}; {order}")
            ax.set_ylabel("Identity R2")
            ax.set_xticks(x, labels, rotation=36, ha="right", fontsize=8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 3: absolute correlation versus correlation difference", fontsize=14)
    save_figure(fig, out / "figures" / "absolute_vs_delta_identity_R2")
    plt.close(fig)

    for order in ORDERS:
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for ax, item in zip(axes.flat, ordered.itertuples(index=False)):
            subset = points[
                (points.dt_ms == 0.25)
                & (points.group == item.group)
                & (points.output_area == item.output_area)
                & (points.order == order)
            ]
            ax.scatter(
                subset.theory_perturbed_r,
                subset.simulation_perturbed_r,
                s=30,
                alpha=0.75,
                color="#247BA0" if "significant" in item.prior_class else "#777777",
            )
            lo = min(subset.theory_perturbed_r.min(), subset.simulation_perturbed_r.min())
            hi = max(subset.theory_perturbed_r.max(), subset.simulation_perturbed_r.max())
            pad = max((hi - lo) * 0.08, 1e-6)
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
            metric = metrics[
                (metrics.dt_ms == 0.25)
                & (metrics.group == item.group)
                & (metrics.output_area == item.output_area)
                & (metrics.order == order)
                & (metrics.scale == "raw_r")
            ].iloc[0]
            ax.text(0.04, 0.95, f"identity R2={metric.R2:.3f}\nN=40 masks", transform=ax.transAxes, va="top", fontsize=8)
            ax.set(
                xlabel=f"Experiment 1 theory r ({order})",
                ylabel="Full55 simulation r",
                title=f"{item.input_area}; delete {item.lesion_area}-{item.degree_type}; read {item.output_area}",
            )
        fig.suptitle(f"Absolute perturbed-network correlation: {order}, dt=0.25 ms", fontsize=14)
        save_figure(fig, out / "figures" / f"absolute_r_scatter_{order}")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)
    width = 0.34
    for dt_ms, offset in ((0.25, -width / 2), (0.05, width / 2)):
        values = []
        error_values = []
        for item in ordered.itertuples(index=False):
            row = noise[
                (noise.dt_ms == dt_ms)
                & (noise.group == item.group)
                & (noise.output_area == item.output_area)
            ].iloc[0]
            values.append(row.median_delta_to_raw_sd_ratio)
            error_values.append(row.median_delta_to_raw_error_sd_ratio)
        axes[0].bar(x + offset, values, width, color=colors[dt_ms], label=f"dt={dt_ms:g} ms")
        axes[1].bar(x + offset, error_values, width, color=colors[dt_ms], label=f"dt={dt_ms:g} ms")
    for ax in axes:
        ax.axhline(1, color="0.25", ls="--", lw=1)
        ax.set_xticks(x, labels, rotation=36, ha="right", fontsize=8)
        ax.legend(frameon=False, fontsize=8)
    axes[0].set(ylabel="median bootstrap SD ratio", title="SD(delta r) / SD(raw perturbed r)")
    axes[1].set(ylabel="median bootstrap error-SD ratio", title="SD(delta error) / SD(raw error)")
    fig.suptitle("Does differencing amplify realization variance?", fontsize=14)
    save_figure(fig, out / "figures" / "differencing_variance_diagnostic")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--bootstrap", type=int, default=BOOTSTRAP)
    args = parser.parse_args()
    print(run(load_config(args.config), bootstrap=args.bootstrap))


if __name__ == "__main__":
    main()
