from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import (
    AREAS9,
    apply_fln_delta,
    feature_from_spectrum,
    frequency_grid,
    input_psd,
    load_model,
    prediction_metrics,
    save_figure,
    save_json,
)
from .exp3 import Z0, ZS, _input_trials
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .nonlinear_network_validation import _fourier_inputs, _run_multioutput_batch
from .pairwise_nonlinear_window_validation import _fit_metrics, _rooted, _window_specs


OUTPUT_NAME = "exp3_targeted_degree10"
N_MASKS = 40
N_TRIALS = 512
N_DELETE = 10
CHUNK_SIZE = 64


GROUPS = [
    {
        "input_area": "A6Va",
        "lesion_area": "MST",
        "degree_type": "in",
        "significant_output": "MST",
        "null_output": "A23a",
        "prior_significant_p": 0.001116,
        "prior_null_p": 0.910684,
    },
    {
        "input_area": "V2",
        "lesion_area": "A6Va",
        "degree_type": "in",
        "significant_output": "A6Va",
        "null_output": "V4",
        "prior_significant_p": 0.001416,
        "prior_null_p": 0.965224,
    },
    {
        "input_area": "PG",
        "lesion_area": "V1",
        "degree_type": "out",
        "significant_output": "V4",
        "null_output": "V2",
        "prior_significant_p": 0.002799,
        "prior_null_p": 0.724919,
    },
]


def _targeted_masks(model, lesion_area: str, degree_type: str, seed: int):
    lesion = model.areas.index(lesion_area)
    if degree_type == "in":
        candidates = [(lesion, int(source)) for source in np.flatnonzero(model.fln[lesion, :])]
    else:
        candidates = [(int(target), lesion) for target in np.flatnonzero(model.fln[:, lesion])]
    if len(candidates) < N_DELETE:
        raise ValueError(f"Only {len(candidates)} {degree_type}-degree edges for {lesion_area}")
    rng = np.random.default_rng(seed)
    masks = []
    seen = set()
    while len(masks) < N_MASKS:
        chosen = tuple(sorted(rng.choice(len(candidates), N_DELETE, replace=False).tolist()))
        if chosen in seen:
            continue
        seen.add(chosen)
        delta_w = np.zeros_like(model.fln)
        edges = []
        for candidate in chosen:
            target, source = candidates[candidate]
            weight = float(model.fln[target, source])
            delta_w[target, source] = -weight
            edges.append((model.areas[source], model.areas[target], weight))
        masks.append((f"mask_{len(masks):03d}", delta_w, apply_fln_delta(model, delta_w) - model.J, edges))
    return masks, len(candidates)


def _transfer_power_eig(j: np.ndarray, input_idx: int, output_indices: list[int], freqs: np.ndarray):
    eigenvalues, vectors = np.linalg.eig(j)
    b = np.zeros(j.shape[0])
    b[input_idx] = 1.0
    coefficients = np.linalg.solve(vectors, b)
    denominators = 1.0 / (
        1j * 2 * np.pi * freqs[:, None] / 1000.0 - eigenvalues[None, :]
    )
    readout = vectors[np.asarray(output_indices), :] * coefficients[None, :]
    response = denominators @ readout.T
    # Keep the same output-major convention used by the direct transfer solver.
    return np.maximum(abs(response.T) ** 2, np.finfo(float).tiny)


def _rho_from_transfer(transfer: np.ndarray, sigma: np.ndarray, freqs: np.ndarray):
    grad = np.empty((2, 4))
    h = 2e-3
    for coordinate in range(4):
        perturbation = np.zeros(4)
        perturbation[coordinate] = h
        values = []
        for u in (perturbation, -perturbation):
            spectrum = transfer * input_psd(freqs, Z0 + ZS * u)
            feature = feature_from_spectrum(freqs, spectrum)
            values.append([feature["peak_omega"], feature["exponent"]])
        grad[:, coordinate] = (np.asarray(values[0]) - np.asarray(values[1])) / (2 * h)
    covariance = float(grad[0] @ sigma @ grad[1])
    variance_p = float(grad[0] @ sigma @ grad[0])
    variance_a = float(grad[1] @ sigma @ grad[1])
    return covariance / np.sqrt(max(variance_p * variance_a, 1e-30))


def _linear_rhos_eig(j, sigmas, freqs, input_idx, output_indices):
    transfers = _transfer_power_eig(j, input_idx, output_indices, freqs)
    return np.asarray(
        [
            _rho_from_transfer(transfers[pos], sigma, freqs)
            for pos, sigma in enumerate(sigmas)
        ]
    )


def _joint_multi_edge_theory(model, delta_j, sigmas, freqs, input_idx, output_indices):
    base = _linear_rhos_eig(model.J, sigmas, freqs, input_idx, output_indices)
    exact = _linear_rhos_eig(model.J + delta_j, sigmas, freqs, input_idx, output_indices) - base
    h1 = 0.02
    plus1 = _linear_rhos_eig(model.J + h1 * delta_j, sigmas, freqs, input_idx, output_indices)
    minus1 = _linear_rhos_eig(model.J - h1 * delta_j, sigmas, freqs, input_idx, output_indices)
    first = (plus1 - minus1) / (2 * h1)
    h2 = 0.05
    plus2 = _linear_rhos_eig(model.J + h2 * delta_j, sigmas, freqs, input_idx, output_indices)
    minus2 = _linear_rhos_eig(model.J - h2 * delta_j, sigmas, freqs, input_idx, output_indices)
    second_derivative = (plus2 - 2 * base + minus2) / h2**2
    second = first + 0.5 * second_derivative
    return first, second, exact


def _networks(model, masks):
    result = [("baseline", model.params, None, [])]
    for mask_id, delta_w, delta_j, edges in masks:
        params = copy.copy(model.params)
        params["fln_mat"] = model.fln + delta_w
        result.append((mask_id, params, delta_j, edges))
    return result


def _simulate_group(cfg, out, model, group, group_pos, z, u, force=False):
    tag = f"{group['input_area']}_{group['lesion_area']}_{group['degree_type']}"
    result_path = out / f"group_{tag}.csv.gz"
    edge_path = out / f"group_{tag}_edges.csv.gz"
    qc_path = out / f"group_{tag}_qc.csv.gz"
    if result_path.exists() and not force:
        print(f"targeted degree group {tag}: cached", flush=True)
        return pd.read_csv(result_path), pd.read_csv(edge_path), pd.read_csv(qc_path)

    masks, candidate_count = _targeted_masks(
        model,
        group["lesion_area"],
        group["degree_type"],
        int(cfg["seed"]) + 1010000 + group_pos * 10000,
    )
    networks = _networks(model, masks)
    input_idx = model.areas.index(group["input_area"])
    output_areas = [group["significant_output"], group["null_output"]]
    output_indices = np.array([model.areas.index(area) for area in output_areas], dtype=np.int64)
    specs = {"short": _window_specs(cfg)["short"]}
    ecfg = cfg["exp3_nonlinear"]
    dt = float(ecfg["dt_ms"])
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    analysis_steps = int(round(specs["short"]["analysis_s"] * 1000.0 / dt))
    features = np.empty((len(networks), len(output_areas), N_TRIALS, 2), dtype=float)
    valid = np.empty((len(networks), len(output_areas), N_TRIALS), dtype=bool)
    qc_rows = []
    phase_seed = int(cfg["seed"]) + 1110000
    for start in range(0, N_TRIALS, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, N_TRIALS)
        inputs = _fourier_inputs(
            z[start:stop],
            dt,
            analysis_steps,
            burn_steps,
            float(ecfg["input_sigma"]),
            phase_seed + 100003 * (start // CHUNK_SIZE),
        )
        for network_pos, (network_id, params, _, _) in enumerate(networks):
            outputs, rect, min_e, min_i, min_de, min_di = _run_multioutput_batch(
                params, inputs, dt, burn_steps, output_indices, input_idx=input_idx
            )
            values, is_valid = _feature_arrays_vectorized(outputs, dt, specs)["short"]
            features[network_pos, :, start:stop] = values
            valid[network_pos, :, start:stop] = is_valid
            qc_rows.append(
                {
                    "group": tag,
                    "trial_start": start,
                    "trial_stop": stop,
                    "network": network_id,
                    "rectification_fraction": float(
                        rect.sum() / (2 * len(model.areas) * inputs.shape[1] * (stop - start))
                    ),
                    "min_rate_E": float(min_e.min()),
                    "min_rate_I": float(min_i.min()),
                    "min_drive_E": float(min_de.min()),
                    "min_drive_I": float(min_di.min()),
                }
            )
        print(f"targeted degree {tag}: trials {start + 1}-{stop}/{N_TRIALS}", flush=True)

    common_valid = [np.all(valid[:, output_pos, :], axis=0) for output_pos in range(len(output_areas))]
    sigmas = [np.cov(u[use], rowvar=False, ddof=1) for use in common_valid]
    baseline_r = [
        float(np.corrcoef(features[0, output_pos, use].T)[0, 1])
        for output_pos, use in enumerate(common_valid)
    ]
    freqs = frequency_grid()
    rows = []
    edge_rows = []
    for mask_pos, (mask_id, _, delta_j, edges) in enumerate(networks[1:], 1):
        first, second, exact = _joint_multi_edge_theory(
            model, delta_j, sigmas, freqs, input_idx, output_indices.tolist()
        )
        for source, target, weight in edges:
            edge_rows.append(
                {
                    "group": tag,
                    "mask_id": mask_id,
                    "source_area": source,
                    "target_area": target,
                    "deleted_weight": weight,
                }
            )
        for output_pos, output_area in enumerate(output_areas):
            use = common_valid[output_pos]
            perturbed_r = float(np.corrcoef(features[mask_pos, output_pos, use].T)[0, 1])
            prior_class = "previously significant" if output_pos == 0 else "previously null"
            rows.append(
                {
                    "group": tag,
                    "input_area": group["input_area"],
                    "lesion_area": group["lesion_area"],
                    "degree_type": group["degree_type"],
                    "output_area": output_area,
                    "prior_class": prior_class,
                    "prior_5edge_p": group["prior_significant_p"] if output_pos == 0 else group["prior_null_p"],
                    "mask_id": mask_id,
                    "n_degree_candidates": candidate_count,
                    "n_deleted_edges": N_DELETE,
                    "n_input_trials": N_TRIALS,
                    "n_valid_common": int(use.sum()),
                    "baseline_actual_r": baseline_r[output_pos],
                    "perturbed_actual_r": perturbed_r,
                    "delta_rho_actual": perturbed_r - baseline_r[output_pos],
                    "delta_rho_first": float(first[output_pos]),
                    "delta_rho_second": float(second[output_pos]),
                    "delta_rho_exact_resolvent": float(exact[output_pos]),
                }
            )
    result = pd.DataFrame(rows)
    edges = pd.DataFrame(edge_rows)
    qc = pd.DataFrame(qc_rows)
    result.to_csv(result_path, index=False, compression="gzip")
    edges.to_csv(edge_path, index=False, compression="gzip")
    qc.to_csv(qc_path, index=False, compression="gzip")
    return result, edges, qc


def _metrics(results: pd.DataFrame):
    rows = []
    keys = ["group", "input_area", "lesion_area", "degree_type", "output_area", "prior_class", "prior_5edge_p"]
    for key, group in results.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        for order, column in (
            ("first", "delta_rho_first"),
            ("second", "delta_rho_second"),
            ("exact_resolvent", "delta_rho_exact_resolvent"),
        ):
            rows.append(
                base
                | {"order": order, "median_abs_actual_delta_rho": float(group.delta_rho_actual.abs().median())}
                | _fit_metrics(group.delta_rho_actual.to_numpy(), group[column].to_numpy())
            )
    return pd.DataFrame(rows)


def run(cfg: dict, force: bool = False) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(_rooted(cfg["model_dir"]))
    z, u = _input_trials(N_TRIALS, int(cfg["seed"]) + 300)
    all_results = []
    all_edges = []
    all_qc = []
    for group_pos, group in enumerate(GROUPS):
        results, edges, qc = _simulate_group(cfg, out, model, group, group_pos, z, u, force=force)
        all_results.append(results)
        all_edges.append(edges)
        all_qc.append(qc)
    results = pd.concat(all_results, ignore_index=True)
    edges = pd.concat(all_edges, ignore_index=True)
    qc = pd.concat(all_qc, ignore_index=True)
    metrics = _metrics(results)
    results.to_csv(out / "targeted_degree10_predictions.csv.gz", index=False, compression="gzip")
    edges.to_csv(out / "targeted_degree10_edges.csv.gz", index=False, compression="gzip")
    qc.to_csv(out / "targeted_degree10_qc.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "targeted_degree10_metrics.csv", index=False)
    _plot(out, results, metrics)
    save_json(
        out / "manifest.json",
        {
            "experiment": "full55_targeted_10_edge_degree_deletion",
            "actual_simulation": "explicit full-55-area threshold-linear time-domain integration",
            "selected_groups": GROUPS,
            "n_masks_per_group": N_MASKS,
            "n_deleted_edges_per_mask": N_DELETE,
            "n_input_trials_per_correlation": N_TRIALS,
            "common_random_inputs_within_and_across_groups": True,
            "theory": "joint 10-edge directional first/second order plus exact perturbed resolvent",
            "dt_ms": cfg["exp3_nonlinear"]["dt_ms"],
            "burn_s": cfg["exp3_nonlinear"]["burn_s"],
            "analysis_s": _window_specs(cfg)["short"]["analysis_s"],
            "runtime_s": time.time() - started,
        },
    )
    return out


def _plot(out: Path, results: pd.DataFrame, metrics: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    combinations = results[
        ["group", "input_area", "lesion_area", "degree_type", "output_area", "prior_class", "prior_5edge_p"]
    ].drop_duplicates()
    significant = combinations[combinations.prior_class == "previously significant"]
    null = combinations[combinations.prior_class == "previously null"]
    ordered = pd.concat([significant, null], ignore_index=True)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for ax, row in zip(axes.flat, ordered.itertuples(index=False)):
        data = results[(results.group == row.group) & (results.output_area == row.output_area)]
        x = data.delta_rho_exact_resolvent.to_numpy()
        y = data.delta_rho_actual.to_numpy()
        ax.scatter(x, y, s=28, alpha=0.72, color="#247BA0" if "significant" in row.prior_class else "#7A7A7A")
        fit = _fit_metrics(y, x)
        if np.std(x) > 0:
            coeff = np.polyfit(x, y, 1)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, np.polyval(coeff, xx), color="#D1495B", lw=2)
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        metric_group = metrics[(metrics.group == row.group) & (metrics.output_area == row.output_area)].set_index("order")
        ax.text(
            0.04,
            0.96,
            f"prior p={row.prior_5edge_p:.3g}\n"
            f"R2 id: 1st={metric_group.loc['first','R2']:.2f}\n"
            f"2nd={metric_group.loc['second','R2']:.2f}, exact={metric_group.loc['exact_resolvent','R2']:.2f}\n"
            f"exact OLS R2={fit['ols_R2']:.2f}",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
        )
        ax.set(
            xlabel="Joint 10-edge exact-resolvent Delta r",
            ylabel="Full55 nonlinear actual Delta r",
            title=f"{row.input_area} input; delete {row.lesion_area} {row.degree_type}; read {row.output_area}",
        )
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
    axes[0, 0].text(-0.24, 1.12, "Previously significant", transform=axes[0, 0].transAxes, fontsize=11, weight="bold")
    axes[1, 0].text(-0.24, 1.12, "Previously null", transform=axes[1, 0].transAxes, fontsize=11, weight="bold")
    fig.suptitle("Experiment 3: concentrated deletion of 10 in/out-degree edges", fontsize=13)
    save_figure(fig, out / "figures" / "exp3_targeted_degree10_scatter")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    labels = [f"{row.input_area}/{row.lesion_area}-{row.degree_type}/{row.output_area}" for row in ordered.itertuples(index=False)]
    x = np.arange(len(ordered))
    width = 0.24
    colors = {"first": "#4C78A8", "second": "#E45756", "exact_resolvent": "#59A14F"}
    for offset, order in zip((-width, 0, width), ("first", "second", "exact_resolvent")):
        values = []
        for row in ordered.itertuples(index=False):
            values.append(metrics[(metrics.group == row.group) & (metrics.output_area == row.output_area) & (metrics.order == order)].R2.iloc[0])
        axes[0].bar(x + offset, values, width, color=colors[order], label=order)
    axes[0].axhline(0, color="0.25", ls=":")
    axes[0].axhline(1, color="0.25", ls="--")
    axes[0].set_xticks(x, labels, rotation=40, ha="right", fontsize=8)
    axes[0].set(ylabel="Identity R2", title="Theory calibration across 40 joint masks")
    axes[0].legend(frameon=False, fontsize=8)
    actual = []
    classes = []
    for row in ordered.itertuples(index=False):
        data = results[(results.group == row.group) & (results.output_area == row.output_area)]
        actual.append(abs(data.delta_rho_actual.to_numpy()))
        classes.append("sig" if "significant" in row.prior_class else "null")
    boxes = axes[1].boxplot(actual, patch_artist=True, tick_labels=labels, showfliers=False)
    for box, cls in zip(boxes["boxes"], classes):
        box.set_facecolor("#F28E2B" if cls == "sig" else "#B8B8B8")
        box.set_alpha(0.75)
    axes[1].set_yscale("log")
    axes[1].set_xticks(x + 1, labels, rotation=40, ha="right", fontsize=8)
    axes[1].set(ylabel="|Actual nonlinear Delta r|", title="Effect-size distribution")
    fig.suptitle("Targeted 10-edge deletion: prior significance, effect size, and identity calibration", fontsize=13)
    save_figure(fig, out / "figures" / "exp3_targeted_degree10_summary")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(load_config(args.config), force=args.force))


if __name__ == "__main__":
    main()
