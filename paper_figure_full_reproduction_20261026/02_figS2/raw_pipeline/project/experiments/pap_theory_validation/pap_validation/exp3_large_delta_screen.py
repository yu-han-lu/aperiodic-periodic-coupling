from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

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
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .exp3_targeted_degree10 import _rho_from_transfer
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .nonlinear_network_validation import _run_multioutput_batch
from .pairwise_nonlinear_window_validation import _window_specs


OUTPUT_NAME = "exp3_large_delta_screen"
DT_MS = 0.25
N_DELETE = 10
N_CANDIDATE_PER_FAMILY = 90
N_TOP = 36
N_CONTROL = 18
N_TRIALS = 256
N_REPEATS = 3
CHUNK_SIZE = 8


def _mask_from_indices(model, edge_list, chosen, mask_id, family):
    delta_w = np.zeros_like(model.fln)
    encoded = []
    for edge_pos in chosen:
        target, source = edge_list[int(edge_pos)]
        weight = float(model.fln[target, source])
        delta_w[target, source] = -weight
        encoded.append((model.areas[source], model.areas[target], weight))
    delta_j = apply_fln_delta(model, delta_w) - model.J
    return {
        "mask_id": mask_id,
        "family": family,
        "delta_w": delta_w,
        "delta_j": delta_j,
        "edges": encoded,
        "delta_j_fro": float(np.linalg.norm(delta_j)),
    }


def _candidate_masks(model, seed):
    rng = np.random.default_rng(seed)
    edge_list = list(zip(*np.nonzero(model.fln)))
    weights = np.asarray([model.fln[t, s] for t, s in edge_list], dtype=float)
    masks = []
    seen = set()

    def add_family(family, probabilities=None):
        attempts = 0
        while sum(item["family"] == family for item in masks) < N_CANDIDATE_PER_FAMILY:
            attempts += 1
            if attempts > 100000:
                raise RuntimeError(f"Could not draw enough unique masks for {family}")
            chosen = tuple(sorted(rng.choice(len(edge_list), N_DELETE, replace=False, p=probabilities).tolist()))
            if chosen in seen:
                continue
            seen.add(chosen)
            masks.append(_mask_from_indices(model, edge_list, chosen, f"{family}_{len(masks):04d}", family))

    add_family("uniform")
    add_family("weight_eta1", weights / weights.sum())
    squared = weights**2
    add_family("weight_eta2", squared / squared.sum())

    targeted = 0
    attempts = 0
    while targeted < N_CANDIDATE_PER_FAMILY:
        attempts += 1
        if attempts > 100000:
            raise RuntimeError("Could not draw enough targeted masks")
        area = int(rng.choice([model.areas.index(name) for name in AREAS9]))
        degree = "in" if rng.random() < 0.5 else "out"
        candidates = (
            [(area, int(source)) for source in np.flatnonzero(model.fln[area])]
            if degree == "in"
            else [(int(target), area) for target in np.flatnonzero(model.fln[:, area])]
        )
        if len(candidates) < N_DELETE:
            continue
        local = tuple(sorted(rng.choice(len(candidates), N_DELETE, replace=False).tolist()))
        chosen_edges = tuple(sorted(candidates[pos] for pos in local))
        key = tuple(sorted(edge_list.index(edge) for edge in chosen_edges))
        if key in seen:
            continue
        seen.add(key)
        mask = _mask_from_indices(model, edge_list, key, f"targeted_{len(masks):04d}", "targeted_degree")
        mask["target_area"] = model.areas[area]
        mask["degree_type"] = degree
        masks.append(mask)
        targeted += 1
    return masks


def _all_transfer_power(j, indices, frequencies):
    eigenvalues, vectors = np.linalg.eig(j)
    b = np.zeros((len(j), len(indices)))
    for column, index in enumerate(indices):
        b[index, column] = 1.0
    coefficients = np.linalg.solve(vectors, b)
    denominators = 1.0 / (1j * 2.0 * np.pi * frequencies[:, None] / 1000.0 - eigenvalues[None, :])
    readout = vectors[np.asarray(indices), :]
    response = np.einsum("fk,ok,ki->foi", denominators, readout, coefficients, optimize=True)
    return np.maximum(abs(response.transpose(2, 1, 0)) ** 2, np.finfo(float).tiny)


def _rho_matrix(j, indices, frequencies, sigma):
    transfers = _all_transfer_power(j, indices, frequencies)
    result = np.empty((len(indices), len(indices)))
    for input_pos in range(len(indices)):
        for output_pos in range(len(indices)):
            result[input_pos, output_pos] = _rho_from_transfer(
                transfers[input_pos, output_pos], sigma, frequencies
            )
    return result


def _screen(model, masks, seed, out):
    _, u = _input_trials(N_TRIALS, seed + 17)
    sigma = np.cov(u, rowvar=False, ddof=1)
    frequencies = frequency_grid()
    indices = [model.areas.index(area) for area in AREAS9]
    baseline = _rho_matrix(model.J, indices, frequencies, sigma)
    rows = []
    edge_rows = []
    for mask_pos, mask in enumerate(masks):
        perturbed = _rho_matrix(model.J + mask["delta_j"], indices, frequencies, sigma)
        delta = perturbed - baseline
        for input_pos, input_area in enumerate(AREAS9):
            for output_pos, output_area in enumerate(AREAS9):
                rows.append(
                    {
                        "mask_id": mask["mask_id"], "family": mask["family"],
                        "input_area": input_area, "output_area": output_area,
                        "baseline_r": baseline[input_pos, output_pos],
                        "perturbed_r": perturbed[input_pos, output_pos],
                        "exact_delta_r": delta[input_pos, output_pos],
                        "abs_exact_delta_r": abs(delta[input_pos, output_pos]),
                        "delta_j_fro": mask["delta_j_fro"],
                        "target_area": mask.get("target_area", ""),
                        "degree_type": mask.get("degree_type", ""),
                    }
                )
        for edge_pos, (source, target, weight) in enumerate(mask["edges"]):
            edge_rows.append(
                {"mask_id": mask["mask_id"], "family": mask["family"], "edge_pos": edge_pos,
                 "source_area": source, "target_area": target, "deleted_weight": weight}
            )
        if (mask_pos + 1) % 10 == 0:
            print(f"exact-resolvent screen {mask_pos + 1}/{len(masks)}", flush=True)
    screen = pd.DataFrame(rows)
    screen.to_csv(out / "exact_resolvent_screen.csv.gz", index=False, compression="gzip")
    pd.DataFrame(edge_rows).to_csv(out / "candidate_edges.csv.gz", index=False, compression="gzip")
    return screen


def _select(screen):
    ordered = screen.sort_values("abs_exact_delta_r", ascending=False)
    top_parts = [ordered.groupby("input_area", sort=False).head(2), ordered.groupby("output_area", sort=False).head(2)]
    top = pd.concat(top_parts).drop_duplicates(["mask_id", "input_area", "output_area"])
    if len(top) < N_TOP:
        top = pd.concat([top, ordered]).drop_duplicates(["mask_id", "input_area", "output_area"]).head(N_TOP)
    else:
        top = top.head(N_TOP)
    top = top.copy()
    top["selection_class"] = "high_effect"

    excluded = set(zip(top.mask_id, top.input_area, top.output_area))
    low_pool = screen[screen.abs_exact_delta_r <= screen.abs_exact_delta_r.quantile(0.35)].copy()
    low_pool = low_pool[
        [tuple(item) not in excluded for item in low_pool[["mask_id", "input_area", "output_area"]].to_numpy()]
    ]
    controls = []
    rng = np.random.default_rng(89031)
    for family, target_count in top.family.value_counts().items():
        n = max(1, int(round(N_CONTROL * target_count / len(top))))
        pool = low_pool[low_pool.family == family]
        if len(pool):
            controls.append(pool.sample(min(n, len(pool)), random_state=int(rng.integers(1, 2**31 - 1))))
    control = pd.concat(controls).drop_duplicates(["mask_id", "input_area", "output_area"]).head(N_CONTROL)
    if len(control) < N_CONTROL:
        extra = low_pool.sample(min(N_CONTROL - len(control), len(low_pool)), random_state=4821)
        control = pd.concat([control, extra]).drop_duplicates(["mask_id", "input_area", "output_area"]).head(N_CONTROL)
    control = control.copy()
    control["selection_class"] = "matched_low_effect"
    selected = pd.concat([top, control], ignore_index=True)
    selected.insert(0, "selection_id", [f"selection_{pos:03d}" for pos in range(len(selected))])
    return selected


def _exact_trial_features(transfer, z):
    frequencies = frequency_grid()
    values = np.empty((len(z), 2))
    valid = np.empty(len(z), dtype=bool)
    for trial, design in enumerate(z):
        feature = feature_from_spectrum(frequencies, transfer * input_psd(frequencies, design))
        values[trial] = feature["peak_omega"], feature["exponent"]
        valid[trial] = bool(feature["peak_valid"])
    return values, valid


def _continuous_transfer(j, input_idx, output_idx):
    frequencies = frequency_grid()
    eigenvalues, vectors = np.linalg.eig(j)
    b = np.zeros(len(j))
    b[input_idx] = 1.0
    coefficients = np.linalg.solve(vectors, b)
    response = (1.0 / (1j * 2.0 * np.pi * frequencies[:, None] / 1000.0 - eigenvalues[None, :])) @ (vectors[output_idx] * coefficients)
    return np.maximum(abs(response) ** 2, np.finfo(float).tiny)


def _discrete_transfer(j, input_idx, output_idx, n_steps, tau_e):
    fs = 1000.0 / DT_MS
    frequencies = np.fft.rfftfreq(n_steps, d=1.0 / fs)
    theta = 2.0 * np.pi * frequencies / fs
    eigenvalues, vectors = np.linalg.eig(j)
    discrete = 1.0 + DT_MS * eigenvalues
    b = np.zeros(len(j))
    b[input_idx] = DT_MS / tau_e
    coefficients = np.linalg.solve(vectors, b)
    denominator = 1.0 / (1.0 - discrete[None, :] * np.exp(-1j * theta[:, None]))
    return denominator @ (vectors[output_idx] * coefficients)


def _pearson(values, use):
    if use.sum() < 8:
        return np.nan
    return float(stats.pearsonr(values[use, 0], values[use, 1]).statistic)


def _validate_repeat(cfg, model, selected, mask_lookup, repeat, out):
    cache_path = out / f"long_validation_repeat{repeat:02d}.csv.gz"
    if cache_path.exists():
        print(f"large-delta long validation repeat {repeat + 1}/{N_REPEATS}: cached", flush=True)
        return pd.read_csv(cache_path)
    z, _ = _input_trials(N_TRIALS, int(cfg["seed"]) + 3500000 + repeat)
    specs = {"long": _window_specs(cfg)["long"]}
    analysis_s = float(specs["long"]["analysis_s"])
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    tau_e = float(model.params["tau_exc"])
    rows = []
    for input_pos, input_area in enumerate(AREAS9):
        group = selected[selected.input_area == input_area].reset_index(drop=True)
        if group.empty:
            continue
        input_idx = model.areas.index(input_area)
        entries = []
        baseline_entry = {}
        for output_area in group.output_area.unique():
            output_idx = model.areas.index(output_area)
            baseline_entry[output_area] = len(entries)
            entries.append(("baseline", output_area, _discrete_transfer(model.J, input_idx, output_idx, n_steps, tau_e)))
        perturbed_entry = {}
        for _, item in group.iterrows():
            mask = mask_lookup[item.mask_id]
            output_idx = model.areas.index(item.output_area)
            perturbed_entry[item.selection_id] = len(entries)
            entries.append((item.selection_id, item.output_area, _discrete_transfer(model.J + mask["delta_j"], input_idx, output_idx, n_steps, tau_e)))
        features = np.empty((len(entries), N_TRIALS, 2))
        valid = np.empty((len(entries), N_TRIALS), dtype=bool)
        for start in range(0, N_TRIALS, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, N_TRIALS)
            inputs = _matched_bandlimited_inputs(
                z[start:stop], DT_MS, analysis_s, burn_s,
                float(cfg["exp3_nonlinear"]["input_sigma"]),
                int(cfg["seed"]) + 3600000 + repeat * 100000 + input_pos * 1000,
                trial_offset=start,
            )
            input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=1)
            outputs = np.empty((stop - start, len(entries), n_steps), dtype=np.float32)
            for entry_pos, (_, _, transfer) in enumerate(entries):
                outputs[:, entry_pos] = np.fft.irfft(input_fft * transfer[None, :], n=n_steps, axis=1).astype(np.float32)
            values, is_valid = _feature_arrays_vectorized(outputs, DT_MS, specs)["long"]
            features[:, start:stop] = values
            valid[:, start:stop] = is_valid
        for _, item in group.iterrows():
            bpos = baseline_entry[item.output_area]
            ppos = perturbed_entry[item.selection_id]
            mask = mask_lookup[item.mask_id]
            output_idx = model.areas.index(item.output_area)
            theory_base, theory_base_valid = _exact_trial_features(_continuous_transfer(model.J, input_idx, output_idx), z)
            theory_pert, theory_pert_valid = _exact_trial_features(_continuous_transfer(model.J + mask["delta_j"], input_idx, output_idx), z)
            common = valid[bpos] & valid[ppos] & theory_base_valid & theory_pert_valid
            simulation_base = _pearson(features[bpos], common)
            simulation_pert = _pearson(features[ppos], common)
            theory_base_r = _pearson(theory_base, common)
            theory_pert_r = _pearson(theory_pert, common)
            rows.append(
                {
                    **item.to_dict(), "repeat": repeat, "n_trials": N_TRIALS,
                    "n_valid_common": int(common.sum()),
                    "theory_baseline_r": theory_base_r, "theory_perturbed_r": theory_pert_r,
                    "theory_delta_r_long_design": theory_pert_r - theory_base_r,
                    "simulation_baseline_r": simulation_base, "simulation_perturbed_r": simulation_pert,
                    "simulation_delta_r": simulation_pert - simulation_base,
                    "solver": "exact_discrete_Euler_time_domain_equivalent_when_unrectified",
                }
            )
        print(f"large-delta repeat {repeat + 1}/{N_REPEATS}: input {input_area}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(cache_path, index=False, compression="gzip")
    return result


def _explicit_recurrence_check(cfg, model, selected, mask_lookup, out):
    path = out / "explicit_recurrence_check.csv"
    if path.exists():
        cached = pd.read_csv(path)
        if {"output_r", "peak_omega_max_abs_error", "exponent_max_abs_error"}.issubset(cached.columns):
            return cached
    rows = []
    specs = {"long": _window_specs(cfg)["long"]}
    analysis_s = float(specs["long"]["analysis_s"])
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    z, _ = _input_trials(8, int(cfg["seed"]) + 3700000)
    for _, item in selected.sort_values("abs_exact_delta_r", ascending=False).head(3).iterrows():
        mask = mask_lookup[item.mask_id]
        input_idx = model.areas.index(item.input_area)
        output_idx = model.areas.index(item.output_area)
        inputs = _matched_bandlimited_inputs(
            z, DT_MS, analysis_s, burn_s, float(cfg["exp3_nonlinear"]["input_sigma"]),
            int(cfg["seed"]) + 3800000, trial_offset=0,
        )
        networks = [("baseline", model.params, model.J), ("perturbed", dict(model.params) | {"fln_mat": model.fln + mask["delta_w"]}, model.J + mask["delta_j"])]
        for network_name, params, j in networks:
            explicit, rect, *_ = _run_multioutput_batch(params, inputs, DT_MS, burn_steps, np.asarray([output_idx]), input_idx=input_idx)
            transfer = _discrete_transfer(j, input_idx, output_idx, n_steps, float(model.params["tau_exc"]))
            accelerated = np.fft.irfft(np.fft.rfft(inputs[:, burn_steps:], axis=1) * transfer[None, :], n=n_steps, axis=1)
            error = explicit[:, 0] - accelerated
            explicit_features, explicit_valid = _feature_arrays_vectorized(explicit, DT_MS, specs)["long"]
            accelerated_features, accelerated_valid = _feature_arrays_vectorized(accelerated[:, None, :], DT_MS, specs)["long"]
            common = explicit_valid[0] & accelerated_valid[0]
            rows.append(
                {"selection_id": item.selection_id, "network": network_name,
                 "max_abs_timeseries_error": float(np.max(abs(error))),
                 "relative_RMSE": float(np.sqrt(np.mean(error**2)) / max(np.std(explicit[:, 0]), 1e-15)),
                 "output_r": float(np.corrcoef(explicit[:, 0].ravel(), accelerated.ravel())[0, 1]),
                 "peak_omega_max_abs_error": float(np.max(abs(explicit_features[0, common, 0] - accelerated_features[0, common, 0]))),
                 "exponent_max_abs_error": float(np.max(abs(explicit_features[0, common, 1] - accelerated_features[0, common, 1]))),
                 "rectification_count": int(rect.sum())}
            )
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def _plots(out, screen, selected, validation):
    plt.style.use("seaborn-v0_8-whitegrid")
    heat = screen.groupby(["input_area", "output_area"]).abs_exact_delta_r.max().unstack().reindex(index=AREAS9, columns=AREAS9)
    fig, ax = plt.subplots(figsize=(7.6, 6.5), constrained_layout=True)
    image = ax.imshow(heat, cmap="magma", aspect="auto")
    ax.set_xticks(range(9), AREAS9, rotation=45, ha="right")
    ax.set_yticks(range(9), AREAS9)
    ax.set(xlabel="output area", ylabel="input area", title="Maximum exact-resolvent |Delta correlation| across screened masks")
    fig.colorbar(image, ax=ax, label="max |Delta r|")
    save_figure(fig, out / "figures" / "large_delta_screen_9x9")
    plt.close(fig)

    mean_validation = validation.groupby(["selection_id", "selection_class"], as_index=False).agg(
        theory_delta=("theory_delta_r_long_design", "mean"), simulation_delta=("simulation_delta_r", "mean")
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), constrained_layout=True)
    colors = {"high_effect": "#D1495B", "matched_low_effect": "#247BA0"}
    for kind, group in mean_validation.groupby("selection_class"):
        axes[0].scatter(group.theory_delta, group.simulation_delta, s=36, alpha=0.8, color=colors[kind], label=kind.replace("_", " "))
    limit = max(abs(mean_validation[["theory_delta", "simulation_delta"]].to_numpy()).max(), 1e-3)
    axes[0].plot([-limit, limit], [-limit, limit], "k--", lw=1)
    metrics = prediction_metrics(mean_validation.simulation_delta, mean_validation.theory_delta)
    axes[0].set(xlabel="exact-resolvent Delta r", ylabel="long-window simulation Delta r", title=f"All selected: identity R2={metrics['R2']:.3f}")
    axes[0].legend(frameon=False, fontsize=8)
    summary = mean_validation.groupby("selection_class").agg(theory=("theory_delta", lambda x: np.median(abs(x))), simulation=("simulation_delta", lambda x: np.median(abs(x))))
    x = np.arange(len(summary))
    axes[1].bar(x - 0.18, summary.theory, 0.36, label="exact theory", color="#6A4C93")
    axes[1].bar(x + 0.18, summary.simulation, 0.36, label="simulation", color="#2A9D8F")
    axes[1].set_xticks(x, [name.replace("_", " ") for name in summary.index], rotation=15)
    axes[1].set(ylabel="median |Delta r|", title="Blind high-effect screen enriches structural signal")
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(fig, out / "figures" / "large_delta_long_validation")
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), constrained_layout=True)
    examples = mean_validation.merge(selected, on=["selection_id", "selection_class"]).sort_values("abs_exact_delta_r", ascending=False).head(6)
    for ax, (_, item) in zip(axes.flat, examples.iterrows()):
        points = validation[validation.selection_id == item.selection_id]
        ax.scatter(points.theory_delta_r_long_design, points.simulation_delta_r, color="#D1495B", s=35)
        lim = max(abs(points[["theory_delta_r_long_design", "simulation_delta_r"]].to_numpy()).max(), 1e-4)
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1)
        ax.set(xlabel="exact Delta r", ylabel="simulation Delta r", title=f"{item.input_area}->{item.output_area}; {item.family}")
    save_figure(fig, out / "figures" / "large_delta_phase_examples")
    plt.close(fig)


def run(cfg, force=False):
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    # Keep the complete 55-area structural network; AREAS9 only defines the
    # input/readout panel. This is essential for 10-edge in/out-degree masks.
    model = load_model(Path(cfg["model_dir"]))
    mask_cache = out / "candidate_masks.npz"
    if mask_cache.exists() and not force:
        masks = _candidate_masks(model, int(cfg["seed"]) + 3300000)
    else:
        masks = _candidate_masks(model, int(cfg["seed"]) + 3300000)
        np.savez_compressed(mask_cache, **{item["mask_id"]: item["delta_w"] for item in masks})
    mask_lookup = {item["mask_id"]: item for item in masks}
    screen_path = out / "exact_resolvent_screen.csv.gz"
    screen = pd.read_csv(screen_path) if screen_path.exists() and not force else _screen(model, masks, int(cfg["seed"]) + 3400000, out)
    selected_path = out / "selected_blind_masks.csv"
    if selected_path.exists() and not force:
        selected = pd.read_csv(selected_path)
    else:
        selected = _select(screen)
        selected.to_csv(selected_path, index=False)
    validations = [_validate_repeat(cfg, model, selected, mask_lookup, repeat, out) for repeat in range(N_REPEATS)]
    validation = pd.concat(validations, ignore_index=True)
    validation.to_csv(out / "long_validation_all_repeats.csv.gz", index=False, compression="gzip")
    check = _explicit_recurrence_check(cfg, model, selected, mask_lookup, out)
    metrics_rows = []
    for selection_class, group in validation.groupby("selection_class"):
        for repeat, phase in group.groupby("repeat"):
            metrics_rows.append({"selection_class": selection_class, "repeat": repeat, **prediction_metrics(phase.simulation_delta_r, phase.theory_delta_r_long_design)})
        means = group.groupby("selection_id").mean(numeric_only=True)
        metrics_rows.append({"selection_class": selection_class, "repeat": "phase_mean", **prediction_metrics(means.simulation_delta_r, means.theory_delta_r_long_design)})
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(out / "long_validation_metrics.csv", index=False)
    phase_matrix = validation.pivot(index="selection_id", columns="repeat", values="simulation_delta_r")
    phase_correlations = phase_matrix.corr()
    phase_correlations.to_csv(out / "phase_repeat_correlations.csv")
    _plots(out, screen, selected, validation)
    save_json(
        out / "manifest.json",
        {
            "experiment": "blind_exact_resolvent_screen_then_long_window_time_domain_validation",
            "candidate_families": {name: N_CANDIDATE_PER_FAMILY for name in ("uniform", "weight_eta1", "weight_eta2", "targeted_degree")},
            "n_candidates": len(masks), "n_edges_deleted": N_DELETE,
            "screen_io_pairs_per_mask": 81, "n_high_effect_selected": int((selected.selection_class == "high_effect").sum()),
            "n_matched_controls": int((selected.selection_class == "matched_low_effect").sum()),
            "n_trials_per_correlation": N_TRIALS, "phase_repeats": N_REPEATS,
            "analysis_s": _window_specs(cfg)["long"]["analysis_s"],
            "welch_segment_s": _window_specs(cfg)["long"]["spectral"]["segment_s"],
            "exact_theory": "full perturbed resolvent of J + Delta J; no first/second structural expansion",
            "time_domain_solver": "exact discrete-Euler transfer, numerically checked against explicit threshold-linear recurrence",
            "max_explicit_relative_RMSE": float(check.relative_RMSE.max()),
            "min_explicit_output_r": float(check.output_r.min()),
            "max_explicit_peak_omega_error": float(check.peak_omega_max_abs_error.max()),
            "max_explicit_exponent_error": float(check.exponent_max_abs_error.max()),
            "explicit_total_rectification_count": int(check.rectification_count.sum()),
            "runtime_s": time.time() - started,
        },
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(load_config(args.config), force=args.force))


if __name__ == "__main__":
    main()
