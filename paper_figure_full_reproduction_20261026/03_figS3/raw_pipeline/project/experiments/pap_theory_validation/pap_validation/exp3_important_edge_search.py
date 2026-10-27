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
    edge_to_delta_j,
    frequency_grid,
    is_stable,
    load_model,
    save_figure,
    save_json,
)
from .exp3 import _input_trials
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .exp3_large_delta_screen import (
    DT_MS,
    _all_transfer_power,
    _continuous_transfer,
    _discrete_transfer,
    _exact_trial_features,
)
from .exp3_targeted_degree10 import _rho_from_transfer
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .nonlinear_network_validation import _run_multioutput_batch
from .pairwise_nonlinear_window_validation import _window_specs


OUTPUT_NAME = "exp3_important_edge_search_long"
IO_PAIRS = [
    ("A6Va", "V2"), ("PG", "V2"), ("A6DR", "V2"),
    ("V4", "V2"), ("A6Va", "V1"), ("V1", "A6Va"),
]
EDGE_COUNTS = (10, 15, 20, 30, 40, 50, 75, 100)
N_DISCOVERY = 512
N_THEORY_HOLDOUT = 2048
N_SIMULATION = 128
N_PILOT_PHASES = 3
N_HOLDOUT_PHASES = 6
CHUNK_SIZE = 8
ANALYSIS_S = 65.536
N_FINALISTS = 12


def _candidate_edges(model):
    edges = list(zip(*np.nonzero(model.fln)))
    ranked = sorted(edges, key=lambda edge: model.fln[edge], reverse=True)
    selected = set(ranked[:500])
    route_areas = {model.areas.index(name) for name in ("V1", "V2", "V4", "PG", "A6Va")}
    for target, source in edges:
        if target in route_areas or source in route_areas:
            selected.add((int(target), int(source)))
    return sorted(selected)


def _delta_w(model, edges):
    result = np.zeros_like(model.fln)
    for target, source in edges:
        result[target, source] = -model.fln[target, source]
    return result


def _single_edge_screen(model, edges, sigma, out):
    path = out / "single_edge_exact_resolvent_screen.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    frequencies = frequency_grid()
    area_indices = [model.areas.index(area) for area in AREAS9]
    baseline_transfers = _all_transfer_power(model.J, area_indices, frequencies)
    baseline = {
        pair: _rho_from_transfer(
            baseline_transfers[AREAS9.index(pair[0]), AREAS9.index(pair[1])], sigma, frequencies
        )
        for pair in IO_PAIRS
    }
    rows = []
    for edge_pos, (target, source) in enumerate(edges):
        delta_j = edge_to_delta_j(model, target, source, -float(model.fln[target, source]))
        transfers = _all_transfer_power(model.J + delta_j, area_indices, frequencies)
        for input_area, output_area in IO_PAIRS:
            perturbed = _rho_from_transfer(
                transfers[AREAS9.index(input_area), AREAS9.index(output_area)], sigma, frequencies
            )
            rows.append(
                {
                    "edge_id": edge_pos, "source_idx": source, "target_idx": target,
                    "source_area": model.areas[source], "target_area": model.areas[target],
                    "weight": model.fln[target, source], "input_area": input_area,
                    "output_area": output_area, "baseline_r": baseline[(input_area, output_area)],
                    "single_delete_r": perturbed, "single_delete_delta_r": perturbed - baseline[(input_area, output_area)],
                }
            )
        if (edge_pos + 1) % 25 == 0:
            print(f"important-edge single screen {edge_pos + 1}/{len(edges)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False, compression="gzip")
    return result


def _common_r(values_a, valid_a, values_b, valid_b):
    use = valid_a & valid_b
    if use.sum() < 16:
        return np.nan, np.nan, int(use.sum())
    left = stats.pearsonr(values_a[use, 0], values_a[use, 1]).statistic
    right = stats.pearsonr(values_b[use, 0], values_b[use, 1]).statistic
    return float(left), float(right), int(use.sum())


def _theory_values(model, input_area, output_area, edges, z):
    input_idx, output_idx = model.areas.index(input_area), model.areas.index(output_area)
    delta_w = _delta_w(model, edges)
    perturbed_j = apply_fln_delta(model, delta_w)
    base_values, base_valid = _exact_trial_features(_continuous_transfer(model.J, input_idx, output_idx), z)
    pert_values, pert_valid = _exact_trial_features(_continuous_transfer(perturbed_j, input_idx, output_idx), z)
    base_r, pert_r, n_valid = _common_r(base_values, base_valid, pert_values, pert_valid)
    return base_r, pert_r, n_valid, perturbed_j, delta_w


def _cumulative_theory_search(model, screen, z, out):
    rows = []
    edge_sets = {}
    for input_area, output_area in IO_PAIRS:
        subset = screen[(screen.input_area == input_area) & (screen.output_area == output_area)].copy()
        for direction in ("positive", "negative"):
            ranked = subset.sort_values("single_delete_delta_r", ascending=(direction == "negative"))
            if direction == "positive":
                ranked = ranked[ranked.single_delete_delta_r > 0]
            else:
                ranked = ranked[ranked.single_delete_delta_r < 0]
            ranked_edges = list(zip(ranked.target_idx.astype(int), ranked.source_idx.astype(int)))
            for count in EDGE_COUNTS:
                if len(ranked_edges) < count:
                    continue
                edges = ranked_edges[:count]
                base_r, pert_r, n_valid, perturbed_j, _ = _theory_values(
                    model, input_area, output_area, edges, z
                )
                mask_id = f"{input_area}_{output_area}_{direction}_{count:03d}"
                rows.append(
                    {
                        "mask_id": mask_id, "input_area": input_area, "output_area": output_area,
                        "direction": direction, "n_edges": count, "baseline_r": base_r,
                        "perturbed_r": pert_r, "delta_r": pert_r - base_r,
                        "abs_delta_r": abs(pert_r - base_r), "n_valid_common": n_valid,
                        "stable": is_stable(perturbed_j),
                    }
                )
                edge_sets[mask_id] = edges
    result = pd.DataFrame(rows)
    result.to_csv(out / "cumulative_theory_search.csv", index=False)
    with (out / "cumulative_edge_sets.json").open("w") as stream:
        json.dump({key: [[int(t), int(s)] for t, s in value] for key, value in edge_sets.items()}, stream, indent=2)
    return result, edge_sets


def _theory_finalists(model, search, edge_sets, z):
    eligible = search[(search.stable) & (search.n_valid_common >= int(0.80 * N_DISCOVERY))].copy()
    eligible = eligible.sort_values("abs_delta_r", ascending=False)
    chosen = []
    for row in eligible.itertuples(index=False):
        base_r, pert_r, n_valid, _, _ = _theory_values(
            model, row.input_area, row.output_area, edge_sets[row.mask_id], z
        )
        chosen.append(
            {
                **row._asdict(), "holdout_theory_baseline_r": base_r,
                "holdout_theory_perturbed_r": pert_r,
                "holdout_theory_delta_r": pert_r - base_r,
                "holdout_theory_n_valid": n_valid,
            }
        )
        if len(chosen) >= N_FINALISTS:
            break
    return pd.DataFrame(chosen)


def _simulate_finalists(cfg, model, finalists, edge_sets, z, out):
    cache = out / "pilot_and_holdout_phase_correlations.csv"
    feature_cache = out / "representative_features.npz"
    if cache.exists() and feature_cache.exists():
        return pd.read_csv(cache), np.load(feature_cache)
    specs = {"long": dict(_window_specs(cfg)["long"])}
    specs["long"]["analysis_s"] = ANALYSIS_S
    analysis_s = ANALYSIS_S
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    tau_e = float(model.params["tau_exc"])
    rows = []
    examples = {}
    total_phases = N_PILOT_PHASES + N_HOLDOUT_PHASES
    for input_area, group in finalists.groupby("input_area", sort=False):
        input_idx = model.areas.index(input_area)
        entries = []
        baseline_positions = {}
        for output_area in group.output_area.unique():
            output_idx = model.areas.index(output_area)
            baseline_positions[output_area] = len(entries)
            entries.append((f"baseline_{output_area}", _discrete_transfer(model.J, input_idx, output_idx, n_steps, tau_e)))
        perturbed_positions = {}
        for row in group.itertuples(index=False):
            edges = edge_sets[row.mask_id]
            _, _, _, perturbed_j, _ = _theory_values(model, row.input_area, row.output_area, edges, z)
            output_idx = model.areas.index(row.output_area)
            perturbed_positions[row.mask_id] = len(entries)
            entries.append((row.mask_id, _discrete_transfer(perturbed_j, input_idx, output_idx, n_steps, tau_e)))
        for phase in range(total_phases):
            features = np.empty((len(entries), len(z), 2))
            valid = np.empty((len(entries), len(z)), dtype=bool)
            for start in range(0, len(z), CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, len(z))
                inputs = _matched_bandlimited_inputs(
                    z[start:stop], DT_MS, analysis_s, burn_s,
                    float(cfg["exp3_nonlinear"]["input_sigma"]),
                    int(cfg["seed"]) + 4400000 + phase * 100000 + AREAS9.index(input_area) * 1000,
                    trial_offset=start,
                )
                input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=1)
                outputs = np.empty((stop - start, len(entries), n_steps), dtype=np.float32)
                for entry_pos, (_, transfer) in enumerate(entries):
                    outputs[:, entry_pos] = np.fft.irfft(
                        input_fft * transfer[None, :], n=n_steps, axis=1
                    ).astype(np.float32)
                values, is_valid = _feature_arrays_vectorized(outputs, DT_MS, specs)["long"]
                features[:, start:stop] = values
                valid[:, start:stop] = is_valid
            for row in group.itertuples(index=False):
                bpos = baseline_positions[row.output_area]
                ppos = perturbed_positions[row.mask_id]
                use = valid[bpos] & valid[ppos]
                baseline_r = stats.pearsonr(features[bpos, use, 0], features[bpos, use, 1]).statistic
                perturbed_r = stats.pearsonr(features[ppos, use, 0], features[ppos, use, 1]).statistic
                rows.append(
                    {
                        "mask_id": row.mask_id, "input_area": row.input_area,
                        "output_area": row.output_area, "phase": phase,
                        "split": "pilot" if phase < N_PILOT_PHASES else "holdout",
                        "n_valid_common": int(use.sum()), "simulation_baseline_r": baseline_r,
                        "simulation_perturbed_r": perturbed_r,
                        "simulation_delta_r": perturbed_r - baseline_r,
                    }
                )
                if phase == N_PILOT_PHASES:
                    examples[f"{row.mask_id}_baseline"] = features[bpos]
                    examples[f"{row.mask_id}_perturbed"] = features[ppos]
                    examples[f"{row.mask_id}_valid"] = use
            print(f"important-edge simulation {input_area}: phase {phase + 1}/{total_phases}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(cache, index=False)
    np.savez_compressed(feature_cache, **examples)
    return result, np.load(feature_cache)


def _choose_with_pilot(finalists, phases):
    rows = []
    for row in finalists.itertuples(index=False):
        pilot = phases[(phases.mask_id == row.mask_id) & (phases.split == "pilot")]
        mean = pilot.simulation_delta_r.mean()
        sd = pilot.simulation_delta_r.std(ddof=1)
        sign_rate = np.mean(np.sign(pilot.simulation_delta_r) == np.sign(row.holdout_theory_delta_r))
        scale = max(abs(row.holdout_theory_delta_r), 0.005)
        score = abs(mean - row.holdout_theory_delta_r) / scale + sd / scale + (1.0 - sign_rate)
        rows.append({"mask_id": row.mask_id, "pilot_mean_delta_r": mean, "pilot_sd_delta_r": sd, "pilot_sign_rate": sign_rate, "pilot_score": score})
    scores = pd.DataFrame(rows).sort_values("pilot_score")
    return scores.iloc[0].mask_id, scores


def _final_summary(finalists, phases, selected_id):
    theory = finalists[finalists.mask_id == selected_id].iloc[0]
    holdout = phases[(phases.mask_id == selected_id) & (phases.split == "holdout")]
    n = len(holdout)
    mean_base = holdout.simulation_baseline_r.mean()
    mean_pert = holdout.simulation_perturbed_r.mean()
    mean_delta = holdout.simulation_delta_r.mean()
    sd_delta = holdout.simulation_delta_r.std(ddof=1)
    sem = sd_delta / np.sqrt(n)
    tcrit = stats.t.ppf(0.975, n - 1)
    return pd.DataFrame(
        [{
            "mask_id": selected_id, "input_area": theory.input_area,
            "output_area": theory.output_area, "n_edges": theory.n_edges,
            "n_simulation_trials_per_phase": N_SIMULATION, "n_pilot_phases": N_PILOT_PHASES,
            "n_holdout_phases": N_HOLDOUT_PHASES,
            "theory_baseline_r": theory.holdout_theory_baseline_r,
            "theory_perturbed_r": theory.holdout_theory_perturbed_r,
            "theory_delta_r": theory.holdout_theory_delta_r,
            "simulation_baseline_r_mean": mean_base,
            "simulation_baseline_r_sd": holdout.simulation_baseline_r.std(ddof=1),
            "simulation_perturbed_r_mean": mean_pert,
            "simulation_perturbed_r_sd": holdout.simulation_perturbed_r.std(ddof=1),
            "simulation_delta_r_mean": mean_delta, "simulation_delta_r_sd": sd_delta,
            "simulation_delta_r_ci_low": mean_delta - tcrit * sem,
            "simulation_delta_r_ci_high": mean_delta + tcrit * sem,
            "holdout_sign_consistency": np.mean(np.sign(holdout.simulation_delta_r) == np.sign(theory.holdout_theory_delta_r)),
            "absolute_theory_simulation_delta_error": abs(mean_delta - theory.holdout_theory_delta_r),
        }]
    )


def _explicit_check(cfg, model, selected, edges, z, out):
    input_idx = model.areas.index(selected.input_area)
    output_idx = model.areas.index(selected.output_area)
    _, _, _, perturbed_j, delta_w = _theory_values(model, selected.input_area, selected.output_area, edges, z)
    specs = {"long": dict(_window_specs(cfg)["long"])}
    specs["long"]["analysis_s"] = ANALYSIS_S
    analysis_s = ANALYSIS_S
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    inputs = _matched_bandlimited_inputs(
        z[:8], DT_MS, analysis_s, burn_s, float(cfg["exp3_nonlinear"]["input_sigma"]),
        int(cfg["seed"]) + 4900000,
    )
    params = dict(model.params)
    params["fln_mat"] = model.fln + delta_w
    explicit, rect, *_ = _run_multioutput_batch(
        params, inputs, DT_MS, burn_steps, np.asarray([output_idx]), input_idx=input_idx
    )
    transfer = _discrete_transfer(perturbed_j, input_idx, output_idx, n_steps, float(model.params["tau_exc"]))
    accelerated = np.fft.irfft(
        np.fft.rfft(inputs[:, burn_steps:], axis=1) * transfer[None, :], n=n_steps, axis=1
    )
    return {
        "output_r": float(np.corrcoef(explicit[:, 0].ravel(), accelerated.ravel())[0, 1]),
        "relative_RMSE": float(np.sqrt(np.mean((explicit[:, 0] - accelerated) ** 2)) / np.std(explicit[:, 0])),
        "rectification_count": int(rect.sum()),
    }


def _plot(out, selected, edges_frame, phases, examples):
    plt.style.use("seaborn-v0_8-whitegrid")
    holdout = phases[(phases.mask_id == selected.mask_id) & (phases.split == "holdout")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), constrained_layout=True)
    ordered = edges_frame.sort_values("weight")
    labels = ordered.source_area + " -> " + ordered.target_area
    axes[0, 0].barh(np.arange(len(ordered)), ordered.weight, color="#247BA0")
    axes[0, 0].set_yticks(np.arange(len(ordered)), labels, fontsize=7)
    axes[0, 0].set(xlabel="deleted FLN weight", title=f"Selected important edges (n={len(ordered)})")

    x = np.arange(2)
    theory = [selected.theory_baseline_r, selected.theory_perturbed_r]
    sim = [selected.simulation_baseline_r_mean, selected.simulation_perturbed_r_mean]
    sim_sd = [selected.simulation_baseline_r_sd, selected.simulation_perturbed_r_sd]
    axes[0, 1].bar(x - 0.18, theory, 0.36, color="#6A4C93", label="exact resolvent")
    axes[0, 1].bar(x + 0.18, sim, 0.36, yerr=sim_sd, color="#2A9D8F", capsize=4, label="long-window simulation")
    axes[0, 1].set_xticks(x, ["baseline", "deleted"])
    axes[0, 1].set(ylabel="P-A correlation r", title="Absolute correlation before and after deletion")
    axes[0, 1].legend(frameon=False)

    axes[1, 0].scatter(holdout.phase - N_PILOT_PHASES + 1, holdout.simulation_delta_r, color="#D1495B", s=45, label="holdout phases")
    axes[1, 0].axhline(selected.theory_delta_r, color="#6A4C93", lw=2, label="exact resolvent")
    axes[1, 0].axhline(0, color="0.25", ls=":")
    axes[1, 0].set(xlabel="independent holdout phase", ylabel="Delta correlation", title=f"mean={selected.simulation_delta_r_mean:.4f}, 95% CI [{selected.simulation_delta_r_ci_low:.4f}, {selected.simulation_delta_r_ci_high:.4f}]")
    axes[1, 0].legend(frameon=False)

    use = examples[f"{selected.mask_id}_valid"].astype(bool)
    base = examples[f"{selected.mask_id}_baseline"][use]
    pert = examples[f"{selected.mask_id}_perturbed"][use]
    axes[1, 1].scatter(base[:, 0] / (2 * np.pi), base[:, 1], s=14, alpha=0.55, color="#247BA0", label="baseline")
    axes[1, 1].scatter(pert[:, 0] / (2 * np.pi), pert[:, 1], s=14, alpha=0.55, color="#D1495B", label="deleted")
    axes[1, 1].set(xlabel="oscillation peak [Hz]", ylabel="aperiodic exponent", title="Representative long-window realization")
    axes[1, 1].legend(frameon=False)
    save_figure(fig, out / "figures" / "important_edge_robust_validation")
    plt.close(fig)


def run(cfg, force=False):
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(Path(cfg["model_dir"]))
    edges = _candidate_edges(model)
    _, u = _input_trials(N_DISCOVERY, int(cfg["seed"]) + 4100000)
    screen = _single_edge_screen(model, edges, np.cov(u, rowvar=False, ddof=1), out)
    z_discovery, _ = _input_trials(N_DISCOVERY, int(cfg["seed"]) + 4200000)
    search, edge_sets = _cumulative_theory_search(model, screen, z_discovery, out)
    z_holdout, _ = _input_trials(N_THEORY_HOLDOUT, int(cfg["seed"]) + 4300000)
    finalists = _theory_finalists(model, search, edge_sets, z_holdout)
    finalists.to_csv(out / "theory_finalists.csv", index=False)
    z_simulation, _ = _input_trials(N_SIMULATION, int(cfg["seed"]) + 4350000)
    phases, examples = _simulate_finalists(cfg, model, finalists, edge_sets, z_simulation, out)
    selected_id, pilot_scores = _choose_with_pilot(finalists, phases)
    pilot_scores.to_csv(out / "pilot_selection_scores.csv", index=False)
    summary = _final_summary(finalists, phases, selected_id)
    summary.to_csv(out / "final_condition_summary.csv", index=False)
    selected = summary.iloc[0]
    selected_edges = edge_sets[selected_id]
    edge_frame = pd.DataFrame(
        [
            {"edge_number": pos + 1, "source_area": model.areas[source],
             "target_area": model.areas[target], "weight": model.fln[target, source],
             "source_idx": source, "target_idx": target}
            for pos, (target, source) in enumerate(selected_edges)
        ]
    )
    edge_frame.to_csv(out / "selected_deleted_edges.csv", index=False)
    explicit = _explicit_check(cfg, model, selected, selected_edges, z_simulation, out)
    _plot(out, selected, edge_frame, phases, examples)
    save_json(
        out / "manifest.json",
        {
            "experiment": "theory-ranked important-edge deletion with pilot/holdout phase validation",
            "selection_rule": "single-edge exact-resolvent ranking; cumulative exact-spectrum theory; no simulation in edge ranking",
            "candidate_edge_count": len(edges), "io_pairs": IO_PAIRS,
            "edge_counts": EDGE_COUNTS, "theory_discovery_trials": N_DISCOVERY,
            "theory_holdout_trials": N_THEORY_HOLDOUT,
            "simulation_trials_per_phase": N_SIMULATION,
            "pilot_phases": N_PILOT_PHASES, "holdout_phases": N_HOLDOUT_PHASES,
            "analysis_s": ANALYSIS_S,
            "welch_segment_s": _window_specs(cfg)["long"]["spectral"]["segment_s"],
            "selected_mask_id": selected_id, "selected_edge_count": len(selected_edges),
            "explicit_recurrence_check": explicit,
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
