from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import apply_fln_delta, load_model, save_figure, save_json
from .exp3 import _input_trials
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .exp3_large_delta_screen import DT_MS, _continuous_transfer, _discrete_transfer, _exact_trial_features
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .pairwise_nonlinear_window_validation import _window_specs


OUTPUT_NAME = "exp3_important_edge_ensemble_validation"
SOURCE_NAME = "exp3_important_edge_search_long"
N_TRIALS = 256
N_ENSEMBLES = 10
PHASES_PER_ENSEMBLE = 24
CHUNK_SIZE = 8


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def _correlation(values, valid):
    return float(stats.pearsonr(values[valid, 0], values[valid, 1]).statistic)


def run(cfg, force=False):
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    feature_path = out / "phase_features.npz"
    model = load_model(Path(cfg["model_dir"]))
    source = _rooted(cfg["output_dir"]) / SOURCE_NAME
    selected = pd.read_csv(source / "final_condition_summary.csv").iloc[0]
    edges = pd.read_csv(source / "selected_deleted_edges.csv")
    edges.to_csv(out / "selected_deleted_edges.csv", index=False)
    delta_w = np.zeros_like(model.fln)
    for row in edges.itertuples(index=False):
        delta_w[int(row.target_idx), int(row.source_idx)] = -model.fln[int(row.target_idx), int(row.source_idx)]
    perturbed_j = apply_fln_delta(model, delta_w)
    input_idx = model.areas.index(selected.input_area)
    output_idx = model.areas.index(selected.output_area)
    z, _ = _input_trials(N_TRIALS, int(cfg["seed"]) + 5100000)
    specs = {"long": _window_specs(cfg)["long"]}
    analysis_s = float(specs["long"]["analysis_s"])
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    total_phases = N_ENSEMBLES * PHASES_PER_ENSEMBLE
    start_phase = 0
    if feature_path.exists() and not force:
        cached = np.load(feature_path)
        old_features = cached["features"]
        old_valid = cached["valid"]
        if old_features.shape[0] >= total_phases:
            features = old_features[:total_phases]
            valid = old_valid[:total_phases]
            start_phase = total_phases
        else:
            start_phase = old_features.shape[0]
            features = np.empty((total_phases, 2, N_TRIALS, 2), dtype=float)
            valid = np.empty((total_phases, 2, N_TRIALS), dtype=bool)
            features[:start_phase] = old_features
            valid[:start_phase] = old_valid
    else:
        features = np.empty((total_phases, 2, N_TRIALS, 2), dtype=float)
        valid = np.empty((total_phases, 2, N_TRIALS), dtype=bool)
    if start_phase < total_phases:
        transfers = np.asarray(
            [
                _discrete_transfer(model.J, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
                _discrete_transfer(perturbed_j, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
            ]
        )
        for phase in range(start_phase, total_phases):
            for start in range(0, N_TRIALS, CHUNK_SIZE):
                stop = min(start + CHUNK_SIZE, N_TRIALS)
                inputs = _matched_bandlimited_inputs(
                    z[start:stop], DT_MS, analysis_s, burn_s,
                    float(cfg["exp3_nonlinear"]["input_sigma"]),
                    int(cfg["seed"]) + 5200000 + phase * 100000,
                    trial_offset=start,
                )
                input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=1)
                outputs = np.empty((stop - start, 2, n_steps), dtype=np.float32)
                for network_pos in range(2):
                    outputs[:, network_pos] = np.fft.irfft(
                        input_fft * transfers[network_pos][None, :], n=n_steps, axis=1
                    ).astype(np.float32)
                values, is_valid = _feature_arrays_vectorized(outputs, DT_MS, specs)["long"]
                features[phase, :, start:stop] = values
                valid[phase, :, start:stop] = is_valid
            print(f"important-edge phase ensemble {phase + 1}/{total_phases}", flush=True)
        np.savez_compressed(feature_path, features=features, valid=valid, z=z)

    theory_base, theory_base_valid = _exact_trial_features(
        _continuous_transfer(model.J, input_idx, output_idx), z
    )
    theory_pert, theory_pert_valid = _exact_trial_features(
        _continuous_transfer(perturbed_j, input_idx, output_idx), z
    )
    global_valid_fraction = valid.mean(axis=0)
    per_ensemble_enough = np.empty((N_ENSEMBLES, 2, N_TRIALS), dtype=bool)
    for ensemble in range(N_ENSEMBLES):
        lo = ensemble * PHASES_PER_ENSEMBLE
        hi = lo + PHASES_PER_ENSEMBLE
        per_ensemble_enough[ensemble] = (
            valid[lo:hi].sum(axis=0) >= int(0.75 * PHASES_PER_ENSEMBLE)
        )
    fixed_use = (
        theory_base_valid
        & theory_pert_valid
        & (global_valid_fraction[0] >= 0.75)
        & (global_valid_fraction[1] >= 0.75)
        & per_ensemble_enough.all(axis=(0, 1))
    )
    fixed_theory_base_r = _correlation(theory_base, fixed_use)
    fixed_theory_pert_r = _correlation(theory_pert, fixed_use)
    rows = []
    ensemble_features = {}
    for ensemble in range(N_ENSEMBLES):
        lo = ensemble * PHASES_PER_ENSEMBLE
        hi = lo + PHASES_PER_ENSEMBLE
        block_features = features[lo:hi]
        block_valid = valid[lo:hi]
        averaged = np.empty((2, N_TRIALS, 2))
        enough = np.empty((2, N_TRIALS), dtype=bool)
        for network_pos in range(2):
            counts = block_valid[:, network_pos].sum(axis=0)
            enough[network_pos] = counts >= int(0.75 * PHASES_PER_ENSEMBLE)
            summed = np.where(
                block_valid[:, network_pos, :, None], block_features[:, network_pos], 0.0
            ).sum(axis=0)
            averaged[network_pos] = summed / np.maximum(counts[:, None], 1)
        use = fixed_use & np.isfinite(averaged).all(axis=(0, 2))
        simulation_base = _correlation(averaged[0], use)
        simulation_pert = _correlation(averaged[1], use)
        rows.append(
            {
                "ensemble": ensemble + 1, "n_phases_averaged": PHASES_PER_ENSEMBLE,
                "n_valid_common": int(use.sum()), "theory_baseline_r": fixed_theory_base_r,
                "simulation_baseline_r": simulation_base, "theory_perturbed_r": fixed_theory_pert_r,
                "simulation_perturbed_r": simulation_pert,
                "theory_delta_r": fixed_theory_pert_r - fixed_theory_base_r,
                "simulation_delta_r": simulation_pert - simulation_base,
            }
        )
        ensemble_features[ensemble] = (averaged, use)
    result = pd.DataFrame(rows)
    result.to_csv(out / "ensemble_correlations.csv", index=False)
    delta_mean = result.simulation_delta_r.mean()
    delta_sd = result.simulation_delta_r.std(ddof=1)
    sem = delta_sd / np.sqrt(N_ENSEMBLES)
    tcrit = stats.t.ppf(0.975, N_ENSEMBLES - 1)
    summary = pd.DataFrame(
        [{
            "mask_id": selected.mask_id, "input_area": selected.input_area,
            "output_area": selected.output_area, "n_deleted_edges": len(edges),
            "n_parameter_trials": N_TRIALS, "n_independent_ensembles": N_ENSEMBLES,
            "phases_per_trial_per_ensemble": PHASES_PER_ENSEMBLE,
            "theory_baseline_r_mean": result.theory_baseline_r.mean(),
            "simulation_baseline_r_mean": result.simulation_baseline_r.mean(),
            "simulation_baseline_r_sd": result.simulation_baseline_r.std(ddof=1),
            "theory_perturbed_r_mean": result.theory_perturbed_r.mean(),
            "simulation_perturbed_r_mean": result.simulation_perturbed_r.mean(),
            "simulation_perturbed_r_sd": result.simulation_perturbed_r.std(ddof=1),
            "theory_delta_r_mean": result.theory_delta_r.mean(),
            "simulation_delta_r_mean": delta_mean, "simulation_delta_r_sd": delta_sd,
            "simulation_delta_r_ci_low": delta_mean - tcrit * sem,
            "simulation_delta_r_ci_high": delta_mean + tcrit * sem,
            "simulation_delta_r_one_sample_p": stats.ttest_1samp(result.simulation_delta_r, 0.0).pvalue,
            "sign_consistency": np.mean(np.sign(result.simulation_delta_r) == np.sign(result.theory_delta_r)),
            "absolute_delta_error": abs(delta_mean - result.theory_delta_r.mean()),
        }]
    )
    summary.to_csv(out / "final_ensemble_summary.csv", index=False)
    _plot(out, summary.iloc[0], edges, result, ensemble_features[0])
    _plot_edge_table(out, edges)
    save_json(
        out / "manifest.json",
        {
            "experiment": "frozen important-edge mask validated by independent phase-ensemble long-window simulations",
            "source_selection": SOURCE_NAME, "selection_uses_current_phase_data": False,
            "analysis_s_per_phase": analysis_s, "n_parameter_trials": N_TRIALS,
            "n_ensembles": N_ENSEMBLES, "phases_per_ensemble": PHASES_PER_ENSEMBLE,
            "total_time_domain_realizations_per_network": N_TRIALS * total_phases,
            "runtime_s": time.time() - started,
        },
    )
    return out


def _plot(out, summary, edges, result, example):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), constrained_layout=True)
    ordered = edges.sort_values("weight").tail(25)
    labels = ordered.source_area + " -> " + ordered.target_area
    axes[0, 0].barh(np.arange(len(ordered)), ordered.weight, color="#247BA0")
    axes[0, 0].set_yticks(np.arange(len(ordered)), labels, fontsize=7)
    axes[0, 0].set(xlabel="deleted FLN weight", title=f"Largest 25 of {len(edges)} selected edges")
    x = np.arange(2)
    axes[0, 1].bar(x - 0.18, [summary.theory_baseline_r_mean, summary.theory_perturbed_r_mean], 0.36, color="#6A4C93", label="exact resolvent")
    axes[0, 1].bar(
        x + 0.18, [summary.simulation_baseline_r_mean, summary.simulation_perturbed_r_mean],
        0.36, yerr=[summary.simulation_baseline_r_sd, summary.simulation_perturbed_r_sd],
        color="#2A9D8F", capsize=4, label="phase-ensemble simulation",
    )
    axes[0, 1].set_xticks(x, ["baseline", "deleted"])
    axes[0, 1].set(ylabel="P-A correlation r", title="Absolute correlation (mean +/- ensemble SD)")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].scatter(result.ensemble, result.simulation_delta_r, s=48, color="#D1495B", label="independent ensembles")
    axes[1, 0].plot(result.ensemble, result.theory_delta_r, color="#6A4C93", lw=2, label="exact resolvent")
    axes[1, 0].axhline(0, color="0.25", ls=":")
    axes[1, 0].set(
        xlabel="holdout ensemble", ylabel="Delta correlation",
        title=f"simulation mean={summary.simulation_delta_r_mean:.4f}; 95% CI [{summary.simulation_delta_r_ci_low:.4f}, {summary.simulation_delta_r_ci_high:.4f}]",
    )
    axes[1, 0].legend(frameon=False)
    averaged, use = example
    axes[1, 1].scatter(averaged[0, use, 0] / (2 * np.pi), averaged[0, use, 1], s=14, alpha=0.55, color="#247BA0", label="baseline")
    axes[1, 1].scatter(averaged[1, use, 0] / (2 * np.pi), averaged[1, use, 1], s=14, alpha=0.55, color="#D1495B", label="deleted")
    axes[1, 1].set(xlabel="oscillation peak [Hz]", ylabel="aperiodic exponent", title="Representative phase-ensemble trial features")
    axes[1, 1].legend(frameon=False)
    save_figure(fig, out / "figures" / "important_edge_phase_ensemble_validation")
    plt.close(fig)


def _plot_edge_table(out, edges):
    ordered = edges.sort_values("edge_number").reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 9), constrained_layout=True)
    for column, ax in enumerate(axes):
        ax.axis("off")
        part = ordered.iloc[column * 25 : (column + 1) * 25]
        lines = [
            f"{int(row.edge_number):02d}. {row.source_area} -> {row.target_area}    FLN={row.weight:.6g}"
            for row in part.itertuples(index=False)
        ]
        ax.text(0.02, 0.98, "\n".join(lines), va="top", family="monospace", fontsize=10, linespacing=1.45)
    fig.suptitle("Complete list of 50 theory-ranked deleted edges", fontsize=14)
    save_figure(fig, out / "figures" / "selected_deleted_edges_complete")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(load_config(args.config), force=args.force))


if __name__ == "__main__":
    main()
