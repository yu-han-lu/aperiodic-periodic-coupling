from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .core import AREAS9, apply_fln_delta, frequency_grid, load_model, save_json
from .exp3 import _input_trials
from .io import load_config, prepare_output
from .nonlinear_network_validation import _fourier_inputs, _run_multioutput_batch
from .pairwise_nonlinear_window_validation import (
    _cov_r,
    _delta_theory,
    _feature_arrays,
    _heatmap_grid,
    _pair_metrics,
    _plot_exp3,
    _rooted,
    _window_specs,
)


LONG_SOURCE = "finite_window_exp3_pairwise_single_phase_12c"
OUTPUT_NAME = "finite_window_exp3_pairwise_single_phase_12c_exact"


def run(cfg: dict) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(_rooted(cfg["model_dir"]), AREAS9)
    specs = _window_specs(cfg)
    short_spec = {"short": specs["short"]}
    ecfg = cfg["exp3_nonlinear"]
    n_trials = 64
    n_masks = 120
    phase_seed_offset = 220000
    dt = float(ecfg["dt_ms"])
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    short_steps = int(round(specs["short"]["analysis_s"] * 1000.0 / dt))
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

    all_features = np.empty((n_masks + 1, len(AREAS9), n_trials, 2), dtype=float)
    all_valid = np.empty((n_masks + 1, len(AREAS9), n_trials), dtype=bool)
    short_rows = []
    short_qc = []
    freqs = frequency_grid()
    for input_pos, input_area in enumerate(AREAS9):
        inputs = _fourier_inputs(
            z, dt, short_steps, burn_steps, float(ecfg["input_sigma"]),
            int(cfg["seed"]) + phase_seed_offset,
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
            values, valid = _feature_arrays(outputs, dt, short_spec)["short"]
            all_features[network_pos] = values
            all_valid[network_pos] = valid
            short_qc.append({
                "window": "short", "input_area": input_area, "network": network_id,
                "rectification_fraction": float(rect.sum() / (2 * len(model.areas) * inputs.shape[1] * n_trials)),
                "min_rate_E": float(min_e.min()), "min_rate_I": float(min_i.min()),
                "min_drive_E": float(min_de.min()), "min_drive_I": float(min_di.min()),
            })
            if network_pos % 10 == 0 or network_pos == n_masks:
                print(f"exact 12c short input={input_area}: network {network_pos}/{n_masks}", flush=True)

        for output_pos, output_area in enumerate(AREAS9):
            use = np.all(all_valid[:, output_pos, :], axis=0)
            if use.sum() < 8:
                continue
            sigma = np.cov(u[use], rowvar=False, ddof=1)
            _, baseline_r = _cov_r(all_features[0, output_pos, use])
            for mask_pos, (perturbation_id, _, delta_j) in enumerate(masks, 1):
                _, perturbed_r = _cov_r(all_features[mask_pos, output_pos, use])
                first, second = _delta_theory(model, sigma, freqs, input_pos, output_pos, delta_j)
                short_rows.append({
                    "input_area": input_area, "output_area": output_area,
                    "window": "short", "perturbation_id": perturbation_id,
                    "n_input_trials": n_trials, "n_valid_common": int(use.sum()),
                    "baseline_actual_r": baseline_r, "perturbed_actual_r": perturbed_r,
                    "delta_rho_actual": perturbed_r - baseline_r,
                    "delta_rho_first": first, "delta_rho_second": second,
                })

    long_source = _rooted(cfg["output_dir"]) / LONG_SOURCE
    prior_results = pd.read_csv(long_source / "delta_correlation_predictions.csv.gz")
    long_rows = prior_results[prior_results.window == "long"].copy()
    results = pd.concat([pd.DataFrame(short_rows), long_rows], ignore_index=True)
    stacked = pd.concat([
        results.assign(order=order, predicted=results[f"delta_rho_{order}"])
        for order in ("first", "second")
    ], ignore_index=True)
    metrics = _pair_metrics(stacked, "delta_rho_actual", "predicted")
    long_qc = pd.read_csv(long_source / "rectification_qc.csv").assign(window="long")
    qc = pd.concat([pd.DataFrame(short_qc), long_qc], ignore_index=True)
    results.to_csv(out / "delta_correlation_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "delta_correlation_R2_by_pair.csv", index=False)
    qc.to_csv(out / "rectification_qc.csv", index=False)
    save_json(out / "manifest.json", {
        "experiment": "exact_Figure_12c_single_phase_9x9_structural_finite_window",
        "actual_simulation": "explicit threshold-linear time-domain Euler integration",
        "areas": AREAS9,
        "n_input_trials": n_trials,
        "n_masks": n_masks,
        "phase_seed_offset": phase_seed_offset,
        "window_input_mode": "short and long Fourier inputs synthesized separately, exactly as Figure 12c",
        "short_simulation": "rerun in this output",
        "long_simulation_source": str(long_source),
        "validity_rule": "per input-output-window intersection across baseline and all 120 masks",
        "short_analysis_s": specs["short"]["analysis_s"],
        "long_analysis_s": specs["long"]["analysis_s"],
        "runtime_s_short_rerun": time.time() - started,
    })
    _plot_exp3(out, results, metrics)
    _heatmap_grid(
        metrics,
        "R2",
        "Exact Figure 12c single phase: identity R2 across 9 inputs x 9 outputs (120 masks)",
        out / "figures" / "exp3_12c_single_phase_9x9_identity_R2",
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
