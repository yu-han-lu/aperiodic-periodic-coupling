from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import AREAS9, apply_fln_delta, feature_from_spectrum, frequency_grid, load_model, save_figure, save_json
from .exp3 import _input_trials
from .io import load_config, prepare_output
from .nonlinear_network_validation import _fourier_inputs, _run_multioutput_batch, _timeseries_features
from .pairwise_nonlinear_window_validation import (
    ROOT,
    WINDOWS,
    _annotated_scatter,
    _delta_theory,
    _heatmap_grid,
    _plot_identity_comparison,
    _pair_metrics,
    _window_specs,
)


def run(cfg: dict) -> Path:
    started = time.time()
    out = prepare_output(cfg, "finite_window_exp3_pairwise_phase_ensemble")
    model = load_model(ROOT / cfg["model_dir"], AREAS9)
    specs = _window_specs(cfg)
    ecfg = cfg["exp3_nonlinear"]
    pcfg = cfg.get("phase_seed_validation", {})
    pair_cfg = cfg.get("pairwise_window_nonlinear", {})
    n_trials = int(pcfg.get("n_trials", 64))
    n_masks = int(pair_cfg.get("exp3_n_masks", 40))
    offsets = list(map(int, pcfg.get("phase_seed_offsets", [220000, 520000, 520001, 520002, 520003, 520004])))
    dt = float(ecfg["dt_ms"])
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    long_steps = int(round(specs["long"]["analysis_s"] * 1000.0 / dt))
    z, u = _input_trials(n_trials, int(cfg["seed"]) + 300)
    source_trials = pd.read_csv(ROOT / cfg["output_dir"] / "exp3_structural_kernel" / "trials.csv.gz")
    selected = source_trials[(source_trials.kind == "random15_delete") & (source_trials.split == "heldout")].head(n_masks)
    masks = []
    for source_row in selected.itertuples(index=False):
        delta_w = np.zeros_like(model.fln)
        for source, target, weight in json.loads(source_row.edges_json):
            delta_w[model.areas.index(target), model.areas.index(source)] = -float(weight)
        masks.append((source_row.perturbation_id, delta_w, apply_fln_delta(model, delta_w) - model.J))

    rows = []
    qc = []
    freqs = frequency_grid()
    output_indices = np.arange(len(AREAS9), dtype=np.int64)
    for input_pos, input_area in enumerate(AREAS9):
        psd_sums: dict[str, np.ndarray] = {}
        psd_freqs: dict[str, np.ndarray] = {}
        networks = [("baseline", model.params, None)]
        for perturbation_id, delta_w, delta_j in masks:
            params = copy.copy(model.params)
            params["fln_mat"] = model.fln + delta_w
            networks.append((perturbation_id, params, delta_j))
        for repeat, offset in enumerate(offsets):
            inputs = _fourier_inputs(
                z, dt, long_steps, burn_steps, float(ecfg["input_sigma"]), int(cfg["seed"]) + offset,
            )
            for network_pos, (network_id, params, _) in enumerate(networks):
                outputs, rect, min_e, min_i, min_de, min_di = _run_multioutput_batch(
                    params, inputs, dt, burn_steps, output_indices, input_idx=input_pos,
                )
                for window, spec in specs.items():
                    n_steps = int(round(spec["analysis_s"] * 1000.0 / dt))
                    for output_pos in range(len(AREAS9)):
                        _, spectra = _timeseries_features(
                            outputs[:, output_pos, :n_steps], dt, spec["spectral"],
                        )
                        spectrum_values = np.stack([spectrum[1] for spectrum in spectra]).astype(np.float32)
                        if window not in psd_sums:
                            psd_sums[window] = np.zeros(
                                (len(networks), len(AREAS9), n_trials, spectrum_values.shape[1]),
                                dtype=np.float32,
                            )
                            psd_freqs[window] = spectra[0][0]
                        psd_sums[window][network_pos, output_pos] += spectrum_values
                qc.append({
                    "input_area": input_area, "phase_repeat": repeat,
                    "phase_seed_offset": offset, "network": network_id,
                    "rectification_fraction": float(rect.sum() / (2 * len(model.areas) * inputs.shape[1] * n_trials)),
                    "min_rate_E": float(min_e.min()), "min_rate_I": float(min_i.min()),
                    "min_drive_E": float(min_de.min()), "min_drive_I": float(min_di.min()),
                })
                if network_pos % 10 == 0 or network_pos == n_masks:
                    print(
                        f"phase ensemble input={input_area} repeat={repeat + 1}/{len(offsets)} "
                        f"network={network_pos}/{n_masks}", flush=True,
                    )

        for window in WINDOWS:
            mean_psd = psd_sums[window] / len(offsets)
            values = np.empty((len(networks), len(AREAS9), n_trials, 2), dtype=float)
            valid = np.empty((len(networks), len(AREAS9), n_trials), dtype=bool)
            for network_pos in range(len(networks)):
                for output_pos in range(len(AREAS9)):
                    for trial in range(n_trials):
                        feature = feature_from_spectrum(
                            psd_freqs[window], mean_psd[network_pos, output_pos, trial],
                        )
                        values[network_pos, output_pos, trial] = feature["peak_omega"], feature["exponent"]
                        valid[network_pos, output_pos, trial] = feature["peak_valid"] and np.isfinite(
                            values[network_pos, output_pos, trial]
                        ).all()
            for output_pos, output_area in enumerate(AREAS9):
                use = np.all(valid[:, output_pos, :], axis=0)
                if use.sum() < 10:
                    continue
                sigma = np.cov(u[use], rowvar=False, ddof=1)
                baseline_r = float(np.corrcoef(values[0, output_pos, use, 0], values[0, output_pos, use, 1])[0, 1])
                for mask_pos, (perturbation_id, _, delta_j) in enumerate(masks, 1):
                    perturbed_r = float(np.corrcoef(
                        values[mask_pos, output_pos, use, 0], values[mask_pos, output_pos, use, 1]
                    )[0, 1])
                    first, second = _delta_theory(model, sigma, freqs, input_pos, output_pos, delta_j)
                    rows.append({
                        "input_area": input_area, "output_area": output_area,
                        "window": window, "perturbation_id": perturbation_id,
                        "n_phase_repeats": len(offsets), "n_input_trials": n_trials,
                        "n_valid_common": int(use.sum()),
                        "baseline_actual_r": baseline_r, "perturbed_actual_r": perturbed_r,
                        "delta_rho_actual": perturbed_r - baseline_r,
                        "delta_rho_first": first, "delta_rho_second": second,
                    })
        del psd_sums

    results = pd.DataFrame(rows)
    stacked = pd.concat([
        results.assign(order=order, predicted=results[f"delta_rho_{order}"])
        for order in ("first", "second")
    ], ignore_index=True)
    metrics = _pair_metrics(stacked, "delta_rho_actual", "predicted")
    results.to_csv(out / "delta_correlation_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "delta_correlation_R2_by_pair.csv", index=False)
    pd.DataFrame(qc).to_csv(out / "rectification_qc.csv", index=False)
    save_json(out / "manifest.json", {
        "experiment": "nine_area_9x9_phase_ensemble_threshold_nonlinear",
        "actual_simulation": "mean PSD across independent threshold-linear time-domain phase realizations before feature extraction",
        "areas": AREAS9, "n_input_trials": n_trials, "n_masks": n_masks,
        "phase_seed_offsets": offsets, "n_phase_repeats": len(offsets),
        "validity_rule": "per input-output-window intersection across baseline and all masks after phase-mean PSD",
        "short_analysis_s": specs["short"]["analysis_s"],
        "long_analysis_s": specs["long"]["analysis_s"],
        "runtime_s": time.time() - started,
    })
    _plot(out, results, metrics)
    return out


def _plot(out: Path, results: pd.DataFrame, metrics: pd.DataFrame) -> None:
    chosen = results[(results.input_area == "V1") & (results.output_area == "PG")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, window, order in zip(
        axes.flat, ("short", "short", "long", "long"), ("first", "second", "first", "second")
    ):
        data = chosen[chosen.window == window]
        _annotated_scatter(
            ax, data[f"delta_rho_{order}"].to_numpy(), data.delta_rho_actual.to_numpy(),
            f"{window.capitalize()} window, {order} order", "#4C78A8" if order == "first" else "#E45756",
        )
        ax.set(xlabel="Predicted Delta correlation", ylabel="Phase-mean-PSD actual Delta correlation")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
    fig.suptitle("Experiment 3 phase ensemble, V1 input -> PG output", fontsize=13)
    save_figure(fig, out / "figures" / "exp3_V1_PG_phase_ensemble_orders")
    plt.close(fig)
    _heatmap_grid(
        metrics, "ols_R2", "Experiment 3 phase ensemble: OLS R2 across 9 inputs x 9 outputs",
        out / "figures" / "exp3_phase_ensemble_9x9_OLS_R2",
    )
    _heatmap_grid(
        metrics, "R2", "Experiment 3 phase ensemble: identity R2 across 9 inputs x 9 outputs",
        out / "figures" / "exp3_phase_ensemble_9x9_identity_R2",
    )
    exp1_metrics = pd.read_csv(
        ROOT / "experiments/pap_theory_validation/outputs/finite_window_exp1_pairwise_nonlinear/correlation_R2_by_pair.csv"
    )
    _plot_identity_comparison(out, exp1_metrics, metrics)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
