from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import AREAS9, apply_fln_delta, feature_from_spectrum, frequency_grid, load_model, prediction_metrics, save_figure, save_json
from .exp3 import _input_trials
from .io import load_config, prepare_output
from .nonlinear_network_validation import _fourier_inputs, _run_batch, _timeseries_features
from .pairwise_nonlinear_window_validation import ROOT, _delta_theory, _fit_metrics, _window_specs


def run(cfg: dict) -> Path:
    started = time.time()
    out = prepare_output(cfg, "exp3_v1pg_phase_seed_validation")
    model = load_model(ROOT / cfg["model_dir"], AREAS9)
    specs = _window_specs(cfg)
    ecfg = cfg["exp3_nonlinear"]
    pcfg = cfg.get("phase_seed_validation", {})
    n_trials = int(pcfg.get("n_trials", 64))
    n_masks = int(pcfg.get("n_masks", 40))
    offsets = list(map(int, pcfg.get("phase_seed_offsets", [220000, 520000, 520001, 520002, 520003, 520004])))
    dt = float(ecfg["dt_ms"])
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    long_steps = int(round(specs["long"]["analysis_s"] * 1000.0 / dt))
    short_steps = int(round(specs["short"]["analysis_s"] * 1000.0 / dt))
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
    psd_sums: dict[str, np.ndarray] = {}
    psd_freqs: dict[str, np.ndarray] = {}
    for repeat, offset in enumerate(offsets):
        inputs = _fourier_inputs(z, dt, long_steps, burn_steps, float(ecfg["input_sigma"]), int(cfg["seed"]) + offset)
        features = {window: [] for window in specs}
        valid = {window: [] for window in specs}
        networks = [("baseline", model.params, None)]
        for perturbation_id, delta_w, delta_j in masks:
            params = copy.copy(model.params)
            params["fln_mat"] = model.fln + delta_w
            networks.append((perturbation_id, params, delta_j))
        for network_pos, (network_id, params, _) in enumerate(networks):
            output, rect, min_e, min_i, min_de, min_di = _run_batch(
                params, inputs, dt, burn_steps, output_idx=5, input_idx=0,
            )
            for window, n_steps in (("short", short_steps), ("long", long_steps)):
                frame, spectra = _timeseries_features(output[:, :n_steps], dt, specs[window]["spectral"])
                values = frame[["peak_omega", "exponent"]].to_numpy()
                features[window].append(values)
                valid[window].append(frame.peak_valid.to_numpy(bool) & np.isfinite(values).all(axis=1))
                spectrum_values = np.stack([spectrum[1] for spectrum in spectra]).astype(np.float64)
                if window not in psd_sums:
                    psd_sums[window] = np.zeros(
                        (len(networks), n_trials, spectrum_values.shape[1]), dtype=np.float64
                    )
                    psd_freqs[window] = spectra[0][0]
                psd_sums[window][network_pos] += spectrum_values
            qc.append({
                "phase_repeat": repeat, "phase_seed_offset": offset, "network": network_id,
                "rectification_fraction": float(rect.sum() / (2 * len(model.areas) * inputs.shape[1] * n_trials)),
                "min_rate_E": float(min_e.min()), "min_rate_I": float(min_i.min()),
                "min_drive_E": float(min_de.min()), "min_drive_I": float(min_di.min()),
            })
        for window in ("short", "long"):
            values = np.asarray(features[window])
            use = np.all(np.asarray(valid[window]), axis=0)
            sigma = np.cov(u[use], rowvar=False, ddof=1)
            base_r = float(np.corrcoef(values[0, use, 0], values[0, use, 1])[0, 1])
            for mask_pos, (perturbation_id, _, delta_j) in enumerate(masks, 1):
                perturbed_r = float(np.corrcoef(values[mask_pos, use, 0], values[mask_pos, use, 1])[0, 1])
                first, second = _delta_theory(model, sigma, freqs, 0, 5, delta_j)
                rows.append({
                    "phase_repeat": repeat, "phase_seed_offset": offset, "window": window,
                    "perturbation_id": perturbation_id, "n_valid_common": int(use.sum()),
                    "delta_rho_actual": perturbed_r - base_r,
                    "delta_rho_first": first, "delta_rho_second": second,
                })
        print(f"phase diagnostic {repeat + 1}/{len(offsets)} offset={offset}", flush=True)

    results = pd.DataFrame(rows)
    metric_rows = []
    for (repeat, offset, window), group in results.groupby(["phase_repeat", "phase_seed_offset", "window"]):
        for order in ("first", "second"):
            metric_rows.append({
                "phase_repeat": repeat, "phase_seed_offset": offset, "window": window, "order": order,
                **_fit_metrics(group.delta_rho_actual.to_numpy(), group[f"delta_rho_{order}"].to_numpy()),
            })
    aggregate = results.groupby(["window", "perturbation_id"], as_index=False).agg(
        delta_rho_actual=("delta_rho_actual", "mean"),
        delta_rho_first=("delta_rho_first", "mean"),
        delta_rho_second=("delta_rho_second", "mean"),
        actual_phase_sd=("delta_rho_actual", "std"),
    )
    aggregate_metrics = []
    for window, group in aggregate.groupby("window"):
        for order in ("first", "second"):
            aggregate_metrics.append({
                "phase_repeat": "ensemble_mean", "phase_seed_offset": -1, "window": window, "order": order,
                **_fit_metrics(group.delta_rho_actual.to_numpy(), group[f"delta_rho_{order}"].to_numpy()),
            })
    psd_ensemble_rows = []
    psd_ensemble_metrics = []
    for window in ("short", "long"):
        mean_psd = psd_sums[window] / len(offsets)
        values = np.empty((len(networks), n_trials, 2), dtype=float)
        valid = np.empty((len(networks), n_trials), dtype=bool)
        for network_pos in range(len(networks)):
            for trial in range(n_trials):
                feature = feature_from_spectrum(psd_freqs[window], mean_psd[network_pos, trial])
                values[network_pos, trial] = feature["peak_omega"], feature["exponent"]
                valid[network_pos, trial] = feature["peak_valid"] and np.isfinite(values[network_pos, trial]).all()
        use = np.all(valid, axis=0)
        sigma = np.cov(u[use], rowvar=False, ddof=1)
        base_r = float(np.corrcoef(values[0, use, 0], values[0, use, 1])[0, 1])
        for mask_pos, (perturbation_id, _, delta_j) in enumerate(masks, 1):
            perturbed_r = float(np.corrcoef(values[mask_pos, use, 0], values[mask_pos, use, 1])[0, 1])
            first, second = _delta_theory(model, sigma, freqs, 0, 5, delta_j)
            psd_ensemble_rows.append({
                "window": window, "perturbation_id": perturbation_id,
                "n_valid_common": int(use.sum()), "delta_rho_actual": perturbed_r - base_r,
                "delta_rho_first": first, "delta_rho_second": second,
            })
        group = pd.DataFrame(psd_ensemble_rows)
        group = group[group.window == window]
        for order in ("first", "second"):
            psd_ensemble_metrics.append({
                "phase_repeat": "psd_ensemble", "phase_seed_offset": -2,
                "window": window, "order": order,
                **_fit_metrics(group.delta_rho_actual.to_numpy(), group[f"delta_rho_{order}"].to_numpy()),
            })
    metrics = pd.concat(
        [pd.DataFrame(metric_rows), pd.DataFrame(aggregate_metrics), pd.DataFrame(psd_ensemble_metrics)],
        ignore_index=True,
    )
    results.to_csv(out / "phase_repeat_results.csv.gz", index=False, compression="gzip")
    aggregate.to_csv(out / "ensemble_mean_results.csv", index=False)
    pd.DataFrame(psd_ensemble_rows).to_csv(out / "psd_ensemble_results.csv", index=False)
    metrics.to_csv(out / "metrics_by_phase_repeat.csv", index=False)
    pd.DataFrame(qc).to_csv(out / "rectification_qc.csv", index=False)
    save_json(out / "manifest.json", {
        "experiment": "V1_to_PG_phase_seed_sensitivity", "n_trials": n_trials, "n_masks": n_masks,
        "phase_seed_offsets": offsets, "actual_simulation": "threshold-linear time-domain integration",
        "short_analysis_s": specs["short"]["analysis_s"], "long_analysis_s": specs["long"]["analysis_s"],
        "runtime_s": time.time() - started,
    })
    _plot(out, metrics)
    return out


def _plot(out: Path, metrics: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    repeats = metrics[~metrics.phase_repeat.isin(["ensemble_mean", "psd_ensemble"])].copy()
    repeats.phase_repeat = repeats.phase_repeat.astype(int)
    x = np.arange(repeats.phase_repeat.nunique())
    for ax, order in zip(axes, ("first", "second")):
        for window, color in (("short", "#E45756"), ("long", "#4C78A8")):
            data = repeats[(repeats.order == order) & (repeats.window == window)].sort_values("phase_repeat")
            ax.plot(x, data.R2, "o-", color=color, label=window)
            ensemble = metrics[(metrics.phase_repeat == "ensemble_mean") & (metrics.order == order) & (metrics.window == window)].iloc[0]
            ax.axhline(ensemble.R2, color=color, ls="--", lw=1.3, label=f"{window} ensemble={ensemble.R2:.2f}")
            psd_ensemble = metrics[(metrics.phase_repeat == "psd_ensemble") & (metrics.order == order) & (metrics.window == window)].iloc[0]
            ax.axhline(psd_ensemble.R2, color=color, ls=":", lw=2.0, label=f"{window} mean-PSD={psd_ensemble.R2:.2f}")
        ax.axhline(0, color="0.35", ls=":")
        ax.set_xticks(x, repeats.drop_duplicates("phase_repeat").sort_values("phase_repeat").phase_seed_offset.astype(int), rotation=35)
        ax.set(xlabel="phase-seed offset", ylabel="Identity R2", title=f"{order.capitalize()} order")
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle("V1 input -> PG output: phase-realization sensitivity", fontsize=13)
    save_figure(fig, out / "figures" / "identity_R2_phase_sensitivity")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
