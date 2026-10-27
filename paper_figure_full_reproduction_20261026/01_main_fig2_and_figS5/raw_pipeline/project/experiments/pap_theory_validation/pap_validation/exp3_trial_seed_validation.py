from __future__ import annotations

import argparse
import copy
import itertools
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal

from .core import (
    AREAS9,
    apply_fln_delta,
    feature_from_spectrum,
    frequency_grid,
    load_model,
    save_figure,
    save_json,
)
from .exp3 import _input_trials
from .io import load_config, prepare_output
from .nonlinear_network_validation import _fourier_inputs, _run_multioutput_batch
from .pairwise_nonlinear_window_validation import (
    _annotated_scatter,
    _delta_theory,
    _fit_metrics,
    _heatmap_grid,
    _pair_metrics,
    _rooted,
    _window_specs,
)


OUTPUT_NAME = "exp3_trial_seed_robustness"
TRIAL_COUNTS = (32, 64, 128, 256, 512)
N_REPEATS = 6
N_MASKS = 40
CHUNK_SIZE = 64


def _feature_arrays_vectorized(
    outputs: np.ndarray, dt_ms: float, specs: dict[str, dict]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Apply the established Welch pipeline in batches over trial and output axes."""
    fs = 1000.0 / dt_ms
    result = {}
    for window, spec in specs.items():
        n_steps = int(round(spec["analysis_s"] * fs))
        spectral = spec["spectral"]
        nperseg = min(n_steps, int(round(float(spectral["segment_s"]) * fs)))
        freq, psd = signal.welch(
            outputs[:, :, :n_steps],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
            axis=-1,
        )
        keep = (freq >= 0.5) & (freq <= 1200.0)
        kept_freq = freq[keep]
        sigma_bins = float(spectral["smoothing_sigma_hz"]) / float(
            kept_freq[1] - kept_freq[0]
        )
        smooth = np.exp(
            ndimage.gaussian_filter1d(
                np.log(np.maximum(psd[:, :, keep], 1e-300)), sigma_bins, axis=-1
            )
        )
        values = np.empty((outputs.shape[1], outputs.shape[0], 2), dtype=float)
        valid = np.empty((outputs.shape[1], outputs.shape[0]), dtype=bool)
        for trial in range(outputs.shape[0]):
            for output_pos in range(outputs.shape[1]):
                feature = feature_from_spectrum(kept_freq, smooth[trial, output_pos])
                values[output_pos, trial] = feature["peak_omega"], feature["exponent"]
                valid[output_pos, trial] = bool(feature["peak_valid"]) and np.isfinite(
                    values[output_pos, trial]
                ).all()
        result[window] = values, valid
    return result


def _load_masks(cfg: dict, model, n_masks: int):
    source = pd.read_csv(
        _rooted(cfg["output_dir"]) / "exp3_structural_kernel" / "trials.csv.gz"
    )
    selected = source[
        (source.kind == "random15_delete") & (source.split == "heldout")
    ].head(n_masks)
    masks = []
    for row in selected.itertuples(index=False):
        delta_w = np.zeros_like(model.fln)
        for source_area, target_area, weight in json.loads(row.edges_json):
            delta_w[
                model.areas.index(target_area), model.areas.index(source_area)
            ] = -float(weight)
        delta_j = apply_fln_delta(model, delta_w) - model.J
        masks.append((row.perturbation_id, delta_w, delta_j))
    return masks


def _networks(model, masks):
    result = [("baseline", model.params, None)]
    for perturbation_id, delta_w, delta_j in masks:
        params = copy.copy(model.params)
        params["fln_mat"] = model.fln + delta_w
        result.append((perturbation_id, params, delta_j))
    return result


def _simulate_feature_pool(
    model,
    networks,
    z: np.ndarray,
    input_pos: int,
    output_indices: np.ndarray,
    dt: float,
    burn_steps: int,
    long_steps: int,
    input_sigma: float,
    phase_seed: int,
    specs: dict,
    chunk_size: int,
    progress_label: str,
):
    n_trials = len(z)
    n_networks = len(networks)
    n_outputs = len(output_indices)
    features = {
        window: np.empty((n_networks, n_outputs, n_trials, 2), dtype=float)
        for window in specs
    }
    valid = {
        window: np.empty((n_networks, n_outputs, n_trials), dtype=bool)
        for window in specs
    }
    qc_rows = []

    for start in range(0, n_trials, chunk_size):
        stop = min(start + chunk_size, n_trials)
        block_seed = phase_seed + 100003 * (start // chunk_size)
        inputs = _fourier_inputs(
            z[start:stop], dt, long_steps, burn_steps, input_sigma, block_seed
        )
        for network_pos, (network_id, params, _) in enumerate(networks):
            outputs, rect, min_e, min_i, min_de, min_di = _run_multioutput_batch(
                params,
                inputs,
                dt,
                burn_steps,
                output_indices,
                input_idx=input_pos,
            )
            extracted = _feature_arrays_vectorized(outputs, dt, specs)
            for window, (values, is_valid) in extracted.items():
                features[window][network_pos, :, start:stop] = values
                valid[window][network_pos, :, start:stop] = is_valid
            qc_rows.append(
                {
                    "label": progress_label,
                    "trial_start": start,
                    "trial_stop": stop,
                    "network": network_id,
                    "rectification_fraction": float(
                        rect.sum()
                        / (2 * len(model.areas) * inputs.shape[1] * (stop - start))
                    ),
                    "min_rate_E": float(min_e.min()),
                    "min_rate_I": float(min_i.min()),
                    "min_drive_E": float(min_de.min()),
                    "min_drive_I": float(min_di.min()),
                }
            )
        print(
            f"{progress_label}: trials {start + 1}-{stop}/{n_trials}", flush=True
        )
    return features, valid, pd.DataFrame(qc_rows)


def _correlation_rows(
    model,
    masks,
    features,
    valid,
    u: np.ndarray,
    input_pos: int,
    input_area: str,
    output_indices: np.ndarray,
    trial_counts,
    repeat: int | str,
    parameter_seed: int,
    phase_seed: int,
):
    freqs = frequency_grid()
    rows = []
    for output_local, output_pos in enumerate(output_indices):
        output_area = AREAS9[int(output_pos)]
        for window in ("short", "long"):
            for n_trials in trial_counts:
                common = np.all(valid[window][:, output_local, :n_trials], axis=0)
                if common.sum() < 8:
                    continue
                sigma = np.cov(u[:n_trials][common], rowvar=False, ddof=1)
                baseline_values = features[window][0, output_local, :n_trials][common]
                baseline_r = float(np.corrcoef(baseline_values.T)[0, 1])
                for mask_pos, (perturbation_id, _, delta_j) in enumerate(masks, 1):
                    perturbed_values = features[window][mask_pos, output_local, :n_trials][common]
                    perturbed_r = float(np.corrcoef(perturbed_values.T)[0, 1])
                    first, second = _delta_theory(
                        model, sigma, freqs, input_pos, int(output_pos), delta_j
                    )
                    rows.append(
                        {
                            "repeat": repeat,
                            "parameter_seed": parameter_seed,
                            "phase_seed": phase_seed,
                            "n_input_trials": n_trials,
                            "n_valid_common": int(common.sum()),
                            "input_area": input_area,
                            "output_area": output_area,
                            "window": window,
                            "perturbation_id": perturbation_id,
                            "baseline_actual_r": baseline_r,
                            "perturbed_actual_r": perturbed_r,
                            "delta_rho_actual": perturbed_r - baseline_r,
                            "delta_rho_first": first,
                            "delta_rho_second": second,
                        }
                    )
    return pd.DataFrame(rows)


def _diagnostic_metrics(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for key, group in rows.groupby(
        ["repeat", "n_input_trials", "window"], sort=False
    ):
        repeat, n_trials, window = key
        for order in ("first", "second"):
            records.append(
                {
                    "repeat": repeat,
                    "n_input_trials": n_trials,
                    "window": window,
                    "order": order,
                    "n_valid_common": int(group.n_valid_common.iloc[0]),
                    **_fit_metrics(
                        group.delta_rho_actual.to_numpy(),
                        group[f"delta_rho_{order}"].to_numpy(),
                    ),
                }
            )
    return pd.DataFrame(records)


def _seed_stability(rows: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (n_trials, window), group in rows.groupby(
        ["n_input_trials", "window"], sort=False
    ):
        matrix = group.pivot(
            index="repeat", columns="perturbation_id", values="delta_rho_actual"
        )
        pairwise = []
        for left, right in itertools.combinations(matrix.index, 2):
            x, y = matrix.loc[left].to_numpy(float), matrix.loc[right].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 4 and np.std(x[ok]) > 0 and np.std(y[ok]) > 0:
                pairwise.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
        within_sd = matrix.std(axis=0, ddof=1)
        repeat_mean = matrix.mean(axis=0)
        signal_sd = float(repeat_mean.std(ddof=1))
        noise_sd = float(within_sd.median())
        records.append(
            {
                "n_input_trials": n_trials,
                "window": window,
                "n_repeats": int(matrix.shape[0]),
                "median_seed_sd_delta_rho": noise_sd,
                "between_mask_sd_of_repeat_mean": signal_sd,
                "noise_to_signal_ratio": noise_sd / max(signal_sd, 1e-15),
                "mean_pairwise_seed_r": float(np.mean(pairwise)),
                "median_pairwise_seed_r": float(np.median(pairwise)),
                "min_pairwise_seed_r": float(np.min(pairwise)),
                "max_pairwise_seed_r": float(np.max(pairwise)),
            }
        )
    return pd.DataFrame(records)


def run_seed_diagnostic(cfg: dict, out: Path, model, masks, force: bool = False):
    specs = _window_specs(cfg)
    ecfg = cfg["exp3_nonlinear"]
    dt = float(ecfg["dt_ms"])
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    long_steps = int(round(specs["long"]["analysis_s"] * 1000.0 / dt))
    networks = _networks(model, masks)
    repeat_dir = out / "seed_repeats"
    repeat_dir.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_qc = []

    for repeat in range(N_REPEATS):
        result_path = repeat_dir / f"repeat_{repeat:02d}.csv.gz"
        qc_path = repeat_dir / f"repeat_{repeat:02d}_qc.csv.gz"
        if result_path.exists() and not force:
            all_rows.append(pd.read_csv(result_path))
            if qc_path.exists():
                all_qc.append(pd.read_csv(qc_path))
            print(f"seed diagnostic repeat {repeat}: cached", flush=True)
            continue
        parameter_seed = int(cfg["seed"]) + 730000 + repeat
        phase_seed = int(cfg["seed"]) + 830000 + repeat * 10000
        z, u = _input_trials(max(TRIAL_COUNTS), parameter_seed)
        features, valid, qc = _simulate_feature_pool(
            model,
            networks,
            z,
            input_pos=0,
            output_indices=np.array([5], dtype=np.int64),
            dt=dt,
            burn_steps=burn_steps,
            long_steps=long_steps,
            input_sigma=float(ecfg["input_sigma"]),
            phase_seed=phase_seed,
            specs=specs,
            chunk_size=CHUNK_SIZE,
            progress_label=f"seed repeat {repeat + 1}/{N_REPEATS}",
        )
        rows = _correlation_rows(
            model,
            masks,
            features,
            valid,
            u,
            input_pos=0,
            input_area="V1",
            output_indices=np.array([5], dtype=np.int64),
            trial_counts=TRIAL_COUNTS,
            repeat=repeat,
            parameter_seed=parameter_seed,
            phase_seed=phase_seed,
        )
        rows.to_csv(result_path, index=False, compression="gzip")
        qc.to_csv(qc_path, index=False, compression="gzip")
        all_rows.append(rows)
        all_qc.append(qc)
    rows = pd.concat(all_rows, ignore_index=True)
    metrics = _diagnostic_metrics(rows)
    stability = _seed_stability(rows)
    rows.to_csv(out / "seed_diagnostic_delta_correlations.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "seed_diagnostic_metrics.csv", index=False)
    stability.to_csv(out / "seed_stability_by_trial_count.csv", index=False)
    if all_qc:
        pd.concat(all_qc, ignore_index=True).to_csv(
            out / "seed_diagnostic_qc.csv.gz", index=False, compression="gzip"
        )
    return rows, metrics, stability


def run_robust_9x9(cfg: dict, out: Path, model, masks, force: bool = False):
    specs = _window_specs(cfg)
    ecfg = cfg["exp3_nonlinear"]
    dt = float(ecfg["dt_ms"])
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    long_steps = int(round(specs["long"]["analysis_s"] * 1000.0 / dt))
    networks = _networks(model, masks)
    input_dir = out / "robust_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    parameter_seed = int(cfg["seed"]) + 300
    phase_seed = int(cfg["seed"]) + 520000
    z, u = _input_trials(max(TRIAL_COUNTS), parameter_seed)
    all_rows = []
    all_qc = []

    for input_pos, input_area in enumerate(AREAS9):
        result_path = input_dir / f"input_{input_area}.csv.gz"
        qc_path = input_dir / f"input_{input_area}_qc.csv.gz"
        if result_path.exists() and not force:
            all_rows.append(pd.read_csv(result_path))
            if qc_path.exists():
                all_qc.append(pd.read_csv(qc_path))
            print(f"robust 9x9 input {input_area}: cached", flush=True)
            continue
        features, valid, qc = _simulate_feature_pool(
            model,
            networks,
            z,
            input_pos=input_pos,
            output_indices=np.arange(len(AREAS9), dtype=np.int64),
            dt=dt,
            burn_steps=burn_steps,
            long_steps=long_steps,
            input_sigma=float(ecfg["input_sigma"]),
            phase_seed=phase_seed,
            specs=specs,
            chunk_size=CHUNK_SIZE,
            progress_label=f"robust 9x9 input {input_area}",
        )
        rows = _correlation_rows(
            model,
            masks,
            features,
            valid,
            u,
            input_pos=input_pos,
            input_area=input_area,
            output_indices=np.arange(len(AREAS9), dtype=np.int64),
            trial_counts=TRIAL_COUNTS,
            repeat="robust_pool",
            parameter_seed=parameter_seed,
            phase_seed=phase_seed,
        )
        rows.to_csv(result_path, index=False, compression="gzip")
        qc.to_csv(qc_path, index=False, compression="gzip")
        all_rows.append(rows)
        all_qc.append(qc)
    rows = pd.concat(all_rows, ignore_index=True)
    stacked = pd.concat(
        [
            rows.assign(order=order, predicted=rows[f"delta_rho_{order}"])
            for order in ("first", "second")
        ],
        ignore_index=True,
    )
    metric_frames = []
    for n_trials, group in stacked.groupby("n_input_trials", sort=True):
        frame = _pair_metrics(group, "delta_rho_actual", "predicted")
        frame.insert(0, "n_input_trials", int(n_trials))
        metric_frames.append(frame)
    metrics = pd.concat(metric_frames, ignore_index=True)
    rows.to_csv(out / "robust_9x9_delta_correlations.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "robust_9x9_metrics.csv", index=False)
    if all_qc:
        pd.concat(all_qc, ignore_index=True).to_csv(
            out / "robust_9x9_qc.csv.gz", index=False, compression="gzip"
        )
    return rows, metrics


def _mean_ci(group: pd.DataFrame, value: str):
    values = group[value].to_numpy(float)
    return float(np.mean(values)), float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def _plot_seed_convergence(out: Path, metrics: pd.DataFrame, stability: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.4), constrained_layout=True)
    colors = {"short": "#D1495B", "long": "#247BA0"}
    markers = {"first": "o", "second": "s"}
    for ax, value, title in (
        (axes[0, 0], "R2", "Identity R2 across structural masks"),
        (axes[0, 1], "ols_R2", "OLS R2 across structural masks"),
    ):
        for (window, order), group in metrics.groupby(["window", "order"]):
            summary = []
            for n_trials, n_group in group.groupby("n_input_trials"):
                mean, lo, hi = _mean_ci(n_group, value)
                summary.append((n_trials, mean, lo, hi))
            summary = np.asarray(summary, float)
            ax.plot(
                summary[:, 0], summary[:, 1], marker=markers[order], color=colors[window],
                ls="-" if order == "first" else "--", label=f"{window}, {order}",
            )
            ax.fill_between(summary[:, 0], summary[:, 2], summary[:, 3], color=colors[window], alpha=0.10)
        ax.axhline(0, color="0.35", ls=":", lw=1)
        if value == "R2":
            ax.axhline(1, color="0.2", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(TRIAL_COUNTS, [str(x) for x in TRIAL_COUNTS])
        ax.set(xlabel="Trials used per correlation", ylabel=value.replace("_", " "), title=title)
        ax.legend(frameon=False, fontsize=8, ncol=2)
    for window, group in stability.groupby("window"):
        group = group.sort_values("n_input_trials")
        axes[1, 0].plot(
            group.n_input_trials, group.median_seed_sd_delta_rho, "o-",
            color=colors[window], label=window,
        )
        axes[1, 1].plot(
            group.n_input_trials, group.mean_pairwise_seed_r, "o-",
            color=colors[window], label=window,
        )
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xticks(TRIAL_COUNTS, [str(x) for x in TRIAL_COUNTS])
    axes[1, 0].set(
        xlabel="Trials used per correlation", ylabel="Median seed SD of actual Delta r",
        title="Trial-seed noise amplitude",
    )
    axes[1, 1].axhline(0, color="0.35", ls=":", lw=1)
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_xticks(TRIAL_COUNTS, [str(x) for x in TRIAL_COUNTS])
    axes[1, 1].set_ylim(-1.05, 1.05)
    axes[1, 1].set(
        xlabel="Trials used per correlation", ylabel="Mean pairwise seed correlation",
        title="Reproducibility of mask-wise actual Delta r",
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 1].legend(frameon=False)
    fig.suptitle(
        "Experiment 3 input-seed sensitivity: V1 input to PG output (6 independent realizations)",
        fontsize=13,
    )
    save_figure(fig, out / "figures" / "exp3_trial_seed_convergence")
    plt.close(fig)


def _plot_robust_comparison(out: Path, rows: pd.DataFrame, metrics: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    final_n = max(TRIAL_COUNTS)
    final_metrics = metrics[metrics.n_input_trials == final_n]
    _heatmap_grid(
        final_metrics,
        "R2",
        f"Experiment 3 robust simulation: identity R2 across 9 x 9 paths ({final_n} trials)",
        out / "figures" / "exp3_robust_9x9_identity_R2",
    )
    _heatmap_grid(
        final_metrics,
        "ols_R2",
        f"Experiment 3 robust simulation: OLS R2 across 9 x 9 paths ({final_n} trials)",
        out / "figures" / "exp3_robust_9x9_OLS_R2",
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for ax, value, window in zip(
        axes.flat,
        ("R2", "R2", "ols_R2", "ols_R2"),
        ("short", "long", "short", "long"),
    ):
        first = metrics[(metrics.window == window) & (metrics.order == "first")]
        second = metrics[(metrics.window == window) & (metrics.order == "second")]
        for data, order, color in (
            (first, "first", "#4C78A8"), (second, "second", "#E45756")
        ):
            medians = data.groupby("n_input_trials")[value].median().reindex(TRIAL_COUNTS)
            q25 = data.groupby("n_input_trials")[value].quantile(0.25).reindex(TRIAL_COUNTS)
            q75 = data.groupby("n_input_trials")[value].quantile(0.75).reindex(TRIAL_COUNTS)
            ax.plot(TRIAL_COUNTS, medians, "o-", color=color, label=order)
            ax.fill_between(TRIAL_COUNTS, q25, q75, color=color, alpha=0.14)
        ax.axhline(0, color="0.35", ls=":", lw=1)
        if value == "R2":
            ax.axhline(1, color="0.2", ls="--", lw=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(TRIAL_COUNTS, [str(x) for x in TRIAL_COUNTS])
        ax.set(
            xlabel="Trials used per correlation",
            ylabel="Identity R2" if value == "R2" else "OLS R2",
            title=f"{window.capitalize()} window",
        )
        ax.legend(frameon=False)
    fig.suptitle("Experiment 3: 81-path theory-simulation agreement versus trial count", fontsize=13)
    save_figure(fig, out / "figures" / "exp3_robust_9x9_trial_count_comparison")
    plt.close(fig)

    chosen = rows[
        (rows.n_input_trials == final_n)
        & (rows.input_area == "V1")
        & (rows.output_area == "PG")
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for ax, window, order in zip(
        axes.flat,
        ("short", "short", "long", "long"),
        ("first", "second", "first", "second"),
    ):
        data = chosen[chosen.window == window]
        _annotated_scatter(
            ax,
            data[f"delta_rho_{order}"].to_numpy(),
            data.delta_rho_actual.to_numpy(),
            f"{window.capitalize()} window, {order} order",
            "#4C78A8" if order == "first" else "#E45756",
        )
        ax.set(xlabel="Predicted Delta correlation", ylabel="Actual nonlinear Delta correlation")
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
    fig.suptitle(f"Robust Experiment 3, V1 input to PG output ({final_n} trials)", fontsize=13)
    save_figure(fig, out / "figures" / "exp3_robust_V1_PG_predictions")
    plt.close(fig)


def run(cfg: dict, force: bool = False) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(_rooted(cfg["model_dir"]), AREAS9)
    masks = _load_masks(cfg, model, N_MASKS)
    diagnostic_rows, diagnostic_metrics, stability = run_seed_diagnostic(
        cfg, out, model, masks, force=force
    )
    _plot_seed_convergence(out, diagnostic_metrics, stability)
    robust_rows, robust_metrics = run_robust_9x9(cfg, out, model, masks, force=force)
    _plot_robust_comparison(out, robust_rows, robust_metrics)
    save_json(
        out / "manifest.json",
        {
            "experiment": "experiment3_trial_seed_and_trial_count_robustness",
            "actual_simulation": "explicit nine-area threshold-linear time-domain integration",
            "areas": AREAS9,
            "diagnostic_path": "V1 input to PG output",
            "diagnostic_independent_repeats": N_REPEATS,
            "trial_counts": list(TRIAL_COUNTS),
            "robust_trial_count": max(TRIAL_COUNTS),
            "n_masks": N_MASKS,
            "mask_selection": "first 40 pre-existing held-out random15_delete masks",
            "chunk_size": CHUNK_SIZE,
            "common_random_numbers_within_each_baseline-mask_comparison": True,
            "input_parameter_and_fourier_phase_seeds_independent_across_diagnostic_repeats": True,
            "dt_ms": cfg["exp3_nonlinear"]["dt_ms"],
            "burn_s": cfg["exp3_nonlinear"]["burn_s"],
            "short_analysis_s": _window_specs(cfg)["short"]["analysis_s"],
            "long_analysis_s": _window_specs(cfg)["long"]["analysis_s"],
            "runtime_s": time.time() - started,
        },
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(load_config(args.config), force=args.force))


if __name__ == "__main__":
    main()
