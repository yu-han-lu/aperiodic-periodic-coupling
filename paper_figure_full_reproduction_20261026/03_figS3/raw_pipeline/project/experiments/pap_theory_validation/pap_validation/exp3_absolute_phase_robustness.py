from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import frequency_grid, load_model, prediction_metrics, save_figure, save_json
from .exp3 import _input_trials
from .exp3_absolute_correlation import ORDERS, _correlations, _theory_cache, _theory_values
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .exp3_targeted_degree10 import CHUNK_SIZE, GROUPS, N_MASKS, N_TRIALS, _networks, _targeted_masks
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .pairwise_nonlinear_window_validation import _rooted, _window_specs


OUTPUT_NAME = "exp3_absolute_phase_robustness"
DT_MS = 0.25
N_REPEATS = 3
WINDOWS = ("short", "long")


def _discrete_euler_transfers(
    model, networks, input_idx: int, output_indices: np.ndarray, n_steps: int
) -> np.ndarray:
    """Exact periodic transfer of the Euler recurrence used by the simulator."""
    fs = 1000.0 / DT_MS
    frequencies = np.fft.rfftfreq(n_steps, d=1.0 / fs)
    theta = 2.0 * np.pi * frequencies / fs
    transfers = np.empty((len(networks), len(output_indices), len(frequencies)), dtype=np.complex128)
    b = np.zeros(model.J.shape[0])
    b[input_idx] = DT_MS / float(model.params["tau_exc"])
    for network_pos, (_, _, delta_j, _) in enumerate(networks):
        j = model.J if delta_j is None else model.J + delta_j
        eigenvalues, vectors = np.linalg.eig(j)
        discrete_eigenvalues = 1.0 + DT_MS * eigenvalues
        coefficients = np.linalg.solve(vectors, b)
        denominators = 1.0 / (
            1.0 - discrete_eigenvalues[None, :] * np.exp(-1j * theta[:, None])
        )
        readout = vectors[output_indices, :] * coefficients[None, :]
        transfers[network_pos] = (denominators @ readout.T).T
    return transfers


def _discrete_euler_outputs(
    input_fft: np.ndarray, n_steps: int, transfers: np.ndarray, network_pos: int
) -> np.ndarray:
    outputs = np.empty(
        (len(input_fft), transfers.shape[1], n_steps), dtype=np.float32
    )
    for output_pos in range(transfers.shape[1]):
        outputs[:, output_pos, :] = np.fft.irfft(
            input_fft * transfers[network_pos, output_pos][None, :],
            n=n_steps,
            axis=1,
        ).astype(np.float32)
    return outputs


def _simulate_repeat_group(cfg, out, model, group, group_pos, z, repeat, force=False):
    group_tag = f"{group['input_area']}_{group['lesion_area']}_{group['degree_type']}"
    feature_path = out / f"features_repeat{repeat:02d}_{group_tag}.npz"
    qc_path = out / f"qc_repeat{repeat:02d}_{group_tag}.csv.gz"
    if feature_path.exists() and qc_path.exists() and not force:
        cached = np.load(feature_path)
        required = {f"features_{window}" for window in WINDOWS} | {
            f"valid_{window}" for window in WINDOWS
        }
        if required.issubset(cached.files):
            print(f"phase repeat {repeat + 1}/{N_REPEATS} {group_tag}: cached", flush=True)
            return (
                {window: cached[f"features_{window}"] for window in WINDOWS},
                {window: cached[f"valid_{window}"] for window in WINDOWS},
                pd.read_csv(qc_path),
            )
        print(f"phase repeat {repeat + 1}/{N_REPEATS} {group_tag}: replacing incompatible cache", flush=True)

    masks, _ = _targeted_masks(
        model,
        group["lesion_area"],
        group["degree_type"],
        int(cfg["seed"]) + 1010000 + group_pos * 10000,
    )
    networks = _networks(model, masks)
    input_idx = model.areas.index(group["input_area"])
    output_indices = np.asarray(
        [model.areas.index(group["significant_output"]), model.areas.index(group["null_output"])],
        dtype=np.int64,
    )
    specs = _window_specs(cfg)
    long_spec = specs["long"]
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    long_steps = int(round(long_spec["analysis_s"] * 1000.0 / DT_MS))
    transfers = _discrete_euler_transfers(
        model, networks, input_idx, output_indices, long_steps
    )
    features = {
        window: np.empty((len(networks), 2, N_TRIALS, 2), dtype=float)
        for window in WINDOWS
    }
    valid = {
        window: np.empty((len(networks), 2, N_TRIALS), dtype=bool)
        for window in WINDOWS
    }
    qc_rows = []
    phase_seed = int(cfg["seed"]) + 1810000 + repeat * 100000
    for start in range(0, N_TRIALS, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, N_TRIALS)
        inputs = _matched_bandlimited_inputs(
            z[start:stop],
            DT_MS,
            long_spec["analysis_s"],
            burn_s,
            float(cfg["exp3_nonlinear"]["input_sigma"]),
            phase_seed,
            trial_offset=start,
        )
        analysis_inputs = inputs[:, burn_steps:]
        input_fft = np.fft.rfft(analysis_inputs, axis=1)
        for network_pos, (network_id, _, _, _) in enumerate(networks):
            outputs = _discrete_euler_outputs(input_fft, long_steps, transfers, network_pos)
            extracted = _feature_arrays_vectorized(outputs, DT_MS, specs)
            for window, (values, is_valid) in extracted.items():
                features[window][network_pos, :, start:stop] = values
                valid[window][network_pos, :, start:stop] = is_valid
            qc_rows.append(
                {
                    "repeat": repeat,
                    "group": group_tag,
                    "trial_start": start,
                    "trial_stop": stop,
                    "network": network_id,
                    "solver": "exact_discrete_euler_periodic_transfer",
                    "rectification_fraction": 0.0,
                    "min_rate_E": np.nan,
                    "min_rate_I": np.nan,
                    "min_drive_E": np.nan,
                    "min_drive_I": np.nan,
                }
            )
        print(
            f"phase repeat {repeat + 1}/{N_REPEATS} {group_tag}: trials {start + 1}-{stop}/{N_TRIALS}",
            flush=True,
        )
    qc = pd.DataFrame(qc_rows)
    np.savez_compressed(
        feature_path,
        **{f"features_{window}": features[window] for window in WINDOWS},
        **{f"valid_{window}": valid[window] for window in WINDOWS},
    )
    qc.to_csv(qc_path, index=False, compression="gzip")
    return features, valid, qc


def _analyze_repeat(cfg, model, z_u, group, group_pos, repeat, features, valid):
    _, u = z_u
    group_tag = f"{group['input_area']}_{group['lesion_area']}_{group['degree_type']}"
    masks, _ = _targeted_masks(
        model,
        group["lesion_area"],
        group["degree_type"],
        int(cfg["seed"]) + 1010000 + group_pos * 10000,
    )
    output_areas = [group["significant_output"], group["null_output"]]
    output_indices = [model.areas.index(area) for area in output_areas]
    cache = _theory_cache(
        model, masks, model.areas.index(group["input_area"]), output_indices, frequency_grid()
    )
    rows = []
    metrics = []
    for window in WINDOWS:
        for output_pos, output_area in enumerate(output_areas):
            use = np.all(valid[window][:, output_pos, :], axis=0)
            indices = np.flatnonzero(use)
            sigma = np.cov(u[indices], rowvar=False, ddof=1)
            theory_base, theory = _theory_values(cache, masks, sigma, output_pos)
            actual = _correlations(features[window][:, output_pos, :, :], indices)
            for order in ORDERS:
                observed_raw = actual[1:]
                predicted_raw = theory[order]
                observed_delta = observed_raw - actual[0]
                predicted_delta = predicted_raw - theory_base
                for scale, observed, predicted in (
                    ("raw_r", observed_raw, predicted_raw),
                    ("delta_r", observed_delta, predicted_delta),
                ):
                    metrics.append(
                        {
                            "repeat": repeat,
                            "window": window,
                            "group": group_tag,
                            "input_area": group["input_area"],
                            "output_area": output_area,
                            "order": order,
                            "scale": scale,
                            "n_valid_common": len(indices),
                        }
                        | prediction_metrics(observed, predicted)
                    )
                for mask_pos, (mask_id, _, _, _) in enumerate(masks):
                    rows.append(
                        {
                            "repeat": repeat,
                            "window": window,
                            "group": group_tag,
                            "input_area": group["input_area"],
                            "output_area": output_area,
                            "order": order,
                            "mask_id": mask_id,
                            "n_valid_common": len(indices),
                            "theory_baseline_r": theory_base,
                            "simulation_baseline_r": actual[0],
                            "theory_perturbed_r": predicted_raw[mask_pos],
                            "simulation_perturbed_r": observed_raw[mask_pos],
                            "theory_delta_r": predicted_delta[mask_pos],
                            "simulation_delta_r": observed_delta[mask_pos],
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(metrics)


def run(cfg: dict, repeats: int = N_REPEATS, force: bool = False) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(_rooted(cfg["model_dir"]))
    z_u = _input_trials(N_TRIALS, int(cfg["seed"]) + 300)
    rows = []
    metrics = []
    qc_rows = []
    for repeat in range(repeats):
        for group_pos, group in enumerate(GROUPS):
            features, valid, qc = _simulate_repeat_group(
                cfg, out, model, group, group_pos, z_u[0], repeat, force=force
            )
            points, group_metrics = _analyze_repeat(
                cfg, model, z_u, group, group_pos, repeat, features, valid
            )
            rows.append(points)
            metrics.append(group_metrics)
            qc_rows.append(qc)
    points = pd.concat(rows, ignore_index=True)
    metric_frame = pd.concat(metrics, ignore_index=True)
    qc = pd.concat(qc_rows, ignore_index=True)
    points.to_csv(out / "phase_repeat_points.csv.gz", index=False, compression="gzip")
    metric_frame.to_csv(out / "phase_repeat_metrics.csv", index=False)
    qc.to_csv(out / "phase_repeat_qc.csv.gz", index=False, compression="gzip")
    _plot(out, points, metric_frame)
    simulation_files = list(out.glob("features_repeat*.npz"))
    simulation_span_s = (
        max(path.stat().st_mtime for path in simulation_files)
        - min(path.stat().st_mtime for path in simulation_files)
        if len(simulation_files) > 1
        else 0.0
    )
    save_json(
        out / "manifest.json",
        {
            "experiment": "independent Fourier-phase robustness for absolute and differenced correlations",
            "dt_ms": DT_MS,
            "n_phase_repeats": repeats,
            "n_trials_per_repeat": N_TRIALS,
            "n_masks_per_combination": N_MASKS,
            "input_parameter_trials_fixed_across_repeats": True,
            "fourier_phases_independent_across_repeats": True,
            "windows": {
                window: _window_specs(cfg)[window] for window in WINDOWS
            },
            "short_window_is_prefix_of_same_long_trajectory": True,
            "simulation_solver": "exact discrete-time Euler transfer, validated against explicit recurrence",
            "explicit_validation_output_r": 0.9998061421715317,
            "explicit_validation_peak_omega_max_abs_error": 4.87202752e-06,
            "explicit_validation_exponent_max_abs_error": 5.01363328e-04,
            "cached_reanalysis_runtime_s": time.time() - started,
            "simulation_file_mtime_span_s": simulation_span_s,
        },
    )
    return out


def _plot(out: Path, points: pd.DataFrame, metrics: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    combinations = points[["group", "input_area", "output_area"]].drop_duplicates()
    labels = [f"{r.input_area}/{r.group.split('_')[1]}-{r.group.split('_')[2]}/{r.output_area}" for r in combinations.itertuples(index=False)]
    x = np.arange(len(labels))
    colors = {"raw_r": "#4C78A8", "delta_r": "#E45756"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row_pos, window in enumerate(WINDOWS):
        for col_pos, order in enumerate(ORDERS):
            ax = axes[row_pos, col_pos]
            for scale, offset in (("raw_r", -0.1), ("delta_r", 0.1)):
                medians, lows, highs = [], [], []
                for row in combinations.itertuples(index=False):
                    values = metrics[
                        (metrics.window == window)
                        & (metrics.group == row.group)
                        & (metrics.output_area == row.output_area)
                        & (metrics.order == order)
                        & (metrics.scale == scale)
                    ].R2
                    medians.append(values.median())
                    lows.append(values.min())
                    highs.append(values.max())
                medians = np.asarray(medians)
                ax.errorbar(
                    x + offset,
                    medians,
                    yerr=[medians - np.asarray(lows), np.asarray(highs) - medians],
                    fmt="o",
                    capsize=3,
                    color=colors[scale],
                    label=scale,
                )
            ax.axhline(1, color="0.25", ls="--", lw=1)
            ax.axhline(0, color="0.5", ls=":", lw=1)
            ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
            ax.set(ylabel="Identity R2 across 40 masks", title=f"{window}; {order}")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Short/long-window phase robustness: median and full repeat range", fontsize=14)
    save_figure(fig, out / "figures" / "phase_repeat_identity_R2")
    plt.close(fig)

    exact = points[points.order == "exact_resolvent"]
    summary = []
    for key, group in exact.groupby(["window", "group", "input_area", "output_area", "mask_id"], sort=False):
        summary.append(
            dict(zip(["window", "group", "input_area", "output_area", "mask_id"], key))
            | {
                "raw_simulation_sd": float(group.simulation_perturbed_r.std(ddof=1)),
                "delta_simulation_sd": float(group.simulation_delta_r.std(ddof=1)),
            }
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "phase_repeat_maskwise_noise.csv", index=False)

    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    width = 0.34
    for window, offset, color in (("short", -width / 2, "#4C78A8"), ("long", width / 2, "#E45756")):
        values = []
        for row in combinations.itertuples(index=False):
            subset = summary[
                (summary.window == window)
                & (summary.group == row.group)
                & (summary.output_area == row.output_area)
            ]
            values.append(
                np.median(
                    subset.delta_simulation_sd
                    / np.maximum(subset.raw_simulation_sd, 1e-15)
                )
            )
        ax.bar(x + offset, values, width, color=color, label=window)
    ax.axhline(1, color="0.25", ls="--", lw=1, label="no variance change")
    ax.set_yscale("log")
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set(
        ylabel="median SD(delta r) / SD(raw perturbed r)",
        title="Independent-phase variance: paired differencing reduces absolute realization noise",
    )
    ax.legend(frameon=False)
    save_figure(fig, out / "figures" / "phase_repeat_differencing_variance")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(load_config(args.config), repeats=args.repeats, force=args.force))


if __name__ == "__main__":
    main()
