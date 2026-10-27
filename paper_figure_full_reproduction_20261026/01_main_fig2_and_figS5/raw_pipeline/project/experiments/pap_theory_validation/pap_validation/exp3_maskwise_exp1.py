from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import finite_derivatives, frequency_grid, prediction_metrics, save_figure, save_json
from .exp1 import Z0, ZHI, ZLO, ZSCALE, _draw_design, _feature_fun
from .exp3_absolute_phase_robustness import (
    DT_MS,
    WINDOWS,
    _discrete_euler_outputs,
    _discrete_euler_transfers,
)
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .exp3_targeted_degree10 import GROUPS, N_MASKS, _networks, _targeted_masks, _transfer_power_eig
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .nonlinear_network_validation import _exp1_conditions
from .pairwise_nonlinear_window_validation import _cov_r, _fit_metrics, _rooted, _taylor_values, _window_specs
from .core import load_model


OUTPUT_NAME = "exp3_maskwise_exp1"
N_TRIALS = 128
N_REPEATS = 3
ORDERS = ("first", "second", "exact_spectrum")


def _continuous_theory(model, networks, input_idx, output_indices, cfg):
    freqs = frequency_grid()
    result = {}
    for network_pos, (network_id, _, delta_j, _) in enumerate(networks):
        j = model.J if delta_j is None else model.J + delta_j
        transfers = _transfer_power_eig(j, input_idx, output_indices.tolist(), freqs)
        for output_pos, transfer in enumerate(transfers):
            fun = _feature_fun(
                transfer,
                freqs,
                tuple(cfg["peak_band_hz"]),
                tuple(cfg["slope_band_hz"]),
            )
            grad, hess, _ = finite_derivatives(fun, np.zeros(4), h=2e-3)
            result[(network_pos, output_pos)] = (fun, grad, hess)
    return result


def _condition_inputs(cfg, repeat, condition_pos, design, eps, rho, n_trials, long_s, burn_s):
    rng = np.random.default_rng(int(cfg["seed"]) + 1000 + condition_pos)
    u = _draw_design(design, n_trials, eps, rng, rho)
    raw = Z0 + u * ZSCALE
    z = np.clip(raw, ZLO, ZHI)
    clipped = np.any(raw != z, axis=1)
    phase_seed = int(cfg["seed"]) + 1910000 + repeat * 100000 + condition_pos * 1000
    inputs = _matched_bandlimited_inputs(
        z,
        DT_MS,
        long_s,
        burn_s,
        float(cfg["exp1_nonlinear"]["input_sigma"]),
        phase_seed,
        trial_offset=0,
    )
    return z, u, clipped, inputs


def _simulate_group_repeat(cfg, out, model, group, group_pos, repeat, n_trials, force=False):
    group_tag = f"{group['input_area']}_{group['lesion_area']}_{group['degree_type']}"
    result_path = out / f"conditions_n{n_trials}_repeat{repeat:02d}_{group_tag}.csv.gz"
    if result_path.exists() and not force:
        print(f"maskwise Experiment 1 repeat {repeat + 1}/{N_REPEATS} {group_tag}: cached", flush=True)
        return pd.read_csv(result_path)

    masks, _ = _targeted_masks(
        model,
        group["lesion_area"],
        group["degree_type"],
        int(cfg["seed"]) + 1010000 + group_pos * 10000,
    )
    networks = _networks(model, masks)
    input_idx = model.areas.index(group["input_area"])
    output_areas = [group["significant_output"], group["null_output"]]
    output_indices = np.asarray([model.areas.index(area) for area in output_areas], dtype=np.int64)
    specs = _window_specs(cfg)
    long_s = specs["long"]["analysis_s"]
    burn_s = float(cfg["exp1_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    long_steps = int(round(long_s * 1000.0 / DT_MS))
    discrete_transfers = _discrete_euler_transfers(
        model, networks, input_idx, output_indices, long_steps
    )
    theory = _continuous_theory(model, networks, input_idx, output_indices, cfg)
    rows = []

    for condition_pos, (design, eps, rho) in enumerate(_exp1_conditions(cfg)):
        z, u, clipped, inputs = _condition_inputs(
            cfg, repeat, condition_pos, design, eps, rho, n_trials, long_s, burn_s
        )
        input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=1)
        features = {
            window: np.empty((len(networks), len(output_areas), n_trials, 2), dtype=float)
            for window in WINDOWS
        }
        valid = {
            window: np.empty((len(networks), len(output_areas), n_trials), dtype=bool)
            for window in WINDOWS
        }
        for network_pos in range(len(networks)):
            outputs = _discrete_euler_outputs(
                input_fft, long_steps, discrete_transfers, network_pos
            )
            extracted = _feature_arrays_vectorized(outputs, DT_MS, specs)
            for window, (values, is_valid) in extracted.items():
                features[window][network_pos] = values
                valid[window][network_pos] = is_valid

        condition = f"{design}_eps{eps:g}_rho{rho:g}"
        for window in WINDOWS:
            for output_pos, output_area in enumerate(output_areas):
                use = np.all(valid[window][:, output_pos, :], axis=0) & ~clipped
                accepted_u = (z[use] - Z0) / ZSCALE
                for network_pos, (network_id, _, _, _) in enumerate(networks):
                    _, actual_r = _cov_r(features[window][network_pos, output_pos, use])
                    fun, grad, hess = theory[(network_pos, output_pos)]
                    first, second = _taylor_values(fun, grad, hess, accepted_u)
                    exact = np.asarray([fun(item) for item in accepted_u])
                    for order, values in (
                        ("first", first),
                        ("second", second),
                        ("exact_spectrum", exact),
                    ):
                        _, predicted_r = _cov_r(values)
                        rows.append(
                            {
                                "repeat": repeat,
                                "window": window,
                                "group": group_tag,
                                "input_area": group["input_area"],
                                "lesion_area": group["lesion_area"],
                                "degree_type": group["degree_type"],
                                "output_area": output_area,
                                "network_id": network_id,
                                "network_class": "baseline" if network_pos == 0 else "10-edge deletion",
                                "order": order,
                                "condition": condition,
                                "design": design,
                                "epsilon": eps,
                                "rho": rho,
                                "n_trials": n_trials,
                                "n_valid_common": int(use.sum()),
                                "predicted_r": predicted_r,
                                "simulation_r": actual_r,
                            }
                        )
        print(
            f"maskwise Experiment 1 repeat {repeat + 1}/{N_REPEATS} {group_tag}: "
            f"condition {condition_pos + 1}/{len(_exp1_conditions(cfg))}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(result_path, index=False, compression="gzip")
    return frame


def _metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = [
        "repeat",
        "window",
        "group",
        "input_area",
        "lesion_area",
        "degree_type",
        "output_area",
        "network_id",
        "network_class",
        "order",
    ]
    for key, group in results.groupby(keys, sort=False):
        rows.append(dict(zip(keys, key)) | _fit_metrics(group.simulation_r, group.predicted_r))
    return pd.DataFrame(rows)


def _phase_agreement(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["window", "group", "output_area", "order", "network_class"]
    for key, group in metrics.groupby(keys, sort=False):
        pivot = group.pivot(index="network_id", columns="repeat", values="R2")
        correlations = []
        for i in pivot.columns:
            for j in pivot.columns:
                if i < j and pivot[i].std() > 0 and pivot[j].std() > 0:
                    correlations.append(np.corrcoef(pivot[i], pivot[j])[0, 1])
        rows.append(
            dict(zip(keys, key))
            | {
                "n_networks": len(pivot),
                "median_pairwise_repeat_r": float(np.median(correlations)) if correlations else np.nan,
                "min_pairwise_repeat_r": float(np.min(correlations)) if correlations else np.nan,
                "max_pairwise_repeat_r": float(np.max(correlations)) if correlations else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run(cfg: dict, repeats: int = N_REPEATS, n_trials: int = N_TRIALS, force: bool = False) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(_rooted(cfg["model_dir"]))
    frames = []
    for repeat in range(repeats):
        for group_pos, group in enumerate(GROUPS):
            frames.append(
                _simulate_group_repeat(
                    cfg, out, model, group, group_pos, repeat, n_trials, force=force
                )
            )
    results = pd.concat(frames, ignore_index=True)
    metrics = _metrics(results)
    agreement = _phase_agreement(metrics)
    results.to_csv(out / "maskwise_exp1_conditions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "maskwise_exp1_identity_metrics.csv", index=False)
    agreement.to_csv(out / "maskwise_exp1_phase_agreement.csv", index=False)
    _plot(out, results, metrics)
    simulation_files = list(out.glob(f"conditions_n{n_trials}_repeat*.csv.gz"))
    simulation_span_s = (
        max(path.stat().st_mtime for path in simulation_files)
        - min(path.stat().st_mtime for path in simulation_files)
        if len(simulation_files) > 1
        else 0.0
    )
    save_json(
        out / "manifest.json",
        {
            "experiment": "Experiment 1 rerun independently in baseline and every S13 10-edge network",
            "n_phase_repeats": repeats,
            "n_trials_per_input_condition": n_trials,
            "n_input_conditions": len(_exp1_conditions(cfg)),
            "n_masks_per_group": N_MASKS,
            "windows": {window: _window_specs(cfg)[window] for window in WINDOWS},
            "theory_orders": ORDERS,
            "simulation_solver": "validated exact discrete-time Euler transfer",
            "short_window_is_prefix_of_same_long_trajectory": True,
            "cached_reanalysis_runtime_s": time.time() - started,
            "simulation_file_mtime_span_s": simulation_span_s,
        },
    )
    return out


def _combinations(results):
    return results[
        ["group", "input_area", "lesion_area", "degree_type", "output_area"]
    ].drop_duplicates()


def _plot(out: Path, results: pd.DataFrame, metrics: pd.DataFrame):
    plt.style.use("seaborn-v0_8-whitegrid")
    combinations = _combinations(results)
    labels = [
        f"{r.input_area}/{r.lesion_area}-{r.degree_type}/{r.output_area}"
        for r in combinations.itertuples(index=False)
    ]
    x = np.arange(len(labels))
    colors = {"baseline": "#222222", "10-edge deletion": "#4C78A8"}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row_pos, window in enumerate(WINDOWS):
        for col_pos, order in enumerate(ORDERS):
            ax = axes[row_pos, col_pos]
            for pos, item in enumerate(combinations.itertuples(index=False)):
                subset = metrics[
                    (metrics.window == window)
                    & (metrics.group == item.group)
                    & (metrics.output_area == item.output_area)
                    & (metrics.order == order)
                ]
                deleted = subset[subset.network_class == "10-edge deletion"].R2
                repeats = sorted(subset.repeat.unique())
                per_repeat = [
                    subset[(subset.network_class == "10-edge deletion") & (subset.repeat == repeat)].R2
                    for repeat in repeats
                ]
                positions = pos + np.linspace(-0.12, 0.12, len(repeats))
                for position, values in zip(positions, per_repeat):
                    ax.scatter(
                        np.full(len(values), position), values, s=8, alpha=0.22, color=colors["10-edge deletion"]
                    )
                baseline = subset[subset.network_class == "baseline"]
                ax.scatter(
                    np.full(len(baseline), pos), baseline.R2, marker="D", s=34,
                    color=colors["baseline"], zorder=5,
                )
                ax.plot(
                    [pos - 0.17, pos + 0.17], [deleted.median(), deleted.median()],
                    color="#D1495B", lw=2,
                )
            ax.axhline(1, color="0.25", ls="--", lw=1)
            ax.axhline(0, color="0.5", ls=":", lw=1)
            ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
            ax.set(ylabel="Identity R2 across 14 input conditions", title=f"{window}; {order}")
    fig.suptitle("Experiment 1 in baseline and every targeted 10-edge network", fontsize=14)
    save_figure(fig, out / "figures" / "maskwise_exp1_identity_R2")
    plt.close(fig)

    for window in WINDOWS:
        for order in ("first", "second"):
            fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
            for ax, item in zip(axes.flat, combinations.itertuples(index=False)):
                candidates = metrics[
                    (metrics.repeat == 0)
                    & (metrics.window == window)
                    & (metrics.group == item.group)
                    & (metrics.output_area == item.output_area)
                    & (metrics.order == order)
                    & (metrics.network_class == "10-edge deletion")
                ].sort_values("R2")
                chosen = candidates.iloc[len(candidates) // 2].network_id
                data = results[
                    (results.repeat == 0)
                    & (results.window == window)
                    & (results.group == item.group)
                    & (results.output_area == item.output_area)
                    & (results.order == order)
                    & (results.network_id == chosen)
                ]
                ax.scatter(data.predicted_r, data.simulation_r, s=38, color="#247BA0", alpha=0.8)
                lo = min(data.predicted_r.min(), data.simulation_r.min())
                hi = max(data.predicted_r.max(), data.simulation_r.max())
                ax.plot([lo, hi], [lo, hi], "k--", lw=1)
                fit = _fit_metrics(data.simulation_r, data.predicted_r)
                trial_count = int(data.n_trials.iloc[0])
                ax.text(
                    0.04, 0.95,
                    f"{chosen}; identity R2={fit['R2']:.3f}\n14 conditions, {trial_count} trials each",
                    transform=ax.transAxes, va="top", fontsize=8,
                )
                ax.set(
                    xlabel=f"Experiment 1 theory r ({order})",
                    ylabel="Simulation r",
                    title=f"{item.input_area}; delete {item.lesion_area}-{item.degree_type}; read {item.output_area}",
                )
            fig.suptitle(f"Representative perturbed networks, {window} window: {order}", fontsize=14)
            save_figure(fig, out / "figures" / f"representative_exp1_scatter_{window}_{order}")
            plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-trials", type=int, default=N_TRIALS)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(load_config(args.config), args.repeats, args.n_trials, args.force))


if __name__ == "__main__":
    main()
