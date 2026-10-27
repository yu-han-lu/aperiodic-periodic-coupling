from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy import stats

from .core import apply_fln_delta, frequency_grid, load_model, save_figure, save_json
from .exp3 import Z0, ZS
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .exp3_large_delta_screen import (
    DT_MS,
    _continuous_transfer,
    _discrete_transfer,
    _exact_trial_features,
)
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .nonlinear_network_validation import _run_multioutput_batch
from .pairwise_nonlinear_window_validation import _window_specs


OUTPUT_NAME = "exp3_emergent_correlation_regime"
INPUT_AREA = "A6Va"
OUTPUT_AREA = "V2"
EPSILON = 0.05
N_TRIALS = 4096
N_ENSEMBLES = 3
PHASES_PER_ENSEMBLE = 2
CHUNK_SIZE = 32

# Frozen after an exact-resolvent-only discovery scan. Each marginal variance is
# one; the matrix is positive definite and was not adjusted using time-domain data.
INPUT_CORRELATION = np.array(
    [
        [1.0, 0.3613, -0.1941, 0.5341],
        [0.3613, 1.0, 0.7878, -0.5540],
        [-0.1941, 0.7878, 1.0, -0.8931],
        [0.5341, -0.5540, -0.8931, 1.0],
    ]
)


def _pearson(values: np.ndarray, use: np.ndarray) -> tuple[float, float]:
    result = stats.pearsonr(values[use, 0], values[use, 1])
    return float(result.statistic), float(result.pvalue)


def _trial_design(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.multivariate_normal(np.zeros(4), INPUT_CORRELATION, size=N_TRIALS) * EPSILON
    z = np.clip(
        Z0 + ZS * u,
        [5.2, 1.0, -1.8, 0.15],
        [19.5, 15.0, 1.0, 3.0],
    )
    return z, u


def _frozen_mask(model):
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    delta_w = np.zeros_like(model.fln)
    edge_rows = []
    for target, source in zip(*np.nonzero(model.fln)):
        delete = (target == output_idx and source != input_idx) or (
            source == input_idx and target != output_idx
        )
        if not delete:
            continue
        delta_w[target, source] = -model.fln[target, source]
        family = "other inputs to V2" if target == output_idx else "other outputs from A6Va"
        edge_rows.append(
            {
                "source_idx": int(source),
                "target_idx": int(target),
                "source_area": model.areas[source],
                "target_area": model.areas[target],
                "weight": float(model.fln[target, source]),
                "mask_family": family,
            }
        )
    return delta_w, pd.DataFrame(edge_rows).sort_values("weight", ascending=False)


def _simulate_features(cfg, model, perturbed_j, z, out: Path, force: bool):
    cache = out / "phase_features.npz"
    total_phases = N_ENSEMBLES * PHASES_PER_ENSEMBLE
    if cache.exists() and not force:
        data = np.load(cache)
        if data["features"].shape[0] == total_phases:
            return data["features"], data["valid"]

    specs = {"long": dict(_window_specs(cfg)["long"])}
    analysis_s = float(specs["long"]["analysis_s"])
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    tau_e = float(model.params["tau_exc"])
    transfers = np.asarray(
        [
            _discrete_transfer(model.J, input_idx, output_idx, n_steps, tau_e),
            _discrete_transfer(perturbed_j, input_idx, output_idx, n_steps, tau_e),
        ]
    )
    features = np.empty((total_phases, 2, N_TRIALS, 2), dtype=np.float64)
    valid = np.empty((total_phases, 2, N_TRIALS), dtype=bool)
    input_sigma = float(cfg["exp3_nonlinear"]["input_sigma"])
    for phase in range(total_phases):
        for start in range(0, N_TRIALS, CHUNK_SIZE):
            stop = min(start + CHUNK_SIZE, N_TRIALS)
            inputs = _matched_bandlimited_inputs(
                z[start:stop],
                DT_MS,
                analysis_s,
                burn_s,
                input_sigma,
                int(cfg["seed"]) + 6300000 + phase * 100000,
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
        print(f"emergent-correlation holdout phase {phase + 1}/{total_phases}", flush=True)
    np.savez_compressed(cache, features=features, valid=valid, z=z)
    return features, valid


def _aggregate(features, valid, theory_valid):
    rows = []
    examples = None
    valid_fraction = valid.mean(axis=0)
    fixed_use = theory_valid & (valid_fraction >= 0.75).all(axis=0)
    for ensemble in range(N_ENSEMBLES):
        lo = ensemble * PHASES_PER_ENSEMBLE
        hi = lo + PHASES_PER_ENSEMBLE
        block_features = features[lo:hi]
        block_valid = valid[lo:hi]
        counts = block_valid.sum(axis=0)
        summed = np.where(block_valid[..., None], block_features, 0.0).sum(axis=0)
        averaged = summed / np.maximum(counts[..., None], 1)
        use = fixed_use & (counts >= int(np.ceil(0.75 * PHASES_PER_ENSEMBLE))).all(axis=0)
        base_r, base_p = _pearson(averaged[0], use)
        pert_r, pert_p = _pearson(averaged[1], use)
        rows.append(
            {
                "ensemble": ensemble + 1,
                "n_valid_common": int(use.sum()),
                "simulation_baseline_r": base_r,
                "simulation_baseline_p": base_p,
                "simulation_deleted_r": pert_r,
                "simulation_deleted_p": pert_p,
                "simulation_delta_r": pert_r - base_r,
            }
        )
        if examples is None:
            examples = (averaged, use)
    return pd.DataFrame(rows), examples, fixed_use


def _explicit_recurrence_qc(cfg, model, perturbed_j, delta_w, z, out: Path):
    specs = {"long": dict(_window_specs(cfg)["long"])}
    analysis_s = float(specs["long"]["analysis_s"])
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    inputs = _matched_bandlimited_inputs(
        z[:4], DT_MS, analysis_s, burn_s,
        float(cfg["exp3_nonlinear"]["input_sigma"]), int(cfg["seed"]) + 6900000,
    )
    rows = []
    networks = [
        ("baseline", model.params, model.J),
        (
            "deleted",
            copy.copy(model.params) | {"fln_mat": model.fln + delta_w},
            perturbed_j,
        ),
    ]
    for name, params, j in networks:
        explicit, rect, *_ = _run_multioutput_batch(
            params, inputs, DT_MS, burn_steps, np.asarray([output_idx]), input_idx=input_idx
        )
        transfer = _discrete_transfer(
            j, input_idx, output_idx, n_steps, float(model.params["tau_exc"])
        )
        accelerated = np.fft.irfft(
            np.fft.rfft(inputs[:, burn_steps:], axis=1) * transfer[None, :],
            n=n_steps,
            axis=1,
        )
        error = explicit[:, 0] - accelerated
        rows.append(
            {
                "network": name,
                "n_trials": len(inputs),
                "rectification_count": int(rect.sum()),
                "timeseries_r": float(np.corrcoef(explicit[:, 0].ravel(), accelerated.ravel())[0, 1]),
                "relative_rmse": float(
                    np.sqrt(np.mean(error**2)) / max(np.std(explicit[:, 0]), 1e-15)
                ),
                "max_abs_error": float(np.max(np.abs(error))),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(out / "explicit_recurrence_qc.csv", index=False)
    return result


def _mean_ci(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    sem = float(stats.sem(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    return mean, mean - half, mean + half


def _plot(out, edges, theory, ensemble, example):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), constrained_layout=True)

    top = edges.head(24).iloc[::-1]
    colors = top.mask_family.map(
        {"other inputs to V2": "#D1495B", "other outputs from A6Va": "#247BA0"}
    )
    axes[0, 0].barh(
        np.arange(len(top)), top.weight, color=colors,
    )
    axes[0, 0].set_yticks(
        np.arange(len(top)), top.source_area + " -> " + top.target_area, fontsize=7
    )
    axes[0, 0].set(
        xlabel="deleted FLN weight",
        title=f"Frozen path-focused mask: {len(edges)} deleted edges",
    )
    axes[0, 0].legend(
        handles=[
            Patch(color="#D1495B", label="other inputs to V2"),
            Patch(color="#247BA0", label="other outputs from A6Va"),
        ],
        frameon=False, fontsize=8, loc="lower right",
    )

    sim_base = ensemble.simulation_baseline_r.to_numpy()
    sim_deleted = ensemble.simulation_deleted_r.to_numpy()
    means = [sim_base.mean(), sim_deleted.mean()]
    errors = [sim_base.std(ddof=1), sim_deleted.std(ddof=1)]
    x = np.arange(2)
    axes[0, 1].bar(
        x - 0.18, [theory["baseline_r"], theory["deleted_r"]], 0.36,
        color="#6A4C93", label="continuous exact resolvent",
    )
    axes[0, 1].bar(
        x + 0.18, means, 0.36, yerr=errors, capsize=4,
        color="#2A9D8F", label="long-window simulation mean +/- SD",
    )
    axes[0, 1].axhline(0, color="0.25", lw=1)
    axes[0, 1].set_xticks(x, ["baseline", "deleted"])
    axes[0, 1].set(ylabel="peak-slope correlation r", title="Theory and simulation")
    axes[0, 1].legend(frameon=False, fontsize=8)
    axes[0, 1].text(
        0.02, 0.03,
        f"theory p: {theory['baseline_p']:.3g} / {theory['deleted_p']:.3g}",
        transform=axes[0, 1].transAxes, fontsize=8,
    )

    for row in ensemble.itertuples(index=False):
        axes[1, 0].plot(
            [0, 1], [row.simulation_baseline_r, row.simulation_deleted_r],
            color="0.65", alpha=0.8, lw=1,
        )
        axes[1, 0].scatter(
            [0, 1], [row.simulation_baseline_r, row.simulation_deleted_r],
            color=["#247BA0", "#D1495B"], s=36, zorder=3,
        )
    axes[1, 0].hlines(
        [theory["baseline_r"], theory["deleted_r"]],
        [-0.22, 0.78], [0.22, 1.22], color="#6A4C93", lw=3,
    )
    axes[1, 0].axhline(0, color="0.25", ls=":")
    axes[1, 0].set_xticks([0, 1], ["baseline", "deleted"])
    axes[1, 0].set(
        xlim=(-0.3, 1.3), ylabel="correlation r",
        title=f"{N_ENSEMBLES} independent phase ensembles",
    )

    averaged, use = example
    scatter_colors = ["#247BA0", "#D1495B"]
    for pos, label in enumerate(("baseline", "deleted")):
        xval = averaged[pos, use, 0] / (2 * np.pi)
        yval = averaged[pos, use, 1]
        fit = stats.linregress(xval, yval)
        axes[1, 1].scatter(
            xval, yval, s=10, alpha=0.34, color=scatter_colors[pos],
            label=f"{label}: r={fit.rvalue:.3f}, p={fit.pvalue:.2g}",
        )
        grid = np.linspace(xval.min(), xval.max(), 100)
        axes[1, 1].plot(grid, fit.intercept + fit.slope * grid, color=scatter_colors[pos], lw=2)
    axes[1, 1].set(
        xlabel="oscillation peak [Hz]", ylabel="aperiodic exponent",
        title="Representative holdout ensemble",
    )
    axes[1, 1].legend(frameon=False)
    fig.suptitle(
        f"Correlation emerges after path-focused deletion ({INPUT_AREA} input -> {OUTPUT_AREA} readout; "
        f"retained power={100 * theory['retained_power']:.3f}%)",
        fontsize=15,
    )
    save_figure(fig, out / "figures" / "emergent_correlation_theory_simulation")
    plt.close(fig)


def run(cfg, force=False):
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(Path(cfg["model_dir"]))
    delta_w, edges = _frozen_mask(model)
    edges.to_csv(out / "deleted_edges.csv", index=False)
    perturbed_j = apply_fln_delta(model, delta_w)
    z, u = _trial_design(int(cfg["seed"]) + 6200000)
    pd.DataFrame(z, columns=["bump_center_hz", "bump_amplitude", "beta", "background_amplitude"]).to_csv(
        out / "input_parameter_trials.csv.gz", index=False, compression="gzip"
    )

    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    theory_base, theory_base_valid = _exact_trial_features(
        _continuous_transfer(model.J, input_idx, output_idx), z
    )
    theory_deleted, theory_deleted_valid = _exact_trial_features(
        _continuous_transfer(perturbed_j, input_idx, output_idx), z
    )
    input_features, input_valid = _exact_trial_features(np.ones_like(frequency_grid()), z)
    theory_valid = theory_base_valid & theory_deleted_valid & input_valid
    input_r, input_p = _pearson(input_features, theory_valid)
    baseline_r, baseline_p = _pearson(theory_base, theory_valid)
    deleted_r, deleted_p = _pearson(theory_deleted, theory_valid)
    theory = {
        "input_r": input_r,
        "input_p": input_p,
        "baseline_r": baseline_r,
        "baseline_p": baseline_p,
        "deleted_r": deleted_r,
        "deleted_p": deleted_p,
        "delta_r": deleted_r - baseline_r,
        "n_valid": int(theory_valid.sum()),
    }
    continuous_base = _continuous_transfer(model.J, input_idx, output_idx)
    continuous_deleted = _continuous_transfer(perturbed_j, input_idx, output_idx)
    frequencies = frequency_grid()
    peak_band = (frequencies >= 5.0) & (frequencies <= 20.0)
    retained_power = float(np.median(continuous_deleted[peak_band] / continuous_base[peak_band]))
    theory["retained_power"] = retained_power

    features, valid = _simulate_features(cfg, model, perturbed_j, z, out, force)
    ensemble, example, fixed_use = _aggregate(features, valid, theory_valid)
    ensemble.to_csv(out / "holdout_ensemble_correlations.csv", index=False)
    qc = _explicit_recurrence_qc(cfg, model, perturbed_j, delta_w, z, out)
    base_mean, base_low, base_high = _mean_ci(ensemble.simulation_baseline_r)
    deleted_mean, deleted_low, deleted_high = _mean_ci(ensemble.simulation_deleted_r)
    delta_mean, delta_low, delta_high = _mean_ci(ensemble.simulation_delta_r)
    summary = pd.DataFrame(
        [
            {
                "input_area": INPUT_AREA,
                "output_area": OUTPUT_AREA,
                "preserved_route": f"{INPUT_AREA} -> {OUTPUT_AREA}",
                "n_deleted_edges": len(edges),
                "epsilon": EPSILON,
                "n_parameter_trials": N_TRIALS,
                "analysis_window_s": float(_window_specs(cfg)["long"]["analysis_s"]),
                "burn_in_s": float(cfg["exp3_nonlinear"]["burn_s"]),
                "n_ensembles": N_ENSEMBLES,
                "phases_per_ensemble": PHASES_PER_ENSEMBLE,
                "n_valid_fixed": int(fixed_use.sum()),
                "input_r": input_r,
                "input_p": input_p,
                "theory_baseline_r": baseline_r,
                "theory_baseline_p": baseline_p,
                "theory_deleted_r": deleted_r,
                "theory_deleted_p": deleted_p,
                "theory_delta_r": deleted_r - baseline_r,
                "simulation_baseline_r_mean": base_mean,
                "simulation_baseline_r_ci_low": base_low,
                "simulation_baseline_r_ci_high": base_high,
                "simulation_deleted_r_mean": deleted_mean,
                "simulation_deleted_r_ci_low": deleted_low,
                "simulation_deleted_r_ci_high": deleted_high,
                "simulation_delta_r_mean": delta_mean,
                "simulation_delta_r_ci_low": delta_low,
                "simulation_delta_r_ci_high": delta_high,
                "baseline_ensemble_significant_fraction": float(
                    np.mean(ensemble.simulation_baseline_p < 0.05)
                ),
                "deleted_ensemble_significant_fraction": float(
                    np.mean(ensemble.simulation_deleted_p < 0.05)
                ),
                "input_covariance_min_eigenvalue": float(np.linalg.eigvalsh(INPUT_CORRELATION).min()),
                "median_5_20hz_retained_transfer_power": retained_power,
                "median_5_20hz_retained_transfer_amplitude": float(np.sqrt(retained_power)),
                "max_rectification_count_qc": int(qc.rectification_count.max()),
            }
        ]
    )
    summary.to_csv(out / "summary.csv", index=False)
    np.savez_compressed(
        out / "theory_features.npz",
        baseline=theory_base,
        deleted=theory_deleted,
        input=input_features,
        valid=theory_valid,
        input_correlation=INPUT_CORRELATION,
        standardized_trials=u,
    )
    _plot(out, edges, theory, ensemble, example)
    save_json(
        out / "manifest.json",
        {
            "experiment": "frozen no-baseline-correlation / post-deletion-correlation regime",
            "selection_rule": "exact-resolvent discovery only; time-domain results are holdout",
            "mask": "delete all other inputs to V2 and all other outputs from A6Va; retain A6Va->V2",
            "input_parameter_order": ["bump_center_hz", "bump_amplitude", "beta", "background_amplitude"],
            "input_correlation": INPUT_CORRELATION.tolist(),
            "epsilon": EPSILON,
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
