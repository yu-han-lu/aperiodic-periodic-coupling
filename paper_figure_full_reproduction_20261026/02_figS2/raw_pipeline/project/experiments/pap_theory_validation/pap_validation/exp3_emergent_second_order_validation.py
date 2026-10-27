from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    apply_fln_delta,
    finite_derivatives,
    frequency_grid,
    load_model,
    network_features,
    save_figure,
    save_json,
)
from .exp3 import Z0, ZS
from .exp3_emergent_correlation_regime import (
    INPUT_AREA,
    OUTPUT_AREA,
    _frozen_mask,
)
from .exp3_large_delta_screen import _continuous_transfer
from .io import load_config, prepare_output


OUTPUT_NAME = "exp3_emergent_second_order_validation"
SOURCE_NAME = "exp3_emergent_correlation_regime"
N_BOOTSTRAP = 2000
DERIVATIVE_STEP = 2e-3
SENSITIVITY_STEP = 4e-3


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def _correlation(values: np.ndarray) -> tuple[float, float]:
    result = stats.pearsonr(values[:, 0], values[:, 1])
    return float(result.statistic), float(result.pvalue)


def _taylor_features(fun, x: np.ndarray, step: float):
    base = fun(np.zeros(x.shape[1]))
    grad, hess, _ = finite_derivatives(fun, np.zeros(x.shape[1]), h=step)
    first = base + x @ grad.T
    second = first + 0.5 * np.einsum("ni,kij,nj->nk", x, hess, x)
    return first, second, grad, hess


def _bootstrap_correlations(feature_maps, rng):
    n = len(next(iter(feature_maps.values()))[0])
    names = tuple(feature_maps)
    samples = {name: np.empty((N_BOOTSTRAP, 3)) for name in names}
    for position in range(N_BOOTSTRAP):
        indices = rng.integers(0, n, n)
        for name, (baseline, deleted) in feature_maps.items():
            base_r = _correlation(baseline[indices])[0]
            deleted_r = _correlation(deleted[indices])[0]
            samples[name][position] = base_r, deleted_r, deleted_r - base_r
    return samples


def _correlation_rows(feature_maps, bootstrap):
    rows = []
    for method, (baseline, deleted) in feature_maps.items():
        for network_pos, (network, values) in enumerate(
            (("baseline", baseline), ("deleted", deleted))
        ):
            r_value, p_value = _correlation(values)
            low, high = np.percentile(bootstrap[method][:, network_pos], [2.5, 97.5])
            rows.append(
                {
                    "method": method,
                    "network": network,
                    "r": r_value,
                    "p": p_value,
                    "bootstrap_ci_low": float(low),
                    "bootstrap_ci_high": float(high),
                    "n_trials": len(values),
                }
            )
        delta = _correlation(deleted)[0] - _correlation(baseline)[0]
        low, high = np.percentile(bootstrap[method][:, 2], [2.5, 97.5])
        rows.append(
            {
                "method": method,
                "network": "deleted_minus_baseline",
                "r": delta,
                "p": np.nan,
                "bootstrap_ci_low": float(low),
                "bootstrap_ci_high": float(high),
                "n_trials": len(baseline),
            }
        )
    return pd.DataFrame(rows)


def _phase_repeat_rows(features, valid, theory_valid, theoretical_maps):
    rows = []
    n_phases = features.shape[0]
    if n_phases % 2:
        raise ValueError("Expected an even number of phase realizations")
    for ensemble in range(n_phases // 2):
        lo, hi = 2 * ensemble, 2 * ensemble + 2
        block_valid = valid[lo:hi]
        counts = block_valid.sum(axis=0)
        use = theory_valid & (counts >= 2).all(axis=0)
        averaged = np.where(
            block_valid[..., None], features[lo:hi], 0.0
        ).sum(axis=0) / np.maximum(counts[..., None], 1)
        maps = {"simulation": (averaged[0, use], averaged[1, use])}
        maps.update(
            {
                name: (values[0][use], values[1][use])
                for name, values in theoretical_maps.items()
            }
        )
        for method, (baseline, deleted) in maps.items():
            base_r, _ = _correlation(baseline)
            deleted_r, _ = _correlation(deleted)
            rows.append(
                {
                    "ensemble": ensemble + 1,
                    "method": method,
                    "n_valid_common": int(use.sum()),
                    "baseline_r": base_r,
                    "deleted_r": deleted_r,
                    "delta_r": deleted_r - base_r,
                }
            )
    return pd.DataFrame(rows)


def _plot_summary(out, correlations):
    plt.style.use("seaborn-v0_8-whitegrid")
    order = ["first_order", "second_order", "exact_resolvent", "simulation"]
    labels = ["First", "Second", "Exact", "Simulation"]
    colors = ["#6A4C93", "#F28E2B", "#D1495B", "#2A9D8F"]
    networks = ["baseline", "deleted", "deleted_minus_baseline"]
    titles = ["Baseline network", "68-edge deletion network", "Deleted - baseline"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 9.0), constrained_layout=True)
    for ax, network, title in zip(axes.flat[:3], networks, titles):
        subset = correlations.set_index(["method", "network"])
        values = np.asarray([subset.loc[(method, network), "r"] for method in order])
        low = np.asarray([subset.loc[(method, network), "bootstrap_ci_low"] for method in order])
        high = np.asarray([subset.loc[(method, network), "bootstrap_ci_high"] for method in order])
        ax.bar(np.arange(4), values, color=colors, width=0.68)
        ax.errorbar(
            np.arange(4), values, yerr=np.vstack((values - low, high - values)),
            fmt="none", ecolor="0.2", capsize=4, lw=1.2,
        )
        ax.axhline(0, color="0.2", lw=1)
        ax.set_xticks(np.arange(4), labels)
        ax.set(ylabel="P-A correlation r" if network != networks[-1] else "Delta r", title=title)
        for position, value in enumerate(values):
            ax.text(position, value, f" {value:.4f}", ha="center", va="bottom", fontsize=8)

    simulation = correlations[correlations.method == "simulation"].set_index("network")
    errors = []
    for method in order[:-1]:
        predicted = correlations[correlations.method == method].set_index("network")
        errors.append([abs(predicted.loc[network, "r"] - simulation.loc[network, "r"]) for network in networks])
    x = np.arange(3)
    width = 0.23
    for pos, (method, values, color) in enumerate(zip(labels[:3], errors, colors[:3])):
        axes[1, 1].bar(x + (pos - 1) * width, values, width, color=color, label=method)
    axes[1, 1].set_xticks(x, ["baseline r", "deleted r", "Delta r"])
    axes[1, 1].set(ylabel="absolute theory-simulation error", title="Does second order improve correspondence?")
    axes[1, 1].legend(frameon=False)
    fig.suptitle(
        "Experiment 3 as two Experiment 1 analyses: A6Va input -> V2 readout",
        fontsize=15,
    )
    save_figure(fig, out / "figures" / "exp3_68edge_second_order_validation")
    plt.close(fig)


def _plot_trial_maps(out, feature_maps):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(13.2, 8.3), constrained_layout=True)
    columns = [
        ("exact_resolvent", "Exact resolvent"),
        ("second_order", "Second-order Taylor"),
        ("simulation", "6-phase long-window simulation"),
    ]
    for row, network in enumerate(("baseline", "deleted")):
        network_pos = 0 if network == "baseline" else 1
        for col, (method, title) in enumerate(columns):
            values = feature_maps[method][network_pos]
            peak_hz = values[:, 0] / (2.0 * np.pi)
            exponent = values[:, 1]
            fit = stats.linregress(peak_hz, exponent)
            axes[row, col].hexbin(peak_hz, exponent, gridsize=42, mincnt=1, cmap="viridis")
            grid = np.linspace(peak_hz.min(), peak_hz.max(), 120)
            axes[row, col].plot(grid, fit.intercept + fit.slope * grid, color="#D1495B", lw=1.8)
            axes[row, col].set(
                xlabel="oscillation peak [Hz]", ylabel="aperiodic exponent",
                title=f"{network}: {title}\nr={fit.rvalue:.4f}, p={fit.pvalue:.2g}",
            )
    fig.suptitle("Trial-level P-A maps on the same frozen 3,224-trial set", fontsize=15)
    save_figure(fig, out / "figures" / "exp3_68edge_trial_maps")
    plt.close(fig)


def _plot_phase_repeats(out, phase_repeats):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.6, 4.8), constrained_layout=True)
    style = {
        "first_order": ("#6A4C93", "o", "First"),
        "second_order": ("#F28E2B", "s", "Second"),
        "exact_resolvent": ("#D1495B", "^", "Exact"),
        "simulation": ("#2A9D8F", "D", "Simulation"),
    }
    for method, group in phase_repeats.groupby("method"):
        color, marker, label = style[method]
        ax.plot(group.ensemble, group.delta_r, marker=marker, color=color, label=label)
    ax.axhline(0, color="0.2", lw=1)
    ax.set(
        xlabel="independent two-phase ensemble", ylabel="Delta r",
        title="Realization check with matched valid trials per ensemble",
        xticks=[1, 2, 3],
    )
    ax.legend(frameon=False, ncol=4)
    save_figure(fig, out / "figures" / "exp3_68edge_phase_repeats")
    plt.close(fig)


def run(cfg, force=False):
    del force  # This validation reuses the frozen, already simulated holdout.
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    source = _rooted(cfg["output_dir"]) / SOURCE_NAME
    theory_cache = np.load(source / "theory_features.npz")
    phase_cache = np.load(source / "phase_features.npz")
    model = load_model(_rooted(cfg["model_dir"]))
    delta_w, deleted_edges = _frozen_mask(model)
    perturbed_j = apply_fln_delta(model, delta_w)
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    frequencies = frequency_grid()

    theory_valid = theory_cache["valid"]
    phase_features = phase_cache["features"]
    phase_valid = phase_cache["valid"]
    fixed_use = theory_valid & (phase_valid.mean(axis=0) >= 0.75).all(axis=0)
    standardized = theory_cache["standardized_trials"]
    center = standardized[fixed_use].mean(axis=0)
    centered = standardized[fixed_use] - center

    exact_maps = (
        theory_cache["baseline"][fixed_use],
        theory_cache["deleted"][fixed_use],
    )
    first_maps = []
    second_maps = []
    derivative_rows = []
    feature_rows = []
    for network, j, exact in (
        ("baseline", model.J, exact_maps[0]),
        ("deleted", perturbed_j, exact_maps[1]),
    ):
        transfer = _continuous_transfer(j, input_idx, output_idx)

        def feature_fun(du):
            feature, _ = network_features(
                j, Z0 + ZS * (center + du), input_idx, output_idx,
                frequencies, transfer,
            )
            return np.asarray([feature["peak_omega"], feature["exponent"]])

        first, second, grad, hess = _taylor_features(
            feature_fun, centered, DERIVATIVE_STEP
        )
        _, _, grad_check, hess_check = _taylor_features(
            feature_fun, centered[:1], SENSITIVITY_STEP
        )
        first_maps.append(first)
        second_maps.append(second)
        derivative_rows.append(
            {
                "network": network,
                "gradient_step_relative_error": float(
                    np.linalg.norm(grad - grad_check) / max(np.linalg.norm(grad), 1e-15)
                ),
                "hessian_step_relative_error": float(
                    np.linalg.norm(hess - hess_check) / max(np.linalg.norm(hess), 1e-15)
                ),
                "center_standardized_norm": float(np.linalg.norm(center)),
            }
        )
        for method, values in (("first_order", first), ("second_order", second)):
            feature_rows.append(
                {
                    "network": network,
                    "method": method,
                    "peak_rmse_hz_vs_exact": float(
                        np.sqrt(np.mean((values[:, 0] - exact[:, 0]) ** 2)) / (2.0 * np.pi)
                    ),
                    "exponent_rmse_vs_exact": float(
                        np.sqrt(np.mean((values[:, 1] - exact[:, 1]) ** 2))
                    ),
                }
            )

    phase_counts = phase_valid.sum(axis=0)
    phase_average = np.where(
        phase_valid[..., None], phase_features, 0.0
    ).sum(axis=0) / np.maximum(phase_counts[..., None], 1)
    simulation_maps = (
        phase_average[0, fixed_use],
        phase_average[1, fixed_use],
    )
    feature_maps = {
        "first_order": tuple(first_maps),
        "second_order": tuple(second_maps),
        "exact_resolvent": exact_maps,
        "simulation": simulation_maps,
    }
    bootstrap = _bootstrap_correlations(
        feature_maps, np.random.default_rng(int(cfg["seed"]) + 7100000)
    )
    correlations = _correlation_rows(feature_maps, bootstrap)
    correlations.to_csv(out / "correlation_validation.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out / "feature_approximation_metrics.csv", index=False)
    pd.DataFrame(derivative_rows).to_csv(out / "derivative_step_validation.csv", index=False)

    all_trial_maps = {
        "first_order": (
            np.full_like(theory_cache["baseline"], np.nan),
            np.full_like(theory_cache["deleted"], np.nan),
        ),
        "second_order": (
            np.full_like(theory_cache["baseline"], np.nan),
            np.full_like(theory_cache["deleted"], np.nan),
        ),
        "exact_resolvent": (theory_cache["baseline"], theory_cache["deleted"]),
    }
    for pos in range(2):
        all_trial_maps["first_order"][pos][fixed_use] = first_maps[pos]
        all_trial_maps["second_order"][pos][fixed_use] = second_maps[pos]
    phase_repeats = _phase_repeat_rows(
        phase_features, phase_valid, fixed_use, all_trial_maps
    )
    phase_repeats.to_csv(out / "phase_repeat_validation.csv", index=False)

    np.savez_compressed(
        out / "validation_features.npz",
        fixed_use=fixed_use,
        center=center,
        baseline_first=first_maps[0],
        baseline_second=second_maps[0],
        baseline_exact=exact_maps[0],
        baseline_simulation=simulation_maps[0],
        deleted_first=first_maps[1],
        deleted_second=second_maps[1],
        deleted_exact=exact_maps[1],
        deleted_simulation=simulation_maps[1],
    )
    _plot_summary(out, correlations)
    _plot_trial_maps(out, feature_maps)
    _plot_phase_repeats(out, phase_repeats)
    save_json(
        out / "manifest.json",
        {
            "experiment": "68-edge emergent-correlation second-order Experiment 1 validation",
            "source": SOURCE_NAME,
            "input_area": INPUT_AREA,
            "output_area": OUTPUT_AREA,
            "n_deleted_edges": int(len(deleted_edges)),
            "n_parameter_trials_total": int(len(theory_valid)),
            "n_frozen_trials": int(fixed_use.sum()),
            "phase_realizations": int(phase_features.shape[0]),
            "analysis_window_s": 32.768,
            "derivative_center": center.tolist(),
            "derivative_step": DERIVATIVE_STEP,
            "bootstrap_repetitions": N_BOOTSTRAP,
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
