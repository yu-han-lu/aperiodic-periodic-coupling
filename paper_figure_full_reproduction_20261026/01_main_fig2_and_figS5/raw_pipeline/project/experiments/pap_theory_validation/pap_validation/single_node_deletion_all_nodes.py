from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import input_psd, load_model
from .exp3_large_delta_screen import _continuous_transfer, _discrete_transfer
from .io import load_config
from .plot_formatting import format_correlation, format_p_value
from .paper_figure_node_deletion import (
    CALIBRATION_PHASES,
    DT_MS,
    EXP3_ANALYSIS_S,
    EXP3_TRIALS,
    HELDOUT_PHASES,
    SEED,
    _all_transfer_power,
    _corr,
    _cov_to_corr,
    _features_from_averaged_psd,
    _node_deleted_j,
    _observed_values,
    _style,
)
from .single_node_deletion_revision import (
    COARSE_BIN_COUNT,
    INPUT_AREA,
    OUTPUT_AREA,
    _design,
    _frequency_grid,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "single_node_deletion_revision"
OUTPUT = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "single_node_deletion_all_nodes"


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _full_error_prediction(signal: np.ndarray, calibration: np.ndarray) -> float:
    error = calibration - signal
    signal_centered = signal - signal.mean(axis=0, keepdims=True)
    error_centered = error - error.mean(axis=0, keepdims=True)
    signal_cov = np.cov(signal, rowvar=False, ddof=1)
    error_cov = np.cov(error, rowvar=False, ddof=1)
    cross = signal_centered.T @ error_centered / (len(signal) - 1)
    return _cov_to_corr(signal_cov + error_cov + cross + cross.T)


def _shared_inputs(selected: dict, frequencies: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = OUTPUT / "shared_longwindow_input_psd_means.npz"
    if path.exists():
        data = np.load(path)
        return data["z"], data["u"], data["calibration_input_psd"], data["heldout_input_psd"]

    z, u = _design(
        np.asarray(selected["sigma"]),
        float(selected["epsilon"]),
        EXP3_TRIALS,
        SEED + 990000,
    )
    deterministic = np.asarray(
        [input_psd(frequencies, trial_z) for trial_z in z], dtype=np.float32
    )
    calibration_sum = np.zeros_like(deterministic)
    heldout_sum = np.zeros_like(deterministic)
    for phase in range(CALIBRATION_PHASES + HELDOUT_PHASES):
        rng = np.random.default_rng(SEED + 1010000 + phase * 100000)
        realization = rng.gamma(
            COARSE_BIN_COUNT,
            1.0 / COARSE_BIN_COUNT,
            size=deterministic.shape,
        ).astype(np.float32)
        if phase < CALIBRATION_PHASES:
            calibration_sum += deterministic * realization
        else:
            heldout_sum += deterministic * realization
        print(f"shared long-window input realization {phase + 1}/{CALIBRATION_PHASES + HELDOUT_PHASES}", flush=True)
    calibration = calibration_sum / CALIBRATION_PHASES
    heldout = heldout_sum / HELDOUT_PHASES
    np.savez_compressed(
        path,
        frequencies=frequencies,
        z=z,
        u=u,
        calibration_input_psd=calibration,
        heldout_input_psd=heldout,
        calibration_phases=CALIBRATION_PHASES,
        heldout_phases=HELDOUT_PHASES,
        seed=SEED,
    )
    return z, u, calibration, heldout


def _network_features(
    transfer_power: np.ndarray,
    frequencies: np.ndarray,
    calibration_input_psd: np.ndarray,
    heldout_input_psd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    calibration, calibration_valid = _features_from_averaged_psd(
        frequencies,
        (calibration_input_psd * transfer_power[None, :])[None, :, :],
    )
    heldout, heldout_valid = _features_from_averaged_psd(
        frequencies,
        (heldout_input_psd * transfer_power[None, :])[None, :, :],
    )
    return calibration[0], calibration_valid[0], heldout[0], heldout_valid[0]


def run_all_nodes(model, selected: dict) -> pd.DataFrame:
    result_path = OUTPUT / "all_single_node_deletion_results.csv"
    feature_path = OUTPUT / "all_single_node_deletion_features.npz"
    if result_path.exists() and feature_path.exists():
        return pd.read_csv(result_path)

    OUTPUT.mkdir(parents=True, exist_ok=True)
    frequencies, frequency_ids, n_steps = _frequency_grid()
    z, u, calibration_input, heldout_input = _shared_inputs(selected, frequencies)
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    tau_e = float(model.params["tau_exc"])

    baseline_continuous = _all_transfer_power(
        model.J, [input_idx, output_idx], frequencies
    )[0, 1]
    baseline_theory, baseline_theory_valid = _observed_values(
        baseline_continuous, frequencies, z
    )
    baseline_discrete = np.abs(
        _discrete_transfer(model.J, input_idx, output_idx, n_steps, tau_e)
    ) ** 2
    baseline_cal, baseline_cal_valid, baseline_hold, baseline_hold_valid = _network_features(
        baseline_discrete[frequency_ids], frequencies, calibration_input, heldout_input
    )

    nodes = [area for area in model.areas if area not in (INPUT_AREA, OUTPUT_AREA)]
    n_nodes = len(nodes)
    shape = (n_nodes, EXP3_TRIALS, 2)
    deleted_theory = np.full(shape, np.nan, dtype=np.float32)
    deleted_cal = np.full(shape, np.nan, dtype=np.float32)
    deleted_hold = np.full(shape, np.nan, dtype=np.float32)
    deleted_theory_valid = np.zeros((n_nodes, EXP3_TRIALS), dtype=bool)
    deleted_cal_valid = np.zeros((n_nodes, EXP3_TRIALS), dtype=bool)
    deleted_hold_valid = np.zeros((n_nodes, EXP3_TRIALS), dtype=bool)
    deleted_transfer_power = np.empty((n_nodes, len(frequencies)), dtype=np.float32)
    rows: list[dict] = []

    for node_pos, node in enumerate(nodes):
        deleted_j, _ = _node_deleted_j(model, [node])
        continuous = _all_transfer_power(
            deleted_j, [input_idx, output_idx], frequencies
        )[0, 1]
        theory, theory_valid = _observed_values(continuous, frequencies, z)
        discrete = np.abs(
            _discrete_transfer(deleted_j, input_idx, output_idx, n_steps, tau_e)
        ) ** 2
        transfer_power = discrete[frequency_ids].astype(np.float32)
        calibration, calibration_valid, heldout, heldout_valid = _network_features(
            transfer_power, frequencies, calibration_input, heldout_input
        )

        use = (
            baseline_theory_valid
            & theory_valid
            & baseline_cal_valid
            & calibration_valid
            & baseline_hold_valid
            & heldout_valid
            & np.isfinite(baseline_theory).all(axis=1)
            & np.isfinite(theory).all(axis=1)
        )
        if use.sum() < 300:
            raise RuntimeError(f"Only {use.sum()} common valid trials for node {node}")

        theory_baseline_r = _full_error_prediction(
            baseline_theory[use], baseline_cal[use]
        )
        theory_deleted_r = _full_error_prediction(theory[use], calibration[use])
        simulation_baseline_r = _corr(baseline_hold[use])
        simulation_deleted_r = _corr(heldout[use])
        rows.append(
            {
                "deleted_node": node,
                "n_valid": int(use.sum()),
                "theory_baseline_r": theory_baseline_r,
                "theory_deleted_r": theory_deleted_r,
                "theory_delta_r": theory_deleted_r - theory_baseline_r,
                "simulation_baseline_r": simulation_baseline_r,
                "simulation_deleted_r": simulation_deleted_r,
                "simulation_delta_r": simulation_deleted_r - simulation_baseline_r,
            }
        )

        deleted_theory[node_pos] = theory
        deleted_cal[node_pos] = calibration
        deleted_hold[node_pos] = heldout
        deleted_theory_valid[node_pos] = theory_valid
        deleted_cal_valid[node_pos] = calibration_valid
        deleted_hold_valid[node_pos] = heldout_valid
        deleted_transfer_power[node_pos] = transfer_power

        pd.DataFrame(rows).to_csv(result_path, index=False)
        print(f"single-node long-window simulation {node_pos + 1}/{n_nodes}: {node}", flush=True)

    result = pd.DataFrame(rows)
    np.savez_compressed(
        feature_path,
        nodes=np.asarray(nodes),
        frequencies=frequencies,
        frequency_ids=frequency_ids,
        z=z,
        u=u,
        baseline_theory=baseline_theory,
        baseline_theory_valid=baseline_theory_valid,
        baseline_calibration=baseline_cal,
        baseline_calibration_valid=baseline_cal_valid,
        baseline_heldout=baseline_hold,
        baseline_heldout_valid=baseline_hold_valid,
        baseline_transfer_power=baseline_discrete[frequency_ids],
        deleted_theory=deleted_theory,
        deleted_theory_valid=deleted_theory_valid,
        deleted_calibration=deleted_cal,
        deleted_calibration_valid=deleted_cal_valid,
        deleted_heldout=deleted_hold,
        deleted_heldout_valid=deleted_hold_valid,
        deleted_transfer_power=deleted_transfer_power,
    )
    return result


def _identity_r2(x: np.ndarray, y: np.ndarray) -> float:
    denominator = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - x) ** 2) / denominator)


def plot_fit(results: pd.DataFrame) -> dict:
    x = results.theory_delta_r.to_numpy(dtype=float)
    y = results.simulation_delta_r.to_numpy(dtype=float)
    fit = stats.linregress(x, y)
    pearson = stats.pearsonr(x, y)
    identity_r2 = _identity_r2(x, y)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = max((hi - lo) * 0.10, 0.002)
    lo -= pad
    hi += pad
    grid = np.linspace(lo, hi, 200)

    fig, ax = plt.subplots(figsize=(7.2, 6.3), constrained_layout=True)
    ax.scatter(
        x,
        y,
        s=92,
        color="#247BA0",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.92,
        zorder=3,
    )
    ax.plot([lo, hi], [lo, hi], color="#D1495B", ls="--", lw=3.2, label="Identity")
    ax.plot(
        grid,
        fit.intercept + fit.slope * grid,
        color="#202020",
        lw=4.0,
        label="Linear fit",
    )
    ax.axhline(0, color="0.55", lw=1.6, zorder=0)
    ax.axvline(0, color="0.55", lw=1.6, zorder=0)
    ax.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel=r"Theory $\Delta r$",
        ylabel=r"Simulation $\Delta r$",
    )
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    ax.tick_params(length=7, width=2.2)
    ax.text(
        0.04,
        0.96,
        (
            f"n={len(results)} nodes\n"
            f"Pearson r={format_correlation(pearson.statistic)}\n"
            f"OLS R$^2$={fit.rvalue ** 2:.3f}\n"
            f"Identity R$^2$={identity_r2:.3f}\n"
            f"p={format_p_value(pearson.pvalue)}"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=16,
    )
    ax.legend(frameon=False, loc="lower right", fontsize=20)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)
    _save(fig, OUTPUT / "single_node_theory_delta_vs_simulation_delta")

    robustness = {}
    for name, subset in (
        ("excluding_largest_effect", results.drop(results.theory_delta_r.abs().idxmax())),
        ("abs_theory_delta_le_0p01", results[results.theory_delta_r.abs() <= 0.01]),
    ):
        sx = subset.theory_delta_r.to_numpy(dtype=float)
        sy = subset.simulation_delta_r.to_numpy(dtype=float)
        sfit = stats.linregress(sx, sy)
        robustness[name] = {
            "n_nodes": int(len(subset)),
            "pearson_r": float(sfit.rvalue),
            "ols_R2": float(sfit.rvalue**2),
            "identity_R2": _identity_r2(sx, sy),
            "ols_slope": float(sfit.slope),
        }
    metrics = {
        "n_nodes": int(len(results)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "ols_slope": float(fit.slope),
        "ols_intercept": float(fit.intercept),
        "ols_R2": float(fit.rvalue**2),
        "identity_R2": identity_r2,
        "mae": float(np.mean(np.abs(y - x))),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "robustness": robustness,
    }
    (OUTPUT / "fit_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    _style()
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 24,
            "axes.linewidth": 2.4,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 20,
            "lines.linewidth": 3.2,
        }
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml")
    model = load_model(ROOT / cfg["model_dir"])
    selected = json.loads((SOURCE / "selected_single_node_observation_matched.json").read_text())
    results = run_all_nodes(model, selected)
    metrics = plot_fit(results)
    validation = {}
    previous_path = SOURCE / "single_node_observation_selected_longwindow_results.csv"
    if previous_path.exists() and (results.deleted_node == "V1").any():
        previous = pd.read_csv(previous_path).set_index("network")
        v1 = results.loc[results.deleted_node == "V1"].iloc[0]
        validation = {
            "reference_node": "V1",
            "theory_delta_difference_from_previous": float(
                v1.theory_delta_r - previous.loc["delta", "theory_plus_realization_r"]
            ),
            "simulation_delta_difference_from_previous": float(
                v1.simulation_delta_r - previous.loc["delta", "simulation_r"]
            ),
        }
    summary = {
        "setup": {
            "input_area": INPUT_AREA,
            "output_area": OUTPUT_AREA,
            "excluded_nodes": [INPUT_AREA, OUTPUT_AREA],
            "analysis_s": EXP3_ANALYSIS_S,
            "dt_ms": DT_MS,
            "n_input_conditions": EXP3_TRIALS,
            "calibration_realizations": CALIBRATION_PHASES,
            "heldout_realizations": HELDOUT_PHASES,
            "coarse_bin_count": COARSE_BIN_COUNT,
            "input_design": selected,
        },
        "metrics": metrics,
        "validation": validation,
        "files": {
            "results": str((OUTPUT / "all_single_node_deletion_results.csv").resolve()),
            "features": str((OUTPUT / "all_single_node_deletion_features.npz").resolve()),
            "shared_inputs": str((OUTPUT / "shared_longwindow_input_psd_means.npz").resolve()),
            "figure": str((OUTPUT / "single_node_theory_delta_vs_simulation_delta.png").resolve()),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
