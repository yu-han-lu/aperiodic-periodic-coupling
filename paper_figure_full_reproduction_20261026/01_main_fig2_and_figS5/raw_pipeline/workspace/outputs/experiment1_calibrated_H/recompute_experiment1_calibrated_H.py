from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from matplotlib.ticker import FormatStrFormatter


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT_ROOT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE_ROOT = PROJECT_ROOT / "experiments/pap_theory_validation"
SOURCE_ROOT = PACKAGE_ROOT / "outputs"
FIXEDPOINT_RESULTS = (
    WORKSPACE / "outputs/fixedpoint_recalibrated_H/fixedpoint_recalibrated_node_deletion.csv"
)
EXACT_ROOT = SOURCE_ROOT / "nonlinear_resimulations_exact/wholebrain_H"

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(WORKSPACE / "outputs/fixedpoint_recalibrated_H"))

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from recompute_fixedpoint_theory_and_plot import (  # noqa: E402
    _drive_matrix,
    _effective_jacobian,
    _fixed_point,
    _nodes,
)
from pap_validation.core import load_model  # noqa: E402
from pap_validation.io import load_config  # noqa: E402
from pap_validation.nonlinear_h_exact import (  # noqa: E402
    _feature_pair,
    _frequency_grid,
    _load_state,
    _payload,
)
from pap_validation.nonlinear_network_validation import _simulation_arrays  # noqa: E402
from pap_validation.nonlinear_resimulations import _deleted_fln, _edge_arrays  # noqa: E402
from pap_validation.paper_figure_node_deletion import (  # noqa: E402
    CALIBRATION_PHASES,
    DT_MS,
    EXP3_ANALYSIS_S,
    EXP3_TRIALS,
    HELDOUT_PHASES,
    SEED,
    WELCH_SEGMENT_S,
    _all_transfer_power,
    _features_from_averaged_psd,
    _observed_values,
)
from pap_validation.single_node_deletion_revision import INPUT_AREA, OUTPUT_AREA  # noqa: E402
from pap_validation.exp3_dt_convergence import _matched_bandlimited_inputs  # noqa: E402


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 13,
        "axes.labelsize": 18,
        "axes.linewidth": 1.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
        "lines.linewidth": 2.4,
    }
)

INPUT_SIGMA = 0.5
BURN_S = 2.048
PHASE_SEED_ROOT = SEED + 1010000
PHASE_SEED_STRIDE = 100000
CHUNK = 16
FIXED_POINT_TOL = 1e-7

STYLES = {
    1: ("#247BA0", "o", "1 node"),
    3: ("#F4A261", "^", "3 nodes"),
    4: ("#2A9D8F", "s", "4 nodes"),
    5: ("#8E5EA2", "D", "5 nodes"),
    12: ("#D1495B", "*", "12 nodes"),
}


def _corr(values: np.ndarray, use: np.ndarray) -> float:
    use = np.asarray(use, bool) & np.isfinite(values).all(axis=1)
    if use.sum() < 8 or np.min(np.std(values[use], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[use, 0], values[use, 1]).statistic)


def _cov_to_corr(covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, float)
    if (
        covariance.shape != (2, 2)
        or not np.isfinite(covariance).all()
        or covariance[0, 0] <= 0
        or covariance[1, 1] <= 0
    ):
        return np.nan
    return float(
        covariance[0, 1]
        / np.sqrt(covariance[0, 0] * covariance[1, 1])
    )


def _identity_r2(predicted: np.ndarray, observed: np.ndarray) -> float:
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def _metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float | int]:
    predicted = np.asarray(predicted, float)
    observed = np.asarray(observed, float)
    use = np.isfinite(predicted) & np.isfinite(observed)
    x = predicted[use]
    y = observed[use]
    fit = stats.linregress(x, y)
    return {
        "n": int(len(x)),
        "pearson_r": float(stats.pearsonr(x, y).statistic),
        "ols_slope": float(fit.slope),
        "ols_intercept": float(fit.intercept),
        "ols_R2": float(fit.rvalue**2),
        "identity_R2": _identity_r2(x, y),
        "mae": float(np.mean(np.abs(y - x))),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
        "predicted_mean": float(x.mean()),
        "observed_mean": float(y.mean()),
        "predicted_sd": float(x.std(ddof=1)),
        "observed_sd": float(y.std(ddof=1)),
    }


def _input_phase_psds(z: np.ndarray, output_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Generate the real Fourier-phase + Hann-Welch input spectra used for calibration."""
    digest = hashlib.sha256(np.ascontiguousarray(z).view(np.uint8)).hexdigest()
    cache_path = output_dir / "shared_input_phase_welch_psd.npz"
    expected = {
        "z_sha256": digest,
        "analysis_s": EXP3_ANALYSIS_S,
        "dt_ms": DT_MS,
        "burn_s": BURN_S,
        "input_sigma": INPUT_SIGMA,
        "calibration_phases": CALIBRATION_PHASES,
        "phase_seed_root": PHASE_SEED_ROOT,
        "phase_seed_stride": PHASE_SEED_STRIDE,
        "welch_segment_s": WELCH_SEGMENT_S,
    }
    if cache_path.exists():
        with np.load(cache_path) as cached:
            metadata = json.loads(str(cached["metadata"].item()))
            if metadata != expected:
                raise RuntimeError("Input phase-Welch cache metadata changed")
            return cached["frequencies"].astype(float), cached["phase_psd"].astype(float)

    fs = 1000.0 / DT_MS
    nperseg = int(round(WELCH_SEGMENT_S * fs))
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    frequencies, keep = _frequency_grid()
    phase_psd = np.empty(
        (CALIBRATION_PHASES, EXP3_TRIALS, len(frequencies)), dtype=np.float32
    )
    for phase in range(CALIBRATION_PHASES):
        for start in range(0, EXP3_TRIALS, CHUNK):
            stop = min(start + CHUNK, EXP3_TRIALS)
            inputs = _matched_bandlimited_inputs(
                z[start:stop],
                DT_MS,
                EXP3_ANALYSIS_S,
                BURN_S,
                INPUT_SIGMA,
                PHASE_SEED_ROOT + phase * PHASE_SEED_STRIDE,
                trial_offset=start,
            )
            actual_frequencies, power = signal.welch(
                inputs[:, burn_steps:],
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend="constant",
                scaling="density",
                axis=-1,
            )
            if not np.array_equal(actual_frequencies[keep], frequencies):
                raise RuntimeError("Welch frequency grid changed")
            phase_psd[phase, start:stop] = power[:, keep]
        print(
            f"input phase-Welch calibration {phase + 1}/{CALIBRATION_PHASES}",
            flush=True,
        )
    np.savez_compressed(
        cache_path,
        frequencies=frequencies,
        phase_psd=phase_psd,
        metadata=np.asarray(json.dumps(expected, sort_keys=True)),
    )
    return frequencies, phase_psd.astype(float)


def _discrete_transfer_at_frequencies(
    jacobian: np.ndarray,
    input_idx: int,
    output_idx: int,
    frequencies: np.ndarray,
    tau_e: float,
) -> np.ndarray:
    """Forward-Euler transfer sampled on the exact Welch grid."""
    fs = 1000.0 / DT_MS
    theta = 2.0 * np.pi * frequencies / fs
    eigenvalues, vectors = np.linalg.eig(jacobian)
    discrete = 1.0 + DT_MS * eigenvalues
    forcing = np.zeros(len(jacobian))
    forcing[input_idx] = DT_MS / tau_e
    coefficients = np.linalg.solve(vectors, forcing)
    denominator = 1.0 / (
        1.0 - discrete[None, :] * np.exp(-1j * theta[:, None])
    )
    response = denominator @ (vectors[output_idx] * coefficients)
    return np.maximum(np.abs(response) ** 2, np.finfo(float).tiny)


def _phase_features(
    transfer_power: np.ndarray,
    frequencies: np.ndarray,
    input_phase_psd: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    features = np.empty((CALIBRATION_PHASES, EXP3_TRIALS, 2), dtype=float)
    valid = np.empty((CALIBRATION_PHASES, EXP3_TRIALS), dtype=bool)
    for phase in range(CALIBRATION_PHASES):
        values, is_valid = _features_from_averaged_psd(
            frequencies,
            (input_phase_psd[phase] * transfer_power[None, :])[None, :, :],
        )
        features[phase] = values[0]
        valid[phase] = is_valid[0]
    return features, valid


def _pooled_within_trial_cov(
    values: np.ndarray,
    valid: np.ndarray,
    trial_use: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    """Experiment1 estimator: center phase repeats separately within every trial."""
    total = np.zeros((2, 2), dtype=float)
    degrees = 0
    contributing_trials = 0
    for trial in np.flatnonzero(trial_use):
        use = valid[:, trial] & np.isfinite(values[:, trial]).all(axis=1)
        if use.sum() < 2:
            continue
        block = values[use, trial]
        centered = block - block.mean(axis=0, keepdims=True)
        total += centered.T @ centered
        degrees += len(block) - 1
        contributing_trials += 1
    if degrees <= 0:
        return np.full((2, 2), np.nan), contributing_trials, degrees
    return total / degrees, contributing_trials, degrees


def _network_calibration(
    jacobian: np.ndarray,
    frequencies: np.ndarray,
    input_phase_psd: np.ndarray,
    z: np.ndarray,
    input_idx: int,
    output_idx: int,
    tau_e: float,
) -> dict[str, np.ndarray]:
    continuous_power = _all_transfer_power(
        jacobian, [input_idx, output_idx], frequencies
    )[0, 1]
    theory, theory_valid = _observed_values(continuous_power, frequencies, z)
    discrete_power = _discrete_transfer_at_frequencies(
        jacobian, input_idx, output_idx, frequencies, tau_e
    )
    phase_features, phase_valid = _phase_features(
        discrete_power, frequencies, input_phase_psd
    )
    return {
        "theory": theory,
        "theory_valid": theory_valid,
        "phase_features": phase_features,
        "phase_valid": phase_valid,
        "continuous_power": continuous_power,
        "discrete_power": discrete_power,
    }


def _recompute(project_root: Path, output_dir: Path) -> pd.DataFrame:
    cfg = load_config(PACKAGE_ROOT / "configs/full.yaml")
    model = load_model(project_root / cfg["model_dir"])
    source = pd.read_csv(FIXEDPOINT_RESULTS)
    selected = source[
        (source.fixedpoint_status == "ok")
        & source.input_active.astype(bool)
        & source.output_active.astype(bool)
        & np.isfinite(source.simulation_delta_r)
    ].copy()

    result_path = output_dir / "experiment1_calibrated_node_deletion.csv"
    if result_path.exists():
        cached = pd.read_csv(result_path)
        if (
            set(cached.row_id.astype(int)) == set(selected.row_id.astype(int))
            and "exp1_calibrated_delta_r" in cached.columns
        ):
            return cached.sort_values("row_id")

    source_features = SOURCE_ROOT / "single_node_deletion_all_nodes/all_single_node_deletion_features.npz"
    with np.load(source_features) as cached:
        z = cached["z"].astype(float)
    frequencies, input_phase_psd = _input_phase_psds(z, output_dir)

    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    tau_e = float(model.params["tau_exc"])
    baseline = _network_calibration(
        model.J, frequencies, input_phase_psd, z, input_idx, output_idx, tau_e
    )

    fingerprint = str(_payload()["fingerprint"])
    exact_frequencies, _ = _frequency_grid()
    baseline_state = _load_state(
        EXACT_ROOT / "states/baseline_shared.npz", fingerprint, exact_frequencies
    )
    base_exact_cal, base_exact_cal_valid, base_exact_hold, base_exact_hold_valid = (
        _feature_pair(baseline_state)
    )

    n = len(model.areas)
    tau = np.r_[
        np.full(n, float(model.params["tau_exc"])),
        np.full(n, float(model.params["tau_inh"])),
    ]
    initial = np.r_[np.full(n, 10.0), np.full(n, 35.0)]
    simulation_arrays = _simulation_arrays(model.params)

    rows: list[dict] = []
    for count, item in enumerate(selected.sort_values("row_id").itertuples(index=False), 1):
        row_id = int(item.row_id)
        deleted_fln = _deleted_fln(model, _nodes(pd.Series(item._asdict())))
        matrix, background = _drive_matrix(
            model, deleted_fln, simulation_arrays, _edge_arrays
        )
        rate, drive, gain, marginal, fp_residual = _fixed_point(
            matrix, background, tau, initial
        )
        effective_j = _effective_jacobian(matrix, gain, tau)
        deleted = _network_calibration(
            effective_j,
            frequencies,
            input_phase_psd,
            z,
            input_idx,
            output_idx,
            tau_e,
        )

        pattern_state = _load_state(
            EXACT_ROOT / "states" / f"pattern_{row_id:03d}.npz",
            fingerprint,
            exact_frequencies,
        )
        deleted_exact_cal, deleted_exact_cal_valid, deleted_exact_hold, deleted_exact_hold_valid = (
            _feature_pair(pattern_state)
        )
        common = (
            base_exact_cal_valid
            & base_exact_hold_valid
            & deleted_exact_cal_valid
            & deleted_exact_hold_valid
            & baseline["theory_valid"]
            & deleted["theory_valid"]
            & np.isfinite(baseline["theory"]).all(axis=1)
            & np.isfinite(deleted["theory"]).all(axis=1)
        )

        base_residual, base_residual_trials, base_degrees = _pooled_within_trial_cov(
            baseline["phase_features"], baseline["phase_valid"], common
        )
        deleted_residual, deleted_residual_trials, deleted_degrees = _pooled_within_trial_cov(
            deleted["phase_features"], deleted["phase_valid"], common
        )
        # The exact H held-out statistic is extracted after averaging five independent
        # phase PSDs.  Under the same finite-window covariance approximation used by
        # Experiment1, the observation covariance of that average scales as 1/5.
        base_signal_cov = np.cov(baseline["theory"][common], rowvar=False, ddof=1)
        deleted_signal_cov = np.cov(deleted["theory"][common], rowvar=False, ddof=1)
        base_predicted_r = _cov_to_corr(base_signal_cov + base_residual / HELDOUT_PHASES)
        deleted_predicted_r = _cov_to_corr(
            deleted_signal_cov + deleted_residual / HELDOUT_PHASES
        )
        base_simulation_r = _corr(base_exact_hold, common)
        deleted_simulation_r = _corr(deleted_exact_hold, common)

        row = item._asdict()
        row.update(
            {
                "exp1_calibrated_baseline_r": base_predicted_r,
                "exp1_calibrated_deleted_r": deleted_predicted_r,
                "exp1_calibrated_delta_r": deleted_predicted_r - base_predicted_r,
                "common_simulation_baseline_r": base_simulation_r,
                "common_simulation_deleted_r": deleted_simulation_r,
                "common_simulation_delta_r": deleted_simulation_r - base_simulation_r,
                "latent_baseline_r": _cov_to_corr(base_signal_cov),
                "latent_deleted_r": _cov_to_corr(deleted_signal_cov),
                "latent_delta_r": _cov_to_corr(deleted_signal_cov) - _cov_to_corr(base_signal_cov),
                "common_n": int(common.sum()),
                "baseline_residual_var_P_single_phase": float(base_residual[0, 0]),
                "baseline_residual_var_A_single_phase": float(base_residual[1, 1]),
                "baseline_residual_cov_PA_single_phase": float(base_residual[0, 1]),
                "deleted_residual_var_P_single_phase": float(deleted_residual[0, 0]),
                "deleted_residual_var_A_single_phase": float(deleted_residual[1, 1]),
                "deleted_residual_cov_PA_single_phase": float(deleted_residual[0, 1]),
                "baseline_residual_contributing_trials": int(base_residual_trials),
                "deleted_residual_contributing_trials": int(deleted_residual_trials),
                "baseline_residual_degrees": int(base_degrees),
                "deleted_residual_degrees": int(deleted_degrees),
                "fixedpoint_recheck_residual": float(fp_residual),
                "fixedpoint_recheck_input_active": bool(gain[input_idx]),
                "fixedpoint_recheck_output_active": bool(gain[output_idx]),
                "fixedpoint_recheck_marginal_count": int(marginal.sum()),
                "source_simulation_delta_absolute_residual": float(
                    abs((deleted_simulation_r - base_simulation_r) - float(item.simulation_delta_r))
                ),
            }
        )
        rows.append(row)
        pd.DataFrame(rows).to_csv(result_path, index=False)
        print(f"Experiment1 calibration {count}/{len(selected)}: row {row_id}", flush=True)

    result = pd.DataFrame(rows).sort_values("row_id")
    result.to_csv(result_path, index=False)
    return result


def _limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    low = float(min(np.nanmin(x), np.nanmin(y)))
    high = float(max(np.nanmax(x), np.nanmax(y)))
    padding = max((high - low) * 0.10, 0.005)
    return low - padding, high + padding


def _scatter_groups(ax: plt.Axes, data: pd.DataFrame, x_name: str, y_name: str) -> None:
    for n_deleted, (color, marker, label) in STYLES.items():
        subset = data[data.n_deleted == n_deleted]
        ax.scatter(
            subset[x_name],
            subset[y_name],
            s=145 if n_deleted == 12 else 76,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.90,
            label=label,
            zorder=3,
        )


def _save(fig: plt.Figure, stem: Path, *, row_groups=None) -> None:
    kwargs = {} if row_groups is None else {"row_groups": row_groups}
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
        **kwargs,
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_h(data: pd.DataFrame, output_dir: Path) -> dict:
    x = data.exp1_calibrated_delta_r.to_numpy(float)
    # Preserve the previously completed exact nonlinear held-out simulation values.
    y = data.simulation_delta_r.to_numpy(float)
    fit = stats.linregress(x, y)
    pearson = stats.pearsonr(x, y)
    identity_r2 = _identity_r2(x, y)
    low, high = _limits(x, y)
    grid = np.linspace(low, high, 200)

    fig, ax = plt.subplots(figsize=(6.35, 5.45), constrained_layout=True)
    _scatter_groups(ax, data, "exp1_calibrated_delta_r", "simulation_delta_r")
    ax.plot([low, high], [low, high], color="#D1495B", ls="--", lw=3.2, label="Identity")
    fitted = fit.intercept + fit.slope * grid
    margin = 0.025 * (high - low)
    visible = (fitted >= low + margin) & (fitted <= high - margin)
    ax.plot(grid[visible], fitted[visible], color="#202020", lw=3.2, label="Linear fit")
    ax.axhline(0, xmin=0.01, color="0.55", lw=1.4, zorder=0)
    ax.axvline(0, ymin=0.05, color="0.55", lw=1.4, zorder=0)
    ax.set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel=r"Theory $\Delta r$",
        ylabel=r"Simulation $\Delta r$",
    )
    ax.text(
        0.24,
        0.96,
        f"n={len(data)} deletion patterns\n"
        f"Pearson r={pearson.statistic:.3f}\n"
        f"OLS $R^2$={fit.rvalue ** 2:.3f}\n"
        f"Identity $R^2$={identity_r2:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
    )
    max_pos = int(np.argmax(np.maximum(x, y)))
    ax.scatter(
        [x[max_pos]],
        [y[max_pos]],
        s=190,
        facecolor="none",
        edgecolor="#D1495B",
        linewidth=2.4,
        zorder=4,
    )
    ax.annotate(
        rf"max $\Delta r$={max(x[max_pos], y[max_pos]):.3f}",
        (x[max_pos], y[max_pos]),
        xytext=(0.55, 0.82),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.15),
        fontsize=11.5,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    ax.text(-0.15, 1.06, "H", transform=ax.transAxes, fontsize=23, fontweight="bold")
    ax.tick_params(labelsize=12, length=5, width=1.5, pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    _save(fig, output_dir / "H_experiment1_calibrator_theory_vs_exact_simulation")
    return _metrics(x, y)


def _plot_diagnostic(data: pd.DataFrame, output_dir: Path) -> dict:
    comparisons = [
        (
            "exp1_calibrated_baseline_r",
            "simulation_baseline_r",
            "Baseline r",
        ),
        (
            "exp1_calibrated_deleted_r",
            "simulation_deleted_r",
            "Deleted r",
        ),
        (
            "exp1_calibrated_delta_r",
            "simulation_delta_r",
            r"$\Delta r$",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.4, 4.9), constrained_layout=False)
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.22, top=0.88, wspace=0.30)
    output = {}
    for panel, (ax, (x_name, y_name, title)) in enumerate(zip(axes, comparisons)):
        x = data[x_name].to_numpy(float)
        y = data[y_name].to_numpy(float)
        result = _metrics(x, y)
        output[title] = result
        low, high = _limits(x, y)
        grid = np.linspace(low, high, 200)
        fit = stats.linregress(x, y)
        _scatter_groups(ax, data, x_name, y_name)
        ax.plot([low, high], [low, high], color="#D1495B", ls="--", lw=2.5)
        ax.plot(grid, fit.intercept + fit.slope * grid, color="#202020", lw=2.5)
        ax.set(
            xlim=(low, high),
            ylim=(low, high),
            xlabel=f"Calibrated theory {title}",
            ylabel=f"Exact simulation {title}",
            title=title,
        )
        diagnostic_ticks = np.linspace(low, high, 7)[1:-1]
        ax.set_xticks(diagnostic_ticks)
        ax.set_yticks(diagnostic_ticks)
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        ax.text(
            -0.13,
            1.07,
            chr(ord("a") + panel),
            transform=ax.transAxes,
            fontsize=16,
            fontweight="bold",
        )
        ax.tick_params(labelsize=10.5, length=4, width=1.2)
        for spine in ax.spines.values():
            spine.set_linewidth(1.3)
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=5,
        fontsize=10.5,
    )
    _save(fig, output_dir / "experiment1_calibrator_pre_post_delta_diagnostic")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = _recompute(args.project_root.resolve(), output_dir)
    metrics = {
        "method": {
            "calibrator": "Experiment1 pooled within-trial feature residual covariance",
            "input_realization": "matched random Fourier phase followed by Hann-Welch",
            "gamma_psd_used": False,
            "calibration_phases": CALIBRATION_PHASES,
            "heldout_phases": HELDOUT_PHASES,
            "heldout_covariance_scaling": f"single-phase residual covariance / {HELDOUT_PHASES}",
            "fixedpoint": "recomputed separately for every deleted network",
            "selection": "stable fixed point, input A6Va active, output V2 active, finite exact held-out simulation",
        },
        "H_delta": _plot_h(data, output_dir),
        "pre_post_delta": _plot_diagnostic(data, output_dir),
        "rows": int(len(data)),
        "common_n_min": int(data.common_n.min()),
        "common_n_max": int(data.common_n.max()),
        "maximum_source_simulation_delta_absolute_residual": float(
            data.source_simulation_delta_absolute_residual.max()
        ),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
