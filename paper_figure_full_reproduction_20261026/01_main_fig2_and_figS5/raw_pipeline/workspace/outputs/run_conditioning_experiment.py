from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit, prange
from scipy import signal, stats


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT_ROOT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE_ROOT = PROJECT_ROOT / "experiments/pap_theory_validation"
SOURCE_ROOT = PACKAGE_ROOT / "outputs"
SELECTED_JSON = SOURCE_ROOT / "single_node_deletion_revision/selected_single_node_observation_matched.json"
FIXEDPOINT_CSV = WORKSPACE / "outputs/fixedpoint_recalibrated_H/fixedpoint_recalibrated_node_deletion.csv"
REFERENCE_CSV = WORKSPACE / "outputs/experiment1_calibrated_H/experiment1_exact_filter_calibrated_node_deletion.csv"

sys.path.insert(0, str(PACKAGE_ROOT))
sys.path.insert(0, str(WORKSPACE / "outputs/fixedpoint_recalibrated_H"))
sys.path.insert(0, str(WORKSPACE / "outputs/experiment1_calibrated_H"))

from recompute_experiment1_calibrated_H import _cov_to_corr, _pooled_within_trial_cov  # noqa: E402
from recompute_fixedpoint_theory_and_plot import (  # noqa: E402
    _drive_matrix,
    _effective_jacobian,
    _fixed_point,
    _nodes,
)
from pap_validation.core import load_model  # noqa: E402
from pap_validation.exp1 import Z0, ZHI, ZLO, ZSCALE  # noqa: E402
from pap_validation.exp3_dt_convergence import _matched_bandlimited_inputs  # noqa: E402
from pap_validation.exp3_large_delta_screen import _discrete_transfer  # noqa: E402
from pap_validation.exp3_trial_seed_validation import _feature_arrays_vectorized  # noqa: E402
from pap_validation.io import load_config  # noqa: E402
from pap_validation.nonlinear_network_validation import _simulation_arrays  # noqa: E402
from pap_validation.nonlinear_resimulations import _deleted_fln, _edge_arrays, _network_run  # noqa: E402
from pap_validation.paper_figure_node_deletion import (  # noqa: E402
    DT_MS,
    SEED,
    SMOOTHING_SIGMA_HZ,
    WELCH_SEGMENT_S,
    _all_transfer_power,
    _features_from_averaged_psd,
    _observed_values,
)
from pap_validation.single_node_deletion_revision import INPUT_AREA, OUTPUT_AREA  # noqa: E402


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.linewidth": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "lines.linewidth": 1.8,
    }
)

INPUT_SIGMA = 0.5
BURN_S = 2.048
CALIBRATION_PHASES = 3
PILOT_HELDOUT_PHASES = 2
LARGE_HELDOUT_PHASES = 5
PHASE_SEED_ROOT = SEED + 1010000
PHASE_SEED_STRIDE = 100000
EPSILON = 0.3
PILOT_CONDITIONS = (
    None,
    1_000_000.0,
    100_000.0,
    10_000.0,
    1_000.0,
    100.0,
    30.0,
    10.0,
)
PILOT_TRIALS = 64
PILOT_ANALYSIS_S = 16.384
LARGE_TRIALS = 128
LARGE_ANALYSIS_S = 32.768
CONFIRM_TRIALS = 128
CONFIRM_ANALYSIS_S = 32.768
CHUNK = 128


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _tag(condition: float | None) -> str:
    if condition is None:
        return "original"
    value = f"{float(condition):g}".replace(".", "p").replace("-", "m")
    return f"cond_{value}"


def _condition_covariance(sigma: np.ndarray, target_condition: float | None) -> tuple[np.ndarray, dict]:
    sigma = (np.asarray(sigma, float) + np.asarray(sigma, float).T) / 2.0
    values, vectors = np.linalg.eigh(sigma)
    top = float(values[-1])
    if target_condition is None:
        adjusted = np.maximum(values, 0.0)
    else:
        adjusted = np.maximum(values, top / float(target_condition))
    conditioned = (vectors * adjusted) @ vectors.T
    conditioned = (conditioned + conditioned.T) / 2.0
    # Retain the original convention: the largest marginal variance is one.
    conditioned /= float(np.max(np.diag(conditioned)))
    final_values = np.linalg.eigvalsh(conditioned)
    numerical_condition = (
        float("inf")
        if final_values[0] <= np.finfo(float).eps * final_values[-1]
        else float(final_values[-1] / final_values[0])
    )
    positive = final_values[final_values > np.finfo(float).eps * final_values[-1]]
    effective_condition = float(final_values[-1] / positive[0])
    return conditioned, {
        "requested_condition": None if target_condition is None else float(target_condition),
        "numerical_condition": numerical_condition,
        "effective_condition": effective_condition,
        "eigenvalues": final_values.tolist(),
        "trace": float(np.trace(conditioned)),
        "diagonal": np.diag(conditioned).tolist(),
    }


def _design(sigma: np.ndarray, n_trials: int, seed: int) -> tuple[np.ndarray, np.ndarray, float]:
    values, vectors = np.linalg.eigh((sigma + sigma.T) / 2.0)
    root = vectors @ np.diag(np.sqrt(np.maximum(values, 0.0))) @ vectors.T
    normals = np.random.default_rng(seed).normal(size=(n_trials, 4))
    u = normals @ root.T * EPSILON
    raw = Z0 + u * ZSCALE
    z = np.clip(raw, ZLO, ZHI)
    clipped = float(np.mean(np.abs(z - raw) > 1e-12))
    return z, u, clipped


def _spec(analysis_s: float) -> dict:
    return {
        "long": {
            "analysis_s": analysis_s,
            "spectral": {
                "mode": "hann_welch",
                "segment_s": WELCH_SEGMENT_S,
                "smoothing_sigma_hz": SMOOTHING_SIGMA_HZ,
            },
        }
    }


def _feature_from_paths(paths: np.ndarray, analysis_s: float) -> tuple[np.ndarray, np.ndarray]:
    values, valid = _feature_arrays_vectorized(paths, DT_MS, _spec(analysis_s))["long"]
    return values[0], valid[0]


def _corr(values: np.ndarray, use: np.ndarray) -> float:
    use = np.asarray(use, bool) & np.isfinite(values).all(axis=1)
    if use.sum() < 8 or np.min(np.std(values[use], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[use, 0], values[use, 1]).statistic)


def _identity_r2(predicted: np.ndarray, observed: np.ndarray) -> float:
    denominator = float(np.sum((observed - observed.mean()) ** 2))
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator) if denominator > 0 else np.nan


def _fit_metrics(frame: pd.DataFrame) -> dict:
    use = np.isfinite(frame.theory_delta_r) & np.isfinite(frame.simulation_delta_r)
    x = frame.loc[use, "theory_delta_r"].to_numpy(float)
    y = frame.loc[use, "simulation_delta_r"].to_numpy(float)
    fit = stats.linregress(x, y) if len(x) >= 3 and np.std(x) > 0 else None
    return {
        "n": int(len(x)),
        "pearson_r": float(stats.pearsonr(x, y).statistic) if len(x) >= 3 else np.nan,
        "ols_slope": float(fit.slope) if fit else np.nan,
        "ols_intercept": float(fit.intercept) if fit else np.nan,
        "ols_R2": float(fit.rvalue**2) if fit else np.nan,
        "identity_R2": _identity_r2(x, y),
        "mae": float(np.mean(np.abs(y - x))),
        "rmse": float(np.sqrt(np.mean((y - x) ** 2))),
    }


def _pilot_rows(source: pd.DataFrame, count: int = 12) -> pd.DataFrame:
    reference = pd.read_csv(REFERENCE_CSV)[["row_id", "simulation_delta_r"]]
    available = source.merge(reference, on="row_id", suffixes=("", "_reference"))
    available = available.sort_values("simulation_delta_r_reference").reset_index(drop=True)
    positions = np.unique(np.round(np.linspace(0, len(available) - 1, count)).astype(int))
    return available.iloc[positions].sort_values("row_id").reset_index(drop=True)


def _load_source(mode: str) -> pd.DataFrame:
    source = pd.read_csv(FIXEDPOINT_CSV)
    source = source[
        (source.fixedpoint_status == "ok")
        & source.input_active.astype(bool)
        & source.output_active.astype(bool)
        & np.isfinite(source.simulation_delta_r)
    ].sort_values("row_id").reset_index(drop=True)
    return _pilot_rows(source) if mode in ("pilot", "confirm", "scan", "fpinit") else source


@njit(parallel=True)
def _threshold_network_euler_batch_fixedpoint(
    inputs: np.ndarray,
    local_ee: np.ndarray,
    local_ei: np.ndarray,
    local_ie: np.ndarray,
    local_ii: np.ndarray,
    edge_target: np.ndarray,
    edge_source: np.ndarray,
    edge_e: np.ndarray,
    edge_i: np.ndarray,
    background_e: np.ndarray,
    background_i: np.ndarray,
    initial_e: np.ndarray,
    initial_i: np.ndarray,
    dt_ms: float,
    tau_e_ms: float,
    tau_i_ms: float,
    burn_steps: int,
    output_indices: np.ndarray,
    input_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact threshold-network integration initialized at its own fixed point."""
    n_trials, n_steps = inputs.shape
    n_area = len(local_ee)
    n_outputs = len(output_indices)
    output = np.empty((n_trials, n_outputs, n_steps - burn_steps), dtype=np.float32)
    rectification = np.zeros(n_trials, dtype=np.int64)
    for trial in prange(n_trials):
        rate_e = initial_e.copy()
        rate_i = initial_i.copy()
        next_e = np.empty(n_area)
        next_i = np.empty(n_area)
        long_e = np.empty(n_area)
        long_i = np.empty(n_area)
        out_pos = 0
        rectified = 0
        for step in range(1, n_steps):
            long_e[:] = 0.0
            long_i[:] = 0.0
            for edge in range(len(edge_target)):
                target = edge_target[edge]
                source = edge_source[edge]
                source_rate = rate_e[source]
                long_e[target] += edge_e[edge] * source_rate
                long_i[target] += edge_i[edge] * source_rate
            for area in range(n_area):
                drive_e = (
                    local_ee[area] * rate_e[area]
                    + local_ei[area] * rate_i[area]
                    + long_e[area]
                    + background_e[area]
                )
                if area == input_idx:
                    drive_e += inputs[trial, step]
                drive_i = (
                    local_ie[area] * rate_e[area]
                    + local_ii[area] * rate_i[area]
                    + long_i[area]
                    + background_i[area]
                )
                if drive_e < 0.0:
                    drive_e = 0.0
                    rectified += 1
                if drive_i < 0.0:
                    drive_i = 0.0
                    rectified += 1
                next_e[area] = rate_e[area] + (-rate_e[area] + drive_e) * dt_ms / tau_e_ms
                next_i[area] = rate_i[area] + (-rate_i[area] + drive_i) * dt_ms / tau_i_ms
            rate_e, next_e = next_e, rate_e
            rate_i, next_i = next_i, rate_i
            if step >= burn_steps:
                for output_pos in range(n_outputs):
                    output[trial, output_pos, out_pos] = rate_e[output_indices[output_pos]] - initial_e[output_indices[output_pos]]
                out_pos += 1
        rectification[trial] = rectified
    return output, rectification


def _network_run_fixedpoint(
    model, inputs: np.ndarray, output_indices: np.ndarray, input_idx: int,
    fixedpoint: np.ndarray, fln: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _simulation_arrays(model.params)
    local_ee, local_ei, local_ie, local_ii = arrays[:4]
    background_e, background_i = arrays[-2:]
    edge_target, edge_source, edge_e, edge_i = _edge_arrays(
        model.params, model.fln if fln is None else fln
    )
    n = len(model.areas)
    return _threshold_network_euler_batch_fixedpoint(
        inputs, local_ee, local_ei, local_ie, local_ii,
        edge_target, edge_source, edge_e, edge_i,
        background_e, background_i, fixedpoint[:n], fixedpoint[n:],
        DT_MS, float(model.params["tau_exc"]), float(model.params["tau_inh"]),
        int(round(BURN_S * 1000.0 / DT_MS)), output_indices, input_idx,
    )


def _network_maps(model, source: pd.DataFrame, z: np.ndarray, analysis_s: float) -> dict:
    n = len(model.areas)
    tau = np.r_[
        np.full(n, float(model.params["tau_exc"])),
        np.full(n, float(model.params["tau_inh"])),
    ]
    initial = np.r_[np.full(n, 10.0), np.full(n, 35.0)]
    simulation_arrays = _simulation_arrays(model.params)
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    fs = 1000.0 / DT_MS
    nperseg = int(round(WELCH_SEGMENT_S * fs))
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    keep = (frequencies >= 0.5) & (frequencies <= 1200.0)
    frequencies = frequencies[keep]
    baseline_matrix, baseline_background = _drive_matrix(
        model, model.fln, simulation_arrays, _edge_arrays
    )
    baseline_fixedpoint, _, baseline_gain, _, baseline_residual = _fixed_point(
        baseline_matrix, baseline_background, tau, initial
    )
    matrices = [_effective_jacobian(baseline_matrix, baseline_gain, tau)]
    flns: list[np.ndarray | None] = [None]
    fixedpoints = [baseline_fixedpoint]
    fixedpoint_residual = [float(baseline_residual)]
    active_e = [int(baseline_gain[:n].sum())]
    active_i = [int(baseline_gain[n:].sum())]
    for item in source.itertuples(index=False):
        row = pd.Series(item._asdict())
        fln = _deleted_fln(model, _nodes(row))
        matrix, background = _drive_matrix(model, fln, simulation_arrays, _edge_arrays)
        fixedpoint, _, gain, _, residual = _fixed_point(matrix, background, tau, initial)
        matrices.append(_effective_jacobian(matrix, gain, tau))
        flns.append(fln)
        fixedpoints.append(fixedpoint)
        fixedpoint_residual.append(float(residual))
        active_e.append(int(gain[:n].sum()))
        active_i.append(int(gain[n:].sum()))
    transfers = np.asarray(
        [_discrete_transfer(j, input_idx, output_idx, n_steps, float(model.params["tau_exc"])) for j in matrices],
        dtype=np.complex64,
    )
    latent = np.empty((len(matrices), len(z), 2), float)
    latent_valid = np.empty((len(matrices), len(z)), bool)
    for pos, j in enumerate(matrices):
        continuous = _all_transfer_power(j, [input_idx, output_idx], frequencies)[0, 1]
        latent[pos], latent_valid[pos] = _observed_values(continuous, frequencies, z)
    return {
        "transfers": transfers,
        "jacobians": np.asarray(matrices),
        "latent": latent,
        "latent_valid": latent_valid,
        "flns": flns,
        "fixedpoints": np.asarray(fixedpoints),
        "fixedpoint_residual": np.asarray(fixedpoint_residual),
        "active_e": np.asarray(active_e),
        "active_i": np.asarray(active_i),
        "frequencies": frequencies,
        "keep": keep,
    }


def _linear_calibration_features(z: np.ndarray, maps: dict, analysis_s: float, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    expected = (CALIBRATION_PHASES, len(maps["transfers"]), len(z), 2)
    if cache.exists():
        with np.load(cache) as data:
            if tuple(data["features"].shape) == expected:
                return data["features"].astype(float), data["valid"].astype(bool)
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    features = np.full(expected, np.nan, np.float32)
    valid = np.zeros(expected[:-1], bool)
    for phase in range(CALIBRATION_PHASES):
        for start in range(0, len(z), CHUNK):
            stop = min(start + CHUNK, len(z))
            inputs = _matched_bandlimited_inputs(
                z[start:stop], DT_MS, analysis_s, BURN_S, INPUT_SIGMA,
                PHASE_SEED_ROOT + phase * PHASE_SEED_STRIDE, trial_offset=start,
            )
            input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
            for network in range(len(maps["transfers"])):
                output = np.fft.irfft(input_fft * maps["transfers"][network][None, :], n=n_steps, axis=-1)
                values, is_valid = _feature_from_paths(output[:, None, :], analysis_s)
                features[phase, network, start:stop] = values
                valid[phase, network, start:stop] = is_valid
        print(f"linear calibration phase {phase + 1}/{CALIBRATION_PHASES}", flush=True)
    _atomic_npz(cache, features=features, valid=valid)
    return features.astype(float), valid


def _nonlinear_heldout_features(
    model, z: np.ndarray, maps: dict, analysis_s: float, heldout_phases: int, cache_dir: Path,
    fixedpoint_init: bool = False, network_indices: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies = maps["frequencies"]
    feature = np.full((len(maps["flns"]), len(z), 2), np.nan, float)
    valid = np.zeros((len(maps["flns"]), len(z)), bool)
    rectification = np.zeros(len(maps["flns"]), np.int64)
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    fs = 1000.0 / DT_MS
    nperseg = int(round(WELCH_SEGMENT_S * fs))
    indices = list(range(len(maps["flns"]))) if network_indices is None else list(network_indices)
    for network in indices:
        fln = maps["flns"][network]
        path = cache_dir / f"network_{network:03d}.npz"
        if path.exists():
            with np.load(path) as data:
                psd_sum = data["psd_sum"].astype(float)
                completed = data["completed"].astype(bool)
                rect = data["rectification"].astype(np.int64)
        else:
            psd_sum = np.zeros((len(z), len(frequencies)), np.float64)
            completed = np.zeros(heldout_phases, bool)
            rect = np.zeros(heldout_phases, np.int64)
        for phase in range(heldout_phases):
            if completed[phase]:
                continue
            phase_psd = np.empty_like(psd_sum)
            phase_rect = 0
            for start in range(0, len(z), CHUNK):
                stop = min(start + CHUNK, len(z))
                inputs = _matched_bandlimited_inputs(
                    z[start:stop], DT_MS, analysis_s, BURN_S, INPUT_SIGMA,
                    PHASE_SEED_ROOT + (CALIBRATION_PHASES + phase) * PHASE_SEED_STRIDE,
                    trial_offset=start,
                )
                if fixedpoint_init:
                    paths, counts = _network_run_fixedpoint(
                        model, inputs, np.asarray([output_idx], np.int64), input_idx,
                        maps["fixedpoints"][network], fln,
                    )
                else:
                    paths, counts = _network_run(
                        model, inputs, np.asarray([output_idx], np.int64), input_idx, fln
                    )
                actual_f, power = signal.welch(
                    paths[:, 0], fs=fs, window="hann", nperseg=nperseg,
                    noverlap=nperseg // 2, detrend="constant", scaling="density", axis=-1,
                )
                if not np.array_equal(actual_f[maps["keep"]], frequencies):
                    raise RuntimeError("Welch frequency grid changed")
                phase_psd[start:stop] = power[:, maps["keep"]]
                phase_rect += int(counts.sum())
            psd_sum += phase_psd
            rect[phase] = phase_rect
            completed[phase] = True
            _atomic_npz(path, psd_sum=psd_sum, completed=completed, rectification=rect)
            print(f"nonlinear network {network + 1}/{len(maps['flns'])}, heldout {phase + 1}/{heldout_phases}", flush=True)
        values, is_valid = _features_from_averaged_psd(
            frequencies, (psd_sum / heldout_phases)[None, :, :]
        )
        feature[network] = values[0]
        valid[network] = is_valid[0]
        rectification[network] = int(rect.sum())
    return feature, valid, rectification


def _assemble(
    source: pd.DataFrame, maps: dict, linear: np.ndarray, linear_valid: np.ndarray,
    nonlinear: np.ndarray, nonlinear_valid: np.ndarray, rectification: np.ndarray,
    heldout_phases: int, n_trials: int, analysis_s: float,
) -> pd.DataFrame:
    rows = []
    total_steps = int(round((analysis_s + BURN_S) * 1000.0 / DT_MS))
    normalizer = heldout_phases * n_trials * total_steps * 2 * 55
    for position, item in enumerate(source.itertuples(index=False), 1):
        common = (
            maps["latent_valid"][0] & maps["latent_valid"][position]
            & nonlinear_valid[0] & nonlinear_valid[position]
            & np.all(linear_valid[:, 0], axis=0) & np.all(linear_valid[:, position], axis=0)
        )
        base_residual, _, _ = _pooled_within_trial_cov(linear[:, 0], linear_valid[:, 0], common)
        del_residual, _, _ = _pooled_within_trial_cov(linear[:, position], linear_valid[:, position], common)
        base_signal = np.cov(maps["latent"][0, common], rowvar=False, ddof=1)
        del_signal = np.cov(maps["latent"][position, common], rowvar=False, ddof=1)
        theory_base = _cov_to_corr(base_signal + base_residual / heldout_phases)
        theory_del = _cov_to_corr(del_signal + del_residual / heldout_phases)
        sim_base = _corr(nonlinear[0], common)
        sim_del = _corr(nonlinear[position], common)
        row = item._asdict()
        row.update(
            {
                "theory_baseline_r": theory_base,
                "theory_deleted_r": theory_del,
                "theory_delta_r": theory_del - theory_base,
                "simulation_baseline_r": sim_base,
                "simulation_deleted_r": sim_del,
                "simulation_delta_r": sim_del - sim_base,
                "common_n": int(common.sum()),
                "baseline_rectification_fraction": float(rectification[0] / normalizer),
                "deleted_rectification_fraction": float(rectification[position] / normalizer),
                "active_E_count_recomputed": int(maps["active_e"][position]),
                "active_I_count_recomputed": int(maps["active_i"][position]),
                "fixedpoint_residual_recomputed": float(maps["fixedpoint_residual"][position]),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _run_variant(
    mode: str, condition: float | None, sigma: np.ndarray, covariance_info: dict,
    model, source: pd.DataFrame, n_trials: int, analysis_s: float, heldout_phases: int,
    fixedpoint_init: bool = False,
) -> tuple[pd.DataFrame, dict]:
    tag = _tag(condition)
    out = HERE / mode / tag
    out.mkdir(parents=True, exist_ok=True)
    z, u, clipped = _design(sigma, n_trials, SEED + 990000)
    maps = _network_maps(model, source, z, analysis_s)
    linear, linear_valid = _linear_calibration_features(
        z, maps, analysis_s, out / "linear_calibration_features.npz"
    )
    nonlinear, nonlinear_valid, rectification = _nonlinear_heldout_features(
        model, z, maps, analysis_s, heldout_phases, out / "nonlinear_states", fixedpoint_init
    )
    result = _assemble(
        source, maps, linear, linear_valid, nonlinear, nonlinear_valid,
        rectification, heldout_phases, n_trials, analysis_s,
    )
    result.insert(0, "covariance_tag", tag)
    result.to_csv(out / "pattern_results.csv", index=False)
    metrics = _fit_metrics(result)
    metrics.update(
        {
            "covariance_tag": tag,
            **covariance_info,
            "clipped_parameter_fraction": clipped,
            "n_trials": n_trials,
            "analysis_s": analysis_s,
            "calibration_phases": CALIBRATION_PHASES,
            "heldout_phases": heldout_phases,
            "fixedpoint_initialized_simulation": bool(fixedpoint_init),
            "mean_deleted_rectification_fraction": float(result.deleted_rectification_fraction.mean()),
            "max_deleted_rectification_fraction": float(result.deleted_rectification_fraction.max()),
        }
    )
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.savez_compressed(out / "design.npz", sigma=sigma, z=z, u=u)
    return result, metrics


def _plot_pilot(metrics: pd.DataFrame) -> None:
    shown = metrics.copy()
    shown["plot_condition"] = shown.requested_condition.fillna(1e8)
    shown = shown.sort_values("plot_condition")
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.15), constrained_layout=True)
    specs = [
        ("identity_R2", "Identity $R^2$"),
        ("ols_slope", "OLS slope"),
        ("mae", r"MAE of $\Delta r$"),
        ("mean_deleted_rectification_fraction", "Rectification fraction"),
    ]
    for ax, (column, label) in zip(axes, specs):
        ax.plot(shown.plot_condition, shown[column], "o-", color="#247BA0", ms=4)
        ax.set_xscale("log")
        ax.set_xlabel("Covariance condition no.")
        ax.set_ylabel(label)
        ax.grid(True, color="#dddddd", linewidth=0.6)
    axes[0].axhline(1, color="#888888", ls="--", lw=1)
    axes[1].axhline(1, color="#888888", ls="--", lw=1)
    for suffix, kwargs in {
        ".png": {"dpi": 300}, ".pdf": {}, ".svg": {}, ".tiff": {"dpi": 600}
    }.items():
        fig.savefig(HERE / f"conditioning_pilot_metrics{suffix}", facecolor="white", **kwargs)
    plt.close(fig)


def _plot_large(result: pd.DataFrame, metrics: dict) -> None:
    x = result.theory_delta_r.to_numpy(float)
    y = result.simulation_delta_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y)
    low = float(min(x[use].min(), y[use].min()))
    high = float(max(x[use].max(), y[use].max()))
    pad = 0.04 * max(high - low, 0.1)
    fig, ax = plt.subplots(figsize=(3.35, 3.15), constrained_layout=True)
    colors = {1: "#247BA0", 3: "#F4A261", 4: "#2A9D8F", 5: "#8E5EA2"}
    markers = {1: "o", 3: "^", 4: "s", 5: "D"}
    for count in sorted(result.n_deleted.unique()):
        take = use & (result.n_deleted.to_numpy(int) == count)
        ax.scatter(x[take], y[take], s=24, marker=markers.get(int(count), "o"),
                   color=colors.get(int(count), "#555555"), alpha=0.85, label=f"{int(count)} nodes")
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#444444", ls="--", lw=1.2)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"Theory $\Delta r$")
    ax.set_ylabel(r"Exact nonlinear simulation $\Delta r$")
    ax.legend(frameon=False, loc="best")
    ax.text(0.03, 0.97, f"r = {metrics['pearson_r']:.2f}\nslope = {metrics['ols_slope']:.2f}\nMAE = {metrics['mae']:.3f}",
            transform=ax.transAxes, va="top", ha="left")
    for suffix, kwargs in {
        ".png": {"dpi": 300}, ".pdf": {}, ".svg": {}, ".tiff": {"dpi": 600}
    }.items():
        fig.savefig(HERE / f"conditioned_covariance_large_theory_vs_simulation{suffix}", facecolor="white", **kwargs)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("pilot", "confirm", "scan", "fpinit", "fpinit_large", "large"), default="pilot")
    parser.add_argument("--condition", type=float)
    parser.add_argument("--cache-only-start", type=int)
    parser.add_argument("--cache-only-end", type=int)
    args = parser.parse_args()
    started = time.time()
    selected = json.loads(SELECTED_JSON.read_text())
    original_sigma = np.asarray(selected["sigma"], float)
    cfg = load_config(PACKAGE_ROOT / "configs/full.yaml")
    model = load_model(PROJECT_ROOT / cfg["model_dir"])
    source = _load_source(args.mode)
    if args.mode == "pilot":
        conditions = PILOT_CONDITIONS
        n_trials, analysis_s, heldout_phases = PILOT_TRIALS, PILOT_ANALYSIS_S, PILOT_HELDOUT_PHASES
    elif args.mode == "confirm":
        conditions = (None, 100.0)
        n_trials, analysis_s, heldout_phases = CONFIRM_TRIALS, CONFIRM_ANALYSIS_S, LARGE_HELDOUT_PHASES
    elif args.mode == "scan":
        if args.condition is None:
            raise SystemExit("--condition is required in scan mode")
        conditions = (args.condition,)
        n_trials, analysis_s, heldout_phases = CONFIRM_TRIALS, CONFIRM_ANALYSIS_S, LARGE_HELDOUT_PHASES
    elif args.mode == "fpinit":
        conditions = (args.condition,)
        n_trials, analysis_s, heldout_phases = CONFIRM_TRIALS, CONFIRM_ANALYSIS_S, LARGE_HELDOUT_PHASES
    elif args.mode == "fpinit_large":
        conditions = (args.condition,)
        n_trials, analysis_s, heldout_phases = LARGE_TRIALS, LARGE_ANALYSIS_S, LARGE_HELDOUT_PHASES
    else:
        if args.condition is None:
            raise SystemExit("--condition is required in large mode")
        conditions = (args.condition,)
        n_trials, analysis_s, heldout_phases = LARGE_TRIALS, LARGE_ANALYSIS_S, LARGE_HELDOUT_PHASES
    if args.cache_only_start is not None or args.cache_only_end is not None:
        if args.mode not in ("large", "fpinit_large"):
            raise SystemExit("cache-only workers require --mode large or fpinit_large")
        start = 0 if args.cache_only_start is None else args.cache_only_start
        end = len(source) + 1 if args.cache_only_end is None else args.cache_only_end
        sigma, _ = _condition_covariance(original_sigma, args.condition)
        z, _, _ = _design(sigma, n_trials, SEED + 990000)
        maps = _network_maps(model, source, z, analysis_s)
        indices = list(range(max(0, start), min(len(maps["flns"]), end)))
        _nonlinear_heldout_features(
            model, z, maps, analysis_s, heldout_phases,
            HERE / args.mode / _tag(args.condition) / "nonlinear_states",
            fixedpoint_init=(args.mode == "fpinit_large"), network_indices=indices,
        )
        print(f"cache-only worker complete: {start}:{end}", flush=True)
        return
    all_results = []
    all_metrics = []
    for condition in conditions:
        sigma, covariance_info = _condition_covariance(original_sigma, condition)
        print(f"START {args.mode} {_tag(condition)}", flush=True)
        result, metrics = _run_variant(
            args.mode, condition, sigma, covariance_info, model, source,
            n_trials, analysis_s, heldout_phases,
            fixedpoint_init=(args.mode in ("fpinit", "fpinit_large")),
        )
        all_results.append(result)
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2), flush=True)
    results = pd.concat(all_results, ignore_index=True)
    metrics_frame = pd.DataFrame(all_metrics)
    results.to_csv(HERE / f"{args.mode}_all_pattern_results.csv", index=False)
    metrics_frame.to_csv(HERE / f"{args.mode}_metrics.csv", index=False)
    if args.mode == "pilot":
        original = metrics_frame.loc[metrics_frame.covariance_tag == "original"].iloc[0]
        metrics_frame["rmse_improvement_fraction"] = 1.0 - metrics_frame.rmse / original.rmse
        metrics_frame["mae_improvement_fraction"] = 1.0 - metrics_frame.mae / original.mae
        metrics_frame["slope_distance_improvement_fraction"] = 1.0 - np.abs(metrics_frame.ols_slope - 1) / abs(original.ols_slope - 1)
        metrics_frame["pilot_pass"] = (
            (metrics_frame.rmse_improvement_fraction >= 0.30)
            & (metrics_frame.mae_improvement_fraction > 0)
            & (metrics_frame.slope_distance_improvement_fraction > 0)
            & (metrics_frame.identity_R2 > original.identity_R2)
            & (metrics_frame.n == original.n)
        )
        metrics_frame.to_csv(HERE / "pilot_metrics.csv", index=False)
        _plot_pilot(metrics_frame)
    elif args.mode == "confirm":
        original = metrics_frame.loc[metrics_frame.covariance_tag == "original"].iloc[0]
        metrics_frame["rmse_improvement_fraction"] = 1.0 - metrics_frame.rmse / original.rmse
        metrics_frame["mae_improvement_fraction"] = 1.0 - metrics_frame.mae / original.mae
        metrics_frame["slope_distance_improvement_fraction"] = 1.0 - np.abs(metrics_frame.ols_slope - 1) / abs(original.ols_slope - 1)
        metrics_frame["confirmation_pass"] = (
            (metrics_frame.rmse_improvement_fraction >= 0.30)
            & (metrics_frame.mae_improvement_fraction > 0)
            & (metrics_frame.slope_distance_improvement_fraction > 0)
            & (metrics_frame.identity_R2 > original.identity_R2)
            & (metrics_frame.n == original.n)
        )
        metrics_frame.to_csv(HERE / "confirm_metrics.csv", index=False)
    else:
        _plot_large(results, all_metrics[0])
    summary = {
        "mode": args.mode,
        "elapsed_s": time.time() - started,
        "pattern_count": int(len(source)),
        "n_trials": n_trials,
        "analysis_s": analysis_s,
        "calibration_phases": CALIBRATION_PHASES,
        "heldout_phases": heldout_phases,
        "metrics": all_metrics,
    }
    (HERE / f"{args.mode}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
