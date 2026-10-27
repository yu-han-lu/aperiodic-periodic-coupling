from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        return lambda fun: fun

    prange = range

from .core import AREAS9, input_psd, load_model
from .exp1 import Z0, ZHI, ZLO, ZSCALE
from .io import load_config
from .nonlinear_network_validation import _fourier_inputs, _simulation_arrays
from .paper_figure_dense_revision import SEED as DENSE_SEED, _sample_exact_correlation
from .two_dimensional_ode_validation import (
    MODELS,
    SEED as ODE_SEED,
    _exp1_trials,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "nonlinear_resimulations"

# These values intentionally retain the sampling rate, burn-in, Welch window,
# and feature bands used in the source figures.  The shorter 16.384-s analysis
# window makes exhaustive nonlinear integration tractable while retaining four
# non-overlapping 4.096-s Welch segments (seven with 50% overlap).
DT_S = 0.00025
DT_MS = 0.25
BURN_S = 2.048
ANALYSIS_S = 16.384
WELCH_SEGMENT_S = 4.096
SMOOTHING_SIGMA_HZ = 0.9765625

ODE_N_CONDITIONS = 201
ODE_N_TRIALS = 64
ODE_FORCING_RMS = 0.02

WB_C_N_CONDITIONS = 101
WB_N_TRIALS = 64
WB_INPUT_RMS = 0.5

SEED = 20260823


@njit(parallel=True)
def _hopf_heun_batch(
    forcing: np.ndarray,
    mu_hz: float,
    omega_hz: float,
    dt: float,
    burn_steps: int,
) -> np.ndarray:
    """Integrate the full cubic Hopf normal form, one trajectory per row."""
    n_trials, n_steps = forcing.shape
    result = np.empty((n_trials, n_steps - burn_steps), dtype=np.float32)
    mu = 2.0 * np.pi * mu_hz
    omega = 2.0 * np.pi * omega_hz
    for trial in prange(n_trials):
        x = 0.0
        y = 0.0
        out_pos = 0
        for step in range(n_steps):
            u = forcing[trial, step]
            r2 = x * x + y * y
            dx1 = mu * x - omega * y - r2 * x + u
            dy1 = omega * x + mu * y - r2 * y
            px = x + dt * dx1
            py = y + dt * dy1
            pr2 = px * px + py * py
            dx2 = mu * px - omega * py - pr2 * px + u
            dy2 = omega * px + mu * py - pr2 * py
            x += 0.5 * dt * (dx1 + dx2)
            y += 0.5 * dt * (dy1 + dy2)
            if step >= burn_steps:
                result[trial, out_pos] = x
                out_pos += 1
    return result


@njit(parallel=True)
def _threshold_network_euler_batch(
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
    dt_ms: float,
    tau_e_ms: float,
    tau_i_ms: float,
    burn_steps: int,
    output_indices: np.ndarray,
    input_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate the 110-state threshold-nonlinear whole-brain ODE.

    The positive-part nonlinearity is evaluated at every step.  Background
    drives are supplied explicitly so node deletion can keep the baseline
    external drive fixed rather than silently homeostatically re-centering the
    network after connections are removed.
    """
    n_trials, n_steps = inputs.shape
    n_area = len(local_ee)
    n_outputs = len(output_indices)
    output = np.empty((n_trials, n_outputs, n_steps - burn_steps), dtype=np.float32)
    rectification = np.zeros(n_trials, dtype=np.int64)
    for trial in prange(n_trials):
        rate_e = np.full(n_area, 10.0)
        rate_i = np.full(n_area, 35.0)
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
                    output[trial, output_pos, out_pos] = rate_e[output_indices[output_pos]] - 10.0
                out_pos += 1
        rectification[trial] = rectified
    return output, rectification


def _corr(values: np.ndarray, valid: np.ndarray | None = None) -> float:
    values = np.asarray(values, float)
    use = np.isfinite(values).all(axis=1)
    if valid is not None:
        use &= np.asarray(valid, bool)
    if use.sum() < 8 or np.min(np.std(values[use], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[use, 0], values[use, 1]).statistic)


def _welch_features(
    paths: np.ndarray,
    *,
    peak_band: tuple[float, float],
    dt_s: float = DT_S,
    exponent: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fs = 1.0 / dt_s
    nperseg = int(round(WELCH_SEGMENT_S * fs))
    frequencies, psd = signal.welch(
        paths,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    keep = (frequencies >= 0.5) & (frequencies <= 1200.0)
    f = frequencies[keep]
    sigma_bins = SMOOTHING_SIGMA_HZ / (f[1] - f[0])
    logp = ndimage.gaussian_filter1d(
        # PSD is float32 here; use a representable floor so silent/deleted
        # outputs remain finite rather than underflowing before the log.
        np.log(np.maximum(psd[:, keep], 1e-30)), sigma_bins, axis=-1
    )
    peak_use = (f >= peak_band[0]) & (f <= peak_band[1])
    peak_ids = np.flatnonzero(peak_use)
    band = logp[:, peak_use]
    local_maxima = (band[:, 1:-1] > band[:, :-2]) & (band[:, 1:-1] > band[:, 2:])
    scores = np.where(local_maxima, band[:, 1:-1], -np.inf)
    local = np.argmax(scores, axis=1) + 1
    ids = peak_ids[local]
    valid = local_maxima.any(axis=1)
    rows = np.arange(len(paths))
    y0 = logp[rows, np.maximum(ids - 1, 0)]
    y1 = logp[rows, ids]
    y2 = logp[rows, np.minimum(ids + 1, len(f) - 1)]
    denom = y0 - 2.0 * y1 + y2
    offset = np.divide(
        0.5 * (y0 - y2), denom,
        out=np.zeros_like(denom), where=np.abs(denom) > 1e-12,
    )
    peak = f[ids] + np.clip(offset, -1.0, 1.0) * (f[1] - f[0])
    slope_use = (f >= 100.0) & (f <= 1000.0)
    x = np.log(f[slope_use])
    xc = x - x.mean()
    slopes = np.sum(logp[:, slope_use] * xc[None, :], axis=1) / np.sum(xc * xc)
    values = np.column_stack([peak, -slopes if exponent else slopes])
    valid &= np.isfinite(values).all(axis=1)
    return values, valid


def _colored_forcing(
    z: np.ndarray,
    *,
    rms: float,
    seed: int,
    aperiodic_only: bool = False,
    floor_ratio: float = 0.0,
) -> np.ndarray:
    analysis_steps = int(round(ANALYSIS_S / DT_S))
    burn_steps = int(round(BURN_S / DT_S))
    frequency = np.fft.rfftfreq(analysis_steps, DT_S)
    f = np.maximum(frequency, 0.5)
    rng = np.random.default_rng(seed)
    values = np.atleast_2d(z).astype(float)
    result = np.empty((len(values), analysis_steps + burn_steps), dtype=np.float32)
    for trial, row in enumerate(values):
        if aperiodic_only:
            beta = row[0]
            q = (f / 10.0) ** beta
            q /= q.mean()
        else:
            center, bump_amp, beta, ap_amp = row
            q = ap_amp * (f / 10.0) ** beta
            q += bump_amp * np.exp(-0.5 * ((f - center) / 2.0) ** 2)
        phase = rng.uniform(0.0, 2.0 * np.pi, len(frequency))
        spectrum = np.sqrt(np.maximum(q, 0.0)) * np.exp(1j * phase)
        spectrum[0] = 0.0
        if analysis_steps % 2 == 0:
            spectrum[-1] = spectrum[-1].real
        colored = np.fft.irfft(spectrum, n=analysis_steps)
        colored *= rms / max(colored.std(), 1e-15)
        if floor_ratio > 0:
            colored += rms * floor_ratio * rng.normal(size=analysis_steps)
        colored -= colored.mean()
        result[trial, :burn_steps] = colored[-burn_steps:]
        result[trial, burn_steps:] = colored
    return result


def _selected_rows(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    ids = np.unique(np.rint(np.linspace(0, len(frame) - 1, n)).astype(int))
    return frame.iloc[ids].copy()


def run_hopf(force: bool = False, limit: int | None = None) -> Path:
    out = OUTPUT / "nonlinear_2d_hopf"
    out.mkdir(parents=True, exist_ok=True)
    source = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "two_dimensional_ode_validation" / "hopf"
    model = MODELS["hopf"]
    burn_steps = int(round(BURN_S / DT_S))

    exp1_out = out / "exp1_nonlinear_heun.csv"
    source_exp1 = pd.read_csv(source / "exp1_1000_conditions.csv")
    selected_exp1 = _selected_rows(source_exp1, ODE_N_CONDITIONS)
    if limit is not None:
        selected_exp1 = selected_exp1.head(limit)
    previous = pd.read_csv(exp1_out) if exp1_out.exists() and not force else pd.DataFrame()
    done = set(previous.source_condition.astype(int)) if len(previous) else set()
    rows = previous.to_dict("records") if len(previous) else []
    for position, (source_id, item) in enumerate(selected_exp1.iterrows(), start=1):
        if int(source_id) in done:
            continue
        z = _exp1_trials(float(item.target_input_rho), int(source_id))[:ODE_N_TRIALS]
        forcing = _colored_forcing(z, rms=ODE_FORCING_RMS, seed=SEED + 10000 + int(source_id))
        paths = _hopf_heun_batch(forcing, model.exp1_parameter, model.omega_hz, DT_S, burn_steps)
        features, valid = _welch_features(paths, peak_band=(5.0, 100.0))
        rows.append(
            {
                "source_condition": int(source_id),
                "target_input_rho": float(item.target_input_rho),
                "theory_r": float(item.theory_r),
                "simulation_r": _corr(features, valid),
                "n_trials": ODE_N_TRIALS,
                "n_valid": int(valid.sum()),
                "forcing_rms": ODE_FORCING_RMS,
            }
        )
        pd.DataFrame(rows).sort_values("source_condition").to_csv(exp1_out, index=False)
        print(f"Hopf exp1 nonlinear {position}/{len(selected_exp1)}", flush=True)

    exp2_out = out / "exp2_nonlinear_heun.csv"
    source_exp2 = pd.read_csv(source / "exp2_1000_conditions_rlt0p6.csv")
    selected_exp2 = _selected_rows(source_exp2, ODE_N_CONDITIONS)
    if limit is not None:
        selected_exp2 = selected_exp2.head(limit)
    previous = pd.read_csv(exp2_out) if exp2_out.exists() and not force else pd.DataFrame()
    done = set(previous.source_condition.astype(int)) if len(previous) else set()
    rows = previous.to_dict("records") if len(previous) else []
    for position, (source_id, item) in enumerate(selected_exp2.iterrows(), start=1):
        if int(source_id) in done:
            continue
        sigma_beta = float(item.sigma_beta)
        rng = np.random.default_rng(ODE_SEED + 500000 + int(item.condition))
        betas = stats.truncnorm.rvs(
            -0.8 / sigma_beta,
            0.8 / sigma_beta,
            loc=0.0,
            scale=sigma_beta,
            size=ODE_N_TRIALS,
            random_state=rng,
        )
        forcing = _colored_forcing(
            betas[:, None],
            rms=ODE_FORCING_RMS,
            seed=SEED + 20000 + int(source_id),
            aperiodic_only=True,
            floor_ratio=float(item.floor_ratio),
        )
        paths = _hopf_heun_batch(forcing, float(item.mu_hz), model.omega_hz, DT_S, burn_steps)
        features, valid = _welch_features(paths, peak_band=(5.0, 100.0))
        rows.append(
            {
                "source_condition": int(source_id),
                "condition": int(item.condition),
                "mu_hz": float(item.mu_hz),
                "sigma_beta": sigma_beta,
                "floor_ratio": float(item.floor_ratio),
                "residual_corrected_theory_r": float(item.residual_corrected_theory_r),
                "simulation_r": _corr(features, valid),
                "n_trials": ODE_N_TRIALS,
                "n_valid": int(valid.sum()),
                "forcing_rms": ODE_FORCING_RMS,
            }
        )
        pd.DataFrame(rows).sort_values("source_condition").to_csv(exp2_out, index=False)
        print(f"Hopf exp2 nonlinear {position}/{len(selected_exp2)}", flush=True)

    exp2 = pd.read_csv(exp2_out)
    score = (
        (exp2.mu_hz + 2.5).abs() / exp2.mu_hz.std()
        + (exp2.residual_corrected_theory_r - 0.30).abs()
        / exp2.residual_corrected_theory_r.std()
    )
    representative = exp2.loc[score.idxmin()]
    source_item = source_exp2.loc[int(representative.source_condition)]
    sigma_beta = float(source_item.sigma_beta)
    rng = np.random.default_rng(ODE_SEED + 500000 + int(source_item.condition))
    betas = stats.truncnorm.rvs(
        -0.8 / sigma_beta,
        0.8 / sigma_beta,
        loc=0.0,
        scale=sigma_beta,
        size=ODE_N_TRIALS,
        random_state=rng,
    )
    forcing = _colored_forcing(
        betas[:, None],
        rms=ODE_FORCING_RMS,
        seed=SEED + 20000 + int(representative.source_condition),
        aperiodic_only=True,
        floor_ratio=float(source_item.floor_ratio),
    )
    paths = _hopf_heun_batch(
        forcing, float(source_item.mu_hz), model.omega_hz, DT_S, burn_steps
    )
    features, valid = _welch_features(paths, peak_band=(5.0, 100.0))
    representative_trials = pd.DataFrame(
        {
            "trial": np.arange(ODE_N_TRIALS),
            "beta": betas,
            "peak_hz": features[:, 0],
            "aperiodic_slope": features[:, 1],
            "valid": valid,
        }
    )
    representative_trials.to_csv(out / "representative_nonlinear_trials.csv", index=False)
    representative_r = _corr(features, valid)
    exp2.loc[exp2.source_condition == representative.source_condition, "simulation_r"] = representative_r
    exp2.to_csv(exp2_out, index=False)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "system": "full cubic Hopf normal form",
                "integrator": "explicit Heun",
                "dt_s": DT_S,
                "burn_s": BURN_S,
                "analysis_s": ANALYSIS_S,
                "welch_segment_s": WELCH_SEGMENT_S,
                "forcing_rms": ODE_FORCING_RMS,
                "exp1_conditions": int(len(pd.read_csv(exp1_out))),
                "exp2_conditions": int(len(exp2)),
                "trials_per_condition": ODE_N_TRIALS,
                "representative_source_condition": int(representative.source_condition),
                "representative_simulation_r": representative_r,
                "simulation_uses_transfer_filter": False,
            },
            indent=2,
        )
    )
    return out


def _edge_arrays(params: dict, fln: np.ndarray) -> tuple[np.ndarray, ...]:
    targets, sources = np.nonzero(fln)
    weights = fln[targets, sources]
    edge_e = params["beta_exc"] * params["muEE"] * np.asarray(params["exc_scale"])[targets] * weights
    edge_i = params["beta_inh"] * params["muIE"] * np.asarray(params["inh_scale"])[targets] * weights
    return (
        targets.astype(np.int64),
        sources.astype(np.int64),
        edge_e.astype(float),
        edge_i.astype(float),
    )


def _network_run(
    model,
    inputs: np.ndarray,
    output_indices: np.ndarray,
    input_idx: int,
    fln: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = _simulation_arrays(model.params)
    local_ee, local_ei, local_ie, local_ii = arrays[:4]
    background_e, background_i = arrays[-2:]
    edge_target, edge_source, edge_e, edge_i = _edge_arrays(
        model.params, model.fln if fln is None else fln
    )
    return _threshold_network_euler_batch(
        inputs,
        local_ee,
        local_ei,
        local_ie,
        local_ii,
        edge_target,
        edge_source,
        edge_e,
        edge_i,
        background_e,
        background_i,
        DT_MS,
        float(model.params["tau_exc"]),
        float(model.params["tau_inh"]),
        int(round(BURN_S / DT_S)),
        np.asarray(output_indices, dtype=np.int64),
        int(input_idx),
    )


def _deleted_fln(model, names: list[str]) -> np.ndarray:
    fln = model.fln.copy()
    for name in names:
        idx = model.areas.index(name)
        fln[idx, :] = 0.0
        fln[:, idx] = 0.0
    return fln


def run_wholebrain(force: bool = False, limit: int | None = None) -> Path:
    out = OUTPUT / "whole_brain_full55"
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml")
    model = load_model(ROOT / cfg["model_dir"])
    input_idx = model.areas.index("V1")
    output_idx = model.areas.index("PG")

    c_path = out / "exp1_nonlinear_full55.csv"
    dense_source = pd.read_csv(
        ROOT
        / "experiments"
        / "pap_theory_validation"
        / "outputs"
        / "paper_figure_dense_revision"
        / "exp1_dense_conditions.csv"
    )
    selected = _selected_rows(dense_source, WB_C_N_CONDITIONS)
    if limit is not None:
        selected = selected.head(limit)
    previous = pd.read_csv(c_path) if c_path.exists() and not force else pd.DataFrame()
    done = set(previous.source_condition.astype(int)) if len(previous) else set()
    rows = previous.to_dict("records") if len(previous) else []
    for position, (source_id, item) in enumerate(selected.iterrows(), start=1):
        if int(source_id) in done:
            continue
        x, y = _sample_exact_correlation(
            WB_N_TRIALS, float(item.target_input_r), DENSE_SEED + int(source_id) * 17
        )
        u = np.zeros((WB_N_TRIALS, 4))
        u[:, 0] = 0.20 * x
        u[:, 2] = 0.20 * y
        z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
        inputs = _fourier_inputs(
            z,
            DT_MS,
            int(round(ANALYSIS_S / DT_S)),
            int(round(BURN_S / DT_S)),
            WB_INPUT_RMS,
            SEED + 100000 + int(source_id),
        )
        paths, rect = _network_run(model, inputs, np.array([output_idx]), input_idx)
        features, valid = _welch_features(
            paths[:, 0], peak_band=(5.0, 20.0), exponent=True
        )
        rows.append(
            {
                "source_condition": int(source_id),
                "target_input_r": float(item.target_input_r),
                "theory_r": float(item.first_residual_corrected_r),
                "simulation_r": _corr(features, valid),
                "n_trials": WB_N_TRIALS,
                "n_valid": int(valid.sum()),
                "rectification_count": int(rect.sum()),
                "input_rms": WB_INPUT_RMS,
            }
        )
        pd.DataFrame(rows).sort_values("source_condition").to_csv(c_path, index=False)
        print(f"Whole-brain exp1 nonlinear {position}/{len(selected)}", flush=True)

    deletion_path = out / "node_deletion_nonlinear_full55.csv"
    deletion_input_idx = model.areas.index("A6Va")
    deletion_output_idx = model.areas.index("V2")
    source_deletions = pd.read_csv(
        ROOT
        / "experiments"
        / "pap_theory_validation"
        / "outputs"
        / "single_node_deletion_all_nodes"
        / "combined_single_multinode_deletion_results.csv"
    )
    if limit is not None:
        source_deletions = source_deletions.head(limit)
    cache = np.load(
        ROOT
        / "experiments"
        / "pap_theory_validation"
        / "outputs"
        / "single_node_deletion_all_nodes"
        / "all_single_node_deletion_features.npz"
    )
    z = cache["z"][:WB_N_TRIALS]
    deletion_inputs = _fourier_inputs(
        z,
        DT_MS,
        int(round(ANALYSIS_S / DT_S)),
        int(round(BURN_S / DT_S)),
        WB_INPUT_RMS,
        SEED + 300000,
    )
    baseline_paths, baseline_rect = _network_run(
        model, deletion_inputs, np.array([deletion_output_idx]), deletion_input_idx
    )
    baseline_features, baseline_valid = _welch_features(
        baseline_paths[:, 0], peak_band=(5.0, 20.0), exponent=True
    )
    previous = pd.read_csv(deletion_path) if deletion_path.exists() and not force else pd.DataFrame()
    done = set(previous.row_id.astype(int)) if len(previous) else set()
    rows = previous.to_dict("records") if len(previous) else []
    for position, (row_id, item) in enumerate(source_deletions.iterrows(), start=1):
        if int(row_id) in done:
            continue
        nodes_text = item.deleted_nodes if isinstance(item.deleted_nodes, str) else item.deleted_node
        nodes = str(nodes_text).split("|")
        paths, rect = _network_run(
            model,
            deletion_inputs,
            np.array([deletion_output_idx]),
            deletion_input_idx,
            _deleted_fln(model, nodes),
        )
        features, valid = _welch_features(
            paths[:, 0], peak_band=(5.0, 20.0), exponent=True
        )
        common = baseline_valid & valid
        baseline_r = _corr(baseline_features, common)
        deleted_r = _corr(features, common)
        rows.append(
            {
                "row_id": int(row_id),
                "deleted_nodes": "|".join(nodes),
                "n_deleted": int(item.n_deleted),
                "theory_delta_r": float(item.theory_delta_r),
                "simulation_baseline_r": baseline_r,
                "simulation_deleted_r": deleted_r,
                "simulation_delta_r": deleted_r - baseline_r,
                "n_trials": WB_N_TRIALS,
                "n_valid": int(common.sum()),
                "baseline_rectification_count": int(baseline_rect.sum()),
                "deleted_rectification_count": int(rect.sum()),
                "input_rms": WB_INPUT_RMS,
            }
        )
        pd.DataFrame(rows).sort_values("row_id").to_csv(deletion_path, index=False)
        print(f"Whole-brain deletion nonlinear {position}/{len(source_deletions)}", flush=True)

    matrix_path = out / "exp1_9x9_nonlinear_full55.csv"
    if force or not matrix_path.exists():
        z_matrix = _exp1_trials(-0.55, 5000)[:WB_N_TRIALS]
        matrix = np.full((len(AREAS9), len(AREAS9)), np.nan)
        matrix_valid = np.zeros_like(matrix, dtype=int)
        rectification = np.zeros(len(AREAS9), dtype=np.int64)
        output_indices = np.array([model.areas.index(area) for area in AREAS9])
        for input_pos, input_area in enumerate(AREAS9):
            inputs = _fourier_inputs(
                z_matrix,
                DT_MS,
                int(round(ANALYSIS_S / DT_S)),
                int(round(BURN_S / DT_S)),
                WB_INPUT_RMS,
                SEED + 400000 + input_pos,
            )
            paths, rect = _network_run(
                model, inputs, output_indices, model.areas.index(input_area)
            )
            rectification[input_pos] = int(rect.sum())
            for output_pos in range(len(AREAS9)):
                features, valid = _welch_features(
                    paths[:, output_pos], peak_band=(5.0, 20.0), exponent=True
                )
                matrix[output_pos, input_pos] = _corr(features, valid)
                matrix_valid[output_pos, input_pos] = int(valid.sum())
            print(f"Whole-brain 9x9 nonlinear input {input_pos + 1}/{len(AREAS9)}", flush=True)
        pd.DataFrame(matrix, index=AREAS9, columns=AREAS9).to_csv(matrix_path)
        pd.DataFrame(matrix_valid, index=AREAS9, columns=AREAS9).to_csv(
            out / "exp1_9x9_nonlinear_valid_counts.csv"
        )
        pd.DataFrame({"input_area": AREAS9, "rectification_count": rectification}).to_csv(
            out / "exp1_9x9_nonlinear_rectification.csv", index=False
        )

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "system": "55-area, 110-state threshold-nonlinear E-I whole-brain ODE",
                "integrator": "forward Euler with positive-part nonlinearity evaluated at every step",
                "dt_ms": DT_MS,
                "burn_s": BURN_S,
                "analysis_s": ANALYSIS_S,
                "welch_segment_s": WELCH_SEGMENT_S,
                "input_rms": WB_INPUT_RMS,
                "exp1_conditions": int(len(pd.read_csv(c_path))),
                "node_deletion_patterns": int(len(pd.read_csv(deletion_path))),
                "trials_per_condition": WB_N_TRIALS,
                "node_deletion_external_background": "frozen at baseline; no post-deletion recentering",
                "node_deletion_input_area": "A6Va",
                "node_deletion_readout_area": "V2",
                "simulation_uses_transfer_filter": False,
            },
            indent=2,
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["hopf", "wholebrain", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    started = time.time()
    result = {}
    if args.part in ("hopf", "both"):
        result["hopf"] = str(run_hopf(args.force, args.limit))
    if args.part in ("wholebrain", "both"):
        result["wholebrain"] = str(run_wholebrain(args.force, args.limit))
    result["runtime_s"] = time.time() - started
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
