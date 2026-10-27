from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import set_num_threads
from scipy import ndimage, signal, stats

FIGURE_SKILL_SCRIPTS = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure/scripts"
sys.path.insert(0, str(FIGURE_SKILL_SCRIPTS))
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
sys.path[:0] = [str(PACKAGE), str(WORKSPACE / "outputs/fixedpoint_recalibrated_H")]

from pap_validation.core import feature_from_spectrum, input_psd, load_model  # noqa: E402
from pap_validation.exp1 import ZHI, ZLO  # noqa: E402
from pap_validation.exp2_pap_correlation_validation import (  # noqa: E402
    _forcing as stable_forcing,
    _input_q,
)
from pap_validation.exp3_dt_convergence import _matched_bandlimited_inputs  # noqa: E402
from pap_validation.io import load_config  # noqa: E402
from pap_validation.nonlinear_network_validation import _simulation_arrays  # noqa: E402
from pap_validation.nonlinear_resimulations import _deleted_fln, _edge_arrays  # noqa: E402
from recompute_fixedpoint_theory_and_plot import (  # noqa: E402
    _drive_matrix,
    _effective_jacobian,
    _fixed_point,
)

sys.path.insert(0, str(WORKSPACE / "outputs/covariance_conditioning_pilot"))
from run_conditioning_experiment import (  # noqa: E402
    _network_run_fixedpoint,
    _threshold_network_euler_batch_fixedpoint,
)


DT_MS = 0.25
BURN_S = 2.048
SEGMENT_S = 4.096
SMOOTH_HZ = 0.9765625
SLOPE_BAND = (100.0, 1000.0)
INPUT_SIGMA_DELETION = 0.5
STABLE_N = 72
STABLE_TRIALS = 48
STABLE_ANALYSIS_S = 16.384
STABLE_CAL_PHASES = 2
STABLE_HELD_PHASES = 2
BAND_TRIALS = 48
BAND_ANALYSIS_S = 16.384
BAND_CAL_PHASES = 2
BAND_HELD_PHASES = 2
BAND_CHUNK = 8
SEED = 73421
BANDS = {
    "Alpha": (10.0, (5.0, 20.0)),
    "Beta": (20.0, (13.0, 30.0)),
    "Low gamma": (40.0, (30.0, 55.0)),
    "High gamma": (80.0, (65.0, 100.0)),
}


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def load_wholebrain():
    cfg = load_config(PACKAGE / "configs/full.yaml")
    return load_model(PROJECT / cfg["model_dir"])


def frequencies_for(analysis_s: float) -> tuple[np.ndarray, np.ndarray]:
    fs = 1000.0 / DT_MS
    nperseg = int(round(SEGMENT_S * fs))
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    keep = (frequencies >= 0.5) & (frequencies <= 1200.0)
    return frequencies[keep], keep


def features_from_psd(
    frequencies: np.ndarray,
    psd: np.ndarray,
    peak_band: tuple[float, float],
    relaxed_peak_qc: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    sigma_bins = SMOOTH_HZ / float(frequencies[1] - frequencies[0])
    smooth = np.exp(
        ndimage.gaussian_filter1d(
            np.log(np.maximum(psd, np.finfo(float).tiny)), sigma_bins, axis=-1
        )
    )
    leading = smooth.shape[:-1]
    values = np.full(leading + (2,), np.nan, dtype=float)
    valid = np.zeros(leading, dtype=bool)
    flat_smooth = smooth.reshape((-1, smooth.shape[-1]))
    flat_values = values.reshape((-1, 2))
    flat_valid = valid.reshape(-1)
    for pos, spectrum in enumerate(flat_smooth):
        feature = feature_from_spectrum(
            frequencies, spectrum, peak_band=peak_band, slope_band=SLOPE_BAND
        )
        flat_values[pos] = feature["peak_omega"], feature["exponent"]
        if relaxed_peak_qc:
            flat_valid[pos] = bool(
                not feature["peak_boundary"]
                and "no_interior_peak" not in feature["feature_failure_reason"]
                and np.isfinite(flat_values[pos]).all()
            )
        else:
            flat_valid[pos] = bool(feature["peak_valid"]) and np.isfinite(flat_values[pos]).all()
    return values, valid


def features_from_paths(
    paths: np.ndarray,
    analysis_s: float,
    peak_band: tuple[float, float],
    relaxed_peak_qc: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    fs = 1000.0 / DT_MS
    nperseg = int(round(SEGMENT_S * fs))
    f, psd = signal.welch(
        paths,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    keep = (f >= 0.5) & (f <= 1200.0)
    return features_from_psd(
        f[keep], psd[..., keep], peak_band, relaxed_peak_qc=relaxed_peak_qc
    )


def continuous_transfer_power_all(
    j: np.ndarray, input_idx: int, output_indices: np.ndarray, frequencies: np.ndarray
) -> np.ndarray:
    eigenvalues, vectors = np.linalg.eig(j)
    b = np.zeros(len(j))
    b[input_idx] = 1.0
    coefficients = np.linalg.solve(vectors, b)
    denominators = 1.0 / (
        1j * 2.0 * np.pi * frequencies[:, None] / 1000.0 - eigenvalues[None, :]
    )
    response = np.einsum(
        "fk,ok,k->of", denominators, vectors[np.asarray(output_indices)], coefficients,
        optimize=True,
    )
    return np.maximum(np.abs(response) ** 2, np.finfo(float).tiny)


def discrete_transfer_all(
    j: np.ndarray,
    input_idx: int,
    output_indices: np.ndarray,
    n_steps: int,
    tau_e: float,
) -> np.ndarray:
    fs = 1000.0 / DT_MS
    frequencies = np.fft.rfftfreq(n_steps, d=1.0 / fs)
    theta = 2.0 * np.pi * frequencies / fs
    eigenvalues, vectors = np.linalg.eig(j)
    discrete = 1.0 + DT_MS * eigenvalues
    b = np.zeros(len(j))
    b[input_idx] = DT_MS / tau_e
    coefficients = np.linalg.solve(vectors, b)
    denominator = 1.0 / (1.0 - discrete[None, :] * np.exp(-1j * theta[:, None]))
    return np.einsum(
        "fk,ok,k->of", denominator, vectors[np.asarray(output_indices)], coefficients,
        optimize=True,
    )


def corr_feature(values: np.ndarray, use: np.ndarray) -> float:
    good = np.asarray(use, bool) & np.isfinite(values).all(axis=-1)
    if good.sum() < 8 or np.min(np.std(values[good], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[good, 0], values[good, 1]).statistic)


def corr_from_cov(cov: np.ndarray) -> float:
    return float(cov[0, 1] / np.sqrt(max(cov[0, 0] * cov[1, 1], 1e-30)))


def pooled_phase_cov(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return np.cov(centered.reshape(-1, 2), rowvar=False, ddof=1)


def stable_arrays(model, s_value: float):
    arrays = list(_simulation_arrays(model.params))
    pg = model.areas.index("PG")
    old_ei = float(arrays[1][pg])
    arrays[1] = arrays[1].copy()
    arrays[1][pg] = old_ei * s_value
    # Preserve the registered 10/35-Hz fixed point while changing only the PG
    # local Jacobian block.
    arrays[8] = arrays[8].copy()
    arrays[8][pg] += (old_ei - arrays[1][pg]) * 35.0
    n = len(model.areas)
    matrix = np.zeros((2 * n, 2 * n), dtype=float)
    ids = np.arange(n)
    matrix[ids, ids] = arrays[0]
    matrix[ids, n + ids] = arrays[1]
    matrix[n + ids, ids] = arrays[2]
    matrix[n + ids, n + ids] = arrays[3]
    np.add.at(matrix, (arrays[4], arrays[5]), arrays[6])
    np.add.at(matrix, (n + arrays[4], arrays[5]), arrays[7])
    tau = np.r_[
        np.full(n, float(model.params["tau_exc"])),
        np.full(n, float(model.params["tau_inh"])),
    ]
    j = (matrix - np.eye(2 * n)) / tau[:, None]
    return arrays, j


def run_stable(force: bool = False, pilot: bool = False) -> Path:
    output = HERE / "stable_complex_full55.csv"
    # Always reassemble the condition table from cached phase features so that
    # harmless reporting-layer changes never require rerunning the trajectories.
    model = load_wholebrain()
    old = pd.read_csv(
        PACKAGE / "outputs/exp2_pap_s32_window_131p072/selected_1000_condition_results_rlt0p6.csv"
    )
    count = 12 if pilot else STABLE_N
    ordered = old.sort_values("residual_corrected_r").reset_index(drop=True)
    positions = np.unique(np.round(np.linspace(0, len(ordered) - 1, count)).astype(int))
    selected = ordered.iloc[positions].sort_values("condition_id").reset_index(drop=True)
    n_trials = 32 if pilot else STABLE_TRIALS
    analysis_s = 8.192 if pilot else STABLE_ANALYSIS_S
    cal_phases = 2 if pilot else STABLE_CAL_PHASES
    held_phases = 2 if pilot else STABLE_HELD_PHASES
    # The original two-population experiment drives the excitatory population
    # of PG itself.  Its whole-brain counterpart must therefore inject at PG;
    # V1->PG is the separate Experiment-1 routing condition and suppresses the
    # local PG resonance in the intact network transfer.
    input_idx = model.areas.index("PG")
    output_idx = model.areas.index("PG")
    n = len(model.areas)
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    theory_f = np.geomspace(0.5, 1200.0, 1200)
    rows = []
    cache = HERE / ("stable_pilot_states" if pilot else "stable_states")
    cache.mkdir(exist_ok=True)
    started = time.time()
    for ordinal, condition in selected.iterrows():
        condition_id = int(condition.condition_id)
        state_path = cache / f"condition_{condition_id:04d}.npz"
        s_value = float(condition.S)
        arrays, j = stable_arrays(model, s_value)
        eig = np.linalg.eigvals(j)
        max_real = float(eig.real.max())
        transfer = continuous_transfer_power_all(
            j, input_idx, np.asarray([output_idx]), theory_f
        )[0]
        sigma_beta = float(condition.sigma_beta)
        rng = np.random.default_rng(SEED + 6100000 + int(round(sigma_beta * 1000)))
        lower, upper = -0.8 / sigma_beta, 0.8 / sigma_beta
        betas = stats.truncnorm.rvs(
            lower, upper, loc=0.0, scale=sigma_beta, size=n_trials, random_state=rng
        )
        theory_values = np.empty((n_trials, 2))
        theory_valid = np.zeros(n_trials, bool)
        for trial, beta in enumerate(betas):
            spectrum = transfer * _input_q(
                theory_f, beta, float(condition.input_sigma), float(condition.floor_ratio)
            )
            value, valid = features_from_psd(
                theory_f, spectrum[None, :], (5.0, 100.0), relaxed_peak_qc=True
            )
            theory_values[trial] = value[0]
            theory_valid[trial] = valid[0]
        if state_path.exists() and not force:
            with np.load(state_path) as saved:
                phase_features = saved["phase_features"].astype(float)
                phase_valid = saved["phase_valid"].astype(bool)
                rectification = saved["rectification"].astype(np.int64)
        else:
            phase_features = np.full((cal_phases + held_phases, n_trials, 2), np.nan)
            phase_valid = np.zeros((cal_phases + held_phases, n_trials), bool)
            rectification = np.zeros(cal_phases + held_phases, np.int64)
            for phase in range(cal_phases + held_phases):
                forcing = stable_forcing(
                    betas,
                    float(condition.input_sigma),
                    float(condition.floor_ratio),
                    SEED + 6200000 + condition_id * 1000 + phase * 100000,
                )
                # stable_forcing uses the registered 32.768-s global constant;
                # retain the requested shorter prefix plus its wrapped burn.
                analysis_steps = int(round(analysis_s * 1000.0 / DT_MS))
                forcing = np.concatenate(
                    [forcing[:, -burn_steps:], forcing[:, burn_steps:burn_steps + analysis_steps]],
                    axis=1,
                )
                paths, counts = _threshold_network_euler_batch_fixedpoint(
                    forcing,
                    arrays[0], arrays[1], arrays[2], arrays[3],
                    arrays[4], arrays[5], arrays[6], arrays[7], arrays[8], arrays[9],
                    np.full(n, 10.0), np.full(n, 35.0), DT_MS,
                    float(model.params["tau_exc"]), float(model.params["tau_inh"]),
                    burn_steps, np.asarray([output_idx], np.int64), input_idx,
                )
                value, valid = features_from_paths(
                    paths, analysis_s, (5.0, 100.0), relaxed_peak_qc=True
                )
                phase_features[phase] = value[:, 0]
                phase_valid[phase] = valid[:, 0]
                rectification[phase] = int(counts.sum())
            atomic_npz(
                state_path,
                phase_features=phase_features,
                phase_valid=phase_valid,
                rectification=rectification,
            )
        common = theory_valid & phase_valid.all(axis=0)
        if common.sum() >= 8:
            residual = pooled_phase_cov(phase_features[:cal_phases, common])
            # Once the local block is embedded in the 110-state network, the
            # finite beta range is not well represented by a single local
            # derivative.  Use the exact full-network transfer features on the
            # sampled trials, then add the independently calibrated finite-window
            # covariance.  This is the same signal+observation decomposition used
            # by the corrected deletion workflow.
            signal_cov = np.cov(theory_values[common], rowvar=False, ddof=1)
            exact_mapping_r = corr_from_cov(signal_cov)
            theory_r = corr_from_cov(signal_cov + residual / held_phases)
            held_rs = [
                corr_feature(phase_features[phase], common)
                for phase in range(cal_phases, cal_phases + held_phases)
            ]
            simulation_r = corr_feature(
                phase_features[cal_phases:cal_phases + held_phases].mean(axis=0),
                common,
            )
        else:
            theory_r = simulation_r = np.nan
            exact_mapping_r = np.nan
            held_rs = []
        total_updates = (
            (cal_phases + held_phases) * n_trials
            * (int(round(analysis_s * 1000.0 / DT_MS)) + burn_steps) * 2 * n
        )
        rows.append(
            {
                **condition.to_dict(),
                "full55_theory_r": theory_r,
                "full55_exact_mapping_r": exact_mapping_r,
                "full55_simulation_r": simulation_r,
                "full55_n_valid": int(common.sum()),
                "full55_heldout_r_json": json.dumps(held_rs),
                "full55_rectification_fraction": float(rectification.sum() / total_updates),
                "full55_max_real_eigenvalue_per_ms": max_real,
                "full55_fixedpoint_preserved": True,
                "full55_n_trials": n_trials,
                "full55_analysis_s": analysis_s,
                "full55_calibration_phases": cal_phases,
                "full55_heldout_phases": held_phases,
            }
        )
        print(
            f"stable full55 {ordinal + 1}/{len(selected)} condition={condition_id} "
            f"valid={common.sum()} rect={rows[-1]['full55_rectification_fraction']:.3g}",
            flush=True,
        )
    result = pd.DataFrame(rows)
    target = HERE / ("stable_complex_full55_pilot.csv" if pilot else "stable_complex_full55.csv")
    result.to_csv(target, index=False)
    valid = result[["full55_theory_r", "full55_simulation_r"]].dropna()
    fit = stats.linregress(valid.full55_theory_r, valid.full55_simulation_r)
    metrics = {
        "n_conditions": int(len(valid)),
        "pearson_r": float(fit.rvalue),
        "ols_slope": float(fit.slope),
        "ols_intercept": float(fit.intercept),
        "ols_R2": float(fit.rvalue**2),
        "runtime_s": time.time() - started,
        "simulation": "complete 55-area x E/I threshold-linear/ReLU Euler integration",
        "input_area": "PG",
        "output_area": "PG",
        "fixedpoint": "10/35 Hz preserved by compensating PG background after local I-to-E scaling",
    }
    target.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return target


def deletion_networks(model, source: pd.DataFrame):
    n = len(model.areas)
    tau = np.r_[
        np.full(n, float(model.params["tau_exc"])),
        np.full(n, float(model.params["tau_inh"])),
    ]
    initial = np.r_[np.full(n, 10.0), np.full(n, 35.0)]
    sim_arrays = _simulation_arrays(model.params)
    networks = []
    for deleted_nodes in [""] + source.deleted_nodes.astype(str).tolist():
        fln = model.fln if not deleted_nodes else _deleted_fln(model, deleted_nodes.split("|"))
        matrix, background = _drive_matrix(model, fln, sim_arrays, _edge_arrays)
        fixedpoint, _, gain, _, residual = _fixed_point(
            matrix, background, tau, initial
        )
        j = _effective_jacobian(matrix, gain, tau)
        networks.append((fln, fixedpoint, gain, j, float(residual)))
    return networks


def linear_phase_features(
    inputs: np.ndarray,
    j: np.ndarray,
    input_idx: int,
    output_indices: np.ndarray,
    analysis_s: float,
    peak_band: tuple[float, float],
    tau_e: float,
) -> tuple[np.ndarray, np.ndarray]:
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / DT_MS))
    transfer = discrete_transfer_all(j, input_idx, output_indices, n_steps, tau_e)
    input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
    values = np.full((len(output_indices), len(inputs), 2), np.nan)
    valid = np.zeros((len(output_indices), len(inputs)), bool)
    for start in range(0, len(output_indices), 5):
        stop = min(start + 5, len(output_indices))
        paths = np.fft.irfft(
            input_fft[:, None, :] * transfer[None, start:stop, :],
            n=n_steps,
            axis=-1,
        ).astype(np.float32)
        local_values, local_valid = features_from_paths(paths, analysis_s, peak_band)
        values[start:stop] = local_values.transpose(1, 0, 2)
        valid[start:stop] = local_valid.T
    return values, valid


def nonlinear_average_features(
    model,
    inputs_by_phase: list[np.ndarray],
    fln: np.ndarray,
    fixedpoint: np.ndarray,
    input_idx: int,
    output_indices: np.ndarray,
    analysis_s: float,
    peak_band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    frequencies, keep = frequencies_for(analysis_s)
    psd_sum = np.zeros(
        (len(output_indices), len(inputs_by_phase[0]), len(frequencies)), dtype=np.float64
    )
    rectification = 0
    fs = 1000.0 / DT_MS
    nperseg = int(round(SEGMENT_S * fs))
    for phase, inputs in enumerate(inputs_by_phase):
        for start in range(0, len(inputs), BAND_CHUNK):
            stop = min(start + BAND_CHUNK, len(inputs))
            paths, counts = _network_run_fixedpoint(
                model, inputs[start:stop], output_indices, input_idx, fixedpoint, fln
            )
            actual_f, psd = signal.welch(
                paths,
                fs=fs,
                window="hann",
                nperseg=nperseg,
                noverlap=nperseg // 2,
                detrend="constant",
                scaling="density",
                axis=-1,
            )
            if not np.array_equal(actual_f[keep], frequencies):
                raise RuntimeError("Welch frequency grid changed")
            psd_sum[:, start:stop] += psd[..., keep].transpose(1, 0, 2)
            rectification += int(counts.sum())
        print(f"  nonlinear heldout phase {phase + 1}/{len(inputs_by_phase)}", flush=True)
    return (*features_from_psd(frequencies, psd_sum / len(inputs_by_phase), peak_band), rectification)


def theoretical_features_all(
    j: np.ndarray,
    z: np.ndarray,
    input_idx: int,
    output_indices: np.ndarray,
    frequencies: np.ndarray,
    peak_band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    transfer = continuous_transfer_power_all(j, input_idx, output_indices, frequencies)
    values = np.full((len(output_indices), len(z), 2), np.nan)
    valid = np.zeros((len(output_indices), len(z)), bool)
    for output_pos in range(len(output_indices)):
        spectra = np.asarray(
            [transfer[output_pos] * input_psd(frequencies, item) for item in z]
        )
        values[output_pos], valid[output_pos] = features_from_psd(
            frequencies, spectra, peak_band
        )
    return values, valid


def run_band(band: str, force: bool = False, pilot: bool = False) -> Path:
    slug = band.lower().replace(" ", "_")
    target = HERE / f"band_{slug}_full55_units.csv.gz"
    if target.exists() and not force and not pilot:
        return target
    center, peak_band = BANDS[band]
    model = load_wholebrain()
    source_all = pd.read_csv(
        WORKSPACE / "outputs/covariance_conditioning_pilot/fpinit_large/cond_100/pattern_results.csv"
    )
    source = source_all[source_all.n_deleted == 1].sort_values("row_id").reset_index(drop=True)
    if pilot:
        source = source.iloc[np.unique(np.round(np.linspace(0, len(source) - 1, 5)).astype(int))]
    design = np.load(
        WORKSPACE / "outputs/covariance_conditioning_pilot/fpinit_large/cond_100/design.npz"
    )
    n_trials = 24 if pilot else BAND_TRIALS
    analysis_s = 8.192 if pilot else BAND_ANALYSIS_S
    cal_phases = 2 if pilot else BAND_CAL_PHASES
    held_phases = 2 if pilot else BAND_HELD_PHASES
    z = design["z"][:n_trials].astype(float)
    z[:, 0] += center - 10.0
    networks = deletion_networks(model, source)
    input_idx = model.areas.index("A6Va")
    output_indices = np.arange(len(model.areas), dtype=np.int64)
    frequencies, _ = frequencies_for(analysis_s)
    cal_inputs = [
        _matched_bandlimited_inputs(
            z, DT_MS, analysis_s, BURN_S, INPUT_SIGMA_DELETION,
            SEED + 1010000 + phase * 100000,
        )
        for phase in range(cal_phases)
    ]
    held_inputs = [
        _matched_bandlimited_inputs(
            z, DT_MS, analysis_s, BURN_S, INPUT_SIGMA_DELETION,
            SEED + 1010000 + (cal_phases + phase) * 100000,
        )
        for phase in range(held_phases)
    ]
    cache = HERE / (f"band_{slug}_pilot_states" if pilot else f"band_{slug}_states")
    cache.mkdir(exist_ok=True)
    all_network = []
    started = time.time()
    for network_pos, (fln, fixedpoint, gain, j, residual) in enumerate(networks):
        state_path = cache / f"network_{network_pos:03d}.npz"
        if state_path.exists() and not force:
            with np.load(state_path) as saved:
                theory = saved["theory"].astype(float)
                theory_valid = saved["theory_valid"].astype(bool)
                calibration = saved["calibration"].astype(float)
                calibration_valid = saved["calibration_valid"].astype(bool)
                simulation = saved["simulation"].astype(float)
                simulation_valid = saved["simulation_valid"].astype(bool)
                rectification = int(saved["rectification"])
        else:
            theory, theory_valid = theoretical_features_all(
                j, z, input_idx, output_indices, frequencies, peak_band
            )
            calibration = np.full((cal_phases, len(output_indices), n_trials, 2), np.nan)
            calibration_valid = np.zeros((cal_phases, len(output_indices), n_trials), bool)
            for phase in range(cal_phases):
                calibration[phase], calibration_valid[phase] = linear_phase_features(
                    cal_inputs[phase], j, input_idx, output_indices, analysis_s,
                    peak_band, float(model.params["tau_exc"]),
                )
            simulation, simulation_valid, rectification = nonlinear_average_features(
                model, held_inputs, fln, fixedpoint, input_idx, output_indices,
                analysis_s, peak_band,
            )
            atomic_npz(
                state_path,
                theory=theory,
                theory_valid=theory_valid,
                calibration=calibration,
                calibration_valid=calibration_valid,
                simulation=simulation,
                simulation_valid=simulation_valid,
                rectification=np.asarray(rectification),
            )
        all_network.append(
            {
                "theory": theory,
                "theory_valid": theory_valid,
                "calibration": calibration,
                "calibration_valid": calibration_valid,
                "simulation": simulation,
                "simulation_valid": simulation_valid,
                "rectification": rectification,
                "gain": gain,
                "residual": residual,
            }
        )
        print(
            f"{band} full55 network {network_pos + 1}/{len(networks)} "
            f"fp_residual={residual:.2g}", flush=True
        )
    rows = []
    base = all_network[0]
    for position, item in enumerate(source.itertuples(index=False), 1):
        deleted = all_network[position]
        deleted_node = str(item.deleted_nodes)
        deleted_idx = model.areas.index(deleted_node)
        for output_idx, output_area in enumerate(model.areas):
            if output_idx == deleted_idx or not bool(deleted["gain"][output_idx]):
                continue
            common = (
                base["theory_valid"][output_idx]
                & deleted["theory_valid"][output_idx]
                & base["simulation_valid"][output_idx]
                & deleted["simulation_valid"][output_idx]
                & base["calibration_valid"][:, output_idx].all(axis=0)
                & deleted["calibration_valid"][:, output_idx].all(axis=0)
            )
            if common.sum() < 8:
                continue
            base_signal = np.cov(base["theory"][output_idx, common], rowvar=False, ddof=1)
            del_signal = np.cov(deleted["theory"][output_idx, common], rowvar=False, ddof=1)
            base_resid = pooled_phase_cov(base["calibration"][:, output_idx, common])
            del_resid = pooled_phase_cov(deleted["calibration"][:, output_idx, common])
            theory_base = corr_from_cov(base_signal + base_resid / held_phases)
            theory_deleted = corr_from_cov(del_signal + del_resid / held_phases)
            simulation_base = corr_feature(base["simulation"][output_idx], common)
            simulation_deleted = corr_feature(deleted["simulation"][output_idx], common)
            rows.append(
                {
                    "band": band,
                    "deleted_node": deleted_node,
                    "output_area": output_area,
                    "direct": bool(model.fln[output_idx, deleted_idx] > 0),
                    "n_valid": int(common.sum()),
                    "theory_baseline_r": theory_base,
                    "theory_deleted_r": theory_deleted,
                    "theory_delta_r": theory_deleted - theory_base,
                    "simulation_baseline_r": simulation_base,
                    "simulation_deleted_r": simulation_deleted,
                    "simulation_delta_r": simulation_deleted - simulation_base,
                }
            )
    result = pd.DataFrame(rows)
    destination = HERE / (f"band_{slug}_full55_units_pilot.csv.gz" if pilot else target.name)
    result.to_csv(destination, index=False, compression="gzip")
    destination.with_suffix("").with_suffix(".metrics.json").write_text(
        json.dumps(
            {
                "band": band,
                "deleted_networks": int(len(source)),
                "output_units": int(len(result)),
                "n_trials": n_trials,
                "analysis_s": analysis_s,
                "calibration_phases": cal_phases,
                "heldout_phases": held_phases,
                "runtime_s": time.time() - started,
                "simulation": "complete 55-area x E/I threshold-linear/ReLU Euler integration",
                "input_area": "A6Va",
                "covariance_condition": 100,
            }, indent=2
        ) + "\n"
    )
    return destination


def sign_flip_p(values: np.ndarray, seed: int, permutations: int = 100000) -> float:
    values = np.asarray(values, float)
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(seed)
    exceed = 0
    completed = 0
    while completed < permutations:
        size = min(2000, permutations - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        exceed += int(np.sum(np.abs(signs @ values / len(values)) >= observed))
        completed += size
    return float((exceed + 1) / (permutations + 1))


def bh(values: np.ndarray) -> np.ndarray:
    p = np.asarray(values, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(ranked, 1.0)
    return q


def assemble_band_statistics() -> tuple[Path, Path]:
    frames = []
    for band in BANDS:
        path = HERE / f"band_{band.lower().replace(' ', '_')}_full55_units.csv.gz"
        frames.append(pd.read_csv(path))
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(HERE / "deletion_bands_full55_raw.csv.gz", index=False, compression="gzip")
    units = []
    statistics = []
    for band_index, band in enumerate(BANDS):
        subset_band = raw[raw.band == band]
        for source_index, (source, column) in enumerate(
            (("Theory", "theory_delta_r"), ("Simulation", "simulation_delta_r"))
        ):
            local = subset_band[["deleted_node", "output_area", "direct", column]].dropna().copy()
            local["value"] = local[column].abs()
            local["source"] = source
            local["band"] = band
            units.append(local[["band", "source", "deleted_node", "output_area", "value", "direct"]])
            node = local.groupby(["deleted_node", "direct"], as_index=False).value.mean()
            pivot = node.pivot(index="deleted_node", columns="direct", values="value").dropna()
            difference = pivot[True].to_numpy() - pivot[False].to_numpy()
            statistics.append(
                {
                    "band": band,
                    "source": source,
                    "n_deleted_nodes": int(len(pivot)),
                    "direct_mean": float(pivot[True].mean()),
                    "non_direct_mean": float(pivot[False].mean()),
                    "mean_difference": float(difference.mean()),
                    "sign_flip_p": sign_flip_p(
                        difference, SEED + 5000 + band_index * 10 + source_index
                    ),
                }
            )
    unit_frame = pd.concat(units, ignore_index=True)
    stat_frame = pd.DataFrame(statistics)
    stat_frame["sign_flip_fdr_q"] = bh(stat_frame.sign_flip_p.to_numpy())
    unit_path = HERE / "deletion_bands_full55_plot_units.csv.gz"
    stat_path = HERE / "deletion_bands_full55_statistics.csv"
    unit_frame.to_csv(unit_path, index=False, compression="gzip")
    stat_frame.to_csv(stat_path, index=False)
    return unit_path, stat_path


def stars(p: float) -> str:
    if p < 1e-4:
        return "****"
    if p < 1e-3:
        return "***"
    if p < 1e-2:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_outputs() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 15,
            "axes.linewidth": 1.7,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
        }
    )
    stable = pd.read_csv(HERE / "stable_complex_full55.csv")
    deletion = pd.read_csv(
        WORKSPACE / "outputs/covariance_conditioning_pilot/fpinit_large/cond_100/pattern_results.csv"
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 8))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.11, top=0.96, wspace=0.22)
    ax = axes[0]
    x = stable.full55_theory_r.to_numpy(float)
    y = stable.full55_simulation_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y)
    fit = stats.linregress(x[use], y[use])
    lo, hi = min(x[use].min(), y[use].min()), max(x[use].max(), y[use].max())
    pad = 0.05 * max(hi - lo, 0.1)
    grid = np.linspace(lo - pad, hi + pad, 200)
    ax.scatter(x[use], y[use], s=42, alpha=0.56, color="#247BA0", edgecolor="none")
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#222222", lw=2.8, label="OLS")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.95, 0.05, f"slope = {fit.slope:.3f}", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=21, fontweight="bold",
    )
    ax.set_xlabel("")
    ax.set_ylabel("")

    ax = axes[1]
    x = deletion.theory_delta_r.to_numpy(float)
    y = deletion.simulation_delta_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y)
    fit_del = stats.linregress(x[use], y[use])
    low, high = min(x[use].min(), y[use].min()), max(x[use].max(), y[use].max())
    pad = 0.04 * max(high - low, 0.1)
    colors = {1: "#247BA0", 3: "#F4A261", 4: "#2A9D8F", 5: "#8E5EA2"}
    markers = {1: "o", 3: "^", 4: "s", 5: "D"}
    for count in sorted(deletion.n_deleted.unique()):
        take = use & (deletion.n_deleted.to_numpy(int) == count)
        ax.scatter(
            x[take], y[take], s=42, marker=markers[int(count)],
            color=colors[int(count)], alpha=0.85, label=f"{int(count)} nodes",
        )
    ax.plot(
        [low - pad, high + pad], [low - pad, high + pad],
        color="#444444", ls="--", lw=2.0,
    )
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.05, 0.95,
        f"r = {fit_del.rvalue:.2f}\nslope = {fit_del.slope:.2f}",
        transform=ax.transAxes, va="top", fontsize=21, fontweight="bold",
    )
    ax.legend(frameon=False, loc="lower right")
    ax.set_xlabel("")
    ax.set_ylabel("")
    for axis in axes:
        axis.tick_params(width=1.7, length=6)
        for spine in axis.spines.values():
            spine.set_linewidth(1.7)
    stem = HERE / "stable_complex_and_deletion_full55_exact_vector_5x4"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=stem.with_suffix(".alignment.json"), strict=True
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)

    unit_path, stat_path = assemble_band_statistics()
    units = pd.read_csv(unit_path)
    stat_frame = pd.read_csv(stat_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.91, wspace=0.12)
    positions = np.arange(len(BANDS), dtype=float)
    offsets = {True: -0.16, False: 0.16}
    colors_dn = {True: "#D1495B", False: "#247BA0"}
    labels = {True: "Direct", False: "Non-direct"}
    annotation_max = 0.0
    for ax, source in zip(axes, ("Theory", "Simulation")):
        for band_pos, band in enumerate(BANDS):
            subset = units[(units.source == source) & (units.band == band)]
            tops = []
            for direct in (True, False):
                values = subset.loc[subset.direct == direct, "value"].to_numpy(float)
                values = values[np.isfinite(values) & (values > 0)]
                box = ax.boxplot(
                    [values], positions=[band_pos + offsets[direct]], widths=0.25,
                    patch_artist=True, showfliers=True,
                    medianprops={"color": "#222222", "linewidth": 2.5},
                    whiskerprops={"linewidth": 1.8}, capprops={"linewidth": 1.8},
                    flierprops={
                        "marker": "o", "markersize": 2.4,
                        "markerfacecolor": colors_dn[direct],
                        "markeredgecolor": "none", "alpha": 0.35,
                    },
                )
                box["boxes"][0].set_facecolor(colors_dn[direct])
                box["boxes"][0].set_alpha(0.32)
                box["boxes"][0].set_linewidth(2.0)
                q1, q3 = np.quantile(values, [0.25, 0.75])
                tops.append(float(np.max(values[values <= q3 + 1.5 * (q3 - q1)])))
            q = float(
                stat_frame.loc[
                    (stat_frame.source == source) & (stat_frame.band == band),
                    "sign_flip_fdr_q",
                ].iloc[0]
            )
            # Put the comparison bracket above every displayed point.  The
            # user explicitly requested retaining outliers and expanding the
            # coordinate range when necessary.
            top = float(subset.loc[subset.value > 0, "value"].max()) * 1.25
            annotation_max = max(annotation_max, top * 1.45)
            ax.text(
                band_pos, top * 1.08, stars(q), ha="center", va="bottom",
                fontsize=17, fontweight="bold",
            )
        ax.set_yscale("log")
        ax.set_xlim(-0.55, len(BANDS) - 0.45)
        ax.set_xticks(positions, ["Alpha", "Beta", "Low\ngamma", "High\ngamma"])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(width=1.7, length=6)
        for spine in ax.spines.values():
            spine.set_linewidth(1.7)
    positive = units.loc[units.value > 0, "value"].to_numpy(float)
    # On a log axis, glyph height is multiplicative; leave roughly one decade
    # above the largest annotation so it cannot collide with the top spine.
    axes[0].set_ylim(float(positive.min() * 0.7), annotation_max * 20.0)
    handles = [
        mpl.lines.Line2D(
            [0], [0], marker="s", color="none", markerfacecolor=colors_dn[v],
            alpha=0.55, markersize=11, label=labels[v],
        )
        for v in (True, False)
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=2)
    stem = HERE / "deletion_bands_direct_nondirect_full55_exact_vector_2x1"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=stem.with_suffix(".alignment.json"), strict=True
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stable", action="store_true")
    parser.add_argument("--band", choices=list(BANDS))
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--threads", type=int, default=3)
    args = parser.parse_args()
    set_num_threads(args.threads)
    if args.stable:
        print(run_stable(force=args.force, pilot=args.pilot))
    if args.band:
        print(run_band(args.band, force=args.force, pilot=args.pilot))
    if args.plot:
        plot_outputs()


if __name__ == "__main__":
    main()
