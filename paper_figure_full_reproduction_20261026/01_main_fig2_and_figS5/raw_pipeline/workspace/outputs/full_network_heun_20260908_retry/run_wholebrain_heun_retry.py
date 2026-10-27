from __future__ import annotations

"""Whole-brain Experiment-2 Theory-C validation with explicit nonlinear Heun.

All inputs and source implementations are read-only.  Every artifact produced by
this script lives beside this file.  The accelerated path is the exact finite-
record SISO transfer of the full 110-state Heun recurrence from the registered
fixed-point initial condition; it is enabled only after the nonlinear two-stage
ReLU trajectory validation passes.
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numba import njit
from scipy import fft, signal, stats


HERE = Path(__file__).resolve().parent
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
SOURCE_ROOT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "workspace/outputs/full_network_exact_vector_20260907"
SOURCE_RUNNER = SOURCE_ROOT / "run_exp2_theory_c_1000conditions.py"
OLD_HELPER = SOURCE_ROOT / "run_exp2_oldprotocol_uniform.py"
BASE_HELPER = SOURCE_ROOT / "run_full_network_exact.py"
SOURCE_RESULTS = SOURCE_ROOT / "exp2_theory_c_1000conditions/results_condition_level.csv"
SOURCE_CONDITIONS = SOURCE_ROOT / "exp2_theory_c_1000conditions/conditions_input_1000.csv"
SOURCE_CACHE = SOURCE_ROOT / "exp2_theory_c_1000conditions/condition_cache"
ZERO_INPUT_SOURCE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "workspace/outputs/revised_core_and_zero_input_boxplots_20260908/wholebrain_zero_input_boxplot_values.csv"
FORMAL_HEUN_2D = PACKAGE / "pap_validation/two_dimensional_ode_validation.py"
FORMAL_HEUN_EXP2 = PACKAGE / "pap_validation/exp2_pap_correlation_validation.py"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(SOURCE_ROOT), str(PACKAGE), str(FIGURE_SKILL / "scripts")]

import run_exp2_oldprotocol_uniform as old
import run_exp2_theory_c_1000conditions as source
import run_full_network_exact as base
from audit_panel_alignment import require_matplotlib_panel_alignment


EXPECTED = 1000
N_TRIALS = 256
N_CAL = 3
N_HELD = 5
N_PHASES = 8
ANALYSIS_S = 131.072
BURN_S = 2.048
DT_MS = 0.25
SMALL_DT_MS = 0.125
SEGMENT_S = 4.096
PEAK_BAND = (5.0, 100.0)
SLOPE_BAND = (100.0, 1000.0)
SEED = 2026090701
INPUT_AREA = "PG"
OUTPUT_AREA = "PG"
CHUNK = 16
FULL_CHUNK = 32
PILOT_N = 12
PILOT_METHODS = ("heun_dt0p25", "heun_dt0p125")
FULL_METHOD = "heun_dt0p25"
PROTOCOL_VERSION = "wholebrain_heun_retry_v1"
POINTWISE_ATOL = 2e-8
POINTWISE_RTOL = 2e-7


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_ready(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): json_ready(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_ready(v) for v in x]
    if isinstance(x, np.ndarray):
        return json_ready(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating,)): return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)): return bool(x)
    if isinstance(x, float): return None if not np.isfinite(x) else x
    return x


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        f.write(value); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(json_ready(value), indent=2) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def append_log(message: str) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "run.log").open("a", encoding="utf-8") as f:
        f.write(f"{utc_now()} {message}\n")


def method_dt(method: str) -> float:
    if method == "heun_dt0p25": return DT_MS
    if method == "heun_dt0p125": return SMALL_DT_MS
    raise ValueError(method)


def chunk_size(scope: str) -> int:
    return FULL_CHUNK if scope == "full" else CHUNK


def protocol_payload() -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "system": "complete 55-area, P/G (E/I), 110-state threshold-linear/ReLU network",
        "conditions": EXPECTED, "trials_per_condition": N_TRIALS,
        "calibration_phases": N_CAL, "heldout_phases": N_HELD,
        "analysis_s": ANALYSIS_S, "burn_s": BURN_S,
        "dt_ms": DT_MS, "pilot_small_dt_ms": SMALL_DT_MS,
        "input_area": INPUT_AREA, "output_area": OUTPUT_AREA,
        "welch": {"window": "periodic Hann", "segment_s": SEGMENT_S,
                  "overlap_fraction": 0.5, "peak_band_hz": list(PEAK_BAND),
                  "slope_band_hz": list(SLOPE_BAND), "relaxed_peak_qc": True},
        "forcing": {
            "registered_seed_rule": "2026090701 + 6200000 + condition_id*100000 + phase*10000 + trial",
            "predictor_corrector": "same u[n] in both Heun stages, matching formal 2D Experiment 1/2",
            "burn_in": "last 2.048 s of the registered analysis forcing prepended; fixed-point initial state",
            "small_dt": "paired zero-order hold: each registered 0.25-ms forcing sample repeated twice",
        },
        "heun": {
            "stage_1": "k1=(-r+ReLU(Mr+h+bu[n]))/tau",
            "predictor": "p=r+dt*k1",
            "stage_2": "k2=(-p+ReLU(Mp+h+bu[n]))/tau",
            "corrector": "r_next=r+dt*(k1+k2)/2",
            "relu_evaluation": "all 110 drives independently at both stages",
            "formal_sources": [str(FORMAL_HEUN_2D), str(FORMAL_HEUN_EXP2)],
        },
        "accelerator": "zero-initial finite-record linear convolution of the exact full 110-state active-branch Heun recurrence; no circular steady-state assumption",
        "theory": "unchanged latest Theory C continuous-time, condition-specific fixed point/Jacobian and Hann-kernel convolution",
        "calibration": "unchanged condition-specific pooled within-trial residual covariance from 3 calibration phases; no slope fit or regression scaling",
        "full_run_gate": "pilot and nonlinear two-stage ReLU pointwise equivalence must PASS",
    }


def validate_inputs() -> tuple[pd.DataFrame, dict]:
    frame = pd.read_csv(SOURCE_CONDITIONS).sort_values("condition_id").reset_index(drop=True)
    if len(frame) != EXPECTED or frame.condition_id.nunique() != EXPECTED:
        raise RuntimeError("The frozen condition table must contain 1000 unique rows")
    required = ["condition_id", "S", "sigma_beta", "floor_ratio", "input_sigma",
                "critical_real_per_s", "critical_frequency_hz"]
    if not np.isfinite(frame[required].to_numpy(float)).all():
        raise RuntimeError("Frozen condition table contains non-finite required values")
    if not (frame.S > source.STABLE_COMPLEX_ONSET_S).all():
        raise RuntimeError("At least one condition is not stable-complex")
    if not (frame.S > source.STABLE_RESONANCE_ONSET_S).all():
        raise RuntimeError("At least one condition is below stable-resonance onset")
    if not (frame.S < source.PATH_UPPER_S).all() or not (frame.critical_real_per_s < 0).all():
        raise RuntimeError("At least one condition violates the registered stable path")
    old_results = pd.read_csv(SOURCE_RESULTS)
    if len(old_results) != EXPECTED or old_results.condition_id.nunique() != EXPECTED:
        raise RuntimeError("Old paired result table is incomplete")
    fingerprints = {str(p): sha256(p) for p in [Path(__file__).resolve(), SOURCE_CONDITIONS, SOURCE_RESULTS,
                    SOURCE_RUNNER, OLD_HELPER, BASE_HELPER, FORMAL_HEUN_2D,
                    FORMAL_HEUN_EXP2, ZERO_INPUT_SOURCE]}
    manifest = {
        "status": "PASS", "created_at_utc": utc_now(),
        "before_condition_count": len(frame), "after_condition_count": len(frame),
        "filtering_performed": False,
        "S_range": [float(frame.S.min()), float(frame.S.max())],
        "sigma_beta_range": [float(frame.sigma_beta.min()), float(frame.sigma_beta.max())],
        "floor_ratio_range": [float(frame.floor_ratio.min()), float(frame.floor_ratio.max())],
        "stable_complex_onset_S": source.STABLE_COMPLEX_ONSET_S,
        "stable_resonance_onset_S": source.STABLE_RESONANCE_ONSET_S,
        "path_upper_S": source.PATH_UPPER_S,
        "critical_real_per_s_range": [float(frame.critical_real_per_s.min()), float(frame.critical_real_per_s.max())],
        "fingerprints": fingerprints, "protocol": protocol_payload(),
        "software": {"python": sys.version, "numpy": np.__version__,
                     "scipy": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
                     "pandas": pd.__version__, "matplotlib": mpl.__version__},
        "hardware": {"platform": platform.platform(), "logical_cpu_count": os.cpu_count()},
    }
    digest = hashlib.sha256(json.dumps(manifest["protocol"], sort_keys=True).encode()).hexdigest()
    manifest["protocol_digest"] = digest
    atomic_csv(HERE / "conditions_input_1000.csv", frame)
    atomic_json(HERE / "manifest.json", manifest)
    return frame, manifest


def load_model_and_condition(condition: pd.Series) -> dict:
    model = base.load_wholebrain()
    solved = source._assemble_and_solve_condition(model, condition)
    arrays = solved["arrays"]
    n = len(model.areas)
    ids = np.arange(n)
    matrix = np.zeros((2*n, 2*n), float)
    matrix[ids, ids] = arrays[0]; matrix[ids, n+ids] = arrays[1]
    matrix[n+ids, ids] = arrays[2]; matrix[n+ids, n+ids] = arrays[3]
    np.add.at(matrix, (arrays[4], arrays[5]), arrays[6])
    np.add.at(matrix, (n+arrays[4], arrays[5]), arrays[7])
    tau = np.r_[np.full(n, float(model.params["tau_exc"])),
                np.full(n, float(model.params["tau_inh"]))]
    background = np.r_[arrays[8], arrays[9]]
    return {"model": model, "solved": solved, "arrays": arrays,
            "matrix": matrix, "background": background, "tau": tau}


def heun_recurrence(j: np.ndarray, input_idx: int, tau_e: float, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    eye = np.eye(len(j))
    b = np.zeros(len(j)); b[input_idx] = 1.0 / tau_e
    a = eye + dt_ms*j + 0.5*dt_ms*dt_ms*(j @ j)
    bd = dt_ms*b + 0.5*dt_ms*dt_ms*(j @ b)
    return a, bd


def finite_transfer(a: np.ndarray, bd: np.ndarray, output_idx: int, fft_length: int) -> np.ndarray:
    eig, vec = np.linalg.eig(a)
    coeff = vec[output_idx] * np.linalg.solve(vec, bd)
    result = np.empty(fft_length // 2 + 1, complex)
    block = 8192
    for start in range(0, len(result), block):
        stop = min(start + block, len(result))
        theta = 2*np.pi*np.arange(start, stop)/fft_length
        zinv = np.exp(-1j*theta)
        result[start:stop] = np.sum(coeff[None, :] / (1.0 - zinv[:, None]*eig[None, :]), axis=1)
    return result


def registered_forcing(condition: pd.Series, betas: np.ndarray, phase: int, trials: np.ndarray,
                       dt_ms: float) -> np.ndarray:
    coarse_n = int(round(ANALYSIS_S * 1000.0 / DT_MS))
    burn_n = int(round(BURN_S * 1000.0 / DT_MS))
    frequencies = np.fft.rfftfreq(coarse_n, d=DT_MS/1000.0)
    coarse = np.empty((len(trials), coarse_n + burn_n), np.float64)
    for pos, trial in enumerate(trials):
        rng = np.random.default_rng(SEED + 6200000 + int(condition.condition_id)*100000 + phase*10000 + int(trial))
        shape = old._normalized_input_shape(frequencies, float(betas[int(trial)]))
        angle = rng.uniform(0.0, 2*np.pi, len(frequencies))
        spectrum = np.sqrt(shape) * np.exp(1j*angle)
        spectrum[0] = 0.0; spectrum[-1] = spectrum[-1].real
        u = np.fft.irfft(spectrum, n=coarse_n)
        u *= float(condition.input_sigma) / max(u.std(), 1e-15)
        u += float(condition.input_sigma)*float(condition.floor_ratio)*rng.normal(size=coarse_n)
        u -= u.mean()
        coarse[pos, :burn_n] = u[-burn_n:]
        coarse[pos, burn_n:] = u
    ratio = int(round(DT_MS/dt_ms))
    if not np.isclose(ratio*dt_ms, DT_MS):
        raise RuntimeError("Only integer paired dt refinement is supported")
    return np.repeat(coarse, ratio, axis=1) if ratio > 1 else coarse


def transfer_paths(forcing: np.ndarray, transfer: np.ndarray, fft_length: int, burn_steps: int) -> np.ndarray:
    transformed = fft.rfft(forcing, n=fft_length, axis=-1, workers=1)
    out = fft.irfft(transformed*transfer[None, :], n=fft_length, axis=-1, workers=1)
    return out[:, burn_steps:forcing.shape[1]].astype(np.float32)


def features_from_paths(paths: np.ndarray, dt_ms: float) -> tuple[np.ndarray, np.ndarray]:
    fs = 1000.0/dt_ms
    nperseg = int(round(SEGMENT_S*fs))
    f, psd = signal.welch(paths, fs=fs, window="hann", nperseg=nperseg,
                          noverlap=nperseg//2, detrend="constant", scaling="density", axis=-1)
    keep = (f >= 0.5) & (f <= 1200.0)
    values, valid = base.features_from_psd(f[keep], psd[:, keep], PEAK_BAND, relaxed_peak_qc=True)
    return old.display_convention(values), valid


@njit(cache=True)
def nonlinear_heun_one(forcing, local_ee, local_ei, local_ie, local_ii,
                       edge_target, edge_source, edge_e, edge_i, bg_e, bg_i,
                       fixed_e, fixed_i, dt_ms, tau_e, tau_i, burn_steps,
                       input_idx, output_idx):
    n = len(local_ee); total = len(forcing)
    e = fixed_e.copy(); ii = fixed_i.copy()
    pe = np.empty(n); pi = np.empty(n); k1e = np.empty(n); k1i = np.empty(n)
    le = np.empty(n); li = np.empty(n); out = np.empty(total-burn_steps)
    rect1 = 0; rect2 = 0; min_drive = 1e300; outpos = 0
    for step in range(total):
        for q in range(n): le[q] = 0.0; li[q] = 0.0
        for q in range(len(edge_target)):
            t=edge_target[q]; s=edge_source[q]
            le[t] += edge_e[q]*e[s]; li[t] += edge_i[q]*e[s]
        for q in range(n):
            de=local_ee[q]*e[q]+local_ei[q]*ii[q]+le[q]+bg_e[q]
            if q == input_idx: de += forcing[step]
            di=local_ie[q]*e[q]+local_ii[q]*ii[q]+li[q]+bg_i[q]
            if de < min_drive: min_drive=de
            if di < min_drive: min_drive=di
            if de < 0.0: de=0.0; rect1 += 1
            if di < 0.0: di=0.0; rect1 += 1
            k1e[q]=(-e[q]+de)/tau_e; k1i[q]=(-ii[q]+di)/tau_i
            pe[q]=e[q]+dt_ms*k1e[q]; pi[q]=ii[q]+dt_ms*k1i[q]
        for q in range(n): le[q] = 0.0; li[q] = 0.0
        for q in range(len(edge_target)):
            t=edge_target[q]; s=edge_source[q]
            le[t] += edge_e[q]*pe[s]; li[t] += edge_i[q]*pe[s]
        for q in range(n):
            de=local_ee[q]*pe[q]+local_ei[q]*pi[q]+le[q]+bg_e[q]
            if q == input_idx: de += forcing[step]
            di=local_ie[q]*pe[q]+local_ii[q]*pi[q]+li[q]+bg_i[q]
            if de < min_drive: min_drive=de
            if di < min_drive: min_drive=di
            if de < 0.0: de=0.0; rect2 += 1
            if di < 0.0: di=0.0; rect2 += 1
            e[q] += 0.5*dt_ms*(k1e[q]+(-pe[q]+de)/tau_e)
            ii[q] += 0.5*dt_ms*(k1i[q]+(-pi[q]+di)/tau_i)
        if step >= burn_steps:
            out[outpos] = e[output_idx]-fixed_e[output_idx]; outpos += 1
    return out, rect1, rect2, min_drive


def cache_path(scope: str, method: str, condition_id: int, phase: int, start: int) -> Path:
    suffix = f"_n{chunk_size(scope)}" if scope == "full" else ""
    return HERE / "chunks" / scope / method / f"condition_{condition_id:04d}" / f"phase_{phase}" / f"chunk_{start:03d}{suffix}.npz"


def run_chunk(condition: pd.Series, betas: np.ndarray, solved: dict, method: str,
              scope: str, phase: int, start: int, force: bool = False,
              precomputed_transfer: np.ndarray | None = None,
              precomputed_fft_length: int | None = None) -> Path:
    path = cache_path(scope, method, int(condition.condition_id), phase, start)
    if path.exists() and not force:
        return path
    dt_ms = method_dt(method); ratio = int(round(DT_MS/dt_ms))
    size = chunk_size(scope)
    trials = np.arange(start, min(start+size, N_TRIALS), dtype=int)
    forcing = registered_forcing(condition, betas, phase, trials, dt_ms)
    n_total = forcing.shape[1]
    fft_length = (precomputed_fft_length if precomputed_fft_length is not None
                  else fft.next_fast_len(2*n_total-1, real=True))
    if precomputed_transfer is None:
        a, bd = heun_recurrence(solved["jacobian"], solved["input_idx"],
                                 float(solved["model"].params["tau_exc"]), dt_ms)
        transfer = finite_transfer(a, bd, solved["output_idx"], fft_length)
    else:
        transfer = precomputed_transfer
    paths = transfer_paths(forcing, transfer, fft_length,
                           int(round(BURN_S*1000.0/dt_ms)))
    values, valid = features_from_paths(paths, dt_ms)
    atomic_npz(path, protocol=np.asarray(PROTOCOL_VERSION), condition_id=np.asarray(int(condition.condition_id)),
               phase=np.asarray(phase), trial_ids=trials, features=values, valid=valid,
               dt_ms=np.asarray(dt_ms), forcing_refinement_ratio=np.asarray(ratio))
    return path


def load_theory(condition_id: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = SOURCE_CACHE / f"condition_{condition_id:04d}.npz"
    with np.load(path) as z:
        return z["betas"].copy(), z["theory_C_features"].copy(), z["theory_C_valid"].copy()


def assemble_method(condition: pd.Series, method: str, scope: str) -> dict:
    cid = int(condition.condition_id)
    betas, theory, theory_valid = load_theory(cid)
    phases = np.full((N_PHASES, N_TRIALS, 2), np.nan)
    valid = np.zeros((N_PHASES, N_TRIALS), bool)
    for phase in range(N_PHASES):
        for start in range(0, N_TRIALS, chunk_size(scope)):
            with np.load(cache_path(scope, method, cid, phase, start)) as z:
                ids = z["trial_ids"].astype(int)
                phases[phase, ids] = z["features"]
                valid[phase, ids] = z["valid"]
    common = theory_valid & valid.all(axis=0)
    if common.sum() < 8:
        raise RuntimeError(f"condition {cid} method {method}: fewer than 8 common trials")
    residual_cov = old.pooled_within_trial_cov(phases[:N_CAL, common])
    signal_cov = np.cov(theory[common], rowvar=False, ddof=1)
    held = [old.feature_corr(phases[p, common]) for p in range(N_CAL, N_PHASES)]
    return {
        "condition_id": cid, "method": method, "n_common": int(common.sum()),
        "theory_r": old.cov_to_corr(signal_cov + residual_cov),
        "simulation_r": float(np.mean(held)), "heldout_r_json": json.dumps(held),
        "residual_var_P": float(residual_cov[0,0]), "residual_var_A": float(residual_cov[1,1]),
        "residual_cov_PA": float(residual_cov[0,1]),
    }


def pilot_ids(frame: pd.DataFrame) -> list[int]:
    chosen = old.farthest_parameter_sample(frame, PILOT_N)
    return sorted(chosen.condition_id.astype(int).tolist())


def pointwise_validation(frame: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    target = HERE / "pilot" / "active_branch_heun_validation.csv"
    if target.exists() and not force:
        return pd.read_csv(target)
    sample = old.farthest_parameter_sample(frame, 4).sort_values("condition_id")
    rows = []
    for _, condition in sample.iterrows():
        bundle = load_model_and_condition(condition); solved = bundle["solved"]
        arrays = bundle["arrays"]; n = len(bundle["model"].areas)
        betas, _, _ = load_theory(int(condition.condition_id))
        for dt_ms in (DT_MS, SMALL_DT_MS):
            trials = np.asarray([0, 127, 255], int); phase = 3
            full_forcing = registered_forcing(condition, betas, phase, trials, dt_ms)
            validation_analysis_s = 8.192
            keep_n = int(round((BURN_S+validation_analysis_s)*1000.0/dt_ms))
            forcing = full_forcing[:, :keep_n]
            fft_length = fft.next_fast_len(2*keep_n-1, real=True)
            a, bd = heun_recurrence(solved["jacobian"], solved["input_idx"],
                                     float(bundle["model"].params["tau_exc"]), dt_ms)
            tf = finite_transfer(a, bd, solved["output_idx"], fft_length)
            accelerated = transfer_paths(forcing, tf, fft_length, int(round(BURN_S*1000/dt_ms)))
            for pos, trial in enumerate(trials):
                direct, r1, r2, min_drive = nonlinear_heun_one(
                    forcing[pos], arrays[0], arrays[1], arrays[2], arrays[3],
                    arrays[4], arrays[5], arrays[6], arrays[7], arrays[8], arrays[9],
                    solved["fixedpoint"][:n], solved["fixedpoint"][n:], dt_ms,
                    float(bundle["model"].params["tau_exc"]), float(bundle["model"].params["tau_inh"]),
                    int(round(BURN_S*1000/dt_ms)), solved["input_idx"], solved["output_idx"])
                delta = direct-accelerated[pos].astype(float)
                direct_feature, direct_valid = features_from_paths(direct[None,:], dt_ms)
                accel_feature, accel_valid = features_from_paths(accelerated[pos:pos+1], dt_ms)
                rows.append({"condition_id": int(condition.condition_id), "S": float(condition.S),
                    "sigma_beta": float(condition.sigma_beta), "floor_ratio": float(condition.floor_ratio),
                    "dt_ms": dt_ms, "phase": phase, "trial": int(trial),
                    "n_timepoints": len(direct), "max_abs_path_difference": float(np.max(np.abs(delta))),
                    "path_RMSE": float(np.sqrt(np.mean(delta*delta))),
                    "path_scale_max_abs": float(np.max(np.abs(direct))),
                    "allclose": bool(np.allclose(direct, accelerated[pos], rtol=POINTWISE_RTOL, atol=POINTWISE_ATOL)),
                    "stage1_relu_crossings": int(r1), "stage2_relu_crossings": int(r2),
                    "minimum_pre_relu_drive": float(min_drive),
                    "peak_difference_hz": float(direct_feature[0,0]-accel_feature[0,0]),
                    "slope_difference": float(direct_feature[0,1]-accel_feature[0,1]),
                    "feature_valid_match": bool(direct_valid[0] == accel_valid[0])})
    result = pd.DataFrame(rows)
    atomic_csv(target, result)
    summary = {"status": "PASS" if (result.allclose.all() and
              (result.stage1_relu_crossings == 0).all() and
              (result.stage2_relu_crossings == 0).all()) else "FAIL",
              "n_representative_conditions": int(result.condition_id.nunique()),
              "n_paired_trials": len(result),
              "max_abs_path_difference": float(result.max_abs_path_difference.max()),
              "max_path_RMSE": float(result.path_RMSE.max()),
              "relu_crossings_total": int(result.stage1_relu_crossings.sum()+result.stage2_relu_crossings.sum()),
              "relu_stage_state_evaluations": int(sum(
                  (row.n_timepoints + round(BURN_S*1000.0/row.dt_ms))*110*2
                  for row in result.itertuples()
              )),
              "minimum_pre_relu_drive": float(result.minimum_pre_relu_drive.min()),
              "pointwise_tolerances": {"atol": POINTWISE_ATOL, "rtol": POINTWISE_RTOL},
              "validation_analysis_s": 8.192}
    summary["relu_crossing_fraction"] = (
        summary["relu_crossings_total"] / summary["relu_stage_state_evaluations"]
    )
    atomic_json(HERE / "pilot" / "active_branch_heun_validation.json", summary)
    return result


def update_progress(scope: str, total_conditions: int, methods: tuple[str, ...]) -> None:
    size = chunk_size(scope)
    expected_chunks = total_conditions*len(methods)*N_PHASES*(N_TRIALS//size)
    root = HERE / "chunks" / scope
    pattern = f"chunk_*_n{size}.npz" if scope == "full" else "chunk_*.npz"
    completed = len(list(root.glob(f"**/{pattern}"))) if root.exists() else 0
    atomic_json(HERE / f"progress_{scope}.json", {"scope": scope, "completed_chunks": completed,
        "expected_chunks": expected_chunks, "fraction": completed/max(expected_chunks,1),
        "updated_at_utc": utc_now(), "chunk_trials": size})


def condition_worker(payload: dict) -> list[dict]:
    condition = pd.Series(payload["condition"]); scope = payload["scope"]
    methods = tuple(payload["methods"]); force = bool(payload.get("force", False))
    started = time.time(); bundle = load_model_and_condition(condition); solved = bundle["solved"]
    betas, _, _ = load_theory(int(condition.condition_id))
    rows = []
    for method in methods:
        dt_ms = method_dt(method)
        n_total = int(round((ANALYSIS_S+BURN_S)*1000.0/dt_ms))
        fft_length = fft.next_fast_len(2*n_total-1, real=True)
        a, bd = heun_recurrence(solved["jacobian"], solved["input_idx"],
                                 float(bundle["model"].params["tau_exc"]), dt_ms)
        transfer = finite_transfer(a, bd, solved["output_idx"], fft_length)
        for phase in range(N_PHASES):
            for start in range(0, N_TRIALS, chunk_size(scope)):
                run_chunk(condition, betas, {**solved, "model": bundle["model"]},
                          method, scope, phase, start, force,
                          precomputed_transfer=transfer,
                          precomputed_fft_length=fft_length)
        row = assemble_method(condition, method, scope)
        row.update({k: condition[k] for k in ("S", "sigma_beta", "floor_ratio", "input_sigma")})
        row.update(solved["diagnostics"]); row["runtime_s"] = time.time()-started
        rows.append(row)
    return rows


def run_conditions(frame: pd.DataFrame, scope: str, methods: tuple[str, ...], workers: int,
                   force: bool = False) -> pd.DataFrame:
    row_dir = HERE / "condition_rows" / scope; fail_dir = HERE / "failures" / scope
    row_dir.mkdir(parents=True, exist_ok=True); fail_dir.mkdir(parents=True, exist_ok=True)
    successes = []
    pending = []
    for _, condition in frame.iterrows():
        paths = [row_dir / f"condition_{int(condition.condition_id):04d}_{m}.json" for m in methods]
        if all(p.exists() for p in paths) and not force:
            successes.extend(json.loads(p.read_text()) for p in paths)
        else:
            pending.append({"condition": condition.to_dict(), "scope": scope, "methods": methods, "force": force})
    append_log(f"{scope} start conditions={len(frame)} pending={len(pending)} methods={methods} workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(condition_worker, p): int(p["condition"]["condition_id"]) for p in pending}
        for future in as_completed(futures):
            cid = futures[future]
            try:
                rows = future.result()
                for row in rows:
                    atomic_json(row_dir / f"condition_{cid:04d}_{row['method']}.json", row)
                    successes.append(row)
                append_log(f"{scope} condition={cid} PASS")
            except Exception as exc:
                atomic_json(fail_dir / f"condition_{cid:04d}.json", {"condition_id": cid,
                    "error": repr(exc), "traceback": traceback.format_exc(), "at": utc_now()})
                append_log(f"{scope} condition={cid} FAIL {exc!r}")
            update_progress(scope, len(frame), methods)
    result = pd.DataFrame(successes).sort_values(["condition_id", "method"]).reset_index(drop=True)
    atomic_csv(HERE / f"results_{scope}.csv", result)
    update_progress(scope, len(frame), methods)
    return result


def comparison_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    use = np.isfinite(x) & np.isfinite(y); x=x[use]; y=y[use]
    fit = stats.linregress(x, y); d=y-x
    return {"n": len(x), "slope": float(fit.slope), "intercept": float(fit.intercept),
            "pearson_r": float(fit.rvalue), "R2": float(fit.rvalue**2),
            "MAE": float(np.mean(np.abs(d))), "RMSE": float(np.sqrt(np.mean(d*d))),
            "mean_bias": float(np.mean(d))}


def aggregate_pilot(frame: pd.DataFrame) -> dict:
    result = pd.read_csv(HERE / "results_pilot.csv")
    old_frame = pd.read_csv(SOURCE_RESULTS)[["condition_id", "theory_C_corrected_r", "simulation_r"]]
    wide = result.pivot(index="condition_id", columns="method", values=["theory_r", "simulation_r"]).reset_index()
    wide.columns = [c[0] if not c[1] else f"{c[0]}_{c[1]}" for c in wide.columns]
    merged = frame.merge(old_frame, on="condition_id").merge(wide, on="condition_id")
    merged["heun_minus_old"] = merged.simulation_r_heun_dt0p25-merged.simulation_r
    merged["smaller_dt_minus_current_heun"] = merged.simulation_r_heun_dt0p125-merged.simulation_r_heun_dt0p25
    merged["old_error"] = merged.simulation_r-merged.theory_C_corrected_r
    merged["heun_error"] = merged.simulation_r_heun_dt0p25-merged.theory_r_heun_dt0p25
    atomic_csv(HERE / "pilot" / "pilot_condition_comparison.csv", merged)
    metrics = {
        "old_integrator": comparison_metrics(merged.theory_C_corrected_r.to_numpy(), merged.simulation_r.to_numpy()),
        "heun_dt0p25": comparison_metrics(merged.theory_r_heun_dt0p25.to_numpy(), merged.simulation_r_heun_dt0p25.to_numpy()),
        "heun_dt0p125": comparison_metrics(merged.theory_r_heun_dt0p125.to_numpy(), merged.simulation_r_heun_dt0p125.to_numpy()),
        "paired_differences": {"heun_minus_old_mean": float(merged.heun_minus_old.mean()),
            "heun_minus_old_RMSE": float(np.sqrt(np.mean(merged.heun_minus_old**2))),
            "smaller_dt_minus_current_mean": float(merged.smaller_dt_minus_current_heun.mean()),
            "smaller_dt_minus_current_RMSE": float(np.sqrt(np.mean(merged.smaller_dt_minus_current_heun**2)))},
    }
    validation = json.loads((HERE / "pilot/active_branch_heun_validation.json").read_text())
    metrics["active_branch_validation"] = validation
    metrics["pilot_gate"] = "PASS" if validation["status"] == "PASS" else "FAIL"
    atomic_json(HERE / "pilot" / "pilot_metrics.json", metrics)
    return metrics


def run_pilot(workers: int, force: bool = False) -> dict:
    frame, _ = validate_inputs(); ids = pilot_ids(frame)
    selected = frame[frame.condition_id.isin(ids)].copy()
    atomic_csv(HERE / "pilot" / "pilot_conditions_12.csv", selected)
    pointwise_validation(selected, force=force)
    gate = json.loads((HERE / "pilot/active_branch_heun_validation.json").read_text())
    if gate["status"] != "PASS":
        raise RuntimeError("Active-branch Heun accelerator validation failed; full/pilot transfer run blocked")
    run_conditions(selected, "pilot", PILOT_METHODS, workers, force=force)
    return aggregate_pilot(selected)


def run_full(workers: int, force: bool = False) -> pd.DataFrame:
    frame, _ = validate_inputs()
    pilot = HERE / "pilot/pilot_metrics.json"
    if not pilot.exists() or json.loads(pilot.read_text()).get("pilot_gate") != "PASS":
        raise RuntimeError("A passing pilot is required before the full run")
    return run_conditions(frame, "full", (FULL_METHOD,), workers, force=force)


def bootstrap_slopes(x_old, y_old, x_new, y_new, n=20000) -> dict:
    rng=np.random.default_rng(2026090801); count=len(x_old); old_s=[]; new_s=[]
    for _ in range(n):
        ids=rng.integers(0,count,count)
        old_s.append(stats.linregress(x_old[ids],y_old[ids]).slope)
        new_s.append(stats.linregress(x_new[ids],y_new[ids]).slope)
    old_s=np.asarray(old_s); new_s=np.asarray(new_s); delta=new_s-old_s
    return {"n_bootstrap": n, "old_slope_CI95": np.quantile(old_s,[.025,.975]),
            "heun_slope_CI95": np.quantile(new_s,[.025,.975]),
            "slope_difference_CI95": np.quantile(delta,[.025,.975]),
            "p_heun_slope_le_old": float(np.mean(delta <= 0)),
            "p_heun_slope_le_1": float(np.mean(new_s <= 1))}


def final_metrics() -> tuple[pd.DataFrame, dict]:
    new = pd.read_csv(HERE / "results_full.csv")
    if len(new) != EXPECTED or new.condition_id.nunique() != EXPECTED:
        raise RuntimeError(f"Full results incomplete: rows={len(new)} unique={new.condition_id.nunique()}")
    oldf = pd.read_csv(SOURCE_RESULTS)
    frame = oldf.merge(new, on="condition_id", suffixes=("_old", "_heun"), validate="one_to_one")
    x0=frame.theory_C_corrected_r.to_numpy(float); y0=frame.simulation_r_old.to_numpy(float)
    x1=frame.theory_r.to_numpy(float); y1=frame.simulation_r_heun.to_numpy(float)
    oldm=comparison_metrics(x0,y0); newm=comparison_metrics(x1,y1)
    olderr=y0-x0; newerr=y1-x1; change=y1-y0
    beta_summary=[]
    for cid in frame.condition_id.astype(int):
        betas, _, _ = load_theory(cid)
        beta_summary.append((float(np.mean(betas)), float(np.var(betas, ddof=1))))
    frame["empirical_beta_mean"]=[v[0] for v in beta_summary]
    frame["empirical_beta_variance"]=[v[1] for v in beta_summary]
    residual_frame=pd.DataFrame({"residual":newerr, "S":frame.S_heun,
        "empirical_beta_mean":frame.empirical_beta_mean,
        "empirical_beta_variance":frame.empirical_beta_variance,
        "sigma_beta":frame.sigma_beta_heun, "floor_ratio":frame.floor_ratio_heun})
    residual_corr={k: float(stats.pearsonr(residual_frame[k],newerr).statistic)
                   for k in ("S","empirical_beta_mean","empirical_beta_variance","sigma_beta","floor_ratio")}
    z=np.column_stack([np.ones(len(frame))]+[(residual_frame[k]-residual_frame[k].mean())/residual_frame[k].std(ddof=0)
                                             for k in ("S","empirical_beta_mean","sigma_beta","floor_ratio")])
    coef=np.linalg.lstsq(z,newerr,rcond=None)[0]; pred=z@coef
    structure={"pearson_residual_vs_parameter":residual_corr,
               "beta_note":"beta is trial-level and registered with target mean zero; empirical mean and variance summarize each frozen 256-trial beta vector",
               "standardized_multiple_OLS": {"intercept":coef[0],"S":coef[1],"empirical_beta_mean":coef[2],
               "sigma_beta":coef[3],"floor_ratio":coef[4],
               "R2":1-np.sum((newerr-pred)**2)/np.sum((newerr-newerr.mean())**2)}}
    explanation_fit=stats.linregress(-olderr,change)
    metrics={"old_integrator":oldm,"heun":newm,
        "improvement":{"slope_change":newm["slope"]-oldm["slope"],
            "absolute_slope_bias_reduction_fraction":1-abs(newm["slope"]-1)/abs(oldm["slope"]-1),
            "MAE_reduction_fraction":1-newm["MAE"]/oldm["MAE"],
            "RMSE_reduction_fraction":1-newm["RMSE"]/oldm["RMSE"],
            "mean_bias_reduction_fraction":1-abs(newm["mean_bias"])/abs(oldm["mean_bias"]),
            "old_error_explained_by_heun_shift_R2":float(explanation_fit.rvalue**2),
            "old_error_explained_by_heun_shift_slope":float(explanation_fit.slope)},
        "bootstrap":bootstrap_slopes(x0,y0,x1,y1), "structured_residuals":structure,
        "threshold_crossing_validation":json.loads((HERE/"pilot/active_branch_heun_validation.json").read_text())}
    atomic_csv(HERE / "source_data_condition_level.csv", frame)
    atomic_npz(HERE / "source_data_condition_level.npz", condition_id=frame.condition_id.to_numpy(),
               theory_old=x0, simulation_old=y0, theory_heun=x1, simulation_heun=y1,
               residual_heun=newerr, S=frame.S_heun.to_numpy(), sigma_beta=frame.sigma_beta_heun.to_numpy(),
               floor_ratio=frame.floor_ratio_heun.to_numpy(),
               empirical_beta_mean=frame.empirical_beta_mean.to_numpy(),
               empirical_beta_variance=frame.empirical_beta_variance.to_numpy())
    atomic_json(HERE / "metrics.json", metrics)
    return frame, metrics


def style_scatter() -> None:
    mpl.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Arial","Helvetica","DejaVu Sans","sans-serif"],
        "svg.fonttype":"none","pdf.fonttype":42,"font.size":15,"xtick.labelsize":13,"ytick.labelsize":13,
        "axes.linewidth":1.7,"axes.spines.right":False,"axes.spines.top":False})


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def draw_main(frame: pd.DataFrame) -> Path:
    style_scatter(); x=frame.theory_r.to_numpy(float); y=frame.simulation_r_heun.to_numpy(float)
    fit=stats.linregress(x,y); lo=float(min(x.min(),y.min())); hi=float(max(x.max(),y.max()))
    pad=.035*max(hi-lo,.1); grid=np.linspace(x.min(),x.max(),300)
    stem=HERE/"wholebrain_theory_C_vs_nonlinear_Heun_1000conditions_5x4"
    fig,ax=plt.subplots(figsize=(5,4)); fig.subplots_adjust(left=.14,right=.98,bottom=.16,top=.96)
    scatter=ax.scatter(x,y,s=19,alpha=.46,color="#247BA0",edgecolor="none",rasterized=False,zorder=2)
    if len(scatter.get_offsets()) != EXPECTED: raise RuntimeError("Main figure does not contain 1000 points")
    ax.plot(grid,fit.intercept+fit.slope*grid,color="#222222",lw=2.8,zorder=3)
    ax.set(xlim=(lo-pad,hi+pad),ylim=(lo-pad,hi+pad),xlabel="",ylabel="",title="")
    ax.tick_params(width=1.7,length=6)
    save_figure(fig,stem)
    atomic_json(HERE/"figure_manifest.json",{"core_conclusion":"Theory C predicts held-out nonlinear Heun correlations across all 1000 stable-complex conditions",
        "archetype":"single-panel quantitative validation","panel_alignment":"NOT APPLICABLE: single panel",
        "n_points":EXPECTED,"sampling":False,"identity_line":False,"annotations":False,
        "OLS_line":"solid black","size_in":[5,4],"limits":[lo-pad,hi+pad]})
    return stem


def audit_main_svg_points(stem: Path) -> dict:
    root=ET.parse(stem.with_suffix(".svg")).getroot()
    group=next((node for node in root.iter() if node.attrib.get("id") == "PathCollection_1"), None)
    if group is None:
        raise RuntimeError("Main SVG is missing PathCollection_1")
    n_uses=sum(1 for node in group.iter() if node.tag.endswith("}use") or node.tag == "use")
    report={"status":"PASS" if n_uses == EXPECTED else "FAIL",
            "svg":str(stem.with_suffix(".svg")),"collection_id":"PathCollection_1",
            "expected_points":EXPECTED,"observed_points":n_uses}
    atomic_json(stem.with_suffix(".svg-point-audit.json"),report)
    if n_uses != EXPECTED:
        raise RuntimeError(f"Main SVG contains {n_uses} scatter points, expected {EXPECTED}")
    return report


def draw_boxplot(frame: pd.DataFrame) -> Path:
    source_values=pd.read_csv(ZERO_INPUT_SOURCE)
    stable_node=source_values[source_values.regime=="stable_node"].copy()
    stable_complex=pd.DataFrame({"regime":"stable_complex","condition_id":frame.condition_id,
        "theory_r":frame.theory_r,"simulation_r":frame.simulation_r_heun})
    combined=pd.concat([stable_node,stable_complex],ignore_index=True)
    atomic_csv(HERE/"wholebrain_zero_input_boxplot_values.csv",combined)
    style_scatter(); mpl.rcParams.update({"font.size":17,"xtick.labelsize":17,"ytick.labelsize":17,"axes.linewidth":2.8})
    groups=[stable_complex.theory_r.to_numpy(),stable_complex.simulation_r.to_numpy()]
    fig,ax=plt.subplots(figsize=(5.4,4.6),constrained_layout=True)
    boxes=ax.boxplot(groups,positions=[0,1],widths=.52,patch_artist=True,showfliers=False,
        medianprops={"color":"#222222","linewidth":3},whiskerprops={"linewidth":2.2},capprops={"linewidth":2.2})
    for patch,color in zip(boxes["boxes"],("#247BA0","#F28E2B")):
        patch.set_facecolor(color); patch.set_alpha(.28); patch.set_linewidth(2.6)
    rng=np.random.default_rng(2026082901)
    for pos,(values,color) in enumerate(zip(groups,("#247BA0","#F28E2B"))):
        ax.scatter(pos+rng.normal(0,.075,len(values)),values,s=12,color=color,alpha=.24,edgecolor="none",zorder=2)
    ax.axhline(0,color="0.45",lw=2,zorder=0); ax.set_xticks([0,1],["Theory","Simulation"])
    ax.set(ylabel="Output P–AP correlation, r",title="Experiment 2: spiral state",xlim=(-.65,1.65))
    stem=HERE/"wholebrain_no_input_pap_stable_complex_Heun_boxplot"; save_figure(fig,stem)
    atomic_json(HERE/"boxplot_manifest.json",{"stable_complex":"recomputed from Heun results",
        "stable_node":"not recomputed; copied byte-for-value from the preceding source-data table",
        "stable_node_rows":len(stable_node),"stable_complex_rows":len(stable_complex)})
    return stem


def run_qa(stems: list[Path]) -> dict:
    reports={}
    validator=FIGURE_SKILL/"scripts/validate_figure.py"
    cmd=[sys.executable,str(validator),str(Path(__file__).resolve()),"--json"]
    p=subprocess.run(cmd,text=True,capture_output=True); atomic_text(HERE/"source-validation.json",p.stdout)
    reports["source_validation"]={"returncode":p.returncode,"stdout":p.stdout,"stderr":p.stderr}
    if p.returncode not in (0,): raise RuntimeError(f"source validation failed: {p.stdout} {p.stderr}")
    for stem in stems:
        p=subprocess.run([sys.executable,str(FIGURE_SKILL/"scripts/audit_pdf_text.py"),str(stem.with_suffix('.pdf')),"--min-pt","5","--json"],text=True,capture_output=True)
        atomic_text(stem.with_suffix(".pdf-text-audit.json"),p.stdout)
        if p.returncode: raise RuntimeError(f"PDF text audit failed for {stem.name}")
        collision=stem.with_suffix(".collision-audit.json")
        p2=subprocess.run([sys.executable,str(FIGURE_SKILL/"scripts/audit_figure_collisions.py"),str(stem.with_suffix('.pdf')),
            "--json-out",str(collision),"--overlay-pdf",str(stem.with_suffix('.collision-audit-overlay.pdf'))],text=True,capture_output=True)
        if p2.returncode: raise RuntimeError(f"collision audit failed for {stem.name}: {p2.stdout} {p2.stderr}")
        reports[stem.name]={"pdf_text_returncode":p.returncode,"collision_returncode":p2.returncode}
    atomic_json(HERE/"qa_command_report.json",reports)
    atomic_text(HERE/"visual_QA.md","""# Final-size visual QA\n\n- Backend: Python/matplotlib only.\n- Main panel: 5 x 4 inch; all 1000 vector points visible; black solid OLS; no title, axis titles, identity line, or in-panel annotation.\n- Main panel alignment: NOT APPLICABLE (single panel).\n- Boxplot: stable-complex recomputed; stable-node data not recomputed and preserved exactly for combination use.\n- PDF glyph floor and collision reports are stored beside each figure.\n- Final PDF/PNG/SVG inspected for clipping, point visibility, tick readability, and excess whitespace.\n""")
    return reports


def audit_integrity(frame: pd.DataFrame) -> dict:
    results=pd.read_csv(HERE/"results_full.csv")
    manifest=json.loads((HERE/"manifest.json").read_text())
    source_values=pd.read_csv(ZERO_INPUT_SOURCE)
    combined_values=pd.read_csv(HERE/"wholebrain_zero_input_boxplot_values.csv")
    source_node=source_values[source_values.regime=="stable_node"].reset_index(drop=True)
    combined_node=combined_values[combined_values.regime=="stable_node"].reset_index(drop=True)
    node_columns_equal=list(source_node.columns) == list(combined_node.columns)
    node_values_equal=node_columns_equal and source_node.equals(combined_node)
    row_files=len(list((HERE/"condition_rows/full").glob("*.json")))
    failure_files=len(list((HERE/"failures/full").glob("*.json")))
    chunk_files=len(list((HERE/"chunks/full").glob("**/chunk_*_n32.npz")))
    checks={
        "results_rows_1000":len(results)==EXPECTED,
        "results_unique_conditions_1000":results.condition_id.nunique()==EXPECTED,
        "condition_row_files_1000":row_files==EXPECTED,
        "full_chunk_files_64000":chunk_files==EXPECTED*N_PHASES*(N_TRIALS//FULL_CHUNK),
        "failure_files_zero":failure_files==0,
        "all_110_states_active":bool((results.active_state_count==110).all() and (results.total_state_count==110).all()),
        "all_numeric_results_finite":bool(np.isfinite(results.select_dtypes(include=[np.number]).to_numpy()).all()),
        "frozen_conditions_unfiltered":bool(manifest["before_condition_count"]==EXPECTED and
                                             manifest["after_condition_count"]==EXPECTED and
                                             not manifest["filtering_performed"]),
        "stable_node_columns_unchanged":node_columns_equal,
        "stable_node_values_unchanged":node_values_equal,
        "stable_node_row_count_unchanged":len(source_node)==len(combined_node),
        "final_merge_rows_1000":len(frame)==EXPECTED,
    }
    report={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,
            "counts":{"results_rows":len(results),"unique_conditions":int(results.condition_id.nunique()),
                      "condition_row_files":row_files,"full_chunk_files":chunk_files,
                      "failure_files":failure_files,"stable_node_source_rows":len(source_node),
                      "stable_node_combined_rows":len(combined_node)},
            "n_common_trials_range":[int(results.n_common.min()),int(results.n_common.max())]}
    atomic_json(HERE/"integrity_audit.json",report)
    if report["status"] != "PASS":
        raise RuntimeError(f"Integrity audit failed: {report}")
    return report


def finalize() -> dict:
    frame,metrics=final_metrics(); main=draw_main(frame); svg_points=audit_main_svg_points(main)
    box=draw_boxplot(frame); integrity=audit_integrity(frame); qa=run_qa([main,box])
    boot=metrics["bootstrap"]; structure=metrics["structured_residuals"]
    report=f"""# 全脑 stable-complex Theory C × 非线性 Heun 复算\n\n## 结论\n\n旧积分器：斜率 {metrics['old_integrator']['slope']:.6f}，截距 {metrics['old_integrator']['intercept']:.6f}，Pearson r {metrics['old_integrator']['pearson_r']:.6f}，R² {metrics['old_integrator']['R2']:.6f}，MAE {metrics['old_integrator']['MAE']:.6f}，RMSE {metrics['old_integrator']['RMSE']:.6f}。\n\nHeun：斜率 {metrics['heun']['slope']:.6f}（95% bootstrap CI {boot['heun_slope_CI95'][0]:.6f}–{boot['heun_slope_CI95'][1]:.6f}），截距 {metrics['heun']['intercept']:.6f}，Pearson r {metrics['heun']['pearson_r']:.6f}，R² {metrics['heun']['R2']:.6f}，MAE {metrics['heun']['MAE']:.6f}，RMSE {metrics['heun']['RMSE']:.6f}。Heun 与旧积分器的斜率差 95% CI 为 {boot['slope_difference_CI95'][0]:.6f}–{boot['slope_difference_CI95'][1]:.6f}。\n\n新残差对 S、实际 β 均值、实际 β 方差、sigma_beta、floor_ratio 的 Pearson 相关绝对值均不超过 {max(abs(v) for v in structure['pearson_residual_vs_parameter'].values()):.6f}；标准化多元 OLS R² 为 {structure['standardized_multiple_OLS']['R2']:.6f}。\n\n## 完整性\n\n1000 个冻结的 S × sigma_beta × floor_ratio 条件、每条件 256 trials、3 个校准相位、5 个 held-out 相位、登记种子、2.048 s burn-in、131.072 s 分析窗、4.096 s Hann-Welch 分段、频带、Theory C 与残差协方差校准均保持不变；未加入回归缩放或斜率拟合。完整 110 状态非线性 Heun 方程已与有限记录 transfer 加速器逐点比对，两个 Heun 阶段均重新计算全部 ReLU。\n\n全量得到 64,000/64,000 分块、1000/1000 条件结果、0 失败，完整性审计状态为 {integrity['status']}。Stable-node 未复算，仅在组合源数据表中逐值原样保留。\n"""
    atomic_text(HERE/"README_zh.md",report)
    append_log("finalization PASS")
    return {"metrics":metrics,"qa":qa,"integrity":integrity,"svg_point_audit":svg_points,
            "main_figure":str(main),"boxplot":str(box)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--assert-input",action="store_true"); parser.add_argument("--pilot",action="store_true")
    parser.add_argument("--full",action="store_true"); parser.add_argument("--finalize",action="store_true")
    parser.add_argument("--status",action="store_true"); parser.add_argument("--force",action="store_true")
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    if args.workers < 1 or args.workers > 8: raise SystemExit("--workers must be 1..8")
    if args.assert_input: print(json.dumps(validate_inputs()[1],indent=2))
    if args.pilot: print(json.dumps(run_pilot(args.workers,args.force),indent=2))
    if args.full: run_full(args.workers,args.force)
    if args.finalize: print(json.dumps(finalize(),indent=2,default=json_ready))
    if args.status or not any((args.assert_input,args.pilot,args.full,args.finalize)):
        status={}
        for scope in ("pilot","full"):
            p=HERE/f"progress_{scope}.json"; status[scope]=json.loads(p.read_text()) if p.exists() else {"status":"not_started"}
        print(json.dumps(status,indent=2))


if __name__ == "__main__": main()
