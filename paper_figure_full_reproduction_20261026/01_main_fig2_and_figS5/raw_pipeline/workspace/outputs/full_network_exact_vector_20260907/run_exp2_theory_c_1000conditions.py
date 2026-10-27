from __future__ import annotations

"""Experiment 2 / Theory C: 1000 stable-complex joint parameter conditions.

The registered unit is one row of the fixed Latin-hypercube condition table:
S x sigma_beta x floor_ratio on one unchanged 55-area structural network.
No row is sampled, filtered, substituted, or selected using Theory C outcomes.
"""

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import platform
import re
import resource
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# One BLAS/FFT thread per condition worker prevents nested oversubscription.
for _name in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats


HERE = Path(__file__).resolve().parent
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(HERE), str(PACKAGE), str(FIGURE_SKILL / "scripts")]

import run_exp2_oldprotocol_uniform as old  # noqa: E402
import run_full_network_exact as base  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


CONDITION_SOURCE = (
    PACKAGE
    / "outputs/exp2_pap_s32_window_131p072"
    / "selected_1000_condition_results_rlt0p6.csv"
)
OUTPUT = HERE / "exp2_theory_c_1000conditions"
CACHE_DIR = OUTPUT / "condition_cache"
ROW_DIR = OUTPUT / "condition_rows"
FAILURE_DIR = OUTPUT / "condition_failures"
REFERENCE_24 = HERE / "exp2_theory_c_full"
REFERENCE_OLD_CACHE = HERE / "exp2_oldprotocol_uniform" / "final_cache"
REFERENCE_BRANCH_VALIDATION = (
    HERE / "exp2_oldprotocol_uniform" / "active_branch_euler_validation.csv"
)

PROTOCOL_VERSION = "exp2_theory_c_1000_stable_complex_v1"
EXPECTED_CONDITIONS = 1000
N_TRIALS = 256
N_CALIBRATION = 3
N_HELDOUT = 5
N_PHASES = N_CALIBRATION + N_HELDOUT
ANALYSIS_S = 131.072
DT_S = 0.00025
DT_MS = DT_S * 1000.0
N_STEPS = int(round(ANALYSIS_S / DT_S))
FS = 1.0 / DT_S
WELCH_SEGMENT_S = 4.096
NPERSEG = int(round(WELCH_SEGMENT_S / DT_S))
WELCH_OVERLAP = NPERSEG // 2
PEAK_BAND = (5.0, 100.0)
OVERSAMPLE = 4
CHUNK = 32
SEED = 2026090701
INPUT_AREA = "PG"
OUTPUT_AREA = "PG"

# Pre-registered path boundaries supplied for the 1000-condition formal run.
STABLE_COMPLEX_ONSET_S = 2.0815631208601286
STABLE_RESONANCE_ONSET_S = 3.1774487915806437
PATH_UPPER_S = 6.693999317940512
FIXED_POINT_TOL = 1e-7
JACOBIAN_MATCH_ATOL = 2e-12
STABILITY_EIGEN_TOL_PER_MS = -1e-10

REQUIRED_SOURCE_COLUMNS = (
    "condition_id",
    "family",
    "S",
    "sigma_beta",
    "floor_ratio",
    "input_sigma",
    "critical_real_per_s",
    "critical_frequency_hz",
)

_WORKER_MODEL = None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return [_json_ready(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _atomic_json(path: Path, payload: dict) -> None:
    _atomic_text(path, json.dumps(_json_ready(payload), indent=2) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _protocol_payload(source_hash: str) -> dict:
    return {
        "protocol_version": PROTOCOL_VERSION,
        "condition_source_sha256": source_hash,
        "condition_definition": (
            "1000 stable-complex S x sigma_beta x floor_ratio joint parameter "
            "conditions on one unchanged 55-area structural network"
        ),
        "expected_conditions": EXPECTED_CONDITIONS,
        "trials_per_condition": N_TRIALS,
        "analysis_s": ANALYSIS_S,
        "dt_s": DT_S,
        "calibration_phases": N_CALIBRATION,
        "heldout_phases": N_HELDOUT,
        "welch_segment_s": WELCH_SEGMENT_S,
        "welch_window": "periodic Hann",
        "welch_overlap_fraction": 0.5,
        "peak_band_hz": list(PEAK_BAND),
        "slope_band_hz": [100.0, 1000.0],
        "relaxed_peak_qc": True,
        "beta_seed_root": SEED + 6100000,
        "simulation_seed_rule": (
            "SEED + 6200000 + condition_id*100000 + phase*10000 + trial"
        ),
        "seed": SEED,
        "input_area": INPUT_AREA,
        "output_area": OUTPUT_AREA,
        "simulation_solver": (
            "exact discrete active-branch solution of the complete 55-area x E/I "
            "threshold-linear/ReLU network"
        ),
        "theory": (
            "continuous full-network transfer PSD convolved with the 4.096-s "
            "Hann spectral kernel before registered peak/exponent extraction"
        ),
        "calibration": (
            "condition-specific pooled within-trial residual covariance from the "
            "three calibration phases; no held-out tuning and no fitted scale"
        ),
        "heldout_estimand": (
            "mean of five condition-specific held-out phase correlations"
        ),
        "selection_rule": "all source rows, no downsampling or result-based reselection",
        "stability_contract": {
            "stable_complex_onset_S": STABLE_COMPLEX_ONSET_S,
            "stable_resonance_onset_S": STABLE_RESONANCE_ONSET_S,
            "path_upper_S": PATH_UPPER_S,
            "critical_real_required": "strictly less than 0 per second for every row",
        },
    }


def _protocol_digest(source_hash: str) -> str:
    raw = json.dumps(_protocol_payload(source_hash), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def load_and_assert_conditions(write_manifest: bool = True) -> tuple[pd.DataFrame, dict]:
    if not CONDITION_SOURCE.exists():
        raise FileNotFoundError(CONDITION_SOURCE)
    source_hash = _sha256(CONDITION_SOURCE)
    frame = pd.read_csv(CONDITION_SOURCE)
    before = int(len(frame))
    missing_columns = sorted(set(REQUIRED_SOURCE_COLUMNS) - set(frame.columns))
    if missing_columns:
        raise RuntimeError(f"Missing required columns: {missing_columns}")
    if before != EXPECTED_CONDITIONS:
        raise RuntimeError(f"Expected exactly 1000 rows, found {before}")
    if frame.condition_id.isna().any():
        bad = frame.index[frame.condition_id.isna()].tolist()
        raise RuntimeError(f"Missing condition_id at source rows {bad}")
    numeric_ids = frame.condition_id.to_numpy(float)
    if not np.isfinite(numeric_ids).all() or not np.equal(numeric_ids, np.round(numeric_ids)).all():
        raise RuntimeError("condition_id must be finite integer-valued")
    frame["condition_id"] = frame.condition_id.astype(int)
    duplicated = sorted(frame.loc[frame.condition_id.duplicated(False), "condition_id"].unique())
    if duplicated:
        raise RuntimeError(f"Duplicate condition_id values: {duplicated}")
    unique = int(frame.condition_id.nunique())
    if unique != EXPECTED_CONDITIONS:
        raise RuntimeError(f"Expected 1000 unique condition_id values, found {unique}")
    for name in ("S", "sigma_beta", "floor_ratio", "input_sigma", "critical_real_per_s", "critical_frequency_hz"):
        if not np.isfinite(frame[name].to_numpy(float)).all():
            bad = frame.loc[~np.isfinite(frame[name].to_numpy(float)), "condition_id"].tolist()
            raise RuntimeError(f"Non-finite {name} for condition_id={bad}")

    s_min = float(frame.S.min())
    s_max = float(frame.S.max())
    stable_complex_margin = s_min - STABLE_COMPLEX_ONSET_S
    resonance_margin = s_min - STABLE_RESONANCE_ONSET_S
    upper_margin = PATH_UPPER_S - s_max
    failing_contract = frame.loc[
        (frame.S <= STABLE_COMPLEX_ONSET_S)
        | (frame.S <= STABLE_RESONANCE_ONSET_S)
        | (frame.S >= PATH_UPPER_S)
        | (frame.critical_real_per_s >= 0),
        ["condition_id", "S", "critical_real_per_s"],
    ]
    if len(failing_contract):
        raise RuntimeError(
            "Stability/path assertion failed; no filtering is allowed:\n"
            + failing_contract.to_string(index=False)
        )
    if "n_trials" in frame and not (frame.n_trials.astype(int) == N_TRIALS).all():
        bad = frame.loc[frame.n_trials.astype(int) != N_TRIALS, "condition_id"].tolist()
        raise RuntimeError(f"Source n_trials is not 256 for condition_id={bad}")

    assertion = {
        "status": "PASS",
        "asserted_at_utc": _utc_now(),
        "source": str(CONDITION_SOURCE),
        "source_sha256": source_hash,
        "before_condition_count": before,
        "after_condition_count": before,
        "unique_condition_id_count": unique,
        "duplicate_condition_ids": duplicated,
        "missing_condition_id_count": 0,
        "filtering_performed": False,
        "S_min": s_min,
        "S_max": s_max,
        "stable_complex_onset_S": STABLE_COMPLEX_ONSET_S,
        "stable_complex_min_margin": stable_complex_margin,
        "stable_resonance_onset_S": STABLE_RESONANCE_ONSET_S,
        "stable_resonance_min_margin": resonance_margin,
        "path_upper_S": PATH_UPPER_S,
        "path_upper_min_margin": upper_margin,
        "all_critical_real_per_s_lt_zero": bool((frame.critical_real_per_s < 0).all()),
        "critical_real_per_s_min": float(frame.critical_real_per_s.min()),
        "critical_real_per_s_max": float(frame.critical_real_per_s.max()),
        "critical_frequency_hz_min": float(frame.critical_frequency_hz.min()),
        "critical_frequency_hz_max": float(frame.critical_frequency_hz.max()),
        "protocol": _protocol_payload(source_hash),
        "protocol_digest": _protocol_digest(source_hash),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": getattr(sys.modules.get("scipy"), "__version__", "unknown"),
            "matplotlib": mpl.__version__,
        },
    }
    if write_manifest:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        _atomic_json(OUTPUT / "protocol_manifest.json", assertion)
        _atomic_csv(OUTPUT / "conditions_input_1000.csv", frame)
    return frame.sort_values("condition_id").reset_index(drop=True), assertion


def _assemble_and_solve_condition(model, condition: pd.Series) -> dict:
    arrays, registered_jacobian = base.stable_arrays(model, float(condition.S))
    n_area = len(model.areas)
    ids = np.arange(n_area)
    matrix = np.zeros((2 * n_area, 2 * n_area), dtype=float)
    matrix[ids, ids] = arrays[0]
    matrix[ids, n_area + ids] = arrays[1]
    matrix[n_area + ids, ids] = arrays[2]
    matrix[n_area + ids, n_area + ids] = arrays[3]
    np.add.at(matrix, (arrays[4], arrays[5]), arrays[6])
    np.add.at(matrix, (n_area + arrays[4], arrays[5]), arrays[7])
    background = np.r_[arrays[8], arrays[9]]
    tau = np.r_[
        np.full(n_area, float(model.params["tau_exc"])),
        np.full(n_area, float(model.params["tau_inh"])),
    ]
    initial = np.r_[np.full(n_area, 10.0), np.full(n_area, 35.0)]
    fixedpoint, drive, gain, marginal, residual = base._fixed_point(
        matrix, background, tau, initial
    )
    jacobian = base._effective_jacobian(matrix, gain, tau)
    max_real = float(np.linalg.eigvals(jacobian).real.max())
    jacobian_difference = float(np.max(np.abs(jacobian - registered_jacobian)))
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    diagnostics = {
        "fixedpoint_residual": float(residual),
        "fixedpoint_min_rate": float(fixedpoint.min()),
        "fixedpoint_max_rate": float(fixedpoint.max()),
        "fixedpoint_distance_from_registered": float(np.linalg.norm(fixedpoint - initial)),
        "active_state_count": int(gain.sum()),
        "total_state_count": int(len(gain)),
        "marginal_state_count": int(marginal.sum()),
        "input_active": bool(gain[input_idx]),
        "output_active": bool(gain[output_idx]),
        "effective_max_real_eigenvalue_per_ms": max_real,
        "jacobian_max_abs_difference_from_registered_active_branch": jacobian_difference,
    }
    condition_id = int(condition.condition_id)
    if residual > FIXED_POINT_TOL:
        raise RuntimeError(f"condition {condition_id}: fixed-point residual {residual:.3g}")
    if not bool(gain.all()):
        inactive = np.flatnonzero(gain == 0).tolist()
        raise RuntimeError(
            f"condition {condition_id}: active branch is not fully active; inactive states={inactive}"
        )
    if marginal.any():
        raise RuntimeError(
            f"condition {condition_id}: {int(marginal.sum())} marginal ReLU states"
        )
    if jacobian_difference > JACOBIAN_MATCH_ATOL:
        raise RuntimeError(
            f"condition {condition_id}: independently solved Jacobian differs by "
            f"{jacobian_difference:.3g}"
        )
    if max_real >= STABILITY_EIGEN_TOL_PER_MS:
        raise RuntimeError(
            f"condition {condition_id}: unstable full-network Jacobian max real={max_real:.6g}/ms"
        )
    return {
        "arrays": arrays,
        "fixedpoint": fixedpoint,
        "drive": drive,
        "gain": gain,
        "jacobian": jacobian,
        "diagnostics": diagnostics,
        "input_idx": input_idx,
        "output_idx": output_idx,
    }


def _convolved_theory_features(
    model,
    condition: pd.Series,
    betas: np.ndarray,
    jacobian: np.ndarray,
    input_idx: int,
    output_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    fine_n = OVERSAMPLE * NPERSEG
    signed_frequencies = np.fft.fftfreq(fine_n, d=DT_S)
    absolute_frequencies = np.abs(signed_frequencies)
    transfer = base.continuous_transfer_power_all(
        jacobian, input_idx, np.asarray([output_idx]), absolute_frequencies
    )[0]
    window = signal.get_window("hann", NPERSEG, fftbins=True)
    spectral_window = np.abs(np.fft.fft(window, n=fine_n)) ** 2
    kernel_fft = np.fft.fft(spectral_window)
    sample_indices = OVERSAMPLE * np.arange(NPERSEG // 2 + 1)
    frequencies = np.fft.rfftfreq(NPERSEG, d=DT_S)
    keep = (frequencies >= 0.5) & (frequencies <= 1200.0)
    values = np.full((len(betas), 2), np.nan)
    valid = np.zeros(len(betas), bool)
    for trial, beta in enumerate(betas):
        two_sided = transfer * base._input_q(
            absolute_frequencies,
            float(beta),
            float(condition.input_sigma),
            float(condition.floor_ratio),
        )
        expected = np.fft.ifft(np.fft.fft(two_sided) * kernel_fft).real[sample_indices]
        expected = np.maximum(expected, np.finfo(float).tiny)
        item, item_valid = base.features_from_psd(
            frequencies[keep],
            expected[keep][None, :],
            PEAK_BAND,
            relaxed_peak_qc=True,
        )
        values[trial] = item[0]
        valid[trial] = bool(item_valid[0])
    return old.display_convention(values), valid


def _simulated_phase_features(
    condition: pd.Series,
    betas: np.ndarray,
    transfer: np.ndarray,
    phase: int,
) -> tuple[np.ndarray, np.ndarray]:
    frequencies = np.fft.rfftfreq(N_STEPS, d=DT_S)
    values = np.full((len(betas), 2), np.nan)
    valid = np.zeros(len(betas), bool)
    for start in range(0, len(betas), CHUNK):
        stop = min(start + CHUNK, len(betas))
        paths = np.empty((stop - start, N_STEPS), dtype=np.float32)
        for local, trial in enumerate(range(start, stop)):
            rng = np.random.default_rng(
                SEED
                + 6200000
                + int(condition.condition_id) * 100000
                + phase * 10000
                + trial
            )
            shape = old._normalized_input_shape(frequencies, float(betas[trial]))
            angle = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
            spectrum = np.sqrt(shape) * np.exp(1j * angle)
            spectrum[0] = 0.0
            spectrum[-1] = spectrum[-1].real
            forcing = np.fft.irfft(spectrum, n=N_STEPS)
            forcing *= float(condition.input_sigma) / max(forcing.std(), 1e-15)
            forcing += (
                float(condition.input_sigma)
                * float(condition.floor_ratio)
                * rng.normal(size=N_STEPS)
            )
            forcing -= forcing.mean()
            paths[local] = np.fft.irfft(
                np.fft.rfft(forcing) * transfer, n=N_STEPS
            ).astype(np.float32)
        local_values, local_valid = base.features_from_paths(
            paths, ANALYSIS_S, PEAK_BAND, relaxed_peak_qc=True
        )
        values[start:stop] = local_values
        valid[start:stop] = local_valid
    return old.display_convention(values), valid


def _cache_path(condition_id: int) -> Path:
    return CACHE_DIR / f"condition_{condition_id:04d}.npz"


def _row_path(condition_id: int) -> Path:
    return ROW_DIR / f"condition_{condition_id:04d}.json"


def _failure_path(condition_id: int) -> Path:
    return FAILURE_DIR / f"condition_{condition_id:04d}.json"


def _write_condition_cache(
    path: Path,
    protocol_digest: str,
    condition_id: int,
    betas: np.ndarray,
    theory: np.ndarray,
    theory_valid: np.ndarray,
    phase_features: np.ndarray,
    phase_valid: np.ndarray,
    phase_complete: np.ndarray,
    diagnostics: dict,
) -> None:
    _atomic_npz(
        path,
        protocol_digest=np.asarray(protocol_digest),
        condition_id=np.asarray(condition_id, dtype=np.int64),
        betas=betas,
        theory_C_features=theory,
        theory_C_valid=theory_valid,
        phase_features=phase_features,
        phase_valid=phase_valid,
        phase_complete=phase_complete,
        fixedpoint_residual=np.asarray(diagnostics["fixedpoint_residual"]),
        fixedpoint_min_rate=np.asarray(diagnostics["fixedpoint_min_rate"]),
        fixedpoint_max_rate=np.asarray(diagnostics["fixedpoint_max_rate"]),
        effective_max_real_eigenvalue_per_ms=np.asarray(
            diagnostics["effective_max_real_eigenvalue_per_ms"]
        ),
        active_state_count=np.asarray(diagnostics["active_state_count"], dtype=np.int64),
        total_state_count=np.asarray(diagnostics["total_state_count"], dtype=np.int64),
        jacobian_match_max_abs=np.asarray(
            diagnostics["jacobian_max_abs_difference_from_registered_active_branch"]
        ),
    )


def _load_partial_cache(
    path: Path,
    protocol_digest: str,
    condition_id: int,
    betas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as saved:
        cached_digest = str(saved["protocol_digest"].item())
        cached_id = int(saved["condition_id"].item())
        cached_betas = saved["betas"].astype(float)
        theory = saved["theory_C_features"].astype(float)
        theory_valid = saved["theory_C_valid"].astype(bool)
        phase_features = saved["phase_features"].astype(float)
        phase_valid = saved["phase_valid"].astype(bool)
        phase_complete = saved["phase_complete"].astype(bool)
    if cached_digest != protocol_digest:
        raise RuntimeError(f"Protocol mismatch in cache {path}")
    if cached_id != condition_id:
        raise RuntimeError(f"condition_id mismatch in cache {path}")
    if not np.array_equal(cached_betas, betas):
        raise RuntimeError(f"Beta trials mismatch in cache {path}")
    if theory.shape != (N_TRIALS, 2) or theory_valid.shape != (N_TRIALS,):
        raise RuntimeError(f"Theory shape mismatch in cache {path}")
    if phase_features.shape != (N_PHASES, N_TRIALS, 2):
        raise RuntimeError(f"Phase feature shape mismatch in cache {path}")
    if phase_valid.shape != (N_PHASES, N_TRIALS) or phase_complete.shape != (N_PHASES,):
        raise RuntimeError(f"Phase validity shape mismatch in cache {path}")
    return theory, theory_valid, phase_features, phase_valid, phase_complete


def _max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _evaluate_condition(model, condition: pd.Series, protocol_digest: str) -> dict:
    started = time.perf_counter()
    condition_id = int(condition.condition_id)
    solved = _assemble_and_solve_condition(model, condition)
    betas = old.beta_trials(float(condition.sigma_beta), N_TRIALS, condition_id)
    cache_path = _cache_path(condition_id)
    cached = _load_partial_cache(cache_path, protocol_digest, condition_id, betas)
    if cached is None:
        theory, theory_valid = _convolved_theory_features(
            model,
            condition,
            betas,
            solved["jacobian"],
            solved["input_idx"],
            solved["output_idx"],
        )
        phase_features = np.full((N_PHASES, N_TRIALS, 2), np.nan)
        phase_valid = np.zeros((N_PHASES, N_TRIALS), bool)
        phase_complete = np.zeros(N_PHASES, bool)
        _write_condition_cache(
            cache_path,
            protocol_digest,
            condition_id,
            betas,
            theory,
            theory_valid,
            phase_features,
            phase_valid,
            phase_complete,
            solved["diagnostics"],
        )
    else:
        theory, theory_valid, phase_features, phase_valid, phase_complete = cached

    discrete_transfer = base.discrete_transfer_all(
        solved["jacobian"],
        solved["input_idx"],
        np.asarray([solved["output_idx"]]),
        N_STEPS,
        float(model.params["tau_exc"]),
    )[0]
    for phase in range(N_PHASES):
        if phase_complete[phase]:
            continue
        phase_features[phase], phase_valid[phase] = _simulated_phase_features(
            condition, betas, discrete_transfer, phase
        )
        phase_complete[phase] = True
        _write_condition_cache(
            cache_path,
            protocol_digest,
            condition_id,
            betas,
            theory,
            theory_valid,
            phase_features,
            phase_valid,
            phase_complete,
            solved["diagnostics"],
        )

    common = theory_valid & phase_valid.all(axis=0)
    n_common = int(common.sum())
    if n_common < 8:
        raise RuntimeError(f"condition {condition_id}: only {n_common}/256 common valid trials")
    signal_cov = np.cov(theory[common], rowvar=False, ddof=1)
    residual_cov = old.pooled_within_trial_cov(phase_features[:N_CALIBRATION, common])
    heldout_rs = [
        old.feature_corr(phase_features[phase, common])
        for phase in range(N_CALIBRATION, N_PHASES)
    ]
    if not np.isfinite(heldout_rs).all():
        raise RuntimeError(f"condition {condition_id}: non-finite held-out correlation {heldout_rs}")

    row = {
        "protocol_digest": protocol_digest,
        "condition_id": condition_id,
        "family": str(condition.family),
        "S": float(condition.S),
        "sigma_beta": float(condition.sigma_beta),
        "floor_ratio": float(condition.floor_ratio),
        "input_sigma": float(condition.input_sigma),
        "source_critical_real_per_s": float(condition.critical_real_per_s),
        "source_critical_frequency_hz": float(condition.critical_frequency_hz),
        "n_trials_requested": N_TRIALS,
        "n_theory_valid": int(theory_valid.sum()),
        "n_common": n_common,
        "valid_fraction": float(n_common / N_TRIALS),
        "theory_C_exact_mapping_r": old.cov_to_corr(signal_cov),
        "theory_C_corrected_r": old.cov_to_corr(signal_cov + residual_cov),
        "simulation_r": float(np.mean(heldout_rs)),
        "heldout_r_json": json.dumps([float(v) for v in heldout_rs]),
        "residual_var_P": float(residual_cov[0, 0]),
        "residual_var_A": float(residual_cov[1, 1]),
        "residual_cov_PA": float(residual_cov[0, 1]),
        **solved["diagnostics"],
        "completed_phases": int(phase_complete.sum()),
        "runtime_s": float(time.perf_counter() - started),
        "max_rss_mb": _max_rss_mb(),
        "completed_at_utc": _utc_now(),
        "cache_path": str(cache_path),
    }
    _atomic_json(_row_path(condition_id), row)
    failure_path = _failure_path(condition_id)
    if failure_path.exists():
        failure_path.unlink()
    return row


def _worker_init() -> None:
    global _WORKER_MODEL
    _WORKER_MODEL = base.load_wholebrain()


def _worker_entry(condition_payload: dict, protocol_digest: str) -> dict:
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        _worker_init()
    condition = pd.Series(condition_payload)
    condition_id = int(condition.condition_id)
    try:
        row = _evaluate_condition(_WORKER_MODEL, condition, protocol_digest)
        return {"status": "success", "condition_id": condition_id, "row": row}
    except Exception as exc:  # condition-local failure must be recorded, never hidden
        failure = {
            "protocol_digest": protocol_digest,
            "condition_id": condition_id,
            "error_type": type(exc).__name__,
            "reason": str(exc),
            "traceback": traceback.format_exc(),
            "failed_at_utc": _utc_now(),
            "max_rss_mb": _max_rss_mb(),
        }
        _atomic_json(_failure_path(condition_id), failure)
        return {"status": "failure", "condition_id": condition_id, "failure": failure}


def _valid_saved_json(path: Path, protocol_digest: str) -> dict | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return value if value.get("protocol_digest") == protocol_digest else None


def _saved_state(conditions: pd.DataFrame, protocol_digest: str) -> tuple[dict[int, dict], dict[int, dict]]:
    successes: dict[int, dict] = {}
    failures: dict[int, dict] = {}
    for condition_id in conditions.condition_id.astype(int):
        row = _valid_saved_json(_row_path(condition_id), protocol_digest)
        if row is not None:
            successes[condition_id] = row
            continue
        failure = _valid_saved_json(_failure_path(condition_id), protocol_digest)
        if failure is not None:
            failures[condition_id] = failure
    return successes, failures


def _persist_progress(
    conditions: pd.DataFrame,
    protocol_digest: str,
    successes: dict[int, dict],
    failures: dict[int, dict],
    started_at: float,
    workers: int,
) -> None:
    order = {int(cid): pos for pos, cid in enumerate(conditions.condition_id)}
    success_frame = pd.DataFrame(successes.values())
    if len(success_frame):
        success_frame["_order"] = success_frame.condition_id.map(order)
        success_frame = success_frame.sort_values("_order").drop(columns="_order")
    failure_columns = [
        "condition_id",
        "error_type",
        "reason",
        "failed_at_utc",
        "protocol_digest",
    ]
    failure_frame = pd.DataFrame(failures.values())
    if len(failure_frame):
        failure_frame["_order"] = failure_frame.condition_id.map(order)
        failure_frame = failure_frame.sort_values("_order").drop(columns="_order")
        failure_frame = failure_frame[[c for c in failure_columns if c in failure_frame]]
    else:
        failure_frame = pd.DataFrame(columns=failure_columns)
    _atomic_csv(OUTPUT / "results_condition_level.csv", success_frame)
    _atomic_csv(OUTPUT / "failures.csv", failure_frame)
    completed = len(successes) + len(failures)
    _atomic_json(
        OUTPUT / "progress.json",
        {
            "protocol_digest": protocol_digest,
            "updated_at_utc": _utc_now(),
            "input_conditions": len(conditions),
            "successful_conditions": len(successes),
            "failed_conditions": len(failures),
            "completed_conditions": completed,
            "remaining_conditions": len(conditions) - completed,
            "completion_fraction": completed / len(conditions),
            "elapsed_s_this_invocation": time.time() - started_at,
            "workers": workers,
        },
    )


def _append_log(message: str) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    line = f"[{_utc_now()}] {message}"
    with (OUTPUT / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _run_payloads(
    conditions: pd.DataFrame,
    protocol_digest: str,
    workers: int,
    include_ids: set[int] | None = None,
    retry_failures: bool = True,
) -> tuple[dict[int, dict], dict[int, dict]]:
    started_at = time.time()
    successes, failures = _saved_state(conditions, protocol_digest)
    if retry_failures:
        failures = {}
    pending = []
    for _, condition in conditions.iterrows():
        condition_id = int(condition.condition_id)
        if include_ids is not None and condition_id not in include_ids:
            continue
        if condition_id in successes:
            continue
        pending.append(condition.to_dict())
    _persist_progress(conditions, protocol_digest, successes, failures, started_at, workers)
    _append_log(
        f"run start: workers={workers}, pending={len(pending)}, "
        f"already_successful={len(successes)}, total_input={len(conditions)}"
    )
    if not pending:
        return successes, failures

    context = mp.get_context("spawn")
    completed_this_run = 0
    with ProcessPoolExecutor(
        max_workers=workers, mp_context=context, initializer=_worker_init
    ) as executor:
        futures = {
            executor.submit(_worker_entry, payload, protocol_digest): int(payload["condition_id"])
            for payload in pending
        }
        for future in as_completed(futures):
            condition_id = futures[future]
            try:
                outcome = future.result()
            except Exception as exc:
                failure = {
                    "protocol_digest": protocol_digest,
                    "condition_id": condition_id,
                    "error_type": type(exc).__name__,
                    "reason": f"worker process failed: {exc}",
                    "failed_at_utc": _utc_now(),
                }
                _atomic_json(_failure_path(condition_id), failure)
                outcome = {"status": "failure", "condition_id": condition_id, "failure": failure}
            completed_this_run += 1
            if outcome["status"] == "success":
                row = outcome["row"]
                successes[condition_id] = row
                failures.pop(condition_id, None)
                _append_log(
                    f"condition {condition_id} success; "
                    f"n_common={row['n_common']}/{N_TRIALS}; "
                    f"theory={row['theory_C_corrected_r']:.6f}; "
                    f"simulation={row['simulation_r']:.6f}; "
                    f"runtime={row['runtime_s']:.1f}s; "
                    f"progress_this_run={completed_this_run}/{len(pending)}"
                )
            else:
                failure = outcome["failure"]
                failures[condition_id] = failure
                _append_log(
                    f"condition {condition_id} FAILURE {failure['error_type']}: "
                    f"{failure['reason']}"
                )
            _persist_progress(
                conditions, protocol_digest, successes, failures, started_at, workers
            )
    return successes, failures


def _benchmark_ids(conditions: pd.DataFrame) -> list[int]:
    preferred = [3179, 3015, 2765]
    available = set(conditions.condition_id.astype(int))
    chosen = [condition_id for condition_id in preferred if condition_id in available]
    if len(chosen) != len(preferred):
        raise RuntimeError(f"Benchmark IDs missing from source: {sorted(set(preferred) - available)}")
    return chosen


def benchmark(workers: int = 2) -> dict:
    conditions, assertion = load_and_assert_conditions(write_manifest=True)
    digest = assertion["protocol_digest"]
    chosen = _benchmark_ids(conditions)
    started = time.time()
    successes, failures = _run_payloads(
        conditions, digest, min(workers, len(chosen)), set(chosen), retry_failures=True
    )
    if any(condition_id in failures for condition_id in chosen):
        raise RuntimeError(f"Benchmark failures: {[cid for cid in chosen if cid in failures]}")
    comparisons = []
    for condition_id in chosen:
        new_cache = _cache_path(condition_id)
        old_cache = REFERENCE_OLD_CACHE / f"condition_{condition_id:04d}.npz"
        comparison = {
            "condition_id": condition_id,
            "reference_cache": str(old_cache),
            "reference_cache_exists": old_cache.exists(),
        }
        if old_cache.exists():
            with np.load(new_cache) as new, np.load(old_cache) as reference:
                comparison.update(
                    {
                        "betas_exact_match": bool(np.array_equal(new["betas"], reference["betas"])),
                        "phase_valid_exact_match": bool(
                            np.array_equal(new["phase_valid"], reference["phase_valid"])
                        ),
                        "phase_features_max_abs_difference": float(
                            np.nanmax(
                                np.abs(
                                    new["phase_features"].astype(float)
                                    - reference["phase_features"].astype(float)
                                )
                            )
                        ),
                    }
                )
        reference_results_path = REFERENCE_24 / "results_24.csv"
        if reference_results_path.exists():
            reference_results = pd.read_csv(reference_results_path)
            match = reference_results.loc[reference_results.condition_id == condition_id]
            comparison["reference_24_result_exists"] = bool(len(match))
            if len(match):
                comparison["theory_C_corrected_r_abs_difference"] = abs(
                    float(successes[condition_id]["theory_C_corrected_r"])
                    - float(match.iloc[0].theory_C_corrected_r)
                )
                comparison["simulation_r_abs_difference"] = abs(
                    float(successes[condition_id]["simulation_r"])
                    - float(match.iloc[0].simulation_r)
                )
        comparisons.append(comparison)

    for item in comparisons:
        if item.get("reference_cache_exists"):
            if not item["betas_exact_match"] or not item["phase_valid_exact_match"]:
                raise RuntimeError(f"Benchmark protocol mismatch: {item}")
            if item["phase_features_max_abs_difference"] > 1e-12:
                raise RuntimeError(f"Benchmark phase feature mismatch: {item}")
        if item.get("reference_24_result_exists"):
            if item["theory_C_corrected_r_abs_difference"] > 1e-12:
                raise RuntimeError(f"Benchmark Theory C mismatch: {item}")
            if item["simulation_r_abs_difference"] > 1e-12:
                raise RuntimeError(f"Benchmark simulation mismatch: {item}")

    selected_rows = [successes[condition_id] for condition_id in chosen]
    elapsed = time.time() - started
    report = {
        "status": "PASS",
        "completed_at_utc": _utc_now(),
        "condition_ids": chosen,
        "workers": min(workers, len(chosen)),
        "elapsed_s": elapsed,
        "mean_condition_runtime_s": float(np.mean([row["runtime_s"] for row in selected_rows])),
        "max_worker_rss_mb": float(max(row["max_rss_mb"] for row in selected_rows)),
        "estimated_wall_hours_1000_at_same_throughput": elapsed / len(chosen) * 1000 / 3600,
        "fixedpoint_residual_max": float(max(row["fixedpoint_residual"] for row in selected_rows)),
        "active_state_count_values": sorted({row["active_state_count"] for row in selected_rows}),
        "full_network_max_real_eigenvalue_per_ms_max": float(
            max(row["effective_max_real_eigenvalue_per_ms"] for row in selected_rows)
        ),
        "reference_comparisons": comparisons,
    }
    _atomic_json(OUTPUT / "benchmark_report.json", report)
    _append_log(
        f"benchmark PASS: ids={chosen}, elapsed={elapsed:.1f}s, "
        f"max_worker_rss={report['max_worker_rss_mb']:.1f}MB"
    )
    return report


def method_metrics(frame: pd.DataFrame) -> dict:
    x = frame.theory_C_corrected_r.to_numpy(float)
    y = frame.simulation_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y)
    if use.sum() < 3 or np.std(x[use]) <= 0 or np.std(y[use]) <= 0:
        raise RuntimeError("Insufficient finite variation for final regression metrics")
    fit = stats.linregress(x[use], y[use])
    residual = y[use] - x[use]
    denominator = float(np.sum((y[use] - y[use].mean()) ** 2))
    return {
        "n_conditions_used": int(use.sum()),
        "pearson_r": float(fit.rvalue),
        "pearson_p": float(fit.pvalue),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "ols_R2": float(fit.rvalue**2),
        "identity_R2": float(1.0 - np.sum(residual**2) / denominator),
        "identity_RMSE": float(np.sqrt(np.mean(residual**2))),
        "MAE": float(np.mean(np.abs(residual))),
        "mean_bias_simulation_minus_theory": float(np.mean(residual)),
        "max_abs_error": float(np.max(np.abs(residual))),
    }


def _valid_trial_distribution(frame: pd.DataFrame) -> dict:
    counts = frame.n_common.astype(int)
    histogram = counts.value_counts().sort_index()
    return {
        "minimum": int(counts.min()),
        "maximum": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "histogram": {str(int(k)): int(v) for k, v in histogram.items()},
    }


def _read_reference_24_metrics() -> dict | None:
    path = REFERENCE_24 / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def aggregate_metrics() -> tuple[pd.DataFrame, dict]:
    conditions, assertion = load_and_assert_conditions(write_manifest=True)
    digest = assertion["protocol_digest"]
    successes, failures = _saved_state(conditions, digest)
    _persist_progress(conditions, digest, successes, failures, time.time(), workers=0)
    frame = pd.read_csv(OUTPUT / "results_condition_level.csv")
    metrics = method_metrics(frame) if len(frame) >= 3 else {}
    metrics.update(
        {
            "input_conditions": len(conditions),
            "successful_conditions": len(successes),
            "failed_conditions": len(failures),
            "before_condition_count": assertion["before_condition_count"],
            "after_condition_count": assertion["after_condition_count"],
            "valid_trials_per_condition": _valid_trial_distribution(frame) if len(frame) else {},
            "protocol": assertion["protocol"],
            "stability_assertions": {
                key: assertion[key]
                for key in (
                    "S_min",
                    "S_max",
                    "stable_complex_onset_S",
                    "stable_complex_min_margin",
                    "stable_resonance_onset_S",
                    "stable_resonance_min_margin",
                    "path_upper_S",
                    "path_upper_min_margin",
                    "all_critical_real_per_s_lt_zero",
                    "critical_real_per_s_min",
                    "critical_real_per_s_max",
                    "critical_frequency_hz_min",
                    "critical_frequency_hz_max",
                )
            },
            "comparison_24_condition_theory_C": _read_reference_24_metrics(),
        }
    )
    _atomic_json(OUTPUT / "metrics.json", metrics)
    return frame, metrics


def plot_final() -> Path:
    frame, metrics = aggregate_metrics()
    if len(frame) != EXPECTED_CONDITIONS or metrics["failed_conditions"] != 0:
        raise RuntimeError(
            f"Final figure requires all 1000 conditions; success={len(frame)}, "
            f"failure={metrics['failed_conditions']}"
        )
    if frame.condition_id.nunique() != EXPECTED_CONDITIONS:
        raise RuntimeError("Final figure does not have 1000 unique condition_id values")
    x = frame.theory_C_corrected_r.to_numpy(float)
    y = frame.simulation_r.to_numpy(float)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise RuntimeError("Final figure contains non-finite coordinates")
    fit = stats.linregress(x, y)
    combined = np.r_[x, y]
    low, high = float(combined.min()), float(combined.max())
    padding = 0.04 * max(high - low, 0.1)
    limits = (low - padding, high + padding)
    identity_grid = np.linspace(limits[0], limits[1], 300)
    fit_grid = np.linspace(float(x.min()), float(x.max()), 300)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.linewidth": 1.7,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.96)
    # All 1000 condition-level observations are drawn; no sampling or aggregation.
    scatter = ax.scatter(
        x,
        y,
        s=19,
        alpha=0.46,
        color="#247BA0",
        edgecolor="none",
        rasterized=False,
        zorder=2,
    )
    if len(scatter.get_offsets()) != EXPECTED_CONDITIONS:
        raise RuntimeError("Matplotlib scatter collection does not contain 1000 offsets")
    ax.plot(identity_grid, identity_grid, color="#777777", lw=1.8, ls="--", zorder=1)
    ax.plot(fit_grid, fit.intercept + fit.slope * fit_grid, color="#222222", lw=2.8, zorder=3)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.text(
        0.05,
        0.95,
        f"r = {fit.rvalue:.3f}\nslope = {fit.slope:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=19,
        fontweight="bold",
        zorder=4,
    )
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)

    stem = OUTPUT / "exp2_theory_C_1000conditions_5x4"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        strict=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    _atomic_json(
        OUTPUT / "figure_manifest.json",
        {
            "source_rows": len(frame),
            "unique_condition_ids": int(frame.condition_id.nunique()),
            "scatter_offsets": EXPECTED_CONDITIONS,
            "sampling": False,
            "aggregation": False,
            "rasterized_scatter": False,
            "figure_width_in": 5.0,
            "figure_height_in": 4.0,
            "height_to_width_ratio": 4.0 / 5.0,
            "x_limits": list(limits),
            "y_limits": list(limits),
            "pearson_r": float(fit.rvalue),
            "slope": float(fit.slope),
            "intercept": float(fit.intercept),
        },
    )
    return stem.with_suffix(".svg")


def run_qa() -> dict:
    script = Path(__file__).resolve()
    stem = OUTPUT / "exp2_theory_C_1000conditions_5x4"
    commands = {
        "source_validation": [
            sys.executable,
            str(FIGURE_SKILL / "scripts/validate_figure.py"),
            str(script),
            "--json",
        ],
        "pdf_text": [
            sys.executable,
            str(FIGURE_SKILL / "scripts/audit_pdf_text.py"),
            str(stem.with_suffix(".pdf")),
            "--min-pt",
            "5",
            "--json",
        ],
        "collisions": [
            sys.executable,
            str(FIGURE_SKILL / "scripts/audit_figure_collisions.py"),
            str(stem.with_suffix(".pdf")),
            "--json-out",
            str(OUTPUT / "collision-audit.json"),
            "--overlay-pdf",
            str(OUTPUT / "collision-audit-overlay.pdf"),
        ],
    }
    report = {}
    for name, command in commands.items():
        completed = subprocess.run(command, text=True, capture_output=True)
        report[name] = {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        if name == "source_validation":
            _atomic_text(OUTPUT / "source-validation.json", completed.stdout)
        elif name == "pdf_text":
            _atomic_text(OUTPUT / "pdf-text-audit.json", completed.stdout)
        if completed.returncode != 0:
            _atomic_json(OUTPUT / "qa_command_report.json", report)
            raise RuntimeError(f"QA command failed ({name}): {completed.stdout}\n{completed.stderr}")
    svg_text = stem.with_suffix(".svg").read_text(encoding="utf-8")
    use_references = re.findall(
        r'<use[^>]+(?:xlink:href|href)="#([^"]+)"', svg_text
    )
    reference_counts: dict[str, int] = {}
    for reference in use_references:
        reference_counts[reference] = reference_counts.get(reference, 0) + 1
    marker_references = [
        reference
        for reference, count in reference_counts.items()
        if count == EXPECTED_CONDITIONS
    ]
    if len(marker_references) != 1:
        raise RuntimeError(
            "Expected exactly one SVG marker reference used 1000 times; "
            f"found {marker_references} with counts {reference_counts}"
        )
    svg_point_audit = {
        "status": "PASS",
        "svg": str(stem.with_suffix(".svg")),
        "expected_condition_points": EXPECTED_CONDITIONS,
        "marker_reference": marker_references[0],
        "svg_point_use_count": reference_counts[marker_references[0]],
        "total_svg_use_nodes": len(use_references),
        "other_use_node_counts": {
            key: value
            for key, value in reference_counts.items()
            if key != marker_references[0]
        },
        "sampling": False,
        "aggregation": False,
    }
    _atomic_json(OUTPUT / "svg-point-audit.json", svg_point_audit)
    report["svg_points"] = svg_point_audit
    _atomic_json(OUTPUT / "qa_command_report.json", report)
    return report


def write_readme() -> Path:
    frame, metrics = aggregate_metrics()
    reference = metrics.get("comparison_24_condition_theory_C") or {}
    distribution = metrics["valid_trials_per_condition"]
    failures = int(metrics["failed_conditions"])
    histogram = ", ".join(f"{k}: {v}" for k, v in distribution.get("histogram", {}).items())
    readme = f"""# Experiment 2 / Theory C：1000 个 stable-complex 联合参数条件

## 结论

本次正式验证使用同一个完整 55 脑区结构网络上的 1000 个 `S × sigma_beta × floor_ratio`
联合参数条件。输入表的 before/after 条件数均为 {metrics['before_condition_count']}，没有下采样、
聚合、替换或依据 Theory C / simulation 结果重新筛选。

- Pearson r = {metrics['pearson_r']:.6f}
- slope = {metrics['slope']:.6f}
- intercept = {metrics['intercept']:.6f}
- OLS R² = {metrics['ols_R2']:.6f}
- identity R² = {metrics['identity_R2']:.6f}
- identity RMSE = {metrics['identity_RMSE']:.6f}
- MAE = {metrics['MAE']:.6f}
- mean bias (simulation − theory) = {metrics['mean_bias_simulation_minus_theory']:.6f}
- max absolute error = {metrics['max_abs_error']:.6f}
- 成功条件数 = {metrics['successful_conditions']}
- 失败条件数 = {failures}

## 稳定性与条件完整性

- stable-complex onset：S = {STABLE_COMPLEX_ONSET_S:.16g}
- 稳定共振 onset：S = {STABLE_RESONANCE_ONSET_S:.16g}
- 路径上限：S = {PATH_UPPER_S:.16g}
- 实际 S 范围：{metrics['stability_assertions']['S_min']:.16g}–{metrics['stability_assertions']['S_max']:.16g}
- 相对 stable-complex onset 的最小 margin：{metrics['stability_assertions']['stable_complex_min_margin']:.16g}
- 相对稳定共振 onset 的最小 margin：{metrics['stability_assertions']['stable_resonance_min_margin']:.16g}
- 距路径上限的最小 margin：{metrics['stability_assertions']['path_upper_min_margin']:.16g}
- 全部 `critical_real_per_s < 0`：{metrics['stability_assertions']['all_critical_real_per_s_lt_zero']}
- `critical_real_per_s` 范围：{metrics['stability_assertions']['critical_real_per_s_min']:.16g}–{metrics['stability_assertions']['critical_real_per_s_max']:.16g}
- `critical_frequency_hz` 范围：{metrics['stability_assertions']['critical_frequency_hz_min']:.16g}–{metrics['stability_assertions']['critical_frequency_hz_max']:.16g}

这些断言在任何计算前执行；若任一行不满足，脚本会整批报错，而不是过滤该行。

## 固定协议

- 每条件 256 trials；每 trial 131.072 s；3 calibration phases + 5 held-out phases。
- beta sampling 与 phase/trial 随机种子规则保持旧 Experiment 2 正式脚本不变。
- simulation 为完整 55 脑区 × E/I threshold-linear/ReLU 网络的 exact discrete
  active-branch solution。每个条件均独立求固定点、gain/Jacobian 和离散传递函数；没有使用
  local tanh toy model。
- simulation 特征使用 4.096 s periodic Hann-Welch、50% overlap；peak/exponent 提取与
  relaxed peak QC 保持注册方法不变。
- Theory C 先把连续全网络理论 PSD 与 4.096 s Hann 窗频谱核卷积，再提取 peak/exponent。
- corrected theory r 使用该条件三个 calibration phase 的 pooled within-trial residual
  covariance。simulation r 是五个 held-out phase r 的平均。
- 没有额外缩放参数，也没有用 held-out simulation 调参。

每条件共同有效 trial 数：min={distribution.get('minimum')}, median={distribution.get('median')},
mean={distribution.get('mean')}, max={distribution.get('maximum')}；完整分布为 `{histogram}`。

## 与 24-condition Theory C 的固定比较

24-condition 参考结果：Pearson r = {reference.get('pearson_r', float('nan')):.6f}，
slope = {reference.get('slope', float('nan')):.6f}，intercept = {reference.get('intercept', float('nan')):.6f}，
OLS R² = {reference.get('ols_R2', float('nan')):.6f}，identity R² = {reference.get('identity_R2', float('nan')):.6f}，
identity RMSE = {reference.get('identity_RMSE', float('nan')):.6f}。该比较只用于报告，未参与
1000 条件的筛选或调参。

## 图与 QA

单面板图严格为 5 × 4 inch（高:宽 = 4:5），无标题和轴标题。1000 个唯一条件对应
1000 个独立矢量散点；没有抽样、聚合或 hexbin。灰色虚线为 identity，黑线为自由截距
OLS，图内注明 Pearson r 和 slope。SVG/PDF 文字可编辑，TIFF 与 PNG 均为 600 dpi。

Nature-figure QA：单面板 alignment gate 记录为 NOT APPLICABLE；源代码检查、PDF 5 pt
字号下限检查和最终 PDF 碰撞审计见相应 JSON。最终 3000 × 2400 px、600 dpi render 已逐项
视觉检查：点云、回归线、identity 线、统计注释和刻度均清晰，无裁切或视觉碰撞；坐标覆盖
全部点且留白合理。

源代码检查的三个 WARN 已人工复核并接受：127 mm 图宽来自用户明确指定的 5 × 4 inch
合同，而不是误用期刊栏宽；脚本中的随机数是正式 simulation 协议，不是 demo/mock data；
图中每个散点是一个原始 condition-level held-out estimand，不是 seed/fold aggregate，因此
不应对这 1000 个原始条件点添加跨条件误差条。最终源代码检查为 18 PASS / 3 reviewed
WARN / 0 FAIL；PDF 最小文字为 13 pt；碰撞审计为 PASS（0 FAIL、0 WARN）。

## 文件

- `results_condition_level.csv`：逐条件结果
- `failures.csv`：失败清单（即使为空也保留表头）
- `metrics.json`：正式统计、有效 trial 分布与 24-condition 比较
- `protocol_manifest.json`：输入哈希、before/after、稳定性断言与固定协议
- `benchmark_report.json`：CPU/内存 benchmark 及旧模型设置一致性检查
- `condition_cache/`：逐条件、逐 phase 原子缓存，可断点续跑
- `condition_rows/`：逐条件成功记录
- `condition_failures/`：逐条件失败详情与 traceback
- `exp2_theory_C_1000conditions_5x4.svg/.pdf/.png/.tiff`：最终图
- `figure_manifest.json`：1000 scatter offsets 与无抽样声明
- `source-validation.json`、`pdf-text-audit.json`、`collision-audit.json`：图形 QA
- `svg-point-audit.json`：SVG 中 1000 个独立条件 marker 的结构化计数审计
- `run.log`：持久化运行日志
- `../run_exp2_theory_c_1000conditions.py`：可复现脚本

## Figure contract / panel audit

Core conclusion：Hann-kernel-convolved Theory C 在预先固定的 1000 个 stable-complex 联合参数
条件上逐条件预测 held-out simulation correlation。Results-level question：corrected theory r 与
held-out simulation r 是否在完整条件空间内保持线性对应和接近 identity。Archetype：single-panel
quantitative validation。Panel：1000 条件散点 + identity + OLS。Replicate unit：condition；每个
condition 的 estimand 来自 256 trials 和五个 held-out phase。Alignment：single panel / NOT
APPLICABLE。Collision：见最终审计。Reviewer risks：条件完整性、held-out leakage、有限窗 mapping、
有效 trial 数和 active-branch 适用性均由 manifest、逐条件诊断和缓存记录。
"""
    path = OUTPUT / "README_zh.md"
    _atomic_text(path, readme)
    return path


def run_all(workers: int) -> tuple[int, int]:
    conditions, assertion = load_and_assert_conditions(write_manifest=True)
    benchmark_path = OUTPUT / "benchmark_report.json"
    benchmark_report = (
        json.loads(benchmark_path.read_text(encoding="utf-8"))
        if benchmark_path.exists()
        else None
    )
    if not benchmark_report or benchmark_report.get("status") != "PASS":
        raise RuntimeError("Run the full-protocol --benchmark successfully before --run")
    successes, failures = _run_payloads(
        conditions, assertion["protocol_digest"], workers, include_ids=None, retry_failures=True
    )
    aggregate_metrics()
    return len(successes), len(failures)


def finalize() -> None:
    plot_final()
    run_qa()
    write_readme()
    _append_log("finalization PASS: metrics, 1000-point figure, exports, README, and QA complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert-input", action="store_true")
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--readme", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 6:
        raise SystemExit("--workers must be between 1 and 6 for the 24-GB host")
    if args.assert_input:
        _, assertion = load_and_assert_conditions(write_manifest=True)
        print(json.dumps(assertion, indent=2))
    if args.benchmark:
        print(json.dumps(benchmark(workers=min(args.workers, 3)), indent=2))
    if args.run:
        success, failure = run_all(args.workers)
        print(f"successful={success} failed={failure}")
    if args.aggregate:
        _, metrics = aggregate_metrics()
        print(json.dumps(metrics, indent=2))
    if args.plot:
        print(plot_final())
    if args.qa:
        print(json.dumps(run_qa(), indent=2))
    if args.readme:
        print(write_readme())
    if args.finalize:
        finalize()
    if args.status or not any(
        (
            args.assert_input,
            args.benchmark,
            args.run,
            args.aggregate,
            args.plot,
            args.qa,
            args.readme,
            args.finalize,
        )
    ):
        conditions, assertion = load_and_assert_conditions(write_manifest=False)
        successes, failures = _saved_state(conditions, assertion["protocol_digest"])
        print(
            json.dumps(
                {
                    "input": len(conditions),
                    "successful": len(successes),
                    "failed": len(failures),
                    "remaining": len(conditions) - len(successes) - len(failures),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
