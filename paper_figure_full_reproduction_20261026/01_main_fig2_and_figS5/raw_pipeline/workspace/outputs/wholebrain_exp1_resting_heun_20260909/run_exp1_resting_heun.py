from __future__ import annotations

"""Whole-brain Experiment 1 and zero-input resting-state reruns with Heun.

The registered model, conditions, seeds, window, Welch estimator, feature
definitions, and calibration/held-out split are preserved.  Only the numerical
recurrence is changed from Euler to explicit Heun.  The fast periodic transfer
path is permitted only after it agrees with the complete 110-state nonlinear
two-stage ReLU Heun trajectory on representative trials.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from scipy import fft, stats


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
SOURCE_EXACT = PACKAGE / "pap_validation/nonlinear_resimulations_exact.py"
SOURCE_DENSE = PACKAGE / "pap_validation/paper_figure_dense_revision.py"
SOURCE_ZERO = PROJECT / "tmp/wholebrain_exp1_exact_zero_input.py"
SOURCE_CONDITIONS = (
    PACKAGE / "outputs/paper_figure_dense_revision/exp1_dense_conditions.csv"
)
OLD_EXP1 = (
    PACKAGE
    / "outputs/nonlinear_resimulations_exact/wholebrain_C/exp1_nonlinear_full55_exact.csv"
)
OLD_RESTING = (
    PROJECT
    / "output/wholebrain_exp1_exact_zero_input_2026-08-29"
    / "experiment1_exact_zero_input_statistics.csv"
)
CONFIG = PACKAGE / "configs/full.yaml"

sys.path[:0] = [
    str(PACKAGE),
    str(WORKSPACE / "outputs"),
    str(FIGURE_SKILL / "scripts"),
]

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from heun_network_common import (  # noqa: E402
    heun_recurrence,
    threshold_network_heun_batch_fixedpoint,
)
from pap_validation.core import (  # noqa: E402
    feature_from_spectrum,
    finite_derivatives,
    input_psd,
    load_model,
    transfer_power,
)
from pap_validation.exp1 import Z0, ZHI, ZLO, ZSCALE, _feature_fun  # noqa: E402
from pap_validation.exp3_dt_convergence import _matched_bandlimited_inputs  # noqa: E402
from pap_validation.exp3_trial_seed_validation import _feature_arrays_vectorized  # noqa: E402
from pap_validation.io import load_config  # noqa: E402
from pap_validation.nonlinear_network_validation import (  # noqa: E402
    _fourier_inputs,
    _simulation_arrays,
)
from pap_validation.paper_figure_dense_revision import (  # noqa: E402
    EXP1_EPSILON,
    EXP1_TRIALS,
    SEED as DENSE_SEED,
    _sample_exact_correlation,
    _welch_features,
)
from pap_validation.realization_residual_validation import (  # noqa: E402
    _cov_to_corr,
    _exp1_designs,
    _pooled_within_trial_cov,
    _spectral_spec,
)


DT_MS = 0.25
BURN_S = 2.048
ANALYSIS_S = 32.768
SEGMENT_S = 4.096
SMOOTHING_SIGMA_HZ = 0.9765625
INPUT_SIGMA = 0.5
N_EXP1 = 1000
N_RESTING = 100
N_CAL = 3
N_HELD = 5
N_PHASES = N_CAL + N_HELD
RESTING_SEED = 2026082902
PROTOCOL_VERSION = "wholebrain_exp1_resting_heun_v1"
BLUE = "#247BA0"
ORANGE = "#F28E2B"
BLACK = "#222222"
_MODEL_BUNDLE_CACHE: dict[str, Any] | None = None
_SOURCE_CONDITION_CACHE: pd.DataFrame | None = None


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_log(message: str) -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    with (HERE / "run.log").open("a", encoding="utf-8") as handle:
        handle.write(f"{utc_now()} {message}\n")


def model_bundle() -> dict[str, Any]:
    global _MODEL_BUNDLE_CACHE
    if _MODEL_BUNDLE_CACHE is not None:
        return _MODEL_BUNDLE_CACHE
    cfg = load_config(CONFIG)
    model = load_model(PROJECT / cfg["model_dir"])
    arrays = _simulation_arrays(model.params)
    input_idx = model.areas.index("V1")
    output_idx = model.areas.index("PG")
    n_steps = int(round(ANALYSIS_S * 1000.0 / DT_MS))
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    total_steps = n_steps + burn_steps
    fft_length = fft.next_fast_len(2 * total_steps - 1, real=True)
    recurrence, forcing = heun_recurrence(
        model.J, input_idx, float(model.params["tau_exc"]), DT_MS
    )
    eigenvalues, vectors = np.linalg.eig(recurrence)
    coefficients = vectors[output_idx] * np.linalg.solve(vectors, forcing)
    transfer = np.empty(fft_length // 2 + 1, complex)
    block = 8192
    for start in range(0, len(transfer), block):
        stop = min(start + block, len(transfer))
        theta = 2.0 * np.pi * np.arange(start, stop) / fft_length
        transfer[start:stop] = np.sum(
            coefficients[None, :]
            / (
                1.0
                - np.exp(-1j * theta)[:, None] * eigenvalues[None, :]
            ),
            axis=1,
        )
    _MODEL_BUNDLE_CACHE = {
        "cfg": cfg,
        "model": model,
        "arrays": arrays,
        "input_idx": input_idx,
        "output_idx": output_idx,
        "transfer": transfer,
        "fft_length": fft_length,
    }
    return _MODEL_BUNDLE_CACHE


def source_conditions() -> pd.DataFrame:
    global _SOURCE_CONDITION_CACHE
    if _SOURCE_CONDITION_CACHE is None:
        _SOURCE_CONDITION_CACHE = pd.read_csv(SOURCE_CONDITIONS).set_index(
            "condition_id"
        )
    return _SOURCE_CONDITION_CACHE


def finite_record_paths(
    inputs: np.ndarray, transfer: np.ndarray, fft_length: int
) -> np.ndarray:
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    transformed = fft.rfft(inputs, n=fft_length, axis=-1, workers=1)
    paths = fft.irfft(
        transformed * transfer[None, :], n=fft_length, axis=-1, workers=1
    )
    return paths[:, burn_steps : inputs.shape[1]].astype(np.float32)


def path_features(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    specs = _spectral_spec(ANALYSIS_S)
    values, valid = _feature_arrays_vectorized(
        paths[:, None, :], DT_MS, specs
    )["long"]
    return values[0].astype(float), valid[0].astype(bool)


def corr(values: np.ndarray, valid: np.ndarray | None = None) -> float:
    values = np.asarray(values, float)
    use = np.isfinite(values).all(axis=1)
    if valid is not None:
        use &= np.asarray(valid, bool)
    if use.sum() < 8 or np.min(np.std(values[use], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[use, 0], values[use, 1]).statistic)


def comparison_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float | int]:
    use = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, float)[use]
    y = np.asarray(y, float)[use]
    fit = stats.linregress(x, y)
    delta = y - x
    return {
        "n": int(len(x)),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "pearson_r": float(fit.rvalue),
        "R2": float(fit.rvalue**2),
        "identity_R2": float(
            1.0 - np.sum(delta * delta) / np.sum((y - y.mean()) ** 2)
        ),
        "MAE": float(np.mean(np.abs(delta))),
        "RMSE": float(np.sqrt(np.mean(delta * delta))),
        "mean_bias": float(np.mean(delta)),
    }


def protocol() -> dict[str, Any]:
    return {
        "version": PROTOCOL_VERSION,
        "model": "complete 55-area x E/I, 110-state threshold-linear/ReLU network",
        "integrator": {
            "name": "explicit Heun",
            "stage_1": "all 110 ReLU drives evaluated at current state",
            "stage_2": "all 110 ReLU drives reevaluated at predictor",
            "forcing": "same u[n] in both stages",
            "dt_ms": DT_MS,
        },
        "routing": {"input_area": "V1", "output_area": "PG"},
        "analysis_s": ANALYSIS_S,
        "burn_s": BURN_S,
        "trials": EXP1_TRIALS,
        "welch": {
            "segment_s": SEGMENT_S,
            "window": "Hann",
            "overlap": "50%",
            "smoothing_sigma_hz": SMOOTHING_SIGMA_HZ,
            "peak_band_hz": [5.0, 20.0],
            "slope_band_hz": [100.0, 1000.0],
        },
        "experiment1": {
            "conditions": N_EXP1,
            "condition_seed_rule": "20260819 + condition_id*17",
            "forcing_seed_rule": "20260819 + 100000 + condition_id*31",
            "theory": "first-order continuous feature map plus Heun finite-window residual covariance",
        },
        "resting_zero_input": {
            "conditions": N_RESTING,
            "calibration_phases": N_CAL,
            "heldout_phases": N_HELD,
            "strict_zero": "sample Pearson input P-AP r <= 1e-12 after common-valid refinement",
            "seed": RESTING_SEED,
        },
        "accelerator": (
            "exact zero-initial finite-record SISO convolution of the full 110-state "
            "active-branch Heun recurrence, including the registered 2.048-s burn-in; "
            "blocked unless direct nonlinear Heun validation has zero ReLU crossings"
        ),
    }


def validate_inputs() -> dict[str, Any]:
    source = pd.read_csv(SOURCE_CONDITIONS)
    old_exp1 = pd.read_csv(OLD_EXP1)
    old_resting = pd.read_csv(OLD_RESTING)
    checks = {
        "source_conditions_1000": len(source) == N_EXP1,
        "source_condition_ids_unique": source.condition_id.nunique() == N_EXP1,
        "old_exp1_rows_1000": len(old_exp1) == N_EXP1,
        "old_resting_rows_100": len(old_resting) == N_RESTING,
        "old_resting_max_abs_input_r_le_1e12": bool(
            old_resting.input_pap_r.abs().max() <= 1e-12
        ),
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at_utc": utc_now(),
        "checks": checks,
        "protocol": protocol(),
        "fingerprints": {
            str(path): sha256(path)
            for path in (
                Path(__file__).resolve(),
                SOURCE_EXACT,
                SOURCE_DENSE,
                SOURCE_ZERO,
                SOURCE_CONDITIONS,
                OLD_EXP1,
                OLD_RESTING,
            )
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_json(HERE / "manifest.json", report)
    if report["status"] != "PASS":
        raise RuntimeError(f"Input validation failed: {checks}")
    return report


def pilot(force: bool = False) -> dict[str, Any]:
    target = HERE / "pilot/active_branch_validation.json"
    if target.exists() and not force:
        return json.loads(target.read_text())
    bundle = model_bundle()
    source = source_conditions()
    arrays = bundle["arrays"]
    model = bundle["model"]
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    validation_steps = int(round(4.096 * 1000.0 / DT_MS))
    rows: list[dict[str, Any]] = []
    for condition_id in (0, 499, 999):
        item = source.loc[condition_id]
        x, y = _sample_exact_correlation(
            EXP1_TRIALS,
            float(item.target_input_r),
            DENSE_SEED + condition_id * 17,
        )
        u = np.zeros((EXP1_TRIALS, 4))
        u[:, 0] = EXP1_EPSILON * x
        u[:, 2] = EXP1_EPSILON * y
        z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
        chosen = np.asarray([0, 31, 63], int)
        all_inputs = _fourier_inputs(
            z,
            DT_MS,
            int(round(ANALYSIS_S * 1000.0 / DT_MS)),
            burn_steps,
            INPUT_SIGMA,
            DENSE_SEED + 100000 + condition_id * 31,
        )
        inputs = all_inputs[chosen]
        direct_input = inputs[:, : burn_steps + validation_steps]
        direct, rect1, rect2 = threshold_network_heun_batch_fixedpoint(
            direct_input,
            *arrays,
            np.full(len(model.areas), 10.0),
            np.full(len(model.areas), 35.0),
            DT_MS,
            float(model.params["tau_exc"]),
            float(model.params["tau_inh"]),
            burn_steps,
            np.asarray([bundle["output_idx"]], np.int64),
            bundle["input_idx"],
        )
        accelerated = finite_record_paths(
            inputs, bundle["transfer"], bundle["fft_length"]
        )[:, :validation_steps]
        delta = direct[:, 0].astype(float) - accelerated.astype(float)
        for pos, trial in enumerate(chosen):
            rows.append(
                {
                    "condition_id": condition_id,
                    "trial": int(trial),
                    "max_abs_path_difference": float(np.max(np.abs(delta[pos]))),
                    "path_RMSE": float(np.sqrt(np.mean(delta[pos] ** 2))),
                    "stage1_relu_crossings": int(rect1[pos]),
                    "stage2_relu_crossings": int(rect2[pos]),
                }
            )
    frame = pd.DataFrame(rows)
    atomic_csv(HERE / "pilot/active_branch_validation.csv", frame)
    gate = {
        "status": "PASS"
        if (
            frame.stage1_relu_crossings.sum() == 0
            and frame.stage2_relu_crossings.sum() == 0
            and frame.max_abs_path_difference.max() <= 2e-5
        )
        else "FAIL",
        "paired_trials": int(len(frame)),
        "representative_conditions": int(frame.condition_id.nunique()),
        "max_abs_path_difference": float(frame.max_abs_path_difference.max()),
        "max_path_RMSE": float(frame.path_RMSE.max()),
        "stage1_relu_crossings": int(frame.stage1_relu_crossings.sum()),
        "stage2_relu_crossings": int(frame.stage2_relu_crossings.sum()),
        "validation_analysis_s": 4.096,
        "path_tolerance": 2e-5,
    }
    atomic_json(target, gate)
    if gate["status"] != "PASS":
        raise RuntimeError(f"Active-branch Heun validation failed: {gate}")
    return gate


def exp1_calibration(force: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = HERE / "exp1_heun_calibration.npz"
    if target.exists() and not force:
        with np.load(target) as data:
            return data["base"], data["gradient"], data["residual_covariance"]
    if pilot()["status"] != "PASS":
        raise RuntimeError("Passing direct nonlinear Heun pilot required")
    bundle = model_bundle()
    cfg = bundle["cfg"]
    _, all_u, all_z = _exp1_designs(cfg)
    theory_f = np.unique(
        np.r_[np.geomspace(0.5, 40.0, 180), np.geomspace(40.1, 1200.0, 220)]
    )
    power = transfer_power(
        bundle["model"].J,
        bundle["input_idx"],
        bundle["output_idx"],
        theory_f,
    )
    feature_map = _feature_fun(
        power, theory_f, tuple(cfg["peak_band_hz"]), tuple(cfg["slope_band_hz"])
    )
    gradient, _, _ = finite_derivatives(feature_map, np.zeros(4), h=2e-3)
    base = feature_map(np.zeros(4))
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    phases = []
    valids = []
    input_pos = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"].index("V1")
    for phase in range(N_CAL):
        phase_values = np.empty((len(all_z), EXP1_TRIALS, 2), float)
        phase_valid = np.zeros((len(all_z), EXP1_TRIALS), bool)
        for condition_pos, z in enumerate(all_z):
            inputs = _matched_bandlimited_inputs(
                z,
                DT_MS,
                ANALYSIS_S,
                BURN_S,
                INPUT_SIGMA,
                int(cfg["seed"])
                + 8100000
                + phase * 100000
                + input_pos * 10000
                + condition_pos,
                trial_offset=0,
            )
            if inputs.shape[1] != burn_steps + int(round(ANALYSIS_S * 1000 / DT_MS)):
                raise RuntimeError("Unexpected calibration forcing length")
            values, valid = path_features(
                finite_record_paths(
                    inputs, bundle["transfer"], bundle["fft_length"]
                )
            )
            phase_values[condition_pos] = values
            phase_valid[condition_pos] = valid
        phases.append(phase_values)
        valids.append(phase_valid)
        append_log(f"Experiment 1 Heun calibration phase {phase + 1}/{N_CAL}")
    values = np.asarray(phases)
    common = np.all(np.asarray(valids), axis=0)
    flattened = np.stack([values[p][common] for p in range(N_CAL)])
    residual_covariance = _pooled_within_trial_cov(flattened)
    atomic_npz(
        target,
        protocol=np.asarray(PROTOCOL_VERSION),
        base=base,
        gradient=gradient,
        residual_covariance=residual_covariance,
        common_valid_count=np.asarray(int(common.sum())),
        theory_u=all_u,
    )
    return base, gradient, residual_covariance


def exp1_cache(condition_id: int) -> Path:
    return HERE / "exp1_conditions" / f"condition_{condition_id:04d}.npz"


def exp1_worker(condition_id: int, force: bool = False) -> dict[str, Any]:
    path = exp1_cache(condition_id)
    if path.exists() and not force:
        with np.load(path) as data:
            return json.loads(str(data["row_json"].item()))
    source = source_conditions()
    item = source.loc[condition_id]
    bundle = model_bundle()
    base, gradient, residual_covariance = exp1_calibration(force=False)
    x, y = _sample_exact_correlation(
        EXP1_TRIALS,
        float(item.target_input_r),
        DENSE_SEED + condition_id * 17,
    )
    u = np.zeros((EXP1_TRIALS, 4))
    u[:, 0] = EXP1_EPSILON * x
    u[:, 2] = EXP1_EPSILON * y
    z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
    u = (z - Z0) / ZSCALE
    theory_values = base + u @ gradient.T
    signal_covariance = np.cov(theory_values, rowvar=False, ddof=1)
    theory_r = float(_cov_to_corr(signal_covariance + residual_covariance))
    inputs = _fourier_inputs(
        z,
        DT_MS,
        int(round(ANALYSIS_S * 1000.0 / DT_MS)),
        int(round(BURN_S * 1000.0 / DT_MS)),
        INPUT_SIGMA,
        DENSE_SEED + 100000 + condition_id * 31,
    )
    features, valid = path_features(
        finite_record_paths(inputs, bundle["transfer"], bundle["fft_length"])
    )
    use = valid & np.isfinite(theory_values).all(axis=1)
    row = {
        "condition_id": condition_id,
        "target_input_r": float(item.target_input_r),
        "theory_r_heun": theory_r,
        "simulation_r_heun": corr(features, use),
        "n_trials": EXP1_TRIALS,
        "n_valid": int(use.sum()),
        "analysis_s": ANALYSIS_S,
        "integrator": "explicit Heun active-branch recurrence validated against full nonlinear ReLU",
    }
    atomic_npz(
        path,
        protocol=np.asarray(PROTOCOL_VERSION),
        features=features.astype(np.float32),
        valid=valid,
        row_json=np.asarray(json.dumps(row)),
    )
    return row


def progress(scope: str, completed: int, total: int) -> None:
    atomic_json(
        HERE / f"progress_{scope}.json",
        {
            "scope": scope,
            "completed": completed,
            "total": total,
            "fraction": completed / total,
            "updated_at_utc": utc_now(),
        },
    )


def run_exp1(workers: int, force: bool = False) -> pd.DataFrame:
    validate_inputs()
    if pilot(force=False)["status"] != "PASS":
        raise RuntimeError("Pilot gate not passed")
    exp1_calibration(force=force)
    completed: dict[int, dict[str, Any]] = {}
    pending = []
    for condition_id in range(N_EXP1):
        path = exp1_cache(condition_id)
        if path.exists() and not force:
            with np.load(path) as data:
                completed[condition_id] = json.loads(str(data["row_json"].item()))
        else:
            pending.append(condition_id)
    progress("experiment1", len(completed), N_EXP1)
    append_log(f"Experiment 1 start pending={len(pending)} workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(exp1_worker, condition_id, force): condition_id
            for condition_id in pending
        }
        for future in as_completed(futures):
            condition_id = futures[future]
            try:
                completed[condition_id] = future.result()
            except Exception as exc:
                atomic_json(
                    HERE / "failures/experiment1" / f"condition_{condition_id:04d}.json",
                    {
                        "condition_id": condition_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                append_log(f"Experiment 1 condition {condition_id} FAIL {exc!r}")
            progress("experiment1", len(completed), N_EXP1)
    frame = pd.DataFrame(completed.values()).sort_values("condition_id")
    atomic_csv(HERE / "experiment1_results_heun.csv", frame)
    if len(frame) != N_EXP1:
        raise RuntimeError(f"Experiment 1 incomplete: {len(frame)}/{N_EXP1}")
    return frame


def standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (values - values.mean()) / values.std(ddof=1)


def input_features(z: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
    result = np.empty((len(z), 2), float)
    for trial, params in enumerate(z):
        feature = feature_from_spectrum(
            frequencies,
            input_psd(frequencies, params),
            (5.0, 20.0),
            (100.0, 1000.0),
        )
        if not feature["peak_valid"]:
            raise RuntimeError(f"Input trial {trial} has no valid peak")
        result[trial] = feature["peak_omega"], feature["exponent"]
    return result


def zero_input_from_scores(
    center_score: np.ndarray,
    exponent_score: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_score = standardize(center_score)
    exponent_score = standardize(exponent_score)
    for _ in range(20):
        u = np.zeros((len(center_score), 4), float)
        u[:, 0] = EXP1_EPSILON * center_score
        u[:, 2] = EXP1_EPSILON * exponent_score
        z = Z0 + u * ZSCALE
        if np.any((z < ZLO) | (z > ZHI)):
            raise RuntimeError("Unexpected clipping in strict-zero design")
        features = input_features(z, frequencies)
        peak_centered = features[:, 0] - features[:, 0].mean()
        score_centered = exponent_score - exponent_score.mean()
        exponent_score = standardize(
            score_centered
            - peak_centered
            * np.dot(peak_centered, score_centered)
            / np.dot(peak_centered, peak_centered)
        )
    u = np.zeros((len(center_score), 4), float)
    u[:, 0] = EXP1_EPSILON * center_score
    u[:, 2] = EXP1_EPSILON * exponent_score
    z = Z0 + u * ZSCALE
    features = input_features(z, frequencies)
    value = float(stats.pearsonr(features[:, 0], features[:, 1]).statistic)
    if abs(value) > 1e-12:
        raise RuntimeError(f"Input P-AP correlation is not zero: {value}")
    return z, u, features


def strict_zero_design(
    condition_id: int, frequencies: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(RESTING_SEED + condition_id * 1009)
    return zero_input_from_scores(
        rng.normal(size=EXP1_TRIALS),
        rng.normal(size=EXP1_TRIALS),
        frequencies,
    )


def resting_phase_features(
    z: np.ndarray,
    trial_ids: np.ndarray,
    transfer: np.ndarray,
    fft_length: int,
    condition_id: int,
    phase: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(ANALYSIS_S * 1000.0 / DT_MS))
    frequency = np.fft.rfftfreq(n_steps, d=DT_MS / 1000.0)
    active = (frequency < 2000.0) & (frequency > 0.0)
    f = np.maximum(frequency, 0.5)[None, :]
    center = z[:, 0, None]
    amplitude = z[:, 1, None]
    beta = z[:, 2, None]
    background_amplitude = z[:, 3, None]
    power = background_amplitude * (f / 10.0) ** beta
    power += amplitude * np.exp(-0.5 * ((f - center) / 2.0) ** 2)
    power[:, ~active] = 0.0
    spectra = np.empty(power.shape, np.complex128)
    phase_base = RESTING_SEED + 1_000_000 + condition_id * 10007 + phase * 101
    for row, trial_id in enumerate(trial_ids):
        rng = np.random.default_rng(phase_base + int(trial_id) * 1_000_003)
        angle = rng.uniform(0.0, 2.0 * np.pi, power.shape[1])
        spectra[row] = np.sqrt(np.maximum(power[row], 0.0)) * np.exp(1j * angle)
    spectra[:, ~active] = 0.0
    analysis_forcing = np.fft.irfft(spectra, n=n_steps, axis=-1)
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    forcing = np.concatenate(
        [analysis_forcing[:, -burn_steps:], analysis_forcing], axis=1
    )
    paths = finite_record_paths(forcing, transfer, fft_length)
    return _welch_features(paths)


def resting_cache(condition_id: int) -> Path:
    return HERE / "resting_conditions" / f"condition_{condition_id:03d}.npz"


def resting_theory_map() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = HERE / "resting_theory_map.npz"
    if target.exists():
        with np.load(target) as data:
            return data["frequencies"], data["base"], data["gradient"]
    bundle = model_bundle()
    frequencies = np.unique(
        np.r_[np.geomspace(0.5, 40.0, 180), np.geomspace(40.1, 1200.0, 220)]
    )
    power = transfer_power(
        bundle["model"].J,
        bundle["input_idx"],
        bundle["output_idx"],
        frequencies,
    )
    feature_map = _feature_fun(
        power, frequencies, (5.0, 20.0), (100.0, 1000.0)
    )
    gradient, _, _ = finite_derivatives(feature_map, np.zeros(4), h=2e-3)
    base = feature_map(np.zeros(4))
    atomic_npz(target, frequencies=frequencies, base=base, gradient=gradient)
    return frequencies, base, gradient


def resting_worker(condition_id: int, force: bool = False) -> dict[str, Any]:
    path = resting_cache(condition_id)
    if path.exists() and not force:
        with np.load(path) as data:
            return json.loads(str(data["row_json"].item()))
    frequencies, base, gradient = resting_theory_map()
    bundle = model_bundle()
    z, u, input_value = strict_zero_design(condition_id, frequencies)
    trial_ids = np.arange(len(z), dtype=int)
    simulated = None
    valid = None
    for _refinement in range(9):
        simulated_phases = []
        valid_phases = []
        for phase in range(N_PHASES):
            phase_value, phase_valid = resting_phase_features(
                z,
                trial_ids,
                bundle["transfer"],
                bundle["fft_length"],
                condition_id,
                phase,
            )
            simulated_phases.append(phase_value)
            valid_phases.append(phase_valid)
        simulated = np.asarray(simulated_phases, float)
        valid = np.asarray(valid_phases, bool)
        common = valid.all(axis=0) & np.isfinite(simulated).all(axis=(0, 2))
        if common.all():
            break
        if common.sum() < 48:
            raise RuntimeError(
                f"Condition {condition_id} retained only {common.sum()} trials"
            )
        z, u, input_value = zero_input_from_scores(
            u[common, 0] / EXP1_EPSILON,
            u[common, 2] / EXP1_EPSILON,
            frequencies,
        )
        trial_ids = trial_ids[common]
    else:
        raise RuntimeError(f"Condition {condition_id} failed common-valid refinement")
    assert simulated is not None and valid is not None
    common = valid.all(axis=0) & np.isfinite(simulated).all(axis=(0, 2))
    calibration = simulated[:N_CAL, common]
    residual_covariance = _pooled_within_trial_cov(calibration)
    first_order = base + u[common] @ gradient.T
    signal_covariance = np.cov(first_order, rowvar=False, ddof=1)
    theory_r = float(_cov_to_corr(signal_covariance + residual_covariance))
    heldout = simulated[N_CAL:, common]
    heldout_r = np.asarray([corr(values) for values in heldout], float)
    ensemble_r = corr(heldout.mean(axis=0))
    input_r = float(stats.pearsonr(input_value[:, 0], input_value[:, 1]).statistic)
    row = {
        "condition_id": condition_id,
        "input_pap_r": input_r,
        "n_trials_initial": EXP1_TRIALS,
        "n_trials_analyzed": int(len(z)),
        "n_valid_common": int(common.sum()),
        "theory_residual_corrected_r_heun": theory_r,
        "simulation_heldout_r_mean_heun": float(heldout_r.mean()),
        "simulation_heldout_r_sd_heun": float(heldout_r.std(ddof=1)),
        "simulation_phase_ensemble_r_heun": ensemble_r,
        "heldout_phase_r_json": json.dumps(heldout_r.tolist()),
        "residual_var_p": float(residual_covariance[0, 0]),
        "residual_var_ap": float(residual_covariance[1, 1]),
        "residual_cov_pap": float(residual_covariance[0, 1]),
    }
    atomic_npz(
        path,
        protocol=np.asarray(PROTOCOL_VERSION),
        z=z,
        u=u,
        trial_ids=trial_ids,
        input_features=input_value,
        simulated=simulated,
        valid=valid,
        row_json=np.asarray(json.dumps(row)),
    )
    return row


def run_resting(workers: int, force: bool = False) -> pd.DataFrame:
    validate_inputs()
    if pilot(force=False)["status"] != "PASS":
        raise RuntimeError("Pilot gate not passed")
    resting_theory_map()
    completed: dict[int, dict[str, Any]] = {}
    pending = []
    for condition_id in range(N_RESTING):
        path = resting_cache(condition_id)
        if path.exists() and not force:
            with np.load(path) as data:
                completed[condition_id] = json.loads(str(data["row_json"].item()))
        else:
            pending.append(condition_id)
    progress("resting", len(completed), N_RESTING)
    append_log(f"Resting zero-input start pending={len(pending)} workers={workers}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(resting_worker, condition_id, force): condition_id
            for condition_id in pending
        }
        for future in as_completed(futures):
            condition_id = futures[future]
            try:
                completed[condition_id] = future.result()
            except Exception as exc:
                atomic_json(
                    HERE / "failures/resting" / f"condition_{condition_id:03d}.json",
                    {
                        "condition_id": condition_id,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    },
                )
                append_log(f"Resting condition {condition_id} FAIL {exc!r}")
            progress("resting", len(completed), N_RESTING)
    frame = pd.DataFrame(completed.values()).sort_values("condition_id")
    atomic_csv(HERE / "resting_zero_input_results_heun.csv", frame)
    if len(frame) != N_RESTING:
        raise RuntimeError(f"Resting run incomplete: {len(frame)}/{N_RESTING}")
    if frame.input_pap_r.abs().max() > 1e-12:
        raise RuntimeError("Strict-zero input invariant failed")
    return frame


def set_scatter_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 14,
            "axes.labelsize": 17,
            "axes.titlesize": 18,
            "axes.linewidth": 2.4,
            "xtick.labelsize": 15,
            "ytick.labelsize": 15,
            "xtick.major.width": 2.4,
            "ytick.major.width": 2.4,
            "figure.facecolor": "white",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def polish(ax: plt.Axes, width: float) -> None:
    ax.tick_params(axis="both", which="major", width=width, length=7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(width)


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_exp1(frame: pd.DataFrame) -> tuple[Path, dict[str, Any]]:
    set_scatter_style()
    x = frame.theory_r_heun.to_numpy(float)
    y = frame.simulation_r_heun.to_numpy(float)
    metrics = comparison_metrics(x, y)
    fit = stats.linregress(x, y)
    low = float(min(x.min(), y.min()) - 0.04)
    high = float(max(x.max(), y.max()) + 0.04)
    grid = np.linspace(float(x.min()), float(x.max()), 300)
    fig, ax = plt.subplots(figsize=(5.25, 4.75), constrained_layout=True)
    ax.scatter(x, y, s=24, alpha=0.42, color=BLUE, edgecolor="none")
    ax.plot(grid, fit.intercept + fit.slope * grid, color=BLACK, lw=3.4)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="", ylabel="", title="")
    polish(ax, 2.4)
    stem = HERE / "wholebrain_experiment1_theory_vs_nonlinear_Heun_1000conditions"
    save_figure(fig, stem)
    return stem, metrics


def draw_resting(frame: pd.DataFrame) -> tuple[Path, dict[str, Any]]:
    set_scatter_style()
    mpl.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 19,
            "axes.titlesize": 19,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "axes.linewidth": 2.8,
        }
    )
    groups = [
        frame.theory_residual_corrected_r_heun.to_numpy(float),
        frame.simulation_heldout_r_mean_heun.to_numpy(float),
    ]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    boxes = ax.boxplot(
        groups,
        positions=[0, 1],
        widths=0.52,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": BLACK, "linewidth": 3.0},
        whiskerprops={"linewidth": 2.2},
        capprops={"linewidth": 2.2},
    )
    for patch, color in zip(boxes["boxes"], (BLUE, ORANGE)):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_linewidth(2.6)
    rng = np.random.default_rng(RESTING_SEED + 900000)
    for position, (values, color) in enumerate(zip(groups, (BLUE, ORANGE))):
        ax.scatter(
            position + rng.normal(0.0, 0.075, len(values)),
            values,
            s=18,
            color=color,
            alpha=0.32,
            edgecolor="none",
            zorder=2,
        )
    all_values = np.concatenate(groups)
    value_span = max(float(all_values.max() - all_values.min()), 1e-3)
    lower = float(all_values.min()) - 0.08 * value_span
    upper = max(
        float(all_values.max()) + 0.08 * value_span,
        0.06 * value_span,
    )
    ax.axhline(0.0, color="0.45", lw=2.0, zorder=0)
    ax.set_xticks([0, 1], ["Theory", "Simulation"])
    ax.set(
        ylabel="Output P–AP correlation, r",
        title="Experiment 1: resting state",
        ylim=(lower, upper),
        xlim=(-0.65, 1.65),
    )
    ax.yaxis.label.set_weight("bold")
    ax.title.set_weight("bold")
    polish(ax, 2.8)
    stem = HERE / "wholebrain_resting_zero_input_theory_vs_nonlinear_Heun_boxplot"
    save_figure(fig, stem)
    metrics = {
        "n_conditions": int(len(frame)),
        "max_abs_input_pap_r": float(frame.input_pap_r.abs().max()),
        "theory_median": float(np.median(groups[0])),
        "theory_mean": float(np.mean(groups[0])),
        "simulation_median": float(np.median(groups[1])),
        "simulation_mean": float(np.mean(groups[1])),
        "theory_range": [float(groups[0].min()), float(groups[0].max())],
        "simulation_range": [float(groups[1].min()), float(groups[1].max())],
    }
    return stem, metrics


def run_qa(stems: list[Path]) -> dict[str, Any]:
    validator = FIGURE_SKILL / "scripts/validate_figure.py"
    source_report = subprocess.run(
        [sys.executable, str(validator), str(Path(__file__).resolve()), "--json"],
        text=True,
        capture_output=True,
    )
    atomic_text(HERE / "source-validation.json", source_report.stdout)
    if source_report.returncode:
        raise RuntimeError(source_report.stdout + source_report.stderr)
    reports: dict[str, Any] = {"source_validation_returncode": source_report.returncode}
    for stem in stems:
        text_result = subprocess.run(
            [
                sys.executable,
                str(FIGURE_SKILL / "scripts/audit_pdf_text.py"),
                str(stem.with_suffix(".pdf")),
                "--min-pt",
                "5",
                "--json",
            ],
            text=True,
            capture_output=True,
        )
        atomic_text(stem.with_suffix(".pdf-text-audit.json"), text_result.stdout)
        collision_result = subprocess.run(
            [
                sys.executable,
                str(FIGURE_SKILL / "scripts/audit_figure_collisions.py"),
                str(stem.with_suffix(".pdf")),
                "--json-out",
                str(stem.with_suffix(".collision-audit.json")),
                "--overlay-pdf",
                str(stem.with_suffix(".collision-audit.pdf")),
            ],
            text=True,
            capture_output=True,
        )
        if text_result.returncode or collision_result.returncode:
            raise RuntimeError(
                f"QA failed for {stem.name}: {text_result.stdout} {collision_result.stdout}"
            )
        reports[stem.name] = {
            "pdf_text": "PASS",
            "collision": "PASS",
        }
    atomic_json(HERE / "qa_report.json", reports)
    return reports


def finalize() -> dict[str, Any]:
    exp1 = pd.read_csv(HERE / "experiment1_results_heun.csv")
    resting = pd.read_csv(HERE / "resting_zero_input_results_heun.csv")
    if len(exp1) != N_EXP1 or len(resting) != N_RESTING:
        raise RuntimeError("Both complete result tables are required")
    exp1_stem, exp1_metrics = draw_exp1(exp1)
    resting_stem, resting_metrics = draw_resting(resting)
    old_exp1 = pd.read_csv(OLD_EXP1)
    old_exp1_metrics = comparison_metrics(
        old_exp1.theory_r.to_numpy(float), old_exp1.simulation_r.to_numpy(float)
    )
    old_resting = pd.read_csv(OLD_RESTING)
    metrics = {
        "experiment1": {
            "old_Euler": old_exp1_metrics,
            "Heun": exp1_metrics,
        },
        "resting_zero_input": {
            "old_Euler_transfer": {
                "theory_median": float(old_resting.theory_residual_corrected_r.median()),
                "simulation_median": float(old_resting.simulation_heldout_r_mean.median()),
            },
            "Heun": resting_metrics,
        },
        "integrity": {
            "experiment1_rows": len(exp1),
            "resting_rows": len(resting),
            "experiment1_failures": len(list((HERE / "failures/experiment1").glob("*.json"))),
            "resting_failures": len(list((HERE / "failures/resting").glob("*.json"))),
            "pilot": pilot(),
        },
    }
    atomic_json(HERE / "metrics.json", metrics)
    qa = run_qa([exp1_stem, resting_stem])
    atomic_text(
        HERE / "README_zh.md",
        "# Whole-brain Experiment 1 与 resting-state Heun 复算\n\n"
        "两项复算均保留原模型、条件、种子、32.768 s 分析窗、0.25 ms 步长、"
        "4.096 s Hann-Welch 分段和特征定义。simulation 改为显式 Heun；"
        "finite-window residual covariance 也使用 Heun calibration 重新估计。\n\n"
        f"Experiment 1: n={exp1_metrics['n']}, slope={exp1_metrics['slope']:.6f}, "
        f"r={exp1_metrics['pearson_r']:.6f}, R²={exp1_metrics['R2']:.6f}.\n\n"
        f"Resting strict-zero input: n={resting_metrics['n_conditions']}, "
        f"max |input r|={resting_metrics['max_abs_input_pap_r']:.3e}, "
        f"theory median={resting_metrics['theory_median']:.6f}, "
        f"simulation median={resting_metrics['simulation_median']:.6f}.\n"
    )
    atomic_json(
        HERE / "complete.json",
        {
            "status": "PASS",
            "completed_at_utc": utc_now(),
            "metrics": metrics,
            "qa": qa,
            "figures": [str(exp1_stem), str(resting_stem)],
        },
    )
    return metrics


def status() -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name in ("experiment1", "resting"):
        path = HERE / f"progress_{name}.json"
        result[name] = json.loads(path.read_text()) if path.exists() else {"status": "not_started"}
    result["pilot"] = (
        json.loads((HERE / "pilot/active_branch_validation.json").read_text())
        if (HERE / "pilot/active_branch_validation.json").exists()
        else {"status": "not_started"}
    )
    result["complete"] = (HERE / "complete.json").exists()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assert-input", action="store_true")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--exp1", action="store_true")
    parser.add_argument("--resting", action="store_true")
    parser.add_argument("--finalize", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        raise SystemExit("--workers must be between 1 and 8")
    if args.assert_input:
        print(json.dumps(validate_inputs(), indent=2, default=json_ready))
    if args.pilot:
        print(json.dumps(pilot(force=args.force), indent=2, default=json_ready))
    if args.exp1:
        run_exp1(args.workers, force=args.force)
    if args.resting:
        run_resting(args.workers, force=args.force)
    if args.finalize:
        print(json.dumps(finalize(), indent=2, default=json_ready))
    if args.status or not any(
        (args.assert_input, args.pilot, args.exp1, args.resting, args.finalize)
    ):
        print(json.dumps(status(), indent=2, default=json_ready))


if __name__ == "__main__":
    main()
