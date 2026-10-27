from __future__ import annotations

"""Figure 2i2 on a protocol-locked, unscreened Experiment-1 condition ensemble.

The protocol is frozen before simulation.  The held-out ensemble is an equally
spaced subset of the registered 1000 Experiment-1 condition IDs; no fit result
is used to select, remove, or weight a condition.  Every intact/deleted network
uses complete nonlinear 110-state explicit Heun simulation.  Theory is the
previously requested second-order input-parameter Taylor map, expanded at each
network's own fixed point/Jacobian.  Calibration and held-out statistics follow
the Experiment-1 implementation.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

for _name, _value in (
    ("OPENBLAS_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OMP_NUM_THREADS", "1"),
    ("VECLIB_MAXIMUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
    ("NUMBA_NUM_THREADS", "3"),
):
    os.environ.setdefault(_name, _value)

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
HISTORY = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "workspace"
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
I2_RUNNER = HISTORY / "outputs/wholebrain_H_heun_20260908/run_H_heun.py"
EXP1_RUNNER = (
    HISTORY
    / "outputs/wholebrain_exp1_resting_heun_20260909/"
    "run_exp1_resting_heun.py"
)
EXP1_CONDITIONS = (
    PACKAGE / "outputs/paper_figure_dense_revision/exp1_dense_conditions.csv"
)
SECOND_ORDER_MAPS = (
    ROOT
    / "outputs/i2_second_order_exp1_matched_exact_heun_20260915/"
    "i2_exp1_second_order_maps.npz"
)
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [
    str(PACKAGE),
    str(HISTORY / "outputs"),
    str(FIGURE_SKILL / "scripts"),
]

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from pap_validation.exp1 import Z0, ZHI, ZLO, ZSCALE  # noqa: E402


PROTOCOL = "i2_predefined_unscreened_exp1_ensemble_second_order_v1"
HELDOUT_IDS = np.rint(np.linspace(0, 999, 12)).astype(int)
N_TRIALS = 64
N_CALIBRATION_PHASES = 3
N_CALIBRATION_CONDITIONS = 14
MIN_VALID = 8
BLUE = "#247BA0"
ORANGE = "#E76F51"
GREEN = "#2A9D8F"
PURPLE = "#8E5EA2"
BLACK = "#202020"
STYLES = {
    1: (BLUE, "o", "1 node"),
    3: ("#F4A261", "^", "3 nodes"),
    4: (GREEN, "s", "4 nodes"),
    5: (PURPLE, "D", "5 nodes"),
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def corr(values: np.ndarray, valid: np.ndarray) -> float:
    values = np.asarray(values, float)
    good = np.asarray(valid, bool) & np.isfinite(values).all(axis=1)
    if good.sum() < MIN_VALID or np.min(np.std(values[good], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[good, 0], values[good, 1]).statistic)


def cov_to_corr(covariance: np.ndarray) -> float:
    covariance = np.asarray(covariance, float)
    if (
        covariance.shape != (2, 2)
        or not np.isfinite(covariance).all()
        or covariance[0, 0] <= 0
        or covariance[1, 1] <= 0
    ):
        return np.nan
    return float(covariance[0, 1] / np.sqrt(covariance[0, 0] * covariance[1, 1]))


def taylor2(
    base: np.ndarray,
    gradient: np.ndarray,
    hessian: np.ndarray,
    u: np.ndarray,
) -> np.ndarray:
    return (
        base[None, :]
        + u @ gradient.T
        + 0.5 * np.einsum("ni,kij,nj->nk", u, hessian, u)
    )


def fit_metrics(theory: np.ndarray, simulation: np.ndarray) -> dict[str, Any]:
    use = np.isfinite(theory) & np.isfinite(simulation)
    x = np.asarray(theory, float)[use]
    y = np.asarray(simulation, float)[use]
    if len(x) < 3 or np.std(x) <= 0 or np.std(y) <= 0:
        return {"n": int(len(x))}
    fit = stats.linregress(x, y)
    error = y - x
    denom = np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(x)),
        "pearson_r": float(fit.rvalue),
        "ols_slope": float(fit.slope),
        "ols_intercept": float(fit.intercept),
        "ols_R2": float(fit.rvalue**2),
        "identity_R2": float(1 - np.sum(error**2) / denom),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "mean_bias": float(np.mean(error)),
    }


def freeze_protocol() -> dict[str, Any]:
    """Freeze all condition IDs, draws, seeds, estimands, and exclusions first."""
    protocol_path = HERE / "protocol_locked.json"
    design_path = HERE / "protocol_locked_design.npz"
    if protocol_path.exists() and design_path.exists():
        locked = json.loads(protocol_path.read_text(encoding="utf-8"))
        if locked["protocol"] != PROTOCOL:
            raise RuntimeError("Existing locked protocol has a different version")
        if locked["design_sha256"] != sha256(design_path):
            raise RuntimeError("Locked design fingerprint changed")
        return locked

    exp1 = load_module("exp1_freeze", EXP1_RUNNER)
    cfg = exp1.model_bundle()["cfg"]
    cal_meta, cal_u, cal_z = exp1._exp1_designs(cfg)
    conditions = pd.read_csv(EXP1_CONDITIONS).set_index("condition_id")
    held_u = np.empty((len(HELDOUT_IDS), N_TRIALS, 4), float)
    held_z = np.empty_like(held_u)
    held_targets = np.empty(len(HELDOUT_IDS), float)
    held_clipped = np.empty(len(HELDOUT_IDS), int)
    for pos, condition_id in enumerate(HELDOUT_IDS):
        target = float(conditions.loc[int(condition_id), "target_input_r"])
        x, y = exp1._sample_exact_correlation(
            N_TRIALS, target, exp1.DENSE_SEED + int(condition_id) * 17
        )
        u = np.zeros((N_TRIALS, 4), float)
        u[:, 0] = exp1.EXP1_EPSILON * x
        u[:, 2] = exp1.EXP1_EPSILON * y
        raw = Z0 + u * ZSCALE
        z = np.clip(raw, ZLO, ZHI)
        held_u[pos] = (z - Z0) / ZSCALE
        held_z[pos] = z
        held_targets[pos] = target
        held_clipped[pos] = int(np.any(raw != z, axis=1).sum())

    if cal_u.shape != (N_CALIBRATION_CONDITIONS, N_TRIALS, 4):
        raise RuntimeError(f"Experiment-1 calibration design changed: {cal_u.shape}")
    atomic_npz(
        design_path,
        calibration_u=cal_u,
        calibration_z=cal_z,
        heldout_condition_ids=HELDOUT_IDS,
        heldout_target_input_r=held_targets,
        heldout_u=held_u,
        heldout_z=held_z,
        heldout_clipped_trials=held_clipped,
    )
    locked = {
        "protocol": PROTOCOL,
        "locked_at_utc_before_simulation": utc_now(),
        "selection_rule": (
            "round(linspace(0,999,12)); fixed before simulation; no theory, "
            "simulation, fit, deletion-effect, or residual statistic enters selection"
        ),
        "heldout_condition_ids": HELDOUT_IDS,
        "heldout_target_input_r": held_targets,
        "heldout_trials_per_condition": N_TRIALS,
        "heldout_clipped_trials": held_clipped,
        "calibration": {
            "conditions": cal_meta,
            "n_conditions": N_CALIBRATION_CONDITIONS,
            "phases": N_CALIBRATION_PHASES,
            "trials_per_condition": N_TRIALS,
            "forcing": "Experiment-1 matched-bandlimited forcing",
            "seed_rule": (
                "cfg.seed + 8100000 + phase*100000 + "
                "input_position*10000 + calibration_condition_position"
            ),
            "residual_estimator": (
                "Experiment-1 unbiased pooled within-trial covariance after "
                "centering the three realization phases within each "
                "calibration-condition x trial unit"
            ),
        },
        "heldout": {
            "forcing": "Experiment-1 Fourier forcing",
            "condition_draw_seed_rule": "20260819 + condition_id*17",
            "forcing_seed_rule": "20260819 + 100000 + condition_id*31",
            "statistic": "Pearson r of per-trial peak omega and exponent",
        },
        "model": "complete 55-area x E/I, 110-state threshold-linear/ReLU network",
        "integrator": "complete nonlinear explicit Heun, both ReLU stages evaluated",
        "routing": {"input_area": "A6Va", "output_area": "V2"},
        "theory": {
            "formula": "m(u)=m0+G u+0.5 u^T H u",
            "coordinates": "four standardized input parameters u",
            "center": "u=0",
            "derivative_step": 0.002,
            "network_specific": (
                "intact and every deleted network use their own fixed point, "
                "active set, Jacobian, gradient, and Hessian"
            ),
        },
        "delta_definition": "r_intact - r_deleted, separately estimated first",
        "i2_estimand": (
            "for each deletion pattern, equal-weight arithmetic mean of its 12 "
            "condition-specific theory delta-r values and separately of its 12 "
            "condition-specific simulation delta-r values"
        ),
        "exclusion_rule": (
            "exclude only a network-condition statistic with fewer than 8 "
            "numerically valid trials or non-finite/zero-variance features; "
            "never exclude using correspondence or effect size"
        ),
        "fit_unit": (
            "82 deletion-pattern means; nested 984 network-condition rows are "
            "not treated as independent OLS observations"
        ),
        "analysis": {
            "dt_ms": 0.25,
            "burn_s": 2.048,
            "analysis_s": 32.768,
            "welch_segment_s": 4.096,
            "welch_window": "Hann",
            "welch_overlap": "50%",
            "smoothing_sigma_hz": 0.9765625,
            "peak_band_hz": [5.0, 20.0],
            "slope_band_hz": [100.0, 1000.0],
        },
        "input_fingerprints": {
            "experiment1_runner": sha256(EXP1_RUNNER),
            "i2_network_runner": sha256(I2_RUNNER),
            "experiment1_condition_table": sha256(EXP1_CONDITIONS),
            "second_order_maps": sha256(SECOND_ORDER_MAPS),
        },
        "design_sha256": sha256(design_path),
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_json(protocol_path, locked)
    return locked


_WORKER_I2 = None
_WORKER_EXP1 = None
_WORKER_MODEL = None
_WORKER_PATTERNS = None
_WORKER_MAPS = None
_WORKER_DESIGN = None
_WORKER_CFG = None


def worker_init() -> None:
    global _WORKER_I2, _WORKER_EXP1, _WORKER_MODEL, _WORKER_PATTERNS
    global _WORKER_MAPS, _WORKER_DESIGN, _WORKER_CFG
    freeze_protocol()
    _WORKER_I2 = load_module("i2_predefined_worker_i2", I2_RUNNER)
    _WORKER_EXP1 = load_module("i2_predefined_worker_exp1", EXP1_RUNNER)
    _WORKER_MODEL, _WORKER_PATTERNS, _unused_z, _WORKER_MAPS, _unused_design = (
        _WORKER_I2.build_context()
    )
    _WORKER_CFG = _WORKER_EXP1.model_bundle()["cfg"]
    with np.load(HERE / "protocol_locked_design.npz") as saved:
        _WORKER_DESIGN = {key: saved[key].copy() for key in saved.files}
    if len(_WORKER_PATTERNS) != 82 or len(_WORKER_MAPS["flns"]) != 83:
        raise RuntimeError("Registered deletion patterns changed")


def network_cache(network: int) -> Path:
    return HERE / "exact_heun_features" / f"network_{network:03d}.npz"


def cache_metadata(network: int) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "network_position": int(network),
        "integrator": "complete nonlinear 110-state explicit Heun",
        "calibration_shape": [3, 14, 64, 2],
        "heldout_condition_ids": HELDOUT_IDS.tolist(),
        "heldout_shape": [12, 64, 2],
        "input_area": "A6Va",
        "output_area": "V2",
        "analysis_s": 32.768,
        "burn_s": 2.048,
        "dt_ms": 0.25,
        "design_sha256": sha256(HERE / "protocol_locked_design.npz"),
    }


def save_network_cache(
    path: Path,
    metadata: dict[str, Any],
    cal_features: np.ndarray,
    cal_valid: np.ndarray,
    cal_rect: np.ndarray,
    cal_done: np.ndarray,
    held_features: np.ndarray,
    held_valid: np.ndarray,
    held_rect: np.ndarray,
    held_done: np.ndarray,
) -> None:
    atomic_npz(
        path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
        calibration_features=cal_features,
        calibration_valid=cal_valid,
        calibration_rectification=cal_rect,
        calibration_completed=cal_done,
        heldout_features=held_features,
        heldout_valid=held_valid,
        heldout_rectification=held_rect,
        heldout_completed=held_done,
    )


def simulate_network(network: int, force: bool = False) -> dict[str, Any]:
    if _WORKER_I2 is None:
        worker_init()
    i2 = _WORKER_I2
    exp1 = _WORKER_EXP1
    model = _WORKER_MODEL
    maps = _WORKER_MAPS
    design = _WORKER_DESIGN
    cache = network_cache(network)
    metadata = cache_metadata(network)

    cal_features = np.full((3, 14, 64, 2), np.nan, np.float32)
    cal_valid = np.zeros((3, 14, 64), bool)
    cal_rect = np.zeros((3, 14), np.int64)
    cal_done = np.zeros((3, 14), bool)
    held_features = np.full((12, 64, 2), np.nan, np.float32)
    held_valid = np.zeros((12, 64), bool)
    held_rect = np.zeros(12, np.int64)
    held_done = np.zeros(12, bool)
    if cache.exists() and not force:
        with np.load(cache) as saved:
            prior = json.loads(str(saved["metadata"].item()))
            if prior != metadata:
                raise RuntimeError(f"Cache metadata mismatch for network {network}")
            cal_features = saved["calibration_features"].copy()
            cal_valid = saved["calibration_valid"].copy()
            cal_rect = saved["calibration_rectification"].copy()
            cal_done = saved["calibration_completed"].copy()
            held_features = saved["heldout_features"].copy()
            held_valid = saved["heldout_valid"].copy()
            held_rect = saved["heldout_rectification"].copy()
            held_done = saved["heldout_completed"].copy()
            if cal_done.all() and held_done.all():
                return {"network": network, "status": "cached"}

    input_idx = model.areas.index(i2.source.INPUT_AREA)
    output_idx = model.areas.index(i2.source.OUTPUT_AREA)
    input_pos = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"].index(
        i2.source.INPUT_AREA
    )
    fixedpoint = maps["fixedpoints"][network]
    fln = maps["flns"][network]
    cal_z = design["calibration_z"]
    # Pure computational batching: every condition keeps its own registered
    # seed/input array; concatenation only amortizes the vectorized solver call.
    batch_size = 1
    for phase in range(3):
        remaining = [pos for pos in range(14) if not cal_done[phase, pos]]
        for batch_start in range(0, len(remaining), batch_size):
            positions = remaining[batch_start : batch_start + batch_size]
            input_blocks = [
                exp1._matched_bandlimited_inputs(
                    cal_z[condition_pos],
                    i2.source.DT_MS,
                    i2.ANALYSIS_S,
                    i2.source.BURN_S,
                    i2.source.INPUT_SIGMA,
                    int(_WORKER_CFG["seed"])
                    + 8100000
                    + phase * 100000
                    + input_pos * 10000
                    + condition_pos,
                    trial_offset=0,
                )
                for condition_pos in positions
            ]
            inputs = np.concatenate(input_blocks, axis=0)
            paths, counts = i2.network_run_fixedpoint_heun(
                model,
                inputs,
                np.asarray([output_idx], np.int64),
                input_idx,
                fixedpoint,
                fln,
            )
            values, valid = i2.source._feature_from_paths(paths, i2.ANALYSIS_S)
            for local_pos, condition_pos in enumerate(positions):
                take = slice(local_pos * N_TRIALS, (local_pos + 1) * N_TRIALS)
                cal_features[phase, condition_pos] = values[take]
                cal_valid[phase, condition_pos] = valid[take]
                cal_rect[phase, condition_pos] = int(counts[take].sum())
                cal_done[phase, condition_pos] = True
            save_network_cache(
                cache, metadata, cal_features, cal_valid, cal_rect, cal_done,
                held_features, held_valid, held_rect, held_done,
            )

    held_z = design["heldout_z"]
    burn_steps = int(round(i2.source.BURN_S * 1000.0 / i2.source.DT_MS))
    n_steps = int(round(i2.ANALYSIS_S * 1000.0 / i2.source.DT_MS))
    remaining = [pos for pos in range(len(HELDOUT_IDS)) if not held_done[pos]]
    for batch_start in range(0, len(remaining), batch_size):
        positions = remaining[batch_start : batch_start + batch_size]
        input_blocks = [
            exp1._fourier_inputs(
                held_z[condition_pos],
                i2.source.DT_MS,
                n_steps,
                burn_steps,
                i2.source.INPUT_SIGMA,
                exp1.DENSE_SEED
                + 100000
                + int(HELDOUT_IDS[condition_pos]) * 31,
            )
            for condition_pos in positions
        ]
        inputs = np.concatenate(input_blocks, axis=0)
        paths, counts = i2.network_run_fixedpoint_heun(
            model,
            inputs,
            np.asarray([output_idx], np.int64),
            input_idx,
            fixedpoint,
            fln,
        )
        values, valid = i2.source._feature_from_paths(paths, i2.ANALYSIS_S)
        for local_pos, condition_pos in enumerate(positions):
            take = slice(local_pos * N_TRIALS, (local_pos + 1) * N_TRIALS)
            held_features[condition_pos] = values[take]
            held_valid[condition_pos] = valid[take]
            held_rect[condition_pos] = int(counts[take].sum())
            held_done[condition_pos] = True
        save_network_cache(
            cache, metadata, cal_features, cal_valid, cal_rect, cal_done,
            held_features, held_valid, held_rect, held_done,
        )
    return {"network": network, "status": "complete"}


def run_simulation(workers: int, start: int, end: int, force: bool) -> None:
    freeze_protocol()
    indices = list(range(max(0, start), min(83, end)))
    completed = 0
    failures: list[dict[str, Any]] = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = {pool.submit(simulate_network, n, force): n for n in indices}
        for future in as_completed(futures):
            network = futures[future]
            try:
                outcome = future.result()
                completed += 1
                print(
                    f"network {network:03d}: {outcome['status']} "
                    f"({completed}/{len(indices)})",
                    flush=True,
                )
            except Exception as exc:
                failures.append(
                    {
                        "network": network,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
            atomic_json(
                HERE / "simulation_progress.json",
                {
                    "protocol": PROTOCOL,
                    "requested_start": start,
                    "requested_end": end,
                    "completed_networks_this_run": completed,
                    "total_networks_this_run": len(indices),
                    "elapsed_s": time.time() - started,
                    "failures": failures,
                },
            )
    if failures:
        raise RuntimeError(f"Simulation failures: {failures}")


def load_network_features(network: int) -> dict[str, np.ndarray]:
    path = network_cache(network)
    if not path.exists():
        raise RuntimeError(f"Missing simulation cache for network {network}")
    with np.load(path) as saved:
        metadata = json.loads(str(saved["metadata"].item()))
        if metadata != cache_metadata(network):
            raise RuntimeError(f"Cache metadata mismatch for network {network}")
        if not saved["calibration_completed"].all() or not saved["heldout_completed"].all():
            raise RuntimeError(f"Incomplete simulation cache for network {network}")
        return {key: saved[key].copy() for key in saved.files if key != "metadata"}


def residual_covariance(features: np.ndarray, valid: np.ndarray, exp1) -> tuple[np.ndarray, int]:
    finite = np.isfinite(features).all(axis=-1)
    common = np.asarray(valid, bool).all(axis=0) & finite.all(axis=0)
    flattened = np.stack([features[phase][common] for phase in range(3)])
    return exp1._pooled_within_trial_cov(flattened), int(common.sum())


def assemble() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    locked = freeze_protocol()
    exp1 = load_module("i2_predefined_assemble_exp1", EXP1_RUNNER)
    i2 = load_module("i2_predefined_assemble_i2", I2_RUNNER)
    model, patterns, _unused_z, maps, _unused_design = i2.build_context()
    with np.load(HERE / "protocol_locked_design.npz") as saved:
        design = {key: saved[key].copy() for key in saved.files}
    with np.load(SECOND_ORDER_MAPS) as saved:
        bases = saved["base"].copy()
        gradients = saved["gradient_u"].copy()
        hessians = saved["hessian_u"].copy()
        if not np.array_equal(saved["theta0"], Z0) or not np.array_equal(saved["theta_scale"], ZSCALE):
            raise RuntimeError("Second-order map coordinates changed")

    network_data = [load_network_features(network) for network in range(83)]
    residuals = np.empty((83, 2, 2), float)
    residual_units = np.empty(83, int)
    for network, item in enumerate(network_data):
        residuals[network], residual_units[network] = residual_covariance(
            item["calibration_features"], item["calibration_valid"], exp1
        )

    condition_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    held_u = design["heldout_u"]
    targets = design["heldout_target_input_r"]
    intact_theory_values = [
        taylor2(bases[0], gradients[0], hessians[0], u) for u in held_u
    ]
    intact_r: dict[int, tuple[float, float, int]] = {}
    for cpos, condition_id in enumerate(HELDOUT_IDS):
        item = network_data[0]
        valid = item["heldout_valid"][cpos] & np.isfinite(item["heldout_features"][cpos]).all(axis=1)
        theory_values = intact_theory_values[cpos]
        theory_r = cov_to_corr(np.cov(theory_values, rowvar=False, ddof=1) + residuals[0])
        simulation_r = corr(item["heldout_features"][cpos], valid)
        intact_r[int(condition_id)] = (theory_r, simulation_r, int(valid.sum()))

    for position, pattern in enumerate(patterns.itertuples(index=False), 1):
        item = network_data[position]
        for cpos, condition_id in enumerate(HELDOUT_IDS):
            valid = item["heldout_valid"][cpos] & np.isfinite(item["heldout_features"][cpos]).all(axis=1)
            theory_values = taylor2(
                bases[position], gradients[position], hessians[position], held_u[cpos]
            )
            theory_deleted = cov_to_corr(
                np.cov(theory_values, rowvar=False, ddof=1) + residuals[position]
            )
            simulation_deleted = corr(item["heldout_features"][cpos], valid)
            theory_intact, simulation_intact, intact_n = intact_r[int(condition_id)]
            if not np.isfinite([theory_intact, simulation_intact, theory_deleted, simulation_deleted]).all():
                exclusion_rows.append(
                    {
                        "network_position": position,
                        "deleted_nodes": pattern.deleted_nodes,
                        "condition_id": int(condition_id),
                        "intact_n_valid": intact_n,
                        "deleted_n_valid": int(valid.sum()),
                        "reason": "pre-specified numerical-validity rule",
                    }
                )
                continue
            row = pattern._asdict()
            row.update(
                {
                    "network_position": position,
                    "condition_id": int(condition_id),
                    "target_input_r": float(targets[cpos]),
                    "theory_intact_r": theory_intact,
                    "simulation_intact_r": simulation_intact,
                    "theory_deleted_r": theory_deleted,
                    "simulation_deleted_r": simulation_deleted,
                    "theory_delta_r": theory_intact - theory_deleted,
                    "simulation_delta_r": simulation_intact - simulation_deleted,
                    "intact_n_valid": intact_n,
                    "deleted_n_valid": int(valid.sum()),
                    "intact_residual_units": int(residual_units[0]),
                    "deleted_residual_units": int(residual_units[position]),
                    "intact_rectification_calibration": int(network_data[0]["calibration_rectification"].sum()),
                    "deleted_rectification_calibration": int(item["calibration_rectification"].sum()),
                    "intact_rectification_heldout": int(network_data[0]["heldout_rectification"][cpos]),
                    "deleted_rectification_heldout": int(item["heldout_rectification"][cpos]),
                }
            )
            condition_rows.append(row)

    condition_frame = pd.DataFrame(condition_rows)
    if condition_frame.empty:
        raise RuntimeError("No valid network-condition rows")
    aggregate_rows: list[dict[str, Any]] = []
    for position, local in condition_frame.groupby("network_position", sort=True):
        first = local.iloc[0]
        aggregate_rows.append(
            {
                "network_position": int(position),
                "n_deleted": int(first.n_deleted),
                "deleted_nodes": first.deleted_nodes,
                "n_conditions": int(len(local)),
                "theory_intact_r_mean": float(local.theory_intact_r.mean()),
                "simulation_intact_r_mean": float(local.simulation_intact_r.mean()),
                "theory_deleted_r_mean": float(local.theory_deleted_r.mean()),
                "simulation_deleted_r_mean": float(local.simulation_deleted_r.mean()),
                "theory_delta_r_mean": float(local.theory_delta_r.mean()),
                "simulation_delta_r_mean": float(local.simulation_delta_r.mean()),
                "theory_deleted_r_sd_across_conditions": float(local.theory_deleted_r.std(ddof=1)),
                "simulation_deleted_r_sd_across_conditions": float(local.simulation_deleted_r.std(ddof=1)),
                "theory_delta_r_sd_across_conditions": float(local.theory_delta_r.std(ddof=1)),
                "simulation_delta_r_sd_across_conditions": float(local.simulation_delta_r.std(ddof=1)),
            }
        )
    aggregate_frame = pd.DataFrame(aggregate_rows)
    intact_frame = (
        condition_frame[["condition_id", "target_input_r", "theory_intact_r", "simulation_intact_r", "intact_n_valid"]]
        .drop_duplicates("condition_id")
        .sort_values("condition_id")
    )

    metrics = {
        "protocol": locked,
        "intact_across_predefined_conditions": fit_metrics(
            intact_frame.theory_intact_r.to_numpy(), intact_frame.simulation_intact_r.to_numpy()
        ),
        "deleted_network_means_across_conditions": fit_metrics(
            aggregate_frame.theory_deleted_r_mean.to_numpy(),
            aggregate_frame.simulation_deleted_r_mean.to_numpy(),
        ),
        "delta_network_means_across_conditions": fit_metrics(
            aggregate_frame.theory_delta_r_mean.to_numpy(),
            aggregate_frame.simulation_delta_r_mean.to_numpy(),
        ),
        "delta_by_n_deleted": {
            str(int(n)): fit_metrics(
                local.theory_delta_r_mean.to_numpy(),
                local.simulation_delta_r_mean.to_numpy(),
            )
            for n, local in aggregate_frame.groupby("n_deleted")
        },
        "registered_deletion_patterns": 82,
        "analyzed_deletion_patterns": int(len(aggregate_frame)),
        "registered_heldout_conditions": int(len(HELDOUT_IDS)),
        "analyzed_network_condition_rows": int(len(condition_frame)),
        "excluded_network_condition_rows": int(len(exclusion_rows)),
        "calibration_residual_units_min": int(residual_units.min()),
        "calibration_residual_units_max": int(residual_units.max()),
        "fixedpoint_residual_max": float(np.max(maps["fixedpoint_residual"])),
        "rectification": {
            "intact_calibration": int(network_data[0]["calibration_rectification"].sum()),
            "intact_heldout": int(network_data[0]["heldout_rectification"].sum()),
            "deleted_calibration_patterns_positive": int(sum(network_data[n]["calibration_rectification"].sum() > 0 for n in range(1, 83))),
            "deleted_heldout_patterns_positive": int(sum(network_data[n]["heldout_rectification"].sum() > 0 for n in range(1, 83))),
        },
    }
    atomic_csv(HERE / "i2_predefined_condition_level_results.csv", condition_frame)
    atomic_csv(HERE / "i2_predefined_network_aggregate_results.csv", aggregate_frame)
    atomic_csv(HERE / "i2_predefined_intact_condition_results.csv", intact_frame)
    atomic_csv(HERE / "i2_predefined_exclusions.csv", pd.DataFrame(exclusion_rows))
    atomic_json(HERE / "i2_predefined_metrics.json", metrics)
    return intact_frame, aggregate_frame, metrics


def limits(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    use = np.isfinite(x) & np.isfinite(y)
    low = float(min(x[use].min(), y[use].min()))
    high = float(max(x[use].max(), y[use].max()))
    pad = max(0.07 * (high - low), 0.004)
    return low - pad, high + pad


def format_axis(ax, xlabel: str, ylabel: str, x: np.ndarray, y: np.ndarray, metrics: dict):
    low, high = limits(x, y)
    grid = np.linspace(low, high, 300)
    ax.plot([low, high], [low, high], color="#777777", ls="--", lw=1.5, zorder=1)
    if metrics.get("n", 0) >= 3:
        ax.plot(
            grid,
            metrics["ols_intercept"] + metrics["ols_slope"] * grid,
            color=BLACK,
            lw=2.1,
            zorder=2,
        )
    ax.set(xlim=(low, high), ylim=(low, high), xlabel=xlabel, ylabel=ylabel)
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.04,
        0.96,
        f"n = {metrics.get('n', 0)}\nr = {metrics.get('pearson_r', np.nan):.2f}\nslope = {metrics.get('ols_slope', np.nan):.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.5,
    )
    ax.tick_params(width=1.3, length=4)


def draw(intact: pd.DataFrame, aggregate: pd.DataFrame, metrics: dict[str, Any]) -> list[Path]:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8.5,
            "axes.linewidth": 1.35,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 7.5,
        }
    )
    outputs: list[Path] = []

    # Main i2: one pre-specified equal-weight condition-ensemble mean per pattern.
    x = aggregate.theory_delta_r_mean.to_numpy(float)
    y = aggregate.simulation_delta_r_mean.to_numpy(float)
    mx = aggregate.theory_delta_r_sd_across_conditions.to_numpy(float)
    my = aggregate.simulation_delta_r_sd_across_conditions.to_numpy(float)
    delta_metrics = metrics["delta_network_means_across_conditions"]
    low, high = limits(x, y)
    grid = np.linspace(low, high, 300)
    fig, ax = plt.subplots(figsize=(3.5, 3.35))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.77)
    for n_deleted in sorted(aggregate.n_deleted.unique()):
        color, marker, label = STYLES[int(n_deleted)]
        take = aggregate.n_deleted.to_numpy(int) == int(n_deleted)
        ax.scatter(
            x[take], y[take], s=38, marker=marker, color=color,
            edgecolor="white", linewidth=0.45, alpha=0.9, label=label, zorder=3,
        )
    ax.plot([low, high], [low, high], color="#777777", ls="--", lw=1.6, zorder=1)
    ax.plot(
        grid,
        delta_metrics["ols_intercept"] + delta_metrics["ols_slope"] * grid,
        color=BLACK,
        lw=2.25,
        zorder=2,
    )
    # A compact uncertainty key: the median horizontal/vertical condition SD.
    span = high - low
    ax.errorbar(
        [low + 0.80 * span], [low + 0.10 * span],
        xerr=[float(np.median(mx))], yerr=[float(np.median(my))],
        fmt="none", ecolor="0.35", elinewidth=0.7, capsize=2.0, zorder=4,
    )
    ax.set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel="Theory mean Δr (intact − deleted)",
        ylabel="Simulation mean Δr (intact − deleted)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.04, 0.96,
        f"n = {delta_metrics['n']} patterns\nr = {delta_metrics['pearson_r']:.2f}\nslope = {delta_metrics['ols_slope']:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
    )
    ax.legend(
        frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2,
        columnspacing=1.0, handletextpad=0.45, borderaxespad=0,
    )
    ax.tick_params(width=1.35, length=4.5)
    main_stem = HERE / "Figure2i2_predefined_unscreened_condition_ensemble"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=main_stem.with_suffix(".alignment.json"),
        overlay_svg=main_stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(main_stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(main_stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(main_stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(main_stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    outputs.append(main_stem)

    # Diagnostic correspondence: intact conditions, deleted pattern means, delta means.
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.22, top=0.84, wspace=0.38)
    xi = intact.theory_intact_r.to_numpy(float)
    yi = intact.simulation_intact_r.to_numpy(float)
    axes[0].scatter(xi, yi, s=38, color=BLUE, edgecolor="white", linewidth=0.5, zorder=3)
    format_axis(
        axes[0], "Theory intact r", "Simulation intact r", xi, yi,
        metrics["intact_across_predefined_conditions"],
    )
    axes[0].set_title("a  Intact network", loc="left", fontweight="bold", fontsize=8.5)

    xd = aggregate.theory_deleted_r_mean.to_numpy(float)
    yd = aggregate.simulation_deleted_r_mean.to_numpy(float)
    for n_deleted in sorted(aggregate.n_deleted.unique()):
        color, marker, _label = STYLES[int(n_deleted)]
        take = aggregate.n_deleted.to_numpy(int) == int(n_deleted)
        axes[1].scatter(
            xd[take], yd[take], s=31, marker=marker, color=color,
            edgecolor="white", linewidth=0.4, alpha=0.9, zorder=3,
        )
    format_axis(
        axes[1], "Theory deleted r (mean)", "Simulation deleted r (mean)", xd, yd,
        metrics["deleted_network_means_across_conditions"],
    )
    axes[1].set_title("b  Deleted networks", loc="left", fontweight="bold", fontsize=8.5)

    for n_deleted in sorted(aggregate.n_deleted.unique()):
        color, marker, label = STYLES[int(n_deleted)]
        take = aggregate.n_deleted.to_numpy(int) == int(n_deleted)
        axes[2].scatter(
            x[take], y[take], s=31, marker=marker, color=color,
            edgecolor="white", linewidth=0.4, alpha=0.9, label=label, zorder=3,
        )
    format_axis(
        axes[2], "Theory Δr (mean)", "Simulation Δr (mean)", x, y,
        metrics["delta_network_means_across_conditions"],
    )
    axes[2].set_title("c  Intact − deleted", loc="left", fontweight="bold", fontsize=8.5)
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(
        handles, labels, frameon=False, loc="upper center",
        bbox_to_anchor=(0.5, 0.995), ncol=4,
        columnspacing=0.8, handletextpad=0.35, borderaxespad=0,
    )
    diag_stem = HERE / "Figure_i2_predefined_ensemble_correspondence_diagnostic"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=diag_stem.with_suffix(".alignment.json"),
        overlay_svg=diag_stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(diag_stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(diag_stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(diag_stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(diag_stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    outputs.append(diag_stem)
    return outputs


def write_figure_contract() -> None:
    atomic_json(
        HERE / "figure_contract.json",
        {
            "core_conclusion": (
                "Assess intact, deleted, and delta-r theory-simulation correspondence "
                "on a pre-defined unscreened Experiment-1 condition ensemble"
            ),
            "figure_archetype": "quantitative validation grid",
            "hero_panel": "delta-r network-pattern means across the 12 fixed conditions",
            "panels": [
                "intact network across 12 pre-defined held-out conditions",
                "82 deleted-network equal-weight condition means",
                "82 delta-r equal-weight condition means",
            ],
            "independent_fit_units": {
                "intact": "12 pre-defined held-out conditions",
                "deleted": "82 deletion-pattern means",
                "delta": "82 deletion-pattern means",
            },
            "uncertainty": (
                "median horizontal/vertical condition-wise standard deviation is shown "
                "as a scale key; every pattern-specific SD is retained in the aggregate table"
            ),
            "reviewer_risk": (
                "network x condition rows are nested and therefore are retained in the "
                "data table but not treated as 984 independent OLS units"
            ),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=83)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    write_figure_contract()
    freeze_protocol()
    if args.freeze:
        return
    if args.simulate:
        run_simulation(args.workers, args.start, args.end, args.force)
    if args.assemble:
        intact, aggregate, metrics = assemble()
        draw(intact, aggregate, metrics)
    if not args.simulate and not args.assemble:
        run_simulation(args.workers, args.start, args.end, args.force)
        intact, aggregate, metrics = assemble()
        draw(intact, aggregate, metrics)


if __name__ == "__main__":
    main()
