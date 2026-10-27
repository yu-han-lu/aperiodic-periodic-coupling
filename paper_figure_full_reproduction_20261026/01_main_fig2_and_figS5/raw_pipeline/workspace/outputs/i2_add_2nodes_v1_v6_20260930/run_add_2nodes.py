from __future__ import annotations

"""Add a protocol-matched two-node deletion stratum to i2 versions 1 and 6.

No existing observation is recomputed or changed.  New two-node patterns are
registered once and used by both branches.  Version 1 retains its complete
nonlinear Heun simulation with five held-out PSDs averaged before feature
extraction and its full transfer-derived feature map.  Version 6 retains its
predefined Experiment-1 condition ensemble, complete nonlinear Heun
calibration/held-out simulation, and second-order input-parameter feature map.
"""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
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
HISTORY = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "workspace"
RERUN = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "rerun"
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
V1_DIR = HISTORY / "outputs/wholebrain_H_heun_20260908"
V1_SCRIPT = V1_DIR / "run_H_heun.py"
V6_DIR = RERUN / "outputs/i2_predefined_exp1_ensemble_20260915"
V6_SCRIPT = V6_DIR / "run_i2_predefined_exp1_ensemble.py"
SECOND_DIR = RERUN / "outputs/i2_second_order_exp1_matched_exact_heun_20260915"
SECOND_SCRIPT = SECOND_DIR / "run_i2_second_order_exp1_matched_exact_heun.py"
EXP1_SCRIPT = HISTORY / "outputs/wholebrain_exp1_resting_heun_20260909/run_exp1_resting_heun.py"
SINGLES_CSV = PACKAGE / "outputs/single_node_deletion_all_nodes/all_single_node_deletion_results.csv"
COUNT_DIR = HISTORY / "outputs/deletion_count_direct_nondirect_20260910"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(PACKAGE), str(HISTORY / "outputs"), str(FIGURE_SKILL / "scripts")]

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


SEED = 20260818
TOP = 40
STRATA = 16
PAIR_PROTOCOL = (
    "eight outcome-blind two-node patterns from the independently pre-registered "
    "deletion-count analysis whose same-protocol version-1 caches had all five "
    "held-out realizations complete (four direct and four non-direct); all had a "
    "stable fixed point with active A6Va input and V2 output"
)
TWO_COLOR = "#D1495B"
TWO_MARKER = "P"
V1_OUT = HERE / "version1_with_2nodes"
V6_OUT = HERE / "version6_with_2nodes"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def selected_pairs() -> pd.DataFrame:
    path = HERE / "two_node_patterns_complete8_used.csv"
    if path.exists():
        return pd.read_csv(path)
    all_new = pd.read_csv(COUNT_DIR / "new_patterns.csv").reset_index(drop=True)
    all_new["source_network_position"] = np.arange(1, len(all_new) + 1)
    selected = all_new.loc[all_new.n_deleted == 2].copy().reset_index(drop=True)
    complete = []
    for item in selected.itertuples(index=False):
        cache = COUNT_DIR / "nonlinear_states" / f"network_{int(item.source_network_position):03d}.npz"
        if not cache.exists():
            complete.append(False)
            continue
        with np.load(cache) as saved:
            complete.append(bool(saved["completed"].all()))
    selected = selected.loc[np.asarray(complete, bool)].reset_index(drop=True)
    if len(selected) != 8 or selected.direct.astype(bool).value_counts().to_dict() != {True: 4, False: 4}:
        raise RuntimeError("Expected eight complete pre-registered pairs balanced 4 direct / 4 non-direct")
    selected["nodes"] = selected.deleted_nodes
    selected["additive_screen_delta_r"] = np.nan
    atomic_csv(path, selected)
    return selected


def contexts() -> tuple[Any, Any, pd.DataFrame, np.ndarray, dict]:
    v1 = load_module("add2_v1", V1_SCRIPT)
    cfg = v1.source.load_config(v1.source.PACKAGE_ROOT / "configs/full.yaml")
    model = v1.source.load_model(PROJECT / cfg["model_dir"])
    with np.load(V1_DIR / "design.npz") as saved:
        z = saved["z"].copy()
    retained_path = HERE / "two_node_patterns_complete8_retained.csv"
    context_path = HERE / "two_node_network_context_complete8.npz"
    if retained_path.exists() and context_path.exists():
        retained = pd.read_csv(retained_path)
        with np.load(context_path) as saved:
            maps = {
                "jacobians": saved["jacobians"].copy(),
                "latent": saved["latent"].copy(),
                "latent_valid": saved["latent_valid"].copy(),
                "flns": [item.copy() for item in saved["flns"]],
                "fixedpoints": saved["fixedpoints"].copy(),
                "fixedpoint_residual": saved["fixedpoint_residual"].copy(),
                "active_e": saved["active_e"].copy(),
                "active_i": saved["active_i"].copy(),
                "frequencies": saved["frequencies"].copy(),
                "keep": saved["keep"].copy(),
            }
        return v1, model, retained, z, maps
    pairs = selected_pairs()
    maps = v1.source._network_maps(model, pairs, z, v1.ANALYSIS_S)
    input_idx = model.areas.index(v1.source.INPUT_AREA)
    output_idx = model.areas.index(v1.source.OUTPUT_AREA)
    stable = np.asarray([np.max(np.linalg.eigvals(j).real) < 0 for j in maps["jacobians"][1:]])
    residual_ok = maps["fixedpoint_residual"][1:] <= 1e-7
    input_active = maps["fixedpoints"][1:, input_idx] > 0
    output_active = maps["fixedpoints"][1:, output_idx] > 0
    retain = stable & residual_ok & input_active & output_active
    audit = pairs.copy()
    audit["stable_full_jacobian"] = stable
    audit["fixedpoint_residual"] = maps["fixedpoint_residual"][1:]
    audit["input_active"] = input_active
    audit["output_active"] = output_active
    audit["retained"] = retain
    atomic_csv(HERE / "two_node_pattern_precheck_complete8.csv", audit)
    retained = pairs.loc[retain].reset_index(drop=True)
    if retained.empty:
        raise RuntimeError("No two-node patterns passed the pre-specified structural checks")
    maps = v1.source._network_maps(model, retained, z, v1.ANALYSIS_S)
    atomic_csv(retained_path, retained)
    atomic_npz(
        context_path,
        jacobians=maps["jacobians"],
        latent=maps["latent"],
        latent_valid=maps["latent_valid"],
        flns=np.stack([model.fln if item is None else item for item in maps["flns"]]),
        fixedpoints=maps["fixedpoints"],
        fixedpoint_residual=maps["fixedpoint_residual"],
        active_e=maps["active_e"],
        active_i=maps["active_i"],
        frequencies=maps["frequencies"],
        keep=maps["keep"],
    )
    return v1, model, retained, z, maps


def fit_metrics(frame: pd.DataFrame, xcol: str, ycol: str) -> dict[str, Any]:
    use = np.isfinite(frame[xcol]) & np.isfinite(frame[ycol])
    x = frame.loc[use, xcol].to_numpy(float)
    y = frame.loc[use, ycol].to_numpy(float)
    fit = stats.linregress(x, y)
    return {
        "n": int(len(x)),
        "pearson_r": float(fit.rvalue),
        "ols_slope": float(fit.slope),
        "ols_intercept": float(fit.intercept),
        "ols_R2": float(fit.rvalue**2),
    }


def v1_worker(position: int) -> dict[str, Any]:
    v1, model, patterns, z, maps = contexts()
    v1.source._network_run_fixedpoint = v1.network_run_fixedpoint_heun
    cache_dir = V1_OUT / "nonlinear_states"
    cache_dir.mkdir(parents=True, exist_ok=True)
    v1.source._nonlinear_heldout_features(
        model,
        z,
        maps,
        v1.ANALYSIS_S,
        v1.N_HELDOUT,
        cache_dir,
        fixedpoint_init=True,
        network_indices=[position],
    )
    return {"position": position, "deleted_nodes": patterns.iloc[position - 1].deleted_nodes}


def run_v1_simulation(workers: int) -> None:
    _v1, _model, patterns, _z, _maps = contexts()
    out = V1_OUT
    cache_dir = out / "nonlinear_states"
    cache_dir.mkdir(parents=True, exist_ok=True)
    baseline_target = cache_dir / "network_000.npz"
    if not baseline_target.exists():
        shutil.copy2(V1_DIR / "nonlinear_states/network_000.npz", baseline_target)
    copied = 0
    for position, pattern in enumerate(patterns.itertuples(index=False), 1):
        source_position = int(pattern.source_network_position)
        source_path = COUNT_DIR / "nonlinear_states" / f"network_{source_position:03d}.npz"
        target_path = cache_dir / f"network_{position:03d}.npz"
        with np.load(source_path) as saved:
            if not saved["completed"].all():
                raise RuntimeError(f"Incomplete pre-existing exact-Heun cache: {source_path}")
        if not target_path.exists() or sha256(target_path) != sha256(source_path):
            shutil.copy2(source_path, target_path)
        copied += 1
    atomic_json(
        out / "simulation_progress.json",
        {
            "status": "complete",
            "two_node_networks": copied,
            "simulation": "reused exact-Heun caches from the same version-1 protocol",
            "source_manifest": str(COUNT_DIR / "selection_manifest.json"),
        },
    )


def draw_v1(combined: pd.DataFrame, metrics: dict[str, Any]) -> Path:
    v1 = load_module("add2_v1_draw", V1_SCRIPT)
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 13,
            "axes.linewidth": 1.7,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    x = combined.theory_delta_r.to_numpy(float)
    y = combined.simulation_delta_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y)
    low = float(min(x[use].min(), y[use].min()))
    high = float(max(x[use].max(), y[use].max()))
    pad = 0.04 * max(high - low, 0.1)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.96)
    colors = {1: "#247BA0", 2: TWO_COLOR, 3: "#F4A261", 4: "#2A9D8F", 5: "#8E5EA2"}
    markers = {1: "o", 2: TWO_MARKER, 3: "^", 4: "s", 5: "D"}
    for count in sorted(combined.n_deleted.unique()):
        take = use & (combined.n_deleted.to_numpy(int) == int(count))
        ax.scatter(x[take], y[take], s=28, marker=markers[int(count)], color=colors[int(count)], alpha=0.85, label=f"{int(count)} nodes")
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#555555", linestyle="--", linewidth=1.7)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.legend(frameon=False, loc="lower right")
    ax.text(0.05, 0.94, f"r = {metrics['pearson_r']:.2f}\nslope = {metrics['ols_slope']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=18, fontweight="bold")
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)
    stem = V1_OUT / "Figure2i2_version1_with_2nodes"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=stem.with_suffix(".alignment.json"), overlay_svg=stem.with_suffix(".alignment.svg"), strict=True)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    return stem


def assemble_v1() -> None:
    v1, model, patterns, z, maps = contexts()
    out = V1_OUT
    maps["transfers"] = v1.heun_transfers(model, maps)
    linear, linear_valid = v1.source._linear_calibration_features(z, maps, v1.ANALYSIS_S, out / "linear_calibration_features_heun.npz")
    v1.source._network_run_fixedpoint = v1.network_run_fixedpoint_heun
    nonlinear, nonlinear_valid, rectification = v1.source._nonlinear_heldout_features(
        model, z, maps, v1.ANALYSIS_S, v1.N_HELDOUT, out / "nonlinear_states", fixedpoint_init=True
    )
    new = v1.source._assemble(patterns, maps, linear, linear_valid, nonlinear, nonlinear_valid, rectification, v1.N_HELDOUT, v1.N_TRIALS, v1.ANALYSIS_S)
    new["baseline_rectification_fraction"] /= 2.0
    new["deleted_rectification_fraction"] /= 2.0
    new.insert(0, "integrator", "explicit Heun")
    new.insert(1, "covariance_tag", "cond_100")
    old = pd.read_csv(V1_DIR / "pattern_results_heun.csv")
    combined = pd.concat([old, new], ignore_index=True, sort=False)
    metrics = fit_metrics(combined, "theory_delta_r", "simulation_delta_r")
    atomic_csv(out / "two_node_results.csv", new)
    atomic_csv(out / "combined_results.csv", combined)
    atomic_json(out / "metrics.json", {"fit": metrics, "n_old": len(old), "n_two_node": len(new), "selection": PAIR_PROTOCOL})
    draw_v1(combined, metrics)


_W_V1 = None
_W_V6 = None
_W_EXP1 = None
_W_MODEL = None
_W_PATTERNS = None
_W_MAPS = None
_W_DESIGN = None
_W_CFG = None


def v6_worker_init() -> None:
    global _W_V1, _W_V6, _W_EXP1, _W_MODEL, _W_PATTERNS, _W_MAPS, _W_DESIGN, _W_CFG
    _W_V1, _W_MODEL, _W_PATTERNS, _z, _W_MAPS = contexts()
    _W_V6 = load_module("add2_v6_worker", V6_SCRIPT)
    _W_EXP1 = load_module("add2_exp1_worker", EXP1_SCRIPT)
    _W_CFG = _W_EXP1.model_bundle()["cfg"]
    with np.load(V6_DIR / "protocol_locked_design.npz") as saved:
        _W_DESIGN = {key: saved[key].copy() for key in saved.files}


def v6_cache_metadata(position: int) -> dict[str, Any]:
    return {
        "protocol": "v6_predefined_unscreened_exp1_ensemble_plus_2nodes",
        "network_position_within_two_node_extension": int(position),
        "deleted_nodes": str(_W_PATTERNS.iloc[position - 1].deleted_nodes) if _W_PATTERNS is not None else None,
        "integrator": "complete nonlinear 110-state explicit Heun",
        "calibration_shape": [3, 14, 64, 2],
        "heldout_condition_ids": np.rint(np.linspace(0, 999, 12)).astype(int).tolist(),
        "heldout_shape": [12, 64, 2],
        "input_area": "A6Va",
        "output_area": "V2",
        "analysis_s": 32.768,
        "burn_s": 2.048,
        "dt_ms": 0.25,
        "design_sha256": sha256(V6_DIR / "protocol_locked_design.npz"),
    }


def v6_cache_path(position: int) -> Path:
    return V6_OUT / "exact_heun_features" / f"two_node_{position:03d}.npz"


def v6_save_cache(path: Path, metadata: dict, cal_features: np.ndarray, cal_valid: np.ndarray, cal_rect: np.ndarray, cal_done: np.ndarray, held_features: np.ndarray, held_valid: np.ndarray, held_rect: np.ndarray, held_done: np.ndarray) -> None:
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


def v6_simulate_one(position: int) -> dict[str, Any]:
    if _W_V1 is None:
        v6_worker_init()
    v1, v6, exp1, model, maps, design = _W_V1, _W_V6, _W_EXP1, _W_MODEL, _W_MAPS, _W_DESIGN
    path = v6_cache_path(position)
    path.parent.mkdir(parents=True, exist_ok=True)
    metadata = v6_cache_metadata(position)
    cal_features = np.full((3, 14, 64, 2), np.nan, np.float32)
    cal_valid = np.zeros((3, 14, 64), bool)
    cal_rect = np.zeros((3, 14), np.int64)
    cal_done = np.zeros((3, 14), bool)
    held_features = np.full((12, 64, 2), np.nan, np.float32)
    held_valid = np.zeros((12, 64), bool)
    held_rect = np.zeros(12, np.int64)
    held_done = np.zeros(12, bool)
    if path.exists():
        with np.load(path) as saved:
            prior = json.loads(str(saved["metadata"].item()))
            if prior != metadata:
                raise RuntimeError(f"Version6 cache metadata mismatch for pair {position}")
            cal_features = saved["calibration_features"].copy()
            cal_valid = saved["calibration_valid"].copy()
            cal_rect = saved["calibration_rectification"].copy()
            cal_done = saved["calibration_completed"].copy()
            held_features = saved["heldout_features"].copy()
            held_valid = saved["heldout_valid"].copy()
            held_rect = saved["heldout_rectification"].copy()
            held_done = saved["heldout_completed"].copy()
            if cal_done.all() and held_done.all():
                return {"position": position, "status": "cached"}
    input_idx = model.areas.index(v1.source.INPUT_AREA)
    output_idx = model.areas.index(v1.source.OUTPUT_AREA)
    input_pos = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"].index(v1.source.INPUT_AREA)
    fixedpoint = maps["fixedpoints"][position]
    fln = maps["flns"][position]
    cal_z = design["calibration_z"]
    for phase in range(3):
        for condition_pos in range(14):
            if cal_done[phase, condition_pos]:
                continue
            inputs = exp1._matched_bandlimited_inputs(
                cal_z[condition_pos], v1.source.DT_MS, v1.ANALYSIS_S,
                v1.source.BURN_S, v1.source.INPUT_SIGMA,
                int(_W_CFG["seed"]) + 8100000 + phase * 100000 + input_pos * 10000 + condition_pos,
                trial_offset=0,
            )
            paths, counts = v1.network_run_fixedpoint_heun(model, inputs, np.asarray([output_idx], np.int64), input_idx, fixedpoint, fln)
            values, valid = v1.source._feature_from_paths(paths, v1.ANALYSIS_S)
            cal_features[phase, condition_pos] = values
            cal_valid[phase, condition_pos] = valid
            cal_rect[phase, condition_pos] = int(counts.sum())
            cal_done[phase, condition_pos] = True
            v6_save_cache(path, metadata, cal_features, cal_valid, cal_rect, cal_done, held_features, held_valid, held_rect, held_done)
    held_ids = np.rint(np.linspace(0, 999, 12)).astype(int)
    held_z = design["heldout_z"]
    burn_steps = int(round(v1.source.BURN_S * 1000.0 / v1.source.DT_MS))
    n_steps = int(round(v1.ANALYSIS_S * 1000.0 / v1.source.DT_MS))
    for condition_pos, condition_id in enumerate(held_ids):
        if held_done[condition_pos]:
            continue
        inputs = exp1._fourier_inputs(
            held_z[condition_pos], v1.source.DT_MS, n_steps, burn_steps,
            v1.source.INPUT_SIGMA, exp1.DENSE_SEED + 100000 + int(condition_id) * 31,
        )
        paths, counts = v1.network_run_fixedpoint_heun(model, inputs, np.asarray([output_idx], np.int64), input_idx, fixedpoint, fln)
        values, valid = v1.source._feature_from_paths(paths, v1.ANALYSIS_S)
        held_features[condition_pos] = values
        held_valid[condition_pos] = valid
        held_rect[condition_pos] = int(counts.sum())
        held_done[condition_pos] = True
        v6_save_cache(path, metadata, cal_features, cal_valid, cal_rect, cal_done, held_features, held_valid, held_rect, held_done)
    return {"position": position, "status": "complete", "deleted_nodes": _W_PATTERNS.iloc[position - 1].deleted_nodes}


def run_v6_simulation(workers: int) -> None:
    _v1, _model, patterns, _z, _maps = contexts()
    pending = []
    for position in range(1, len(patterns) + 1):
        path = v6_cache_path(position)
        complete = False
        if path.exists():
            with np.load(path) as saved:
                complete = bool(saved["calibration_completed"].all() and saved["heldout_completed"].all())
        if not complete:
            pending.append(position)
    if not pending:
        return
    failures = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=v6_worker_init) as pool:
        futures = {pool.submit(v6_simulate_one, position): position for position in pending}
        for done, future in enumerate(as_completed(futures), 1):
            position = futures[future]
            try:
                result = future.result()
                print(f"version6 pair {position}/{len(patterns)} {result['status']}", flush=True)
            except Exception as exc:
                failures.append({"position": position, "error": repr(exc), "traceback": traceback.format_exc()})
            atomic_json(V6_OUT / "simulation_progress.json", {"completed_this_run": done, "pending_at_start": len(pending), "elapsed_s": time.time() - started, "failures": failures})
    if failures:
        raise RuntimeError(f"Version6 simulation failures: {failures}")


def load_v6_cache(position: int) -> dict[str, np.ndarray]:
    path = v6_cache_path(position)
    with np.load(path) as saved:
        if not saved["calibration_completed"].all() or not saved["heldout_completed"].all():
            raise RuntimeError(f"Incomplete version6 cache for pair {position}")
        return {key: saved[key].copy() for key in saved.files if key != "metadata"}


def cov_to_corr(cov: np.ndarray) -> float:
    if cov.shape != (2, 2) or not np.isfinite(cov).all() or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return np.nan
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def draw_v6(combined: pd.DataFrame, metrics: dict[str, Any]) -> Path:
    v6 = load_module("add2_v6_draw", V6_SCRIPT)
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
    x = combined.theory_delta_r_mean.to_numpy(float)
    y = combined.simulation_delta_r_mean.to_numpy(float)
    mx = combined.theory_delta_r_sd_across_conditions.to_numpy(float)
    my = combined.simulation_delta_r_sd_across_conditions.to_numpy(float)
    low, high = v6.limits(x, y)
    grid = np.linspace(low, high, 300)
    fig, ax = plt.subplots(figsize=(3.5, 3.35))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.77)
    styles = dict(v6.STYLES)
    styles[2] = (TWO_COLOR, TWO_MARKER, "2 nodes")
    for n_deleted in sorted(combined.n_deleted.unique()):
        color, marker, label = styles[int(n_deleted)]
        take = combined.n_deleted.to_numpy(int) == int(n_deleted)
        ax.scatter(x[take], y[take], s=38, marker=marker, color=color, edgecolor="white", linewidth=0.45, alpha=0.9, label=label, zorder=3)
    ax.plot([low, high], [low, high], color="#777777", ls="--", lw=1.6, zorder=1)
    ax.plot(grid, metrics["ols_intercept"] + metrics["ols_slope"] * grid, color=v6.BLACK, lw=2.25, zorder=2)
    span = high - low
    ax.errorbar([low + 0.80 * span], [low + 0.10 * span], xerr=[float(np.median(mx))], yerr=[float(np.median(my))], fmt="none", ecolor="0.35", elinewidth=0.7, capsize=2.0, zorder=4)
    ax.set(xlim=(low, high), ylim=(low, high), xlabel="Theory mean Δr (intact − deleted)", ylabel="Simulation mean Δr (intact − deleted)")
    ax.set_aspect("equal", adjustable="box")
    ax.text(0.04, 0.96, f"n = {metrics['n']} patterns\nr = {metrics['pearson_r']:.2f}\nslope = {metrics['ols_slope']:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=8.5)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=2, columnspacing=1.0, handletextpad=0.45, borderaxespad=0)
    ax.tick_params(width=1.35, length=4.5)
    stem = V6_OUT / "Figure2i2_version6_with_2nodes"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=stem.with_suffix(".alignment.json"), overlay_svg=stem.with_suffix(".alignment.svg"), tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True)
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    return stem


def assemble_v6() -> None:
    v1, model, patterns, _z, maps = contexts()
    v6 = load_module("add2_v6_assemble", V6_SCRIPT)
    second = load_module("add2_second", SECOND_SCRIPT)
    exp1 = load_module("add2_exp1_assemble", EXP1_SCRIPT)
    with np.load(V6_DIR / "protocol_locked_design.npz") as saved:
        design = {key: saved[key].copy() for key in saved.files}
    baseline = v6.load_network_features(0)
    baseline_residual, baseline_units = v6.residual_covariance(baseline["calibration_features"], baseline["calibration_valid"], exp1)
    with np.load(SECOND_DIR / "i2_exp1_second_order_maps.npz") as saved:
        base0 = saved["base"][0].copy()
        gradient0 = saved["gradient_u"][0].copy()
        hessian0 = saved["hessian_u"][0].copy()
    input_idx = model.areas.index(v1.source.INPUT_AREA)
    output_idx = model.areas.index(v1.source.OUTPUT_AREA)
    new_maps = [second.second_order_map(j, input_idx, output_idx, v1.source) for j in maps["jacobians"][1:]]
    held_u = design["heldout_u"]
    held_ids = np.rint(np.linspace(0, 999, 12)).astype(int)
    rows = []
    aggregate_rows = []
    for position, pattern in enumerate(patterns.itertuples(index=False), 1):
        item = load_v6_cache(position)
        residual, units = v6.residual_covariance(item["calibration_features"], item["calibration_valid"], exp1)
        local_rows = []
        for cpos, condition_id in enumerate(held_ids):
            intact_values = v6.taylor2(base0, gradient0, hessian0, held_u[cpos])
            deleted_values = v6.taylor2(new_maps[position - 1]["base"], new_maps[position - 1]["gradient"], new_maps[position - 1]["hessian"], held_u[cpos])
            theory_intact = cov_to_corr(np.cov(intact_values, rowvar=False, ddof=1) + baseline_residual)
            theory_deleted = cov_to_corr(np.cov(deleted_values, rowvar=False, ddof=1) + residual)
            intact_valid = baseline["heldout_valid"][cpos] & np.isfinite(baseline["heldout_features"][cpos]).all(axis=1)
            deleted_valid = item["heldout_valid"][cpos] & np.isfinite(item["heldout_features"][cpos]).all(axis=1)
            simulation_intact = v6.corr(baseline["heldout_features"][cpos], intact_valid)
            simulation_deleted = v6.corr(item["heldout_features"][cpos], deleted_valid)
            if not np.isfinite([theory_intact, theory_deleted, simulation_intact, simulation_deleted]).all():
                continue
            row = pattern._asdict()
            row.update(
                {
                    "network_position": 1000 + position,
                    "condition_id": int(condition_id),
                    "theory_intact_r": theory_intact,
                    "theory_deleted_r": theory_deleted,
                    "simulation_intact_r": simulation_intact,
                    "simulation_deleted_r": simulation_deleted,
                    "theory_delta_r": theory_intact - theory_deleted,
                    "simulation_delta_r": simulation_intact - simulation_deleted,
                    "intact_residual_units": int(baseline_units),
                    "deleted_residual_units": int(units),
                }
            )
            rows.append(row)
            local_rows.append(row)
        if not local_rows:
            continue
        local = pd.DataFrame(local_rows)
        aggregate_rows.append(
            {
                "network_position": 1000 + position,
                "n_deleted": 2,
                "deleted_nodes": pattern.deleted_nodes,
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
    condition_new = pd.DataFrame(rows)
    aggregate_new = pd.DataFrame(aggregate_rows)
    old = pd.read_csv(V6_DIR / "i2_predefined_network_aggregate_results.csv")
    combined = pd.concat([old, aggregate_new], ignore_index=True, sort=False)
    metrics = fit_metrics(combined, "theory_delta_r_mean", "simulation_delta_r_mean")
    atomic_csv(V6_OUT / "two_node_condition_level_results.csv", condition_new)
    atomic_csv(V6_OUT / "two_node_network_aggregate_results.csv", aggregate_new)
    atomic_csv(V6_OUT / "combined_network_aggregate_results.csv", combined)
    atomic_json(V6_OUT / "metrics.json", {"fit": metrics, "n_old": len(old), "n_two_node": len(aggregate_new), "selection": PAIR_PROTOCOL})
    np.savez_compressed(
        V6_OUT / "two_node_second_order_maps.npz",
        base=np.asarray([m["base"] for m in new_maps]),
        gradient_u=np.asarray([m["gradient"] for m in new_maps]),
        hessian_u=np.asarray([m["hessian"] for m in new_maps]),
    )
    draw_v6(combined, metrics)


def write_manifest() -> None:
    atomic_json(
        HERE / "manifest.json",
        {
            "change": "add two-node deletion simulations to existing i2 versions 1 and 6",
            "unchanged_existing_points": True,
            "two_node_registration": PAIR_PROTOCOL,
            "seed": SEED,
            "version1": {
                "theory": "full transfer-derived feature map plus fixed-Jacobian linear calibration covariance divided by five",
                "simulation": "complete nonlinear explicit Heun; five held-out PSDs averaged before feature extraction",
                "delta": "deleted minus intact",
            },
            "version6": {
                "theory": "network-specific second-order four-parameter feature map plus complete nonlinear Heun calibration residual covariance",
                "simulation": "complete nonlinear explicit Heun for 12 predefined Experiment-1 held-out conditions",
                "delta": "intact minus deleted; equal-weight mean across 12 conditions",
            },
            "source_fingerprints": {
                "version1_program": sha256(V1_SCRIPT),
                "version1_existing_results": sha256(V1_DIR / "pattern_results_heun.csv"),
                "version6_program": sha256(V6_SCRIPT),
                "version6_existing_results": sha256(V6_DIR / "i2_predefined_network_aggregate_results.csv"),
            },
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--v1-simulate", action="store_true")
    parser.add_argument("--v1-assemble", action="store_true")
    parser.add_argument("--v6-simulate", action="store_true")
    parser.add_argument("--v6-assemble", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    write_manifest()
    if args.prepare:
        _v1, _model, patterns, _z, _maps = contexts()
        print(f"retained two-node patterns: {len(patterns)}", flush=True)
    if args.v1_simulate:
        run_v1_simulation(args.workers)
    if args.v1_assemble:
        assemble_v1()
    if args.v6_simulate:
        run_v6_simulation(args.workers)
    if args.v6_assemble:
        assemble_v6()
    if not any((args.prepare, args.v1_simulate, args.v1_assemble, args.v6_simulate, args.v6_assemble)):
        contexts()
        run_v1_simulation(args.workers)
        assemble_v1()
        run_v6_simulation(args.workers)
        assemble_v6()


if __name__ == "__main__":
    main()
