from __future__ import annotations

"""Recompute Fig. 2i with condition-matched nonlinear calibration.

The held-out inputs, held-out nonlinear Heun features, deletion patterns,
second-order transfer-derived feature maps, correlation convention and plot
format are frozen to the existing version-6 + two-node log-axis branch.
Only calibration is changed: for every held-out condition, three independent
complete nonlinear Heun realizations use the exact same 64 parameter vectors.
The resulting condition-specific single-realization covariance is added as Ce,
without division by the number of realizations.
"""

import argparse
import hashlib
import importlib.util
import json
import os
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

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


HERE = Path(__file__).resolve().parent
HISTORY = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "workspace"
RERUN = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "rerun"
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
V6_DIR = RERUN / "outputs/i2_predefined_exp1_ensemble_20260915"
V6_SCRIPT = V6_DIR / "run_i2_predefined_exp1_ensemble.py"
SECOND_DIR = RERUN / "outputs/i2_second_order_exp1_matched_exact_heun_20260915"
ADD2_DIR = HISTORY / "outputs/i2_add_2nodes_v1_v6_20260930"
ADD2_SCRIPT = ADD2_DIR / "run_add_2nodes.py"
EXP1_SCRIPT = HISTORY / "outputs/wholebrain_exp1_resting_heun_20260909/run_exp1_resting_heun.py"
I2_RUNNER = HISTORY / "outputs/wholebrain_H_heun_20260908/run_H_heun.py"
PLOT_SCRIPT = ADD2_DIR / "plot_abs_log_axes.py"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"

OUT = HERE / "fig2i_matched_calibration_Ce"
CACHE = OUT / "matched_calibration_features"
PROTOCOL = "fig2i_v6_plus2_matched_condition_calibration_Ce_v1"
HELDOUT_IDS = np.rint(np.linspace(0, 999, 12)).astype(int)
N_PHASES = 3
N_CONDITIONS = 12
N_TRIALS = 64
MIN_VALID = 8
CAL_SEED_OFFSET = 2_000_000

sys.path[:0] = [str(PACKAGE), str(HISTORY / "outputs"), str(FIGURE_SKILL / "scripts")]
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temp, **arrays)
    temp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False)
    temp.replace(path)


_W_V6 = None
_W_ADD2 = None
_W_EXP1 = None
_W_I2 = None
_W_MODEL = None
_W_OLD_PATTERNS = None
_W_OLD_MAPS = None
_W_NEW_PATTERNS = None
_W_NEW_MAPS = None
_W_DESIGN = None


def worker_init() -> None:
    global _W_V6, _W_ADD2, _W_EXP1, _W_I2, _W_MODEL
    global _W_OLD_PATTERNS, _W_OLD_MAPS, _W_NEW_PATTERNS, _W_NEW_MAPS, _W_DESIGN
    _W_V6 = load_module("fig2i_matched_v6", V6_SCRIPT)
    _W_ADD2 = load_module("fig2i_matched_add2", ADD2_SCRIPT)
    _W_EXP1 = load_module("fig2i_matched_exp1", EXP1_SCRIPT)
    _W_I2 = load_module("fig2i_matched_i2", I2_RUNNER)
    _W_MODEL, _W_OLD_PATTERNS, _unused_z, _W_OLD_MAPS, _unused_design = _W_I2.build_context()
    _v1, model2, _W_NEW_PATTERNS, _unused_z2, _W_NEW_MAPS = _W_ADD2.contexts()
    if model2.areas != _W_MODEL.areas:
        raise RuntimeError("Old and two-node model area orders differ")
    with np.load(V6_DIR / "protocol_locked_design.npz") as saved:
        _W_DESIGN = {key: saved[key].copy() for key in saved.files}


def network_count() -> int:
    if _W_MODEL is None:
        worker_init()
    return 1 + len(_W_OLD_PATTERNS) + len(_W_NEW_PATTERNS)


def descriptor(global_position: int) -> tuple[np.ndarray, np.ndarray | None, str, int]:
    """Return fixed point, FLN, label and source-local position."""
    if global_position <= len(_W_OLD_PATTERNS):
        label = "intact" if global_position == 0 else str(_W_OLD_PATTERNS.iloc[global_position - 1].deleted_nodes)
        return (
            _W_OLD_MAPS["fixedpoints"][global_position],
            _W_OLD_MAPS["flns"][global_position],
            label,
            global_position,
        )
    new_position = global_position - len(_W_OLD_PATTERNS)
    label = str(_W_NEW_PATTERNS.iloc[new_position - 1].deleted_nodes)
    return (
        _W_NEW_MAPS["fixedpoints"][new_position],
        _W_NEW_MAPS["flns"][new_position],
        label,
        new_position,
    )


def cache_path(global_position: int) -> Path:
    return CACHE / f"network_{global_position:03d}.npz"


def metadata(global_position: int) -> dict[str, Any]:
    _fixedpoint, _fln, label, local_position = descriptor(global_position)
    branch = "registered_82" if global_position <= len(_W_OLD_PATTERNS) else "two_node_extension"
    return {
        "protocol": PROTOCOL,
        "global_network_position": global_position,
        "source_branch": branch,
        "source_local_position": local_position,
        "deleted_nodes": label,
        "shape": [N_PHASES, N_CONDITIONS, N_TRIALS, 2],
        "heldout_condition_ids": HELDOUT_IDS.tolist(),
        "model": "complete nonlinear 55-area x E/I threshold-linear network",
        "integrator": "explicit Heun; rectification evaluated at predictor and corrector stages",
        "input_area": "A6Va",
        "output_area": "V2",
        "dt_ms": 0.25,
        "burn_s": 2.048,
        "analysis_s": 32.768,
        "calibration_seed_rule": (
            "DENSE_SEED + 2,000,000 + phase*100,000 + condition_id*31"
        ),
        "input_parameter_vectors": "exact heldout_z vectors from frozen version-6 design",
        "forcing": "Experiment-1 Fourier forcing, independently regenerated for calibration",
        "design_sha256": sha256(V6_DIR / "protocol_locked_design.npz"),
    }


def save_cache(
    path: Path,
    meta: dict[str, Any],
    features: np.ndarray,
    valid: np.ndarray,
    rectification: np.ndarray,
    completed: np.ndarray,
) -> None:
    atomic_npz(
        path,
        metadata=np.asarray(json.dumps(meta, sort_keys=True)),
        features=features,
        valid=valid,
        rectification=rectification,
        completed=completed,
    )


def simulate_one(global_position: int, force: bool = False) -> dict[str, Any]:
    if _W_MODEL is None:
        worker_init()
    path = cache_path(global_position)
    meta = metadata(global_position)
    features = np.full((N_PHASES, N_CONDITIONS, N_TRIALS, 2), np.nan, np.float32)
    valid = np.zeros((N_PHASES, N_CONDITIONS, N_TRIALS), bool)
    rectification = np.zeros((N_PHASES, N_CONDITIONS), np.int64)
    completed = np.zeros((N_PHASES, N_CONDITIONS), bool)
    if path.exists() and not force:
        with np.load(path) as saved:
            prior = json.loads(str(saved["metadata"].item()))
            if prior != meta:
                raise RuntimeError(f"Cache metadata mismatch for network {global_position}")
            features = saved["features"].copy()
            valid = saved["valid"].copy()
            rectification = saved["rectification"].copy()
            completed = saved["completed"].copy()
        if completed.all():
            return {"network": global_position, "status": "cached"}

    fixedpoint, fln, label, _local_position = descriptor(global_position)
    input_idx = _W_MODEL.areas.index(_W_I2.source.INPUT_AREA)
    output_idx = _W_MODEL.areas.index(_W_I2.source.OUTPUT_AREA)
    held_z = _W_DESIGN["heldout_z"]
    burn_steps = int(round(_W_I2.source.BURN_S * 1000.0 / _W_I2.source.DT_MS))
    n_steps = int(round(_W_I2.ANALYSIS_S * 1000.0 / _W_I2.source.DT_MS))
    for phase in range(N_PHASES):
        for cpos, condition_id in enumerate(HELDOUT_IDS):
            if completed[phase, cpos]:
                continue
            seed = int(_W_EXP1.DENSE_SEED + CAL_SEED_OFFSET + phase * 100_000 + int(condition_id) * 31)
            inputs = _W_EXP1._fourier_inputs(
                held_z[cpos],
                _W_I2.source.DT_MS,
                n_steps,
                burn_steps,
                _W_I2.source.INPUT_SIGMA,
                seed,
            )
            paths, counts = _W_I2.network_run_fixedpoint_heun(
                _W_MODEL,
                inputs,
                np.asarray([output_idx], np.int64),
                input_idx,
                fixedpoint,
                fln,
            )
            values, good = _W_I2.source._feature_from_paths(paths, _W_I2.ANALYSIS_S)
            features[phase, cpos] = values
            valid[phase, cpos] = good
            rectification[phase, cpos] = int(counts.sum())
            completed[phase, cpos] = True
            save_cache(path, meta, features, valid, rectification, completed)
    return {"network": global_position, "status": "complete", "deleted_nodes": label}


def run_simulation(workers: int, start: int, end: int, force: bool) -> None:
    worker_init()
    total = network_count()
    indices = list(range(max(0, start), min(total, end)))
    started = time.time()
    failures: list[dict[str, Any]] = []
    completed = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = {pool.submit(simulate_one, pos, force): pos for pos in indices}
        for future in as_completed(futures):
            pos = futures[future]
            try:
                result = future.result()
                completed += 1
                print(f"Fig2i network {pos:03d}: {result['status']} ({completed}/{len(indices)})", flush=True)
            except Exception as exc:
                failures.append({"network": pos, "error": repr(exc), "traceback": traceback.format_exc()})
            atomic_json(
                OUT / "simulation_progress.json",
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


def load_calibration(global_position: int) -> dict[str, np.ndarray]:
    path = cache_path(global_position)
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    with np.load(path) as saved:
        if json.loads(str(saved["metadata"].item())) != metadata(global_position):
            raise RuntimeError(f"Metadata mismatch for {path}")
        if not saved["completed"].all():
            raise RuntimeError(f"Incomplete {path}")
        return {key: saved[key].copy() for key in saved.files if key != "metadata"}


def condition_residual_covariance(item: dict[str, np.ndarray], exp1) -> tuple[np.ndarray, np.ndarray]:
    covariances = np.full((N_CONDITIONS, 2, 2), np.nan)
    units = np.zeros(N_CONDITIONS, int)
    for cpos in range(N_CONDITIONS):
        features = np.asarray(item["features"][:, cpos], float)
        good = np.asarray(item["valid"][:, cpos], bool).all(axis=0)
        good &= np.isfinite(features).all(axis=(0, 2))
        if good.sum() < MIN_VALID:
            continue
        covariances[cpos] = exp1._pooled_within_trial_cov(features[:, good])
        units[cpos] = int(good.sum())
    return covariances, units


def corr(values: np.ndarray, valid: np.ndarray) -> float:
    values = np.asarray(values, float)
    good = np.asarray(valid, bool) & np.isfinite(values).all(axis=1)
    if good.sum() < MIN_VALID or np.min(np.std(values[good], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[good, 0], values[good, 1]).statistic)


def cov_to_corr(cov: np.ndarray) -> float:
    if cov.shape != (2, 2) or not np.isfinite(cov).all() or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return np.nan
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def taylor2(base: np.ndarray, gradient: np.ndarray, hessian: np.ndarray, u: np.ndarray) -> np.ndarray:
    return base[None, :] + u @ gradient.T + 0.5 * np.einsum("ni,kij,nj->nk", u, hessian, u)


def load_old_heldout(v6, position: int) -> dict[str, np.ndarray]:
    return v6.load_network_features(position)


def load_new_heldout(add2, position: int) -> dict[str, np.ndarray]:
    return add2.load_v6_cache(position)


def aggregate_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for position, local in frame.groupby("global_network_position", sort=True):
        first = local.iloc[0]
        rows.append(
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
    return pd.DataFrame(rows)


def assemble() -> None:
    worker_init()
    v6, add2, exp1 = _W_V6, _W_ADD2, _W_EXP1
    with np.load(V6_DIR / "protocol_locked_design.npz") as saved:
        held_u = saved["heldout_u"].copy()
    with np.load(SECOND_DIR / "i2_exp1_second_order_maps.npz") as saved:
        old_base = saved["base"].copy()
        old_gradient = saved["gradient_u"].copy()
        old_hessian = saved["hessian_u"].copy()
    with np.load(ADD2_DIR / "version6_with_2nodes/two_node_second_order_maps.npz") as saved:
        new_base = saved["base"].copy()
        new_gradient = saved["gradient_u"].copy()
        new_hessian = saved["hessian_u"].copy()

    baseline_held = load_old_heldout(v6, 0)
    baseline_cal = load_calibration(0)
    baseline_ce, baseline_units = condition_residual_covariance(baseline_cal, exp1)
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []

    def process_pattern(
        global_position: int,
        n_deleted: int,
        deleted_nodes: str,
        held: dict[str, np.ndarray],
        base_map: np.ndarray,
        gradient_map: np.ndarray,
        hessian_map: np.ndarray,
    ) -> None:
        deleted_cal = load_calibration(global_position)
        deleted_ce, deleted_units = condition_residual_covariance(deleted_cal, exp1)
        for cpos, condition_id in enumerate(HELDOUT_IDS):
            intact_values = taylor2(old_base[0], old_gradient[0], old_hessian[0], held_u[cpos])
            deleted_values = taylor2(base_map, gradient_map, hessian_map, held_u[cpos])
            theory_intact = cov_to_corr(np.cov(intact_values, rowvar=False, ddof=1) + baseline_ce[cpos])
            theory_deleted = cov_to_corr(np.cov(deleted_values, rowvar=False, ddof=1) + deleted_ce[cpos])
            intact_valid = baseline_held["heldout_valid"][cpos] & np.isfinite(baseline_held["heldout_features"][cpos]).all(axis=1)
            deleted_valid = held["heldout_valid"][cpos] & np.isfinite(held["heldout_features"][cpos]).all(axis=1)
            simulation_intact = corr(baseline_held["heldout_features"][cpos], intact_valid)
            simulation_deleted = corr(held["heldout_features"][cpos], deleted_valid)
            if not np.isfinite([theory_intact, theory_deleted, simulation_intact, simulation_deleted]).all():
                exclusions.append(
                    {
                        "global_network_position": global_position,
                        "deleted_nodes": deleted_nodes,
                        "condition_id": int(condition_id),
                        "reason": "pre-specified numerical-validity rule",
                    }
                )
                continue
            rows.append(
                {
                    "global_network_position": global_position,
                    "n_deleted": n_deleted,
                    "deleted_nodes": deleted_nodes,
                    "condition_id": int(condition_id),
                    "theory_intact_r": theory_intact,
                    "theory_deleted_r": theory_deleted,
                    "simulation_intact_r": simulation_intact,
                    "simulation_deleted_r": simulation_deleted,
                    "theory_delta_r": theory_intact - theory_deleted,
                    "simulation_delta_r": simulation_intact - simulation_deleted,
                    "intact_calibration_units": int(baseline_units[cpos]),
                    "deleted_calibration_units": int(deleted_units[cpos]),
                    "Ce_scaling": 1.0,
                }
            )

    for position, pattern in enumerate(_W_OLD_PATTERNS.itertuples(index=False), 1):
        process_pattern(
            position,
            int(pattern.n_deleted),
            str(pattern.deleted_nodes),
            load_old_heldout(v6, position),
            old_base[position],
            old_gradient[position],
            old_hessian[position],
        )
    offset = len(_W_OLD_PATTERNS)
    for position, pattern in enumerate(_W_NEW_PATTERNS.itertuples(index=False), 1):
        process_pattern(
            offset + position,
            2,
            str(pattern.deleted_nodes),
            load_new_heldout(add2, position),
            new_base[position - 1],
            new_gradient[position - 1],
            new_hessian[position - 1],
        )

    condition_frame = pd.DataFrame(rows)
    aggregate = aggregate_rows(condition_frame)
    atomic_csv(OUT / "condition_level_results.csv", condition_frame)
    atomic_csv(OUT / "network_aggregate_results.csv", aggregate)
    atomic_csv(OUT / "exclusions.csv", pd.DataFrame(exclusions))
    metrics = log_metrics(
        aggregate["theory_delta_r_mean"].to_numpy(float),
        aggregate["simulation_delta_r_mean"].to_numpy(float),
    )
    atomic_json(OUT / "metrics_abs_log.json", metrics)
    atomic_json(
        OUT / "protocol_manifest.json",
        {
            "protocol": PROTOCOL,
            "only_changed_component": (
                "calibration inputs now use the exact 12 held-out condition parameter ensembles; "
                "three independent nonlinear Heun realizations estimate a condition-specific Ce"
            ),
            "heldout": "reused unchanged version-6 and two-node complete nonlinear Heun features",
            "theory": "unchanged network-specific second-order four-parameter Taylor maps",
            "Ce_scaling": "Ce added directly (multiplier 1; no division by 2 or 5)",
            "calibration_seed_rule": (
                "DENSE_SEED + 2,000,000 + phase*100,000 + condition_id*31"
            ),
            "conditions": HELDOUT_IDS,
            "trials_per_condition": N_TRIALS,
            "calibration_realizations": N_PHASES,
            "n_network_patterns_plotted": len(aggregate),
            "metrics": metrics,
        },
    )
    draw(aggregate)


def log_metrics(theory: np.ndarray, simulation: np.ndarray) -> dict[str, Any]:
    good = np.isfinite(theory) & np.isfinite(simulation) & (theory != 0) & (simulation != 0)
    x = np.log10(np.abs(theory[good]))
    y = np.log10(np.abs(simulation[good]))
    fit = stats.linregress(x, y)
    return {
        "n": int(good.sum()),
        "pearson_r_log10_abs": float(fit.rvalue),
        "log_log_slope": float(fit.slope),
        "log_log_intercept": float(fit.intercept),
        "R2_log10_abs": float(fit.rvalue**2),
    }


def draw(aggregate: pd.DataFrame) -> None:
    raw_x = aggregate["theory_delta_r_mean"].to_numpy(float)
    raw_y = aggregate["simulation_delta_r_mean"].to_numpy(float)
    x = np.abs(raw_x)
    y = np.abs(raw_y)
    use = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if int(use.sum()) != len(aggregate):
        raise ValueError("Fig. 2i contains zero or non-finite values that cannot be shown on log axes")
    metrics = log_metrics(raw_x, raw_y)
    values = np.r_[x[use], y[use]]
    low = float(10 ** (np.log10(values.min()) - 0.10))
    high = float(10 ** (np.log10(values.max()) + 0.10))
    grid = np.logspace(np.log10(low), np.log10(high), 400)

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
    fig, ax = plt.subplots(figsize=(3.5, 3.35))
    fig.subplots_adjust(left=0.18, right=0.97, bottom=0.16, top=0.77)
    styles = {
        1: ("#2B8CBE", "o", "1 node"),
        2: ("#D1495B", "P", "2 nodes"),
        3: ("#FDAE6B", "^", "3 nodes"),
        4: ("#41AB5D", "s", "4 nodes"),
        5: ("#9E9AC8", "D", "5 nodes"),
        12: ("0.45", "|", "12 nodes"),
    }
    counts = aggregate["n_deleted"].to_numpy(int)
    for n_deleted in sorted(aggregate["n_deleted"].unique()):
        color, marker, label = styles[int(n_deleted)]
        take = use & (counts == int(n_deleted))
        ax.scatter(
            x[take],
            y[take],
            s=38,
            marker=marker,
            color=color,
            edgecolor="white",
            linewidth=0.45,
            alpha=0.9,
            label=label,
            zorder=3,
        )
    ax.plot(grid, grid, color="#777777", ls="--", lw=1.6, zorder=1)
    fitted = 10 ** metrics["log_log_intercept"] * grid ** metrics["log_log_slope"]
    ax.plot(grid, fitted, color="#202020", lw=2.25, zorder=2)

    mx = aggregate["theory_delta_r_sd_across_conditions"].to_numpy(float)
    my = aggregate["simulation_delta_r_sd_across_conditions"].to_numpy(float)
    med_xerr = float(np.nanmedian(mx))
    med_yerr = float(np.nanmedian(my))
    log_low, log_high = np.log10(low), np.log10(high)
    span = log_high - log_low
    x0 = max(10 ** (log_low + 0.80 * span), 1.25 * med_xerr)
    y0 = max(10 ** (log_low + 0.10 * span), 1.25 * med_yerr)
    if x0 + med_xerr < high and y0 + med_yerr < high:
        ax.errorbar(
            [x0],
            [y0],
            xerr=[med_xerr],
            yerr=[med_yerr],
            fmt="none",
            ecolor="0.35",
            elinewidth=0.7,
            capsize=2.0,
            zorder=4,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel="|Theory mean Δr (intact − deleted)|",
        ylabel="|Simulation mean Δr (intact − deleted)|",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.04,
        0.96,
        f"n = {metrics['n']} patterns\nr = {metrics['pearson_r_log10_abs']:.2f}\nslope = {metrics['log_log_slope']:.2f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
    )
    ax.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=2,
        columnspacing=1.0,
        handletextpad=0.45,
        borderaxespad=0,
    )
    ax.tick_params(width=1.35, length=4.5)
    source = aggregate.copy()
    source["abs_theory_delta_r_mean"] = x
    source["abs_simulation_delta_r_mean"] = y
    source["log10_abs_theory_delta_r_mean"] = np.log10(x)
    source["log10_abs_simulation_delta_r_mean"] = np.log10(y)
    atomic_csv(OUT / "plot_source_data_abs_log.csv", source)
    atomic_json(OUT / "plot_metrics_abs_log.json", metrics)
    stem = OUT / "Figure2i_matched_calibration_Ce"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=10_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.all or args.simulate:
        run_simulation(args.workers, args.start, args.end, args.force)
    if args.all or args.assemble:
        assemble()
    if not (args.all or args.simulate or args.assemble):
        raise SystemExit("Choose --simulate, --assemble, or --all")


if __name__ == "__main__":
    main()
