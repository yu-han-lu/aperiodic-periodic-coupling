from __future__ import annotations

"""Recompute Fig. 2j under a single-realization observation design.

All frozen inputs, deletion patterns, model parameters, numerical settings,
calibration trajectories, first-order feature maps and graphical styling are
retained.  The two complete nonlinear Heun held-out realizations are now kept
separate through feature extraction.  Correlations and deletion effects are
calculated within each realization and then averaged.  Accordingly, the
single-realization residual covariance Ce is added directly to the theoretical
signal covariance rather than divided by two.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
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
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from numba import set_num_threads
from scipy import signal, stats


HERE = Path(__file__).resolve().parent
HISTORY = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "workspace"
RERUN = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "rerun"
SOURCE_DIR = HISTORY / "outputs/deletion_count_direct_nondirect_20260910"
SOURCE_SCRIPT = SOURCE_DIR / "run_full55_count_direct_nondirect.py"
FIRST_DIR = RERUN / "outputs/first_order_rerun_20260914"
FIRST_SCRIPT = FIRST_DIR / "run_first_order_rerun.py"
FIRST_MAPS = FIRST_DIR / "i1_first_order_maps.npz"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
OUT = HERE / "fig2j_single_realization_Ce"
CACHE = OUT / "heldout_phase_features"
PROTOCOL = "fig2j_first_order_single_realization_Ce_v1"
N_PHASES = 2
N_TRIALS = 512
MIN_VALID = 8
COLORS = {
    ("Theory", True): "#D1495B",
    ("Theory", False): "#247BA0",
    ("Simulation", True): "#F28E2B",
    ("Simulation", False): "#2A9D8F",
}
BLACK = "#1A1A1A"

sys.path[:0] = [str(HISTORY / "outputs"), str(FIGURE_SKILL / "scripts")]
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


def atomic_csv(path: Path, frame: pd.DataFrame, compression=None) -> None:
    suffix = ".tmp.gz" if compression == "gzip" else ".tmp"
    temp = path.with_name(path.name + f".{os.getpid()}{suffix}")
    frame.to_csv(temp, index=False, compression=compression)
    temp.replace(path)


_W_COUNT = None
_W_MODEL = None
_W_SELECTED = None
_W_NETWORKS = None
_W_INPUT_IDX = None
_W_OUTPUTS = None
_W_HELD_INPUTS = None


def worker_init() -> None:
    global _W_COUNT, _W_MODEL, _W_SELECTED, _W_NETWORKS
    global _W_INPUT_IDX, _W_OUTPUTS, _W_HELD_INPUTS
    _W_COUNT = load_module("fig2j_count_source", SOURCE_SCRIPT)
    (
        _W_MODEL,
        _W_SELECTED,
        _unused_z,
        _W_NETWORKS,
        _W_INPUT_IDX,
        _W_OUTPUTS,
        _unused_frequencies,
        _unused_cal_inputs,
        _W_HELD_INPUTS,
    ) = _W_COUNT.build_context()


def cache_path(position: int) -> Path:
    return CACHE / f"network_{position:03d}.npz"


def metadata(position: int) -> dict[str, Any]:
    if position == 0:
        deleted_nodes = "intact"
    else:
        deleted_nodes = str(_W_SELECTED.iloc[position - 1].deleted_nodes)
    return {
        "protocol": PROTOCOL,
        "network_position": position,
        "deleted_nodes": deleted_nodes,
        "shape": [N_PHASES, 55, N_TRIALS, 2],
        "integrator": "complete nonlinear 110-state explicit Heun",
        "heldout_inputs": "unchanged frozen inputs from deletion-count analysis",
        "input_area": "A6Va",
        "outputs": "all 55 excitatory populations",
        "dt_ms": float(_W_COUNT.base.DT_MS),
        "burn_s": float(_W_COUNT.base.BURN_S),
        "analysis_s": float(_W_COUNT.ANALYSIS_S),
        "Welch_segment_s": float(_W_COUNT.base.SEGMENT_S),
        "Welch_overlap": "50%",
        "feature_extraction": "unchanged; now applied separately to each held-out phase",
        "source_script_sha256": sha256(SOURCE_SCRIPT),
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


def nonlinear_phase_features(
    model,
    inputs: np.ndarray,
    fln: np.ndarray,
    fixedpoint: np.ndarray,
    input_idx: int,
    output_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    set_num_threads(int(os.environ.get("NUMBA_NUM_THREADS", "3")))
    count = _W_COUNT
    frequencies, keep = count.base.frequencies_for(count.ANALYSIS_S)
    values = np.full((len(output_indices), len(inputs), 2), np.nan)
    valid = np.zeros((len(output_indices), len(inputs)), bool)
    psd_all = np.empty((len(output_indices), len(inputs), len(frequencies)), np.float32)
    rectification = 0
    fs = 1000.0 / count.base.DT_MS
    nperseg = int(round(count.base.SEGMENT_S * fs))
    arrays = count.base._simulation_arrays(model.params)
    edge_target, edge_source, edge_e, edge_i = count.base._edge_arrays(model.params, fln)
    n_area = len(model.areas)
    burn_steps = int(round(count.base.BURN_S * 1000.0 / count.base.DT_MS))
    for start in range(0, len(inputs), count.base.BAND_CHUNK):
        stop = min(start + count.base.BAND_CHUNK, len(inputs))
        paths, rect1, rect2 = count.band.threshold_network_heun_batch_fixedpoint(
            inputs[start:stop],
            arrays[0], arrays[1], arrays[2], arrays[3],
            edge_target, edge_source, edge_e, edge_i,
            arrays[8], arrays[9],
            fixedpoint[:n_area], fixedpoint[n_area:],
            count.base.DT_MS,
            float(model.params["tau_exc"]),
            float(model.params["tau_inh"]),
            burn_steps,
            output_indices,
            input_idx,
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
        psd_all[:, start:stop] = psd[..., keep].transpose(1, 0, 2)
        rectification += int(rect1.sum() + rect2.sum())
    values, valid = count.base.features_from_psd(frequencies, psd_all, count.PEAK_BAND)
    return values, valid, rectification


def simulate_one(position: int, force: bool = False) -> dict[str, Any]:
    if _W_MODEL is None:
        worker_init()
    path = cache_path(position)
    meta = metadata(position)
    features = np.full((N_PHASES, 55, N_TRIALS, 2), np.nan, np.float32)
    valid = np.zeros((N_PHASES, 55, N_TRIALS), bool)
    rectification = np.zeros(N_PHASES, np.int64)
    completed = np.zeros(N_PHASES, bool)
    if path.exists() and not force:
        with np.load(path) as saved:
            if json.loads(str(saved["metadata"].item())) != meta:
                raise RuntimeError(f"Cache metadata mismatch for network {position}")
            features = saved["features"].copy()
            valid = saved["valid"].copy()
            rectification = saved["rectification"].copy()
            completed = saved["completed"].copy()
        if completed.all():
            return {"network": position, "status": "cached"}
    fln, fixedpoint, _gain, _jacobian, _residual = _W_NETWORKS[position]
    for phase in range(N_PHASES):
        if completed[phase]:
            continue
        values, good, rect = nonlinear_phase_features(
            _W_MODEL,
            _W_HELD_INPUTS[phase],
            fln,
            fixedpoint,
            _W_INPUT_IDX,
            _W_OUTPUTS,
        )
        features[phase] = values
        valid[phase] = good
        rectification[phase] = rect
        completed[phase] = True
        save_cache(path, meta, features, valid, rectification, completed)
    return {"network": position, "status": "complete"}


def run_simulation(workers: int, start: int, end: int, force: bool) -> None:
    worker_init()
    indices = list(range(max(0, start), min(len(_W_NETWORKS), end)))
    failures: list[dict[str, Any]] = []
    completed = 0
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers, initializer=worker_init) as pool:
        futures = {pool.submit(simulate_one, position, force): position for position in indices}
        for future in as_completed(futures):
            position = futures[future]
            try:
                result = future.result()
                completed += 1
                print(f"Fig2j network {position:03d}: {result['status']} ({completed}/{len(indices)})", flush=True)
            except Exception as exc:
                failures.append({"network": position, "error": repr(exc), "traceback": traceback.format_exc()})
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


def load_phase_features(position: int) -> dict[str, np.ndarray]:
    path = cache_path(position)
    if not path.exists():
        raise RuntimeError(f"Missing {path}")
    with np.load(path) as saved:
        if not saved["completed"].all():
            raise RuntimeError(f"Incomplete {path}")
        return {key: saved[key].copy() for key in saved.files if key != "metadata"}


def corr(values: np.ndarray, use: np.ndarray) -> float:
    good = np.asarray(use, bool) & np.isfinite(values).all(axis=1)
    if good.sum() < MIN_VALID or np.min(np.std(values[good], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[good, 0], values[good, 1]).statistic)


def corr_from_cov(cov: np.ndarray) -> float:
    denom = float(cov[0, 0] * cov[1, 1])
    if not np.isfinite(cov).all() or denom <= 0:
        return np.nan
    return float(cov[0, 1] / np.sqrt(denom))


def pooled_phase_cov(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return np.cov(centered.reshape(-1, 2), rowvar=False, ddof=1)


def assemble() -> None:
    worker_init()
    count = _W_COUNT
    states = [count.load_state(count.STATE_ROOT / f"network_{position:03d}.npz") for position in range(61)]
    held = [load_phase_features(position) for position in range(61)]
    with np.load(FIRST_MAPS) as saved:
        gradients = saved["gradient_u"].copy()
        gradient_valid = saved["valid"].copy()
    with np.load(count.LATEST_ROOT / "design_512.npz") as saved:
        u = saved["u"].astype(float)
    if gradients.shape[:2] != (61, 55):
        raise RuntimeError(f"Unexpected first-order gradient shape {gradients.shape}")

    intact_state = states[0]
    intact_held = held[0]
    rows: list[dict[str, Any]] = []
    exclusions: list[dict[str, Any]] = []
    for position, pattern in enumerate(_W_SELECTED.itertuples(index=False), 1):
        deleted_state = states[position]
        deleted_held = held[position]
        deleted_nodes = str(pattern.deleted_nodes).split("|")
        deleted_indices = [_W_MODEL.areas.index(node) for node in deleted_nodes]
        gain = _W_NETWORKS[position][2]
        for output_idx, output_area in enumerate(_W_MODEL.areas):
            if output_idx in deleted_indices or not bool(gain[output_idx]):
                continue
            if not (gradient_valid[0, output_idx] and gradient_valid[position, output_idx]):
                exclusions.append({"network_position": position, "output_area": output_area, "reason": "invalid first-order derivative map"})
                continue
            common = (
                intact_held["valid"][:, output_idx].all(axis=0)
                & deleted_held["valid"][:, output_idx].all(axis=0)
                & np.isfinite(intact_held["features"][:, output_idx]).all(axis=(0, 2))
                & np.isfinite(deleted_held["features"][:, output_idx]).all(axis=(0, 2))
                & intact_state["calibration_valid"][:, output_idx].all(axis=0)
                & deleted_state["calibration_valid"][:, output_idx].all(axis=0)
            )
            if common.sum() < MIN_VALID:
                exclusions.append({"network_position": position, "output_area": output_area, "reason": f"fewer than {MIN_VALID} common trials ({int(common.sum())})"})
                continue
            sigma = np.cov(u[common], rowvar=False, ddof=1)
            base_signal = gradients[0, output_idx] @ sigma @ gradients[0, output_idx].T
            del_signal = gradients[position, output_idx] @ sigma @ gradients[position, output_idx].T
            base_ce = pooled_phase_cov(intact_state["calibration"][:, output_idx, common])
            del_ce = pooled_phase_cov(deleted_state["calibration"][:, output_idx, common])
            theory_base = corr_from_cov(base_signal + base_ce)
            theory_deleted = corr_from_cov(del_signal + del_ce)
            phase_base = np.asarray([
                corr(intact_held["features"][phase, output_idx], common)
                for phase in range(N_PHASES)
            ])
            phase_deleted = np.asarray([
                corr(deleted_held["features"][phase, output_idx], common)
                for phase in range(N_PHASES)
            ])
            phase_delta = phase_deleted - phase_base
            if not np.isfinite(np.r_[theory_base, theory_deleted, phase_base, phase_deleted]).all():
                exclusions.append({"network_position": position, "output_area": output_area, "reason": "non-finite correlation after common masking"})
                continue
            rows.append(
                {
                    "deleted_nodes": pattern.deleted_nodes,
                    "n_deleted": int(pattern.n_deleted),
                    "output_area": output_area,
                    "direct": bool(any(_W_MODEL.fln[output_idx, deleted_idx] > 0 for deleted_idx in deleted_indices)),
                    "n_valid": int(common.sum()),
                    "theory_baseline_r": theory_base,
                    "theory_deleted_r": theory_deleted,
                    "theory_delta_r": theory_deleted - theory_base,
                    "simulation_baseline_r": float(phase_base.mean()),
                    "simulation_deleted_r": float(phase_deleted.mean()),
                    "simulation_delta_r": float(phase_delta.mean()),
                    "simulation_baseline_r_phase1": float(phase_base[0]),
                    "simulation_baseline_r_phase2": float(phase_base[1]),
                    "simulation_deleted_r_phase1": float(phase_deleted[0]),
                    "simulation_deleted_r_phase2": float(phase_deleted[1]),
                    "simulation_delta_r_phase1": float(phase_delta[0]),
                    "simulation_delta_r_phase2": float(phase_delta[1]),
                    "Ce_scaling": 1.0,
                }
            )
    frame = pd.DataFrame(rows)
    atomic_csv(OUT / "raw_units.csv.gz", frame, compression="gzip")
    atomic_csv(OUT / "exclusions.csv", pd.DataFrame(exclusions))
    atomic_json(
        OUT / "protocol_manifest.json",
        {
            "protocol": PROTOCOL,
            "only_changed_components": [
                "held-out features extracted separately in each of the two unchanged nonlinear Heun realizations",
                "r and delta-r calculated within each realization and then equally averaged",
                "single-realization calibration covariance Ce added directly to theoretical covariance",
            ],
            "unchanged": [
                "512 input parameter vectors and input time series",
                "60 deletion patterns, 12 at each deletion count 1-5",
                "network model, fixed points, Jacobians, first-order maps",
                "two fixed-Jacobian Heun calibration realizations",
                "complete nonlinear Heun held-out solver and all spectral/feature settings",
                "validity hierarchy, direct definition, statistical unit/test/FDR, and figure format",
            ],
            "Ce_scaling": 1.0,
            "simulation_aggregation": "mean of two realization-specific delta-r values",
            "theory": "four-parameter first-order feature map G Cov(u) G^T plus Ce",
            "raw_units": len(frame),
            "exclusions": len(exclusions),
        },
    )
    draw(frame)


def sign_flip_p(values: np.ndarray, seed: int, permutations: int = 100_000) -> float:
    values = np.asarray(values, float)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    for start in range(0, permutations, 10_000):
        n = min(10_000, permutations - start)
        signs = rng.choice((-1.0, 1.0), size=(n, len(values)))
        extreme += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed))
    return float((extreme + 1) / (permutations + 1))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty_like(q)
    result[order] = np.minimum(q, 1.0)
    return result


def stars(q: float) -> str:
    if q < 1e-4:
        return "****"
    if q < 1e-3:
        return "***"
    if q < 1e-2:
        return "**"
    if q < 0.05:
        return "*"
    return "ns"


def draw(raw: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.linewidth": 1.7,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 9.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    units = []
    for source_name, column in (("Theory", "theory_delta_r"), ("Simulation", "simulation_delta_r")):
        local = raw[["n_deleted", "deleted_nodes", "output_area", "direct", column]].copy()
        local["source"] = source_name
        local["value"] = local[column].abs()
        units.append(local[["n_deleted", "deleted_nodes", "output_area", "direct", "source", "value"]])
    units = pd.concat(units, ignore_index=True)
    zero_count = int((units.value == 0).sum())
    units = units[np.isfinite(units.value) & (units.value > 0)].copy()
    atomic_csv(OUT / "plot_units_absolute_delta_r.csv.gz", units, compression="gzip")
    network_units = units.groupby(["n_deleted", "deleted_nodes", "source", "direct"], as_index=False).value.mean()
    atomic_csv(OUT / "network_level_means.csv", network_units)

    statistics = []
    for count in range(1, 6):
        for source_index, source_name in enumerate(("Theory", "Simulation")):
            local = network_units[(network_units.n_deleted == count) & (network_units.source == source_name)]
            paired = local.pivot(index="deleted_nodes", columns="direct", values="value").dropna()
            difference = paired[True].to_numpy() - paired[False].to_numpy()
            statistics.append(
                {
                    "n_deleted": count,
                    "source": source_name,
                    "n_deletion_networks": len(paired),
                    "direct_mean": float(paired[True].mean()),
                    "non_direct_mean": float(paired[False].mean()),
                    "mean_paired_difference": float(difference.mean()),
                    "sign_flip_p": sign_flip_p(difference, 73421 + 7000 + count * 10 + source_index),
                    "fraction_positive": float(np.mean(difference > 0)),
                }
            )
    stat_frame = pd.DataFrame(statistics)
    stat_frame["sign_flip_fdr_q"] = bh(stat_frame.sign_flip_p.to_numpy())
    atomic_csv(OUT / "significance.csv", stat_frame)
    summary = units.groupby(["n_deleted", "source", "direct"]).value.agg(["count", "mean", "median", "std", "min", "max"]).reset_index()
    atomic_csv(OUT / "group_summary.csv", summary)

    fig, ax = plt.subplots(figsize=(183 / 25.4, 91.5 / 25.4))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84)
    offsets = {
        ("Theory", True): -0.30,
        ("Theory", False): -0.10,
        ("Simulation", True): 0.10,
        ("Simulation", False): 0.30,
    }
    rng = np.random.default_rng(20260910 + 21)
    annotation_max = 0.0
    for count in range(1, 6):
        for source_name in ("Theory", "Simulation"):
            for direct in (True, False):
                values = units.loc[
                    (units.n_deleted == count) & (units.source == source_name) & (units.direct == direct),
                    "value",
                ].to_numpy()
                position = count - 1 + offsets[(source_name, direct)]
                color = COLORS[(source_name, direct)]
                box = ax.boxplot(
                    [values],
                    positions=[position],
                    widths=0.16,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": BLACK, "linewidth": 2.4},
                    whiskerprops={"color": color, "linewidth": 1.7},
                    capprops={"color": color, "linewidth": 1.7},
                )
                box["boxes"][0].set_facecolor(color)
                box["boxes"][0].set_edgecolor(color)
                box["boxes"][0].set_alpha(0.34)
                box["boxes"][0].set_linewidth(1.9)
                ax.scatter(
                    np.full(len(values), position) + rng.uniform(-0.055, 0.055, len(values)),
                    values,
                    s=5.5,
                    color=color,
                    edgecolor="none",
                    alpha=0.28,
                    rasterized=False,
                    zorder=2,
                )
            source_values = units.loc[(units.n_deleted == count) & (units.source == source_name), "value"].to_numpy()
            top = float(source_values.max() * 1.18)
            left = count - 1 + offsets[(source_name, True)]
            right = count - 1 + offsets[(source_name, False)]
            ax.plot([left, left, right, right], [top / 1.13, top, top, top / 1.13], color=BLACK, linewidth=1.35, clip_on=False)
            q = float(stat_frame.loc[(stat_frame.n_deleted == count) & (stat_frame.source == source_name), "sign_flip_fdr_q"].iloc[0])
            ax.text((left + right) / 2, top * 1.08, stars(q), ha="center", va="bottom", fontsize=12, fontweight="bold")
            annotation_max = max(annotation_max, top * 1.42)
    ax.set_yscale("log")
    ax.set_xlim(-0.55, 4.55)
    ax.set_xticks(np.arange(5), ["1", "2", "3", "4", "5"])
    ax.set(xlabel="", ylabel="", title="")
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)
    positive = units.value.to_numpy()
    ax.set_ylim(float(positive.min() * 0.70), annotation_max * 8)
    order = [
        ("Theory", True, "Theory direct"),
        ("Theory", False, "Theory non-direct"),
        ("Simulation", True, "Simulation direct"),
        ("Simulation", False, "Simulation non-direct"),
    ]
    handles = [
        Line2D(
            [0], [0], marker="s", color="none",
            markerfacecolor=COLORS[(source, direct)], markeredgecolor="none",
            alpha=0.72, markersize=8.5, label=label,
        )
        for source, direct, label in order
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, handletextpad=0.45, columnspacing=1.4)
    stem = OUT / "Figure2j_single_realization_Ce"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        strict=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    atomic_json(OUT / "plot_audit.json", {"zero_absolute_effects_omitted_from_log_plot": zero_count})


def qa() -> None:
    stem = OUT / "Figure2j_single_realization_Ce"
    validator = subprocess.run(
        [sys.executable, str(FIGURE_SKILL / "scripts/validate_figure.py"), str(Path(__file__).resolve()), "--json"],
        text=True,
        capture_output=True,
    )
    (OUT / "source-validation.json").write_text(validator.stdout or validator.stderr, encoding="utf-8")
    if validator.returncode != 0:
        raise RuntimeError(f"Source validation failed: {validator.stdout} {validator.stderr}")
    text_audit = subprocess.run(
        [sys.executable, str(FIGURE_SKILL / "scripts/audit_pdf_text.py"), str(stem.with_suffix(".pdf")), "--min-pt", "5", "--json"],
        text=True,
        capture_output=True,
    )
    (OUT / "pdf-text-audit.json").write_text(text_audit.stdout or text_audit.stderr, encoding="utf-8")
    collision = subprocess.run(
        [
            sys.executable,
            str(FIGURE_SKILL / "scripts/audit_figure_collisions.py"),
            str(stem.with_suffix(".pdf")),
            "--json-out", str(OUT / "collision-audit.json"),
            "--overlay-pdf", str(OUT / "collision-audit.pdf"),
        ],
        text=True,
        capture_output=True,
    )
    if text_audit.returncode != 0 or collision.returncode != 0:
        raise RuntimeError(f"QA failed: {text_audit.stdout} {collision.stdout}")
    root = ET.parse(stem.with_suffix(".svg")).getroot()
    images = [node for node in root.iter() if node.tag.endswith("}image")]
    if images:
        raise RuntimeError("SVG unexpectedly embeds raster images")
    atomic_json(OUT / "qa_summary.json", {"status": "PASS", "embedded_raster_images": 0})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--simulate", action="store_true")
    parser.add_argument("--assemble", action="store_true")
    parser.add_argument("--qa", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=61)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.all or args.simulate:
        run_simulation(args.workers, args.start, args.end, args.force)
    if args.all or args.assemble:
        assemble()
    if args.all or args.qa:
        qa()
    if not (args.all or args.simulate or args.assemble or args.qa):
        raise SystemExit("Choose --simulate, --assemble, --qa, or --all")


if __name__ == "__main__":
    main()
