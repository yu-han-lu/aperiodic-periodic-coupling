from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parents[1]
PROJECT_ROOT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
CONDITIONING_SCRIPT = (
    TASK_ROOT / "outputs/covariance_conditioning_pilot/run_conditioning_experiment.py"
)
COMMON_ROOT = TASK_ROOT / "outputs"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(COMMON_ROOT), str(FIGURE_SKILL / "scripts")]

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from heun_network_common import (  # noqa: E402
    heun_discrete_transfer_all,
    threshold_network_heun_batch_fixedpoint,
)


def load_conditioning_module():
    spec = importlib.util.spec_from_file_location(
        "conditioning_euler_source", CONDITIONING_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {CONDITIONING_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


source = load_conditioning_module()

TARGET_CONDITION = 100.0
N_TRIALS = source.LARGE_TRIALS
ANALYSIS_S = source.LARGE_ANALYSIS_S
N_HELDOUT = source.LARGE_HELDOUT_PHASES
OUTPUT_STEM = HERE / "wholebrain_H_fixedpoint_kappa100_Heun_5x4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def build_context() -> tuple[object, pd.DataFrame, np.ndarray, dict, dict]:
    selected = json.loads(source.SELECTED_JSON.read_text())
    original_sigma = np.asarray(selected["sigma"], float)
    sigma, covariance_info = source._condition_covariance(
        original_sigma, TARGET_CONDITION
    )
    cfg = source.load_config(source.PACKAGE_ROOT / "configs/full.yaml")
    model = source.load_model(PROJECT_ROOT / cfg["model_dir"])
    patterns = source._load_source("fpinit_large")
    if len(patterns) != 82:
        raise RuntimeError(f"Expected 82 registered deletion patterns, found {len(patterns)}")
    z, u, clipped = source._design(sigma, N_TRIALS, source.SEED + 990000)
    maps = source._network_maps(model, patterns, z, ANALYSIS_S)
    if "jacobians" not in maps or len(maps["jacobians"]) != len(patterns) + 1:
        raise RuntimeError("The fixed-point map did not expose every Jacobian")
    design_info = {
        "sigma": sigma,
        "u": u,
        "clipped_parameter_fraction": clipped,
        **covariance_info,
    }
    return model, patterns, z, maps, design_info


def network_run_fixedpoint_heun(
    model,
    inputs: np.ndarray,
    output_indices: np.ndarray,
    input_idx: int,
    fixedpoint: np.ndarray,
    fln: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    arrays = source._simulation_arrays(model.params)
    edge_target, edge_source, edge_e, edge_i = source._edge_arrays(
        model.params, model.fln if fln is None else fln
    )
    n_area = len(model.areas)
    paths, rect1, rect2 = threshold_network_heun_batch_fixedpoint(
        inputs,
        arrays[0], arrays[1], arrays[2], arrays[3],
        edge_target, edge_source, edge_e, edge_i,
        arrays[8], arrays[9],
        fixedpoint[:n_area], fixedpoint[n_area:],
        source.DT_MS,
        float(model.params["tau_exc"]),
        float(model.params["tau_inh"]),
        int(round(source.BURN_S * 1000.0 / source.DT_MS)),
        output_indices,
        input_idx,
    )
    return paths, rect1 + rect2


def nonlinear_cache(
    start: int | None = None, end: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    model, _, z, maps, _ = build_context()
    source._network_run_fixedpoint = network_run_fixedpoint_heun
    indices = None
    if start is not None or end is not None:
        first = 0 if start is None else max(0, start)
        last = len(maps["flns"]) if end is None else min(len(maps["flns"]), end)
        indices = list(range(first, last))
    result = source._nonlinear_heldout_features(
        model,
        z,
        maps,
        ANALYSIS_S,
        N_HELDOUT,
        HERE / "nonlinear_states",
        fixedpoint_init=True,
        network_indices=indices,
    )
    if indices is not None:
        atomic_json(
            HERE / f"worker_{indices[0]:03d}_{indices[-1] + 1:03d}.json",
            {
                "status": "complete",
                "network_start": indices[0],
                "network_end_exclusive": indices[-1] + 1,
                "integrator": "explicit Heun",
            },
        )
        return None
    return result


def heun_transfers(model, maps: dict) -> np.ndarray:
    n_steps = int(round(ANALYSIS_S * 1000.0 / source.DT_MS))
    input_idx = model.areas.index(source.INPUT_AREA)
    output_idx = model.areas.index(source.OUTPUT_AREA)
    values = []
    for pos, jacobian in enumerate(maps["jacobians"]):
        values.append(
            heun_discrete_transfer_all(
                jacobian,
                input_idx,
                np.asarray([output_idx], np.int64),
                n_steps,
                float(model.params["tau_exc"]),
                source.DT_MS,
            )[0]
        )
        print(f"Heun linear transfer {pos + 1}/{len(maps['jacobians'])}", flush=True)
    return np.asarray(values, dtype=np.complex64)


def metrics(frame: pd.DataFrame) -> dict:
    use = np.isfinite(frame.theory_delta_r) & np.isfinite(frame.simulation_delta_r)
    x = frame.loc[use, "theory_delta_r"].to_numpy(float)
    y = frame.loc[use, "simulation_delta_r"].to_numpy(float)
    fit = stats.linregress(x, y)
    difference = y - x
    return {
        "n": int(len(x)),
        "pearson_r": float(fit.rvalue),
        "ols_slope": float(fit.slope),
        "ols_intercept": float(fit.intercept),
        "ols_R2": float(fit.rvalue**2),
        "identity_R2": float(
            1.0
            - np.sum(difference**2)
            / np.sum((y - np.mean(y)) ** 2)
        ),
        "mae": float(np.mean(np.abs(difference))),
        "rmse": float(np.sqrt(np.mean(difference**2))),
        "mean_bias": float(np.mean(difference)),
    }


def draw(frame: pd.DataFrame, fit: dict) -> None:
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
    x = frame.theory_delta_r.to_numpy(float)
    y = frame.simulation_delta_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y)
    low = float(min(x[use].min(), y[use].min()))
    high = float(max(x[use].max(), y[use].max()))
    pad = 0.04 * max(high - low, 0.1)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=0.12, right=0.97, bottom=0.12, top=0.96)
    colors = {1: "#247BA0", 3: "#F4A261", 4: "#2A9D8F", 5: "#8E5EA2"}
    markers = {1: "o", 3: "^", 4: "s", 5: "D"}
    for count in sorted(frame.n_deleted.unique()):
        take = use & (frame.n_deleted.to_numpy(int) == int(count))
        ax.scatter(
            x[take], y[take], s=28, marker=markers[int(count)],
            color=colors[int(count)], alpha=0.85, label=f"{int(count)} nodes",
        )
    ax.plot(
        [low - pad, high + pad], [low - pad, high + pad],
        color="#555555", linestyle="--", linewidth=1.7,
    )
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.05, 0.94,
        f"r = {fit['pearson_r']:.2f}\nslope = {fit['ols_slope']:.2f}",
        transform=ax.transAxes, ha="left", va="top",
        fontsize=18, fontweight="bold",
    )
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=OUTPUT_STEM.with_suffix(".alignment.json"),
        overlay_svg=OUTPUT_STEM.with_suffix(".alignment.svg"),
        strict=True,
    )
    fig.savefig(OUTPUT_STEM.with_suffix(".svg"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def assemble() -> None:
    started = time.time()
    model, patterns, z, maps, design_info = build_context()
    maps["transfers"] = heun_transfers(model, maps)
    linear, linear_valid = source._linear_calibration_features(
        z,
        maps,
        ANALYSIS_S,
        HERE / "linear_calibration_features_heun.npz",
    )
    source._network_run_fixedpoint = network_run_fixedpoint_heun
    nonlinear, nonlinear_valid, rectification = source._nonlinear_heldout_features(
        model,
        z,
        maps,
        ANALYSIS_S,
        N_HELDOUT,
        HERE / "nonlinear_states",
        fixedpoint_init=True,
    )
    result = source._assemble(
        patterns,
        maps,
        linear,
        linear_valid,
        nonlinear,
        nonlinear_valid,
        rectification,
        N_HELDOUT,
        N_TRIALS,
        ANALYSIS_S,
    )
    # The shared assembler normalizes a one-stage Euler count.  Heun evaluates
    # the ReLU twice, so divide the reported fractions by two.
    result["baseline_rectification_fraction"] /= 2.0
    result["deleted_rectification_fraction"] /= 2.0
    result.insert(0, "integrator", "explicit Heun")
    result.insert(1, "covariance_tag", "cond_100")
    atomic_csv(HERE / "pattern_results_heun.csv", result)
    fit = metrics(result)
    fit.update(
        {
            "integrator": "complete 55-area x E/I threshold-linear/ReLU explicit Heun",
            "dt_ms": source.DT_MS,
            "n_trials": N_TRIALS,
            "analysis_s": ANALYSIS_S,
            "calibration_phases": source.CALIBRATION_PHASES,
            "heldout_phases": N_HELDOUT,
            "fixedpoint_initialized": True,
            "covariance_condition": TARGET_CONDITION,
            "pattern_count": int(len(patterns)),
            "elapsed_assembly_s": time.time() - started,
        }
    )
    atomic_json(HERE / "metrics.json", fit)
    np.savez_compressed(
        HERE / "design.npz",
        sigma=design_info["sigma"],
        z=z,
        u=design_info["u"],
    )
    draw(result, fit)
    manifest = {
        "status": "complete",
        "system": "complete 55-area x E/I (110-state) threshold-linear/ReLU network",
        "integrator": {
            "name": "explicit Heun",
            "stage_1": "k1=(-r+ReLU(Mr+h+bu[n]))/tau",
            "predictor": "p=r+dt*k1",
            "stage_2": "k2=(-p+ReLU(Mp+h+bu[n]))/tau",
            "corrector": "r_next=r+dt*(k1+k2)/2",
            "same_forcing_sample_in_both_stages": True,
            "all_110_relu_drives_recomputed_at_each_stage": True,
        },
        "protocol": {
            "patterns": int(len(patterns)),
            "trials": N_TRIALS,
            "analysis_s": ANALYSIS_S,
            "burn_s": source.BURN_S,
            "dt_ms": source.DT_MS,
            "calibration_phases": source.CALIBRATION_PHASES,
            "heldout_phases": N_HELDOUT,
            "covariance_condition": TARGET_CONDITION,
            "fixedpoint_initialized": True,
            "input_area": source.INPUT_AREA,
            "output_area": source.OUTPUT_AREA,
        },
        "source_fingerprints": {
            "runner": sha256(Path(__file__).resolve()),
            "common_heun": sha256(TASK_ROOT / "outputs/heun_network_common.py"),
            "original_protocol": sha256(CONDITIONING_SCRIPT),
            "registered_patterns": sha256(
                TASK_ROOT
                / "outputs/covariance_conditioning_pilot/fpinit_large/cond_100/pattern_results.csv"
            ),
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "matplotlib": mpl.__version__,
            "platform": platform.platform(),
        },
    }
    atomic_json(HERE / "manifest.json", manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-start", type=int)
    parser.add_argument("--worker-end", type=int)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    if args.worker_start is not None or args.worker_end is not None:
        nonlinear_cache(args.worker_start, args.worker_end)
    elif args.assemble:
        assemble()
    else:
        nonlinear_cache()
        assemble()


if __name__ == "__main__":
    main()
