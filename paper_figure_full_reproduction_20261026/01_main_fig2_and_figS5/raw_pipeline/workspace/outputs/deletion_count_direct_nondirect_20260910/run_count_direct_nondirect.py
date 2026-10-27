from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import optimize


HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parents[1]
H_ROOT = TASK_ROOT / "outputs/wholebrain_H_heun_20260908"
H_RUNNER = H_ROOT / "run_H_heun.py"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(TASK_ROOT / "outputs"), str(FIGURE_SKILL / "scripts")]

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


TARGET_PER_CLASS = 12
SELECTION_SEED = 20260910
COLORS = {
    ("Theory", True): "#D1495B",
    ("Theory", False): "#247BA0",
    ("Simulation", True): "#F28E2B",
    ("Simulation", False): "#2A9D8F",
}
BLACK = "#1A1A1A"
STEM = HERE / "direct_nondirect_by_deleted_node_count_exact_heun_2x1"


def load_h_runner():
    spec = importlib.util.spec_from_file_location("wholebrain_h_heun", H_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {H_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


h = load_h_runner()
source = h.source
ORIGINAL_FIXED_POINT = source._fixed_point


def fixed_point_exact_fast(matrix, background, tau, initial, allow_fallback=True):
    """Solve the same ReLU fixed point, avoiding a long ODE fallback when possible."""
    identity = np.eye(len(initial))

    def residual(rate):
        return rate - np.maximum(matrix @ rate + background, 0.0)

    def residual_jac(rate):
        gain = (matrix @ rate + background > 0.0).astype(float)
        return identity - gain[:, None] * matrix

    candidate = optimize.root(
        residual,
        initial,
        jac=residual_jac,
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 2500},
    ).x
    tolerance = 1e-7
    if np.max(np.abs(residual(candidate))) > tolerance or np.min(candidate) < -tolerance:
        seed = np.maximum(candidate, 0.0)
        solved = False
        for _ in range(200):
            drive = matrix @ seed + background
            gain = (drive > 0.0).astype(float)
            system = identity - gain[:, None] * matrix
            rhs = gain * background
            try:
                updated = np.linalg.solve(system, rhs)
            except np.linalg.LinAlgError:
                break
            if np.max(np.abs(residual(updated))) <= tolerance and np.min(updated) >= -tolerance:
                candidate = updated
                solved = True
                break
            seed = 0.5 * np.maximum(seed, 0.0) + 0.5 * np.maximum(updated, 0.0)
        if not solved:
            if allow_fallback:
                candidate, drive, gain, marginal, fixed_residual = ORIGINAL_FIXED_POINT(
                    matrix, background, tau, initial
                )
                return candidate, drive, gain, marginal, fixed_residual
    fixed_residual = float(np.max(np.abs(residual(candidate))))
    candidate[np.abs(candidate) < 1e-10] = 0.0
    drive = matrix @ candidate + background
    gain = (drive > 1e-8).astype(float)
    marginal = np.abs(drive) <= 1e-8
    return candidate, drive, gain, marginal, fixed_residual


source._fixed_point = fixed_point_exact_fast


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temp, index=False)
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_model_and_design():
    cfg = source.load_config(source.PACKAGE_ROOT / "configs/full.yaml")
    model = source.load_model(h.PROJECT_ROOT / cfg["model_dir"])
    with np.load(H_ROOT / "design.npz") as data:
        z = data["z"].astype(float)
        sigma = data["sigma"].astype(float)
    return model, z, sigma


def row_nodes(row: pd.Series) -> tuple[str, ...]:
    return tuple(source._nodes(row))


def is_direct(model, nodes: tuple[str, ...]) -> bool:
    output_idx = model.areas.index(source.OUTPUT_AREA)
    return any(model.fln[output_idx, model.areas.index(node)] > 0 for node in nodes)


def stable_input_output_active(model, nodes: tuple[str, ...]) -> tuple[bool, dict]:
    n = len(model.areas)
    tau = np.r_[
        np.full(n, float(model.params["tau_exc"])),
        np.full(n, float(model.params["tau_inh"])),
    ]
    initial = np.r_[np.full(n, 10.0), np.full(n, 35.0)]
    arrays = source._simulation_arrays(model.params)
    fln = source._deleted_fln(model, list(nodes))
    matrix, background = source._drive_matrix(
        model, fln, arrays, source._edge_arrays
    )
    fixedpoint, _, gain, _, residual = fixed_point_exact_fast(
        matrix, background, tau, initial, allow_fallback=False
    )
    jacobian = source._effective_jacobian(matrix, gain, tau)
    max_real = float(np.max(np.linalg.eigvals(jacobian).real))
    input_idx = model.areas.index(source.INPUT_AREA)
    output_idx = model.areas.index(source.OUTPUT_AREA)
    ok = bool(
        residual < 1e-7
        and gain[input_idx] > 0
        and gain[output_idx] > 0
        and max_real < 0
    )
    return ok, {
        "fixedpoint_residual_screen": float(residual),
        "effective_max_real_eigenvalue_per_ms_screen": max_real,
        "active_E_count_screen": int(gain[:n].sum()),
        "active_I_count_screen": int(gain[n:].sum()),
        "input_active_screen": bool(gain[input_idx] > 0),
        "output_active_screen": bool(gain[output_idx] > 0),
        "fixedpoint_output_rate_screen": float(fixedpoint[output_idx]),
    }


def prepare() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    model, _, _ = load_model_and_design()
    registered = pd.read_csv(H_ROOT / "pattern_results_heun.csv")
    registered["direct"] = [
        is_direct(model, row_nodes(row)) for _, row in registered.iterrows()
    ]
    registered["combo_key"] = [
        "|".join(sorted(row_nodes(row))) for _, row in registered.iterrows()
    ]
    rng = np.random.default_rng(SELECTION_SEED)
    existing_parts = []
    needed: dict[tuple[int, bool], int] = {}
    for count in range(1, 6):
        for direct in (True, False):
            available = registered[
                (registered.n_deleted == count) & (registered.direct == direct)
            ].copy()
            if len(available) > TARGET_PER_CLASS:
                take = np.sort(
                    rng.choice(len(available), size=TARGET_PER_CLASS, replace=False)
                )
                available = available.iloc[take]
            existing_parts.append(available)
            needed[(count, direct)] = TARGET_PER_CLASS - len(available)
    selected_existing = pd.concat(existing_parts, ignore_index=True)
    selected_existing["selection_source"] = "existing exact-Heun result"

    eligible = registered.loc[registered.n_deleted == 1, "deleted_node"].tolist()
    direct_nodes = [node for node in eligible if is_direct(model, (node,))]
    nondirect_nodes = [node for node in eligible if not is_direct(model, (node,))]
    existing_keys = set(registered.combo_key)
    selected_keys = set(selected_existing.combo_key)
    new_rows = []
    new_id = 1000
    for count in range(1, 6):
        for direct in (True, False):
            target = needed[(count, direct)]
            attempts = 0
            while target > 0:
                attempts += 1
                if attempts > 200000:
                    raise RuntimeError(f"Could not find enough valid n={count}, direct={direct} patterns")
                if direct:
                    nodes = tuple(sorted(rng.choice(eligible, size=count, replace=False)))
                    if not any(node in direct_nodes for node in nodes):
                        continue
                else:
                    nodes = tuple(sorted(rng.choice(nondirect_nodes, size=count, replace=False)))
                key = "|".join(nodes)
                if key in existing_keys or key in selected_keys:
                    continue
                ok, screen = stable_input_output_active(model, nodes)
                if not ok:
                    continue
                new_rows.append(
                    {
                        "row_id": new_id,
                        "deleted_node": nodes[0] if count == 1 else np.nan,
                        "deleted_nodes": key,
                        "n_deleted": count,
                        "source": "balanced deletion-count condition",
                        "combo_id": key,
                        "combo_key": key,
                        "direct": direct,
                        "selection_source": "new exact-Heun rerun",
                        **screen,
                    }
                )
                new_id += 1
                selected_keys.add(key)
                target -= 1
                atomic_csv(HERE / "new_patterns.partial.csv", pd.DataFrame(new_rows))
                print(
                    f"accepted n={count}, direct={direct}: "
                    f"{needed[(count, direct)] - target}/{needed[(count, direct)]}",
                    flush=True,
                )

    new_patterns = pd.DataFrame(new_rows)
    atomic_csv(HERE / "selected_existing_results.csv", selected_existing)
    atomic_csv(HERE / "new_patterns.csv", new_patterns)
    selection = pd.concat(
        [
            selected_existing[
                ["row_id", "deleted_node", "deleted_nodes", "n_deleted", "combo_key", "direct", "selection_source"]
            ],
            new_patterns[
                ["row_id", "deleted_node", "deleted_nodes", "n_deleted", "combo_key", "direct", "selection_source"]
            ],
        ],
        ignore_index=True,
    ).sort_values(["n_deleted", "direct", "combo_key"])
    atomic_csv(HERE / "pattern_selection.csv", selection)
    cache = HERE / "nonlinear_states"
    cache.mkdir(exist_ok=True)
    baseline = cache / "network_000.npz"
    if not baseline.exists():
        shutil.copy2(H_ROOT / "nonlinear_states/network_000.npz", baseline)
    counts = (
        selection.groupby(["n_deleted", "direct"]).size().rename("n").reset_index()
    )
    atomic_json(
        HERE / "selection_manifest.json",
        {
            "target_per_class": TARGET_PER_CLASS,
            "selection_seed": SELECTION_SEED,
            "new_pattern_count": int(len(new_patterns)),
            "reused_pattern_count": int(len(selected_existing)),
            "counts": counts.to_dict(orient="records"),
            "direct_definition": (
                "V2 receives a positive FLN edge from at least one deleted node in the intact network"
            ),
            "screen": "stable fixed point with active A6Va input and V2 output",
        },
    )
    print(counts.to_string(index=False), flush=True)
    print(f"new patterns: {len(new_patterns)}", flush=True)


def build_new_context():
    model, z, sigma = load_model_and_design()
    patterns = pd.read_csv(HERE / "new_patterns.csv")
    maps = source._network_maps(model, patterns, z, h.ANALYSIS_S)
    return model, patterns, z, sigma, maps


def worker(start: int, end: int) -> None:
    model, patterns, z, _, maps = build_new_context()
    source._network_run_fixedpoint = h.network_run_fixedpoint_heun
    indices = list(range(max(1, start), min(len(patterns) + 1, end)))
    source._nonlinear_heldout_features(
        model,
        z,
        maps,
        h.ANALYSIS_S,
        h.N_HELDOUT,
        HERE / "nonlinear_states",
        fixedpoint_init=True,
        network_indices=indices,
    )
    atomic_json(
        HERE / f"worker_{indices[0]:03d}_{indices[-1] + 1:03d}.json",
        {
            "status": "complete",
            "network_start": indices[0],
            "network_end_exclusive": indices[-1] + 1,
            "integrator": "explicit Heun",
        },
    )


def assemble() -> None:
    model, patterns, z, sigma, maps = build_new_context()
    maps["transfers"] = h.heun_transfers(model, maps)
    linear, linear_valid = source._linear_calibration_features(
        z,
        maps,
        h.ANALYSIS_S,
        h.ANALYSIS_S,
        HERE / "linear_calibration_features_heun.npz",
    )
    source._network_run_fixedpoint = h.network_run_fixedpoint_heun
    nonlinear, nonlinear_valid, rectification = source._nonlinear_heldout_features(
        model,
        z,
        maps,
        h.ANALYSIS_S,
        h.N_HELDOUT,
        HERE / "nonlinear_states",
        fixedpoint_init=True,
    )
    new_result = source._assemble(
        patterns,
        maps,
        linear,
        linear_valid,
        nonlinear,
        nonlinear_valid,
        rectification,
        h.N_HELDOUT,
        h.N_TRIALS,
        h.ANALYSIS_S,
    )
    new_result["baseline_rectification_fraction"] /= 2.0
    new_result["deleted_rectification_fraction"] /= 2.0
    new_result.insert(0, "integrator", "explicit Heun")
    new_result.insert(1, "covariance_tag", "cond_100")
    atomic_csv(HERE / "new_pattern_results_heun.csv", new_result)

    existing = pd.read_csv(HERE / "selected_existing_results.csv")
    common_columns = sorted(set(existing.columns) | set(new_result.columns))
    combined = pd.concat(
        [existing.reindex(columns=common_columns), new_result.reindex(columns=common_columns)],
        ignore_index=True,
    )
    combined["direct"] = combined["direct"].astype(bool)
    combined = combined.sort_values(["n_deleted", "direct", "combo_key"]).reset_index(drop=True)
    atomic_csv(HERE / "combined_exact_heun_results.csv", combined)
    plot(combined)
    atomic_json(
        HERE / "manifest.json",
        {
            "status": "complete",
            "system": "complete 55-area x E/I (110-state) threshold-linear/ReLU network",
            "integrator": "explicit Heun",
            "protocol": {
                "trials": h.N_TRIALS,
                "analysis_s": h.ANALYSIS_S,
                "burn_s": source.BURN_S,
                "dt_ms": source.DT_MS,
                "calibration_phases": source.CALIBRATION_PHASES,
                "heldout_phases": h.N_HELDOUT,
                "covariance_condition": h.TARGET_CONDITION,
                "fixedpoint_initialized": True,
                "input_area": source.INPUT_AREA,
                "output_area": source.OUTPUT_AREA,
            },
            "groups": "1 through 5 deleted nodes; 12 direct and 12 non-direct networks per count",
            "direct_definition": "V2 receives from at least one deleted node in the intact FLN matrix",
            "quantity_plotted": "absolute deletion-induced correlation change |delta r|",
            "source_fingerprints": {
                "runner": sha256(Path(__file__).resolve()),
                "wholebrain_H_runner": sha256(H_RUNNER),
                "wholebrain_H_results": sha256(H_ROOT / "pattern_results_heun.csv"),
            },
            "software": {
                "python": sys.version,
                "numpy": np.__version__,
                "matplotlib": mpl.__version__,
                "platform": platform.platform(),
            },
        },
    )


def plot(frame: pd.DataFrame) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 12,
            "axes.linewidth": 1.7,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11.5,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    long_rows = []
    for source_name, column in (
        ("Theory", "theory_delta_r"),
        ("Simulation", "simulation_delta_r"),
    ):
        local = frame[["n_deleted", "direct", "combo_key", column]].copy()
        local["source"] = source_name
        local["value"] = local[column].abs()
        long_rows.append(local[["n_deleted", "direct", "combo_key", "source", "value"]])
    units = pd.concat(long_rows, ignore_index=True)
    units = units[np.isfinite(units.value) & (units.value > 0)].copy()
    atomic_csv(HERE / "plot_units_absolute_delta_r.csv", units)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.16, top=0.83)
    offsets = {
        ("Theory", True): -0.30,
        ("Theory", False): -0.10,
        ("Simulation", True): 0.10,
        ("Simulation", False): 0.30,
    }
    rng = np.random.default_rng(SELECTION_SEED + 1)
    for count in range(1, 6):
        for source_name in ("Theory", "Simulation"):
            for direct in (True, False):
                values = units.loc[
                    (units.n_deleted == count)
                    & (units.source == source_name)
                    & (units.direct == direct),
                    "value",
                ].to_numpy(float)
                position = count - 1 + offsets[(source_name, direct)]
                color = COLORS[(source_name, direct)]
                box = ax.boxplot(
                    [values],
                    positions=[position],
                    widths=0.16,
                    patch_artist=True,
                    showfliers=False,
                    medianprops={"color": BLACK, "linewidth": 2.4},
                    whiskerprops={"color": color, "linewidth": 1.8},
                    capprops={"color": color, "linewidth": 1.8},
                )
                box["boxes"][0].set_facecolor(color)
                box["boxes"][0].set_edgecolor(color)
                box["boxes"][0].set_alpha(0.34)
                box["boxes"][0].set_linewidth(2.0)
                jitter = rng.uniform(-0.045, 0.045, size=len(values))
                ax.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=18,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.78,
                    zorder=3,
                )
    ax.set_yscale("log")
    ax.set_xlim(-0.55, 4.55)
    ax.set_xticks(np.arange(5), ["1", "2", "3", "4", "5"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)
    positive = units.value.to_numpy(float)
    ax.set_ylim(float(positive.min() * 0.65), float(positive.max() * 1.55))
    legend_order = [
        ("Theory", True, "Theory direct"),
        ("Theory", False, "Theory non-direct"),
        ("Simulation", True, "Simulation direct"),
        ("Simulation", False, "Simulation non-direct"),
    ]
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            color="none",
            markerfacecolor=COLORS[(source_name, direct)],
            markeredgecolor="none",
            alpha=0.72,
            markersize=10,
            label=label,
        )
        for source_name, direct, label in legend_order
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=STEM.with_suffix(".alignment.json"),
        overlay_svg=STEM.with_suffix(".alignment.svg"),
        strict=True,
    )
    fig.savefig(STEM.with_suffix(".svg"), facecolor="white")
    fig.savefig(STEM.with_suffix(".pdf"), facecolor="white")
    fig.savefig(STEM.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(STEM.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)

    summary = (
        units.groupby(["n_deleted", "source", "direct"])
        .value.agg(["count", "mean", "median", "std", "min", "max"])
        .reset_index()
    )
    atomic_csv(HERE / "group_summary.csv", summary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--worker-start", type=int)
    parser.add_argument("--worker-end", type=int)
    parser.add_argument("--assemble", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.worker_start is not None and args.worker_end is not None:
        worker(args.worker_start, args.worker_end)
    elif args.assemble:
        assemble()
    else:
        raise SystemExit("Choose --prepare, --worker-start/--worker-end, or --assemble")


if __name__ == "__main__":
    main()
