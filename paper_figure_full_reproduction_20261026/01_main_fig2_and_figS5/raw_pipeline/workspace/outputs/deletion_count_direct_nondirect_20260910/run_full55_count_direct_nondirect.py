from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from numba import set_num_threads


HERE = Path(__file__).resolve().parent
TASK_ROOT = HERE.parents[1]
LATEST_ROOT = TASK_ROOT / "outputs/deletion_bands_robustness_512_heun_20260908"
BAND_RUNNER = TASK_ROOT / "outputs/deletion_bands_robustness_20260908/run_larger_sample_bands.py"
SELECTOR_RUNNER = HERE / "run_count_direct_nondirect.py"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(TASK_ROOT / "outputs"), str(FIGURE_SKILL / "scripts")]

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


band = load_module("band_runner", BAND_RUNNER)
selector = load_module("deletion_selector", SELECTOR_RUNNER)
base = band.base
base._fixed_point = selector.fixed_point_exact_fast


N_TRIALS = 512
N_PER_COUNT = 12
SEED = 20260910
CENTER_HZ = 10.0
PEAK_BAND = (5.0, 20.0)
ANALYSIS_S = base.BAND_ANALYSIS_S
CAL_PHASES = base.BAND_CAL_PHASES
HELD_PHASES = base.BAND_HELD_PHASES
STATE_ROOT = HERE / "full55_states_512"
STEM = HERE / "direct_nondirect_by_deleted_node_count_full55_exact_heun_512trials_2x1"
COLORS = {
    ("Theory", True): "#D1495B",
    ("Theory", False): "#247BA0",
    ("Simulation", True): "#F28E2B",
    ("Simulation", False): "#2A9D8F",
}
BLACK = "#1A1A1A"


def atomic_csv(path: Path, frame: pd.DataFrame, compression=None) -> None:
    suffix = ".tmp.gz" if compression == "gzip" else ".tmp"
    temp = path.with_name(path.name + f".{os.getpid()}{suffix}")
    frame.to_csv(temp, index=False, compression=compression)
    temp.replace(path)


def atomic_json(path: Path, payload: dict) -> None:
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)


def model_and_design():
    model = base.load_wholebrain()
    with np.load(LATEST_ROOT / "design_512.npz") as saved:
        z = saved["z"].astype(float)
    if len(z) != N_TRIALS:
        raise RuntimeError(f"Expected {N_TRIALS} registered inputs, found {len(z)}")
    z[:, 0] += CENTER_HZ - 10.0
    return model, z


def select_patterns() -> pd.DataFrame:
    all_patterns = pd.read_csv(HERE / "pattern_selection.csv")
    rng = np.random.default_rng(SEED + 20)
    selected = []
    for count in range(1, 6):
        available = all_patterns[all_patterns.n_deleted == count].copy()
        positions = np.sort(rng.choice(len(available), size=N_PER_COUNT, replace=False))
        selected.append(available.iloc[positions])
    result = pd.concat(selected, ignore_index=True)
    result = result.sort_values(["n_deleted", "combo_key"]).reset_index(drop=True)
    return result


def load_state(path: Path) -> dict:
    with np.load(path) as saved:
        return {
            "theory": saved["theory"].astype(float),
            "theory_valid": saved["theory_valid"].astype(bool),
            "calibration": saved["calibration"].astype(float),
            "calibration_valid": saved["calibration_valid"].astype(bool),
            "simulation": saved["simulation"].astype(float),
            "simulation_valid": saved["simulation_valid"].astype(bool),
            "rectification": int(saved["rectification"]),
        }


def prepare() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    STATE_ROOT.mkdir(exist_ok=True)
    selected = select_patterns()
    atomic_csv(HERE / "full55_pattern_selection.csv", selected)
    registered_path = (
        TASK_ROOT
        / "outputs/covariance_conditioning_pilot/fpinit_large/cond_100/pattern_results.csv"
    )
    registered_states = LATEST_ROOT / "band_alpha_states_512"
    reuse = []
    if registered_path.exists() and registered_states.exists():
        registered = pd.read_csv(registered_path)
        single = registered[registered.n_deleted == 1].sort_values("row_id").reset_index(drop=True)
        row_to_position = {int(row_id): pos + 1 for pos, row_id in enumerate(single.row_id)}
        reuse = [(0, 0)]
        for new_position, item in enumerate(selected.itertuples(index=False), 1):
            if int(item.n_deleted) == 1:
                reuse.append((new_position, row_to_position[int(item.row_id)]))
    for destination_position, registered_position in reuse:
        old = load_state(
            registered_states / f"network_{registered_position:03d}.npz"
        )
        if old["theory"].shape != (55, N_TRIALS, 2):
            raise RuntimeError("Registered full-output cache has an unexpected shape")
        base.atomic_npz(
            STATE_ROOT / f"network_{destination_position:03d}.npz",
            **old,
        )
    atomic_json(
        HERE / "full55_prepare_manifest.json",
        {
            "patterns_per_deleted_count": N_PER_COUNT,
            "total_patterns": int(len(selected)),
            "reused_exact_Heun_states": int(len(reuse)),
            "states_to_simulate": int(len(selected) + 1 - len(reuse)),
            "trials": N_TRIALS,
            "analysis_s": ANALYSIS_S,
            "calibration_phases": CAL_PHASES,
            "heldout_phases": HELD_PHASES,
            "peak_center_hz": CENTER_HZ,
            "peak_fit_band_hz": PEAK_BAND,
        },
    )
    print(selected.groupby("n_deleted").size().to_string(), flush=True)
    print(f"reused {len(reuse)} full-output states", flush=True)


def build_context():
    model, z = model_and_design()
    selected = pd.read_csv(HERE / "full55_pattern_selection.csv")
    networks = base.deletion_networks(model, selected)
    input_idx = model.areas.index("A6Va")
    output_indices = np.arange(len(model.areas), dtype=np.int64)
    frequencies, _ = base.frequencies_for(ANALYSIS_S)
    cal_inputs = [
        base._matched_bandlimited_inputs(
            z,
            base.DT_MS,
            ANALYSIS_S,
            base.BURN_S,
            base.INPUT_SIGMA_DELETION,
            base.SEED + 1010000 + phase * 100000,
        )
        for phase in range(CAL_PHASES)
    ]
    held_inputs = [
        base._matched_bandlimited_inputs(
            z,
            base.DT_MS,
            ANALYSIS_S,
            base.BURN_S,
            base.INPUT_SIGMA_DELETION,
            base.SEED + 1010000 + (CAL_PHASES + phase) * 100000,
        )
        for phase in range(HELD_PHASES)
    ]
    return model, selected, z, networks, input_idx, output_indices, frequencies, cal_inputs, held_inputs


def run_worker(start: int, end: int) -> None:
    set_num_threads(int(os.environ.get("COUNT_NUMBA_THREADS", "3")))
    (
        model,
        selected,
        z,
        networks,
        input_idx,
        output_indices,
        frequencies,
        cal_inputs,
        held_inputs,
    ) = build_context()
    started = time.time()
    for network_pos in range(max(0, start), min(len(networks), end)):
        destination = STATE_ROOT / f"network_{network_pos:03d}.npz"
        if destination.exists():
            continue
        fln, fixedpoint, _, jacobian, _ = networks[network_pos]
        theory, theory_valid = base.theoretical_features_all(
            jacobian, z, input_idx, output_indices, frequencies, PEAK_BAND
        )
        calibration = np.full((CAL_PHASES, 55, N_TRIALS, 2), np.nan)
        calibration_valid = np.zeros((CAL_PHASES, 55, N_TRIALS), bool)
        for phase in range(CAL_PHASES):
            calibration[phase], calibration_valid[phase] = band.heun_linear_phase_features(
                cal_inputs[phase],
                jacobian,
                input_idx,
                output_indices,
                ANALYSIS_S,
                PEAK_BAND,
                float(model.params["tau_exc"]),
            )
        simulation, simulation_valid, rectification = band.heun_nonlinear_average_features(
            model,
            held_inputs,
            fln,
            fixedpoint,
            input_idx,
            output_indices,
            ANALYSIS_S,
            PEAK_BAND,
        )
        base.atomic_npz(
            destination,
            theory=theory,
            theory_valid=theory_valid,
            calibration=calibration,
            calibration_valid=calibration_valid,
            simulation=simulation,
            simulation_valid=simulation_valid,
            rectification=np.asarray(rectification),
        )
        print(
            f"network {network_pos + 1}/{len(networks)} complete; "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    atomic_json(
        HERE / f"full55_worker_{start:03d}_{end:03d}.json",
        {
            "status": "complete",
            "network_start": start,
            "network_end_exclusive": end,
            "integrator": "explicit Heun",
        },
    )


def assemble() -> None:
    model, selected, _, networks, _, _, _, _, _ = build_context()
    states = []
    for network_pos in range(len(networks)):
        path = STATE_ROOT / f"network_{network_pos:03d}.npz"
        if not path.exists():
            raise RuntimeError(f"Missing {path}")
        states.append(load_state(path))
    intact = states[0]
    rows = []
    for position, item in enumerate(selected.itertuples(index=False), 1):
        deleted = states[position]
        deleted_nodes = str(item.deleted_nodes).split("|")
        deleted_indices = [model.areas.index(node) for node in deleted_nodes]
        gain = networks[position][2]
        for output_idx, output_area in enumerate(model.areas):
            if output_idx in deleted_indices or not bool(gain[output_idx]):
                continue
            common = (
                intact["theory_valid"][output_idx]
                & deleted["theory_valid"][output_idx]
                & intact["simulation_valid"][output_idx]
                & deleted["simulation_valid"][output_idx]
                & intact["calibration_valid"][:, output_idx].all(axis=0)
                & deleted["calibration_valid"][:, output_idx].all(axis=0)
            )
            if common.sum() < 8:
                continue
            base_signal = np.cov(intact["theory"][output_idx, common], rowvar=False, ddof=1)
            del_signal = np.cov(deleted["theory"][output_idx, common], rowvar=False, ddof=1)
            base_residual = base.pooled_phase_cov(
                intact["calibration"][:, output_idx, common]
            )
            del_residual = base.pooled_phase_cov(
                deleted["calibration"][:, output_idx, common]
            )
            theory_baseline = base.corr_from_cov(base_signal + base_residual / HELD_PHASES)
            theory_deleted = base.corr_from_cov(del_signal + del_residual / HELD_PHASES)
            simulation_baseline = base.corr_feature(intact["simulation"][output_idx], common)
            simulation_deleted = base.corr_feature(deleted["simulation"][output_idx], common)
            rows.append(
                {
                    "deleted_nodes": item.deleted_nodes,
                    "n_deleted": int(item.n_deleted),
                    "output_area": output_area,
                    "direct": bool(
                        any(model.fln[output_idx, deleted_idx] > 0 for deleted_idx in deleted_indices)
                    ),
                    "n_valid": int(common.sum()),
                    "theory_baseline_r": theory_baseline,
                    "theory_deleted_r": theory_deleted,
                    "theory_delta_r": theory_deleted - theory_baseline,
                    "simulation_baseline_r": simulation_baseline,
                    "simulation_deleted_r": simulation_deleted,
                    "simulation_delta_r": simulation_deleted - simulation_baseline,
                }
            )
    raw = pd.DataFrame(rows)
    atomic_csv(HERE / "full55_raw_units_512.csv.gz", raw, compression="gzip")
    plot(raw)
    atomic_json(
        HERE / "full55_manifest.json",
        {
            "status": "complete",
            "system": "complete 55-area x E/I (110-state) threshold-linear/ReLU network",
            "integrator": "explicit Heun",
            "trials": N_TRIALS,
            "analysis_s": ANALYSIS_S,
            "burn_s": base.BURN_S,
            "dt_ms": base.DT_MS,
            "calibration_phases": CAL_PHASES,
            "heldout_phases": HELD_PHASES,
            "input_area": "A6Va",
            "outputs": "all non-deleted, active areas with valid paired features",
            "input_peak_center_hz": CENTER_HZ,
            "peak_fit_band_hz": PEAK_BAND,
            "direct_definition": (
                "the output receives a positive intact-network FLN edge from at least one deleted node"
            ),
            "quantity_plotted": "absolute deletion-induced correlation change |delta r|",
            "statistics": {
                "unit": "deletion network",
                "within_network_summary": "mean |delta r| across direct or non-direct outputs",
                "comparison": "paired two-sided sign-flip test of direct minus non-direct means",
                "permutations": 100000,
                "multiple_comparison_correction": "Benjamini-Hochberg FDR across 5 deletion counts x 2 sources",
                "display": "ns, *, **, ***, or **** from FDR-adjusted q values",
            },
            "patterns_per_deleted_count": N_PER_COUNT,
            "raw_output_units": int(len(raw)),
            "software": {
                "python": sys.version,
                "numpy": np.__version__,
                "matplotlib": mpl.__version__,
                "platform": platform.platform(),
            },
        },
    )


def plot(raw: pd.DataFrame) -> None:
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
    for source_name, column in (
        ("Theory", "theory_delta_r"),
        ("Simulation", "simulation_delta_r"),
    ):
        local = raw[["n_deleted", "deleted_nodes", "output_area", "direct", column]].copy()
        local["source"] = source_name
        local["value"] = local[column].abs()
        units.append(local[["n_deleted", "deleted_nodes", "output_area", "direct", "source", "value"]])
    units = pd.concat(units, ignore_index=True)
    units = units[np.isfinite(units.value) & (units.value > 0)].copy()
    atomic_csv(HERE / "full55_plot_units_absolute_delta_r_512.csv.gz", units, compression="gzip")
    network_units = (
        units.groupby(["n_deleted", "deleted_nodes", "source", "direct"], as_index=False)
        .value.mean()
    )
    atomic_csv(HERE / "full55_network_level_means_512.csv", network_units)

    statistics = []
    for count in range(1, 6):
        for source_index, source_name in enumerate(("Theory", "Simulation")):
            local = network_units[
                (network_units.n_deleted == count)
                & (network_units.source == source_name)
            ]
            paired = local.pivot(
                index="deleted_nodes", columns="direct", values="value"
            ).dropna()
            difference = paired[True].to_numpy(float) - paired[False].to_numpy(float)
            statistics.append(
                {
                    "n_deleted": count,
                    "source": source_name,
                    "n_deletion_networks": int(len(paired)),
                    "direct_mean": float(paired[True].mean()),
                    "non_direct_mean": float(paired[False].mean()),
                    "mean_paired_difference": float(difference.mean()),
                    "sign_flip_p": base.sign_flip_p(
                        difference,
                        base.SEED + 7000 + count * 10 + source_index,
                    ),
                    "fraction_positive": float(np.mean(difference > 0)),
                }
            )
    stat_frame = pd.DataFrame(statistics)
    stat_frame["sign_flip_fdr_q"] = base.bh(stat_frame.sign_flip_p.to_numpy())
    atomic_csv(HERE / "full55_direct_nondirect_significance_512.csv", stat_frame)

    # Nature-compatible double-column width while preserving the requested 2:1 ratio.
    fig, ax = plt.subplots(figsize=(183.0 / 25.4, 91.5 / 25.4))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84)
    offsets = {
        ("Theory", True): -0.30,
        ("Theory", False): -0.10,
        ("Simulation", True): 0.10,
        ("Simulation", False): 0.30,
    }
    rng = np.random.default_rng(SEED + 21)
    annotation_max = 0.0
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
                    whiskerprops={"color": color, "linewidth": 1.7},
                    capprops={"color": color, "linewidth": 1.7},
                )
                box["boxes"][0].set_facecolor(color)
                box["boxes"][0].set_edgecolor(color)
                box["boxes"][0].set_alpha(0.34)
                box["boxes"][0].set_linewidth(1.9)
                jitter = rng.uniform(-0.055, 0.055, size=len(values))
                ax.scatter(
                    np.full(len(values), position) + jitter,
                    values,
                    s=5.5,
                    color=color,
                    edgecolor="none",
                    alpha=0.28,
                    rasterized=False,
                    zorder=2,
                )
            source_values = units.loc[
                (units.n_deleted == count) & (units.source == source_name), "value"
            ].to_numpy(float)
            top = float(source_values.max() * 1.18)
            left = count - 1 + offsets[(source_name, True)]
            right = count - 1 + offsets[(source_name, False)]
            ax.plot(
                [left, left, right, right],
                [top / 1.13, top, top, top / 1.13],
                color=BLACK,
                linewidth=1.35,
                clip_on=False,
            )
            q_value = float(
                stat_frame.loc[
                    (stat_frame.n_deleted == count)
                    & (stat_frame.source == source_name),
                    "sign_flip_fdr_q",
                ].iloc[0]
            )
            ax.text(
                (left + right) / 2.0,
                top * 1.08,
                base.stars(q_value),
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
            )
            annotation_max = max(annotation_max, top * 1.42)
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
    # A full log-decade of headroom keeps data-derived brackets clear of the top spine.
    ax.set_ylim(float(positive.min() * 0.70), annotation_max * 8.0)
    legend_order = [
        ("Theory", True, "Theory direct"),
        ("Theory", False, "Theory non-direct"),
        ("Simulation", True, "Simulation direct"),
        ("Simulation", False, "Simulation non-direct"),
    ]
    handles = [
        Line2D(
            [0], [0], marker="s", color="none",
            markerfacecolor=COLORS[(source_name, direct)], markeredgecolor="none",
            alpha=0.72, markersize=8.5, label=label,
        )
        for source_name, direct, label in legend_order
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="upper center",
        ncol=4,
        handletextpad=0.45,
        columnspacing=1.4,
    )
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
    atomic_csv(HERE / "full55_group_summary_512.csv", summary)


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
        run_worker(args.worker_start, args.worker_end)
    elif args.assemble:
        assemble()
    else:
        raise SystemExit("Choose --prepare, worker bounds, or --assemble")


if __name__ == "__main__":
    main()
