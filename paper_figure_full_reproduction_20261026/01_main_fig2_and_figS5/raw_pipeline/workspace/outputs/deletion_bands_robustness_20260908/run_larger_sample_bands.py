from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from numba import set_num_threads


SCRIPT_ROOT = Path(__file__).resolve().parent
TASK_ROOT = SCRIPT_ROOT.parents[1]
SOURCE_48_ROOT = TASK_ROOT / "outputs/full_network_exact_vector_20260907"
BASE_SCRIPT = SOURCE_48_ROOT / "run_full_network_exact.py"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path.insert(0, str(FIGURE_SKILL / "scripts"))
sys.path.insert(0, str(TASK_ROOT / "outputs"))

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from heun_network_common import (  # noqa: E402
    heun_discrete_transfer_all,
    threshold_network_heun_batch_fixedpoint,
)


def load_base():
    spec = importlib.util.spec_from_file_location("full55_band_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_base()

from run_conditioning_experiment import (  # noqa: E402
    SEED as CONDITIONING_SEED,
    _design as conditioning_design,
)

TARGET_TRIALS = int(os.environ.get("ROBUST_TARGET_TRIALS", "128"))
REUSED_TRIALS = int(os.environ.get("ROBUST_REUSED_TRIALS", "48"))
OUTPUT_TAG = os.environ.get("ROBUST_OUTPUT_TAG", str(TARGET_TRIALS))
RESULT_ROOT = Path(os.environ.get("ROBUST_RESULT_ROOT", str(SCRIPT_ROOT))).resolve()
REGISTERED_ROOT = Path(
    os.environ.get("ROBUST_REGISTERED_ROOT", str(SOURCE_48_ROOT))
).resolve()
REGISTERED_STATE_SUFFIX = os.environ.get("ROBUST_REGISTERED_STATE_SUFFIX", "")
INTEGRATOR = os.environ.get("ROBUST_INTEGRATOR", "euler").strip().lower()
if INTEGRATOR not in {"euler", "heun"}:
    raise ValueError(f"Unsupported ROBUST_INTEGRATOR={INTEGRATOR!r}")
BANDS = base.BANDS

COLORS = {
    ("Theory", True): "#D1495B",
    ("Theory", False): "#247BA0",
    ("Simulation", True): "#F28E2B",
    ("Simulation", False): "#2A9D8F",
}
BLACK = "#222222"


def old_state_path(band: str, network_pos: int) -> Path:
    slug = band.lower().replace(" ", "_")
    return (
        REGISTERED_ROOT
        / f"band_{slug}_states{REGISTERED_STATE_SUFFIX}/network_{network_pos:03d}.npz"
    )


def new_state_path(band: str, network_pos: int) -> Path:
    slug = band.lower().replace(" ", "_")
    return RESULT_ROOT / f"band_{slug}_states_{OUTPUT_TAG}/network_{network_pos:03d}.npz"


def load_state(path: Path) -> dict[str, np.ndarray | int]:
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


def load_or_extend_design() -> np.ndarray:
    registered_path = (
        TASK_ROOT
        / "outputs/covariance_conditioning_pilot/fpinit_large/cond_100/design.npz"
    )
    with np.load(registered_path) as saved:
        sigma = saved["sigma"].astype(float)
        registered_z = saved["z"].astype(float)
    if TARGET_TRIALS <= len(registered_z):
        return registered_z[:TARGET_TRIALS]

    z, u, clipped = conditioning_design(
        sigma, TARGET_TRIALS, CONDITIONING_SEED + 990000
    )
    if not np.array_equal(z[: len(registered_z)], registered_z):
        raise RuntimeError(
            "The deterministically extended input design changed the registered prefix"
        )
    design_path = RESULT_ROOT / f"design_{OUTPUT_TAG}.npz"
    if not design_path.exists():
        base.atomic_npz(
            design_path,
            sigma=sigma,
            z=z,
            u=u,
            clipped_parameter_fraction=np.asarray(clipped),
        )
    return z


def assert_reused_prefix(
    old: dict[str, np.ndarray | int],
    theory: np.ndarray,
    theory_valid: np.ndarray,
    calibration: np.ndarray,
    calibration_valid: np.ndarray,
) -> None:
    checks = (
        ("theory", theory[:, :REUSED_TRIALS], old["theory"]),
        ("theory_valid", theory_valid[:, :REUSED_TRIALS], old["theory_valid"]),
        ("calibration", calibration[:, :, :REUSED_TRIALS], old["calibration"]),
        (
            "calibration_valid",
            calibration_valid[:, :, :REUSED_TRIALS],
            old["calibration_valid"],
        ),
    )
    for name, fresh, registered in checks:
        if fresh.dtype == bool:
            equal = np.array_equal(fresh, registered)
        else:
            equal = np.allclose(fresh, registered, rtol=2e-6, atol=2e-7, equal_nan=True)
        if not equal:
            raise RuntimeError(
                f"The registered first-{REUSED_TRIALS} prefix changed for {name}"
            )


def heun_linear_phase_features(
    inputs: np.ndarray,
    j: np.ndarray,
    input_idx: int,
    output_indices: np.ndarray,
    analysis_s: float,
    peak_band: tuple[float, float],
    tau_e: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Match the existing linear calibration using the Heun recurrence."""

    burn_steps = int(round(base.BURN_S * 1000.0 / base.DT_MS))
    n_steps = int(round(analysis_s * 1000.0 / base.DT_MS))
    transfer = heun_discrete_transfer_all(
        j, input_idx, output_indices, n_steps, tau_e, base.DT_MS
    )
    input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
    values = np.full((len(output_indices), len(inputs), 2), np.nan)
    valid = np.zeros((len(output_indices), len(inputs)), bool)
    for start in range(0, len(output_indices), 5):
        stop = min(start + 5, len(output_indices))
        paths = np.fft.irfft(
            input_fft[:, None, :] * transfer[None, start:stop, :],
            n=n_steps,
            axis=-1,
        ).astype(np.float32)
        local_values, local_valid = base.features_from_paths(
            paths, analysis_s, peak_band
        )
        values[start:stop] = local_values.transpose(1, 0, 2)
        valid[start:stop] = local_valid.T
    return values, valid


def heun_nonlinear_average_features(
    model,
    inputs_by_phase: list[np.ndarray],
    fln: np.ndarray,
    fixedpoint: np.ndarray,
    input_idx: int,
    output_indices: np.ndarray,
    analysis_s: float,
    peak_band: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Full 110-state ReLU Heun trajectories, averaged in PSD space."""

    frequencies, keep = base.frequencies_for(analysis_s)
    psd_sum = np.zeros(
        (len(output_indices), len(inputs_by_phase[0]), len(frequencies)),
        dtype=np.float64,
    )
    rectification = 0
    fs = 1000.0 / base.DT_MS
    nperseg = int(round(base.SEGMENT_S * fs))
    arrays = base._simulation_arrays(model.params)
    edge_target, edge_source, edge_e, edge_i = base._edge_arrays(model.params, fln)
    n_area = len(model.areas)
    burn_steps = int(round(base.BURN_S * 1000.0 / base.DT_MS))
    for phase, inputs in enumerate(inputs_by_phase):
        for start in range(0, len(inputs), base.BAND_CHUNK):
            stop = min(start + base.BAND_CHUNK, len(inputs))
            paths, rect1, rect2 = threshold_network_heun_batch_fixedpoint(
                inputs[start:stop],
                arrays[0], arrays[1], arrays[2], arrays[3],
                edge_target, edge_source, edge_e, edge_i,
                arrays[8], arrays[9],
                fixedpoint[:n_area], fixedpoint[n_area:],
                base.DT_MS,
                float(model.params["tau_exc"]),
                float(model.params["tau_inh"]),
                burn_steps,
                output_indices,
                input_idx,
            )
            actual_f, psd = base.signal.welch(
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
            psd_sum[:, start:stop] += psd[..., keep].transpose(1, 0, 2)
            rectification += int(rect1.sum() + rect2.sum())
        print(
            f"  nonlinear Heun heldout phase {phase + 1}/{len(inputs_by_phase)}",
            flush=True,
        )
    values, valid = base.features_from_psd(
        frequencies, psd_sum / len(inputs_by_phase), peak_band
    )
    return values, valid, rectification


def run_band(band: str) -> Path:
    set_num_threads(int(os.environ.get("ROBUST_NUMBA_THREADS", "1")))
    started = time.time()
    slug = band.lower().replace(" ", "_")
    output = RESULT_ROOT / f"band_{slug}_full55_units_{OUTPUT_TAG}.csv.gz"
    if output.exists():
        return output

    center, peak_band = BANDS[band]
    model = base.load_wholebrain()
    source_all = pd.read_csv(
        TASK_ROOT
        / "outputs/covariance_conditioning_pilot/fpinit_large/cond_100/pattern_results.csv"
    )
    source = (
        source_all[source_all.n_deleted == 1]
        .sort_values("row_id")
        .reset_index(drop=True)
    )
    if len(source) != 48:
        raise RuntimeError(f"Expected the registered 48 deletion networks, found {len(source)}")

    z = load_or_extend_design()
    z[:, 0] += center - 10.0

    networks = base.deletion_networks(model, source)
    input_idx = model.areas.index("A6Va")
    output_indices = np.arange(len(model.areas), dtype=np.int64)
    frequencies, _ = base.frequencies_for(base.BAND_ANALYSIS_S)
    cal_inputs = [
        base._matched_bandlimited_inputs(
            z,
            base.DT_MS,
            base.BAND_ANALYSIS_S,
            base.BURN_S,
            base.INPUT_SIGMA_DELETION,
            base.SEED + 1010000 + phase * 100000,
        )
        for phase in range(base.BAND_CAL_PHASES)
    ]
    held_inputs = [
        base._matched_bandlimited_inputs(
            z,
            base.DT_MS,
            base.BAND_ANALYSIS_S,
            base.BURN_S,
            base.INPUT_SIGMA_DELETION,
            base.SEED
            + 1010000
            + (base.BAND_CAL_PHASES + phase) * 100000,
        )
        for phase in range(base.BAND_HELD_PHASES)
    ]

    all_network: list[dict[str, np.ndarray | int | float]] = []
    for network_pos, (fln, fixedpoint, gain, j, residual) in enumerate(networks):
        destination = new_state_path(band, network_pos)
        if destination.exists():
            state = load_state(destination)
            if state["theory"].shape[1] != TARGET_TRIALS:
                raise RuntimeError(f"Incomplete cache shape in {destination}")
        else:
            if INTEGRATOR == "heun":
                theory, theory_valid = base.theoretical_features_all(
                    j, z, input_idx, output_indices, frequencies, peak_band
                )
                calibration = np.full(
                    (
                        base.BAND_CAL_PHASES,
                        len(output_indices),
                        TARGET_TRIALS,
                        2,
                    ),
                    np.nan,
                )
                calibration_valid = np.zeros(
                    (
                        base.BAND_CAL_PHASES,
                        len(output_indices),
                        TARGET_TRIALS,
                    ),
                    bool,
                )
                for phase in range(base.BAND_CAL_PHASES):
                    calibration[phase], calibration_valid[phase] = (
                        heun_linear_phase_features(
                            cal_inputs[phase],
                            j,
                            input_idx,
                            output_indices,
                            base.BAND_ANALYSIS_S,
                            peak_band,
                            float(model.params["tau_exc"]),
                        )
                    )
                simulation, simulation_valid, rectification = (
                    heun_nonlinear_average_features(
                        model,
                        held_inputs,
                        fln,
                        fixedpoint,
                        input_idx,
                        output_indices,
                        base.BAND_ANALYSIS_S,
                        peak_band,
                    )
                )
                state = {
                    "theory": theory,
                    "theory_valid": theory_valid,
                    "calibration": calibration,
                    "calibration_valid": calibration_valid,
                    "simulation": simulation,
                    "simulation_valid": simulation_valid,
                    "rectification": int(rectification),
                }
                base.atomic_npz(destination, **state)
                all_network.append(
                    {
                        **state,
                        "gain": gain,
                        "residual": float(residual),
                    }
                )
                print(
                    f"{band}: network {network_pos + 1}/{len(networks)}; "
                    f"integrator=Heun; elapsed={time.time() - started:.1f}s",
                    flush=True,
                )
                continue

            old_path = old_state_path(band, network_pos)
            if not old_path.exists():
                raise RuntimeError(
                    f"Missing registered {REUSED_TRIALS}-trial state: {old_path}"
                )
            old = load_state(old_path)
            if old["theory"].shape[1] != REUSED_TRIALS:
                raise RuntimeError(f"Unexpected registered cache shape in {old_path}")

            extra_z = z[REUSED_TRIALS:]
            extra_theory, extra_theory_valid = base.theoretical_features_all(
                j, extra_z, input_idx, output_indices, frequencies, peak_band
            )
            theory = np.concatenate(
                [old["theory"], extra_theory], axis=1
            )
            theory_valid = np.concatenate(
                [old["theory_valid"], extra_theory_valid], axis=1
            )
            extra_calibration = np.full(
                (
                    base.BAND_CAL_PHASES,
                    len(output_indices),
                    TARGET_TRIALS - REUSED_TRIALS,
                    2,
                ),
                np.nan,
            )
            extra_calibration_valid = np.zeros(
                (
                    base.BAND_CAL_PHASES,
                    len(output_indices),
                    TARGET_TRIALS - REUSED_TRIALS,
                ),
                bool,
            )
            for phase in range(base.BAND_CAL_PHASES):
                extra_calibration[phase], extra_calibration_valid[phase] = (
                    base.linear_phase_features(
                    cal_inputs[phase][REUSED_TRIALS:],
                    j,
                    input_idx,
                    output_indices,
                    base.BAND_ANALYSIS_S,
                    peak_band,
                    float(model.params["tau_exc"]),
                )
                )
            calibration = np.concatenate(
                [old["calibration"], extra_calibration], axis=2
            )
            calibration_valid = np.concatenate(
                [old["calibration_valid"], extra_calibration_valid], axis=2
            )
            extra_inputs = [values[REUSED_TRIALS:] for values in held_inputs]
            extra_simulation, extra_valid, extra_rectification = (
                base.nonlinear_average_features(
                    model,
                    extra_inputs,
                    fln,
                    fixedpoint,
                    input_idx,
                    output_indices,
                    base.BAND_ANALYSIS_S,
                    peak_band,
                )
            )
            simulation = np.concatenate(
                [old["simulation"], extra_simulation], axis=1
            )
            simulation_valid = np.concatenate(
                [old["simulation_valid"], extra_valid], axis=1
            )
            state = {
                "theory": theory,
                "theory_valid": theory_valid,
                "calibration": calibration,
                "calibration_valid": calibration_valid,
                "simulation": simulation,
                "simulation_valid": simulation_valid,
                "rectification": int(old["rectification"]) + int(extra_rectification),
            }
            base.atomic_npz(destination, **state)

        all_network.append(
            {
                **state,
                "gain": gain,
                "residual": float(residual),
            }
        )
        print(
            f"{band}: network {network_pos + 1}/{len(networks)}; "
            f"elapsed={time.time() - started:.1f}s",
            flush=True,
        )

    rows: list[dict[str, object]] = []
    intact = all_network[0]
    for position, item in enumerate(source.itertuples(index=False), 1):
        deleted = all_network[position]
        deleted_node = str(item.deleted_nodes)
        deleted_idx = model.areas.index(deleted_node)
        for output_idx, output_area in enumerate(model.areas):
            if output_idx == deleted_idx or not bool(deleted["gain"][output_idx]):
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
            base_signal = np.cov(
                intact["theory"][output_idx, common], rowvar=False, ddof=1
            )
            del_signal = np.cov(
                deleted["theory"][output_idx, common], rowvar=False, ddof=1
            )
            base_resid = base.pooled_phase_cov(
                intact["calibration"][:, output_idx, common]
            )
            del_resid = base.pooled_phase_cov(
                deleted["calibration"][:, output_idx, common]
            )
            theory_base = base.corr_from_cov(
                base_signal + base_resid / base.BAND_HELD_PHASES
            )
            theory_deleted = base.corr_from_cov(
                del_signal + del_resid / base.BAND_HELD_PHASES
            )
            simulation_base = base.corr_feature(
                intact["simulation"][output_idx], common
            )
            simulation_deleted = base.corr_feature(
                deleted["simulation"][output_idx], common
            )
            rows.append(
                {
                    "band": band,
                    "deleted_node": deleted_node,
                    "output_area": output_area,
                    "direct": bool(model.fln[output_idx, deleted_idx] > 0),
                    "n_valid": int(common.sum()),
                    "theory_baseline_r": theory_base,
                    "theory_deleted_r": theory_deleted,
                    "theory_delta_r": theory_deleted - theory_base,
                    "simulation_baseline_r": simulation_base,
                    "simulation_deleted_r": simulation_deleted,
                    "simulation_delta_r": simulation_deleted - simulation_base,
                }
            )

    result = pd.DataFrame(rows)
    result.to_csv(output, index=False, compression="gzip")
    metrics = {
        "band": band,
        "deleted_networks": int(len(source)),
        "output_units": int(len(result)),
        "target_trials": TARGET_TRIALS,
        "registered_trials_reused": REUSED_TRIALS if INTEGRATOR == "euler" else 0,
        "new_trials_simulated": (
            TARGET_TRIALS - REUSED_TRIALS if INTEGRATOR == "euler" else TARGET_TRIALS
        ),
        "analysis_s": base.BAND_ANALYSIS_S,
        "calibration_phases": base.BAND_CAL_PHASES,
        "heldout_phases": base.BAND_HELD_PHASES,
        "input_area": "A6Va",
        "runtime_s": time.time() - started,
        "simulation": (
            "complete 55-area x E/I threshold-linear/ReLU explicit Heun integration"
            if INTEGRATOR == "heun"
            else "complete 55-area x E/I threshold-linear/ReLU Euler integration"
        ),
    }
    (RESULT_ROOT / f"band_{slug}_metrics_{OUTPUT_TAG}.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )
    return output


def assemble_statistics() -> tuple[pd.DataFrame, pd.DataFrame]:
    frames = [
        pd.read_csv(
            RESULT_ROOT
            / f"band_{band.lower().replace(' ', '_')}_full55_units_{OUTPUT_TAG}.csv.gz"
        )
        for band in BANDS
    ]
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(
        RESULT_ROOT / f"deletion_bands_full55_raw_{OUTPUT_TAG}.csv.gz",
        index=False,
        compression="gzip",
    )
    units = []
    stats_rows = []
    for band_index, band in enumerate(BANDS):
        subset_band = raw[raw.band == band]
        for source_index, (source, column) in enumerate(
            (("Theory", "theory_delta_r"), ("Simulation", "simulation_delta_r"))
        ):
            local = subset_band[
                ["deleted_node", "output_area", "direct", column]
            ].dropna().copy()
            local["value"] = local[column].abs()
            local["source"] = source
            local["band"] = band
            units.append(
                local[["band", "source", "deleted_node", "output_area", "value", "direct"]]
            )
            node = local.groupby(["deleted_node", "direct"], as_index=False).value.mean()
            pivot = node.pivot(index="deleted_node", columns="direct", values="value").dropna()
            difference = pivot[True].to_numpy() - pivot[False].to_numpy()
            stats_rows.append(
                {
                    "band": band,
                    "source": source,
                    "n_deleted_nodes": int(len(pivot)),
                    "direct_mean": float(pivot[True].mean()),
                    "non_direct_mean": float(pivot[False].mean()),
                    "mean_difference": float(difference.mean()),
                    "sign_flip_p": base.sign_flip_p(
                        difference,
                        base.SEED + 5000 + band_index * 10 + source_index,
                    ),
                    "fraction_positive": float(np.mean(difference > 0)),
                }
            )
    unit_frame = pd.concat(units, ignore_index=True)
    stat_frame = pd.DataFrame(stats_rows)
    stat_frame["sign_flip_fdr_q"] = base.bh(stat_frame.sign_flip_p.to_numpy())
    unit_frame.to_csv(
        RESULT_ROOT / f"deletion_bands_full55_plot_units_{OUTPUT_TAG}.csv.gz",
        index=False,
        compression="gzip",
    )
    stat_frame.to_csv(
        RESULT_ROOT / f"deletion_bands_full55_statistics_{OUTPUT_TAG}.csv",
        index=False,
    )
    return unit_frame, stat_frame


def plot() -> None:
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
    units, stat_frame = assemble_statistics()
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.subplots_adjust(left=0.075, right=0.985, bottom=0.16, top=0.84)
    positions = np.arange(len(BANDS), dtype=float)
    offsets = {
        ("Theory", True): -0.30,
        ("Theory", False): -0.10,
        ("Simulation", True): 0.10,
        ("Simulation", False): 0.30,
    }
    annotation_max = 0.0
    for band_pos, band in enumerate(BANDS):
        for source in ("Theory", "Simulation"):
            subset = units[(units.source == source) & (units.band == band)]
            for direct in (True, False):
                values = subset.loc[subset.direct == direct, "value"].to_numpy(float)
                values = values[np.isfinite(values) & (values > 0)]
                color = COLORS[(source, direct)]
                box = ax.boxplot(
                    [values],
                    positions=[band_pos + offsets[(source, direct)]],
                    widths=0.16,
                    patch_artist=True,
                    showfliers=True,
                    medianprops={"color": BLACK, "linewidth": 2.5},
                    whiskerprops={"linewidth": 1.8},
                    capprops={"linewidth": 1.8},
                    flierprops={
                        "marker": "o",
                        "markersize": 2.4,
                        "markerfacecolor": color,
                        "markeredgecolor": "none",
                        "alpha": 0.35,
                    },
                )
                box["boxes"][0].set_facecolor(color)
                box["boxes"][0].set_alpha(0.34)
                box["boxes"][0].set_linewidth(2.0)
            q_value = float(
                stat_frame.loc[
                    (stat_frame.source == source) & (stat_frame.band == band),
                    "sign_flip_fdr_q",
                ].iloc[0]
            )
            top = float(subset.loc[subset.value > 0, "value"].max()) * 1.30
            left = band_pos + offsets[(source, True)]
            right = band_pos + offsets[(source, False)]
            ax.plot(
                [left, left, right, right],
                [top / 1.12, top, top, top / 1.12],
                color=BLACK,
                lw=1.7,
                clip_on=False,
            )
            annotation_max = max(annotation_max, top * 1.55)
            ax.text(
                (left + right) / 2.0,
                top * 1.08,
                base.stars(q_value),
                ha="center",
                va="bottom",
                fontsize=17,
                fontweight="bold",
            )
    ax.set_yscale("log")
    ax.set_xlim(-0.55, len(BANDS) - 0.45)
    ax.set_xticks(positions, ["Alpha", "Beta", "Low\ngamma", "High\ngamma"])
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)
    positive = units.loc[units.value > 0, "value"].to_numpy(float)
    ax.set_ylim(float(positive.min() * 0.7), annotation_max * 20.0)
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
            markerfacecolor=COLORS[(source, direct)],
            markeredgecolor="none",
            alpha=0.72,
            markersize=10,
            label=label,
        )
        for source, direct, label in legend_order
    ]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4)
    stem = (
        RESULT_ROOT
        / f"deletion_bands_direct_nondirect_{OUTPUT_TAG}trials_combined_4colors_no_titles"
    )
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

    if INTEGRATOR == "euler" and REUSED_TRIALS > 0:
        registered_stats = REGISTERED_ROOT / (
            "deletion_bands_full55_statistics.csv"
            if REUSED_TRIALS == 48
            else f"deletion_bands_full55_statistics_{REUSED_TRIALS}.csv"
        )
        old_stats = pd.read_csv(registered_stats)
        comparison = old_stats.merge(
            stat_frame,
            on=["band", "source"],
            suffixes=(f"_{REUSED_TRIALS}", f"_{TARGET_TRIALS}"),
        )
        comparison.to_csv(
            RESULT_ROOT / f"statistics_{REUSED_TRIALS}_vs_{TARGET_TRIALS}.csv",
            index=False,
        )
    (RESULT_ROOT / "manifest.json").write_text(
        json.dumps(
            {
                "change": (
                    f"full rerun with {TARGET_TRIALS} trials using explicit Heun"
                    if INTEGRATOR == "heun"
                    else f"trials increased from {REUSED_TRIALS} to {TARGET_TRIALS}"
                ),
                "integrator": INTEGRATOR,
                "reused_registered_prefix_trials": (
                    REUSED_TRIALS if INTEGRATOR == "euler" else 0
                ),
                "new_trials_per_band": (
                    TARGET_TRIALS - REUSED_TRIALS
                    if INTEGRATOR == "euler"
                    else TARGET_TRIALS
                ),
                "fixed_settings": {
                    "deleted_networks": 48,
                    "input_area": "A6Va",
                    "analysis_s": base.BAND_ANALYSIS_S,
                    "dt_ms": base.DT_MS,
                    "calibration_phases": base.BAND_CAL_PHASES,
                    "heldout_phases": base.BAND_HELD_PHASES,
                    "covariance_condition": 100,
                },
                "statistical_unit": "deleted node; direct and non-direct output means paired within node",
                "multiple_testing": "Benjamini-Hochberg across 4 bands x 2 sources",
                "colors": {
                    f"{source}_{'direct' if direct else 'non_direct'}": COLORS[(source, direct)]
                    for source, direct, _ in legend_order
                },
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--band", choices=list(BANDS))
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    if args.band:
        print(run_band(args.band))
    if args.plot:
        plot()


if __name__ == "__main__":
    main()
