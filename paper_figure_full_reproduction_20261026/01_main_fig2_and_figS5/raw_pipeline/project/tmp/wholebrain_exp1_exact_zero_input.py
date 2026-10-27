from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
sys.path.insert(0, str(ROOT))

from experiments.pap_theory_validation.pap_validation.core import (
    feature_from_spectrum,
    finite_derivatives,
    input_psd,
    load_model,
    transfer_power,
)
from experiments.pap_theory_validation.pap_validation.exp1 import (
    Z0,
    ZHI,
    ZLO,
    ZSCALE,
    _feature_fun,
)
from experiments.pap_theory_validation.pap_validation.exp3_large_delta_screen import (
    _discrete_transfer,
)
from experiments.pap_theory_validation.pap_validation.io import load_config
from experiments.pap_theory_validation.pap_validation.paper_figure_dense_revision import (
    DT_MS,
    EXP1_ANALYSIS_S,
    EXP1_EPSILON,
    EXP1_TRIALS,
    SMOOTHING_SIGMA_HZ,
    WELCH_SEGMENT_S,
    _welch_features,
)
from experiments.pap_theory_validation.pap_validation.realization_residual_validation import (
    _cov_to_corr,
)


OUTPUT = ROOT / "output" / "wholebrain_exp1_exact_zero_input_2026-08-29"
CACHE = (
    ROOT
    / "experiments"
    / "pap_theory_validation"
    / "outputs"
    / "exp1_exact_zero_input_pap"
)
PREVIEW = ROOT / "tmp" / "wholebrain_exp1_exact_zero_input_preview"

N_CONDITIONS = 100
N_CALIBRATION_PHASES = 3
N_HELDOUT_PHASES = 5
SEED = 2026082902
DESIGN_VERSION = 3

BLUE = "#247BA0"
ORANGE = "#F28E2B"
BLACK = "#222222"


def _standardize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    return (values - values.mean()) / values.std(ddof=1)


def _input_features(z: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
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


def _zero_input_from_scores(
    center_score: np.ndarray,
    exponent_score: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center_score = _standardize(center_score)
    exponent_score = _standardize(exponent_score)
    for _ in range(20):
        u = np.zeros((len(center_score), 4), float)
        u[:, 0] = EXP1_EPSILON * center_score
        u[:, 2] = EXP1_EPSILON * exponent_score
        z = Z0 + u * ZSCALE
        if np.any((z < ZLO) | (z > ZHI)):
            raise RuntimeError("Unexpected clipping in exact-zero input design")
        features = _input_features(z, frequencies)
        peak_centered = features[:, 0] - features[:, 0].mean()
        score_centered = exponent_score - exponent_score.mean()
        exponent_score = _standardize(
            score_centered
            - peak_centered
            * np.dot(peak_centered, score_centered)
            / np.dot(peak_centered, peak_centered)
        )

    u = np.zeros((len(center_score), 4), float)
    u[:, 0] = EXP1_EPSILON * center_score
    u[:, 2] = EXP1_EPSILON * exponent_score
    z = Z0 + u * ZSCALE
    features = _input_features(z, frequencies)
    input_r = float(stats.pearsonr(features[:, 0], features[:, 1]).statistic)
    if abs(input_r) > 1e-12:
        raise RuntimeError(f"Input P-AP correlation is not zero: {input_r}")
    return z, u, features


def _strict_zero_input_design(
    condition_id: int, frequencies: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Make the sample correlation of ideal input P and AP zero to machine precision."""
    rng = np.random.default_rng(SEED + condition_id * 1009)
    return _zero_input_from_scores(
        rng.normal(size=EXP1_TRIALS),
        rng.normal(size=EXP1_TRIALS),
        frequencies,
    )


def _pearson(values: np.ndarray) -> tuple[float, float]:
    result = stats.pearsonr(values[:, 0], values[:, 1])
    return float(result.statistic), float(result.pvalue)


def _simulation_features_with_locked_trial_phases(
    z: np.ndarray,
    trial_ids: np.ndarray,
    discrete_transfer: np.ndarray,
    condition_id: int,
    phase: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Use a fixed phase realization for each original trial through refinements."""
    n_steps = int(round(EXP1_ANALYSIS_S * 1000.0 / DT_MS))
    frequency = np.fft.rfftfreq(n_steps, d=DT_MS / 1000.0)
    active = (frequency < 2000.0) & (frequency > 0.0)
    f = np.maximum(frequency, 0.5)[None, :]
    center = z[:, 0, None]
    amplitude = z[:, 1, None]
    beta = z[:, 2, None]
    background_amplitude = z[:, 3, None]
    input_power = background_amplitude * (f / 10.0) ** beta
    input_power += amplitude * np.exp(-0.5 * ((f - center) / 2.0) ** 2)
    input_power[:, ~active] = 0.0

    phase_base = SEED + 1_000_000 + condition_id * 10007 + phase * 101
    random_phase = np.empty_like(input_power)
    for row, trial_id in enumerate(trial_ids):
        rng = np.random.default_rng(phase_base + int(trial_id) * 1_000_003)
        random_phase[row] = rng.uniform(0.0, 2.0 * np.pi, input_power.shape[1])
    spectrum = np.sqrt(np.maximum(input_power, 0.0)) * np.exp(1j * random_phase)
    spectrum[:, ~active] = 0.0
    paths = np.fft.irfft(
        spectrum * discrete_transfer[None, :], n=n_steps, axis=-1
    ).astype(np.float32)
    return _welch_features(paths)


def _condition_cache_path(condition_id: int) -> Path:
    return CACHE / "conditions" / f"condition_{condition_id:03d}.npz"


def _run_condition(
    condition_id: int,
    frequencies: np.ndarray,
    discrete_transfer: np.ndarray,
    base: np.ndarray,
    gradient: np.ndarray,
    force: bool,
) -> dict[str, object]:
    cache_path = _condition_cache_path(condition_id)
    use_cache = False
    if cache_path.exists() and not force:
        payload = np.load(cache_path)
        use_cache = (
            "design_version" in payload
            and int(payload["design_version"]) == DESIGN_VERSION
        )
        if use_cache:
            z = payload["z"]
            u = payload["u"]
            trial_ids = payload["trial_ids"].astype(int)
            input_features = payload["input_features"]
            simulated = payload["simulated"]
            valid = payload["valid"].astype(bool)
    if not use_cache:
        z, u, input_features = _strict_zero_input_design(condition_id, frequencies)
        trial_ids = np.arange(len(z), dtype=int)
        for refinement in range(9):
            simulated_phases = []
            valid_phases = []
            for phase in range(N_CALIBRATION_PHASES + N_HELDOUT_PHASES):
                phase_features, phase_valid = _simulation_features_with_locked_trial_phases(
                    z,
                    trial_ids,
                    discrete_transfer,
                    condition_id,
                    phase,
                )
                simulated_phases.append(phase_features)
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
            z, u, input_features = _zero_input_from_scores(
                u[common, 0] / EXP1_EPSILON,
                u[common, 2] / EXP1_EPSILON,
                frequencies,
            )
            trial_ids = trial_ids[common]
        else:
            raise RuntimeError(
                f"Condition {condition_id} did not reach a stable all-valid trial set"
            )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            design_version=np.asarray(DESIGN_VERSION),
            z=z,
            u=u,
            trial_ids=trial_ids,
            input_features=input_features,
            simulated=simulated,
            valid=valid,
        )

    input_r = float(stats.pearsonr(input_features[:, 0], input_features[:, 1]).statistic)
    common = valid.all(axis=0) & np.isfinite(simulated).all(axis=(0, 2))
    if not common.all():
        raise RuntimeError(f"Condition {condition_id} cache is not all-valid")
    if common.sum() < 48:
        raise RuntimeError(
            f"Condition {condition_id} has only {common.sum()} common valid trials"
        )

    calibration = simulated[:N_CALIBRATION_PHASES, common]
    calibration_centered = calibration - calibration.mean(axis=0, keepdims=True)
    residual_covariance = np.cov(
        calibration_centered.reshape(-1, 2), rowvar=False, ddof=1
    )

    first_order = base + u[common] @ gradient.T
    signal_covariance = np.cov(first_order, rowvar=False, ddof=1)
    theory_r = float(_cov_to_corr(signal_covariance + residual_covariance))

    heldout = simulated[N_CALIBRATION_PHASES:, common]
    phase_tests = [_pearson(values) for values in heldout]
    phase_r = np.asarray([value[0] for value in phase_tests], float)
    phase_p = np.asarray([value[1] for value in phase_tests], float)
    heldout_mean_r = float(phase_r.mean())
    ensemble = heldout.mean(axis=0)
    ensemble_r, ensemble_p = _pearson(ensemble)

    return {
        "condition_id": condition_id,
        "input_pap_r": input_r,
        "n_trials_initial": EXP1_TRIALS,
        "n_trials_analyzed": int(len(z)),
        "n_valid_common": int(common.sum()),
        "theory_residual_corrected_r": theory_r,
        "simulation_heldout_r_mean": heldout_mean_r,
        "simulation_heldout_r_sd": float(phase_r.std(ddof=1)),
        "simulation_phase_ensemble_r": ensemble_r,
        "simulation_phase_ensemble_p": ensemble_p,
        "heldout_phase_r_json": json.dumps(phase_r.tolist()),
        "heldout_phase_p_json": json.dumps(phase_p.tolist()),
        "residual_var_p": float(residual_covariance[0, 0]),
        "residual_var_ap": float(residual_covariance[1, 1]),
        "residual_cov_pap": float(residual_covariance[0, 1]),
    }


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "axes.labelsize": 19,
            "axes.titlesize": 19,
            "axes.linewidth": 2.8,
            "xtick.labelsize": 17,
            "ytick.labelsize": 17,
            "xtick.major.width": 2.6,
            "ytick.major.width": 2.6,
            "svg.fonttype": "path",
            "figure.facecolor": "white",
        }
    )


def _polish(ax: plt.Axes) -> None:
    ax.yaxis.label.set_weight("bold")
    ax.title.set_weight("bold")
    ax.tick_params(axis="both", which="major", width=2.6, length=7)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)


def _plot(results: pd.DataFrame) -> Path:
    _set_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)

    theory = results.theory_residual_corrected_r.to_numpy(float)
    simulation = results.simulation_heldout_r_mean.to_numpy(float)
    heldout_p = np.concatenate(
        [np.asarray(json.loads(value), float) for value in results.heldout_phase_p_json]
    )
    ensemble_fraction = float(np.mean(results.simulation_phase_ensemble_p < 0.05))
    heldout_fraction = float(np.mean(heldout_p < 0.05))

    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    groups = [theory, simulation]
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

    rng = np.random.default_rng(SEED + 900000)
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
    lower = min(-0.30, float(np.nanmin(all_values)) - 0.06)
    upper = max(0.12, float(np.nanmax(all_values)) + 0.06)
    label_y = lower + 0.025 * (upper - lower)
    for position, values in enumerate(groups):
        ax.text(
            position,
            label_y,
            f"median={np.median(values):.3f}",
            ha="center",
            va="bottom",
            fontsize=13.5,
            fontweight="bold",
        )

    ax.axhline(0.0, color="0.45", lw=2.0, zorder=0)
    ax.set_xticks([0, 1], ["Theory", "Simulation"])
    ax.set(
        ylabel="Output P–AP correlation, r",
        title="Experiment 1: resting state",
        ylim=(lower, upper),
        xlim=(-0.65, 1.65),
    )
    ax.text(
        0.04,
        0.96,
        (
            "Input P–AP: r=0 exactly\n"
            f"Ensemble: {ensemble_fraction:.1%} p<0.05\n"
            f"Held-out phases: {heldout_fraction:.1%} p<0.05"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=13.5,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.90, "pad": 2.0},
    )
    _polish(ax)

    svg = OUTPUT / "experiment1_exact_zero_input_output_pa_correlation.svg"
    fig.savefig(svg, bbox_inches="tight", facecolor="white")
    fig.savefig(
        PREVIEW / "experiment1_exact_zero_input_output_pa_correlation.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    return svg


def run(force: bool = False, limit: int | None = None) -> Path:
    cfg = load_config(
        ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml"
    )
    model = load_model(ROOT / cfg["model_dir"])
    input_idx = model.areas.index("V1")
    output_idx = model.areas.index("PG")

    theory_frequencies = np.unique(
        np.r_[np.geomspace(0.5, 40.0, 180), np.geomspace(40.1, 1200.0, 220)]
    )
    power = transfer_power(model.J, input_idx, output_idx, theory_frequencies)
    feature_map = _feature_fun(
        power, theory_frequencies, (5.0, 20.0), (100.0, 1000.0)
    )
    gradient, _, _ = finite_derivatives(feature_map, np.zeros(4), h=2e-3)
    base = feature_map(np.zeros(4))

    n_steps = int(round(EXP1_ANALYSIS_S * 1000.0 / DT_MS))
    discrete_transfer = _discrete_transfer(
        model.J,
        input_idx,
        output_idx,
        n_steps,
        float(model.params["tau_exc"]),
    )

    count = N_CONDITIONS if limit is None else min(limit, N_CONDITIONS)
    rows = []
    for condition_id in range(count):
        rows.append(
            _run_condition(
                condition_id,
                theory_frequencies,
                discrete_transfer,
                base,
                gradient,
                force,
            )
        )
        print(f"exact-zero Experiment 1 {condition_id + 1}/{count}", flush=True)

    results = pd.DataFrame(rows)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT / "experiment1_exact_zero_input_statistics.csv", index=False)
    results.to_csv(CACHE / "condition_results.csv", index=False)

    heldout_p = np.concatenate(
        [np.asarray(json.loads(value), float) for value in results.heldout_phase_p_json]
    )
    summary = {
        "experiment": "whole-brain Experiment 1 exact-zero input P-AP correlation",
        "input_area": "V1",
        "output_area": "PG",
        "n_conditions": int(len(results)),
        "trials_per_condition_initial": EXP1_TRIALS,
        "trials_per_condition_analyzed_min": int(results.n_trials_analyzed.min()),
        "trials_per_condition_analyzed_max": int(results.n_trials_analyzed.max()),
        "calibration_phases": N_CALIBRATION_PHASES,
        "heldout_phases": N_HELDOUT_PHASES,
        "total_windowed_trial_realizations_analyzed": int(
            results.n_trials_analyzed.sum()
            * (N_CALIBRATION_PHASES + N_HELDOUT_PHASES)
        ),
        "analysis_window_s": EXP1_ANALYSIS_S,
        "dt_ms": DT_MS,
        "welch_segment_s": WELCH_SEGMENT_S,
        "welch_overlap": "50%",
        "smoothing_sigma_hz": SMOOTHING_SIGMA_HZ,
        "peak_band_hz": [5.0, 20.0],
        "slope_band_hz": [100.0, 1000.0],
        "input_zero_definition": "sample Pearson r of deterministic input PSD features P and AP is zero within 1e-12 on the final common-valid trial set in every condition",
        "max_abs_input_pap_r": float(results.input_pap_r.abs().max()),
        "theory_median_r": float(results.theory_residual_corrected_r.median()),
        "simulation_median_r": float(results.simulation_heldout_r_mean.median()),
        "simulation_mean_r": float(results.simulation_heldout_r_mean.mean()),
        "phase_ensemble_fraction_p_lt_0p05": float(
            np.mean(results.simulation_phase_ensemble_p < 0.05)
        ),
        "heldout_phase_fraction_p_lt_0p05": float(np.mean(heldout_p < 0.05)),
        "seed": SEED,
    }
    (OUTPUT / "analysis_manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    (CACHE / "manifest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    return _plot(results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(run(force=args.force, limit=args.limit))


if __name__ == "__main__":
    main()
