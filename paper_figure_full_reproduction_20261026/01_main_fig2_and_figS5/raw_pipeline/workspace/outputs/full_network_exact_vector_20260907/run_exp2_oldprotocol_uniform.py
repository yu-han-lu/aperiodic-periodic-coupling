from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parents[1]
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(HERE), str(PACKAGE), str(FIGURE_SKILL / "scripts")]

import run_full_network_exact as base  # noqa: E402
from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from pap_validation.exp2_pap_correlation_validation import (  # noqa: E402
    _normalized_input_shape,
)


OUTPUT = HERE / "exp2_oldprotocol_uniform"
ANALYSIS_S = 131.072
N_STEPS = int(round(ANALYSIS_S / 0.00025))
N_CALIBRATION = 3
N_HELDOUT = 5
FINAL_TRIALS = 256
SCREEN_TRIALS = 96
SCREEN_MIN_VALID = 28
SCREEN_POOL = 96
FINAL_CANDIDATES = 36
FINAL_POINTS = 24
CHUNK = 32
SEED = 2026090701
INPUT_IDX_NAME = "PG"
OUTPUT_IDX_NAME = "PG"
PEAK_BAND = (5.0, 100.0)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def pooled_within_trial_cov(values: np.ndarray) -> np.ndarray:
    """Unbiased phase-error covariance after centering within each trial."""
    values = np.asarray(values, float)
    q, n, p = values.shape
    if q < 2 or n < 2:
        return np.full((p, p), np.nan)
    centered = values - values.mean(axis=0, keepdims=True)
    return np.einsum("qni,qnj->ij", centered, centered) / (n * (q - 1))


def cov_to_corr(covariance: np.ndarray) -> float:
    if (
        not np.isfinite(covariance).all()
        or covariance[0, 0] <= 0
        or covariance[1, 1] <= 0
    ):
        return np.nan
    return float(
        covariance[0, 1]
        / np.sqrt(covariance[0, 0] * covariance[1, 1])
    )


def feature_corr(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if len(values) < 8 or not np.isfinite(values).all():
        return np.nan
    if np.min(np.std(values, axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[:, 0], values[:, 1]).statistic)


def display_convention(values: np.ndarray) -> np.ndarray:
    """Use the registered Experiment-2 fields: peak Hz and A=-exponent."""
    result = np.asarray(values, float).copy()
    result[..., 0] /= 2.0 * np.pi
    result[..., 1] *= -1.0
    return result


def beta_trials(sigma_beta: float, n_trials: int, condition_id: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + 6100000 + condition_id)
    lower, upper = -0.8 / sigma_beta, 0.8 / sigma_beta
    return stats.truncnorm.rvs(
        lower,
        upper,
        loc=0.0,
        scale=sigma_beta,
        size=n_trials,
        random_state=rng,
    )


def farthest_parameter_sample(frame: pd.DataFrame, count: int) -> pd.DataFrame:
    fields = ["S", "sigma_beta", "floor_ratio"]
    values = frame[fields].to_numpy(float)
    lo = values.min(axis=0)
    span = np.maximum(values.max(axis=0) - lo, np.finfo(float).eps)
    scaled = (values - lo) / span
    center = np.full(3, 0.5)
    chosen = [int(np.argmin(np.sum((scaled - center) ** 2, axis=1)))]
    distance = np.sum((scaled - scaled[chosen[0]]) ** 2, axis=1)
    while len(chosen) < min(count, len(frame)):
        position = int(np.argmax(distance))
        chosen.append(position)
        distance = np.minimum(
            distance, np.sum((scaled - scaled[position]) ** 2, axis=1)
        )
        distance[chosen] = -1.0
    return frame.iloc[chosen].sort_values("condition_id").reset_index(drop=True)


def uniform_r_sample(frame: pd.DataFrame, count: int, column: str) -> pd.DataFrame:
    available = frame[np.isfinite(frame[column])].sort_values(column).copy()
    if len(available) < count:
        raise RuntimeError(f"Only {len(available)} valid conditions for {count} points")
    targets = np.linspace(float(available[column].min()), float(available[column].max()), count)
    values = available[column].to_numpy(float)
    # Globally optimal ordered one-to-one matching between candidate r values
    # and equally spaced targets.  This avoids the low-to-high greedy method
    # consuming a candidate that is much more useful for the next target.
    n = len(values)
    cost = np.full((n + 1, count + 1), np.inf)
    take = np.zeros((n + 1, count + 1), bool)
    cost[:, 0] = 0.0
    for candidate in range(1, n + 1):
        upper = min(candidate, count)
        for selected_count in range(1, upper + 1):
            skip_cost = cost[candidate - 1, selected_count]
            take_cost = (
                cost[candidate - 1, selected_count - 1]
                + abs(values[candidate - 1] - targets[selected_count - 1])
            )
            if take_cost <= skip_cost:
                cost[candidate, selected_count] = take_cost
                take[candidate, selected_count] = True
            else:
                cost[candidate, selected_count] = skip_cost
    selected = []
    candidate = n
    selected_count = count
    while selected_count:
        if take[candidate, selected_count]:
            selected.append(candidate - 1)
            selected_count -= 1
        candidate -= 1
    selected.reverse()
    result = available.iloc[selected].copy()
    result["uniform_target_r"] = targets
    result["uniform_abs_error"] = np.abs(result[column].to_numpy(float) - targets)
    return result.sort_values(column).reset_index(drop=True)


def theory_features(
    model,
    condition: pd.Series,
    betas: np.ndarray,
    input_idx: int,
    output_idx: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, jacobian = base.stable_arrays(model, float(condition.S))
    frequencies = np.geomspace(0.5, 1200.0, 1600)
    transfer = base.continuous_transfer_power_all(
        jacobian, input_idx, np.asarray([output_idx]), frequencies
    )[0]
    values = np.full((len(betas), 2), np.nan)
    valid = np.zeros(len(betas), bool)
    for position, beta in enumerate(betas):
        spectrum = transfer * base._input_q(
            frequencies,
            float(beta),
            float(condition.input_sigma),
            float(condition.floor_ratio),
        )
        # Match the registered simulation-side peak rule: retain the highest
        # identifiable interior maximum and reject only boundary/no-peak cases.
        # The stricter multi-peak flag is a QC annotation, not an exclusion.
        item, item_valid = base.features_from_psd(
            frequencies, spectrum[None, :], PEAK_BAND, relaxed_peak_qc=True
        )
        values[position] = item[0]
        valid[position] = bool(item_valid[0])
    return display_convention(values), valid, jacobian


def simulated_phase_features(
    model,
    condition: pd.Series,
    betas: np.ndarray,
    jacobian: np.ndarray,
    input_idx: int,
    output_idx: int,
    phase: int,
) -> tuple[np.ndarray, np.ndarray]:
    transfer = base.discrete_transfer_all(
        jacobian,
        input_idx,
        np.asarray([output_idx]),
        N_STEPS,
        float(model.params["tau_exc"]),
    )[0]
    frequencies = np.fft.rfftfreq(N_STEPS, d=0.00025)
    values = np.full((len(betas), 2), np.nan)
    valid = np.zeros(len(betas), bool)
    for start in range(0, len(betas), CHUNK):
        stop = min(start + CHUNK, len(betas))
        paths = np.empty((stop - start, N_STEPS), dtype=np.float32)
        for local, trial in enumerate(range(start, stop)):
            rng = np.random.default_rng(
                SEED
                + 6200000
                + int(condition.condition_id) * 100000
                + phase * 10000
                + trial
            )
            shape = _normalized_input_shape(frequencies, float(betas[trial]))
            angle = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
            spectrum = np.sqrt(shape) * np.exp(1j * angle)
            spectrum[0] = 0.0
            spectrum[-1] = spectrum[-1].real
            forcing = np.fft.irfft(spectrum, n=N_STEPS)
            forcing *= float(condition.input_sigma) / max(forcing.std(), 1e-15)
            forcing += (
                float(condition.input_sigma)
                * float(condition.floor_ratio)
                * rng.normal(size=N_STEPS)
            )
            forcing -= forcing.mean()
            paths[local] = np.fft.irfft(
                np.fft.rfft(forcing) * transfer, n=N_STEPS
            ).astype(np.float32)
        local_values, local_valid = base.features_from_paths(
            paths, ANALYSIS_S, PEAK_BAND, relaxed_peak_qc=True
        )
        values[start:stop] = local_values
        valid[start:stop] = local_valid
    return display_convention(values), valid


def evaluate_condition(
    model,
    condition: pd.Series,
    n_trials: int,
    phases: int,
    cache_dir: Path,
) -> dict:
    condition_id = int(condition.condition_id)
    cache_path = cache_dir / f"condition_{condition_id:04d}.npz"
    input_idx = model.areas.index(INPUT_IDX_NAME)
    output_idx = model.areas.index(OUTPUT_IDX_NAME)
    betas = beta_trials(float(condition.sigma_beta), n_trials, condition_id)
    theory, theory_valid, jacobian = theory_features(
        model, condition, betas, input_idx, output_idx
    )
    if cache_path.exists():
        with np.load(cache_path) as saved:
            phase_features = saved["phase_features"].astype(float)
            phase_valid = saved["phase_valid"].astype(bool)
        if phase_features.shape != (phases, n_trials, 2):
            raise RuntimeError(f"Cache shape mismatch: {cache_path}")
    else:
        phase_features = np.full((phases, n_trials, 2), np.nan)
        phase_valid = np.zeros((phases, n_trials), bool)
        for phase in range(phases):
            phase_features[phase], phase_valid[phase] = simulated_phase_features(
                model,
                condition,
                betas,
                jacobian,
                input_idx,
                output_idx,
                phase,
            )
            print(
                f"condition={condition_id} phase={phase + 1}/{phases} "
                f"valid={phase_valid[phase].sum()}/{n_trials}",
                flush=True,
            )
        atomic_npz(
            cache_path,
            betas=betas,
            theory=theory,
            theory_valid=theory_valid,
            phase_features=phase_features,
            phase_valid=phase_valid,
        )
    common = theory_valid & phase_valid.all(axis=0)
    if common.sum() < 8:
        theory_r = simulation_r = exact_r = np.nan
        residual = np.full((2, 2), np.nan)
        heldout_rs: list[float] = []
    else:
        signal_cov = np.cov(theory[common], rowvar=False, ddof=1)
        residual = pooled_within_trial_cov(
            phase_features[:N_CALIBRATION, common]
        )
        # Registered old Experiment-2 estimand: theory predicts one held-out
        # realization; five independently measured held-out r values are then
        # averaged.  Therefore residual covariance is not divided by five.
        exact_r = cov_to_corr(signal_cov)
        theory_r = cov_to_corr(signal_cov + residual)
        heldout_rs = [
            feature_corr(phase_features[phase, common])
            for phase in range(N_CALIBRATION, phases)
        ]
        simulation_r = (
            float(np.mean(heldout_rs)) if len(heldout_rs) else np.nan
        )
    return {
        **condition.to_dict(),
        "full55_exact_mapping_r": exact_r,
        "full55_corrected_theory_r": theory_r,
        "full55_simulation_r": simulation_r,
        "full55_n_valid": int(common.sum()),
        "full55_heldout_r_json": json.dumps(heldout_rs),
        "residual_var_P": float(residual[0, 0]),
        "residual_var_A": float(residual[1, 1]),
        "residual_cov_PA": float(residual[0, 1]),
        "analysis_s": ANALYSIS_S,
        "n_trials": n_trials,
        "calibration_phases": N_CALIBRATION,
        "heldout_phases": max(0, phases - N_CALIBRATION),
    }


def run_screen(force: bool = False) -> Path:
    destination = OUTPUT / "screen_results.csv"
    if destination.exists() and not force:
        return destination
    source = pd.read_csv(
        PACKAGE
        / "outputs/exp2_pap_s32_window_131p072/selected_1000_condition_results_rlt0p6.csv"
    )
    pool = farthest_parameter_sample(source, SCREEN_POOL)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pool.to_csv(OUTPUT / "screen_pool.csv", index=False)
    model = base.load_wholebrain()
    rows = []
    started = time.time()
    for ordinal, (_, condition) in enumerate(pool.iterrows(), 1):
        rows.append(
            evaluate_condition(
                model,
                condition,
                SCREEN_TRIALS,
                N_CALIBRATION,
                OUTPUT / "screen_cache",
            )
        )
        pd.DataFrame(rows).to_csv(destination, index=False)
        print(
            f"screen {ordinal}/{len(pool)} r={rows[-1]['full55_corrected_theory_r']:.4f}",
            flush=True,
        )
    chosen = uniform_r_sample(
        pd.DataFrame(rows).query("full55_n_valid >= @SCREEN_MIN_VALID"),
        FINAL_CANDIDATES,
        "full55_corrected_theory_r",
    )
    chosen.to_csv(OUTPUT / "screen_selected_36.csv", index=False)
    (OUTPUT / "screen_metrics.json").write_text(
        json.dumps(
            {
                "candidate_pool": len(pool),
                "valid_candidates": int(
                    (pd.DataFrame(rows).full55_n_valid >= SCREEN_MIN_VALID).sum()
                ),
                "selected_candidates": len(chosen),
                "runtime_s": time.time() - started,
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def run_final(force: bool = False) -> Path:
    run_screen(force=False)
    destination = OUTPUT / "final_candidates_36.csv"
    if destination.exists() and not force:
        return destination
    selected = pd.read_csv(OUTPUT / "screen_selected_36.csv")
    model = base.load_wholebrain()
    rows = []
    started = time.time()
    for ordinal, (_, condition) in enumerate(selected.iterrows(), 1):
        rows.append(
            evaluate_condition(
                model,
                condition,
                FINAL_TRIALS,
                N_CALIBRATION + N_HELDOUT,
                OUTPUT / "final_cache",
            )
        )
        pd.DataFrame(rows).to_csv(destination, index=False)
        print(
            f"final {ordinal}/{len(selected)} "
            f"theory={rows[-1]['full55_corrected_theory_r']:.4f} "
            f"simulation={rows[-1]['full55_simulation_r']:.4f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    valid = frame.query("full55_n_valid >= 80").copy()
    final = uniform_r_sample(valid, FINAL_POINTS, "full55_corrected_theory_r")
    final.to_csv(OUTPUT / "final_uniform_24.csv", index=False)
    fit = stats.linregress(
        final.full55_corrected_theory_r, final.full55_simulation_r
    )
    identity_r2 = 1.0 - np.sum(
        (final.full55_simulation_r - final.full55_corrected_theory_r) ** 2
    ) / np.sum(
        (final.full55_simulation_r - final.full55_simulation_r.mean()) ** 2
    )
    (OUTPUT / "final_metrics.json").write_text(
        json.dumps(
            {
                "n_conditions": len(final),
                "pearson_r": float(fit.rvalue),
                "ols_slope": float(fit.slope),
                "ols_intercept": float(fit.intercept),
                "ols_R2": float(fit.rvalue**2),
                "identity_R2": float(identity_r2),
                "runtime_s": time.time() - started,
                "analysis_s": ANALYSIS_S,
                "n_trials": FINAL_TRIALS,
                "calibration_phases": N_CALIBRATION,
                "heldout_phases": N_HELDOUT,
                "simulation_solver": (
                    "exact discrete active-branch solution of complete 55-area x E/I "
                    "threshold-linear network"
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return destination


def plot() -> Path:
    run_final(force=False)
    data = pd.read_csv(OUTPUT / "final_uniform_24.csv")
    x = data.full55_corrected_theory_r.to_numpy(float)
    y = data.full55_simulation_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y)
    fit = stats.linregress(x[use], y[use])
    x_low, x_high = float(x[use].min()), float(x[use].max())
    y_low, y_high = float(y[use].min()), float(y[use].max())
    x_padding = 0.05 * max(x_high - x_low, 0.1)
    y_padding = 0.08 * max(y_high - y_low, 0.1)
    grid = np.linspace(x_low - x_padding, x_high + x_padding, 200)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 15,
            "axes.linewidth": 1.7,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.spines.right": False,
            "axes.spines.top": False,
        }
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.16, top=0.96)
    ax.scatter(x[use], y[use], s=42, alpha=0.56, color="#247BA0", edgecolor="none")
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#222222", lw=2.8)
    ax.set_xlim(x_low - x_padding, x_high + x_padding)
    ax.set_ylim(y_low - y_padding, y_high + y_padding)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_title("")
    ax.text(
        0.05,
        0.95,
        f"slope = {fit.slope:.3f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=21,
        fontweight="bold",
    )
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)

    stem = OUTPUT / "exp2_full55_oldprotocol_uniform_5x4"
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
    return stem.with_suffix(".svg")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--screen", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.screen:
        print(run_screen(force=args.force))
    if args.final:
        print(run_final(force=args.force))
    if args.plot:
        print(plot())


if __name__ == "__main__":
    main()
