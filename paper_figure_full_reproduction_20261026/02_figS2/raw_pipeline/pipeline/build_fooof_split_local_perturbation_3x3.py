from __future__ import annotations

"""Local first-order FOOOF--IRASA/SPLIT validation around fixed centres."""

import argparse
import json
import multiprocessing as mp
from pathlib import Path
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT_CODE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project/experiments/pap_theory_validation/pap_validation"
FIGURE_SCRIPTS = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure/scripts"
for source in (HERE, PROJECT_CODE, FIGURE_SCRIPTS):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import build_two_fooof_split_3x3_revisions as renderer  # noqa: E402
import redraw_fooof_irasa_split_3x3 as shared  # noqa: E402


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_local_first_order_3x3_2026-09-13"
)
STEM = "fooof_irasa_split_local_first_order_nature_neuroscience_3x3"
BASE_SEED = shared.frozen.BASE_SEED + 14_000_000
N_GROUPS = 20
N_CALIBRATION = 120
N_HELDOUT = 120
PILOT_GROUPS = 2
PILOT_N = 32
PILOT_EPSILONS = (0.0025, 0.005, 0.01, 0.02)
DEFAULT_EPSILON = 0.01
WIDTH0_HZ = 3.75
LOW_BAND_STRENGTH0 = 3.3
HIGH_GAMMA_STRENGTH0 = 31.0

mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
EXPORT_CONTRACT = ("figure.svg", "figure.pdf", "figure.tiff")
RASTER_DPI = 600


def initialize() -> float:
    shared.fixed.load_sources()
    shared.matched.install_variant()
    shared.frozen.official._fooof_group_class()
    knees = shared.frozen.load_empirical_knees()
    return float(np.median(knees))


def orthogonal_gaussian(rng: np.random.Generator, n: int, dimensions: int = 4) -> np.ndarray:
    """Return exactly centred perturbations with identity sample covariance."""
    values = rng.normal(size=(n, dimensions))
    values -= values.mean(axis=0, keepdims=True)
    covariance = np.cov(values, rowvar=False, ddof=1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) <= 0:
        raise RuntimeError("Degenerate local perturbation design")
    whitening = eigenvectors @ np.diag(eigenvalues ** -0.5) @ eigenvectors.T
    values = values @ whitening
    if not np.allclose(values.mean(axis=0), 0.0, atol=1e-12):
        raise AssertionError("Perturbations are not centred")
    if not np.allclose(np.cov(values, rowvar=False, ddof=1), np.eye(dimensions), atol=1e-10):
        raise AssertionError("Perturbations are not orthogonal")
    return values


def centre_parameters(display_band: str, knee0_hz: float) -> np.ndarray:
    low, high, _ = shared.researcher.BANDS[display_band]
    center0 = 0.5 * (low + high)
    strength0 = HIGH_GAMMA_STRENGTH0 if display_band == "High gamma" else LOW_BAND_STRENGTH0
    return np.asarray((center0, WIDTH0_HZ, strength0, knee0_hz), dtype=float)


def local_parameters(
    rng: np.random.Generator,
    display_band: str,
    knee0_hz: float,
    n: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centre = centre_parameters(display_band, knee0_hz)
    standardized = orthogonal_gaussian(rng, n)
    sd = epsilon * centre
    parameters = centre[None, :] + standardized * sd[None, :]
    if np.any(parameters[:, 1:] <= 0):
        raise AssertionError("Local perturbation generated a non-positive parameter")
    low, high, _ = shared.researcher.BANDS[display_band]
    if np.any((parameters[:, 0] <= low) | (parameters[:, 0] >= high)):
        raise AssertionError("Local centre perturbation left the target frequency band")
    return parameters, centre, sd


def local_batch(
    rng: np.random.Generator,
    display_band: str,
    knee0_hz: float,
    n: int,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parameters, centre, sd = local_parameters(rng, display_band, knee0_hz, n, epsilon)
    signals: list[np.ndarray] = []
    originals: list[np.ndarray] = []
    for center, width, strength, knee in parameters:
        signal, frequency, log_psd = shared.composite_trial(
            rng, float(center), float(width), float(strength), float(knee)
        )
        signals.append(signal)
        if not np.all(np.diff(frequency) > 0):
            raise AssertionError("Welch frequency grid is not increasing")
        originals.append(np.interp(shared.frozen.base.FREQ, frequency, log_psd))
    irasa_frequency, irasa_power = shared.irasa_aperiodic_batch(np.asarray(signals))
    if not np.all(np.diff(irasa_frequency) > 0):
        raise AssertionError("IRASA frequency grid is not increasing")
    irasa_log = np.log10(np.maximum(irasa_power, np.finfo(float).tiny))
    fractals = np.asarray(
        [np.interp(shared.frozen.base.FREQ, irasa_frequency, row) for row in irasa_log]
    )
    return np.asarray(originals), fractals, parameters, centre, sd


def correlation(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if int(good.sum()) < 8:
        return np.nan
    return float(stats.spearmanr(values[good, 0], values[good, 1]).statistic)


def feature_linearity(predicted: np.ndarray, fitted: np.ndarray) -> dict[str, float]:
    good = np.isfinite(predicted).all(axis=1) & np.isfinite(fitted).all(axis=1)
    if int(good.sum()) < 8:
        return {"p_r": np.nan, "ap_r": np.nan, "p_slope": np.nan, "ap_slope": np.nan}
    result: dict[str, float] = {}
    for index, name in enumerate(("p", "ap")):
        fit = stats.linregress(predicted[good, index], fitted[good, index])
        result[f"{name}_r"] = float(fit.rvalue)
        result[f"{name}_slope"] = float(fit.slope)
    return result


def local_jacobians(
    calibration_original: np.ndarray,
    calibration_fractal: np.ndarray,
    display_band: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate both estimator Jacobians once at the group centre spectrum."""
    mean_original = calibration_original.mean(axis=0)
    mean_fractal = calibration_fractal.mean(axis=0)
    fooof_j = shared.frozen.official_multipeak_influence(mean_original, display_band)
    split_j = np.asarray(
        shared.six.split_weights(
            mean_original[None, :], mean_fractal[None, :], display_band.lower()
        )
    )
    if not np.isfinite(fooof_j).all() or not np.isfinite(split_j).all():
        raise RuntimeError(f"Non-finite centre-point Jacobian in {display_band}")
    return fooof_j, split_j


def project(
    original: np.ndarray,
    fractal: np.ndarray,
    reference_original: np.ndarray,
    reference_fractal: np.ndarray,
    fooof_j: np.ndarray,
    split_j: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    d_original = original - reference_original[None, :]
    d_fractal = fractal - reference_fractal[None, :]
    d_residual = d_original - d_fractal
    fooof_linear = d_original @ fooof_j.T
    split_linear = np.column_stack((d_residual @ split_j[0], d_fractal @ split_j[1]))
    return fooof_linear, split_linear


def run_group(
    repeat: int,
    display_band: str,
    epsilon: float,
    n_calibration: int,
    n_heldout: int,
    seed_offset: int,
) -> tuple[dict[str, float | int | str], dict[str, object]]:
    band_index = list(shared.researcher.BANDS).index(display_band)
    knee0 = float(np.median(shared.frozen.EMPIRICAL_KNEES))
    seed = BASE_SEED + seed_offset + repeat * 1000 + band_index * 100
    calibration = local_batch(
        np.random.default_rng(seed), display_band, knee0, n_calibration, epsilon
    )
    heldout = local_batch(
        np.random.default_rng(seed + 1), display_band, knee0, n_heldout, epsilon
    )
    cal_o, cal_f, cal_params, centre, sd = calibration
    held_o, held_f, held_params, _, _ = heldout
    fooof_j, split_j = local_jacobians(cal_o, cal_f, display_band)
    mean_o, mean_f = cal_o.mean(axis=0), cal_f.mean(axis=0)
    fooof_cal_linear, split_cal_linear = project(
        cal_o, cal_f, mean_o, mean_f, fooof_j, split_j
    )
    fooof_held_linear, split_held_linear = project(
        held_o, held_f, mean_o, mean_f, fooof_j, split_j
    )
    fooof_cal_fit = shared.frozen.official_fit(cal_o, display_band)
    split_cal_fit = shared.six.fit_split_batch(cal_o, cal_f, display_band.lower())
    fooof_held_fit = shared.frozen.official_fit(held_o, display_band)
    split_held_fit = shared.six.fit_split_batch(held_o, held_f, display_band.lower())
    fooof_pred = correlation(fooof_cal_linear)
    split_pred = correlation(split_cal_linear)
    fooof_linearity = feature_linearity(fooof_held_linear, fooof_held_fit)
    split_linearity = feature_linearity(split_held_linear, split_held_fit)
    row: dict[str, float | int | str] = {
        "repeat": repeat,
        "band": display_band.lower(),
        "epsilon": epsilon,
        "fooof_pred": fooof_pred,
        "fooof_emp": correlation(fooof_held_fit),
        "fooof_corrected": shared.six.corrected_corr(fooof_held_fit, fooof_pred),
        "split_pred": split_pred,
        "split_emp": correlation(split_held_fit),
        "fooof_train_valid_n": int(np.isfinite(fooof_cal_fit).all(axis=1).sum()),
        "split_train_valid_n": int(np.isfinite(split_cal_fit).all(axis=1).sum()),
        "fooof_held_valid_n": int(np.isfinite(fooof_held_fit).all(axis=1).sum()),
        "split_held_valid_n": int(np.isfinite(split_held_fit).all(axis=1).sum()),
        "fooof_valid": float(np.isfinite(fooof_held_fit).all(axis=1).mean()),
        "split_valid": float(np.isfinite(split_held_fit).all(axis=1).mean()),
        "true_center_ap_spearman_calibration": float(
            stats.spearmanr(cal_params[:, 0], cal_params[:, 3]).statistic
        ),
        "true_center_ap_spearman_heldout": float(
            stats.spearmanr(held_params[:, 0], held_params[:, 3]).statistic
        ),
        "center0_hz": float(centre[0]),
        "width0_hz": float(centre[1]),
        "strength0": float(centre[2]),
        "knee0_hz": float(centre[3]),
        "center_sd_hz": float(sd[0]),
        "width_sd_hz": float(sd[1]),
        "strength_sd": float(sd[2]),
        "knee_sd_hz": float(sd[3]),
        "n_train": n_calibration,
        "n_heldout": n_heldout,
        **{f"fooof_linearity_{key}": value for key, value in fooof_linearity.items()},
        **{f"split_linearity_{key}": value for key, value in split_linearity.items()},
    }
    state: dict[str, object] = {
        "train_o": cal_o,
        "train_f": cal_f,
        "train_parameters": cal_params,
        "held_o": held_o,
        "held_f": held_f,
        "held_parameters": held_params,
        "influence": fooof_j,
        "split_influence": split_j,
        "fooof_projected": fooof_cal_linear,
        "split_projected": split_cal_linear,
        "fooof": fooof_held_fit,
        "split": split_held_fit,
        "v_fooof": np.cov(fooof_cal_linear.T, ddof=1),
        "v_split": np.cov(split_cal_linear.T, ddof=1),
    }
    return row, state


def pilot_worker(task: tuple[float, int, str]) -> dict[str, float | int | str]:
    epsilon, repeat, display_band = task
    row, _ = run_group(
        repeat, display_band, epsilon, PILOT_N, PILOT_N, seed_offset=40_000_000
    )
    return row


def run_pilot() -> pd.DataFrame:
    tasks = [
        (epsilon, repeat, band)
        for epsilon in PILOT_EPSILONS
        for repeat in range(PILOT_GROUPS)
        for band in shared.researcher.BANDS
    ]
    rows: list[dict[str, float | int | str]] = []
    with mp.get_context("fork").Pool(4) as pool:
        for completed, row in enumerate(pool.imap_unordered(pilot_worker, tasks), 1):
            rows.append(row)
            print(f"local-scale pilot {completed}/{len(tasks)}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["epsilon", "repeat", "band"])
    frame.to_csv(OUTPUT / "local_scale_pilot_groups.csv", index=False)
    summaries = []
    for epsilon, group in frame.groupby("epsilon"):
        summaries.append(
            {
                "epsilon": epsilon,
                "mean_fooof_abs_rho_error": float(np.nanmean(abs(group.fooof_pred - group.fooof_emp))),
                "mean_split_abs_rho_error": float(np.nanmean(abs(group.split_pred - group.split_emp))),
                "median_fooof_p_linearity_r": float(np.nanmedian(group.fooof_linearity_p_r)),
                "median_fooof_ap_linearity_r": float(np.nanmedian(group.fooof_linearity_ap_r)),
                "median_split_p_linearity_r": float(np.nanmedian(group.split_linearity_p_r)),
                "median_split_ap_linearity_r": float(np.nanmedian(group.split_linearity_ap_r)),
                "mean_fooof_valid": float(group.fooof_valid.mean()),
                "mean_split_valid": float(group.split_valid.mean()),
            }
        )
    summary = pd.DataFrame(summaries).sort_values("epsilon")
    summary.to_csv(OUTPUT / "local_scale_pilot_summary.csv", index=False)
    print(summary.to_string(index=False), flush=True)
    return summary


def full_worker(task: tuple[int, str, float]) -> dict[str, float | int | str]:
    repeat, display_band, epsilon = task
    row, _ = run_group(
        repeat, display_band, epsilon, N_CALIBRATION, N_HELDOUT, seed_offset=80_000_000
    )
    return row


def reconstruct(repeat: int, display_band: str, epsilon: float) -> dict[str, object]:
    _, state = run_group(
        repeat, display_band, epsilon, N_CALIBRATION, N_HELDOUT, seed_offset=80_000_000
    )
    fooof = np.asarray(state["fooof"])
    split = np.asarray(state["split"])
    both_good = np.isfinite(fooof).all(axis=1) & np.isfinite(split).all(axis=1)
    if not np.any(both_good):
        raise RuntimeError("No representative held-out trial valid for both estimators")
    target = np.nanmedian(fooof[both_good, 0])
    indices = np.flatnonzero(both_good)
    component_index = int(indices[np.argmin(abs(fooof[both_good, 0] - target))])
    components = shared.frozen.official_fit_components(
        np.asarray(state["held_o"])[component_index]
    )
    state.update(
        {
            "repeat": repeat,
            "band": display_band.lower(),
            "held_c": np.asarray(state["held_parameters"])[:, 0],
            "held_knee": np.asarray(state["held_parameters"])[:, 3],
            "train_c": np.asarray(state["train_parameters"])[:, 0],
            "train_knee": np.asarray(state["train_parameters"])[:, 3],
            "component_trial_index": component_index,
            "fooof_component_frequency": components["frequency"],
            "fooof_source_psd": components["source"],
            "fooof_aperiodic_fit": components["aperiodic"],
            "fooof_periodic_fit": components["periodic"],
        }
    )
    return state


def validation_metrics(groups: pd.DataFrame) -> dict[str, object]:
    metrics: dict[str, object] = {}
    for method in ("fooof", "split"):
        good = np.isfinite(groups[f"{method}_pred"]) & np.isfinite(groups[f"{method}_emp"])
        fit = stats.linregress(groups.loc[good, f"{method}_pred"], groups.loc[good, f"{method}_emp"])
        metrics[method] = {
            "n_groups": int(good.sum()),
            "pearson_r": float(fit.rvalue),
            "ols_slope": float(fit.slope),
            "R2": float(fit.rvalue**2),
            "mean_absolute_error": float(
                np.mean(abs(groups.loc[good, f"{method}_pred"] - groups.loc[good, f"{method}_emp"]))
            ),
        }
    return metrics


def run_full(epsilon: float) -> Path:
    checkpoint = OUTPUT / f"full_groups_epsilon_{epsilon:.4f}.csv"
    if checkpoint.exists():
        groups = pd.read_csv(checkpoint)
    else:
        tasks = [
            (repeat, band, epsilon)
            for repeat in range(N_GROUPS)
            for band in shared.researcher.BANDS
        ]
        rows: list[dict[str, float | int | str]] = []
        with mp.get_context("fork").Pool(4) as pool:
            for completed, row in enumerate(pool.imap_unordered(full_worker, tasks), 1):
                rows.append(row)
                pd.DataFrame(rows).sort_values(["repeat", "band"]).to_csv(
                    OUTPUT / "full_checkpoint_partial.csv", index=False
                )
                print(f"local first-order full groups {completed}/{len(tasks)}", flush=True)
        groups = pd.DataFrame(rows).sort_values(["repeat", "band"]).reset_index(drop=True)
        groups.to_csv(checkpoint, index=False)
    groups.to_csv(OUTPUT / "all_group_results.csv", index=False)
    shared.six.aggregate(groups).to_csv(OUTPUT / "cross_band_summary.csv", index=False)
    representative_repeat = shared.frozen.choose_representative(groups)
    representative = reconstruct(representative_repeat, "High gamma", epsilon)
    svg, metrics = renderer.render_3x3(
        groups,
        representative,
        OUTPUT,
        STEM,
        "FOOOF local first-order theory",
        "SPLIT local first-order theory",
        rasterize_pdf_linework=True,
    )
    metrics.update(
        {
            "epsilon_relative_sd": epsilon,
            "central_parameters_by_band": {
                band: centre_parameters(band, float(np.median(shared.frozen.EMPIRICAL_KNEES))).tolist()
                for band in shared.researcher.BANDS
            },
            "design": "four independent exactly centred perturbations with identity sample covariance",
            "jacobian": "one estimator Jacobian evaluated at the calibration mean spectrum for each band-by-group condition",
            "calibration_trials_per_group": N_CALIBRATION,
            "heldout_trials_per_group": N_HELDOUT,
            "groups_per_band": N_GROUPS,
            "validation_by_method": validation_metrics(groups),
        }
    )
    (OUTPUT / f"{STEM}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return svg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    initialize()
    if args.pilot:
        run_pilot()
    if args.full:
        print(run_full(args.epsilon))
    if not args.pilot and not args.full:
        parser.error("Choose --pilot and/or --full")


if __name__ == "__main__":
    main()
