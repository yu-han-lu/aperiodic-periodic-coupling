from __future__ import annotations

"""Test total-covariance theory on the selected broad high-gamma distribution."""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import build_fooof_split_local_perturbation_3x3 as local
from pilot_four_parameter_plus_realization_theory import (
    condition_means_and_seed_covariance,
    fit_taylor,
    replicated_spectra,
)


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_split_broad_high_gamma_total_covariance_pilot_2026-09-13"
)
DISPLAY_BAND = "High gamma"
STRENGTH_RANGE = (28.0, 34.0)
N_REPEATS = 4
N_CALIBRATION_CONDITIONS = 48
N_CALIBRATION_REALIZATIONS = 6
N_HELDOUT_CONDITIONS = 96
N_HELDOUT_REALIZATIONS = 3
N_MONTE_CARLO = 50_000
BASE_SEED = local.BASE_SEED + 230_000_000


def sample_parameters(rng: np.random.Generator, n: int) -> np.ndarray:
    low, high, _ = local.shared.researcher.BANDS[DISPLAY_BAND]
    centers = rng.uniform(low, high, n)
    widths = rng.uniform(1.5, 6.0, n)
    strengths = rng.uniform(*STRENGTH_RANGE, n)
    knees = local.shared.frozen.sample_empirical_knees_independent_of_centers(
        rng, centers
    )
    # Consume the same index draw as the selected generator before the signal
    # innovations, preserving its random-stream structure.
    rng.integers(0, len(local.shared.fixed.FRACTAL_RESIDUALS), n)
    return np.column_stack([centers, widths, strengths, knees])


def fit_estimators(original: np.ndarray, fractal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "FOOOF": local.shared.frozen.official_fit(original, DISPLAY_BAND),
        "SPLIT": local.shared.six.fit_split_batch(
            original, fractal, DISPLAY_BAND.lower()
        ),
    }


def correlation(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if int(good.sum()) < 8:
        return np.nan
    return float(stats.spearmanr(values[good, 0], values[good, 1]).statistic)


def propagate(
    rng: np.random.Generator,
    held_standardized: np.ndarray,
    polynomial,
    model,
    residuals: np.ndarray,
) -> tuple[float, float]:
    conditional = model.predict(polynomial.transform(held_standardized))
    parameter_only = correlation(conditional)
    draw = rng.integers(0, len(held_standardized), N_MONTE_CARLO)
    residual_draw = rng.integers(0, len(residuals), N_MONTE_CARLO)
    total = conditional[draw] + residuals[residual_draw]
    return parameter_only, correlation(total)


def run_repeat(repeat: int) -> list[dict[str, float | int | str]]:
    seed = BASE_SEED + repeat * 100_000
    calibration_parameters = sample_parameters(
        np.random.default_rng(seed), N_CALIBRATION_CONDITIONS
    )
    held_parameters = sample_parameters(
        np.random.default_rng(seed + 1), N_HELDOUT_CONDITIONS
    )
    location = calibration_parameters.mean(axis=0)
    scale = calibration_parameters.std(axis=0, ddof=1)
    cal_standardized = (calibration_parameters - location) / scale
    held_standardized = (held_parameters - location) / scale
    cal_o, cal_f, cal_index = replicated_spectra(
        np.random.default_rng(seed + 2),
        calibration_parameters,
        N_CALIBRATION_REALIZATIONS,
    )
    held_o, held_f, _ = replicated_spectra(
        np.random.default_rng(seed + 3),
        held_parameters,
        N_HELDOUT_REALIZATIONS,
    )
    cal_fit = fit_estimators(cal_o, cal_f)
    held_fit = fit_estimators(held_o, held_f)
    rows: list[dict[str, float | int | str]] = []
    for estimator in ("FOOOF", "SPLIT"):
        means, seed_covariance, counts, residuals = condition_means_and_seed_covariance(
            cal_fit[estimator], cal_index, N_CALIBRATION_CONDITIONS
        )
        for order in (1, 2):
            polynomial, model = fit_taylor(
                cal_standardized, means, counts, order
            )
            parameter_rho, total_rho = propagate(
                np.random.default_rng(seed + 100 + order + (0 if estimator == "FOOOF" else 10)),
                held_standardized,
                polynomial,
                model,
                residuals,
            )
            rows.append(
                {
                    "repeat": repeat,
                    "estimator": estimator,
                    "order": order,
                    "parameter_only_rho": parameter_rho,
                    "parameter_plus_realization_rho": total_rho,
                    "empirical_rho": correlation(held_fit[estimator]),
                    "abs_total_error": abs(total_rho - correlation(held_fit[estimator])),
                    "valid_fraction": float(
                        np.isfinite(held_fit[estimator]).all(axis=1).mean()
                    ),
                    "seed_cross_covariance": float(seed_covariance[0, 1]),
                    "true_center_knee_rho": float(
                        stats.spearmanr(held_parameters[:, 0], held_parameters[:, 3]).statistic
                    ),
                }
            )
    return rows


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    local.initialize()
    rows: list[dict[str, float | int | str]] = []
    with mp.get_context("fork").Pool(4) as pool:
        for completed, result in enumerate(pool.imap_unordered(run_repeat, range(N_REPEATS)), 1):
            rows.extend(result)
            print(f"broad high-gamma repeats {completed}/{N_REPEATS}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["repeat", "estimator", "order"])
    frame.to_csv(OUTPUT / "pilot_results.csv", index=False)
    summary = (
        frame.groupby(["estimator", "order"], as_index=False)
        .agg(
            theory_rho_mean=("parameter_plus_realization_rho", "mean"),
            empirical_rho_mean=("empirical_rho", "mean"),
            empirical_rho_sd=("empirical_rho", "std"),
            mean_abs_error=("abs_total_error", "mean"),
            valid_fraction=("valid_fraction", "mean"),
            true_center_knee_rho_mean=("true_center_knee_rho", "mean"),
        )
    )
    summary.to_csv(OUTPUT / "pilot_summary.csv", index=False)
    criteria = {
        "fooof_spurious_rho_target": 0.15,
        "split_absolute_rho_target": 0.08,
        "theory_mean_absolute_error_target": 0.08,
    }
    (OUTPUT / "pilot_metadata.json").write_text(
        json.dumps(
            {
                "parameter_distribution": {
                    "center_hz": "Uniform(60,150)",
                    "width_hz": "Uniform(1.5,6.0)",
                    "strength": "Uniform(28,34)",
                    "knee_hz": "empirical distribution, independently permuted relative to center",
                },
                "criteria": criteria,
                "n_calibration_conditions": N_CALIBRATION_CONDITIONS,
                "n_calibration_realizations": N_CALIBRATION_REALIZATIONS,
                "n_heldout_conditions": N_HELDOUT_CONDITIONS,
                "n_heldout_realizations": N_HELDOUT_REALIZATIONS,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False), flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
