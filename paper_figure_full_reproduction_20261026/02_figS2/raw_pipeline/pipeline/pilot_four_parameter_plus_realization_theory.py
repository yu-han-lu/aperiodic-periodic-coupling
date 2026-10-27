from __future__ import annotations

"""Pilot the law-of-total-covariance correction for FOOOF and SPLIT.

The four physical input parameters vary locally around the selected high-gamma
centre.  Multiple independent time-domain realizations are generated at every
identical parameter vector.  Calibration estimates (i) the first- or
second-order parameter-to-estimator map and (ii) the within-parameter
realization covariance.  Independent held-out simulations test whether adding
the latter improves the predicted P--A correlation.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures


HERE = Path(__file__).resolve().parent
PROJECT_CODE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project/experiments/pap_theory_validation/pap_validation"
for source in (HERE, PROJECT_CODE):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

import build_fooof_split_local_perturbation_3x3 as local  # noqa: E402
from pilot_local_covariance_sweep import exact_correlated_gaussian  # noqa: E402
from pilot_targeted_covariance_path import covariance_at, T_VALUES  # noqa: E402


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_split_four_parameter_plus_realization_pilot_eps005_2026-09-13"
)
DISPLAY_BAND = "High gamma"
EPSILON = 0.05
N_CALIBRATION_CONDITIONS = 48
N_CALIBRATION_REALIZATIONS = 12
N_HELDOUT_CONDITIONS = 128
N_HELDOUT_REALIZATIONS = 4
BASE_SEED = local.BASE_SEED + 180_000_000


def parameters_for_covariance(
    rng: np.random.Generator,
    n: int,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    knee0 = float(np.median(local.shared.frozen.EMPIRICAL_KNEES))
    centre = local.centre_parameters(DISPLAY_BAND, knee0)
    sd = EPSILON * centre
    standardized = exact_correlated_gaussian(rng, n, covariance)
    parameters = centre[None, :] + standardized * sd[None, :]
    return parameters, standardized, centre, sd


def replicated_spectra(
    rng: np.random.Generator,
    parameters: np.ndarray,
    n_realizations: int,
    chunk_size: int = 64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate complete s(t), Welch and IRASA outputs in bounded-memory chunks."""
    expanded = np.repeat(parameters, n_realizations, axis=0)
    condition_index = np.repeat(np.arange(len(parameters)), n_realizations)
    original_chunks: list[np.ndarray] = []
    fractal_chunks: list[np.ndarray] = []
    for start in range(0, len(expanded), chunk_size):
        stop = min(start + chunk_size, len(expanded))
        signals: list[np.ndarray] = []
        originals: list[np.ndarray] = []
        for center, width, strength, knee in expanded[start:stop]:
            time_series, frequency, log_psd = local.shared.composite_trial(
                rng, float(center), float(width), float(strength), float(knee)
            )
            signals.append(time_series)
            if not np.all(np.diff(frequency) > 0):
                raise AssertionError("Welch frequency grid is not increasing")
            originals.append(
                np.interp(local.shared.frozen.base.FREQ, frequency, log_psd)
            )
        irasa_frequency, irasa_power = local.shared.irasa_aperiodic_batch(
            np.asarray(signals)
        )
        if not np.all(np.diff(irasa_frequency) > 0):
            raise AssertionError("IRASA frequency grid is not increasing")
        irasa_log = np.log10(np.maximum(irasa_power, np.finfo(float).tiny))
        fractals = np.asarray(
            [
                np.interp(local.shared.frozen.base.FREQ, irasa_frequency, row)
                for row in irasa_log
            ]
        )
        original_chunks.append(np.asarray(originals))
        fractal_chunks.append(fractals)
    return (
        np.vstack(original_chunks),
        np.vstack(fractal_chunks),
        condition_index,
    )


def fit_estimators(original: np.ndarray, fractal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "FOOOF": local.shared.frozen.official_fit(original, DISPLAY_BAND),
        "SPLIT": local.shared.six.fit_split_batch(
            original, fractal, DISPLAY_BAND.lower()
        ),
    }


def condition_means_and_seed_covariance(
    fitted: np.ndarray,
    condition_index: np.ndarray,
    n_conditions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means = np.full((n_conditions, 2), np.nan)
    counts = np.zeros(n_conditions, dtype=int)
    residuals: list[np.ndarray] = []
    residual_condition: list[np.ndarray] = []
    for condition in range(n_conditions):
        rows = fitted[condition_index == condition]
        good = np.isfinite(rows).all(axis=1)
        rows = rows[good]
        counts[condition] = len(rows)
        if len(rows):
            means[condition] = rows.mean(axis=0)
        if len(rows) >= 2:
            residuals.append(rows - rows.mean(axis=0, keepdims=True))
            residual_condition.append(np.full(len(rows), condition, dtype=int))
    if not residuals:
        return means, np.full((2, 2), np.nan), counts, np.empty((0, 2))
    residual = np.vstack(residuals)
    degrees_of_freedom = sum(max(count - 1, 0) for count in counts)
    seed_covariance = residual.T @ residual / max(degrees_of_freedom, 1)
    return means, seed_covariance, counts, residual


def fit_taylor(
    standardized: np.ndarray,
    conditional_means: np.ndarray,
    counts: np.ndarray,
    order: int,
) -> tuple[PolynomialFeatures, Ridge]:
    good = np.isfinite(conditional_means).all(axis=1) & (counts >= 2)
    polynomial = PolynomialFeatures(degree=order, include_bias=True)
    design = polynomial.fit_transform(standardized[good])
    model = Ridge(alpha=1e-4, fit_intercept=False)
    model.fit(design, conditional_means[good], sample_weight=counts[good])
    return polynomial, model


def parameter_covariance_from_taylor(
    polynomial: PolynomialFeatures,
    model: Ridge,
    covariance: np.ndarray,
    order: int,
) -> np.ndarray:
    """Evaluate J Sigma J' + 1/2 tr(Ha Sigma Hb Sigma)."""
    powers = polynomial.powers_
    coefficients = np.asarray(model.coef_)
    jacobian = np.zeros((2, 4))
    hessian = np.zeros((2, 4, 4))
    for term, power in enumerate(powers):
        degree = int(power.sum())
        if degree == 1:
            variable = int(np.flatnonzero(power)[0])
            jacobian[:, variable] = coefficients[:, term]
        elif order >= 2 and degree == 2:
            variables = np.flatnonzero(power)
            if len(variables) == 1:
                variable = int(variables[0])
                hessian[:, variable, variable] = 2.0 * coefficients[:, term]
            else:
                left, right = map(int, variables)
                hessian[:, left, right] = coefficients[:, term]
                hessian[:, right, left] = coefficients[:, term]
    result = jacobian @ covariance @ jacobian.T
    if order >= 2:
        for left in range(2):
            for right in range(2):
                result[left, right] += 0.5 * np.trace(
                    hessian[left] @ covariance @ hessian[right] @ covariance
                )
    return result


def spearman_from_covariance(covariance: np.ndarray) -> float:
    denominator = np.sqrt(covariance[0, 0] * covariance[1, 1])
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    pearson = float(np.clip(covariance[0, 1] / denominator, -0.999, 0.999))
    return float(6.0 / np.pi * np.arcsin(pearson / 2.0))


def empirical_spearman(fitted: np.ndarray) -> float:
    good = np.isfinite(fitted).all(axis=1)
    if int(good.sum()) < 8:
        return np.nan
    return float(stats.spearmanr(fitted[good, 0], fitted[good, 1]).statistic)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    local.initialize()

    cal_parameters, cal_standardized, centre, sd = parameters_for_covariance(
        np.random.default_rng(BASE_SEED),
        N_CALIBRATION_CONDITIONS,
        np.eye(4),
    )
    cal_o, cal_f, cal_condition = replicated_spectra(
        np.random.default_rng(BASE_SEED + 1),
        cal_parameters,
        N_CALIBRATION_REALIZATIONS,
    )
    calibration_fit = fit_estimators(cal_o, cal_f)

    calibration: dict[str, dict[str, object]] = {}
    for estimator, values in calibration_fit.items():
        means, seed_covariance, counts, residuals = condition_means_and_seed_covariance(
            values, cal_condition, N_CALIBRATION_CONDITIONS
        )
        calibration[estimator] = {
            "means": means,
            "seed_covariance": seed_covariance,
            "counts": counts,
            "residuals": residuals,
            "models": {
                order: fit_taylor(cal_standardized, means, counts, order)
                for order in (1, 2)
            },
        }

    rows: list[dict[str, float | int | str]] = []
    for condition_number, t_value in enumerate(T_VALUES):
        target_covariance = covariance_at(float(t_value))
        held_parameters, _, _, _ = parameters_for_covariance(
            np.random.default_rng(BASE_SEED + 100 + condition_number),
            N_HELDOUT_CONDITIONS,
            target_covariance,
        )
        held_o, held_f, _ = replicated_spectra(
            np.random.default_rng(BASE_SEED + 200 + condition_number),
            held_parameters,
            N_HELDOUT_REALIZATIONS,
        )
        held_fit = fit_estimators(held_o, held_f)
        for estimator, values in held_fit.items():
            seed_covariance = np.asarray(calibration[estimator]["seed_covariance"])
            for order in (1, 2):
                polynomial, model = calibration[estimator]["models"][order]
                parameter_covariance = parameter_covariance_from_taylor(
                    polynomial, model, target_covariance, order
                )
                total_covariance = parameter_covariance + seed_covariance
                rows.append(
                    {
                        "condition": condition_number,
                        "t": float(t_value),
                        "estimator": estimator,
                        "order": order,
                        "parameter_only_rho": spearman_from_covariance(
                            parameter_covariance
                        ),
                        "parameter_plus_seed_rho": spearman_from_covariance(
                            total_covariance
                        ),
                        "empirical_rho": empirical_spearman(values),
                        "valid_n": int(np.isfinite(values).all(axis=1).sum()),
                        "valid_fraction": float(
                            np.isfinite(values).all(axis=1).mean()
                        ),
                        "parameter_cross_covariance": float(
                            parameter_covariance[0, 1]
                        ),
                        "seed_cross_covariance": float(seed_covariance[0, 1]),
                        "parameter_peak_variance": float(parameter_covariance[0, 0]),
                        "seed_peak_variance": float(seed_covariance[0, 0]),
                        "parameter_ap_variance": float(parameter_covariance[1, 1]),
                        "seed_ap_variance": float(seed_covariance[1, 1]),
                    }
                )
        print(f"held-out covariance condition {condition_number + 1}/{len(T_VALUES)}", flush=True)

    frame = pd.DataFrame(rows)
    frame.to_csv(OUTPUT / "pilot_condition_results.csv", index=False)
    summary = (
        frame.assign(
            parameter_only_error=lambda x: abs(x.parameter_only_rho - x.empirical_rho),
            parameter_plus_seed_error=lambda x: abs(
                x.parameter_plus_seed_rho - x.empirical_rho
            ),
        )
        .groupby(["estimator", "order"], as_index=False)
        .agg(
            parameter_only_mae=("parameter_only_error", "mean"),
            parameter_plus_seed_mae=("parameter_plus_seed_error", "mean"),
            empirical_rho_min=("empirical_rho", "min"),
            empirical_rho_max=("empirical_rho", "max"),
            mean_valid_fraction=("valid_fraction", "mean"),
            mean_seed_cross_covariance=("seed_cross_covariance", "mean"),
            mean_seed_peak_variance=("seed_peak_variance", "mean"),
            mean_seed_ap_variance=("seed_ap_variance", "mean"),
        )
    )
    summary.to_csv(OUTPUT / "pilot_summary.csv", index=False)
    metadata = {
        "band": DISPLAY_BAND,
        "epsilon": EPSILON,
        "centre": centre.tolist(),
        "physical_sd": sd.tolist(),
        "parameter_order": ["peak_center", "peak_width", "peak_strength", "ou_knee"],
        "n_calibration_conditions": N_CALIBRATION_CONDITIONS,
        "n_calibration_realizations": N_CALIBRATION_REALIZATIONS,
        "n_heldout_conditions_per_covariance": N_HELDOUT_CONDITIONS,
        "n_heldout_realizations": N_HELDOUT_REALIZATIONS,
        "covariance_identity": (
            "Cov(Y)=Cov_theta(E[Y|theta])+E_theta[Cov(Y|theta)]"
        ),
    }
    (OUTPUT / "pilot_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(OUTPUT, flush=True)


if __name__ == "__main__":
    main()
