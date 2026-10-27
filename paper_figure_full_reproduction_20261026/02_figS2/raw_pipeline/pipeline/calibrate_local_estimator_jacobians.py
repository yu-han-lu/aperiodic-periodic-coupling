from __future__ import annotations

"""Estimate one held-out-safe local estimator Jacobian at the fixed centre."""

import json
from pathlib import Path

import numpy as np

import build_fooof_split_local_perturbation_3x3 as local
import pilot_anisotropic_nuisance_covariance as anisotropic


OUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_local_covariance_calibration_2026-09-13"
)
DISPLAY_BAND = "High gamma"
EPSILON = 0.02
N_CALIBRATION = 1024
SEED = local.BASE_SEED + 160_000_000


def estimator_linear_model(
    standardized_parameters: np.ndarray, fitted: np.ndarray
) -> dict[str, object]:
    good = np.isfinite(fitted).all(axis=1)
    design = np.column_stack((np.ones(int(good.sum())), standardized_parameters[good]))
    coefficients = np.linalg.lstsq(design, fitted[good], rcond=None)[0]
    predicted = design @ coefficients
    residual = fitted[good] - predicted
    return {
        "valid_n": int(good.sum()),
        "valid_fraction": float(good.mean()),
        "intercept": coefficients[0].tolist(),
        "jacobian_standardized": coefficients[1:].T.tolist(),
        "residual_covariance": np.cov(residual.T, ddof=1).tolist(),
        "feature_covariance": np.cov(fitted[good].T, ddof=1).tolist(),
        "feature_r2": [
            float(1.0 - np.var(residual[:, column], ddof=1) / np.var(fitted[good, column], ddof=1))
            for column in range(2)
        ],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local.initialize()
    original, fractal, parameters, centre, sd = anisotropic.spectral_batch(
        np.random.default_rng(SEED), N_CALIBRATION, np.eye(4)
    )
    standardized = (parameters - centre[None, :]) / sd[None, :]
    fooof = local.shared.frozen.official_fit(original, DISPLAY_BAND)
    split = local.shared.six.fit_split_batch(original, fractal, DISPLAY_BAND.lower())
    result = {
        "band": DISPLAY_BAND.lower(),
        "relative_sd": anisotropic.RELATIVE_SD.tolist(),
        "n_calibration": N_CALIBRATION,
        "centre": centre.tolist(),
        "physical_sd": sd.tolist(),
        "parameter_order": ["peak_center", "peak_width", "peak_strength", "ou_knee"],
        "sample_parameter_covariance": np.cov(standardized.T, ddof=1).tolist(),
        "fooof": estimator_linear_model(standardized, fooof),
        "split": estimator_linear_model(standardized, split),
    }
    path = OUT / "high_gamma_anisotropic_local_estimator_calibration.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
