from __future__ import annotations

"""Test a local centre--width covariance sweep with small true P/AP changes."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import build_fooof_split_local_perturbation_3x3 as local
from pilot_local_covariance_sweep import exact_correlated_gaussian


OUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_anisotropic_covariance_pilot_2026-09-13"
)
DISPLAY_BAND = "High gamma"
N_CALIBRATION = 384
N_HELDOUT = 384
RHO_VALUES = (-0.9, -0.45, 0.0, 0.45, 0.9)
RELATIVE_SD = np.asarray((0.02, 0.20, 0.05, 0.02))


def covariance_at(rho: float) -> np.ndarray:
    covariance = np.eye(4)
    covariance[0, 1] = covariance[1, 0] = rho
    if covariance[0, 3] != 0:
        raise AssertionError("True centre--knee covariance must remain zero")
    return covariance


def spectral_batch(
    rng: np.random.Generator, n: int, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    knee0 = float(np.median(local.shared.frozen.EMPIRICAL_KNEES))
    centre = local.centre_parameters(DISPLAY_BAND, knee0)
    sd = RELATIVE_SD * centre
    standardized = exact_correlated_gaussian(rng, n, covariance)
    parameters = centre[None, :] + standardized * sd[None, :]
    if np.any(parameters[:, 1:] <= 0):
        raise AssertionError("Non-positive local parameter")
    signals = []
    originals = []
    for center, width, strength, knee in parameters:
        signal, frequency, log_psd = local.shared.composite_trial(
            rng, float(center), float(width), float(strength), float(knee)
        )
        signals.append(signal)
        originals.append(np.interp(local.shared.frozen.base.FREQ, frequency, log_psd))
    irasa_frequency, irasa_power = local.shared.irasa_aperiodic_batch(np.asarray(signals))
    fractals = np.asarray(
        [
            np.interp(
                local.shared.frozen.base.FREQ,
                irasa_frequency,
                np.log10(np.maximum(row, np.finfo(float).tiny)),
            )
            for row in irasa_power
        ]
    )
    return np.asarray(originals), fractals, parameters, centre, sd


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local.initialize()
    reference = local.reconstruct(3, DISPLAY_BAND, 0.01)
    fooof_j = np.asarray(reference["influence"])
    split_j = np.asarray(reference["split_influence"])
    rows = []
    for condition, rho in enumerate(RHO_VALUES):
        covariance = covariance_at(rho)
        seed = local.BASE_SEED + 180_000_000
        cal_o, cal_f, cal_params, centre, sd = spectral_batch(
            np.random.default_rng(seed), N_CALIBRATION, covariance
        )
        held_o, held_f, held_params, _, _ = spectral_batch(
            np.random.default_rng(seed + 1), N_HELDOUT, covariance
        )
        fooof_cal_linear, split_cal_linear = local.project(
            cal_o, cal_f, cal_o.mean(axis=0), cal_f.mean(axis=0), fooof_j, split_j
        )
        fooof_held = local.shared.frozen.official_fit(held_o, DISPLAY_BAND)
        split_held = local.shared.six.fit_split_batch(
            held_o, held_f, DISPLAY_BAND.lower()
        )
        row = {
            "condition": condition,
            "rho_center_width": rho,
            "fooof_theory": local.correlation(fooof_cal_linear),
            "fooof_empirical": local.correlation(fooof_held),
            "split_theory": local.correlation(split_cal_linear),
            "split_empirical": local.correlation(split_held),
            "fooof_valid": float(np.isfinite(fooof_held).all(axis=1).mean()),
            "split_valid": float(np.isfinite(split_held).all(axis=1).mean()),
            "true_center_knee_cal": float(
                stats.spearmanr(cal_params[:, 0], cal_params[:, 3]).statistic
            ),
            "true_center_knee_held": float(
                stats.spearmanr(held_params[:, 0], held_params[:, 3]).statistic
            ),
            "centre": json.dumps(centre.tolist()),
            "physical_sd": json.dumps(sd.tolist()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "anisotropic_center_width_results.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
