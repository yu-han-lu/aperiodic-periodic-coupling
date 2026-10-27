from __future__ import annotations

"""Pilot one covariance path that preserves true centre--knee independence."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import build_fooof_split_local_perturbation_3x3 as local
from pilot_local_covariance_sweep import exact_correlated_gaussian


OUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_targeted_covariance_path_pilot_2026-09-13"
)
DISPLAY_BAND = "High gamma"
EPSILON = 0.02
N_CALIBRATION = 384
N_HELDOUT = 384
T_VALUES = np.linspace(-0.72, 0.72, 7)
PAIRS = ((0, 1), (0, 2), (1, 2), (1, 3), (2, 3))
D0 = np.asarray((-0.40112554, -0.34984947, -0.00840086, 0.03284833, 0.00829349))
DIRECTION = np.asarray((0.65846063, -0.75140925, -0.01501410, 0.03748573, -0.01353261))


def covariance_at(t_value: float) -> np.ndarray:
    covariance = np.eye(4)
    for value, (left, right) in zip(D0 + t_value * DIRECTION, PAIRS):
        covariance[left, right] = covariance[right, left] = value
    if covariance[0, 3] != 0:
        raise AssertionError("True centre--knee covariance must remain zero")
    if np.min(np.linalg.eigvalsh(covariance)) < 0.05:
        raise AssertionError("Covariance path is too close to singularity")
    return covariance


def spectral_batch(
    rng: np.random.Generator, n: int, covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knee0 = float(np.median(local.shared.frozen.EMPIRICAL_KNEES))
    centre = local.centre_parameters(DISPLAY_BAND, knee0)
    sd = EPSILON * centre
    standardized = exact_correlated_gaussian(rng, n, covariance)
    parameters = centre[None, :] + standardized * sd[None, :]
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
    return np.asarray(originals), fractals, parameters


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    local.initialize()
    reference = local.reconstruct(3, DISPLAY_BAND, 0.01)
    fooof_j = np.asarray(reference["influence"])
    split_j = np.asarray(reference["split_influence"])
    rows = []
    for condition_index, t_value in enumerate(T_VALUES):
        covariance = covariance_at(float(t_value))
        seed = local.BASE_SEED + 140_000_000
        cal_o, cal_f, cal_params = spectral_batch(
            np.random.default_rng(seed), N_CALIBRATION, covariance
        )
        held_o, held_f, held_params = spectral_batch(
            np.random.default_rng(seed + 1), N_HELDOUT, covariance
        )
        fooof_cal, split_cal = local.project(
            cal_o,
            cal_f,
            cal_o.mean(axis=0),
            cal_f.mean(axis=0),
            fooof_j,
            split_j,
        )
        fooof_held = local.shared.frozen.official_fit(held_o, DISPLAY_BAND)
        split_held = local.shared.six.fit_split_batch(
            held_o, held_f, DISPLAY_BAND.lower()
        )
        row = {
            "condition": condition_index,
            "t": float(t_value),
            "minimum_eigenvalue": float(np.min(np.linalg.eigvalsh(covariance))),
            "fooof_theory": local.correlation(fooof_cal),
            "fooof_empirical": local.correlation(fooof_held),
            "split_theory": local.correlation(split_cal),
            "split_empirical": local.correlation(split_held),
            "fooof_valid": float(np.isfinite(fooof_held).all(axis=1).mean()),
            "split_valid": float(np.isfinite(split_held).all(axis=1).mean()),
            "true_center_knee_cal": float(
                stats.spearmanr(cal_params[:, 0], cal_params[:, 3]).statistic
            ),
            "true_center_knee_held": float(
                stats.spearmanr(held_params[:, 0], held_params[:, 3]).statistic
            ),
            "covariance": json.dumps(covariance.tolist()),
        }
        rows.append(row)
        print(json.dumps(row), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "targeted_path_results.csv", index=False)
    print(frame.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
