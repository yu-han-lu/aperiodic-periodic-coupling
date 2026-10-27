from __future__ import annotations

"""Screen local covariance families for estimator-induced P--AP coupling."""

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import build_fooof_split_local_perturbation_3x3 as local


OUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_local_covariance_pilot_2026-09-13"
)
DISPLAY_BAND = "High gamma"
EPSILON = 0.05
N_CALIBRATION = 192
N_HELDOUT = 192
RHO_VALUES = (-0.8, -0.4, 0.0, 0.4, 0.8)
PAIR_FAMILIES = {
    "center_strength": (0, 2),
    "center_width": (0, 1),
    "strength_knee": (2, 3),
    "width_knee": (1, 3),
}
REFERENCE_FOOOF_J: np.ndarray | None = None
REFERENCE_SPLIT_J: np.ndarray | None = None


def exact_correlated_gaussian(
    rng: np.random.Generator, n: int, covariance: np.ndarray
) -> np.ndarray:
    base = local.orthogonal_gaussian(rng, n)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if np.min(eigenvalues) <= 1e-8:
        raise ValueError("Target covariance is not positive definite")
    square_root = eigenvectors @ np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    values = base @ square_root
    if not np.allclose(np.cov(values, rowvar=False, ddof=1), covariance, atol=1e-10):
        raise AssertionError("Target covariance was not reproduced")
    return values


def covariance_matrix(pair: tuple[int, int], rho: float) -> np.ndarray:
    covariance = np.eye(4)
    covariance[pair[0], pair[1]] = rho
    covariance[pair[1], pair[0]] = rho
    return covariance


def parameter_batch(
    rng: np.random.Generator,
    n: int,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    knee0 = float(np.median(local.shared.frozen.EMPIRICAL_KNEES))
    centre = local.centre_parameters(DISPLAY_BAND, knee0)
    sd = EPSILON * centre
    standardized = exact_correlated_gaussian(rng, n, covariance)
    parameters = centre[None, :] + standardized * sd[None, :]
    return parameters, centre, sd


def spectral_batch(
    rng: np.random.Generator,
    n: int,
    covariance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parameters, centre, sd = parameter_batch(rng, n, covariance)
    signals = []
    originals = []
    for center, width, strength, knee in parameters:
        signal, frequency, log_psd = local.shared.composite_trial(
            rng, float(center), float(width), float(strength), float(knee)
        )
        signals.append(signal)
        originals.append(np.interp(local.shared.frozen.base.FREQ, frequency, log_psd))
    irasa_frequency, irasa_power = local.shared.irasa_aperiodic_batch(np.asarray(signals))
    irasa_log = np.log10(np.maximum(irasa_power, np.finfo(float).tiny))
    fractals = np.asarray(
        [
            np.interp(local.shared.frozen.base.FREQ, irasa_frequency, row)
            for row in irasa_log
        ]
    )
    return np.asarray(originals), fractals, parameters, centre, sd


def run_condition(family: str, rho: float, condition_index: int) -> dict[str, object]:
    covariance = covariance_matrix(PAIR_FAMILIES[family], rho)
    # Common random numbers make the covariance-sweep contrast identifiable:
    # only the target covariance changes across conditions.
    seed = local.BASE_SEED + 120_000_000
    cal_o, cal_f, cal_parameters, _, _ = spectral_batch(
        np.random.default_rng(seed), N_CALIBRATION, covariance
    )
    held_o, held_f, held_parameters, _, _ = spectral_batch(
        np.random.default_rng(seed + 1), N_HELDOUT, covariance
    )
    if REFERENCE_FOOOF_J is None or REFERENCE_SPLIT_J is None:
        raise RuntimeError("Reference Jacobians have not been initialized")
    fooof_j, split_j = REFERENCE_FOOOF_J, REFERENCE_SPLIT_J
    fooof_cal, split_cal = local.project(
        cal_o, cal_f, cal_o.mean(axis=0), cal_f.mean(axis=0), fooof_j, split_j
    )
    fooof_held = local.shared.frozen.official_fit(held_o, DISPLAY_BAND)
    split_held = local.shared.six.fit_split_batch(held_o, held_f, DISPLAY_BAND.lower())
    return {
        "family": family,
        "rho": rho,
        "fooof_theory": local.correlation(fooof_cal),
        "fooof_empirical": local.correlation(fooof_held),
        "split_theory": local.correlation(split_cal),
        "split_empirical": local.correlation(split_held),
        "fooof_valid": float(np.isfinite(fooof_held).all(axis=1).mean()),
        "split_valid": float(np.isfinite(split_held).all(axis=1).mean()),
        "true_center_knee_cal": float(
            stats.spearmanr(cal_parameters[:, 0], cal_parameters[:, 3]).statistic
        ),
        "true_center_knee_held": float(
            stats.spearmanr(held_parameters[:, 0], held_parameters[:, 3]).statistic
        ),
    }


def main() -> None:
    global REFERENCE_FOOOF_J, REFERENCE_SPLIT_J
    OUT.mkdir(parents=True, exist_ok=True)
    local.initialize()
    reference = local.reconstruct(3, DISPLAY_BAND, 0.01)
    REFERENCE_FOOOF_J = np.asarray(reference["influence"])
    REFERENCE_SPLIT_J = np.asarray(reference["split_influence"])
    tasks = list(itertools.product(PAIR_FAMILIES, RHO_VALUES))
    rows = []
    for index, (family, rho) in enumerate(tasks):
        row = run_condition(family, rho, index)
        rows.append(row)
        print(json.dumps(row), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pilot_results.csv", index=False)
    summary = (
        frame.assign(
            fooof_separation=lambda x: x.fooof_theory.abs() - x.split_theory.abs(),
            fooof_error=lambda x: (x.fooof_theory - x.fooof_empirical).abs(),
            split_error=lambda x: (x.split_theory - x.split_empirical).abs(),
        )
        .sort_values("fooof_separation", ascending=False)
    )
    summary.to_csv(OUT / "pilot_ranked.csv", index=False)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
