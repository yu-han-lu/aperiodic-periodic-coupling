from __future__ import annotations

"""Small pilot for a valid ~105-Hz SPLIT Jacobian and direct covariance theory."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import optimize, stats


WORKSPACE = Path(__file__).resolve().parent
PROJECT_CODE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project/experiments/pap_theory_validation/pap_validation"
for path in [WORKSPACE, PROJECT_CODE]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import redraw_fooof_irasa_split_3x3 as shared  # noqa: E402


OUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "split_105hz_direct_covariance_pilot_2026-09-13"
)
N_REPEATS = 4
N_INPUTS = 60
N_GAUSSIAN_CONTROL = 120
PILOT_SEED = shared.frozen.BASE_SEED + 9_000_000
DISPLAY_BAND = "High gamma"
BAND = "high gamma"
TARGET_HZ = 105.0


def select_valid_reference(
    original: np.ndarray,
    fractal: np.ndarray,
    true_centers: np.ndarray,
) -> tuple[int, np.ndarray]:
    fitted = shared.six.fit_split_batch(original, fractal, BAND)
    valid = np.isfinite(fitted).all(axis=1)
    if not np.any(valid):
        raise RuntimeError("No valid SPLIT trial is available for a local Jacobian")
    valid_indices = np.flatnonzero(valid)
    index = int(
        valid_indices[
            np.argmin(np.abs(true_centers[valid_indices] - TARGET_HZ))
        ]
    )
    return index, fitted


def reference_peak_parameters(
    original_row: np.ndarray, fractal_row: np.ndarray
) -> tuple[float, float, float, float]:
    mask = shared.frozen.base.band_mask(BAND)
    frequency = shared.frozen.base.FREQ[mask]
    residual = (original_row - fractal_row)[mask]
    center0 = float(frequency[np.argmax(residual)])
    initial = np.array([center0, max(float(residual.max()), 0.65), 1.5])
    fit = optimize.least_squares(
        lambda values: residual
        - shared.frozen.base.gaussian(frequency, *values),
        initial,
        bounds=([frequency[0], 0.0, 0.75], [frequency[-1], 3.0, 3.0]),
        max_nfev=250,
    ).x
    fitted_residual = residual - shared.frozen.base.gaussian(frequency, *fit)
    return float(fit[0]), float(fit[1]), float(fit[2]), float(
        np.std(fitted_residual)
    )


def direct_theory(
    original: np.ndarray,
    fractal: np.ndarray,
    reference_index: int,
) -> dict[str, float]:
    # Passing exactly one valid trial makes the SPLIT influence calculation use
    # that trial, instead of fitting an unaligned average over 60--150 Hz peaks.
    j_peak, j_ap = shared.six.split_weights(
        original[reference_index : reference_index + 1],
        fractal[reference_index : reference_index + 1],
        BAND,
    )
    residual = original - fractal
    x_residual = residual - residual.mean(axis=0, keepdims=True)
    x_fractal = fractal - fractal.mean(axis=0, keepdims=True)

    # These projections are algebraically equivalent to applying the directly
    # estimated covariance blocks, without constructing 1000 x 1000 matrices.
    delta_peak = x_residual @ j_peak
    delta_ap = x_fractal @ j_ap
    denominator = len(original) - 1.0
    var_peak = float(delta_peak @ delta_peak / denominator)
    var_ap = float(delta_ap @ delta_ap / denominator)
    covariance = float(delta_peak @ delta_ap / denominator)
    pearson = float(covariance / np.sqrt(var_peak * var_ap))
    gaussian_copula_spearman = float(
        (6.0 / np.pi) * np.arcsin(np.clip(pearson, -1.0, 1.0) / 2.0)
    )
    projected_spearman = float(stats.spearmanr(delta_peak, delta_ap).statistic)
    transfer = x_residual.T @ delta_ap / denominator
    contribution = j_peak * transfer
    return {
        "theory_covariance_pa": covariance,
        "theory_variance_p": var_peak,
        "theory_variance_a": var_ap,
        "theory_pearson_r": pearson,
        "theory_gaussian_copula_spearman_rho": gaussian_copula_spearman,
        "theory_projected_spearman_rho": projected_spearman,
        "contribution_sum": float(contribution.sum()),
        "jacobian_peak_max_abs": float(np.max(np.abs(j_peak))),
        "jacobian_ap_max_abs": float(np.max(np.abs(j_ap))),
    }


def run_repeat(repeat: int) -> dict[str, float | int]:
    rng = np.random.default_rng(PILOT_SEED + repeat * 100)
    original, fractal, true_centers, knees = shared.shared_signal_batch(
        rng, DISPLAY_BAND, N_INPUTS
    )
    reference_index, fitted = select_valid_reference(
        original, fractal, true_centers
    )
    valid = np.isfinite(fitted).all(axis=1)
    empirical_rho = float(
        stats.spearmanr(fitted[valid, 0], fitted[valid, 1]).statistic
    )
    center, height, sigma, residual_sd = reference_peak_parameters(
        original[reference_index], fractal[reference_index]
    )
    theory = direct_theory(original, fractal, reference_index)

    control_rng = np.random.default_rng(PILOT_SEED + repeat * 100 + 1)
    mixed_o, mixed_f = shared.frozen.covariance_mc_spectra(
        control_rng, original, fractal, n_mc=N_GAUSSIAN_CONTROL
    )
    mixed_fitted = shared.six.fit_split_batch(mixed_o, mixed_f, BAND)
    mixed_valid = np.isfinite(mixed_fitted).all(axis=1)

    return {
        "repeat": repeat,
        "n_real_inputs": N_INPUTS,
        "n_direct_valid": int(valid.sum()),
        "direct_detection_rate": float(valid.mean()),
        "direct_empirical_spearman_rho": empirical_rho,
        "n_gaussian_control": N_GAUSSIAN_CONTROL,
        "n_gaussian_valid": int(mixed_valid.sum()),
        "gaussian_detection_rate": float(mixed_valid.mean()),
        "reference_index": reference_index,
        "reference_true_center_hz": float(true_centers[reference_index]),
        "reference_knee_hz": float(knees[reference_index]),
        "reference_split_center_hz": center,
        "reference_split_height": height,
        "reference_split_sigma_hz": sigma,
        "reference_split_residual_sd": residual_sd,
        "reference_split_height_to_residual_sd": float(height / residual_sd),
        **theory,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    shared.fixed.load_sources()
    shared.matched.install_variant()
    shared.frozen.load_empirical_knees()
    rows = [run_repeat(repeat) for repeat in range(N_REPEATS)]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "pilot_repeat_results.csv", index=False)
    summary = {
        "design": {
            "n_repeats": N_REPEATS,
            "n_real_inputs_per_repeat": N_INPUTS,
            "n_gaussian_control_per_repeat": N_GAUSSIAN_CONTROL,
            "input_distribution": "same as Figure c high-gamma empirical-knee condition",
            "split_jacobian_rule": "valid real trial minimizing abs(true center - 105 Hz)",
            "covariance_rule": "direct sample covariance of real post-Welch/post-IRASA spectra; no Gaussian spectra enter theory",
        },
        "aggregate": {
            "direct_valid_total": int(frame.n_direct_valid.sum()),
            "direct_total": int(N_REPEATS * N_INPUTS),
            "direct_detection_rate": float(
                frame.n_direct_valid.sum() / (N_REPEATS * N_INPUTS)
            ),
            "gaussian_valid_total": int(frame.n_gaussian_valid.sum()),
            "gaussian_total": int(N_REPEATS * N_GAUSSIAN_CONTROL),
            "gaussian_detection_rate": float(
                frame.n_gaussian_valid.sum()
                / (N_REPEATS * N_GAUSSIAN_CONTROL)
            ),
            "median_reference_true_center_hz": float(
                frame.reference_true_center_hz.median()
            ),
            "median_reference_split_center_hz": float(
                frame.reference_split_center_hz.median()
            ),
            "median_reference_split_height_to_residual_sd": float(
                frame.reference_split_height_to_residual_sd.median()
            ),
            "mean_theory_projected_spearman_rho": float(
                frame.theory_projected_spearman_rho.mean()
            ),
            "mean_direct_empirical_spearman_rho": float(
                frame.direct_empirical_spearman_rho.mean()
            ),
        },
        "repeats": rows,
    }
    (OUT / "pilot_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
