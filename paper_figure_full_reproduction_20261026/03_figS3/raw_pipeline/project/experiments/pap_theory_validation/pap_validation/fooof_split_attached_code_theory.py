from __future__ import annotations

"""Theory validation for FOOOF versus the attached IRASA-SPLIT code path.

The SPLIT estimator follows the smooth one-peak / one-segment branch of:
  * c_oscillation_fit.m: Gaussian peak fit to log10(original)-log10(fractal)
    over 1--30 Hz, peak width 1.5--6 Hz, height >= 0.6, threshold 2 SD.
  * d_split_analysis.py: aperiodic slopes from a continuous piecewise-linear
    fit to log10(fractal) versus log10(f).  The surrogate is a pure power law,
    for which the penalized selector chooses one segment, so the exponent is
    the negative OLS slope over 0.5--200 Hz.

FOOOF is represented by its fixed-mode, one-Gaussian local least-squares
branch.  The conditional covariance theory is the sandwich/delta covariance
of the fitted parameters.  Peak/segment selection is checked empirically by
the full branch thresholds; it is not treated as differentiable.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, optimize, stats
from matplotlib.patches import FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "fooof_split_attached_code_theory_2026-08-30"
SEED = 20260830

FREQ = np.arange(0.2, 200.0 + 0.2, 0.2)
PEAK_MASK = (FREQ >= 1.0) & (FREQ <= 30.0)
FP = FREQ[PEAK_MASK]

BANDS = {
    "alpha": {"fit_range": (6.0, 15.0), "center_range": (8.0, 12.0), "representative": 10.0},
    "beta": {"fit_range": (12.0, 35.0), "center_range": (13.0, 30.0), "representative": 21.0},
    "low gamma": {"fit_range": (25.0, 70.0), "center_range": (30.0, 60.0), "representative": 45.0},
    "high gamma": {"fit_range": (50.0, 170.0), "center_range": (60.0, 150.0), "representative": 100.0},
}
BAND_COLORS = {
    "alpha": "#E9A03B",
    "beta": "#E76F51",
    "low gamma": "#4C78A8",
    "high gamma": "#343A69",
}

BASE = dict(offset=-0.35, exponent=1.65, center=5.0, height=0.90, sigma=1.55)
BASE_NOISE = dict(sd_original=0.045, sd_fractal=0.035, shared_fraction=0.55, ar1=0.86)


def band_mask(name: str) -> np.ndarray:
    lo, hi = BANDS[name]["fit_range"]
    return (FREQ >= lo) & (FREQ <= hi)


def band_for_center(center: float) -> str:
    if center <= 12.0:
        return "alpha"
    if center <= 30.0:
        return "beta"
    if center <= 60.0:
        return "low gamma"
    return "high gamma"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 17,
            "axes.labelsize": 18,
            "axes.titlesize": 18,
            "axes.linewidth": 2.4,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 14,
            "lines.linewidth": 3.0,
            "xtick.major.width": 2.2,
            "ytick.major.width": 2.2,
            "xtick.major.size": 6,
            "ytick.major.size": 6,
            "svg.fonttype": "none",
        }
    )


def gaussian(f: np.ndarray, center: float, height: float, sigma: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((f - center) / sigma) ** 2)


def mean_log_spectra(params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    ap = params["offset"] - params["exponent"] * np.log10(FREQ)
    osc = gaussian(FREQ, params["center"], params["height"], params["sigma"])
    return ap + osc, ap


def ar1_draws(rng: np.random.Generator, n: int, m: int, rho: float) -> np.ndarray:
    z = rng.normal(size=(n, m))
    x = np.empty_like(z)
    x[:, 0] = z[:, 0]
    scale = np.sqrt(1.0 - rho**2)
    for j in range(1, m):
        x[:, j] = rho * x[:, j - 1] + scale * z[:, j]
    return x


def noisy_log_spectra(
    rng: np.random.Generator,
    params: dict[str, float],
    noise: dict[str, float],
    n: int,
) -> tuple[np.ndarray, np.ndarray]:
    mean_original, mean_fractal = mean_log_spectra(params)
    m = len(FREQ)
    shared = ar1_draws(rng, n, m, noise["ar1"])
    only_o = ar1_draws(rng, n, m, noise["ar1"])
    only_f = ar1_draws(rng, n, m, noise["ar1"])
    q = noise["shared_fraction"]
    eo = noise["sd_original"] * (np.sqrt(q) * shared + np.sqrt(1.0 - q) * only_o)
    ef = noise["sd_fractal"] * (np.sqrt(q) * shared + np.sqrt(1.0 - q) * only_f)
    return mean_original + eo, mean_fractal + ef


def noisy_log_spectra_with_latents(
    rng: np.random.Generator,
    params: dict[str, float],
    noise: dict[str, float],
    centers: np.ndarray,
    exponents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(centers)
    ap = params["offset"] - exponents[:, None] * np.log10(FREQ)[None, :]
    osc = params["height"] * np.exp(
        -0.5 * ((FREQ[None, :] - centers[:, None]) / params["sigma"]) ** 2
    )
    shared = ar1_draws(rng, n, len(FREQ), noise["ar1"])
    only_o = ar1_draws(rng, n, len(FREQ), noise["ar1"])
    only_f = ar1_draws(rng, n, len(FREQ), noise["ar1"])
    q = noise["shared_fraction"]
    eo = noise["sd_original"] * (np.sqrt(q) * shared + np.sqrt(1.0 - q) * only_o)
    ef = noise["sd_fractal"] * (np.sqrt(q) * shared + np.sqrt(1.0 - q) * only_f)
    return ap + osc + eo, ap + ef


def exact_uncorrelated_normals(
    rng: np.random.Generator, n: int
) -> tuple[np.ndarray, np.ndarray]:
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    z1 = (z1 - z1.mean()) / z1.std(ddof=1)
    z2 = z2 - z2.mean()
    z2 = z2 - z1 * (np.dot(z1, z2) / np.dot(z1, z1))
    z2 = z2 / z2.std(ddof=1)
    return z1, z2


def _ap_guess(y: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([np.ones(len(f)), -np.log10(f)])
    return tuple(np.linalg.lstsq(design, y, rcond=None)[0])


def fit_fooof_local(
    y_original: np.ndarray, fit_mask: np.ndarray | None = None
) -> tuple[float, float]:
    """Fixed-mode, one-Gaussian FOOOF local branch; return center, exponent."""
    use_mask = PEAK_MASK if fit_mask is None else fit_mask
    fit_freq = FREQ[use_mask]
    y = y_original[use_mask]
    off0, exp0 = _ap_guess(y, fit_freq)
    flat = y - (off0 - exp0 * np.log10(fit_freq))
    c0 = float(fit_freq[np.argmax(flat)])
    h0 = float(max(np.max(flat), 0.65))
    x0 = np.array([off0, np.clip(exp0, 0.05, 4.5), c0, h0, 1.5])

    def residual(x: np.ndarray) -> np.ndarray:
        off, exponent, center, height, sigma = x
        model = off - exponent * np.log10(fit_freq) + gaussian(fit_freq, center, height, sigma)
        return y - model

    lo = np.array([-8.0, 0.0, fit_freq[0], 0.0, 0.75])
    hi = np.array([8.0, 5.0, fit_freq[-1], 3.0, 3.0])
    try:
        fit = optimize.least_squares(residual, x0, bounds=(lo, hi), max_nfev=350)
        off, exponent, center, height, sigma = fit.x
        flat_final = y - (off - exponent * np.log10(fit_freq))
        if height < 0.60 or height <= 2.0 * np.std(flat_final - gaussian(fit_freq, center, height, sigma)):
            return np.nan, np.nan
        return float(center), float(exponent)
    except (ValueError, RuntimeError):
        return np.nan, np.nan


def fit_split_local(
    y_original: np.ndarray,
    y_fractal: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Attached code's one-peak / one-segment branch; return center, exponent."""
    use_mask = PEAK_MASK if fit_mask is None else fit_mask
    fit_freq = FREQ[use_mask]
    residual_log = (y_original - y_fractal)[use_mask]
    c0 = float(fit_freq[np.argmax(residual_log)])
    h0 = float(max(np.max(residual_log), 0.65))
    x0 = np.array([c0, h0, 1.5])

    def peak_residual(x: np.ndarray) -> np.ndarray:
        return residual_log - gaussian(fit_freq, x[0], x[1], x[2])

    try:
        fit = optimize.least_squares(
            peak_residual,
            x0,
            bounds=(
                np.array([fit_freq[0], 0.0, 0.75]),
                np.array([fit_freq[-1], 3.0, 3.0]),
            ),
            max_nfev=250,
        )
        center, height, sigma = fit.x
        resid = peak_residual(fit.x)
        if height < 0.60 or height <= 2.0 * np.std(resid):
            return np.nan, np.nan
        _, exponent = _ap_guess(y_fractal, FREQ)
        return float(center), float(exponent)
    except (ValueError, RuntimeError):
        return np.nan, np.nan


def covariance_blocks(noise: dict[str, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(FREQ))
    corr = noise["ar1"] ** np.abs(idx[:, None] - idx[None, :])
    so, sf, q = noise["sd_original"], noise["sd_fractal"], noise["shared_fraction"]
    return so**2 * corr, sf**2 * corr, q * so * sf * corr


def theory_covariances(
    params: dict[str, float],
    noise: dict[str, float],
    fit_mask: np.ndarray | None = None,
    covariance_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cov_o, cov_f, cov_of = (
        covariance_blocks(noise) if covariance_cache is None else covariance_cache
    )
    use_mask = PEAK_MASK if fit_mask is None else fit_mask
    fit_freq = FREQ[use_mask]
    c, h, s = params["center"], params["height"], params["sigma"]
    g = np.exp(-0.5 * ((fit_freq - c) / s) ** 2)

    # FOOOF Jacobian: offset, exponent, center, height, sigma.
    d_fooof = np.column_stack(
        [
            np.ones(len(fit_freq)),
            -np.log10(fit_freq),
            h * g * (fit_freq - c) / s**2,
            g,
            h * g * (fit_freq - c) ** 2 / s**3,
        ]
    )
    influence_fooof = np.linalg.pinv(d_fooof)
    j_fooof = np.vstack([influence_fooof[2], influence_fooof[1]])
    v_fooof = j_fooof @ cov_o[np.ix_(use_mask, use_mask)] @ j_fooof.T

    # SPLIT peak is fit to original-fractal; exponent is fit to fractal.
    d_peak = np.column_stack(
        [h * g * (fit_freq - c) / s**2, g, h * g * (fit_freq - c) ** 2 / s**3]
    )
    j_center = np.linalg.pinv(d_peak)[0]
    d_ap = np.column_stack([np.ones(len(FREQ)), -np.log10(FREQ)])
    j_exponent = np.linalg.pinv(d_ap)[1]

    peak_idx = np.flatnonzero(use_mask)
    cov_residual = cov_o + cov_f - cov_of - cov_of.T
    cov_residual_fractal = cov_of - cov_f
    v_split = np.array(
        [
            [j_center @ cov_residual[np.ix_(peak_idx, peak_idx)] @ j_center,
             j_center @ cov_residual_fractal[np.ix_(peak_idx, np.arange(len(FREQ)))] @ j_exponent],
            [j_exponent @ cov_residual_fractal[np.ix_(peak_idx, np.arange(len(FREQ)))].T @ j_center,
             j_exponent @ cov_f @ j_exponent],
        ]
    )

    weights = {
        "fooof_center": influence_fooof[2],
        "fooof_exponent": influence_fooof[1],
        "split_center": j_center,
        "split_exponent": j_exponent,
    }
    return v_fooof, v_split, weights


def corr_from_cov(v: np.ndarray) -> float:
    return float(v[0, 1] / np.sqrt(v[0, 0] * v[1, 1]))


def spearman_from_pearson(r: float) -> float:
    return float(6.0 / np.pi * np.arcsin(np.clip(r, -1.0, 1.0) / 2.0))


def fit_many(
    y_o: np.ndarray,
    y_f: np.ndarray,
    fit_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    fooof = np.array([fit_fooof_local(row, fit_mask) for row in y_o])
    split = np.array(
        [fit_split_local(row_o, row_f, fit_mask) for row_o, row_f in zip(y_o, y_f)]
    )
    return fooof, split


def empirical_corr(values: np.ndarray, rank: bool = False) -> float:
    good = np.isfinite(values).all(axis=1)
    if good.sum() < 8:
        return np.nan
    test = stats.spearmanr if rank else stats.pearsonr
    return float(test(values[good, 0], values[good, 1]).statistic)


def conditional_experiment(rng: np.random.Generator, n: int = 1400) -> dict:
    params = dict(BASE)
    params["center"] = BANDS["alpha"]["representative"]
    use_mask = band_mask("alpha")
    y_o, y_f = noisy_log_spectra(rng, params, BASE_NOISE, n)
    fooof, split = fit_many(y_o, y_f, use_mask)
    vf, vs, weights = theory_covariances(params, BASE_NOISE, use_mask)
    return {
        "params": params,
        "fit_mask": use_mask,
        "y_original": y_o,
        "y_fractal": y_f,
        "fooof": fooof,
        "split": split,
        "v_fooof": vf,
        "v_split": vs,
        "weights": weights,
    }


def _estimate_noise_from_training(y_o: np.ndarray, y_f: np.ndarray) -> dict[str, float]:
    eo = y_o - y_o.mean(axis=0, keepdims=True)
    ef = y_f - y_f.mean(axis=0, keepdims=True)
    sd_o = float(np.sqrt(np.mean(eo**2)))
    sd_f = float(np.sqrt(np.mean(ef**2)))
    cross = float(np.mean(eo * ef) / max(sd_o * sd_f, 1e-12))
    lag_o = float(np.sum(eo[:, 1:] * eo[:, :-1]) / max(np.sum(eo[:, :-1] ** 2), 1e-12))
    lag_f = float(np.sum(ef[:, 1:] * ef[:, :-1]) / max(np.sum(ef[:, :-1] ** 2), 1e-12))
    return {
        "sd_original": sd_o,
        "sd_fractal": sd_f,
        "shared_fraction": float(np.clip(cross, 0.01, 0.98)),
        "ar1": float(np.clip((lag_o + lag_f) / 2.0, 0.05, 0.98)),
    }


def _corrected_corr(values: np.ndarray, artifact_cov: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if good.sum() < 8:
        return np.nan
    # Remove the theory-predicted spurious cross-covariance while retaining
    # held-out marginal variances.  This targets coupling bias directly and
    # avoids redefining the observed P/AP scales during correction.
    cov = np.cov(values[good].T, ddof=1)
    cov[0, 1] -= artifact_cov[0, 1]
    cov[1, 0] -= artifact_cov[1, 0]
    if cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return np.nan
    return corr_from_cov(cov)


def heldout_calibration_experiment(
    rng: np.random.Generator,
    n_conditions_per_band: int = 7,
    n_train: int = 260,
    n_heldout: int = 700,
) -> pd.DataFrame:
    """Estimate noise only in training spectra; validate and correct held-out fits."""
    rows: list[dict[str, float | str]] = []
    condition = 0
    for band_name, spec in BANDS.items():
        use_mask = band_mask(band_name)
        c_lo, c_hi = spec["center_range"]
        for _ in range(n_conditions_per_band):
            params = {
                "offset": BASE["offset"],
                "exponent": float(rng.uniform(1.15, 2.25)),
                "center": float(rng.uniform(c_lo, c_hi)),
                "height": float(rng.uniform(0.76, 1.12)),
                "sigma": float(rng.uniform(0.95, 2.55)),
            }
            noise = {
                "sd_original": float(rng.uniform(0.028, 0.065)),
                "sd_fractal": float(rng.uniform(0.022, 0.055)),
                "shared_fraction": float(rng.uniform(0.25, 0.80)),
                "ar1": float(rng.uniform(0.68, 0.92)),
            }
            train_o, train_f = noisy_log_spectra(rng, params, noise, n_train)
            estimated_noise = _estimate_noise_from_training(train_o, train_f)
            z_center, z_exponent = exact_uncorrelated_normals(rng, n_heldout)
            center_sd = {"alpha": 0.45, "beta": 0.80, "low gamma": 1.50,
                         "high gamma": 3.00}[band_name]
            exponent_sd = 0.055
            latent_centers = np.clip(params["center"] + center_sd * z_center, c_lo, c_hi)
            latent_exponents = params["exponent"] + exponent_sd * z_exponent
            held_o, held_f = noisy_log_spectra_with_latents(
                rng, params, noise, latent_centers, latent_exponents
            )
            fooof, split = fit_many(held_o, held_f, use_mask)
            vf, vs, _ = theory_covariances(params, estimated_noise, use_mask)
            latent_cov = np.diag(
                [np.var(latent_centers, ddof=1), np.var(latent_exponents, ddof=1)]
            )
            vf_total = latent_cov + vf
            vs_total = latent_cov + vs
            rows.append(
                {
                    "condition": condition,
                    "band": band_name,
                    "center": params["center"],
                    "fooof_pred": corr_from_cov(vf_total),
                    "fooof_artifact_r": corr_from_cov(vf),
                    "fooof_heldout": empirical_corr(fooof),
                    "fooof_corrected": _corrected_corr(fooof, vf),
                    "split_pred": corr_from_cov(vs_total),
                    "split_artifact_r": corr_from_cov(vs),
                    "split_heldout": empirical_corr(split),
                    "split_corrected": _corrected_corr(split, vs),
                    "fooof_valid": np.isfinite(fooof).all(axis=1).mean(),
                    "split_valid": np.isfinite(split).all(axis=1).mean(),
                    "n_train": n_train,
                    "n_heldout": n_heldout,
                }
            )
            condition += 1
    return pd.DataFrame(rows)


def uniform_theory_calibration_experiment(
    rng: np.random.Generator,
    n_conditions: int = 25,
    n_train: int = 260,
    n_heldout: int = 1600,
) -> pd.DataFrame:
    """Validate a theory-only, near-uniform 0.02--0.50 FOOOF error-r design.

    Candidate conditions are selected using the analytic covariance only. The
    held-out simulations are generated after selection and therefore cannot
    influence which points enter the calibration plot.
    """
    targets = np.linspace(0.02, 0.50, n_conditions)
    names = list(BANDS)
    cached_covariance = covariance_blocks(BASE_NOISE)
    candidate_by_band: dict[str, list[dict[str, object]]] = {}

    # Moving the peak relative to the fitted spectral window continuously
    # changes how strongly peak-center and aperiodic-slope errors mix.
    for band_name, spec in BANDS.items():
        center = float(spec["representative"])
        candidates: list[dict[str, object]] = []
        for width in np.geomspace(1.35, 95.0, 150):
            fit_lo = max(0.2, center - float(width))
            fit_hi = min(200.0, center + 2.0 * float(width))
            use_mask = (FREQ >= fit_lo) & (FREQ <= fit_hi)
            if use_mask.sum() < 28:
                continue
            params = dict(BASE)
            params["center"] = center
            vf, vs, _ = theory_covariances(
                params, BASE_NOISE, use_mask, covariance_cache=cached_covariance
            )
            fooof_r = corr_from_cov(vf)
            if not np.isfinite(fooof_r) or fooof_r < -0.02 or fooof_r > 0.65:
                continue
            candidates.append(
                {
                    "band": band_name,
                    "center": center,
                    "fit_lo": float(FREQ[use_mask][0]),
                    "fit_hi": float(FREQ[use_mask][-1]),
                    "fit_mask": use_mask,
                    "params": params,
                    "v_fooof": vf,
                    "v_split": vs,
                    "fooof_r": fooof_r,
                }
            )
        candidate_by_band[band_name] = candidates

    rows: list[dict[str, float | str]] = []
    used: set[tuple[str, float, float]] = set()
    for condition, target in enumerate(targets):
        # Very narrow high-frequency windows make the nonlinear FOOOF fit
        # numerically underidentified. Keep all four bands at low r, then use
        # stable low-frequency conditions for the upper calibration range.
        if target <= 0.10:
            band_name = names[condition % 4]
        elif target <= 0.20:
            band_name = names[(condition - 5) % 3]
        else:
            band_name = names[(condition - 10) % 2]
        available = [
            c for c in candidate_by_band[band_name]
            if (str(c["band"]), float(c["fit_lo"]), float(c["fit_hi"])) not in used
        ]
        chosen = min(available, key=lambda c: abs(float(c["fooof_r"]) - target))
        used.add((band_name, float(chosen["fit_lo"]), float(chosen["fit_hi"])))
        params = chosen["params"]
        use_mask = chosen["fit_mask"]

        # Training data estimate only the covariance removed by correction;
        # the displayed theory uses the known generative covariance.
        train_o, train_f = noisy_log_spectra(rng, params, BASE_NOISE, n_train)
        estimated_noise = _estimate_noise_from_training(train_o, train_f)
        held_o, held_f = noisy_log_spectra(rng, params, BASE_NOISE, n_heldout)
        fooof, split = fit_many(held_o, held_f, use_mask)
        vf = chosen["v_fooof"]
        vs = chosen["v_split"]
        vf_est, vs_est, _ = theory_covariances(params, estimated_noise, use_mask)
        rows.append(
            {
                "condition": condition,
                "target_r": float(target),
                "band": band_name,
                "center": float(chosen["center"]),
                "fit_lo": float(chosen["fit_lo"]),
                "fit_hi": float(chosen["fit_hi"]),
                "fooof_pred": corr_from_cov(vf),
                "fooof_artifact_r": corr_from_cov(vf),
                "fooof_heldout": empirical_corr(fooof),
                "fooof_corrected": _corrected_corr(fooof, vf_est),
                "split_pred": corr_from_cov(vs),
                "split_artifact_r": corr_from_cov(vs),
                "split_heldout": empirical_corr(split),
                "split_corrected": _corrected_corr(split, vs_est),
                "fooof_valid": np.isfinite(fooof).all(axis=1).mean(),
                "split_valid": np.isfinite(split).all(axis=1).mean(),
                "n_train": n_train,
                "n_heldout": n_heldout,
            }
        )
    return pd.DataFrame(rows).sort_values("fooof_pred").reset_index(drop=True)


def _fisher_interval(r: float, n: int) -> tuple[float, float]:
    if not np.isfinite(r) or n <= 3:
        return np.nan, np.nan
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    half = 1.96 / np.sqrt(n - 3)
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def center_sweep(rng: np.random.Generator, n_each: int = 1500) -> pd.DataFrame:
    centers = np.array(
        [8, 9, 10, 11, 12, 14, 18, 22, 26, 30, 32, 39, 46, 53, 60, 70, 85, 105, 127, 150],
        dtype=float,
    )
    rows = []
    for center in centers:
        band_name = band_for_center(center)
        use_mask = band_mask(band_name)
        params = dict(BASE)
        params["center"] = float(center)
        y_o, y_f = noisy_log_spectra(rng, params, BASE_NOISE, n_each)
        fooof, split = fit_many(y_o, y_f, use_mask)
        vf, vs, _ = theory_covariances(params, BASE_NOISE, use_mask)
        fr = empirical_corr(fooof)
        sr = empirical_corr(split)
        nf = int(np.isfinite(fooof).all(axis=1).sum())
        ns = int(np.isfinite(split).all(axis=1).sum())
        flo, fhi = _fisher_interval(fr, nf)
        slo, shi = _fisher_interval(sr, ns)
        rows.append(
            {
                "center": center,
                "band": band_name,
                "fooof_pred": corr_from_cov(vf),
                "fooof_emp": fr,
                "fooof_lo": flo,
                "fooof_hi": fhi,
                "split_pred": corr_from_cov(vs),
                "split_emp": sr,
                "split_lo": slo,
                "split_hi": shi,
                "n_fooof": nf,
                "n_split": ns,
            }
        )
    return pd.DataFrame(rows)


def covariance_factor_experiment(
    rng: np.random.Generator, n_each: int = 1800
) -> pd.DataFrame:
    """Empirical and theoretical factors in |V_PA| = |rho| * sd(P) * sd(A)."""
    rows: list[dict[str, float | str]] = []
    for band_name, spec in BANDS.items():
        params = dict(BASE)
        params["center"] = float(spec["representative"])
        use_mask = band_mask(band_name)
        y_o, y_f = noisy_log_spectra(rng, params, BASE_NOISE, n_each)
        fooof, split = fit_many(y_o, y_f, use_mask)
        vf, vs, _ = theory_covariances(params, BASE_NOISE, use_mask)
        for method, values, theory in (("FOOOF", fooof, vf), ("SPLIT", split, vs)):
            good = np.isfinite(values).all(axis=1)
            empirical = np.cov(values[good].T, ddof=1)
            for source, cov in (("empirical", empirical), ("theory", theory)):
                rows.append(
                    {
                        "band": band_name,
                        "method": method,
                        "source": source,
                        "abs_rho": abs(corr_from_cov(cov)),
                        "sd_p": float(np.sqrt(cov[0, 0])),
                        "sd_ap": float(np.sqrt(cov[1, 1])),
                        "abs_cov": abs(float(cov[0, 1])),
                        "n": int(good.sum()),
                    }
                )
    return pd.DataFrame(rows)


def targeted_null_conditions(n_conditions: int = 120) -> list[dict[str, object]]:
    """Choose conditions by theory, with expected FOOOF rho from -0.08 to 0.50."""
    latent_scale = 0.25
    candidates: list[dict[str, object]] = []
    for ar1 in np.linspace(0.70, 0.985, 100):
        noise = dict(BASE_NOISE)
        noise["ar1"] = float(ar1)
        cached_covariance = covariance_blocks(noise)
        for band_name, spec in BANDS.items():
            params = dict(BASE)
            params["center"] = float(spec["representative"])
            use_mask = band_mask(band_name)
            vf, vs, _ = theory_covariances(
                params, noise, use_mask, covariance_cache=cached_covariance
            )
            conditional_r = corr_from_cov(vf)
            expected_pearson = conditional_r / (1.0 + latent_scale**2)
            candidates.append(
                {
                    "band": band_name,
                    "params": params,
                    "noise": noise,
                    "fit_mask": use_mask,
                    "v_fooof": vf,
                    "v_split": vs,
                    "theory_error_r": conditional_r,
                    "expected_observed_r": spearman_from_pearson(expected_pearson),
                    "latent_scale": latent_scale,
                }
            )

    targets = np.linspace(-0.08, 0.50, n_conditions)
    selected: list[dict[str, object]] = []
    used: set[tuple[str, float]] = set()
    for target in targets:
        available = [
            c for c in candidates
            if (str(c["band"]), float(c["noise"]["ar1"])) not in used
        ]
        chosen = min(
            available, key=lambda c: abs(float(c["expected_observed_r"]) - target)
        )
        used.add((str(chosen["band"]), float(chosen["noise"]["ar1"])))
        condition = dict(chosen)
        condition["target_r"] = float(target)
        selected.append(condition)
    return selected


def simulate_targeted_null_condition(
    rng: np.random.Generator,
    condition: dict[str, object],
    dataset: int,
    n_trials: int = 1600,
) -> dict[str, float | int | str]:
    params = condition["params"]
    noise = condition["noise"]
    vf = condition["v_fooof"]
    vs = condition["v_split"]
    latent_scale = float(condition["latent_scale"])
    z_center, z_exponent = exact_uncorrelated_normals(rng, n_trials)
    true_c = (
        float(params["center"])
        + latent_scale * np.sqrt(float(vf[0, 0])) * z_center
    )
    true_e = (
        float(params["exponent"])
        + latent_scale * np.sqrt(float(vf[1, 1])) * z_exponent
    )
    y_o, y_f = noisy_log_spectra_with_latents(
        rng, params, noise, true_c, true_e
    )
    fooof, split = fit_many(y_o, y_f, condition["fit_mask"])

    def summarize(values: np.ndarray, artifact: np.ndarray) -> tuple[float, float, float, float]:
        good = np.isfinite(values).all(axis=1)
        z = values[good]
        raw_test = stats.spearmanr(z[:, 0], z[:, 1])
        corrected_pearson = _corrected_corr(z, artifact)
        corrected_rank = spearman_from_pearson(corrected_pearson)
        return (
            float(raw_test.statistic),
            float(raw_test.pvalue),
            corrected_rank,
            float(good.mean()),
        )

    fr, fp, fc, fv = summarize(fooof, vf)
    sr, sp, sc, sv = summarize(split, vs)
    return {
        "dataset": dataset,
        "n_trials": n_trials,
        "band": str(condition["band"]),
        "ar1": float(noise["ar1"]),
        "target_r": float(condition["target_r"]),
        "theory_error_r": float(condition["theory_error_r"]),
        "expected_observed_r": float(condition["expected_observed_r"]),
        "latent_r": float(stats.spearmanr(true_c, true_e).statistic),
        "fooof_r": fr,
        "fooof_p": fp,
        "fooof_corrected_r": fc,
        "fooof_valid": fv,
        "split_r": sr,
        "split_p": sp,
        "split_corrected_r": sc,
        "split_valid": sv,
    }


def null_datasets(
    rng: np.random.Generator, n_datasets: int = 120, n_trials: int = 1600
) -> pd.DataFrame:
    conditions = targeted_null_conditions(n_datasets)
    rows = [
        simulate_targeted_null_condition(rng, condition, dataset, n_trials)
        for dataset, condition in enumerate(conditions)
    ]
    return pd.DataFrame(rows)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=24, fontweight="bold")


def thicken(ax: plt.Axes) -> None:
    ax.tick_params(width=2.2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)


def _diagram_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    edge: str,
    fill: str = "white",
    fontsize: float = 11.5,
    linewidth: float = 2.2,
) -> None:
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.014,rounding_size=0.018",
        transform=ax.transAxes,
        facecolor=fill,
        edgecolor=edge,
        linewidth=linewidth,
        clip_on=False,
    )
    ax.add_patch(patch)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#202020",
        linespacing=1.15,
    )


def _diagram_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str,
    label: str | None = None,
    label_xy: tuple[float, float] | None = None,
    fontsize: float = 10.5,
) -> None:
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=2.2, mutation_scale=13),
    )
    if label and label_xy:
        ax.text(
            label_xy[0],
            label_xy[1],
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=color,
            fontweight="bold",
        )


def draw_error_pathway_panel(ax: plt.Axes, cond: dict, standalone: bool = False) -> None:
    """Draw the common-error source for FOOOF and the SPLIT cancellation path."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fs = 15.0 if standalone else 10.2
    fs_small = 12.0 if standalone else 8.8
    lw = 2.8 if standalone else 2.1

    fooof_emp = empirical_corr(cond["fooof"])
    split_emp = empirical_corr(cond["split"])
    fooof_pred = corr_from_cov(cond["v_fooof"])
    split_pred = corr_from_cov(cond["v_split"])
    fooof_summary = (
        f"Shared-error covariance\nwPᵀΩo wA > 0\ntheory r = {fooof_pred:.2f}   empirical r = {fooof_emp:.2f}"
        if standalone
        else f"Shared original error\nCov(δP, δA) > 0\nr = {fooof_pred:.2f} theory / {fooof_emp:.2f} empirical"
    )
    split_summary = (
        f"Fractal contribution\npartly cancels\ntheory r = {split_pred:.2f}   empirical r = {split_emp:.2f}"
        if standalone
        else f"Opposite fractal sign\npartial cancellation\nr = {split_pred:.2f} theory / {split_emp:.2f} empirical"
    )

    ax.text(0.01, 0.95, "FOOOF", transform=ax.transAxes, ha="left", va="top",
            fontsize=fs + 2, fontweight="bold", color="#247BA0")
    _diagram_box(ax, (0.02, 0.66), 0.22, 0.17, "Original PSD\nerror  εo(f)", "#555555",
                 fontsize=fs, linewidth=lw)
    _diagram_box(ax, (0.34, 0.77), 0.27, 0.14, "Periodic error\nδP = wPᵀεo", "#247BA0",
                 fontsize=fs, linewidth=lw)
    _diagram_box(ax, (0.34, 0.55), 0.27, 0.14, "Aperiodic error\nδA = wAᵀεo", "#D1495B",
                 fontsize=fs, linewidth=lw)
    _diagram_arrow(ax, (0.24, 0.745), (0.34, 0.84), "#247BA0", "same εo", (0.29, 0.84), fs_small)
    _diagram_arrow(ax, (0.24, 0.745), (0.34, 0.62), "#D1495B", "same εo", (0.29, 0.65), fs_small)
    _diagram_box(
        ax,
        (0.72, 0.63),
        0.25,
        0.20,
        fooof_summary,
        "#7B2CBF",
        fill="#F3EAFB",
        fontsize=fs_small,
        linewidth=lw,
    )
    _diagram_arrow(ax, (0.61, 0.84), (0.72, 0.76), "#247BA0")
    _diagram_arrow(ax, (0.61, 0.62), (0.72, 0.70), "#D1495B")

    ax.plot([0.01, 0.99], [0.48, 0.48], transform=ax.transAxes, color="#C0C0C0", lw=1.5)
    ax.text(0.01, 0.44, "SPLIT", transform=ax.transAxes, ha="left", va="top",
            fontsize=fs + 2, fontweight="bold", color="#2A9D8F")
    _diagram_box(ax, (0.02, 0.26), 0.22, 0.12, "Original error  εo", "#555555",
                 fontsize=fs, linewidth=lw)
    _diagram_box(ax, (0.02, 0.07), 0.22, 0.12, "Fractal error  εf", "#D1495B",
                 fontsize=fs, linewidth=lw)
    _diagram_box(ax, (0.34, 0.27), 0.29, 0.14, "Periodic error\nδP = wPᵀ(εo − εf)", "#2A9D8F",
                 fontsize=fs, linewidth=lw)
    _diagram_box(ax, (0.34, 0.06), 0.29, 0.14, "Aperiodic error\nδA = wAᵀεf", "#E07A5F",
                 fontsize=fs, linewidth=lw)
    _diagram_arrow(ax, (0.24, 0.32), (0.34, 0.34), "#247BA0", "+", (0.29, 0.35), fs)
    _diagram_arrow(ax, (0.24, 0.13), (0.34, 0.30), "#D1495B", "−", (0.29, 0.22), fs)
    _diagram_arrow(ax, (0.24, 0.13), (0.34, 0.13), "#E07A5F", "+", (0.29, 0.11), fs)
    _diagram_box(
        ax,
        (0.72, 0.13),
        0.25,
        0.20,
        split_summary,
        "#2A9D8F",
        fill="#E8F5F2",
        fontsize=fs_small,
        linewidth=lw,
    )
    _diagram_arrow(ax, (0.63, 0.34), (0.72, 0.27), "#2A9D8F")
    _diagram_arrow(ax, (0.63, 0.13), (0.72, 0.19), "#E07A5F")

    if standalone:
        ax.text(
            0.5,
            0.005,
            "FOOOF sends the same spectral error into P and AP; SPLIT sends the shared fractal error into P with the opposite sign.",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12.5,
            color="#333333",
        )


def draw_influence_mechanism_panel(ax: plt.Axes, standalone: bool = False) -> None:
    """Show the algorithmic influence functions and their covariance overlap."""
    ax.set_axis_off()
    params = dict(BASE)
    params["center"] = float(BANDS["alpha"]["representative"])
    use_mask = band_mask("alpha")
    idx = np.flatnonzero(use_mask)
    fit_freq = FREQ[use_mask]
    vf, vs, weights = theory_covariances(params, BASE_NOISE, use_mask)
    cov_o, cov_f, cov_of = covariance_blocks(BASE_NOISE)

    fooof_jp = np.asarray(weights["fooof_center"])
    fooof_ja = np.asarray(weights["fooof_exponent"])
    split_jp = np.asarray(weights["split_center"])
    split_ja = np.asarray(weights["split_exponent"])

    fooof_jp_full = np.zeros_like(FREQ)
    fooof_ja_full = np.zeros_like(FREQ)
    split_jp_full = np.zeros_like(FREQ)
    fooof_jp_full[use_mask] = fooof_jp
    fooof_ja_full[use_mask] = fooof_ja
    split_jp_full[use_mask] = split_jp

    # Exact signed frequency contributions. Their sums equal V_{P,AP}.
    fooof_transfer = cov_o[np.ix_(idx, idx)] @ fooof_ja
    fooof_contribution = fooof_jp * fooof_transfer
    residual_fractal_cross = cov_of - cov_f
    split_transfer = residual_fractal_cross[np.ix_(idx, np.arange(len(FREQ)))] @ split_ja
    split_contribution = split_jp * split_transfer

    def normalize(values: np.ndarray) -> np.ndarray:
        scale = float(np.max(np.abs(values)))
        return values / scale if scale > 0 else values

    def scientific_math(value: float) -> str:
        if value == 0:
            return "0"
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.1f}\times10^{{{exponent}}}"

    left = 0.075
    gap = 0.075
    width = (1.0 - 2 * left - gap) / 2
    top_y, top_h = 0.55, 0.34
    bottom_y, bottom_h = 0.085, 0.32
    axes_top = [
        ax.inset_axes([left, top_y, width, top_h]),
        ax.inset_axes([left + width + gap, top_y, width, top_h]),
    ]
    axes_bottom = [
        ax.inset_axes([left, bottom_y, width, bottom_h]),
        ax.inset_axes([left + width + gap, bottom_y, width, bottom_h]),
    ]
    influence_sets = [
        (fooof_jp_full, fooof_ja_full, "FOOOF", "same original PSD"),
        (split_jp_full, split_ja, "SPLIT", "residual P / fractal AP"),
    ]
    contributions = [
        (fooof_contribution, vf, "#247BA0"),
        (split_contribution, vs, "#D1495B"),
    ]
    small = 10.5 if not standalone else 13.0
    title_size = 12.5 if not standalone else 16.0
    label_size = 10.5 if not standalone else 13.5

    for k, (sub, (jp, ja, method, subtitle)) in enumerate(zip(axes_top, influence_sets)):
        sub.plot(fit_freq, normalize(jp[use_mask]), color="#247BA0", lw=2.4,
                 label=r"$J_P(f)$")
        sub.plot(fit_freq, normalize(ja[use_mask]), color="#D1495B", lw=2.4,
                 label=r"$J_{AP}(f)$")
        sub.axhline(0, color="0.45", lw=1.0)
        sub.set_xlim(float(fit_freq[0]), float(fit_freq[-1]))
        sub.set_ylim(-1.08, 1.08)
        sub.set_xticks([6, 10, 15])
        sub.tick_params(labelsize=small, width=1.5, length=4)
        sub.set_title(method, fontsize=title_size, fontweight="bold", pad=11,
                      color="#247BA0" if method == "FOOOF" else "#2A9D8F")
        sub.text(0.5, 1.01, subtitle, transform=sub.transAxes, ha="center", va="bottom",
                 fontsize=small - 1, color="0.30")
        if k == 0:
            sub.set_ylabel("Normalized influence", fontsize=label_size)
            sub.legend(frameon=False, fontsize=small, loc="lower left", ncol=2,
                       handlelength=1.4, columnspacing=0.8)
        else:
            sub.set_yticklabels([])
        for spine in sub.spines.values():
            spine.set_linewidth(1.6)

    for k, (sub, (contribution, cov, color)) in enumerate(zip(axes_bottom, contributions)):
        cn = normalize(contribution)
        sub.fill_between(fit_freq, 0, cn, where=cn >= 0, color="#6A4C93", alpha=0.78)
        sub.fill_between(fit_freq, 0, cn, where=cn < 0, color="#E76F51", alpha=0.78)
        sub.plot(fit_freq, cn, color=color, lw=1.8)
        sub.axhline(0, color="0.35", lw=1.1)
        sub.set_xlim(float(fit_freq[0]), float(fit_freq[-1]))
        sub.set_ylim(-1.08, 1.08)
        sub.set_xticks([6, 10, 15])
        sub.tick_params(labelsize=small, width=1.5, length=4)
        sub.set_xlabel("Frequency (Hz)", fontsize=label_size)
        if k == 0:
            sub.set_ylabel("Normalized signed\ncontribution", fontsize=label_size)
        else:
            sub.set_yticklabels([])
        sub.text(
            0.98,
            0.94,
            rf"$\sum_f c(f)=V_{{P,AP}}={scientific_math(float(cov[0,1]))}$" + "\n" +
            rf"$\rho_{{art}}={corr_from_cov(cov):.2f}$",
            transform=sub.transAxes,
            ha="right",
            va="top",
            fontsize=small,
        )
        for spine in sub.spines.values():
            spine.set_linewidth(1.6)

    ax.text(
        0.5,
        1.025,
        r"$V_{P,AP}=\sum_f J_P(f)\,[\Sigma J_{AP}^{T}](f)$",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15.0 if not standalone else 19.0,
    )
    if standalone:
        ax.text(0.995, 0.98, "all axes: 6–15 Hz; SPLIT AP fit: 0.2–200 Hz",
                transform=ax.transAxes, ha="right", va="top", fontsize=small,
                color="0.35")


def draw_influence_mechanism_external(axes: list[plt.Axes]) -> None:
    """Draw B1--B4 into four ordinary, equal-sized grid cells."""
    params = dict(BASE)
    params["center"] = float(BANDS["alpha"]["representative"])
    use_mask = band_mask("alpha")
    idx = np.flatnonzero(use_mask)
    fit_freq = FREQ[use_mask]
    vf, vs, weights = theory_covariances(params, BASE_NOISE, use_mask)
    cov_o, cov_f, cov_of = covariance_blocks(BASE_NOISE)

    fooof_jp = np.asarray(weights["fooof_center"])
    fooof_ja = np.asarray(weights["fooof_exponent"])
    split_jp = np.asarray(weights["split_center"])
    split_ja = np.asarray(weights["split_exponent"])
    fooof_transfer = cov_o[np.ix_(idx, idx)] @ fooof_ja
    split_transfer = (cov_of - cov_f)[np.ix_(idx, np.arange(len(FREQ)))] @ split_ja
    fooof_contribution = fooof_jp * fooof_transfer
    split_contribution = split_jp * split_transfer

    def normalize(values: np.ndarray) -> np.ndarray:
        scale = float(np.max(np.abs(values)))
        return values / scale if scale > 0 else values

    def scientific_math(value: float) -> str:
        exponent = int(np.floor(np.log10(abs(value))))
        mantissa = value / (10.0**exponent)
        return rf"{mantissa:.1f}\times10^{{{exponent}}}"

    top_specs = [
        (fooof_jp, fooof_ja, "B1", "FOOOF influence functions", "same original PSD"),
        (split_jp, split_ja[use_mask], "B2", "SPLIT influence functions",
         "residual P / fractal AP"),
    ]
    for ax, (jp, ja, label, title, subtitle) in zip(axes[:2], top_specs):
        ax.plot(fit_freq, normalize(jp), color="#247BA0", lw=3.0, label=r"$J_P(f)$")
        ax.plot(fit_freq, normalize(ja), color="#D1495B", lw=3.0, label=r"$J_{AP}(f)$")
        ax.axhline(0, color="0.45", lw=1.2)
        ax.set(xlim=(float(fit_freq[0]), float(fit_freq[-1])), ylim=(-1.08, 1.08),
               xticks=[6, 10, 15], xlabel="Frequency (Hz)", ylabel="Normalized influence")
        ax.set_title(title, fontsize=17, fontweight="bold", pad=18,
                     color="#247BA0" if "FOOOF" in title else "#2A9D8F")
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12.5, color="0.30")
        ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=19,
                fontweight="bold", ha="left", va="top")
        ax.legend(frameon=False, fontsize=13, loc="lower left", ncol=2,
                  handlelength=1.5, columnspacing=0.9)

    bottom_specs = [
        (fooof_contribution, vf, "B3", "FOOOF covariance contribution", "#247BA0"),
        (split_contribution, vs, "B4", "SPLIT covariance contribution", "#D1495B"),
    ]
    for ax, (contribution, cov, label, title, color) in zip(axes[2:], bottom_specs):
        cn = normalize(contribution)
        ax.fill_between(fit_freq, 0, cn, where=cn >= 0, color="#6A4C93", alpha=0.78)
        ax.fill_between(fit_freq, 0, cn, where=cn < 0, color="#E76F51", alpha=0.78)
        ax.plot(fit_freq, cn, color=color, lw=2.4)
        ax.axhline(0, color="0.35", lw=1.2)
        ax.set(xlim=(float(fit_freq[0]), float(fit_freq[-1])), ylim=(-1.08, 1.08),
               xticks=[6, 10, 15], xlabel="Frequency (Hz)",
               ylabel="Normalized signed contribution")
        ax.set_title(title, fontsize=17, fontweight="bold", pad=12,
                     color="#247BA0" if "FOOOF" in title else "#2A9D8F")
        ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=19,
                fontweight="bold", ha="left", va="top")
        ax.text(
            0.97,
            0.94,
            rf"$\sum_f c(f)=V_{{P,AP}}={scientific_math(float(cov[0,1]))}$" + "\n" +
            rf"$\rho_{{art}}={corr_from_cov(cov):.2f}$",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=13,
        )


def ellipse_from_cov(ax: plt.Axes, mean: np.ndarray, cov: np.ndarray, color: str) -> None:
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    from matplotlib.patches import Ellipse

    ell = Ellipse(mean, 4 * np.sqrt(vals[0]), 4 * np.sqrt(vals[1]), angle=angle,
                  fill=False, edgecolor=color, linewidth=3.0, linestyle="--")
    ax.add_patch(ell)


def plot_six_panel(
    cond: dict,
    cal: pd.DataFrame,
    sweep: pd.DataFrame,
    null: pd.DataFrame,
    factors: pd.DataFrame,
) -> Path:
    fig, axes = plt.subplots(3, 3, figsize=(17.0, 15.5), constrained_layout=True)
    mean_o, mean_f = mean_log_spectra(cond["params"])

    ax = axes[0, 0]
    ax.plot(FREQ, mean_o, color="#333333", label="Original log PSD")
    ax.plot(FREQ, mean_f, color="#D1495B", label="IRASA fractal")
    ax.plot(FREQ, mean_o - mean_f, color="#247BA0", label="Oscillatory residual")
    alpha_lo, alpha_hi = BANDS["alpha"]["fit_range"]
    ax.axvspan(alpha_lo, alpha_hi, color="#247BA0", alpha=0.10)
    ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel="log10 power")
    ax.set_xticks([0.2, 1, 10, 30, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, loc="lower left")
    add_panel_label(ax, "A")

    b_axes = [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]
    draw_influence_mechanism_external(b_axes)

    ax = axes[1, 0]
    f = cond["fooof"]
    s = cond["split"]
    gf = np.isfinite(f).all(axis=1)
    gs = np.isfinite(s).all(axis=1)
    ax.scatter(f[gf, 0], f[gf, 1], s=15, alpha=0.16, color="#247BA0", label="FOOOF")
    ax.scatter(s[gs, 0], s[gs, 1], s=15, alpha=0.16, color="#2A9D8F", label="SPLIT")
    ellipse_from_cov(ax, np.nanmean(f, axis=0), cond["v_fooof"], "#247BA0")
    ellipse_from_cov(ax, np.nanmean(s, axis=0), cond["v_split"], "#2A9D8F")
    ax.set(xlabel="Estimated peak center (Hz)", ylabel="Estimated exponent")
    ax.legend(frameon=False, loc="upper left")
    rf = empirical_corr(f)
    rs = empirical_corr(s)
    ax.text(0.04, 0.13, f"FOOOF: r = {rf:.2f}", transform=ax.transAxes,
            color="#247BA0", fontsize=14, fontweight="bold")
    ax.text(0.04, 0.06, f"SPLIT: r = {rs:.2f}", transform=ax.transAxes,
            color="#2A9D8F", fontsize=14, fontweight="bold")
    add_panel_label(ax, "C")

    ax = axes[2, 0]
    ax.scatter(
        cal.fooof_pred,
        cal.fooof_heldout,
        s=54,
        marker="o",
        color="#247BA0",
        edgecolor="black",
        linewidth=0.55,
        alpha=0.84,
        label="FOOOF",
    )
    both = np.r_[cal.fooof_pred, cal.fooof_heldout]
    lo, hi = np.nanmin(both) - 0.06, np.nanmax(both) + 0.06
    ax.plot([lo, hi], [lo, hi], color="0.25", ls="--", lw=2.0)
    ax.axhline(0, color="0.45", lw=1.3)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Theory-predicted error r",
           ylabel="Held-out error r")
    linear_ticks = [-0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ax.set_xticks(linear_ticks)
    ax.set_yticks(linear_ticks)
    ax.legend(frameon=False, loc="upper left", fontsize=12.5)
    fit = stats.linregress(cal.fooof_pred, cal.fooof_heldout)
    ax.text(
        0.97,
        0.05,
        f"FOOOF: Pearson r = {fit.rvalue:.2f}\n"
        f"OLS slope = {fit.slope:.2f}; $R^2$ = {fit.rvalue**2:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="#247BA0",
        fontsize=12.5,
        fontweight="bold",
    )
    add_panel_label(ax, "D")

    ax = axes[2, 1]
    for band_name, spec in BANDS.items():
        lo_band, hi_band = spec["center_range"]
        ax.axvspan(lo_band, hi_band, color=BAND_COLORS[band_name], alpha=0.075, lw=0)
        ax.text(np.sqrt(lo_band * hi_band), 0.98, band_name, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=10.5, color=BAND_COLORS[band_name],
                fontweight="bold")
    ax.fill_between(sweep.center, sweep.fooof_lo, sweep.fooof_hi,
                    color="#247BA0", alpha=0.13, linewidth=0)
    ax.fill_between(sweep.center, sweep.split_lo, sweep.split_hi,
                    color="#2A9D8F", alpha=0.13, linewidth=0)
    ax.plot(sweep.center, sweep.fooof_emp, "o-", color="#247BA0", label="FOOOF empirical")
    ax.plot(sweep.center, sweep.fooof_pred, "--", color="#247BA0", label="FOOOF theory")
    ax.plot(sweep.center, sweep.split_emp, "o-", color="#2A9D8F", label="SPLIT empirical")
    ax.plot(sweep.center, sweep.split_pred, "--", color="#2A9D8F", label="SPLIT theory")
    ax.axhline(0, color="0.35", lw=1.5)
    grouped_sweep = "n_groups" in sweep.columns and "n_trials" in sweep.columns
    if grouped_sweep:
        ax.set(
            xscale="log",
            xlabel="Oscillation band center (Hz)",
            ylabel="Estimated p–ap Spearman rho",
            ylim=(-1.08, 1.12),
        )
    else:
        ax.set(xscale="log", xlabel="True peak center (Hz)", ylabel="Conditional error r")
    ax.set_xticks([8, 12, 30, 60, 100, 150])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if grouped_sweep:
        sample_label = (
            f"{int(sweep.n_groups.iloc[0])} groups; "
            f"n = {int(sweep.n_trials.iloc[0])} trials/group"
        )
        ax.legend(
            frameon=False,
            ncol=1,
            loc="lower right",
            title=sample_label,
            title_fontsize=10.5,
        )
    else:
        sample_label = f"n = {int(sweep.n_fooof.min())} per center"
        ax.text(0.98, 0.04, sample_label,
                transform=ax.transAxes, ha="right", va="bottom", fontsize=10.5)
        ax.legend(frameon=False, ncol=1, loc="best")
    add_panel_label(ax, "E")

    ax = axes[2, 2]
    series = [null.fooof_r.dropna(), null.fooof_corrected_r.dropna(), null.split_r.dropna()]
    vp = ax.violinplot(series, positions=[0, 1, 2], widths=0.72, showmeans=True, showextrema=False)
    colors = ["#247BA0", "#D1495B", "#2A9D8F"]
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.78)
    vp["cmeans"].set_color("black")
    vp["cmeans"].set_linewidth(3)
    ax.axhline(0, color="0.25", lw=1.8)
    ax.set(xticks=[0, 1, 2], xticklabels=["FOOOF", "FOOOF\ncorrected", "SPLIT"], ylabel="Null-dataset Spearman rho")
    if "n_trials" in null:
        ax.text(
            0.98,
            0.97,
            f"{len(null)} conditions; n = {int(null.n_trials.iloc[0])} trials/condition",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=10.5,
        )
    add_panel_label(ax, "F")

    for ax in axes.ravel():
        thicken(ax)
    fig.canvas.draw()
    positions = [sub.get_position() for sub in b_axes]
    x0 = min(pos.x0 for pos in positions)
    y0 = min(pos.y0 for pos in positions)
    x1 = max(pos.x1 for pos in positions)
    y1 = max(pos.y1 for pos in positions)
    group_pad_x, group_pad_y = 0.012, 0.014
    fig.add_artist(
        Rectangle(
            (x0 - group_pad_x, y0 - group_pad_y),
            (x1 - x0) + 2 * group_pad_x,
            (y1 - y0) + 2 * group_pad_y,
            transform=fig.transFigure,
            fill=False,
            edgecolor="0.55",
            linewidth=2.0,
            clip_on=False,
            zorder=0,
        )
    )
    fig.text(x0 - group_pad_x, y1 + 0.018, "B", fontsize=26, fontweight="bold",
             ha="left", va="bottom")
    fig.text(
        (x0 + x1) / 2,
        y1 + 0.018,
        r"$V_{P,AP}=\sum_f J_P(f)\,[\Sigma J_{AP}^{T}](f)$",
        fontsize=18,
        ha="center",
        va="bottom",
    )
    path = OUT / "figure2_theory_validation_nine_panels.svg"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure2_theory_validation_nine_panels_preview.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_standalone_panel_b(factors: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(12.5, 8.0), constrained_layout=True)
    draw_influence_mechanism_panel(ax, standalone=True)
    path = OUT / "figure2_panel_B_influence_mechanism.svg"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "figure2_panel_B_influence_mechanism_preview.png", dpi=190,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_four_panel(cond: dict, null: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.5), constrained_layout=True)
    mean_o, mean_f = mean_log_spectra(BASE)

    ax = axes[0, 0]
    idx = 2
    ax.plot(FREQ, cond["y_original"][idx], color="0.60", lw=1.8, label="Original PSD")
    ax.plot(FREQ, cond["y_fractal"][idx], color="#D1495B", lw=2.8, label="IRASA fractal")
    ax.plot(FREQ, cond["y_original"][idx] - cond["y_fractal"][idx], color="#2A9D8F", lw=2.8, label="SPLIT oscillation")
    ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel="log10 power")
    ax.set_xticks([0.2, 1, 10, 30, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, loc="lower left")
    add_panel_label(ax, "A")

    ax = axes[0, 1]
    fooof_emp = empirical_corr(cond["fooof"])
    split_emp = empirical_corr(cond["split"])
    fooof_pred = corr_from_cov(cond["v_fooof"])
    split_pred = corr_from_cov(cond["v_split"])
    x = np.arange(2)
    width = 0.34
    b1 = ax.bar(x - width / 2, [fooof_emp, split_emp], width, color="#247BA0", label="Empirical")
    b2 = ax.bar(x + width / 2, [fooof_pred, split_pred], width, color="#D1495B", label="Theory")
    ax.axhline(0, color="0.25", lw=1.8)
    ax.set(xticks=x, xticklabels=["FOOOF", "SPLIT"], ylabel="Conditional error r")
    ax.legend(frameon=False, loc="best")
    all_values = np.array([fooof_emp, split_emp, fooof_pred, split_pred])
    span = max(float(np.nanmax(all_values) - np.nanmin(all_values)), 0.12)
    ax.set_ylim(min(0.0, float(np.nanmin(all_values))) - 0.18 * span,
                max(0.0, float(np.nanmax(all_values))) + 0.24 * span)
    for bars in (b1, b2):
        for bar in bars:
            val = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2,
                    val + np.sign(val if val != 0 else 1) * 0.055 * span,
                    f"{val:.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=13)
    add_panel_label(ax, "B")

    ax = axes[1, 0]
    arrays = [null.fooof_r, null.split_r]
    vp = ax.violinplot(arrays, positions=[0, 1], widths=0.72, showmeans=True, showextrema=False)
    for body, color in zip(vp["bodies"], ["#247BA0", "#2A9D8F"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.80)
    vp["cmeans"].set_color("black")
    vp["cmeans"].set_linewidth(3)
    ax.axhline(0, color="0.25", lw=1.8)
    ax.set(xticks=[0, 1], xticklabels=["FOOOF", "SPLIT"], ylabel="Estimated null Spearman rho")
    add_panel_label(ax, "C")

    ax = axes[1, 1]
    fpr = [(null.fooof_p < 0.05).mean(), (null.split_p < 0.05).mean()]
    bars = ax.bar(["FOOOF", "SPLIT"], fpr, color=["#247BA0", "#2A9D8F"], width=0.62)
    ax.axhline(0.05, color="#D1495B", ls="--", lw=3, label="Nominal 0.05")
    ax.set(ylabel="False-positive rate", ylim=(0, max(0.22, max(fpr) * 1.25)))
    for bar, val in zip(bars, fpr):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.012, f"{val:.2f}", ha="center", fontsize=14)
    ax.legend(frameon=False, loc="upper right")
    add_panel_label(ax, "D")

    for ax in axes.ravel():
        thicken(ax)
    path = OUT / "fooof_split_comparison_four_panels_attached_split.svg"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fooof_split_comparison_four_panels_attached_split_preview.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def metrics(cond: dict, cal: pd.DataFrame, null: pd.DataFrame) -> dict:
    vf, vs = cond["v_fooof"], cond["v_split"]
    cal_f = stats.linregress(cal.fooof_pred, cal.fooof_heldout)
    cal_s = stats.linregress(cal.split_pred, cal.split_heldout)
    return {
        "scope": "simulation-only; smooth one-peak/one-segment branches of attached code",
        "band_specific_oscillation_ranges_hz": {
            name: list(spec["fit_range"]) for name, spec in BANDS.items()
        },
        "attached_aperiodic_range_hz": [0.2, 200.0],
        "conditional": {
            "fooof_empirical_r": empirical_corr(cond["fooof"]),
            "fooof_theory_r": corr_from_cov(vf),
            "split_empirical_r": empirical_corr(cond["split"]),
            "split_theory_r": corr_from_cov(vs),
            "fooof_valid_fraction": float(np.isfinite(cond["fooof"]).all(axis=1).mean()),
            "split_valid_fraction": float(np.isfinite(cond["split"]).all(axis=1).mean()),
        },
        "calibration": {
            "fooof_slope": float(cal_f.slope),
            "fooof_intercept": float(cal_f.intercept),
            "fooof_r": float(cal_f.rvalue),
            "split_slope": float(cal_s.slope),
            "split_intercept": float(cal_s.intercept),
            "split_r": float(cal_s.rvalue),
            "fooof_mean_abs_heldout_r": float(cal.fooof_heldout.abs().mean()),
            "fooof_mean_abs_corrected_r": float(cal.fooof_corrected.abs().mean()),
            "split_mean_abs_heldout_r": float(cal.split_heldout.abs().mean()),
            "split_mean_abs_corrected_r": float(cal.split_corrected.abs().mean()),
            "n_train": int(cal.n_train.iloc[0]),
            "n_heldout": int(cal.n_heldout.iloc[0]),
        },
        "null": {
            "n_datasets": int(len(null)),
            "fooof_mean_spearman": float(null.fooof_r.mean()),
            "fooof_corrected_mean_spearman": float(null.fooof_corrected_r.mean()),
            "split_mean_spearman": float(null.split_r.mean()),
            "fooof_false_positive_rate": float((null.fooof_p < 0.05).mean()),
            "split_false_positive_rate": float((null.split_p < 0.05).mean()),
        },
        "important_limitations": [
            "No empirical original/fractal spectra or per-window FOOOF outputs were supplied, so real-contact panels are not claimed.",
            "Panels D/E extend the supplied 1-30 Hz oscillation search to explicitly stated band-specific ranges; this is an analysis extension, not the literal original script.",
            "The supplied aperiodic script stores every piecewise slope but does not specify which segment downstream analysis pairs with a peak.",
        ],
    }


def main() -> None:
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    cond = conditional_experiment(rng)
    factors = covariance_factor_experiment(rng)
    cal = uniform_theory_calibration_experiment(rng)
    sweep = center_sweep(rng)
    null = null_datasets(rng)
    six = plot_six_panel(cond, cal, sweep, null, factors)
    panel_b = plot_standalone_panel_b(factors)
    cal.to_csv(OUT / "uniform_theory_calibration_conditions.csv", index=False)
    sweep.to_csv(OUT / "center_sweep.csv", index=False)
    factors.to_csv(OUT / "covariance_factor_decomposition.csv", index=False)
    null.to_csv(OUT / "null_dataset_results.csv", index=False)
    result = metrics(cond, cal, null)
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"six_panel": str(six), "panel_b": str(panel_b),
                      "metrics": result}, indent=2))


if __name__ == "__main__":
    main()
