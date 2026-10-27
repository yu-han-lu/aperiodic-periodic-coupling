from __future__ import annotations

"""Corrected knee=2-Hz six-panel figure.

Two confounds in the first knee=2 rendering are removed here:
1. FOOOF is the researcher's official knee-mode, max_n_peaks=3 estimator,
   instead of the single-peak sequential approximation.
2. Each simulation group has an independent 30-s OU realization, instead of
   repeatedly sampling trials around one unusually shaped finite OU record.

The local theory uses the active multi-peak FOOOF model Jacobian at the group
mean PSD.  Train spectra estimate the error covariance; independent held-out
spectra test the predicted p--ap correlation.
"""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

import fooof_split_attached_code_theory as base
import fooof_split_researcher_e_gamma_fix as fixed
import fooof_split_researcher_ou_six_panel as six
import fooof_split_researcher_pipeline_matched as matched
import knee_position_official_fooof3_band_sweep as sweep
import reproduce_gamma_artifact as researcher
import rerun_researcher_official_fooof_bands as official


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "fooof_split_knee2_corrected_official3_six_panel_2026-09-01"
TRUE_KNEE_HZ = 2.0
BASE_SEED = 20260901 + 77000


def install_independent_ou(repeat: int) -> float:
    """Install one independent knee=2 OU realization into the shared generator."""
    rng = np.random.default_rng(BASE_SEED + repeat * 10000 + 777)
    innovations = rng.normal(size=sweep.N)
    x = sweep.ou_from_innovations(innovations, TRUE_KNEE_HZ)
    frequency, psd = signal.welch(
        x,
        fs=researcher.FS,
        window="hamming",
        nperseg=len(x),
        noverlap=len(x) // 2,
        nfft=32768,
        detrend="constant",
    )
    fixed.BASE_SIGNAL = x
    fixed.BASE_FREQ = frequency
    fixed.BASE_PSD = psd
    fixed.BASE_LOG_PSD = np.interp(base.FREQ, frequency, np.log10(psd))
    ap = researcher.fit_ap_knee(fixed.BASE_LOG_PSD, base.FREQ)
    return float(10.0 ** (ap[2] / ap[1]))


def official_fit(original: np.ndarray, display_band: str) -> np.ndarray:
    return official.official_fooof(original, display_band, max_n_peaks=3)


def official_multipeak_influence(
    mean_log_psd: np.ndarray, display_band: str
) -> np.ndarray:
    """Local [target CF, exponent] Jacobian for the active official FOOOF model."""
    FOOOFGroup = official._fooof_group_class()
    fit = (base.FREQ >= 0.2) & (base.FREQ <= 197.0)
    frequency = base.FREQ[fit]
    fg = FOOOFGroup(
        aperiodic_mode="knee",
        max_n_peaks=3,
        peak_width_limits=[1.5, 6.0],
        min_peak_height=0.6,
        peak_threshold=2.0,
        verbose=False,
    )
    fg.fit(frequency, 10.0 ** mean_log_psd[None, fit], [0.2, 197.0])
    model = fg.get_fooof(0, regenerate=True)
    ap = np.asarray(model.aperiodic_params_, dtype=float)
    peaks = np.asarray(model.peak_params_, dtype=float)
    peaks = np.empty((0, 3)) if peaks.size == 0 else np.atleast_2d(peaks)
    offset, knee, exponent = ap
    powered = frequency**exponent
    denominator = knee + powered
    ap_design = np.column_stack(
        [
            np.ones_like(frequency),
            -1.0 / (np.log(10.0) * denominator),
            -(powered * np.log(frequency)) / (np.log(10.0) * denominator),
        ]
    )
    columns = [ap_design]
    for center, height, bandwidth in peaks:
        sigma = bandwidth / 2.0
        gaussian = np.exp(-0.5 * ((frequency - center) / sigma) ** 2)
        columns.append(
            np.column_stack(
                [
                    height * gaussian * (frequency - center) / sigma**2,
                    gaussian,
                    0.5 * height * gaussian * (frequency - center) ** 2 / sigma**3,
                ]
            )
        )
    design = np.column_stack(columns)
    parameter_map = np.linalg.pinv(design)
    lo, hi, _ = researcher.BANDS[display_band]
    candidates = np.flatnonzero((peaks[:, 0] >= lo) & (peaks[:, 0] <= hi))
    if not len(candidates):
        return np.full((2, len(base.FREQ)), np.nan)
    target = int(candidates[np.argmax(peaks[candidates, 1])])
    center_row = 3 + 3 * target
    influence = np.zeros((2, len(base.FREQ)))
    influence[0, fit] = parameter_map[center_row]
    influence[1, fit] = parameter_map[2]
    return influence


def worker(repeat: int) -> list[dict[str, float | int | str]]:
    fitted_base_knee = install_independent_ou(repeat)
    rows: list[dict[str, float | int | str]] = []
    for band_index, display_band in enumerate(researcher.BANDS):
        band = display_band.lower()
        train_rng = np.random.default_rng(six.SEED + repeat * 100 + band_index * 10)
        held_rng = np.random.default_rng(six.SEED + repeat * 100 + band_index * 10 + 1)
        train_o, train_f, _ = fixed.generate_batch(train_rng, display_band, six.N_TRAIN)
        held_o, held_f, _ = fixed.generate_batch(held_rng, display_band, six.N_HELDOUT)

        influence = official_multipeak_influence(train_o.mean(axis=0), display_band)
        fooof_projected = (train_o - train_o.mean(axis=0)) @ influence.T
        split_projected = fixed.split_projected_errors(train_o, train_f, band)
        vf = np.cov(fooof_projected.T, ddof=1)
        vs = np.cov(split_projected.T, ddof=1)
        fooof_pred = six.corr(fooof_projected)
        fooof = official_fit(held_o, display_band)
        split = six.fit_split_batch(held_o, held_f, band)
        rows.append(
            {
                "repeat": repeat,
                "band": band,
                "fooof_pred": fooof_pred,
                "fooof_emp": six.corr(fooof),
                "fooof_corrected": six.corrected_corr(fooof, fooof_pred),
                "split_pred": six.corr(split_projected),
                "split_emp": six.corr(split),
                "fooof_valid": float(np.isfinite(fooof).all(axis=1).mean()),
                "split_valid": float(np.isfinite(split).all(axis=1).mean()),
                "vf00": float(vf[0, 0]),
                "vf01": float(vf[0, 1]),
                "vf11": float(vf[1, 1]),
                "vs00": float(vs[0, 0]),
                "vs01": float(vs[0, 1]),
                "vs11": float(vs[1, 1]),
                "fitted_base_knee_hz": fitted_base_knee,
                "n_train": six.N_TRAIN,
                "n_heldout": six.N_HELDOUT,
            }
        )
    return rows


def reconstruct_group(repeat: int, display_band: str) -> dict[str, object]:
    install_independent_ou(repeat)
    band_index = list(researcher.BANDS).index(display_band)
    band = display_band.lower()
    train_rng = np.random.default_rng(six.SEED + repeat * 100 + band_index * 10)
    held_rng = np.random.default_rng(six.SEED + repeat * 100 + band_index * 10 + 1)
    train_o, train_f, train_c = fixed.generate_batch(train_rng, display_band, six.N_TRAIN)
    held_o, held_f, held_c = fixed.generate_batch(held_rng, display_band, six.N_HELDOUT)
    influence = official_multipeak_influence(train_o.mean(axis=0), display_band)
    fooof_projected = (train_o - train_o.mean(axis=0)) @ influence.T
    split_projected = fixed.split_projected_errors(train_o, train_f, band)
    fooof = official_fit(held_o, display_band)
    split = six.fit_split_batch(held_o, held_f, band)
    return {
        "repeat": repeat,
        "band": band,
        "peak_range": researcher.BANDS[display_band][2],
        "train_o": train_o,
        "train_f": train_f,
        "train_c": train_c,
        "held_o": held_o,
        "held_f": held_f,
        "held_c": held_c,
        "influence": influence,
        "fooof_projected": fooof_projected,
        "split_projected": split_projected,
        "fooof": fooof,
        "split": split,
        "v_fooof": np.cov(fooof_projected.T, ddof=1),
        "v_split": np.cov(split_projected.T, ddof=1),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed.load_sources()  # supplies the empirical IRASA residual library
    matched.install_variant()
    official._fooof_group_class()
    with mp.get_context("fork").Pool(4) as pool:
        nested = pool.map(worker, range(six.N_GROUPS))
    groups = pd.DataFrame([row for part in nested for row in part])
    selected_repeat = six.choose_c_group(groups)
    representative = reconstruct_group(selected_repeat, "High gamma")
    six.OUT = OUT
    original_svg, metrics = six.make_figure(groups, representative)
    svg = OUT / "figure2_fooof_split_ou_welch_knee2_official3_corrected_six_panels.svg"
    png = OUT / "figure2_fooof_split_ou_welch_knee2_official3_corrected_six_panels_preview.png"
    original_png = OUT / "figure2_fooof_split_ou_welch_six_panels_preview.png"
    original_svg.replace(svg)
    original_png.replace(png)
    metrics.update(
        {
            "figure": str(svg),
            "preview": str(png),
            "true_knee_hz": TRUE_KNEE_HZ,
            "fooof": "official FOOOF 1.1.1 knee mode, max_n_peaks=3",
            "group_base_design": "20 independent 30-s OU realizations",
            "median_fitted_base_knee_hz": float(groups.fitted_base_knee_hz.median()),
            "fitted_base_knee_q025_q975": [
                float(groups.fitted_base_knee_hz.quantile(0.025)),
                float(groups.fitted_base_knee_hz.quantile(0.975)),
            ],
            "correction_reason": "remove one-peak estimator mismatch and single finite-OU-realization pseudoreplication",
        }
    )
    groups.to_csv(OUT / "all_group_results.csv", index=False)
    six.aggregate(groups).to_csv(OUT / "cross_band_summary.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
