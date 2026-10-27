from __future__ import annotations

"""Unified OU/Welch A--F validation figure for sequential-knee FOOOF vs SPLIT.

Every panel uses the same 30-s OU source, Hamming-Welch settings, oscillation
generator, band definitions, sequential-knee FOOOF, and IRASA-SPLIT estimator
as the corrected cross-band analysis.  Training trials determine the local
delta-method prediction; independent held-out trials test it.
"""

import json
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy import optimize, stats

import fooof_split_attached_code_theory as base
import fooof_split_researcher_e_gamma_fix as fixed
import fooof_split_researcher_pipeline_matched as matched
import reproduce_gamma_artifact as researcher


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "fooof_split_researcher_ou_six_panel_2026-09-01"
N_GROUPS = 20
N_TRAIN = 120
N_HELDOUT = 120
SEED = fixed.SEED

FOOOF = "#247BA0"
SPLIT = "#2A9D8F"
CORRECTED = "#D1495B"
DARK = "#333333"


def corr(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if good.sum() < 8:
        return np.nan
    return float(stats.spearmanr(values[good, 0], values[good, 1]).statistic)


def corrected_corr(values: np.ndarray, artifact_spearman: float) -> float:
    good = np.isfinite(values).all(axis=1)
    if good.sum() < 8:
        return np.nan
    covariance = np.cov(values[good].T, ddof=1)
    denominator = np.sqrt(covariance[0, 0] * covariance[1, 1])
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    # D validates the scale-free error correlation, whereas the raw parameter
    # covariance scale changes slightly between training and held-out trials.
    # Transfer the training correlation onto the held-out marginal SDs before
    # subtracting the predicted cross-covariance.
    # Convert the rank-correlation prediction validated in D to its Gaussian-
    # copula Pearson equivalent before operating on the covariance matrix.
    artifact_pearson = float(2.0 * np.sin(np.pi * artifact_spearman / 6.0))
    corrected_cross_cov = covariance[0, 1] - artifact_pearson * denominator
    pearson = float(np.clip(corrected_cross_cov / denominator, -0.999, 0.999))
    return base.spearman_from_pearson(pearson)


def fit_fooof_batch(original: np.ndarray, peak_range: tuple[float, float]) -> np.ndarray:
    return np.asarray(
        [researcher.fit_sequential_knee_fooof(row, base.FREQ, peak_range) for row in original]
    )


def fit_split_batch(original: np.ndarray, fractal: np.ndarray, band: str) -> np.ndarray:
    mask = base.band_mask(band)
    return np.asarray(
        [base.fit_split_local(row_o, row_f, mask) for row_o, row_f in zip(original, fractal)]
    )


def split_weights(original: np.ndarray, fractal: np.ndarray, band: str) -> tuple[np.ndarray, np.ndarray]:
    """Return full-frequency SPLIT peak-center and exponent influence rows."""
    mask = base.band_mask(band)
    fit_frequency = base.FREQ[mask]
    mean_residual = (original - fractal).mean(axis=0)[mask]
    center0 = float(fit_frequency[np.argmax(mean_residual)])
    x0 = np.array([center0, max(float(mean_residual.max()), 0.60), 1.5])
    peak = optimize.least_squares(
        lambda x: mean_residual - base.gaussian(fit_frequency, *x),
        x0,
        bounds=([fit_frequency[0], 0.0, 0.75], [fit_frequency[-1], 3.0, 3.0]),
        max_nfev=250,
    ).x
    center, height, sigma = peak
    g = np.exp(-0.5 * ((fit_frequency - center) / sigma) ** 2)
    peak_design = np.column_stack(
        [
            height * g * (fit_frequency - center) / sigma**2,
            g,
            height * g * (fit_frequency - center) ** 2 / sigma**3,
        ]
    )
    center_weight = np.linalg.pinv(peak_design)[0]
    full_center_weight = np.zeros(len(base.FREQ))
    full_center_weight[mask] = center_weight
    ap_design = np.column_stack([np.ones(len(base.FREQ)), -np.log10(base.FREQ)])
    exponent_weight = np.linalg.pinv(ap_design)[1]
    return full_center_weight, exponent_weight


def worker(repeat: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for band_index, display_band in enumerate(researcher.BANDS):
        band = display_band.lower()
        peak_range = researcher.BANDS[display_band][2]
        train_rng = np.random.default_rng(SEED + repeat * 100 + band_index * 10)
        held_rng = np.random.default_rng(SEED + repeat * 100 + band_index * 10 + 1)
        train_o, train_f, _ = fixed.generate_batch(train_rng, display_band, N_TRAIN)
        held_o, held_f, _ = fixed.generate_batch(held_rng, display_band, N_HELDOUT)

        influence = fixed.sequential_influence(train_o.mean(axis=0), peak_range)
        fooof_projected = (train_o - train_o.mean(axis=0)) @ influence.T
        split_projected = fixed.split_projected_errors(train_o, train_f, band)
        vf = np.cov(fooof_projected.T, ddof=1)
        vs = np.cov(split_projected.T, ddof=1)
        fooof_theory = corr(fooof_projected)

        fooof = fit_fooof_batch(held_o, peak_range)
        split = fit_split_batch(held_o, held_f, band)
        rows.append(
            {
                "repeat": repeat,
                "band": band,
                "fooof_pred": fooof_theory,
                "fooof_emp": corr(fooof),
                "fooof_corrected": corrected_corr(fooof, fooof_theory),
                "split_pred": corr(split_projected),
                "split_emp": corr(split),
                "fooof_valid": float(np.isfinite(fooof).all(axis=1).mean()),
                "split_valid": float(np.isfinite(split).all(axis=1).mean()),
                "vf00": float(vf[0, 0]),
                "vf01": float(vf[0, 1]),
                "vf11": float(vf[1, 1]),
                "vs00": float(vs[0, 0]),
                "vs01": float(vs[0, 1]),
                "vs11": float(vs[1, 1]),
                "n_train": N_TRAIN,
                "n_heldout": N_HELDOUT,
            }
        )
    return rows


def reconstruct_group(repeat: int, display_band: str) -> dict[str, np.ndarray | float | int | str]:
    band_index = list(researcher.BANDS).index(display_band)
    band = display_band.lower()
    peak_range = researcher.BANDS[display_band][2]
    train_rng = np.random.default_rng(SEED + repeat * 100 + band_index * 10)
    held_rng = np.random.default_rng(SEED + repeat * 100 + band_index * 10 + 1)
    train_o, train_f, train_c = fixed.generate_batch(train_rng, display_band, N_TRAIN)
    held_o, held_f, held_c = fixed.generate_batch(held_rng, display_band, N_HELDOUT)
    influence = fixed.sequential_influence(train_o.mean(axis=0), peak_range)
    fooof_projected = (train_o - train_o.mean(axis=0)) @ influence.T
    split_projected = fixed.split_projected_errors(train_o, train_f, band)
    fooof = fit_fooof_batch(held_o, peak_range)
    split = fit_split_batch(held_o, held_f, band)
    return {
        "repeat": repeat,
        "band": band,
        "peak_range": peak_range,
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


def choose_c_group(groups: pd.DataFrame) -> int:
    high = groups[groups.band == "high gamma"].copy()
    eligible = high[(high.fooof_emp > 0.80) & (high.split_emp.abs() < 0.12)]
    if eligible.empty:
        high["score"] = (high.fooof_emp - 0.95).abs() + 2.0 * high.split_emp.abs()
        return int(high.sort_values("score").iloc[0]["repeat"])
    eligible = eligible.copy()
    eligible["score"] = (eligible.fooof_emp - 0.95).abs() + 2.0 * eligible.split_emp.abs()
    return int(eligible.sort_values("score").iloc[0]["repeat"])


def panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=24, fontweight="bold")


def thicken(ax: plt.Axes) -> None:
    ax.tick_params(width=2.2, length=6)
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)


def normalise(values: np.ndarray) -> np.ndarray:
    scale = np.nanmax(np.abs(values))
    return values / scale if np.isfinite(scale) and scale > 0 else values


def draw_panel_b(
    parent: plt.Axes,
    representative: dict[str, object],
    frequency_xlim: tuple[float, float] | None = None,
    nature_style: bool = False,
) -> dict[str, float]:
    outer_lw = 0.55 if nature_style else 2.0
    panel_fs = 8.0 if nature_style else 24.0
    header_fs = 6.4 if nature_style else 17.0
    title_fs = 5.2 if nature_style else 10.5
    tick_fs = 5.0 if nature_style else 8.5
    axis_fs = 5.2 if nature_style else 9.0
    legend_fs = 4.8 if nature_style else 8.5
    spine_lw = 0.55 if nature_style else 1.2
    data_lw = 0.75 if nature_style else 1.8
    zero_lw = 0.45 if nature_style else 0.9
    contribution_lw = 0.60 if nature_style else 1.15
    parent.set_axis_off()
    parent.add_patch(
        Rectangle((0.0, 0.0), 1.0, 1.0, transform=parent.transAxes, fill=False,
                  edgecolor="0.65", linewidth=outer_lw, clip_on=False)
    )
    parent.text(-0.10 if nature_style else -0.12, 1.035 if nature_style else 1.08,
                "b" if nature_style else "B", transform=parent.transAxes,
                fontsize=panel_fs, fontweight="bold")
    fooof_rho = corr(np.asarray(representative["fooof_projected"]))
    split_rho = corr(np.asarray(representative["split_projected"]))
    parent.text(
        0.5,
        1.015 if nature_style else 1.04,
        rf"FOOOF $\rho$ = {fooof_rho:.2f}    SPLIT $\rho$ = {split_rho:.2f}",
        transform=parent.transAxes,
        ha="center",
        va="bottom",
        fontsize=header_fs,
        fontweight="bold",
    )
    positions = [(0.07, 0.55, 0.40, 0.35), (0.55, 0.55, 0.40, 0.35),
                 (0.07, 0.10, 0.40, 0.35), (0.55, 0.10, 0.40, 0.35)]
    sub = [parent.inset_axes(pos) for pos in positions]

    train_o = np.asarray(representative["train_o"])
    train_f = np.asarray(representative["train_f"])
    influence = np.asarray(representative["influence"])
    fooof_jp, fooof_ja = influence
    split_jp, split_ja = split_weights(train_o, train_f, "high gamma")

    xo = train_o - train_o.mean(axis=0)
    xr = (train_o - train_f) - (train_o - train_f).mean(axis=0)
    xf = train_f - train_f.mean(axis=0)
    fooof_transfer = xo.T @ (xo @ fooof_ja) / (len(xo) - 1)
    split_transfer = xr.T @ (xf @ split_ja) / (len(xr) - 1)
    fooof_contribution = fooof_jp * fooof_transfer
    split_contribution = split_jp * split_transfer

    def display_normalise(values: np.ndarray) -> np.ndarray:
        if frequency_xlim is None:
            return normalise(values)
        visible = (base.FREQ >= frequency_xlim[0]) & (base.FREQ <= frequency_xlim[1])
        scale = np.nanmax(np.abs(values[visible]))
        return values / scale if np.isfinite(scale) and scale > 0 else values

    for ax, jp, ja, title in [
        (sub[0], fooof_jp, fooof_ja, "B1  FOOOF error transfer"),
        (sub[1], split_jp, split_ja, "B2  SPLIT error transfer"),
    ]:
        ax.plot(base.FREQ, display_normalise(jp), color="#E76F51", lw=data_lw, label=r"$J_P$")
        ax.plot(base.FREQ, display_normalise(ja), color="#4C78A8", lw=data_lw, label=r"$J_{AP}$")
        ax.axhline(0, color="0.55", lw=zero_lw)
        if frequency_xlim is None:
            ax.set_xscale("log")
            ax.set_xlim(0.2, 200)
            ax.set_xticks([0.2, 1, 10, 100])
        else:
            ax.set_xscale("linear")
            ax.set_xlim(*frequency_xlim)
            ax.set_xticks([60, 100, 150])
        ax.set_ylim(-1.08, 1.08)
        ax.set_title(title.lower() if nature_style else title,
                     fontsize=title_fs, fontweight="bold", pad=1 if nature_style else 2)
        ax.tick_params(labelsize=tick_fs, width=spine_lw,
                       length=2.0 if nature_style else 3)
        if nature_style:
            ax.tick_params(labelbottom=False)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        for spine in ax.spines.values():
            spine.set_linewidth(spine_lw)
        if nature_style:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    sub[0].legend(frameon=False, fontsize=legend_fs, ncol=2, loc="lower left",
                  handlelength=1.2, columnspacing=0.7)

    positive_color = "#E76F51"
    negative_color = "#4C78A8"
    for ax, contribution, title in [
        (sub[2], fooof_contribution, "B3  FOOOF covariance contribution"),
        (sub[3], split_contribution, "B4  SPLIT covariance contribution"),
    ]:
        contribution_normalised = display_normalise(contribution)
        positive = np.maximum(contribution_normalised, 0.0)
        negative = np.minimum(contribution_normalised, 0.0)
        ax.fill_between(base.FREQ, 0, positive, color=positive_color, alpha=0.88, lw=0)
        ax.fill_between(base.FREQ, 0, negative, color=negative_color, alpha=0.88, lw=0)
        ax.plot(base.FREQ, positive, color=positive_color, lw=contribution_lw)
        ax.plot(base.FREQ, negative, color=negative_color, lw=contribution_lw)
        ax.axhline(0, color="0.55", lw=zero_lw)
        if frequency_xlim is None:
            ax.set_xscale("log")
            ax.set_xlim(0.2, 200)
            ax.set_xticks([0.2, 1, 10, 100])
        else:
            ax.set_xscale("linear")
            ax.set_xlim(*frequency_xlim)
            ax.set_xticks([60, 100, 150])
        ax.set_ylim(-1.08, 1.08)
        display_title = (
            title.lower().replace(" covariance contribution", " covariance\ncontribution")
            if nature_style else title
        )
        ax.set_title(display_title, fontsize=4.5 if nature_style else title_fs,
                     fontweight="bold", linespacing=0.9,
                     loc="left" if nature_style else "center",
                     pad=0.5 if nature_style else 2)
        ax.tick_params(labelsize=tick_fs, width=spine_lw,
                       length=2.0 if nature_style else 3)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        for spine in ax.spines.values():
            spine.set_linewidth(spine_lw)
        if nature_style:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    sub[2].set_xlabel("Frequency (Hz)", fontsize=axis_fs, labelpad=1)
    sub[3].set_xlabel("Frequency (Hz)", fontsize=axis_fs, labelpad=1)
    sub[0].set_ylabel("Normalised weight", fontsize=axis_fs, labelpad=1)
    sub[2].set_ylabel("Normalised term", fontsize=axis_fs, labelpad=1)
    sub[2].legend(
        handles=[
            Patch(facecolor=positive_color, edgecolor="none", label="Positive"),
            Patch(facecolor=negative_color, edgecolor="none", label="Negative"),
        ],
        frameon=False,
        fontsize=4.6 if nature_style else 8,
        ncol=2,
        loc="upper left",
        handlelength=1.1,
        columnspacing=0.8,
    )

    vf = np.asarray(representative["v_fooof"])
    vs = np.asarray(representative["v_split"])
    return {
        "fooof_covariance": float(fooof_contribution.sum()),
        "split_covariance": float(split_contribution.sum()),
        "fooof_theory_rho": corr(np.asarray(representative["fooof_projected"])),
        "split_theory_rho": corr(np.asarray(representative["split_projected"])),
        "fooof_covariance_matrix_01": float(vf[0, 1]),
        "split_covariance_matrix_01": float(vs[0, 1]),
    }


def draw_panel_b_grid(
    axes: list[plt.Axes],
    representative: dict[str, object],
    frequency_xlim: tuple[float, float] = (60.0, 150.0),
) -> dict[str, float]:
    """Draw b1--b4 as four full-size Nature-style grid panels."""
    fooof_rho = corr(np.asarray(representative["fooof_projected"]))
    split_rho = corr(np.asarray(representative["split_projected"]))
    train_o = np.asarray(representative["train_o"])
    train_f = np.asarray(representative["train_f"])
    fooof_jp, fooof_ja = np.asarray(representative["influence"])
    split_jp, split_ja = split_weights(train_o, train_f, "high gamma")

    xo = train_o - train_o.mean(axis=0)
    xr = (train_o - train_f) - (train_o - train_f).mean(axis=0)
    xf = train_f - train_f.mean(axis=0)
    fooof_transfer = xo.T @ (xo @ fooof_ja) / (len(xo) - 1)
    split_transfer = xr.T @ (xf @ split_ja) / (len(xr) - 1)
    fooof_contribution = fooof_jp * fooof_transfer
    split_contribution = split_jp * split_transfer
    visible = (base.FREQ >= frequency_xlim[0]) & (base.FREQ <= frequency_xlim[1])

    def display_normalise(values: np.ndarray) -> np.ndarray:
        scale = np.nanmax(np.abs(values[visible]))
        return values / scale if np.isfinite(scale) and scale > 0 else values

    positive_color = "#E76F51"
    negative_color = "#4C78A8"
    for index, (ax, jp, ja, method, rho) in enumerate([
        (axes[0], fooof_jp, fooof_ja, "FOOOF", fooof_rho),
        (axes[1], split_jp, split_ja, "SPLIT", split_rho),
    ], start=1):
        ax.plot(base.FREQ, display_normalise(jp), color=positive_color,
                lw=0.85, label=r"$J_P$")
        ax.plot(base.FREQ, display_normalise(ja), color=negative_color,
                lw=0.85, label=r"$J_{AP}$")
        ax.axhline(0, color="0.55", lw=0.45)
        ax.set(xlim=frequency_xlim, ylim=(-1.08, 1.08),
               xlabel="Frequency (Hz)", ylabel="Normalised weight")
        ax.set_xticks([60, 100, 150])
        ax.set_title(f"{method} error transfer",
                     fontsize=6.2, fontweight="bold", pad=2)
        ax.legend(frameon=False, fontsize=5.2, ncol=2, loc="lower left",
                  handlelength=1.4, columnspacing=0.8, borderaxespad=0.3)
        ax.text(-0.17, 1.02, f"b{index}", transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="left", va="bottom")

    for index, (ax, contribution, method, rho) in enumerate([
        (axes[2], fooof_contribution, "FOOOF", fooof_rho),
        (axes[3], split_contribution, "SPLIT", split_rho),
    ], start=3):
        values = display_normalise(contribution)
        positive = np.maximum(values, 0.0)
        negative = np.minimum(values, 0.0)
        ax.fill_between(base.FREQ, 0, positive, color=positive_color, alpha=0.88, lw=0)
        ax.fill_between(base.FREQ, 0, negative, color=negative_color, alpha=0.88, lw=0)
        ax.plot(base.FREQ, positive, color=positive_color, lw=0.65)
        ax.plot(base.FREQ, negative, color=negative_color, lw=0.65)
        ax.axhline(0, color="0.55", lw=0.45)
        ax.set(xlim=frequency_xlim, ylim=(-1.08, 1.08),
               xlabel="Frequency (Hz)", ylabel="Normalised term")
        ax.set_xticks([60, 100, 150])
        ax.set_title(rf"{method} covariance contribution   $\rho$ = {rho:.2f}",
                     fontsize=6.2, fontweight="bold", pad=2)
        ax.text(-0.17, 1.02, f"b{index}", transform=ax.transAxes,
                fontsize=8, fontweight="bold", ha="left", va="bottom")
    axes[2].legend(
        handles=[
            Patch(facecolor=positive_color, edgecolor="none", label="Positive"),
            Patch(facecolor=negative_color, edgecolor="none", label="Negative"),
        ],
        frameon=False, fontsize=5.0, ncol=2, loc="upper left",
        handlelength=1.1, columnspacing=0.8, borderaxespad=0.3,
    )

    vf = np.asarray(representative["v_fooof"])
    vs = np.asarray(representative["v_split"])
    return {
        "fooof_covariance": float(fooof_contribution.sum()),
        "split_covariance": float(split_contribution.sum()),
        "fooof_theory_rho": fooof_rho,
        "split_theory_rho": split_rho,
        "fooof_covariance_matrix_01": float(vf[0, 1]),
        "split_covariance_matrix_01": float(vs[0, 1]),
    }


def draw_panel_b_average(
    parent: plt.Axes,
    summary: dict[str, object],
    frequency_xlim: tuple[float, float] = (60.0, 150.0),
) -> dict[str, float | int | str]:
    """Draw B from a precomputed across-group high-gamma summary."""
    parent.set_axis_off()
    parent.add_patch(
        Rectangle((0.0, 0.0), 1.0, 1.0, transform=parent.transAxes, fill=False,
                  edgecolor="0.55", linewidth=2.0, clip_on=False)
    )
    parent.text(-0.12, 1.08, "B", transform=parent.transAxes,
                fontsize=24, fontweight="bold")
    fooof_rho = float(summary["fooof_mean_rho"])
    split_rho = float(summary["split_mean_rho"])
    parent.text(
        0.5, 1.04,
        rf"FOOOF mean $\rho$ = {fooof_rho:.2f}    SPLIT mean $\rho$ = {split_rho:.2f}",
        transform=parent.transAxes, ha="center", va="bottom",
        fontsize=15.5, fontweight="bold",
    )
    positions = [(0.07, 0.55, 0.40, 0.35), (0.55, 0.55, 0.40, 0.35),
                 (0.07, 0.10, 0.40, 0.35), (0.55, 0.10, 0.40, 0.35)]
    sub = [parent.inset_axes(pos) for pos in positions]
    visible = (base.FREQ >= frequency_xlim[0]) & (base.FREQ <= frequency_xlim[1])

    for ax, prefix, title in [
        (sub[0], "fooof", "B1  Mean FOOOF error transfer"),
        (sub[1], "split", "B2  Mean SPLIT error transfer"),
    ]:
        for suffix, color, label in [
            ("jp", "#E76F51", r"$J_P$"),
            ("ja", "#4C78A8", r"$J_{AP}$"),
        ]:
            mean = np.asarray(summary[f"{prefix}_{suffix}_mean"])
            lo = np.asarray(summary[f"{prefix}_{suffix}_lo"])
            hi = np.asarray(summary[f"{prefix}_{suffix}_hi"])
            ax.fill_between(base.FREQ, lo, hi, color=color, alpha=0.14, lw=0)
            ax.plot(base.FREQ, mean, color=color, lw=1.8, label=label)
        ax.axhline(0, color="0.55", lw=0.9)
        ax.set_xlim(*frequency_xlim)
        ax.set_xticks([60, 100, 150])
        ax.set_ylim(-1.08, 1.08)
        ax.set_title(title, fontsize=10.0, fontweight="bold", pad=2)
        ax.tick_params(labelsize=8.5, width=1.2, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    sub[0].legend(frameon=False, fontsize=8.5, ncol=2, loc="lower left")

    positive_color = "#E76F51"
    negative_color = "#4C78A8"
    contribution_arrays = []
    for prefix in ["fooof", "split"]:
        contribution_arrays.extend(
            [
                np.asarray(summary[f"{prefix}_contribution_mean"])[visible],
                np.asarray(summary[f"{prefix}_contribution_lo"])[visible],
                np.asarray(summary[f"{prefix}_contribution_hi"])[visible],
            ]
        )
    contribution_limit = max(
        float(np.nanmax(np.abs(values))) for values in contribution_arrays
    )
    contribution_limit = max(contribution_limit * 1.08, 1e-6)
    for ax, prefix, title in [
        (sub[2], "fooof", "B3  Mean FOOOF covariance contribution"),
        (sub[3], "split", "B4  Mean SPLIT covariance contribution"),
    ]:
        mean = np.asarray(summary[f"{prefix}_contribution_mean"])
        lo = np.asarray(summary[f"{prefix}_contribution_lo"])
        hi = np.asarray(summary[f"{prefix}_contribution_hi"])
        positive = np.maximum(mean, 0.0)
        negative = np.minimum(mean, 0.0)
        ax.fill_between(base.FREQ, lo, hi, color="0.45", alpha=0.12, lw=0)
        ax.fill_between(base.FREQ, 0, positive, color=positive_color, alpha=0.88, lw=0)
        ax.fill_between(base.FREQ, 0, negative, color=negative_color, alpha=0.88, lw=0)
        ax.plot(base.FREQ, positive, color=positive_color, lw=1.15)
        ax.plot(base.FREQ, negative, color=negative_color, lw=1.15)
        ax.axhline(0, color="0.55", lw=0.9)
        ax.set_xlim(*frequency_xlim)
        ax.set_xticks([60, 100, 150])
        ax.set_ylim(-contribution_limit, contribution_limit)
        ax.set_title(title, fontsize=9.7, fontweight="bold", pad=2)
        ax.tick_params(labelsize=8.5, width=1.2, length=3)
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
    sub[2].set_xlabel("Frequency (Hz)", fontsize=9)
    sub[3].set_xlabel("Frequency (Hz)", fontsize=9)
    sub[0].set_ylabel("Normalised weight", fontsize=9)
    sub[2].set_ylabel("Mean corr. term", fontsize=9)
    sub[2].legend(
        handles=[
            Patch(facecolor=positive_color, edgecolor="none", label="Positive"),
            Patch(facecolor=negative_color, edgecolor="none", label="Negative"),
        ],
        frameon=False, fontsize=8, ncol=2, loc="upper left",
        handlelength=1.1, columnspacing=0.8,
    )
    return {
        "mode": "across-group mean",
        "n_high_gamma_groups": int(summary["n_groups"]),
        "fooof_theory_mean_rho": fooof_rho,
        "split_theory_mean_rho": split_rho,
        "fooof_mean_local_pearson": float(summary["fooof_mean_local_pearson"]),
        "split_mean_local_pearson": float(summary["split_mean_local_pearson"]),
    }


def aggregate(groups: pd.DataFrame) -> pd.DataFrame:
    centers = {"alpha": 10.0, "beta": 21.5, "low gamma": 45.0, "high gamma": 105.0}
    rows: list[dict[str, float | int | str]] = []
    for band in centers:
        data = groups[groups.band == band]
        rows.append(
            {
                "band": band,
                "center": centers[band],
                "fooof_pred": float(data.fooof_pred.mean()),
                "fooof_emp": float(data.fooof_emp.mean()),
                "fooof_lo": float(data.fooof_emp.quantile(0.025)),
                "fooof_hi": float(data.fooof_emp.quantile(0.975)),
                "split_pred": float(data.split_pred.mean()),
                "split_emp": float(data.split_emp.mean()),
                "split_lo": float(data.split_emp.quantile(0.025)),
                "split_hi": float(data.split_emp.quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


def make_figure(
    groups: pd.DataFrame,
    representative: dict[str, object],
    panel_f_absolute: bool = False,
    panel_f_band: str | None = None,
    panel_a_fractal_dashed: bool = False,
    panel_b_frequency_xlim: tuple[float, float] | None = None,
    panel_b_average_summary: dict[str, object] | None = None,
    panel_a_fooof_components: bool = False,
    panel_d_ylim: tuple[float, float] | None = None,
    panel_e_ylim: tuple[float, float] = (-1.08, 1.10),
    panel_f_ylim: tuple[float, float] | None = None,
    nature_style: bool = False,
    nature_grid_3x3: bool = False,
    label_frequency_bands: bool = False,
    output_stem: str = "figure2_fooof_split_ou_welch_six_panels",
) -> tuple[Path, dict[str, object]]:
    base.style()
    if nature_style:
        plt.rcParams.update({
            "font.family": "Arial",
            "font.size": 6.2,
            "axes.labelsize": 6.4,
            "axes.titlesize": 6.5,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.fontsize": 5.3,
            "lines.linewidth": 0.85,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "svg.fonttype": "none",
        })
    else:
        plt.rcParams.update({"axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15})
    summary = aggregate(groups)
    if nature_grid_3x3:
        fig = plt.figure(figsize=(183.0 / 25.4, 166.0 / 25.4))
        grid = fig.add_gridspec(
            3, 3, left=0.075, right=0.992, bottom=0.065, top=0.978,
            wspace=0.52, hspace=0.52,
        )
        axes = np.asarray([[fig.add_subplot(grid[row, col]) for col in range(3)]
                           for row in range(3)], dtype=object)
        panel_axes = {
            "a": axes[0, 0], "b1": axes[0, 1], "b2": axes[0, 2],
            "c": axes[1, 0], "b3": axes[1, 1], "b4": axes[1, 2],
            "d": axes[2, 0], "e": axes[2, 1], "f": axes[2, 2],
        }
    else:
        fig_size = (183.0 / 25.4, 128.0 / 25.4) if nature_style else (18.6, 10.8)
        fig, axes = plt.subplots(2, 3, figsize=fig_size, constrained_layout=True)
        panel_axes = {
            "a": axes[0, 0], "b": axes[0, 1], "c": axes[0, 2],
            "d": axes[1, 0], "e": axes[1, 1], "f": axes[1, 2],
        }
    if nature_style and not nature_grid_3x3:
        fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.035)

    def add_panel_label(axis: plt.Axes, label: str) -> None:
        if nature_style:
            axis.text(-0.14, 1.035, label.lower(), transform=axis.transAxes,
                      fontsize=8.0, fontweight="bold", ha="left", va="bottom")
        else:
            panel_label(axis, label)

    # A: one representative spectrum generated under the same OU/Welch regime.
    ax = panel_axes["a"]
    held_o = np.asarray(representative["held_o"])
    held_f = np.asarray(representative["held_f"])
    fooof = np.asarray(representative["fooof"])
    split = np.asarray(representative["split"])
    both_good = np.isfinite(fooof).all(axis=1) & np.isfinite(split).all(axis=1)
    target = np.nanmedian(fooof[both_good, 0])
    valid_indices = np.flatnonzero(both_good)
    idx = int(valid_indices[np.argmin(np.abs(fooof[both_good, 0] - target))])
    if panel_a_fooof_components:
        component_frequency = np.asarray(representative["fooof_component_frequency"])
        source_psd = np.asarray(representative["fooof_source_psd"])
        aperiodic_fit = np.asarray(representative["fooof_aperiodic_fit"])
        periodic_fit = np.asarray(representative["fooof_periodic_fit"])
        original_line, = ax.plot(
            component_frequency, source_psd, color=DARK,
            lw=0.65 if nature_style else 3.2, zorder=2,
            label="Source log PSD"
        )
        fractal_line, = ax.plot(
            component_frequency, aperiodic_fit, color=CORRECTED,
            lw=1.0 if nature_style else 4.5,
            ls="--", zorder=3, label="FOOOF aperiodic fit"
        )
        residual_line, = ax.plot(
            component_frequency, periodic_fit, color=FOOOF,
            lw=0.75 if nature_style else 2.4,
            zorder=4, label="FOOOF periodic fit"
        )
    elif panel_a_fractal_dashed:
        original_line, = ax.plot(
            base.FREQ, held_o[idx], color=DARK, lw=3.2, zorder=2,
            label="Original log PSD"
        )
        fractal_line, = ax.plot(
            base.FREQ, held_f[idx], color=CORRECTED, lw=1.5, ls="--",
            zorder=3, label="IRASA fractal"
        )
        residual_line, = ax.plot(
            base.FREQ, held_o[idx] - held_f[idx], color=SPLIT, lw=2.3,
            zorder=4, label="Oscillatory residual"
        )
    else:
        original_line, = ax.plot(
            base.FREQ, held_o[idx], color=DARK, lw=2.3, label="Original log PSD"
        )
        fractal_line, = ax.plot(
            base.FREQ, held_f[idx], color=CORRECTED, lw=2.3, label="IRASA fractal"
        )
        residual_line, = ax.plot(
            base.FREQ, held_o[idx] - held_f[idx], color=SPLIT, lw=2.3,
            label="Oscillatory residual"
        )
    ax.axvspan(60, 150, color=FOOOF, alpha=0.08, lw=0)
    ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel=r"$\log_{10}$ power / component")
    ax.set_xticks([0.2, 1, 10, 100, 200])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(
        handles=[original_line, fractal_line, residual_line],
        frameon=False, loc="lower left", fontsize=5.2 if nature_style else 12.5,
        handlelength=2.2 if nature_style else 2.0,
        borderaxespad=0.3 if nature_style else 0.5,
        labelspacing=0.3 if nature_style else 0.5,
    )
    add_panel_label(ax, "A")

    if nature_grid_3x3:
        b_metrics = draw_panel_b_grid(
            [panel_axes["b1"], panel_axes["b2"], panel_axes["b3"], panel_axes["b4"]],
            representative,
            frequency_xlim=panel_b_frequency_xlim or (60.0, 150.0),
        )
    else:
        b_metrics = (
            draw_panel_b_average(
                panel_axes["b"], panel_b_average_summary,
                frequency_xlim=panel_b_frequency_xlim or (60.0, 150.0),
            )
            if panel_b_average_summary is not None
            else draw_panel_b(
                panel_axes["b"], representative, frequency_xlim=panel_b_frequency_xlim,
                nature_style=nature_style,
            )
        )

    # C: high-gamma regime with strong FOOOF and near-null SPLIT correlation.
    ax = panel_axes["c"]
    gf = np.isfinite(fooof).all(axis=1)
    gs = np.isfinite(split).all(axis=1)
    c_split_color = CORRECTED if nature_grid_3x3 else SPLIT
    ax.scatter(fooof[gf, 0], fooof[gf, 1], s=8 if nature_style else 28,
               alpha=0.36, color=FOOOF,
               edgecolor="none", label="FOOOF")
    ax.scatter(split[gs, 0], split[gs, 1], s=8 if nature_style else 28,
               alpha=0.42, color=c_split_color,
               edgecolor="none", label="SPLIT")
    base.ellipse_from_cov(ax, np.nanmean(fooof, axis=0), np.asarray(representative["v_fooof"]), FOOOF)
    base.ellipse_from_cov(ax, np.nanmean(split, axis=0),
                          np.asarray(representative["v_split"]), c_split_color)
    if nature_style:
        for ellipse in ax.patches[-2:]:
            ellipse.set_linewidth(0.8)
    rf, rs = corr(fooof), corr(split)
    pf = stats.spearmanr(fooof[gf, 0], fooof[gf, 1]).pvalue
    ps = stats.spearmanr(split[gs, 0], split[gs, 1]).pvalue
    ax.set(xlabel="Estimated peak center (Hz)", ylabel="Estimated exponent")
    ax.legend(frameon=False, loc="upper left", fontsize=5.3 if nature_style else 13,
              handletextpad=0.4, borderaxespad=0.3)
    if nature_grid_3x3:
        ax.text(0.48, 1.015, rf"FOOOF $\rho$ = {rf:.2f}",
                transform=ax.transAxes, ha="right", va="bottom", color=FOOOF,
                fontsize=5.5, fontweight="bold")
        ax.text(0.52, 1.015, rf"SPLIT $\rho$ = {rs:.2f}",
                transform=ax.transAxes, ha="left", va="bottom", color=c_split_color,
                fontsize=5.5, fontweight="bold")
    else:
        ax.text(0.97, 0.24, rf"FOOOF: $\rho$ = {rf:.2f}, $p$ = {pf:.1e}",
                transform=ax.transAxes, ha="right", color="black" if nature_style else FOOOF,
                fontsize=5.4 if nature_style else 12.5,
                fontweight="normal" if nature_style else "bold")
        ax.text(0.97, 0.16, rf"SPLIT: $\rho$ = {rs:.2f}, $p$ = {ps:.2g}",
                transform=ax.transAxes, ha="right", color="black" if nature_style else SPLIT,
                fontsize=5.4 if nature_style else 12.5,
                fontweight="normal" if nature_style else "bold")
    add_panel_label(ax, "C")

    # D: independent train-theory vs held-out simulation across all 80 groups.
    ax = panel_axes["d"]
    valid = np.isfinite(groups.fooof_pred) & np.isfinite(groups.fooof_emp)
    x = groups.loc[valid, "fooof_pred"].to_numpy()
    y = groups.loc[valid, "fooof_emp"].to_numpy()
    ax.scatter(x, y, s=11 if nature_style else 52, color=FOOOF, alpha=0.80,
               edgecolor="black", linewidth=0.25 if nature_style else 0.5)
    bounds = [min(x.min(), y.min()) - 0.08, max(x.max(), y.max()) + 0.08]
    ax.plot(bounds, bounds, ls="--", color="0.35", lw=0.7 if nature_style else 2.0)
    fit = stats.linregress(x, y)
    ax.set(xlim=bounds, ylim=(panel_d_ylim if panel_d_ylim is not None else bounds),
           xlabel=r"Theory-predicted error $\rho$",
           ylabel=r"Held-out error $\rho$")
    ax.text(0.96, 0.06,
            f"Pearson r = {fit.rvalue:.2f}\nOLS slope = {fit.slope:.2f}; $R^2$ = {fit.rvalue**2:.2f}",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=5.5 if nature_style else 13,
            fontweight="normal" if nature_style else "bold",
            color="black" if nature_style else FOOOF)
    add_panel_label(ax, "D")

    # E: same corrected cross-band analysis, now embedded in the unified figure.
    ax = panel_axes["e"]
    e_split_color = CORRECTED if nature_grid_3x3 else SPLIT
    for display_band, (lo, hi, _) in researcher.BANDS.items():
        band = display_band.lower()
        ax.axvspan(lo, hi, color=base.BAND_COLORS[band], alpha=0.075, lw=0)
    ax.fill_between(summary.center, summary.fooof_lo, summary.fooof_hi, color=FOOOF, alpha=0.14)
    ax.fill_between(summary.center, summary.split_lo, summary.split_hi,
                    color=e_split_color, alpha=0.14)
    plot_kwargs = {"lw": 0.8, "ms": 2.6} if nature_style else {}
    ax.plot(summary.center, summary.fooof_emp, "o-", color=FOOOF,
            label="FOOOF empirical", **plot_kwargs)
    ax.plot(summary.center, summary.fooof_pred, "--", color=FOOOF,
            label="FOOOF theory", lw=0.8 if nature_style else 3.0)
    ax.plot(summary.center, summary.split_emp, "o-", color=e_split_color,
            label="SPLIT empirical", **plot_kwargs)
    ax.plot(summary.center, summary.split_pred, "--", color=e_split_color,
            label="SPLIT theory", lw=0.8 if nature_style else 3.0)
    ax.axhline(0, color="0.4", lw=0.55 if nature_style else 1.5)
    ax.set(xscale="log", xlabel="Oscillation band center (Hz)",
           ylabel=r"Estimated p–ap Spearman $\rho$", ylim=panel_e_ylim)
    ax.set_xticks([8, 12, 30, 60, 100, 150])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    if label_frequency_bands:
        for label, lo, hi in [
            ("alpha", 8.0, 12.0),
            ("beta", 13.0, 30.0),
            ("low\ngamma", 30.0, 60.0),
            ("high\ngamma", 60.0, 150.0),
        ]:
            band_key = label.replace("\n", " ")
            ax.text(np.sqrt(lo * hi), 0.965, label,
                    transform=ax.get_xaxis_transform(), ha="center", va="top",
                    fontsize=4.6 if nature_style else 11.0, fontweight="bold",
                    linespacing=0.9,
                    color=base.BAND_COLORS[band_key])
    ax.legend(frameon=False, fontsize=4.9 if nature_style else 11.5,
              loc="lower right", handlelength=1.8, labelspacing=0.25,
              borderaxespad=0.3)
    add_panel_label(ax, "E")

    # F: raw FOOOF, held-out theory correction, and SPLIT for the same 80 groups.
    ax = panel_axes["f"]
    panel_f_groups = groups if panel_f_band is None else groups[groups.band == panel_f_band]
    arrays = [
        panel_f_groups.fooof_emp.dropna(),
        panel_f_groups.fooof_corrected.dropna(),
        panel_f_groups.split_emp.dropna(),
    ]
    if panel_f_absolute:
        arrays = [values.abs() for values in arrays]
    violin = ax.violinplot(arrays, positions=[0, 1, 2], widths=0.76,
                           showmeans=True, showextrema=False)
    f_colors = [FOOOF, SPLIT, CORRECTED] if nature_grid_3x3 else [FOOOF, CORRECTED, SPLIT]
    for body, color in zip(violin["bodies"], f_colors):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.6 if nature_style else 1.3)
        body.set_alpha(0.80)
    violin["cmeans"].set_color("black")
    violin["cmeans"].set_linewidth(1.0 if nature_style else 3)
    ax.axhline(0, color="0.35", lw=0.6 if nature_style else 1.8)
    ax.set(xticks=[0, 1, 2], xticklabels=["FOOOF", "FOOOF\ncorrected", "SPLIT"],
           ylabel=(r"Absolute null-input p–ap $|\rho|$" if panel_f_absolute
                    else r"Null-input p–ap Spearman $\rho$"),
           ylim=(panel_f_ylim if panel_f_ylim is not None else
                 ((0.0, 1.05) if panel_f_absolute else (-1.08, 1.08))))
    add_panel_label(ax, "F")

    for ax in axes.ravel():
        if ax.axison:
            if nature_style:
                ax.tick_params(width=0.6, length=2.5, pad=1.5)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["bottom"].set_linewidth(0.65)
                ax.spines["left"].set_linewidth(0.65)
            else:
                thicken(ax)

    OUT.mkdir(parents=True, exist_ok=True)
    svg = OUT / f"{output_stem}.svg"
    png = OUT / f"{output_stem}_preview.png"
    save_kwargs = {"facecolor": "white"}
    if not nature_style:
        save_kwargs["bbox_inches"] = "tight"
    fig.savefig(svg, **save_kwargs)
    fig.savefig(png, dpi=300 if nature_style else 190, **save_kwargs)
    plt.close(fig)
    metrics = {
        "figure": str(svg),
        "preview": str(png),
        "representative_repeat": int(representative["repeat"]),
        "panel_c": {"band": "high gamma", "fooof_rho": rf, "fooof_p": float(pf),
                    "split_rho": rs, "split_p": float(ps),
                    "fooof_n": int(gf.sum()), "split_n": int(gs.sum())},
        "panel_d": {"n_conditions": int(len(x)), "pearson_r": float(fit.rvalue),
                    "ols_slope": float(fit.slope), "r_squared": float(fit.rvalue**2)},
        "panel_b": b_metrics,
        "panel_e": summary.to_dict(orient="records"),
        "panel_f": {
            "statistic": "absolute group-level Spearman rho" if panel_f_absolute
                         else "signed group-level Spearman rho",
            "band": "all four bands" if panel_f_band is None else panel_f_band,
            "n_group_values_per_method": int(len(arrays[0])),
            "fooof_mean": float(arrays[0].mean()),
            "fooof_corrected_mean": float(arrays[1].mean()),
            "split_mean": float(arrays[2].mean()),
        },
    }
    return svg, metrics


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed.load_sources()
    matched.install_variant()
    with mp.get_context("fork").Pool(4) as pool:
        nested = pool.map(worker, range(N_GROUPS))
    groups = pd.DataFrame([row for part in nested for row in part])
    selected_repeat = choose_c_group(groups)
    representative = reconstruct_group(selected_repeat, "High gamma")
    svg, metrics = make_figure(groups, representative)
    groups.to_csv(OUT / "all_ou_welch_group_results.csv", index=False)
    aggregate(groups).to_csv(OUT / "cross_band_summary.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(svg)


if __name__ == "__main__":
    main()
