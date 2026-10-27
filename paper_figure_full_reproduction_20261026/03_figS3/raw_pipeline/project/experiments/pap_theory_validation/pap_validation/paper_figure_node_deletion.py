from __future__ import annotations

import argparse
import ast
import copy
import json
import time
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib import ticker
from scipy import ndimage, signal, stats

from .core import AREAS9, apply_fln_delta, feature_from_spectrum, frequency_grid, input_psd, load_model
from .exp1 import Z0, ZHI, ZLO, ZSCALE
from .exp2_pap_correlation_validation import (
    ANALYSIS_S as EXP2_ANALYSIS_S,
    BURN_S as EXP2_BURN_S,
    DT_S as EXP2_DT_S,
    _extract_features as exp2_extract_features,
    _forced_ei_batch_parallel,
    _forcing as exp2_forcing,
    _input_q as exp2_input_q,
    _theory_feature as exp2_theory_feature,
    _transfer_power as exp2_transfer_power,
)
from .exp2_two_stage import _jacobian as exp2_jacobian
from .exp2_two_stage import _matrix as exp2_matrix
from .exp2_two_stage import _setup as exp2_setup
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .exp3_large_delta_screen import _all_transfer_power, _continuous_transfer, _discrete_transfer, _exact_trial_features
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config
from .nonlinear_network_validation import _run_multioutput_batch
from .plot_formatting import format_correlation, format_p_value
from .realization_residual_validation import _corr, _cov_to_corr, _pooled_within_trial_cov


OUTPUT_NAME = "paper_figure_node_deletion"
CALIBRATION_PHASES = 3
HELDOUT_PHASES = 5
TOTAL_PHASES = CALIBRATION_PHASES + HELDOUT_PHASES
DT_MS = 0.25
EXP3_ANALYSIS_S = 65.536
EXP3_TRIALS = 512
EXP3_CHUNK = 16
WELCH_SEGMENT_S = 4.096
SMOOTHING_SIGMA_HZ = 0.9765625
SEED = 20260807
PSD_AXES_LINEWIDTH = 2.4
PSD_LABELSIZE = 17
PSD_TITLESIZE = 18
PSD_TICKSIZE = 15


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def _output(cfg: dict) -> Path:
    out = _rooted(cfg["output_dir"]) / OUTPUT_NAME
    (out / "panels").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "exp3_phase_cache").mkdir(parents=True, exist_ok=True)
    return out


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "axes.linewidth": 1.9,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12.5,
            "lines.linewidth": 3.0,
            "xtick.major.width": 1.7,
            "ytick.major.width": 1.7,
            "savefig.dpi": 300,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.12, label, transform=ax.transAxes, fontsize=23, fontweight="bold", va="top")


def _style_psd_axes(axes) -> None:
    for ax in np.atleast_1d(axes).ravel():
        ax.xaxis.label.set_size(PSD_LABELSIZE)
        ax.yaxis.label.set_size(PSD_LABELSIZE)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_weight("bold")
        ax.title.set_size(PSD_TITLESIZE)
        ax.title.set_weight("bold")
        ax.tick_params(axis="both", which="major", labelsize=PSD_TICKSIZE, width=PSD_AXES_LINEWIDTH, length=6)
        ax.tick_params(axis="both", which="minor", width=1.5, length=3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(PSD_AXES_LINEWIDTH)


def _sparse_log_y_ticks(ax: plt.Axes, count: int = 3) -> None:
    low, high = ax.get_ylim()
    lo = int(np.floor(np.log10(max(low, 1e-300))))
    hi = int(np.ceil(np.log10(max(high, 1e-300))))
    if hi <= lo:
        hi = lo + 1
    exponents = np.unique(np.rint(np.linspace(lo, hi, count)).astype(int))
    ax.set_ylim(10.0**lo, 10.0**hi)
    ax.yaxis.set_major_locator(ticker.FixedLocator(10.0 ** exponents))
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())


def _welch(x: np.ndarray, dt_s: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = int(round(WELCH_SEGMENT_S / dt_s))
    return signal.welch(
        x,
        fs=1.0 / dt_s,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
    )


def _smooth_psd(freq: np.ndarray, psd: np.ndarray) -> np.ndarray:
    keep = (freq >= 0.5) & (freq <= 1200.0)
    f = freq[keep]
    bins = SMOOTHING_SIGMA_HZ / (f[1] - f[0])
    smooth = np.exp(ndimage.gaussian_filter1d(np.log(np.maximum(psd[keep], 1e-300)), bins))
    return f, smooth


def _fit_slope_line(freq: np.ndarray, psd: np.ndarray, band=(100.0, 1000.0)):
    use = (freq >= band[0]) & (freq <= band[1])
    fit = stats.linregress(np.log10(freq[use]), np.log10(psd[use]))
    line_f = np.geomspace(band[0], band[1], 80)
    line_y = 10 ** (fit.intercept + fit.slope * np.log10(line_f))
    return fit, line_f, line_y


def make_panel_a(cfg: dict, model, out: Path) -> dict:
    analysis_s = 32.768
    burn_s = float(cfg["exp1_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000 / DT_MS))
    n_steps = int(round(analysis_s * 1000 / DT_MS))
    z = Z0[None, :]
    x = _matched_bandlimited_inputs(
        z,
        DT_MS,
        analysis_s,
        burn_s,
        float(cfg["exp1_nonlinear"]["input_sigma"]),
        SEED + 10,
    )[0]
    input_idx = model.areas.index("V1")
    output_idx = model.areas.index("PG")
    transfer = _discrete_transfer(model.J, input_idx, output_idx, n_steps, float(model.params["tau_exc"]))
    y = np.fft.irfft(np.fft.rfft(x[burn_steps:]) * transfer, n=n_steps)
    fi, pi = _welch(x[burn_steps:], DT_MS / 1000.0)
    fo, po = _welch(y, DT_MS / 1000.0)
    fi, pi = _smooth_psd(fi, pi)
    fo, po = _smooth_psd(fo, po)
    feature = feature_from_spectrum(fo, po, peak_band=(5.0, 20.0), slope_band=(100.0, 1000.0))
    fit, line_f, line_y = _fit_slope_line(fo, po)

    theory_f = np.geomspace(0.5, 1200.0, 1000)
    background = Z0[3] * (theory_f / 10.0) ** Z0[2]
    bump = Z0[1] * np.exp(-0.5 * ((theory_f - Z0[0]) / 2.0) ** 2)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    ax = axes[0]
    empirical_norm = np.max(pi[(fi >= .5) & (fi <= 1200)])
    empirical = pi / empirical_norm
    total = background + bump
    norm = total.max()
    ax.loglog(fi, empirical, color="#222222", ls="--", lw=3.1, label="input PSD", zorder=3)
    ax.loglog(theory_f, background / norm, color="#78ABC1", ls="-", lw=5.0, label="aperiodic fit")
    ax.loglog(theory_f, np.maximum(bump / norm, 1e-4), color="#D1495B", ls="--", lw=3.0, label="target bump")
    ax.set(xlim=(0.7, 1200), ylim=(1e-4, 2), xlabel="Frequency (Hz)", ylabel="Normalized PSD", title="Input: bump + aperiodic")
    ax.set_xticks([1, 10, 100, 1000])
    ax.legend(frameon=False, loc="lower left")
    _panel_label(ax, "A")
    ax = axes[1]
    ax.loglog(line_f, line_y, color="#78ABC1", ls="-", lw=5.0, alpha=.95,
              label=f"slope={fit.slope:.2f}", zorder=1)
    ax.loglog(fo, po, color="#222222", ls="--", lw=3.1, label="exact PSD", zorder=3)
    peak = feature["peak_hz"]
    peak_y = float(np.interp(peak, fo, po))
    ax.scatter([peak], [peak_y], s=55, color="#D1495B", zorder=4, label=f"peak={peak:.1f} Hz")
    ax.set(xlim=(0.7, 1200), ylim=(1e-18, 1e-4), xlabel="Frequency (Hz)", ylabel="PSD")
    ax.set_xticks([1, 10, 100, 1000])
    ax.set_yticks([1e-15, 1e-10, 1e-5])
    ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10))
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())
    ax.legend(frameon=False, loc="lower left")
    _style_psd_axes(axes)
    _save(fig, out / "panels" / "A_baseline_input_output_psd")
    np.savez_compressed(out / "panel_A_psd_data.npz", input_freq=fi, input_psd=pi, output_freq=fo, output_psd=po)
    return {"peak_hz": peak, "slope": float(fit.slope)}


def make_panel_b(out: Path) -> dict:
    source = out.parent / "realization_residual_validation" / "exp1_condition_results.csv.gz"
    data = pd.read_csv(source)
    pair = data[(data.input_area == "V1") & (data.output_area == "PG")].copy()
    fig, ax = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
    pred = "first_residual_corrected_r"
    x = pair[pred].to_numpy()
    y = pair.heldout_r_mean.to_numpy()
    identity_r2 = 1.0 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
    metrics = {pred: float(identity_r2)}
    ax.scatter(x, y, s=48, marker="o", color="#247BA0", edgecolor="white", linewidth=.5,
               label=f"1st order + realization (R2={identity_r2:.3f})")
    lo = min(pair.heldout_r_mean.min(), pair[pred].min()) - .04
    hi = max(pair.heldout_r_mean.max(), pair[pred].max()) + .04
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="identity")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Theory correlation", ylabel="Simulation correlation", title="Experiment 1: V1 input, PG readout")
    ax.legend(frameon=False, loc="upper left")
    _panel_label(ax, "B")
    _save(fig, out / "panels" / "B_exp1_V1_PG_theory_simulation")
    return metrics


def make_panel_c(out: Path, condition_pos: int = 1) -> dict:
    phase_dir = out.parent / "realization_residual_validation" / "exp1_phase_cache"
    matrix = np.full((9, 9), np.nan)
    pvalues = np.full((9, 9), np.nan)
    for i, input_area in enumerate(AREAS9):
        heldout = []
        heldout_valid = []
        for phase in range(CALIBRATION_PHASES, TOTAL_PHASES):
            d = np.load(phase_dir / f"phase{phase:02d}_input_{input_area}.npz")
            heldout.append(d["features"][condition_pos])
            heldout_valid.append(d["valid"][condition_pos])
        features = np.mean(np.asarray(heldout), axis=0)
        valid = np.all(np.asarray(heldout_valid), axis=0)
        for j in range(9):
            use = valid[j] & np.isfinite(features[j]).all(axis=1)
            result = stats.pearsonr(features[j, use, 0], features[j, use, 1])
            matrix[j, i] = result.statistic
            pvalues[j, i] = result.pvalue
    fig, ax = plt.subplots(figsize=(5.0, 4.3), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(9), AREAS9, rotation=45, ha="right")
    ax.set_yticks(range(9), AREAS9)
    ax.set(xlabel="Input area", ylabel="Readout area",
           title="Experiment 1: rank-one input, 32.768-s window")
    for row in range(9):
        for col in range(9):
            if pvalues[row, col] < 0.05:
                color = "white" if abs(matrix[row, col]) > .48 else "black"
                ax.text(col, row, format_correlation(matrix[row, col]), ha="center", va="center", fontsize=6.5, color=color, fontweight="bold")
    cb = fig.colorbar(image, ax=ax, shrink=.78, pad=.02)
    cb.set_label("Pearson r")
    cb.set_ticks([-1, 0, 1])
    ax.text(-0.18, 1.22, "C", transform=ax.transAxes, fontsize=23, fontweight="bold", va="top")
    _save(fig, out / "panels" / "C_exp1_9x9_correlation_matrix")
    pd.DataFrame(matrix, index=AREAS9, columns=AREAS9).to_csv(out / "panel_C_correlation_matrix.csv")
    pd.DataFrame(pvalues, index=AREAS9, columns=AREAS9).to_csv(out / "panel_C_pvalue_matrix.csv")
    return {
        "condition_pos": condition_pos,
        "condition": "rank_one_eps0.2_rho0",
        "analysis_s": 32.768,
        "n_significant": int((pvalues < .05).sum()),
        "min_r": float(np.nanmin(matrix)),
        "max_r": float(np.nanmax(matrix)),
    }


def make_panel_d(out: Path) -> dict:
    source = out.parent / "exp2_two_stage_scan" / "two_stage_eigen_cross_spectrum.csv"
    scan = pd.read_csv(source)
    manifest = json.loads((out.parent / "exp2_two_stage_scan" / "manifest.json").read_text())
    complex_onset = float(manifest["complex_eigenvalue_onset_kappa_I"])
    hopf = float(manifest["hopf_E_E_gain"])
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7))
    fig.subplots_adjust(left=.10, right=.91, top=.86, bottom=.25, wspace=.62)
    legend_lines = None
    for ax, stage, boundary, xlabel in (
        (axes[0], "I_to_E", complex_onset, r"I-to-E multiplier, $s_{EI}$"),
        (axes[1], "E_to_E", hopf, r"E-to-E multiplier, $s_{EE}$"),
    ):
        d = scan[scan.stage == stage]
        ax.plot(d.scan_value, d.critical_real_per_s, color="#247BA0", label=r"$\mathrm{Re}(\lambda)$")
        ax2 = ax.twinx()
        ax2.plot(d.scan_value, d.critical_imag_per_s, color="#D1495B", label=r"$|\mathrm{Im}(\lambda)|$")
        ax.axhline(0, color="0.25", ls="--", lw=.75)
        ax.axvline(boundary, color="#6A4C93", ls=":", lw=1.1)
        ax.set(
            xlabel=xlabel,
            ylabel=r"$\mathrm{Re}(\lambda)$ (s$^{-1}$)",
            title="Complex onset" if stage == "I_to_E" else "Hopf crossing",
        )
        ax2.set_ylabel(r"$|\mathrm{Im}(\lambda)|$ (s$^{-1}$)", color="#D1495B")
        lines = ax.lines[:1] + ax2.lines[:1]
        legend_lines = lines
    fig.legend(
        legend_lines,
        [line.get_label() for line in legend_lines],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(.5, .015),
        ncol=2,
    )
    _panel_label(axes[0], "D")
    _save(fig, out / "panels" / "D_exp2_eigenvalue_transitions")
    return {"complex_onset_s_EI": complex_onset, "hopf_s_EE": hopf}


def _exp2_single_input(beta=-0.5, input_sigma=.02, floor_ratio=.1, seed=SEED + 50):
    forcing = exp2_forcing(np.asarray([beta]), input_sigma, floor_ratio, seed)
    burn = int(round(EXP2_BURN_S / EXP2_DT_S))
    return forcing, forcing[0, burn:]


def make_panel_e(out: Path) -> dict:
    original, tau, kappa_target, _ = exp2_setup()
    manifest = json.loads((out.parent / "exp2_two_stage_scan" / "manifest.json").read_text())
    hopf = float(manifest["hopf_E_E_gain"])
    forcing, input_trace = _exp2_single_input()
    regimes = [
        ("Real modes", 1.0, 1.0, "#247BA0"),
        ("Stable complex", kappa_target, 1.0, "#2A9D8F"),
        ("Beyond Hopf", kappa_target, hopf + .03, "#D1495B"),
    ]
    fi, pi = _welch(input_trace, EXP2_DT_S)
    paths = []
    for _, kappa, see, _ in regimes:
        w = exp2_matrix(original, kappa, see)
        paths.append(_forced_ei_batch_parallel(w, tau, forcing, EXP2_DT_S, int(round(EXP2_BURN_S / EXP2_DT_S)))[0])
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.1), constrained_layout=True)
    axes[0].loglog(fi, pi, color="#222222")
    axes[0].set(title="Common input", xlabel="Frequency (Hz)", ylabel="PSD", xlim=(.7, 1200))
    axes[0].set_xticks([1, 10, 100, 1000])
    _sparse_log_y_ticks(axes[0], 3)
    peak_rows = []
    for ax, path, (name, kappa, see, color) in zip(axes[1:], paths, regimes):
        f, p = _welch(path, EXP2_DT_S)
        ax.loglog(f, p, color=color)
        band = (f >= .5) & (f <= 100)
        peak = float(f[band][np.argmax(p[band])])
        ax.axvline(peak, color="0.25", ls="--", lw=1.2)
        ax.set(title=name, xlabel="Frequency (Hz)", xlim=(.7, 1200))
        ax.set_xticks([1, 10, 100, 1000])
        _sparse_log_y_ticks(ax, 3)
        peak_rows.append({"regime": name, "s_EI": kappa, "s_EE": see, "peak_hz": peak})
    _style_psd_axes(axes)
    _panel_label(axes[0], "E")
    _save(fig, out / "panels" / "E_exp2_three_regime_psd")
    pd.DataFrame(peak_rows).to_csv(out / "panel_E_regimes.csv", index=False)
    return {"regimes": peak_rows}


def make_panel_f(out: Path) -> dict:
    original, tau, kappa_target, _ = exp2_setup()
    manifest = json.loads((out.parent / "exp2_two_stage_scan" / "manifest.json").read_text())
    hopf = float(manifest["hopf_E_E_gain"])
    forcing, input_trace = _exp2_single_input(beta=-0.5, floor_ratio=.1, seed=SEED + 60)
    w = exp2_matrix(original, kappa_target, hopf - .05)
    output_trace = _forced_ei_batch_parallel(w, tau, forcing, EXP2_DT_S, int(round(EXP2_BURN_S / EXP2_DT_S)))[0]
    fi, pi = _welch(input_trace, EXP2_DT_S)
    fo, po = _welch(output_trace, EXP2_DT_S)
    fo, po = _smooth_psd(fo, po)
    feature = feature_from_spectrum(fo, po, peak_band=(5.0, 100.0), slope_band=(100.0, 1000.0))
    fit, line_f, line_y = _fit_slope_line(fo, po)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), constrained_layout=True)
    axes[0].loglog(fi, pi, color="#222222")
    axes[0].set(xlim=(.7, 1200), xlabel="Frequency (Hz)", ylabel="PSD", title="Aperiodic-only input")
    axes[0].set_xticks([1, 10, 100, 1000])
    axes[1].loglog(line_f, line_y, color="#78ABC1", ls="-", lw=3.6, alpha=.9,
                   label=f"fit slope={fit.slope:.2f}", zorder=1)
    axes[1].loglog(fo, po, color="#222222", ls="--", lw=1.5,
                   label="stable-complex PSD", zorder=3)
    peak = feature["peak_hz"]
    axes[1].scatter([peak], [np.interp(peak, fo, po)], s=55, color="#D1495B", zorder=4, label=f"peak={peak:.1f} Hz")
    axes[1].set(xlim=(.7, 1200), xlabel="Frequency (Hz)", ylabel="PSD", title="Stable-complex output")
    axes[1].set_xticks([1, 10, 100, 1000])
    _style_psd_axes(axes)
    axes[1].legend(frameon=False, loc="lower left", fontsize=12)
    _panel_label(axes[0], "F")
    _save(fig, out / "panels" / "F_exp2_nearcritical_input_output_psd")
    return {"s_EI": kappa_target, "s_EE": hopf - .05, "peak_hz": peak, "slope": float(fit.slope)}


def _exp2_1000_results(out: Path) -> pd.DataFrame:
    result_path = out / "exp2_1000_valid_trial_conditions.csv"
    if result_path.exists():
        return pd.read_csv(result_path)
    source = out.parent / "exp2_pap_correlation_validation"
    condition_ids = [227, 229, 233, 235, 239, 241, 245, 247, 251, 253]
    original, tau, _, _ = exp2_setup()
    rows = []
    for condition_id in condition_ids:
        d = np.load(source / "condition_features" / f"condition_{condition_id:04d}.npz", allow_pickle=True)
        betas = d["betas"]
        nonlinear = d["nonlinear"].astype(float)
        valid = d["nonlinear_valid"]
        s_value, sigma_beta, floor_ratio, input_sigma = d["condition"].astype(float)
        w = exp2_matrix(original, s_value, 1.0)
        j = exp2_jacobian(original, tau, s_value, 1.0)
        f = np.geomspace(.5, 1200.0, 1600)
        transfer = exp2_transfer_power(j, float(tau[0]), f)
        theory = np.asarray([exp2_theory_feature(transfer, f, beta, input_sigma, floor_ratio)[0] for beta in betas])
        common = valid.all(axis=0) & np.isfinite(theory).all(axis=1)
        ids = np.flatnonzero(common)[:100]
        if len(ids) != 100:
            raise RuntimeError(f"Condition {condition_id} has only {len(ids)} common valid trials")
        calibration = nonlinear[:CALIBRATION_PHASES, ids]
        residual_cov = _pooled_within_trial_cov(calibration)
        signal_cov = np.cov(theory[ids], rowvar=False, ddof=1)
        predicted = _cov_to_corr(signal_cov + residual_cov)
        observed_phase = np.asarray([_corr(nonlinear[p, ids]) for p in range(CALIBRATION_PHASES, TOTAL_PHASES)])
        rows.append(
            {
                "condition_id": condition_id,
                "S": s_value,
                "sigma_beta": sigma_beta,
                "floor_ratio": floor_ratio,
                "input_sigma": input_sigma,
                "n_valid_used": 100,
                "theory_plus_realization_r": predicted,
                "simulation_r": float(observed_phase.mean()),
                "simulation_r_sd": float(observed_phase.std(ddof=1)),
                "latent_theory_r": _cov_to_corr(signal_cov),
                "residual_only_r": _cov_to_corr(residual_cov),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(result_path, index=False)
    return result


def make_panel_g(out: Path) -> dict:
    data = _exp2_1000_results(out)
    x = data.theory_plus_realization_r.to_numpy()
    y = data.simulation_r.to_numpy()
    identity_r2 = 1.0 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
    fit = stats.linregress(x, y)
    lo = min(x.min(), y.min()) - .05
    hi = max(x.max(), y.max()) + .05
    fig, ax = plt.subplots(figsize=(4.2, 3.8), constrained_layout=True)
    scatter = ax.scatter(x, y, c=data.sigma_beta, cmap="viridis", s=65, edgecolor="white", linewidth=.5)
    ax.errorbar(x, y, yerr=data.simulation_r_sd, fmt="none", ecolor="0.45", lw=1, capsize=2, zorder=0)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="identity")
    grid = np.linspace(lo, hi, 100)
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#D1495B", label="OLS")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Theory + realization correlation", ylabel="Nonlinear simulation correlation", title="Experiment 2: 1000 valid trials")
    ax.text(.04, .96, f"identity R2={identity_r2:.3f}\nOLS R2={fit.rvalue**2:.3f}\np={format_p_value(fit.pvalue)}", transform=ax.transAxes, va="top")
    ax.legend(frameon=False, loc="lower right")
    cb = fig.colorbar(scatter, ax=ax, shrink=.78, pad=.02)
    cb.set_label("SD(beta)")
    ax.text(-0.18, 1.22, "G", transform=ax.transAxes, fontsize=23, fontweight="bold", va="top")
    _save(fig, out / "panels" / "G_exp2_1000trial_theory_simulation")
    return {"identity_R2": float(identity_r2), "ols_R2": float(fit.rvalue**2), "p": float(fit.pvalue), "n_conditions": 10, "n_trials_total": 1000}


def _node_deleted_j(model, nodes: list[str]):
    delta = np.zeros_like(model.fln)
    for name in nodes:
        idx = model.areas.index(name)
        delta[idx, :] = -model.fln[idx, :]
        delta[:, idx] = -model.fln[:, idx]
    return apply_fln_delta(model, delta), delta


def _feature_gradients(transfers: np.ndarray, frequencies: np.ndarray, h: float = 2e-3) -> np.ndarray:
    gradients = np.empty((9, 9, 2, 4), dtype=float)
    for q in range(4):
        zp = Z0 + h * ZSCALE * np.eye(4)[q]
        zm = Z0 - h * ZSCALE * np.eye(4)[q]
        for i in range(9):
            for j in range(9):
                fp = feature_from_spectrum(frequencies, transfers[i, j] * input_psd(frequencies, zp))
                fm = feature_from_spectrum(frequencies, transfers[i, j] * input_psd(frequencies, zm))
                gradients[i, j, :, q] = (
                    np.asarray([fp["peak_omega"], fp["exponent"]])
                    - np.asarray([fm["peak_omega"], fm["exponent"]])
                ) / (2 * h)
    return gradients


def _pair_gradient(transfer: np.ndarray, frequencies: np.ndarray, h: float = 2e-3) -> np.ndarray:
    gradient = np.empty((2, 4), dtype=float)
    for q in range(4):
        zp = Z0 + h * ZSCALE * np.eye(4)[q]
        zm = Z0 - h * ZSCALE * np.eye(4)[q]
        fp = feature_from_spectrum(frequencies, transfer * input_psd(frequencies, zp))
        fm = feature_from_spectrum(frequencies, transfer * input_psd(frequencies, zm))
        gradient[:, q] = (
            np.asarray([fp["peak_omega"], fp["exponent"]])
            - np.asarray([fm["peak_omega"], fm["exponent"]])
        ) / (2 * h)
    return gradient


def _linear_corr(gradient: np.ndarray, weights: np.ndarray) -> float:
    gp, ga = gradient
    cov = float(np.sum(weights * gp * ga))
    vp = float(np.sum(weights * gp * gp))
    va = float(np.sum(weights * ga * ga))
    return cov / np.sqrt(vp * va) if vp > 0 and va > 0 else np.nan


def _screen_node_deletions(model, out: Path, force: bool = False) -> pd.DataFrame:
    path = out / "exp3_node_screen_fullcov_v2.csv.gz"
    if path.exists() and not force:
        return pd.read_csv(path)
    rng = np.random.default_rng(SEED + 100)
    frequencies = frequency_grid()
    indices = [model.areas.index(area) for area in AREAS9]
    base_transfers = _all_transfer_power(model.J, indices, frequencies)
    base_grad = _feature_gradients(base_transfers, frequencies)
    candidates = []
    seen = set()
    eligible = [area for area in model.areas]
    for size in (1, 2, 3, 5, 8, 12):
        target = 55 if size < 5 else 80
        while sum(len(c) == size for c in candidates) < target:
            nodes = tuple(sorted(rng.choice(eligible, size=size, replace=False).tolist()))
            if nodes not in seen:
                seen.add(nodes)
                candidates.append(nodes)
    rows = []
    for pos, nodes in enumerate(candidates):
        j, _ = _node_deleted_j(model, list(nodes))
        transfers = _all_transfer_power(j, indices, frequencies)
        pert_grad = _feature_gradients(transfers, frequencies)
        for i, input_area in enumerate(AREAS9):
            if input_area in nodes:
                continue
            for o, output_area in enumerate(AREAS9):
                if output_area in nodes:
                    continue
                gb = base_grad[i, o]
                gd = pert_grad[i, o]
                if np.linalg.matrix_rank(gb) < 2:
                    continue
                right_inverse = np.linalg.pinv(gb)
                for variance_ratio in np.geomspace(.02, 50.0, 17):
                    target_cov = np.diag([variance_ratio, 1.0])
                    sigma = right_inverse @ target_cov @ right_inverse.T
                    sigma /= max(float(np.max(np.diag(sigma))), 1e-15)
                    cb = gb @ sigma @ gb.T
                    cd = gd @ sigma @ gd.T
                    rb = _cov_to_corr(cb)
                    rp = _cov_to_corr(cd)
                    rows.append(
                        {
                            "nodes": json.dumps(nodes),
                            "n_nodes": len(nodes),
                            "input_area": input_area,
                            "output_area": output_area,
                            "variance_ratio": float(variance_ratio),
                            "sigma_json": json.dumps(sigma.tolist()),
                            "linear_baseline_r": rb,
                            "linear_deleted_r": rp,
                            "linear_delta_r": rp - rb,
                        }
                    )
        if (pos + 1) % 20 == 0:
            print(f"Node-deletion analytic screen {pos + 1}/{len(candidates)}", flush=True)
    result = pd.DataFrame(rows).sort_values("linear_deleted_r", key=lambda x: abs(x), ascending=False)
    result.to_csv(path, index=False, compression="gzip")
    return result


def _exact_candidate_refinement(model, screen: pd.DataFrame, out: Path) -> dict:
    path = out / "exp3_selected_candidate_robust_v3.json"
    if path.exists():
        return json.loads(path.read_text())
    existing = out / "exp3_exact_candidate_refinement.csv"
    if existing.exists():
        candidates = pd.read_csv(existing)
    else:
        candidates = None
    rng = np.random.default_rng(SEED + 200)
    normals = rng.normal(size=(1024, 4))
    rows = []
    shortlist = screen.head(36)
    frequencies = frequency_grid()
    cache = {}
    for candidate_pos, (_, item) in enumerate(shortlist.iterrows() if candidates is None else []):
        nodes = tuple(json.loads(item.nodes))
        key = (nodes, item.input_area, item.output_area)
        input_idx = model.areas.index(item.input_area)
        output_idx = model.areas.index(item.output_area)
        if key not in cache:
            j, _ = _node_deleted_j(model, list(nodes))
            cache[key] = (
                j,
                _continuous_transfer(model.J, input_idx, output_idx),
                _continuous_transfer(j, input_idx, output_idx),
            )
        j, tb, td = cache[key]
        gb = _pair_gradient(tb, frequencies)
        right_inverse = np.linalg.pinv(gb)
        variance_ratio = float(item.variance_ratio)
        for baseline_target_r in (-.25, -.15, -.08, 0.0, .08, .15, .25):
            target_cov = np.asarray(
                [
                    [variance_ratio, baseline_target_r * np.sqrt(variance_ratio)],
                    [baseline_target_r * np.sqrt(variance_ratio), 1.0],
                ]
            )
            sigma = right_inverse @ target_cov @ right_inverse.T
            sigma = (sigma + sigma.T) / 2
            sigma /= max(float(np.max(np.diag(sigma))), 1e-15)
            eigenvalues, eigenvectors = np.linalg.eigh(sigma)
            root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
            for epsilon in (.10, .20, .30, .40, .50):
                u = normals @ root.T * epsilon
                z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
                vb, okb = _exact_trial_features(tb, z)
                vd, okd = _exact_trial_features(td, z)
                use = okb & okd & np.isfinite(vb).all(axis=1) & np.isfinite(vd).all(axis=1)
                if use.sum() < 900:
                    continue
                rb = _corr(vb[use])
                rd = _corr(vd[use])
                score = abs(rd) - 2.5 * abs(rb)
                rows.append(
                    {
                        "score": score,
                        "nodes": list(nodes),
                        "input_area": item.input_area,
                        "output_area": item.output_area,
                        "variance_ratio": variance_ratio,
                        "baseline_target_r": baseline_target_r,
                        "sigma": sigma.tolist(),
                        "epsilon": epsilon,
                        "exact_baseline_r": rb,
                        "exact_deleted_r": rd,
                        "exact_delta_r": rd - rb,
                        "n_valid": int(use.sum()),
                    }
                )
        print(f"Exact candidate refinement {candidate_pos + 1}/{len(shortlist)}", flush=True)
    if candidates is None:
        candidates = pd.DataFrame(rows)
        candidates.to_csv(out / "exp3_exact_candidate_refinement.csv", index=False)
    eligible = candidates[
        (candidates.exact_baseline_r.abs() <= .10)
        & (candidates.exact_deleted_r.abs() >= .20)
        & (candidates.epsilon >= .40)
        & (candidates.output_area != "V1")
    ]
    if eligible.empty:
        eligible = candidates[
            (candidates.exact_baseline_r.abs() <= .12)
            & (candidates.exact_deleted_r.abs() >= .20)
            & (candidates.output_area != "V1")
        ]
    selected = eligible.sort_values("score", ascending=False).iloc[0].to_dict()
    if isinstance(selected["nodes"], str):
        selected["nodes"] = list(ast.literal_eval(selected["nodes"]))
    else:
        selected["nodes"] = list(selected["nodes"])
    if isinstance(selected["sigma"], str):
        selected["sigma"] = ast.literal_eval(selected["sigma"])
    path.write_text(json.dumps(selected, indent=2))
    return selected


def _exp3_design(selected: dict, n_trials=EXP3_TRIALS):
    rng = np.random.default_rng(SEED + 300)
    sigma = np.asarray(selected["sigma"], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh((sigma + sigma.T) / 2)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
    u = rng.normal(size=(n_trials, 4)) @ root.T * float(selected["epsilon"])
    z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
    return z, u, sigma


def _sigma_for_target(model, input_area: str, output_area: str, variance_ratio: float, target_r: float):
    frequencies = frequency_grid()
    transfer = _continuous_transfer(
        model.J, model.areas.index(input_area), model.areas.index(output_area)
    )
    gradient = _pair_gradient(transfer, frequencies)
    right_inverse = np.linalg.pinv(gradient)
    target_cov = np.asarray(
        [
            [variance_ratio, target_r * np.sqrt(variance_ratio)],
            [target_r * np.sqrt(variance_ratio), 1.0],
        ]
    )
    sigma = right_inverse @ target_cov @ right_inverse.T
    sigma = (sigma + sigma.T) / 2
    sigma /= max(float(np.max(np.diag(sigma))), 1e-15)
    return sigma


def _pilot_refine_regime(cfg, model, initial: dict, out: Path) -> dict:
    selected_path = out / "exp3_selected_candidate_pilot_v4.json"
    if selected_path.exists():
        return json.loads(selected_path.read_text())
    result_path = out / "exp3_pilot_regime_screen_v4.csv"
    deleted_j, _ = _node_deleted_j(model, initial["nodes"])
    input_idx = model.areas.index(initial["input_area"])
    output_idx = model.areas.index(initial["output_area"])
    n_trials = 128
    n_steps = int(round(EXP3_ANALYSIS_S * 1000 / DT_MS))
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000 / DT_MS))
    transfers = [
        _discrete_transfer(model.J, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
        _discrete_transfer(deleted_j, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
    ]
    rng = np.random.default_rng(SEED + 510)
    normals = rng.normal(size=(n_trials, 4))
    rows = []
    condition = 0
    for target_r in (.25, .40, .55, .70, .85, .95):
        sigma = _sigma_for_target(
            model, initial["input_area"], initial["output_area"],
            float(initial["variance_ratio"]), target_r,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
        for epsilon in (.30, .40, .50):
            u = normals @ root.T * epsilon
            z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
            phase_rs = np.full((2, 2), np.nan)
            phase_ns = np.zeros((2, 2), dtype=int)
            for phase in range(2):
                features = np.empty((2, n_trials, 2), dtype=float)
                valid = np.empty((2, n_trials), dtype=bool)
                for start in range(0, n_trials, EXP3_CHUNK):
                    stop = min(start + EXP3_CHUNK, n_trials)
                    inputs = _matched_bandlimited_inputs(
                        z[start:stop], DT_MS, EXP3_ANALYSIS_S, burn_s,
                        float(cfg["exp3_nonlinear"]["input_sigma"]),
                        SEED + 520000 + condition * 10000 + phase * 100000,
                        trial_offset=start,
                    )
                    input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
                    outputs = np.empty((stop - start, 2, n_steps), dtype=np.float32)
                    for network_pos, transfer in enumerate(transfers):
                        outputs[:, network_pos] = np.fft.irfft(
                            input_fft * transfer[None, :], n=n_steps, axis=-1
                        ).astype(np.float32)
                    values, is_valid = _feature_arrays_vectorized(outputs, DT_MS, _spectral_spec())["long"]
                    features[:, start:stop] = values
                    valid[:, start:stop] = is_valid
                for network_pos in range(2):
                    phase_ns[phase, network_pos] = int(valid[network_pos].sum())
                    phase_rs[phase, network_pos] = _corr(features[network_pos, valid[network_pos]])
            baseline_r = float(np.nanmean(phase_rs[:, 0]))
            deleted_r = float(np.nanmean(phase_rs[:, 1]))
            rows.append(
                {
                    "condition": condition,
                    "target_r": target_r,
                    "epsilon": epsilon,
                    "baseline_r": baseline_r,
                    "deleted_r": deleted_r,
                    "delta_r": deleted_r - baseline_r,
                    "baseline_phase_sd": float(np.nanstd(phase_rs[:, 0], ddof=1)),
                    "deleted_phase_sd": float(np.nanstd(phase_rs[:, 1], ddof=1)),
                    "minimum_valid": int(phase_ns.min()),
                    "sigma": json.dumps(sigma.tolist()),
                    "score": abs(deleted_r) - 2.5 * abs(baseline_r),
                }
            )
            print(
                f"Experiment 3 pilot {condition + 1}/18: target={target_r:.2f}, "
                f"epsilon={epsilon:.2f}, baseline={baseline_r:.3f}, deleted={deleted_r:.3f}",
                flush=True,
            )
            condition += 1
    result = pd.DataFrame(rows)
    result.to_csv(result_path, index=False)
    eligible = result[(result.baseline_r.abs() <= .15) & (result.deleted_r.abs() >= .25)]
    if eligible.empty:
        eligible = result[result.baseline_r.abs() <= .20]
    best = eligible.sort_values("score", ascending=False).iloc[0]
    selected = dict(initial)
    selected.update(
        {
            "pilot_target_r": float(best.target_r),
            "epsilon": float(best.epsilon),
            "sigma": json.loads(best.sigma),
            "pilot_baseline_r": float(best.baseline_r),
            "pilot_deleted_r": float(best.deleted_r),
            "pilot_delta_r": float(best.delta_r),
            "pilot_n_trials": n_trials,
            "pilot_n_realizations": 2,
        }
    )
    selected_path.write_text(json.dumps(selected, indent=2))
    return selected


def _spectral_spec():
    return {
        "long": {
            "analysis_s": EXP3_ANALYSIS_S,
            "spectral": {
                "mode": "hann_welch",
                "segment_s": WELCH_SEGMENT_S,
                "smoothing_sigma_hz": SMOOTHING_SIGMA_HZ,
            },
        }
    }


def _pooled_cov_with_missing(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Pool within-trial realization covariance without imputing invalid peaks."""
    total = np.zeros((values.shape[-1], values.shape[-1]), dtype=float)
    degrees = 0
    for trial in range(values.shape[1]):
        use = valid[:, trial]
        if use.sum() < 2:
            continue
        block = values[use, trial].astype(float)
        centered = block - block.mean(axis=0, keepdims=True)
        total += centered.T @ centered
        degrees += len(block) - 1
    if degrees <= 0:
        return np.full((values.shape[-1], values.shape[-1]), np.nan)
    return total / degrees


def _full_error_layer(theory: np.ndarray, observed: np.ndarray, valid: np.ndarray, use: np.ndarray):
    """Estimate finite-window error, including signal-error cross covariance."""
    signal_cov = np.cov(theory[use], rowvar=False, ddof=1)
    x_rows = []
    e_rows = []
    for phase in range(observed.shape[0]):
        phase_use = use & valid[phase]
        x = theory[phase_use]
        x_rows.append(x)
        e_rows.append(observed[phase, phase_use] - x)
    x = np.concatenate(x_rows, axis=0)
    e = np.concatenate(e_rows, axis=0)
    xc = x - x.mean(axis=0, keepdims=True)
    ec = e - e.mean(axis=0, keepdims=True)
    error_cov = np.cov(e, rowvar=False, ddof=1)
    cross = xc.T @ ec / (len(x) - 1)
    total_cov = signal_cov + error_cov + cross + cross.T
    return signal_cov, error_cov, cross, total_cov


def _simulate_exp3_phase(cfg, model, deleted_j, selected, z, phase, out, force=False):
    cache = out / "exp3_phase_cache" / f"pilot_v4_phase{phase:02d}.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        return d["features"], d["valid"]
    n_steps = int(round(EXP3_ANALYSIS_S * 1000 / DT_MS))
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000 / DT_MS))
    input_idx = model.areas.index(selected["input_area"])
    output_idx = model.areas.index(selected["output_area"])
    transfers = [
        _discrete_transfer(model.J, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
        _discrete_transfer(deleted_j, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
    ]
    features = np.empty((2, len(z), 2), dtype=np.float32)
    valid = np.empty((2, len(z)), dtype=bool)
    for start in range(0, len(z), EXP3_CHUNK):
        stop = min(start + EXP3_CHUNK, len(z))
        inputs = _matched_bandlimited_inputs(
            z[start:stop],
            DT_MS,
            EXP3_ANALYSIS_S,
            burn_s,
            float(cfg["exp3_nonlinear"]["input_sigma"]),
            SEED + 400000 + phase * 100000,
            trial_offset=start,
        )
        input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
        outputs = np.empty((stop - start, 2, n_steps), dtype=np.float32)
        for network_pos, transfer in enumerate(transfers):
            outputs[:, network_pos] = np.fft.irfft(input_fft * transfer[None, :], n=n_steps, axis=-1).astype(np.float32)
        values, is_valid = _feature_arrays_vectorized(outputs, DT_MS, _spectral_spec())["long"]
        features[:, start:stop] = values
        valid[:, start:stop] = is_valid
        if start % 128 == 0:
            print(f"Experiment 3 phase {phase + 1}/{TOTAL_PHASES}: trials {start + 1}-{stop}/{len(z)}", flush=True)
    np.savez_compressed(cache, features=features, valid=valid)
    return features, valid


def _selected_psd_averaged_v5(model, initial: dict, out: Path) -> dict:
    path = out / "exp3_selected_candidate_psd_averaged_v5.json"
    if path.exists():
        return json.loads(path.read_text())
    selected = dict(initial)
    selected["pilot_target_r"] = 0.0
    selected["epsilon"] = 0.4
    selected["sigma"] = _sigma_for_target(
        model,
        selected["input_area"],
        selected["output_area"],
        float(selected["variance_ratio"]),
        0.0,
    ).tolist()
    selected["observation"] = "PSD averaged before feature extraction: 3 calibration and 5 held-out segments"
    path.write_text(json.dumps(selected, indent=2))
    return selected


def _simulate_exp3_psd_phase(cfg, model, deleted_j, selected, z, phase, out, force=False):
    cache = out / "exp3_phase_cache" / f"psd_averaged_v5_phase{phase:02d}.npz"
    if cache.exists() and not force:
        d = np.load(cache)
        return d["frequencies"], d["psd"]
    n_steps = int(round(EXP3_ANALYSIS_S * 1000 / DT_MS))
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000 / DT_MS))
    input_idx = model.areas.index(selected["input_area"])
    output_idx = model.areas.index(selected["output_area"])
    transfers = [
        _discrete_transfer(model.J, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
        _discrete_transfer(deleted_j, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
    ]
    fs = 1000.0 / DT_MS
    nperseg = int(round(WELCH_SEGMENT_S * fs))
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    keep = (frequencies >= .5) & (frequencies <= 1200.0)
    kept_frequencies = frequencies[keep]
    psds = np.empty((2, len(z), int(keep.sum())), dtype=np.float32)
    for start in range(0, len(z), EXP3_CHUNK):
        stop = min(start + EXP3_CHUNK, len(z))
        inputs = _matched_bandlimited_inputs(
            z[start:stop], DT_MS, EXP3_ANALYSIS_S, burn_s,
            float(cfg["exp3_nonlinear"]["input_sigma"]),
            SEED + 610000 + phase * 100000,
            trial_offset=start,
        )
        input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
        outputs = np.empty((stop - start, 2, n_steps), dtype=np.float32)
        for network_pos, transfer in enumerate(transfers):
            outputs[:, network_pos] = np.fft.irfft(
                input_fft * transfer[None, :], n=n_steps, axis=-1
            ).astype(np.float32)
        _, p = signal.welch(
            outputs,
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
            axis=-1,
        )
        psds[:, start:stop] = p[:, :, keep].transpose(1, 0, 2)
        if start % 128 == 0:
            print(
                f"Experiment 3 PSD segment {phase + 1}/{TOTAL_PHASES}: "
                f"trials {start + 1}-{stop}/{len(z)}",
                flush=True,
            )
    np.savez_compressed(cache, frequencies=kept_frequencies, psd=psds)
    return kept_frequencies, psds


def _features_from_averaged_psd(frequencies: np.ndarray, psd: np.ndarray):
    sigma_bins = SMOOTHING_SIGMA_HZ / (frequencies[1] - frequencies[0])
    smooth = np.exp(
        ndimage.gaussian_filter1d(
            np.log(np.maximum(psd, 1e-300)), sigma_bins, axis=-1
        )
    )
    features = np.empty((psd.shape[0], psd.shape[1], 2), dtype=float)
    valid = np.empty((psd.shape[0], psd.shape[1]), dtype=bool)
    for network in range(psd.shape[0]):
        for trial in range(psd.shape[1]):
            feature = feature_from_spectrum(
                frequencies,
                smooth[network, trial],
                peak_band=(5.0, 20.0),
                slope_band=(100.0, 1000.0),
            )
            features[network, trial] = feature["peak_omega"], feature["exponent"]
            valid[network, trial] = bool(feature["peak_valid"])
    return features, valid


def _observed_feature(transfer: np.ndarray, frequencies: np.ndarray, z: np.ndarray):
    sigma_bins = SMOOTHING_SIGMA_HZ / (frequencies[1] - frequencies[0])
    spectrum = transfer * input_psd(frequencies, z)
    smooth = np.exp(
        ndimage.gaussian_filter1d(np.log(np.maximum(spectrum, 1e-300)), sigma_bins)
    )
    feature = feature_from_spectrum(
        frequencies, smooth, peak_band=(5.0, 20.0), slope_band=(100.0, 1000.0)
    )
    return np.asarray([feature["peak_omega"], feature["exponent"]]), bool(feature["peak_valid"])


def _observed_gradients(transfers: np.ndarray, frequencies: np.ndarray, h: float = 2e-3):
    gradients = np.empty((9, 9, 2, 4), dtype=float)
    for q in range(4):
        zp = Z0 + h * ZSCALE * np.eye(4)[q]
        zm = Z0 - h * ZSCALE * np.eye(4)[q]
        for i in range(9):
            for o in range(9):
                fp, _ = _observed_feature(transfers[i, o], frequencies, zp)
                fm, _ = _observed_feature(transfers[i, o], frequencies, zm)
                gradients[i, o, :, q] = (fp - fm) / (2 * h)
    return gradients


def _sigma_from_gradient(
    gradient: np.ndarray, variance_ratio: float, target_r: float, null_weight: float = 0.0
):
    right_inverse = np.linalg.pinv(gradient)
    target_cov = np.asarray(
        [
            [variance_ratio, target_r * np.sqrt(variance_ratio)],
            [target_r * np.sqrt(variance_ratio), 1.0],
        ]
    )
    sigma = right_inverse @ target_cov @ right_inverse.T
    if null_weight > 0:
        projector = np.eye(gradient.shape[1]) - right_inverse @ gradient
        null_cov = projector @ projector.T
        scale = np.trace(sigma) / max(float(np.trace(null_cov)), 1e-15)
        sigma += null_weight * scale * null_cov
    sigma = (sigma + sigma.T) / 2
    sigma /= max(float(np.max(np.diag(sigma))), 1e-15)
    return sigma


def _observed_node_screen(model, out: Path):
    path = out / "exp3_observation_matched_nullspace_screen_v7.csv.gz"
    if path.exists():
        return pd.read_csv(path)
    source = pd.read_csv(out / "exp3_node_screen_fullcov_v2.csv.gz")
    node_sets = source.nodes.drop_duplicates().tolist()
    n_steps = int(round(EXP3_ANALYSIS_S * 1000 / DT_MS))
    full_f = np.fft.rfftfreq(n_steps, d=DT_MS / 1000.0)
    coarse_step = 16
    ids = np.arange(
        int(np.ceil(.5 / (full_f[1] - full_f[0]))),
        int(np.floor(1200.0 / (full_f[1] - full_f[0]))) + 1,
        coarse_step,
    )
    frequencies = full_f[ids]
    indices = [model.areas.index(area) for area in AREAS9]
    base_transfer = _all_transfer_power(model.J, indices, frequencies)
    base_grad = _observed_gradients(base_transfer, frequencies)
    rows = []
    for pos, encoded in enumerate(node_sets):
        nodes = list(json.loads(encoded))
        deleted_j, _ = _node_deleted_j(model, nodes)
        deleted_transfer = _all_transfer_power(deleted_j, indices, frequencies)
        deleted_grad = _observed_gradients(deleted_transfer, frequencies)
        for i, input_area in enumerate(AREAS9):
            if input_area in nodes:
                continue
            for o, output_area in enumerate(AREAS9):
                if output_area in nodes or np.linalg.matrix_rank(base_grad[i, o]) < 2:
                    continue
                for variance_ratio in (.02, .05, .10, .20, .50, 1.0, 2.0, 5.0, 20.0):
                    for null_weight in (0.0, .1, 1.0, 10.0, 100.0):
                        sigma = _sigma_from_gradient(
                            base_grad[i, o], variance_ratio, 0.0, null_weight
                        )
                        rb = _cov_to_corr(base_grad[i, o] @ sigma @ base_grad[i, o].T)
                        rd = _cov_to_corr(deleted_grad[i, o] @ sigma @ deleted_grad[i, o].T)
                        rows.append(
                            {
                                "nodes": encoded,
                                "n_nodes": len(nodes),
                                "input_area": input_area,
                                "output_area": output_area,
                                "variance_ratio": variance_ratio,
                                "null_weight": null_weight,
                                "linear_baseline_r": rb,
                                "linear_deleted_r": rd,
                                "linear_delta_r": rd - rb,
                            }
                        )
        if (pos + 1) % 20 == 0:
            print(f"Observation-matched node screen {pos + 1}/{len(node_sets)}", flush=True)
    result = pd.DataFrame(rows).sort_values(
        "linear_deleted_r", key=lambda x: abs(x), ascending=False
    )
    result.to_csv(path, index=False, compression="gzip")
    np.savez_compressed(out / "exp3_observation_grid_v7.npz", frequencies=frequencies, frequency_ids=ids)
    return result


def _observed_values(transfer, frequencies, z):
    values = np.empty((len(z), 2), dtype=float)
    valid = np.empty(len(z), dtype=bool)
    for trial, design in enumerate(z):
        values[trial], valid[trial] = _observed_feature(transfer, frequencies, design)
    return values, valid


def _select_observation_matched_candidate(model, screen: pd.DataFrame, out: Path):
    path = out / "exp3_selected_observation_matched_nullspace_v9.json"
    if path.exists():
        return json.loads(path.read_text())
    grid = np.load(out / "exp3_observation_grid_v7.npz")
    frequencies = grid["frequencies"]
    indices = [model.areas.index(area) for area in AREAS9]
    base_transfer = _all_transfer_power(model.J, indices, frequencies)
    base_grad = _observed_gradients(base_transfer, frequencies)
    shortlist = (
        screen[screen.null_weight > 0]
        .drop_duplicates(["nodes", "input_area", "output_area"])
        .head(30)
    )
    normals = np.random.default_rng(SEED + 300).normal(size=(EXP3_TRIALS, 4))
    rows = []
    for pos, (_, item) in enumerate(shortlist.iterrows()):
        nodes = list(json.loads(item.nodes))
        deleted_j, _ = _node_deleted_j(model, nodes)
        i = AREAS9.index(item.input_area)
        o = AREAS9.index(item.output_area)
        deleted_transfer = _all_transfer_power(deleted_j, indices, frequencies)[i, o]
        for target_r in (-.4, 0.0, .4):
            sigma = _sigma_from_gradient(
                base_grad[i, o], float(item.variance_ratio), target_r,
                float(item.null_weight),
            )
            eigenvalues, eigenvectors = np.linalg.eigh(sigma)
            root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0)))
            for epsilon in (.50, .75, 1.00, 1.50):
                u = normals @ root.T * epsilon
                z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
                baseline, valid_b = _observed_values(base_transfer[i, o], frequencies, z)
                deleted, valid_d = _observed_values(deleted_transfer, frequencies, z)
                use = valid_b & valid_d
                if use.sum() < 400:
                    continue
                rb = _corr(baseline[use])
                rd = _corr(deleted[use])
                peak_sd_hz = float(np.std(deleted[use, 0], ddof=1) / (2 * np.pi))
                rows.append(
                    {
                        "nodes": item.nodes,
                        "input_area": item.input_area,
                        "output_area": item.output_area,
                        "variance_ratio": float(item.variance_ratio),
                        "null_weight": float(item.null_weight),
                        "target_r": target_r,
                        "epsilon": epsilon,
                        "sigma": json.dumps(sigma.tolist()),
                        "baseline_r": rb,
                        "deleted_r": rd,
                        "delta_r": rd - rb,
                        "deleted_peak_sd_hz": peak_sd_hz,
                        "n_valid": int(use.sum()),
                        "score": abs(rd) - 2.5 * abs(rb) + .03 * min(peak_sd_hz, 5.0),
                    }
                )
        print(f"Observation-matched exact refinement {pos + 1}/{len(shortlist)}", flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out / "exp3_observation_matched_nullspace_refinement_v9.csv", index=False)
    eligible = result[
        (result.baseline_r.abs() <= .10)
        & (result.deleted_r.abs() >= .40)
        & (result.deleted_peak_sd_hz >= .5)
    ]
    if eligible.empty:
        eligible = result[(result.baseline_r.abs() <= .15) & (result.deleted_r.abs() >= .25)]
    selected_row = eligible.sort_values("score", ascending=False).iloc[0]
    selected = {
        "nodes": list(json.loads(selected_row.nodes)),
        "input_area": selected_row.input_area,
        "output_area": selected_row.output_area,
        "variance_ratio": float(selected_row.variance_ratio),
        "null_weight": float(selected_row.null_weight),
        "target_r": float(selected_row.target_r),
        "epsilon": float(selected_row.epsilon),
        "sigma": json.loads(selected_row.sigma),
        "observation_matched_baseline_r": float(selected_row.baseline_r),
        "observation_matched_deleted_r": float(selected_row.deleted_r),
        "observation_matched_delta_r": float(selected_row.delta_r),
        "deleted_peak_sd_hz": float(selected_row.deleted_peak_sd_hz),
        "n_screen_valid": int(selected_row.n_valid),
        "coarse_bin_count": 16,
    }
    path.write_text(json.dumps(selected, indent=2))
    return selected


def _stochastic_psd_phase(model, deleted_j, selected, frequencies, frequency_ids, phase, out, force=False):
    path = out / "exp3_phase_cache" / f"observation_matched_nullspace_v9_phase{phase:02d}.npz"
    if path.exists() and not force:
        return np.load(path)["psd"]
    z, _, _ = _exp3_design(selected)
    n_steps = int(round(EXP3_ANALYSIS_S * 1000 / DT_MS))
    input_idx = model.areas.index(selected["input_area"])
    output_idx = model.areas.index(selected["output_area"])
    transfers = [
        np.abs(_discrete_transfer(j, input_idx, output_idx, n_steps, float(model.params["tau_exc"]))) ** 2
        for j in (model.J, deleted_j)
    ]
    rng = np.random.default_rng(SEED + 710000 + phase * 100000)
    psd = np.empty((2, EXP3_TRIALS, len(frequencies)), dtype=np.float32)
    shape = int(selected["coarse_bin_count"])
    for trial in range(EXP3_TRIALS):
        realization = rng.gamma(shape, 1.0 / shape, size=len(frequencies))
        input_power = input_psd(frequencies, z[trial]) * realization
        for network_pos in range(2):
            psd[network_pos, trial] = input_power * transfers[network_pos][frequency_ids]
    np.savez_compressed(path, psd=psd)
    print(f"Observation-matched stochastic PSD phase {phase + 1}/{TOTAL_PHASES}", flush=True)
    return psd


def run_exp3_observation_matched(cfg: dict, model, out: Path, force=False):
    result_path = out / "exp3_node_deletion_observation_matched_nullspace_v9.csv"
    selected_path = out / "exp3_selected_observation_matched_nullspace_v9.json"
    if result_path.exists() and selected_path.exists() and not force:
        return pd.read_csv(result_path), json.loads(selected_path.read_text())
    screen = _observed_node_screen(model, out)
    selected = _select_observation_matched_candidate(model, screen, out)
    deleted_j, delta_w = _node_deleted_j(model, selected["nodes"])
    grid = np.load(out / "exp3_observation_grid_v7.npz")
    frequencies = grid["frequencies"]
    frequency_ids = grid["frequency_ids"]
    indices = [model.areas.index(area) for area in AREAS9]
    transfers = {
        "baseline": _all_transfer_power(model.J, indices, frequencies)[AREAS9.index(selected["input_area"]), AREAS9.index(selected["output_area"])],
        "node_deleted": _all_transfer_power(deleted_j, indices, frequencies)[AREAS9.index(selected["input_area"]), AREAS9.index(selected["output_area"])],
    }
    z, u, sigma = _exp3_design(selected)
    theory = {}
    theory_valid = np.ones(EXP3_TRIALS, dtype=bool)
    for name in ("baseline", "node_deleted"):
        theory[name], valid = _observed_values(transfers[name], frequencies, z)
        theory_valid &= valid
    phase_psds = np.asarray(
        [
            _stochastic_psd_phase(
                model, deleted_j, selected, frequencies, frequency_ids, phase, out, force
            )
            for phase in range(TOTAL_PHASES)
        ]
    )
    calibration, calibration_valid = _features_from_averaged_psd(
        frequencies, phase_psds[:CALIBRATION_PHASES].mean(axis=0)
    )
    heldout, heldout_valid = _features_from_averaged_psd(
        frequencies, phase_psds[CALIBRATION_PHASES:].mean(axis=0)
    )
    use = theory_valid & calibration_valid.all(axis=0) & heldout_valid.all(axis=0)
    if use.sum() < 350:
        raise RuntimeError(f"Only {use.sum()} common valid observation-matched trials")
    combo_rs = {"baseline": [], "node_deleted": []}
    for combo in itertools.combinations(range(HELDOUT_PHASES), CALIBRATION_PHASES):
        features, valid = _features_from_averaged_psd(
            frequencies, phase_psds[CALIBRATION_PHASES + np.asarray(combo)].mean(axis=0)
        )
        for network_pos, name in enumerate(("baseline", "node_deleted")):
            combo_use = use & valid[network_pos]
            combo_rs[name].append(_corr(features[network_pos, combo_use]))
    rows = []
    values = {}
    for network_pos, name in enumerate(("baseline", "node_deleted")):
        signal = theory[name][use]
        error = calibration[network_pos, use] - signal
        sc = signal - signal.mean(axis=0, keepdims=True)
        ec = error - error.mean(axis=0, keepdims=True)
        signal_cov = np.cov(signal, rowvar=False, ddof=1)
        error_cov = np.cov(error, rowvar=False, ddof=1)
        cross = sc.T @ ec / (len(signal) - 1)
        predicted = _cov_to_corr(signal_cov + error_cov + cross + cross.T)
        observed = _corr(heldout[network_pos, use])
        repeats = np.asarray(combo_rs[name])
        values[name] = (predicted, observed, repeats)
        rows.append(
            {
                "network": name,
                "n_valid": int(use.sum()),
                "latent_exact_r": _corr(signal),
                "residual_only_r": _corr(error),
                "signal_error_cross_fro": float(np.linalg.norm(cross)),
                "theory_plus_realization_r": predicted,
                "simulation_r": observed,
                "simulation_r_sd": float(repeats.std(ddof=1)),
                "heldout_three_segment_r_json": json.dumps(repeats.tolist()),
            }
        )
    rows.append(
        {
            "network": "delta",
            "n_valid": int(use.sum()),
            "latent_exact_r": np.nan,
            "residual_only_r": np.nan,
            "signal_error_cross_fro": np.nan,
            "theory_plus_realization_r": values["node_deleted"][0] - values["baseline"][0],
            "simulation_r": values["node_deleted"][1] - values["baseline"][1],
            "simulation_r_sd": float(np.std(values["node_deleted"][2] - values["baseline"][2], ddof=1)),
            "heldout_three_segment_r_json": json.dumps((values["node_deleted"][2] - values["baseline"][2]).tolist()),
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(result_path, index=False)
    np.savez_compressed(
        out / "exp3_observation_matched_nullspace_v9.npz",
        frequencies=frequencies,
        phase_psds=phase_psds,
        calibration_features=calibration,
        heldout_features=heldout,
        calibration_valid=calibration_valid,
        heldout_valid=heldout_valid,
        use=use,
        z=z,
        u=u,
        sigma=sigma,
        baseline_theory=theory["baseline"],
        deleted_theory=theory["node_deleted"],
    )
    _exp3_explicit_qc(cfg, model, deleted_j, delta_w, selected, z, out)
    return result, selected


def run_exp3_psd_averaged(cfg: dict, model, out: Path, force=False):
    results_path = out / "exp3_node_deletion_psd_averaged_v5.csv"
    selected_path = out / "exp3_selected_candidate_psd_averaged_v5.json"
    if results_path.exists() and selected_path.exists() and not force:
        return pd.read_csv(results_path), json.loads(selected_path.read_text())
    screen = _screen_node_deletions(model, out, force=False)
    initial = _exact_candidate_refinement(model, screen, out)
    selected = _selected_psd_averaged_v5(model, initial, out)
    deleted_j, delta_w = _node_deleted_j(model, selected["nodes"])
    z, u, sigma = _exp3_design(selected)
    input_idx = model.areas.index(selected["input_area"])
    output_idx = model.areas.index(selected["output_area"])
    theory = {}
    theory_valid = np.ones(len(z), dtype=bool)
    for name, j in (("baseline", model.J), ("node_deleted", deleted_j)):
        values, valid = _exact_trial_features(
            _continuous_transfer(j, input_idx, output_idx), z
        )
        theory[name] = values
        theory_valid &= valid & np.isfinite(values).all(axis=1)
    phase_psds = []
    frequencies = None
    for phase in range(TOTAL_PHASES):
        frequencies, phase_psd = _simulate_exp3_psd_phase(
            cfg, model, deleted_j, selected, z, phase, out, force=force
        )
        phase_psds.append(phase_psd)
    phase_psds = np.asarray(phase_psds)
    calibration_features, calibration_valid = _features_from_averaged_psd(
        frequencies, phase_psds[:CALIBRATION_PHASES].mean(axis=0)
    )
    heldout_features, heldout_valid = _features_from_averaged_psd(
        frequencies, phase_psds[CALIBRATION_PHASES:].mean(axis=0)
    )
    use = theory_valid & calibration_valid.all(axis=0) & heldout_valid.all(axis=0)
    if use.sum() < 450:
        raise RuntimeError(f"Only {use.sum()} common valid PSD-averaged trials")
    rows = []
    network_results = {}
    heldout_combo_rs = {"baseline": [], "node_deleted": []}
    for combo in itertools.combinations(range(HELDOUT_PHASES), CALIBRATION_PHASES):
        combo_features, combo_valid = _features_from_averaged_psd(
            frequencies,
            phase_psds[CALIBRATION_PHASES + np.asarray(combo)].mean(axis=0),
        )
        for network_pos, name in enumerate(("baseline", "node_deleted")):
            combo_use = use & combo_valid[network_pos]
            heldout_combo_rs[name].append(_corr(combo_features[network_pos, combo_use]))
    for network_pos, name in enumerate(("baseline", "node_deleted")):
        signal = theory[name][use]
        error = calibration_features[network_pos, use] - signal
        signal_cov = np.cov(signal, rowvar=False, ddof=1)
        error_cov = np.cov(error, rowvar=False, ddof=1)
        sc = signal - signal.mean(axis=0, keepdims=True)
        ec = error - error.mean(axis=0, keepdims=True)
        cross = sc.T @ ec / (len(signal) - 1)
        predicted = _cov_to_corr(signal_cov + error_cov + cross + cross.T)
        observed = _corr(heldout_features[network_pos, use])
        combo_rs = np.asarray(heldout_combo_rs[name])
        network_results[name] = {
            "theory": predicted,
            "simulation": observed,
            "sd": float(combo_rs.std(ddof=1)),
            "combo_rs": combo_rs,
        }
        rows.append(
            {
                "network": name,
                "n_valid": int(use.sum()),
                "latent_exact_r": _corr(signal),
                "residual_only_r": _corr(error),
                "signal_error_cross_fro": float(np.linalg.norm(cross)),
                "theory_plus_realization_r": predicted,
                "simulation_r": observed,
                "simulation_r_sd": float(combo_rs.std(ddof=1)),
                "heldout_three_segment_r_json": json.dumps(combo_rs.tolist()),
            }
        )
    rows.append(
        {
            "network": "delta",
            "n_valid": int(use.sum()),
            "latent_exact_r": theory["node_deleted"][use, 0].shape[0] * np.nan,
            "residual_only_r": np.nan,
            "signal_error_cross_fro": np.nan,
            "theory_plus_realization_r": network_results["node_deleted"]["theory"] - network_results["baseline"]["theory"],
            "simulation_r": network_results["node_deleted"]["simulation"] - network_results["baseline"]["simulation"],
            "simulation_r_sd": float(np.std(network_results["node_deleted"]["combo_rs"] - network_results["baseline"]["combo_rs"], ddof=1)),
            "heldout_three_segment_r_json": json.dumps((network_results["node_deleted"]["combo_rs"] - network_results["baseline"]["combo_rs"]).tolist()),
        }
    )
    results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False)
    np.savez_compressed(
        out / "exp3_node_deletion_psd_averaged_v5.npz",
        frequencies=frequencies,
        phase_psds=phase_psds,
        calibration_features=calibration_features,
        heldout_features=heldout_features,
        calibration_valid=calibration_valid,
        heldout_valid=heldout_valid,
        use=use,
        z=z,
        u=u,
        sigma=sigma,
        baseline_theory=theory["baseline"],
        deleted_theory=theory["node_deleted"],
    )
    _exp3_explicit_qc(cfg, model, deleted_j, delta_w, selected, z, out)
    return results, selected


def _exp3_explicit_qc(cfg, model, deleted_j, delta_w, selected, z, out):
    path = out / "exp3_explicit_recurrence_qc_psd_averaged_v5.csv"
    if path.exists():
        return pd.read_csv(path)
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000 / DT_MS))
    analysis_s = 4.096
    n_steps = int(round(analysis_s * 1000 / DT_MS))
    input_idx = model.areas.index(selected["input_area"])
    output_idx = model.areas.index(selected["output_area"])
    inputs = _matched_bandlimited_inputs(z[:4], DT_MS, analysis_s, burn_s, float(cfg["exp3_nonlinear"]["input_sigma"]), SEED + 900)
    rows = []
    deleted_params = copy.copy(model.params)
    deleted_params["fln_mat"] = model.fln + delta_w
    for name, params, j in (("baseline", model.params, model.J), ("node_deleted", deleted_params, deleted_j)):
        explicit, rect, *_ = _run_multioutput_batch(params, inputs, DT_MS, burn_steps, np.asarray([output_idx]), input_idx=input_idx)
        transfer = _discrete_transfer(j, input_idx, output_idx, n_steps, float(model.params["tau_exc"]))
        accelerated = np.fft.irfft(np.fft.rfft(inputs[:, burn_steps:], axis=-1) * transfer[None, :], n=n_steps, axis=-1)
        error = explicit[:, 0] - accelerated
        rows.append(
            {
                "network": name,
                "n_trials": 4,
                "rectification_count": int(rect.sum()),
                "timeseries_r": float(np.corrcoef(explicit[:, 0].ravel(), accelerated.ravel())[0, 1]),
                "relative_rmse": float(np.sqrt(np.mean(error**2)) / max(np.std(explicit[:, 0]), 1e-15)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(path, index=False)
    return result


def run_exp3(cfg: dict, model, out: Path, force=False):
    results_path = out / "exp3_node_deletion_long_results_pilot_v4.csv"
    selected_path = out / "exp3_selected_candidate_pilot_v4.json"
    if results_path.exists() and selected_path.exists() and not force:
        return pd.read_csv(results_path), json.loads(selected_path.read_text())
    screen = _screen_node_deletions(model, out, force=force)
    initial = _exact_candidate_refinement(model, screen, out)
    selected = _pilot_refine_regime(cfg, model, initial, out)
    deleted_j, delta_w = _node_deleted_j(model, selected["nodes"])
    z, u, sigma = _exp3_design(selected)
    input_idx = model.areas.index(selected["input_area"])
    output_idx = model.areas.index(selected["output_area"])
    theory = {}
    theory_valid = np.ones(len(z), dtype=bool)
    for name, j in (("baseline", model.J), ("node_deleted", deleted_j)):
        values, valid = _exact_trial_features(_continuous_transfer(j, input_idx, output_idx), z)
        theory[name] = values
        theory_valid &= valid & np.isfinite(values).all(axis=1)
    features = np.empty((TOTAL_PHASES, 2, len(z), 2), dtype=np.float32)
    valid = np.empty((TOTAL_PHASES, 2, len(z)), dtype=bool)
    for phase in range(TOTAL_PHASES):
        features[phase], valid[phase] = _simulate_exp3_phase(cfg, model, deleted_j, selected, z, phase, out, force=force)
    enough_calibration = (valid[:CALIBRATION_PHASES].sum(axis=0) >= 2).all(axis=0)
    enough_heldout = (valid[CALIBRATION_PHASES:].sum(axis=0) >= 3).all(axis=0)
    use = theory_valid & enough_calibration & enough_heldout
    if use.sum() < 350:
        raise RuntimeError(f"Only {use.sum()} common valid trials in Experiment 3")
    rows = []
    network_results = {}
    for network_pos, name in enumerate(("baseline", "node_deleted")):
        observed = features[:, network_pos].astype(float)
        signal_cov, residual_cov, cross_cov, total_cov = _full_error_layer(
            theory[name],
            observed[:CALIBRATION_PHASES],
            valid[:CALIBRATION_PHASES, network_pos],
            use,
        )
        predicted = _cov_to_corr(total_cov)
        phase_counts = []
        phase_r = []
        for phase in range(CALIBRATION_PHASES, TOTAL_PHASES):
            phase_use = use & valid[phase, network_pos]
            phase_counts.append(int(phase_use.sum()))
            phase_r.append(_corr(observed[phase, phase_use]))
        phase_r = np.asarray(phase_r)
        network_results[name] = {"theory": predicted, "simulation": float(phase_r.mean()), "sd": float(phase_r.std(ddof=1)), "phase_r": phase_r}
        rows.append(
            {
                "network": name,
                "n_valid": int(use.sum()),
                "latent_exact_r": _cov_to_corr(signal_cov),
                "residual_only_r": _cov_to_corr(residual_cov),
                "signal_error_cross_fro": float(np.linalg.norm(cross_cov)),
                "theory_plus_realization_r": predicted,
                "simulation_r": float(phase_r.mean()),
                "simulation_r_sd": float(phase_r.std(ddof=1)),
                "heldout_phase_r_json": json.dumps(phase_r.tolist()),
                "heldout_phase_n_json": json.dumps(phase_counts),
            }
        )
    rows.append(
        {
            "network": "delta",
            "n_valid": int(use.sum()),
            "latent_exact_r": np.nan,
            "residual_only_r": np.nan,
            "signal_error_cross_fro": np.nan,
            "theory_plus_realization_r": network_results["node_deleted"]["theory"] - network_results["baseline"]["theory"],
            "simulation_r": network_results["node_deleted"]["simulation"] - network_results["baseline"]["simulation"],
            "simulation_r_sd": float(np.std(network_results["node_deleted"]["phase_r"] - network_results["baseline"]["phase_r"], ddof=1)),
            "heldout_phase_r_json": json.dumps((network_results["node_deleted"]["phase_r"] - network_results["baseline"]["phase_r"]).tolist()),
            "heldout_phase_n_json": "",
        }
    )
    results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False)
    np.savez_compressed(out / "exp3_node_deletion_features_pilot_v4.npz", features=features, valid=valid, use=use, z=z, u=u, sigma=sigma, baseline_theory=theory["baseline"], deleted_theory=theory["node_deleted"])
    _exp3_explicit_qc(cfg, model, deleted_j, delta_w, selected, z, out)
    return results, selected


def make_panel_h_i(model, out: Path, results: pd.DataFrame, selected: dict) -> dict:
    deleted_j, _ = _node_deleted_j(model, selected["nodes"])
    n = len(model.areas)
    baseline = model.J[:n, :n].copy()
    deleted = deleted_j[:n, :n].copy()
    np.fill_diagonal(baseline, 0)
    np.fill_diagonal(deleted, 0)
    deleted_mask = np.zeros_like(deleted, dtype=bool)
    for name in selected["nodes"]:
        idx = model.areas.index(name)
        deleted_mask[idx, :] = model.fln[idx, :] > 0
        deleted_mask[:, idx] |= model.fln[:, idx] > 0
    vmax = float(np.nanpercentile(baseline[baseline > 0], 99))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#B8B8B8")
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), constrained_layout=True)
    im = axes[0].imshow(baseline, cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    axes[1].imshow(np.ma.array(deleted, mask=deleted_mask), cmap=cmap, vmin=0, vmax=vmax, aspect="equal")
    for ax, title in zip(axes, ("Baseline long-range J", "Node-deleted long-range J")):
        ticks = [0, 10, 20, 30, 40, 50, 54]
        ax.set_xticks(ticks, [model.areas[i] for i in ticks], rotation=45, ha="right")
        ax.set_yticks(ticks, [model.areas[i] for i in ticks])
        ax.set(xlabel="Source", ylabel="Target", title=title)
    cb = fig.colorbar(im, ax=axes, shrink=.76, pad=.02)
    cb.set_label(r"E-to-E coupling (ms$^{-1}$)")
    cb.set_ticks([0, vmax])
    _panel_label(axes[0], "H")
    _save(fig, out / "panels" / "H_exp3_baseline_deleted_J_matrices")

    ordered = results.set_index("network").loc[["baseline", "node_deleted", "delta"]]
    x = np.arange(3)
    width = .34
    fig, ax = plt.subplots(figsize=(4.8, 3.8), constrained_layout=True)
    ax.bar(x - width / 2, ordered.theory_plus_realization_r, width, color="#247BA0", label="theory")
    ax.bar(x + width / 2, ordered.simulation_r, width, color="#D1495B", label="simulation")
    ax.errorbar(x + width / 2, ordered.simulation_r, yerr=ordered.simulation_r_sd, fmt="none", ecolor="0.2", capsize=3, lw=1.2)
    ax.axhline(0, color="0.25", lw=1)
    ax.set_xticks(x, ["Baseline", "Node-deleted", "Delta"])
    ax.set(ylabel="P-A correlation r", title=f"Experiment 3: {selected['input_area']} input to {selected['output_area']} readout")
    ax.legend(frameon=False, loc="upper left")
    _panel_label(ax, "I")
    _save(fig, out / "panels" / "I_exp3_node_deletion_correlations")
    return {"deleted_nodes": selected["nodes"], "input_area": selected["input_area"], "output_area": selected["output_area"]}


def make_combined(out: Path) -> None:
    panel_files = [
        "A_baseline_input_output_psd",
        "B_exp1_V1_PG_theory_simulation",
        "C_exp1_9x9_correlation_matrix",
        "D_exp2_eigenvalue_transitions",
        "E_exp2_three_regime_psd",
        "F_exp2_nearcritical_input_output_psd",
        "G_exp2_1000trial_theory_simulation",
        "H_exp3_baseline_deleted_J_matrices",
        "I_exp3_node_deletion_correlations",
    ]
    fig, axes = plt.subplots(3, 3, figsize=(17.0, 14.2), constrained_layout=True)
    for ax, name in zip(axes.flat, panel_files):
        image = plt.imread(out / "panels" / f"{name}.png")
        ax.imshow(image)
        ax.axis("off")
    fig.suptitle("Network filtering links periodic and aperiodic dynamics", fontsize=18, fontweight="bold")
    _save(fig, out / "figures" / "paper_figure_A_to_I")


def run(cfg: dict, force: bool = False) -> Path:
    started = time.time()
    _style()
    out = _output(cfg)
    model = load_model(_rooted(cfg["model_dir"]))
    summary = {
        "panel_A": make_panel_a(cfg, model, out),
        "panel_B": make_panel_b(out),
        "panel_C": make_panel_c(out),
        "panel_D": make_panel_d(out),
        "panel_E": make_panel_e(out),
        "panel_F": make_panel_f(out),
        "panel_G": make_panel_g(out),
    }
    exp3_results, selected = run_exp3_observation_matched(cfg, model, out, force=force)
    summary["panels_H_I"] = make_panel_h_i(model, out, exp3_results, selected)
    make_combined(out)
    summary.update(
        {
            "runtime_s": time.time() - started,
            "seed": SEED,
            "exp3_analysis_s": EXP3_ANALYSIS_S,
            "exp3_trials": EXP3_TRIALS,
            "calibration_phases": CALIBRATION_PHASES,
            "heldout_phases": HELDOUT_PHASES,
            "source_directory": str(out.resolve()),
        }
    )
    (out / "manifest.json").write_text(json.dumps(summary, indent=2))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(run(cfg, force=args.force))


if __name__ == "__main__":
    main()
