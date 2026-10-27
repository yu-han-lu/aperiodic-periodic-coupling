#!/usr/bin/env python3
"""
Standard-condition marmoset cortex simulation from Li, Li & Wang (2025).

This file extracts the standard simulation path from pre_functions_clean.py into
a compact, runnable script:

1. Load marmoset FLN/SLN/hierarchy data.
2. Build the standard 55-area E/I firing-rate model.
3. Compute eigenmode time constants and localization statistics.
4. Compute analytical functional connectivity for the linear system.
5. Run pulse and resting/noise simulations.
6. Save summary tables, arrays, and basic figures.

The standard condition follows the paper notebooks:
MACAQUE_CASE=0, FIT_HIER=1, original FLN, no shuffle/ablation,
diverse_hi=41.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from typing import Any

import numpy as np


def load_data(datafile: Path) -> dict[str, Any]:
    """Load connectome data and normalize the structural hierarchy."""
    with datafile.open("rb") as f:
        p = pickle.load(f, encoding="latin1")
    p["hier_vals"] = np.asarray(p["hier_vals"], dtype=float)
    p["hier_vals"] = p["hier_vals"] / np.max(p["hier_vals"])
    p["fln_mat"] = np.asarray(p["fln_mat"], dtype=float)
    p["sln_mat"] = np.asarray(p["sln_mat"], dtype=float)
    p["areas"] = list(p["areas"])
    p["n_area"] = len(p["areas"])
    return p


def apply_standard_parameters(
    p_in: dict[str, Any],
    hier_fit_file: Path | None = None,
    fit_hier: bool = True,
    diverse_hi: int = 41,
    eta_val: float = 0.68,
    eta_local_val: float = 0.68,
    long_amp: float = 1.0,
    str_factor: float = 1.0,
) -> dict[str, Any]:
    """Set the standard marmoset model parameters used by the notebooks."""
    p = p_in.copy()
    p["fln_mat"] = p_in["fln_mat"].copy()
    p["sln_mat"] = p_in["sln_mat"].copy()
    p["areas"] = list(p_in["areas"])
    p["hier_vals"] = p_in["hier_vals"].copy()

    if fit_hier:
        if hier_fit_file is None:
            raise ValueError("fit_hier=True requires hier_fit_file.")
        p["hier_vals"] = np.load(hier_fit_file).astype(float)
    p["hier_vals_inh"] = p["hier_vals"].copy()

    p["beta_exc"] = 0.066
    p["beta_inh"] = 0.351
    p["tau_exc"] = 20.0
    p["tau_inh"] = 10.0
    p["wEE"] = 24.4 * str_factor
    p["wIE"] = 12.2 * str_factor
    p["wEI"] = 19.7 * str_factor
    p["wII"] = 12.5 * str_factor
    p["muEE"] = 33.7 * long_amp * str_factor
    p["muIE"] = 25.3 * long_amp * str_factor
    p["eta"] = eta_val
    p["eta_local"] = eta_local_val
    p["eta_inh"] = eta_val
    p["eta_inh_local"] = eta_local_val

    if diverse_hi == 41:
        p["muEE"] *= 2.0
        p["wIE"] = 11.660380103696516
        p["eta_inh_local"] = 0.76
        p["muIE"] = 49.81041566951567
        p["eta_inh"] = 0.76
        p["eta"] += 5e-3
        p["eta_local"] += 5e-3
    elif diverse_hi != 0:
        raise ValueError("This clean standard script implements diverse_hi=0 or 41 only.")

    p["exc_scale"] = 1.0 + p["eta"] * p["hier_vals"]
    p["local_exc_scale"] = 1.0 + p["eta_local"] * p["hier_vals"]
    p["inh_scale"] = 1.0 + p["eta_inh"] * p["hier_vals_inh"]
    p["local_inh_scale"] = 1.0 + p["eta_inh_local"] * p["hier_vals"]
    return p


def build_connectivity_matrix(p: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Build interleaved W and block-ordered W_EI."""
    n = p["n_area"]
    local_EE = p["beta_exc"] * p["wEE"] * p["local_exc_scale"]
    local_EI = -p["beta_exc"] * p["wEI"] * np.ones(n)
    local_IE = p["beta_inh"] * p["wIE"] * p["local_inh_scale"]
    local_II = -p["beta_inh"] * p["wII"] * np.ones(n)

    fln_scaled_E = (p["exc_scale"] * p["fln_mat"].T).T
    fln_scaled_I = (p["inh_scale"] * p["fln_mat"].T).T

    W = np.zeros((2 * n, 2 * n), dtype=float)
    for i in range(n):
        W[2 * i, 2 * i] = (local_EE[i] - 1.0) / p["tau_exc"]
        W[2 * i, 2 * i + 1] = local_EI[i] / p["tau_exc"]
        W[2 * i + 1, 2 * i] = local_IE[i] / p["tau_inh"]
        W[2 * i + 1, 2 * i + 1] = (local_II[i] - 1.0) / p["tau_inh"]
        W[2 * i, ::2] += p["beta_exc"] * p["muEE"] * fln_scaled_E[i, :] / p["tau_exc"]
        W[2 * i + 1, ::2] += p["beta_inh"] * p["muIE"] * fln_scaled_I[i, :] / p["tau_inh"]

    W_EI = np.zeros_like(W)
    W_EI[:n, :n] = W[0::2, 0::2]
    W_EI[:n, n:] = W[0::2, 1::2]
    W_EI[n:, :n] = W[1::2, 0::2]
    W_EI[n:, n:] = W[1::2, 1::2]
    return W, W_EI


def inverse_participation_ratio(vec: np.ndarray, power: int = 4) -> float:
    """Inverse participation ratio used in the original code."""
    v = np.abs(vec)
    return float(np.sum(v**power) / (np.sum(v**2) ** 2 + 2e-16))


def eigen_statistics(W_EI: np.ndarray, n_area: int) -> dict[str, np.ndarray | float | int]:
    """Compute eigenvalues, sorted time constants, and IPR values."""
    eigvals, eigvecs = np.linalg.eig(W_EI)
    tau = -1.0 / np.real(eigvals)
    order = np.argsort(-tau)
    tau_sorted = tau[order]
    eigvals_sorted = eigvals[order]
    eigvecs_sorted = eigvecs[:, order]
    ipr = np.array([inverse_participation_ratio(eigvecs_sorted[:, i]) for i in range(2 * n_area)])
    return {
        "eigvals": eigvals_sorted,
        "eigvecs": eigvecs_sorted,
        "tau": tau_sorted,
        "ipr": ipr,
        "max_real_eig": float(np.max(np.real(eigvals))),
        "positive_real_eig_count": int(np.sum(np.real(eigvals) > 0)),
    }


def analytical_functional_connectivity(
    W_EI: np.ndarray,
    n_area: int,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytical FC for the linear system in E/I block order."""
    eigvals, eigvecs = np.linalg.eig(W_EI)
    inv_eigvecs = np.linalg.inv(eigvecs)
    B = np.zeros_like(W_EI)
    for i in range(n_area):
        B[i, i] = sigma

    Q = inv_eigvecs @ B @ B.conj().T @ inv_eigvecs.conj().T
    M = np.zeros_like(Q, dtype=complex)
    for i in range(2 * n_area):
        for j in range(2 * n_area):
            M[i, j] = -Q[i, j] / (eigvals[i] + eigvals[j].conj())

    cov = eigvecs @ M @ eigvecs.conj().T
    cov_E = cov[:n_area, :n_area]
    corr_E = np.zeros((n_area, n_area), dtype=float)
    for i in range(n_area):
        for j in range(n_area):
            denom = np.sqrt(cov[i, i] * cov[j, j])
            corr_E[i, j] = np.real(cov[i, j] / denom)
    return corr_E, cov_E


def gaussian_noise(mu: float, sigma: float, n_t: int) -> np.ndarray:
    """Match the original random.gauss-based noise generator."""
    return np.array([random.gauss(mu, sigma) for _ in range(n_t)])


def powerlaw_noise(
    n_t: int,
    target_psd_slope: float,
    sigma: float,
    rng: np.random.Generator,
    mu: float = 0.0,
) -> np.ndarray:
    """Generate zero-mean noise with PSD approximately proportional to f^slope."""
    white = rng.normal(0.0, 1.0, n_t)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n_t)
    scale = np.ones_like(freqs)
    nonzero = freqs > 0
    scale[nonzero] = freqs[nonzero] ** (target_psd_slope / 2.0)
    scale[~nonzero] = 0.0
    colored = np.fft.irfft(spectrum * scale, n=n_t)
    colored = colored - np.mean(colored)
    std = np.std(colored)
    if std > 0:
        colored = colored / std * sigma
    return colored + mu


def square_wave_input(
    t_ms: np.ndarray,
    frequency_hz: float,
    amplitude: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a zero-mean symmetric square wave in model input units."""
    t_seconds = t_ms / 1000.0
    phase_signal = np.sin(2.0 * np.pi * frequency_hz * t_seconds + phase)
    return amplitude * np.where(phase_signal >= 0.0, 1.0, -1.0)


def sine_wave_input(
    t_ms: np.ndarray,
    frequency_hz: float,
    amplitude: float,
    phase: float = 0.0,
) -> np.ndarray:
    """Generate a zero-mean sinusoidal input in model input units."""
    t_seconds = t_ms / 1000.0
    return amplitude * np.sin(2.0 * np.pi * frequency_hz * t_seconds + phase)


def simulate_stimulus(
    p: dict[str, Any],
    stim_area: str = "V1",
    T: float = 6000.0,
    dt: float = 0.05,
    pulse_input: bool = True,
    total_input: bool = False,
    seed: int = 1,
    stim_noise: str = "white",
    stim_noise_slope: float = 0.5,
    stim_noise_sigma: float = 0.5,
    square_frequency_hz: float = 1.0,
    square_amplitude: float = 0.5,
    sine_frequency_hz: float = 1.0,
    sine_amplitude: float = 0.5,
    sine_white_sigma: float = 0.5,
    sine_powerlaw_sigma: float = 0.5,
    sine_powerlaw_slope: float = 0.0,
) -> dict[str, Any]:
    """Run the standard threshold-linear E/I firing-rate simulation."""
    random.seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n = p["n_area"]
    local_EE = p["beta_exc"] * p["wEE"] * p["local_exc_scale"]
    local_EI = -p["beta_exc"] * p["wEI"] * np.ones(n)
    local_IE = p["beta_inh"] * p["wIE"] * p["local_inh_scale"]
    local_II = -p["beta_inh"] * p["wII"] * np.ones(n)
    fln_scaled_E = (p["exc_scale"] * p["fln_mat"].T).T
    fln_scaled_I = (p["inh_scale"] * p["fln_mat"].T).T

    t_plot = np.linspace(0.0, T, int(T / dt) + 1)
    n_t = len(t_plot)

    r_exc_base = 10.0
    r_exc_tgt = r_exc_base * np.ones(n)
    r_inh_tgt = 35.0 * np.ones(n)
    longrange_E0 = fln_scaled_E @ r_exc_tgt
    longrange_I0 = fln_scaled_I @ r_exc_tgt
    I_bkg_exc = r_exc_tgt - (
        local_EE * r_exc_tgt + local_EI * r_inh_tgt + p["beta_exc"] * p["muEE"] * longrange_E0
    )
    I_bkg_inh = r_inh_tgt - (
        local_IE * r_exc_tgt + local_II * r_inh_tgt + p["beta_inh"] * p["muIE"] * longrange_I0
    )

    I_stim_exc = np.zeros((n_t, n), dtype=float)
    stim_idx = p["areas"].index(stim_area)
    if pulse_input:
        time_idx = (t_plot > 200.0) & (t_plot <= 400.0)
        I_stim_exc[time_idx, stim_idx] = 41.187
    else:
        for i in range(n):
            I_stim_exc[:, i] = gaussian_noise(0.0, 1e-5, n_t)
        if not total_input:
            if stim_noise == "white":
                I_stim_exc[:, stim_idx] = gaussian_noise(0.0, stim_noise_sigma, n_t)
            elif stim_noise in {"powerlaw", "fbm", "fractional_brownian"}:
                I_stim_exc[:, stim_idx] = powerlaw_noise(
                    n_t,
                    target_psd_slope=stim_noise_slope,
                    sigma=stim_noise_sigma,
                    rng=rng,
                )
            elif stim_noise == "square":
                I_stim_exc[:, stim_idx] = square_wave_input(
                    t_plot,
                    frequency_hz=square_frequency_hz,
                    amplitude=square_amplitude,
                )
            elif stim_noise == "sine_plus_white":
                I_stim_exc[:, stim_idx] = sine_wave_input(
                    t_plot,
                    frequency_hz=sine_frequency_hz,
                    amplitude=sine_amplitude,
                ) + gaussian_noise(0.0, sine_white_sigma, n_t)
            elif stim_noise == "sine_plus_powerlaw":
                I_stim_exc[:, stim_idx] = sine_wave_input(
                    t_plot,
                    frequency_hz=sine_frequency_hz,
                    amplitude=sine_amplitude,
                ) + powerlaw_noise(
                    n_t,
                    target_psd_slope=sine_powerlaw_slope,
                    sigma=sine_powerlaw_sigma,
                    rng=rng,
                )
            else:
                raise ValueError(f"Unknown stim_noise: {stim_noise}")

    r_exc = np.zeros((n_t, n), dtype=float)
    r_inh = np.zeros((n_t, n), dtype=float)
    r_exc[0] = r_exc_tgt
    r_inh[0] = r_inh_tgt

    for i_t in range(1, n_t):
        longrange_E = fln_scaled_E @ r_exc[i_t - 1]
        longrange_I = fln_scaled_I @ r_exc[i_t - 1]
        I_exc = (
            local_EE * r_exc[i_t - 1]
            + local_EI * r_inh[i_t - 1]
            + p["beta_exc"] * p["muEE"] * longrange_E
            + I_bkg_exc
            + I_stim_exc[i_t]
        )
        I_inh = (
            local_IE * r_exc[i_t - 1]
            + local_II * r_inh[i_t - 1]
            + p["beta_inh"] * p["muIE"] * longrange_I
            + I_bkg_inh
        )
        d_r_exc = -r_exc[i_t - 1] + np.maximum(I_exc, 0.0)
        d_r_inh = -r_inh[i_t - 1] + np.maximum(I_inh, 0.0)
        r_exc[i_t] = r_exc[i_t - 1] + d_r_exc * dt / p["tau_exc"]
        r_inh[i_t] = r_inh[i_t - 1] + d_r_inh * dt / p["tau_inh"]

    max_rate = np.max(r_exc - r_exc_base, axis=0)
    decay_time = np.zeros(n, dtype=float)
    if pulse_input:
        pulse_end = np.where(time_idx)[0][-1]
        for area_idx in range(n):
            trace = r_exc[pulse_end:-1, area_idx].copy() - np.min(r_exc[:, area_idx])
            peak = np.max(trace)
            if peak <= 0:
                decay_time[area_idx] = np.nan
                continue
            normalized = trace / peak
            above = np.where(normalized > 1.0 / np.e)[0]
            decay_time[area_idx] = above[-1] * dt if len(above) else 0.0

    return {
        "I_stim_exc": I_stim_exc,
        "r_exc": r_exc,
        "r_inh": r_inh,
        "stim_idx": stim_idx,
        "dt": dt,
        "t_plot": t_plot,
        "decay_time": decay_time,
        "max_rate": max_rate,
    }


def write_area_table(path: Path, p: dict[str, Any], max_rate: np.ndarray, decay_time: np.ndarray, ipr: np.ndarray) -> None:
    """Write per-area statistics as CSV."""
    header = "area,hier_val,max_rate_hz_above_baseline,decay_time_ms,slowmode_ipr\n"
    rows = []
    for i, area in enumerate(p["areas"]):
        rows.append(f"{area},{p['hier_vals'][i]:.10g},{max_rate[i]:.10g},{decay_time[i]:.10g},{ipr[i]:.10g}\n")
    path.write_text(header + "".join(rows), encoding="utf-8")


def maybe_make_figures(
    outdir: Path,
    p: dict[str, Any],
    tau: np.ndarray,
    corr: np.ndarray,
    sim: dict[str, Any],
    resting: dict[str, Any] | None = None,
) -> None:
    """Save compact reproduction figures."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "./.mplconfig")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(p["hier_vals"], tau[: p["n_area"]], s=24)
    ax.set_xlabel("Fitted excitation gradient h_i")
    ax.set_ylabel("Eigenmode time constant (ms)")
    ax.set_yscale("log")
    fig.tight_layout()
    fig.savefig(outdir / "standard_tau_vs_gradient.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(corr - np.diag(np.diag(corr)), cmap="coolwarm", aspect="auto")
    ax.set_title("Analytical FC, diagonal removed")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(outdir / "standard_functional_connectivity.png", dpi=180)
    plt.close(fig)

    if resting is not None:
        selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
        fig, axes = plt.subplots(len(selected) + 1, 1, sharex=True, figsize=(8, 8), facecolor=(1, 1, 1))
        t = resting["t_plot"]
        axes[0].plot(t, resting["I_stim_exc"][:, resting["stim_idx"]], color="k", lw=0.8)
        axes[0].text(0.9, 0.55, "Input", transform=axes[0].transAxes, fontsize=10)
        for ax, area in zip(axes[1:], selected):
            idx = p["areas"].index(area)
            ax.plot(t, resting["r_exc"][:, idx], lw=0.8)
            ax.text(0.9, 0.55, area, transform=ax.transAxes, fontsize=10)
        for ax in axes:
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        axes[-1].set_xlabel("Time (ms)")
        fig.text(0.01, 0.5, "Firing rate (Hz)", va="center", rotation="vertical")
        fig.tight_layout(rect=(0.04, 0.02, 1, 1))
        fig.savefig(outdir / "standard_fig3a_resting_activity.png", dpi=180)
        plt.close(fig)

        from scipy import signal

        dt_seconds = resting["dt"] / 1000.0
        fs = 1.0 / dt_seconds
        nperseg = min(len(t), int(fs))
        fig, axes = plt.subplots(3, 3, figsize=(10, 8), facecolor=(1, 1, 1))
        psd_rows = ["area,frequency_hz,psd\n"]
        slope_rows = ["area,fit_min_hz,fit_max_hz,slope,intercept,r2,n_points\n"]
        fit_min_hz = 100.0
        fit_max_hz = 1000.0
        for ax, area in zip(axes.flat, selected):
            idx = p["areas"].index(area)
            freq, psd = signal.welch(resting["r_exc"][:, idx], fs=fs, nperseg=nperseg, detrend="constant")
            valid = (freq > 0) & (psd > 0)
            fit_mask = valid & (freq >= fit_min_hz) & (freq <= fit_max_hz)
            slope = np.nan
            intercept = np.nan
            r2 = np.nan
            if np.sum(fit_mask) >= 2:
                log_f = np.log10(freq[fit_mask])
                log_psd = np.log10(psd[fit_mask])
                slope, intercept = np.polyfit(log_f, log_psd, 1)
                pred = slope * log_f + intercept
                ss_res = np.sum((log_psd - pred) ** 2)
                ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                fit_freq = np.array([fit_min_hz, fit_max_hz])
                fit_psd = 10 ** (slope * np.log10(fit_freq) + intercept)
                ax.loglog(fit_freq, fit_psd, color="crimson", lw=1.2, linestyle="--")
            ax.loglog(freq[valid], psd[valid], lw=0.9)
            ax.set_title(f"{area}: slope={slope:.2f}", fontsize=10)
            ax.set_xlabel("Frequency (Hz)", fontsize=8)
            ax.set_ylabel("PSD", fontsize=8)
            ax.tick_params(labelsize=7)
            for f_i, p_i in zip(freq[valid], psd[valid]):
                psd_rows.append(f"{area},{f_i:.10g},{p_i:.10g}\n")
            slope_rows.append(
                f"{area},{fit_min_hz:.10g},{fit_max_hz:.10g},"
                f"{slope:.10g},{intercept:.10g},{r2:.10g},{int(np.sum(fit_mask))}\n"
            )
        fig.tight_layout()
        fig.savefig(outdir / "standard_fig3a_resting_psd_loglog.png", dpi=180)
        plt.close(fig)
        (outdir / "standard_fig3a_resting_psd.csv").write_text("".join(psd_rows), encoding="utf-8")
        (outdir / "standard_fig3a_resting_psd_slope.csv").write_text("".join(slope_rows), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(8, 4))
    selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    for area in selected:
        if area in p["areas"]:
            idx = p["areas"].index(area)
            trace = sim["r_exc"][:, idx] - 10.0
            peak = np.nanmax(trace)
            if peak > 0:
                trace = 100.0 * trace / peak
            ax.plot(sim["t_plot"], trace, label=area, lw=1)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized excitatory response (% peak)")
    ax.set_ylim(bottom=0)
    ax.legend(ncol=3, fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "standard_v1_pulse_traces.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    max_rate = np.maximum(sim["max_rate"], np.finfo(float).tiny)
    ax.bar(np.arange(p["n_area"]), max_rate)
    ax.set_yscale("log")
    ax.set_xticks(np.arange(p["n_area"]))
    ax.set_xticklabels(p["areas"], rotation=90, fontsize=6)
    ax.set_ylabel("Max rate above baseline (Hz)")
    fig.tight_layout()
    fig.savefig(outdir / "standard_v1_max_response.png", dpi=180)
    plt.close(fig)


def fit_loglog_slope(freq: np.ndarray, psd: np.ndarray, fit_min_hz: float, fit_max_hz: float) -> dict[str, float | int]:
    """Fit log10(PSD) = slope * log10(freq) + intercept in a frequency band."""
    mask = (freq > 0) & (psd > 0) & (freq >= fit_min_hz) & (freq <= fit_max_hz)
    if np.sum(mask) < 2:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan, "n_points": int(np.sum(mask))}
    log_f = np.log10(freq[mask])
    log_psd = np.log10(psd[mask])
    slope, intercept = np.polyfit(log_f, log_psd, 1)
    pred = slope * log_f + intercept
    ss_res = np.sum((log_psd - pred) ** 2)
    ss_tot = np.sum((log_psd - np.mean(log_psd)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"slope": float(slope), "intercept": float(intercept), "r2": float(r2), "n_points": int(np.sum(mask))}


def plot_psd_slope_grid(
    outdir: Path,
    p: dict[str, Any],
    sim: dict[str, Any],
    filename_stem: str,
    title: str,
) -> None:
    """Plot PSD for Fig. 3A areas with 10-100 Hz and 100-1000 Hz fitted slopes."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "./.mplconfig")
    import matplotlib.pyplot as plt
    from scipy import signal

    selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    fit_bands = [(10.0, 100.0, "crimson"), (100.0, 1000.0, "darkorange")]
    fs = 1.0 / (sim["dt"] / 1000.0)
    nperseg = min(len(sim["t_plot"]), int(fs))

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor=(1, 1, 1))
    slope_rows = ["condition,area,fit_min_hz,fit_max_hz,slope,intercept,r2,n_points\n"]
    for ax, area in zip(axes.flat, selected):
        idx = p["areas"].index(area)
        freq, psd = signal.welch(sim["r_exc"][:, idx], fs=fs, nperseg=nperseg, detrend="constant")
        valid = (freq > 0) & (psd > 0)
        ax.loglog(freq[valid], psd[valid], color="steelblue", lw=0.9)

        label_parts = []
        for fit_min, fit_max, color in fit_bands:
            fit = fit_loglog_slope(freq, psd, fit_min, fit_max)
            slope = fit["slope"]
            intercept = fit["intercept"]
            if np.isfinite(slope) and np.isfinite(intercept):
                fit_freq = np.array([fit_min, fit_max])
                fit_psd = 10 ** (slope * np.log10(fit_freq) + intercept)
                ax.loglog(fit_freq, fit_psd, color=color, lw=1.4, linestyle="--")
            label_parts.append(f"{int(fit_min)}-{int(fit_max)}Hz: {slope:.2f}")
            slope_rows.append(
                f"{filename_stem},{area},{fit_min:.10g},{fit_max:.10g},"
                f"{slope:.10g},{intercept:.10g},{fit['r2']:.10g},{fit['n_points']}\n"
            )

        ax.set_title(area, fontsize=10)
        ax.text(0.03, 0.06, "\n".join(label_parts), transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / f"{filename_stem}.png", dpi=180)
    plt.close(fig)
    (outdir / f"{filename_stem}_slopes.csv").write_text("".join(slope_rows), encoding="utf-8")


def compute_psd_band_slopes(
    p: dict[str, Any],
    sim: dict[str, Any],
    areas: list[str],
    fit_bands: list[tuple[float, float]],
) -> dict[str, dict[tuple[float, float], dict[str, float | int]]]:
    """Compute PSD log-log slopes for selected areas and frequency bands."""
    from scipy import signal

    fs = 1.0 / (sim["dt"] / 1000.0)
    nperseg = min(len(sim["t_plot"]), int(fs))
    slopes: dict[str, dict[tuple[float, float], dict[str, float | int]]] = {}
    for area in areas:
        idx = p["areas"].index(area)
        freq, psd = signal.welch(sim["r_exc"][:, idx], fs=fs, nperseg=nperseg, detrend="constant")
        slopes[area] = {}
        for fit_min, fit_max in fit_bands:
            slopes[area][(fit_min, fit_max)] = fit_loglog_slope(freq, psd, fit_min, fit_max)
    return slopes


def generate_beta_sweep_psd_slopes(outdir: Path, p: dict[str, Any], args: argparse.Namespace) -> None:
    """Sweep V1 power-law/fBM-like input PSD slope beta and plot output PSD slopes by area."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "./.mplconfig")
    import matplotlib.pyplot as plt

    selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    fit_bands = [(10.0, 100.0), (100.0, 1000.0)]
    betas = np.round(np.arange(args.beta_min, args.beta_max + args.beta_step / 2.0, args.beta_step), 10)
    colors = {(10.0, 100.0): "crimson", (100.0, 1000.0): "darkorange"}
    labels = {(10.0, 100.0): "10-100 Hz", (100.0, 1000.0): "100-1000 Hz"}

    results: dict[str, dict[tuple[float, float], list[float]]] = {
        area: {band: [] for band in fit_bands} for area in selected
    }
    rows = ["beta,area,fit_min_hz,fit_max_hz,slope,intercept,r2,n_points\n"]

    for beta in betas:
        sim = simulate_stimulus(
            p,
            stim_area="V1",
            T=args.rest_T,
            pulse_input=False,
            total_input=False,
            seed=args.seed,
            stim_noise="fractional_brownian",
            stim_noise_slope=float(beta),
            stim_noise_sigma=args.beta_sweep_sigma,
        )
        slopes = compute_psd_band_slopes(p, sim, selected, fit_bands)
        for area in selected:
            for band in fit_bands:
                fit = slopes[area][band]
                results[area][band].append(float(fit["slope"]))
                rows.append(
                    f"{beta:.10g},{area},{band[0]:.10g},{band[1]:.10g},"
                    f"{fit['slope']:.10g},{fit['intercept']:.10g},{fit['r2']:.10g},{fit['n_points']}\n"
                )

    (outdir / "beta_sweep_v1_fbm_psd_slopes.csv").write_text("".join(rows), encoding="utf-8")

    for area in selected:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor=(1, 1, 1))
        for band in fit_bands:
            ax.plot(
                betas,
                results[area][band],
                marker="o",
                lw=1.6,
                ms=4,
                color=colors[band],
                label=labels[band],
            )
        ax.axhline(0.0, color="0.75", lw=0.8)
        ax.set_title(f"{area}: output PSD slope vs input beta")
        ax.set_xlabel("Input PSD slope beta")
        ax.set_ylabel("Output PSD slope")
        ax.set_xticks(betas)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(outdir / f"beta_sweep_psd_slope_{area}.png", dpi=180)
        plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor=(1, 1, 1), sharex=True)
    for ax, area in zip(axes.flat, selected):
        for band in fit_bands:
            ax.plot(betas, results[area][band], marker="o", lw=1.2, ms=3, color=colors[band], label=labels[band])
        ax.axhline(0.0, color="0.8", lw=0.7)
        ax.set_title(area, fontsize=10)
        ax.tick_params(labelsize=7)
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False)
    fig.supxlabel("Input PSD slope beta")
    fig.supylabel("Output PSD slope")
    fig.suptitle(f"V1 power-law/fBM-like input sigma={args.beta_sweep_sigma}: PSD slope beta sweep", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(outdir / "beta_sweep_psd_slope_3x3_summary.png", dpi=180)
    plt.close(fig)


def generate_square_wave_psd(outdir: Path, p: dict[str, Any], args: argparse.Namespace) -> None:
    """Drive V1 with a square wave and plot PSD, fitted slopes, and dominant frequency."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "./.mplconfig")
    import matplotlib.pyplot as plt
    from scipy import signal

    selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    fit_bands = [(10.0, 100.0, "crimson"), (100.0, 1000.0, "darkorange")]
    sim = simulate_stimulus(
        p,
        stim_area="V1",
        T=args.square_T,
        pulse_input=False,
        total_input=False,
        seed=args.seed,
        stim_noise="square",
        square_frequency_hz=args.square_frequency,
        square_amplitude=args.square_amplitude,
    )

    fs = 1.0 / (sim["dt"] / 1000.0)
    nperseg = min(len(sim["t_plot"]), int(fs * args.square_welch_seconds))
    peak_min_hz = args.square_peak_min_hz
    peak_max_hz = args.square_peak_max_hz

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor=(1, 1, 1))
    slope_rows = ["area,fit_min_hz,fit_max_hz,slope,intercept,r2,n_points\n"]
    peak_rows = ["area,dominant_frequency_hz,dominant_psd\n"]
    for ax, area in zip(axes.flat, selected):
        idx = p["areas"].index(area)
        freq, psd = signal.welch(sim["r_exc"][:, idx], fs=fs, nperseg=nperseg, detrend="constant")
        valid = (freq > 0) & (psd > 0)
        peak_mask = valid & (freq >= peak_min_hz) & (freq <= peak_max_hz)
        if np.any(peak_mask):
            peak_candidates = np.where(peak_mask)[0]
            peak_idx = peak_candidates[int(np.argmax(psd[peak_candidates]))]
            peak_freq = float(freq[peak_idx])
            peak_psd = float(psd[peak_idx])
        else:
            peak_freq = np.nan
            peak_psd = np.nan

        ax.loglog(freq[valid], psd[valid], color="steelblue", lw=0.9)
        if np.isfinite(peak_freq):
            ax.axvline(peak_freq, color="black", lw=1.0, linestyle=":", alpha=0.85)

        label_parts = []
        for fit_min, fit_max, color in fit_bands:
            fit = fit_loglog_slope(freq, psd, fit_min, fit_max)
            slope = fit["slope"]
            intercept = fit["intercept"]
            if np.isfinite(slope) and np.isfinite(intercept):
                fit_freq = np.array([fit_min, fit_max])
                fit_psd = 10 ** (slope * np.log10(fit_freq) + intercept)
                ax.loglog(fit_freq, fit_psd, color=color, lw=1.4, linestyle="--")
            label_parts.append(f"{int(fit_min)}-{int(fit_max)}Hz: {slope:.2f}")
            slope_rows.append(
                f"{area},{fit_min:.10g},{fit_max:.10g},"
                f"{slope:.10g},{intercept:.10g},{fit['r2']:.10g},{fit['n_points']}\n"
            )

        peak_rows.append(f"{area},{peak_freq:.10g},{peak_psd:.10g}\n")
        ax.set_title(f"{area}: peak {peak_freq:.2f} Hz", fontsize=10)
        ax.text(0.03, 0.06, "\n".join(label_parts), transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"V1 square-wave input: {args.square_frequency:g} Hz, amplitude={args.square_amplitude:g}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "square_wave_v1_psd_3x3.png", dpi=180)
    plt.close(fig)
    (outdir / "square_wave_v1_psd_slopes.csv").write_text("".join(slope_rows), encoding="utf-8")
    (outdir / "square_wave_v1_dominant_frequencies.csv").write_text("".join(peak_rows), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 2.6), facecolor=(1, 1, 1))
    ax.plot(sim["t_plot"], sim["I_stim_exc"][:, sim["stim_idx"]], color="black", lw=0.8)
    ax.set_xlim(0, min(args.square_T, 5000.0))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Input")
    ax.set_title(f"V1 square-wave input ({args.square_frequency:g} Hz, amplitude={args.square_amplitude:g})")
    fig.tight_layout()
    fig.savefig(outdir / "square_wave_v1_input_trace.png", dpi=180)
    plt.close(fig)


def generate_sine_white_psd(outdir: Path, p: dict[str, Any], args: argparse.Namespace) -> None:
    """Drive V1 with a sine wave plus white noise and plot PSD diagnostics."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "./.mplconfig")
    import matplotlib.pyplot as plt
    from scipy import signal

    selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    fit_bands = [(10.0, 100.0, "crimson"), (100.0, 1000.0, "darkorange")]
    sim = simulate_stimulus(
        p,
        stim_area="V1",
        T=args.sine_white_T,
        pulse_input=False,
        total_input=False,
        seed=args.seed,
        stim_noise="sine_plus_white",
        sine_frequency_hz=args.sine_frequency,
        sine_amplitude=args.sine_amplitude,
        sine_white_sigma=args.sine_white_sigma,
    )

    fs = 1.0 / (sim["dt"] / 1000.0)
    nperseg = min(len(sim["t_plot"]), int(fs * args.sine_white_welch_seconds))
    peak_min_hz = args.sine_white_peak_min_hz
    peak_max_hz = args.sine_white_peak_max_hz

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor=(1, 1, 1))
    slope_rows = ["area,fit_min_hz,fit_max_hz,slope,intercept,r2,n_points\n"]
    peak_rows = ["area,dominant_frequency_hz,dominant_psd\n"]
    for ax, area in zip(axes.flat, selected):
        idx = p["areas"].index(area)
        freq, psd = signal.welch(sim["r_exc"][:, idx], fs=fs, nperseg=nperseg, detrend="constant")
        valid = (freq > 0) & (psd > 0)
        peak_mask = valid & (freq >= peak_min_hz) & (freq <= peak_max_hz)
        if np.any(peak_mask):
            peak_candidates = np.where(peak_mask)[0]
            peak_idx = peak_candidates[int(np.argmax(psd[peak_candidates]))]
            peak_freq = float(freq[peak_idx])
            peak_psd = float(psd[peak_idx])
        else:
            peak_freq = np.nan
            peak_psd = np.nan

        ax.loglog(freq[valid], psd[valid], color="steelblue", lw=0.9)
        if np.isfinite(peak_freq):
            ax.axvline(peak_freq, color="black", lw=1.0, linestyle=":", alpha=0.85)

        label_parts = []
        for fit_min, fit_max, color in fit_bands:
            fit = fit_loglog_slope(freq, psd, fit_min, fit_max)
            slope = fit["slope"]
            intercept = fit["intercept"]
            if np.isfinite(slope) and np.isfinite(intercept):
                fit_freq = np.array([fit_min, fit_max])
                fit_psd = 10 ** (slope * np.log10(fit_freq) + intercept)
                ax.loglog(fit_freq, fit_psd, color=color, lw=1.4, linestyle="--")
            label_parts.append(f"{int(fit_min)}-{int(fit_max)}Hz: {slope:.2f}")
            slope_rows.append(
                f"{area},{fit_min:.10g},{fit_max:.10g},"
                f"{slope:.10g},{intercept:.10g},{fit['r2']:.10g},{fit['n_points']}\n"
            )

        peak_rows.append(f"{area},{peak_freq:.10g},{peak_psd:.10g}\n")
        ax.set_title(f"{area}: peak {peak_freq:.2f} Hz", fontsize=10)
        ax.text(0.03, 0.06, "\n".join(label_parts), transform=ax.transAxes, fontsize=7)
        ax.set_xlabel("Frequency (Hz)", fontsize=8)
        ax.set_ylabel("PSD", fontsize=8)
        ax.tick_params(labelsize=7)

    fig.suptitle(
        f"V1 sine+white input: {args.sine_frequency:g} Hz, amp={args.sine_amplitude:g}, noise sigma={args.sine_white_sigma:g}",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "sine_white_v1_psd_3x3.png", dpi=180)
    plt.close(fig)
    (outdir / "sine_white_v1_psd_slopes.csv").write_text("".join(slope_rows), encoding="utf-8")
    (outdir / "sine_white_v1_dominant_frequencies.csv").write_text("".join(peak_rows), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(9, 2.6), facecolor=(1, 1, 1))
    ax.plot(sim["t_plot"], sim["I_stim_exc"][:, sim["stim_idx"]], color="black", lw=0.7)
    ax.set_xlim(0, min(args.sine_white_T, 5000.0))
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Input")
    ax.set_title(
        f"V1 sine+white input ({args.sine_frequency:g} Hz, amp={args.sine_amplitude:g}, noise sigma={args.sine_white_sigma:g})"
    )
    fig.tight_layout()
    fig.savefig(outdir / "sine_white_v1_input_trace.png", dpi=180)
    plt.close(fig)


def compute_psd_metrics(
    p: dict[str, Any],
    sim: dict[str, Any],
    areas: list[str],
    peak_min_hz: float,
    peak_max_hz: float,
    welch_seconds: float,
) -> list[dict[str, float | str]]:
    """Compute dominant frequency and band-limited PSD slopes for selected areas."""
    from scipy import signal

    fit_bands = [(10.0, 100.0), (100.0, 1000.0)]
    fs = 1.0 / (sim["dt"] / 1000.0)
    nperseg = min(len(sim["t_plot"]), int(fs * welch_seconds))
    rows: list[dict[str, float | str]] = []
    for area in areas:
        idx = p["areas"].index(area)
        freq, psd = signal.welch(sim["r_exc"][:, idx], fs=fs, nperseg=nperseg, detrend="constant")
        valid = (freq > 0) & (psd > 0)
        peak_mask = valid & (freq >= peak_min_hz) & (freq <= peak_max_hz)
        if np.any(peak_mask):
            peak_candidates = np.where(peak_mask)[0]
            peak_idx = peak_candidates[int(np.argmax(psd[peak_candidates]))]
            dominant_frequency = float(freq[peak_idx])
            dominant_psd = float(psd[peak_idx])
        else:
            dominant_frequency = np.nan
            dominant_psd = np.nan
        fit_10_100 = fit_loglog_slope(freq, psd, 10.0, 100.0)
        fit_100_1000 = fit_loglog_slope(freq, psd, 100.0, 1000.0)
        rows.append(
            {
                "area": area,
                "dominant_frequency_hz": dominant_frequency,
                "dominant_psd": dominant_psd,
                "slope_10_100": float(fit_10_100["slope"]),
                "r2_10_100": float(fit_10_100["r2"]),
                "slope_100_1000": float(fit_100_1000["slope"]),
                "r2_100_1000": float(fit_100_1000["r2"]),
            }
        )
    return rows


def write_sine_powerlaw_sweep_csv(path: Path, sweep_name: str, sweep_value_name: str, rows: list[dict[str, float | str]]) -> None:
    """Write sine+powerlaw sweep metrics to CSV."""
    header = (
        f"sweep,sweep_value,omega_hz,beta,area,dominant_frequency_hz,dominant_psd,"
        "slope_10_100,r2_10_100,slope_100_1000,r2_100_1000\n"
    )
    lines = [header]
    for row in rows:
        lines.append(
            f"{sweep_name},{row[sweep_value_name]:.10g},{row['omega_hz']:.10g},{row['beta']:.10g},"
            f"{row['area']},{row['dominant_frequency_hz']:.10g},{row['dominant_psd']:.10g},"
            f"{row['slope_10_100']:.10g},{row['r2_10_100']:.10g},"
            f"{row['slope_100_1000']:.10g},{row['r2_100_1000']:.10g}\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def plot_sine_powerlaw_metric(
    outdir: Path,
    rows: list[dict[str, float | str]],
    x_key: str,
    y_key: str,
    areas: list[str],
    filename: str,
    title: str,
    x_label: str,
    y_label: str,
) -> None:
    """Plot one sweep metric for all selected areas."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=(1, 1, 1))
    for area in areas:
        area_rows = sorted([row for row in rows if row["area"] == area], key=lambda row: float(row[x_key]))
        x = [float(row[x_key]) for row in area_rows]
        y = [float(row[y_key]) for row in area_rows]
        ax.plot(x, y, marker="o", lw=1.4, ms=3, label=area)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.tick_params(labelsize=8)
    ax.legend(ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=180)
    plt.close(fig)


def sample_truncated_normal(
    rng: np.random.Generator,
    mean: float,
    std: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """Sample a truncated normal by rejection sampling for small trial counts."""
    values: list[float] = []
    while len(values) < size:
        draws = rng.normal(mean, std, max(size, 32))
        accepted = draws[(draws >= low) & (draws <= high)]
        values.extend(float(x) for x in accepted)
    return np.array(values[:size], dtype=float)


def generate_random_sine_powerlaw_trials(outdir: Path, p: dict[str, Any], args: argparse.Namespace) -> None:
    """Run random omega/beta sine+powerlaw trials and regress PSD slope against dominant frequency."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "./.mplconfig")
    import matplotlib.pyplot as plt
    from scipy import stats

    selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    rng = np.random.default_rng(args.seed)
    omegas = sample_truncated_normal(
        rng,
        mean=args.random_omega_mean,
        std=np.sqrt(args.random_omega_variance),
        low=args.random_omega_min,
        high=args.random_omega_max,
        size=args.random_trials,
    )
    betas = sample_truncated_normal(
        rng,
        mean=args.random_beta_mean,
        std=args.random_beta_std,
        low=args.random_beta_min,
        high=args.random_beta_max,
        size=args.random_trials,
    )

    metric_rows: list[dict[str, float | str | int]] = []
    for trial_idx, (omega, beta) in enumerate(zip(omegas, betas), start=1):
        sim = simulate_stimulus(
            p,
            stim_area="V1",
            T=args.random_trial_T,
            pulse_input=False,
            total_input=False,
            seed=args.seed + trial_idx,
            stim_noise="sine_plus_powerlaw",
            sine_frequency_hz=float(omega),
            sine_amplitude=args.random_sine_amplitude,
            sine_powerlaw_sigma=args.random_powerlaw_sigma,
            sine_powerlaw_slope=float(beta),
        )
        rows = compute_psd_metrics(
            p,
            sim,
            selected,
            args.random_peak_min_hz,
            args.random_peak_max_hz,
            args.random_welch_seconds,
        )
        for row in rows:
            row["trial"] = trial_idx
            row["omega_hz"] = float(omega)
            row["beta"] = float(beta)
            metric_rows.append(row)

    metric_header = (
        "trial,omega_hz,beta,area,dominant_frequency_hz,dominant_psd,"
        "slope_100_1000,r2_100_1000\n"
    )
    metric_lines = [metric_header]
    for row in metric_rows:
        metric_lines.append(
            f"{row['trial']},{row['omega_hz']:.10g},{row['beta']:.10g},{row['area']},"
            f"{row['dominant_frequency_hz']:.10g},{row['dominant_psd']:.10g},"
            f"{row['slope_100_1000']:.10g},{row['r2_100_1000']:.10g}\n"
        )
    (outdir / "random_sine_powerlaw_trial_metrics.csv").write_text("".join(metric_lines), encoding="utf-8")

    regression_rows = ["area,n,slope,intercept,r_value,r_squared,p_value,std_err\n"]
    regression_by_area: dict[str, dict[str, float]] = {}
    for area in selected:
        area_rows = [row for row in metric_rows if row["area"] == area]
        x = np.array([float(row["dominant_frequency_hz"]) for row in area_rows], dtype=float)
        y = np.array([float(row["slope_100_1000"]) for row in area_rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) >= 3:
            reg = stats.linregress(x[valid], y[valid])
            regression_by_area[area] = {
                "slope": float(reg.slope),
                "intercept": float(reg.intercept),
                "r_value": float(reg.rvalue),
                "r_squared": float(reg.rvalue**2),
                "p_value": float(reg.pvalue),
                "std_err": float(reg.stderr),
                "n": int(np.sum(valid)),
            }
        else:
            regression_by_area[area] = {
                "slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "r_squared": np.nan,
                "p_value": np.nan,
                "std_err": np.nan,
                "n": int(np.sum(valid)),
            }
        reg = regression_by_area[area]
        regression_rows.append(
            f"{area},{reg['n']},{reg['slope']:.10g},{reg['intercept']:.10g},"
            f"{reg['r_value']:.10g},{reg['r_squared']:.10g},{reg['p_value']:.10g},{reg['std_err']:.10g}\n"
        )
    (outdir / "random_sine_powerlaw_regression.csv").write_text("".join(regression_rows), encoding="utf-8")

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor=(1, 1, 1))
    for ax, area in zip(axes.flat, selected):
        area_rows = [row for row in metric_rows if row["area"] == area]
        x = np.array([float(row["dominant_frequency_hz"]) for row in area_rows], dtype=float)
        y = np.array([float(row["slope_100_1000"]) for row in area_rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=16, alpha=0.7, color="steelblue", edgecolor="none")
        reg = regression_by_area[area]
        if np.sum(valid) >= 3 and np.isfinite(reg["slope"]):
            x_line = np.linspace(np.min(x[valid]), np.max(x[valid]), 100)
            y_line = reg["slope"] * x_line + reg["intercept"]
            ax.plot(x_line, y_line, color="crimson", lw=1.4)
        ax.set_title(area, fontsize=10)
        ax.text(
            0.04,
            0.06,
            f"r={reg['r_value']:.2f}\np={reg['p_value']:.2g}",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_xlabel("Dominant frequency (Hz)", fontsize=8)
        ax.set_ylabel("PSD slope, 100-1000 Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.18, lw=0.6)
    fig.suptitle(
        f"Random sine+fBM trials: omega~truncN(5, var=2), beta~truncN(0, {args.random_beta_std:g}^2)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outdir / "random_sine_powerlaw_freq_vs_slope_regression.png", dpi=180)
    plt.close(fig)


def run_random_sine_powerlaw_for_input_area(
    outdir: Path,
    p: dict[str, Any],
    args: argparse.Namespace,
    input_area: str,
    input_area_index: int,
    omegas: np.ndarray,
    betas: np.ndarray,
) -> tuple[list[dict[str, float | str | int]], list[dict[str, float | str | int]]]:
    """Run random sine+powerlaw trials for one input area and save its 3x3 regression figure."""
    import matplotlib.pyplot as plt
    from scipy import stats

    selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    metric_rows: list[dict[str, float | str | int]] = []
    for trial_idx, (omega, beta) in enumerate(zip(omegas, betas), start=1):
        sim = simulate_stimulus(
            p,
            stim_area=input_area,
            T=args.random_trial_T,
            pulse_input=False,
            total_input=False,
            seed=args.seed + input_area_index * 100000 + trial_idx,
            stim_noise="sine_plus_powerlaw",
            sine_frequency_hz=float(omega),
            sine_amplitude=args.random_sine_amplitude,
            sine_powerlaw_sigma=args.random_powerlaw_sigma,
            sine_powerlaw_slope=float(beta),
        )
        rows = compute_psd_metrics(
            p,
            sim,
            selected,
            args.random_peak_min_hz,
            args.random_peak_max_hz,
            args.random_welch_seconds,
        )
        for row in rows:
            row["input_area"] = input_area
            row["trial"] = trial_idx
            row["omega_hz"] = float(omega)
            row["beta"] = float(beta)
            metric_rows.append(row)

    regression_rows: list[dict[str, float | str | int]] = []
    regression_by_area: dict[str, dict[str, float | int]] = {}
    for output_area in selected:
        area_rows = [row for row in metric_rows if row["area"] == output_area]
        x = np.array([float(row["dominant_frequency_hz"]) for row in area_rows], dtype=float)
        y = np.array([float(row["slope_100_1000"]) for row in area_rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.sum(valid) >= 3:
            reg = stats.linregress(x[valid], y[valid])
            reg_row = {
                "input_area": input_area,
                "output_area": output_area,
                "n": int(np.sum(valid)),
                "slope": float(reg.slope),
                "intercept": float(reg.intercept),
                "r_value": float(reg.rvalue),
                "r_squared": float(reg.rvalue**2),
                "p_value": float(reg.pvalue),
                "std_err": float(reg.stderr),
            }
        else:
            reg_row = {
                "input_area": input_area,
                "output_area": output_area,
                "n": int(np.sum(valid)),
                "slope": np.nan,
                "intercept": np.nan,
                "r_value": np.nan,
                "r_squared": np.nan,
                "p_value": np.nan,
                "std_err": np.nan,
            }
        regression_rows.append(reg_row)
        regression_by_area[output_area] = reg_row

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), facecolor=(1, 1, 1))
    for ax, output_area in zip(axes.flat, selected):
        area_rows = [row for row in metric_rows if row["area"] == output_area]
        x = np.array([float(row["dominant_frequency_hz"]) for row in area_rows], dtype=float)
        y = np.array([float(row["slope_100_1000"]) for row in area_rows], dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=16, alpha=0.7, color="steelblue", edgecolor="none")
        reg = regression_by_area[output_area]
        if np.sum(valid) >= 3 and np.isfinite(float(reg["slope"])):
            x_line = np.linspace(np.min(x[valid]), np.max(x[valid]), 100)
            y_line = float(reg["slope"]) * x_line + float(reg["intercept"])
            ax.plot(x_line, y_line, color="crimson", lw=1.4)
        ax.set_title(output_area, fontsize=10)
        ax.text(
            0.04,
            0.06,
            f"r={float(reg['r_value']):.2f}\np={float(reg['p_value']):.2g}",
            transform=ax.transAxes,
            fontsize=8,
        )
        ax.set_xlabel("Dominant frequency (Hz)", fontsize=8)
        ax.set_ylabel("PSD slope, 100-1000 Hz", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.18, lw=0.6)
    fig.suptitle(f"Input to {input_area}: random sine+fBM trials, frequency-slope regression", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    safe_area = input_area.replace("/", "_")
    fig.savefig(outdir / f"random_sine_powerlaw_input_{safe_area}_freq_vs_slope_regression.png", dpi=180)
    plt.close(fig)

    return metric_rows, regression_rows


def generate_random_sine_powerlaw_input_area_sweep(outdir: Path, p: dict[str, Any], args: argparse.Namespace) -> None:
    """Run the random omega/beta regression simulation for each selected input area."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "./.mplconfig")

    input_areas = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    rng = np.random.default_rng(args.seed)
    omegas = sample_truncated_normal(
        rng,
        mean=args.random_omega_mean,
        std=np.sqrt(args.random_omega_variance),
        low=args.random_omega_min,
        high=args.random_omega_max,
        size=args.random_trials,
    )
    betas = sample_truncated_normal(
        rng,
        mean=args.random_beta_mean,
        std=args.random_beta_std,
        low=args.random_beta_min,
        high=args.random_beta_max,
        size=args.random_trials,
    )

    all_metric_rows: list[dict[str, float | str | int]] = []
    all_regression_rows: list[dict[str, float | str | int]] = []
    for input_area_index, input_area in enumerate(input_areas):
        metric_rows, regression_rows = run_random_sine_powerlaw_for_input_area(
            outdir,
            p,
            args,
            input_area,
            input_area_index,
            omegas,
            betas,
        )
        all_metric_rows.extend(metric_rows)
        all_regression_rows.extend(regression_rows)

    metric_header = (
        "input_area,trial,omega_hz,beta,output_area,dominant_frequency_hz,dominant_psd,"
        "slope_100_1000,r2_100_1000\n"
    )
    metric_lines = [metric_header]
    for row in all_metric_rows:
        metric_lines.append(
            f"{row['input_area']},{row['trial']},{row['omega_hz']:.10g},{row['beta']:.10g},{row['area']},"
            f"{row['dominant_frequency_hz']:.10g},{row['dominant_psd']:.10g},"
            f"{row['slope_100_1000']:.10g},{row['r2_100_1000']:.10g}\n"
        )
    (outdir / "random_sine_powerlaw_input_area_sweep_metrics.csv").write_text("".join(metric_lines), encoding="utf-8")

    regression_header = "input_area,output_area,n,slope,intercept,r_value,r_squared,p_value,std_err\n"
    regression_lines = [regression_header]
    for row in all_regression_rows:
        regression_lines.append(
            f"{row['input_area']},{row['output_area']},{row['n']},{row['slope']:.10g},{row['intercept']:.10g},"
            f"{row['r_value']:.10g},{row['r_squared']:.10g},{row['p_value']:.10g},{row['std_err']:.10g}\n"
        )
    (outdir / "random_sine_powerlaw_input_area_sweep_regression.csv").write_text(
        "".join(regression_lines),
        encoding="utf-8",
    )


def generate_sine_powerlaw_sweeps(outdir: Path, p: dict[str, Any], args: argparse.Namespace) -> None:
    """Sweep sine frequency and power-law noise beta for V1 sine+fBM-like input."""
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.setdefault("MPLCONFIGDIR", "./.mplconfig")

    selected = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]
    omega_values = np.arange(args.sine_powerlaw_omega_min, args.sine_powerlaw_omega_max + 0.5, 1.0)
    beta_values = np.round(
        np.arange(
            args.sine_powerlaw_beta_min,
            args.sine_powerlaw_beta_max + args.sine_powerlaw_beta_step / 2.0,
            args.sine_powerlaw_beta_step,
        ),
        10,
    )

    omega_rows: list[dict[str, float | str]] = []
    for omega in omega_values:
        sim = simulate_stimulus(
            p,
            stim_area="V1",
            T=args.sine_powerlaw_T,
            pulse_input=False,
            total_input=False,
            seed=args.seed,
            stim_noise="sine_plus_powerlaw",
            sine_frequency_hz=float(omega),
            sine_amplitude=args.sine_powerlaw_amplitude,
            sine_powerlaw_sigma=args.sine_powerlaw_sigma,
            sine_powerlaw_slope=args.sine_powerlaw_base_beta,
        )
        for row in compute_psd_metrics(
            p,
            sim,
            selected,
            args.sine_powerlaw_peak_min_hz,
            args.sine_powerlaw_peak_max_hz,
            args.sine_powerlaw_welch_seconds,
        ):
            row["omega_hz"] = float(omega)
            row["beta"] = args.sine_powerlaw_base_beta
            omega_rows.append(row)
    write_sine_powerlaw_sweep_csv(outdir / "sine_powerlaw_omega_sweep_metrics.csv", "omega", "omega_hz", omega_rows)

    beta_rows: list[dict[str, float | str]] = []
    for beta in beta_values:
        sim = simulate_stimulus(
            p,
            stim_area="V1",
            T=args.sine_powerlaw_T,
            pulse_input=False,
            total_input=False,
            seed=args.seed,
            stim_noise="sine_plus_powerlaw",
            sine_frequency_hz=args.sine_powerlaw_base_omega,
            sine_amplitude=args.sine_powerlaw_amplitude,
            sine_powerlaw_sigma=args.sine_powerlaw_sigma,
            sine_powerlaw_slope=float(beta),
        )
        for row in compute_psd_metrics(
            p,
            sim,
            selected,
            args.sine_powerlaw_peak_min_hz,
            args.sine_powerlaw_peak_max_hz,
            args.sine_powerlaw_welch_seconds,
        ):
            row["omega_hz"] = args.sine_powerlaw_base_omega
            row["beta"] = float(beta)
            beta_rows.append(row)
    write_sine_powerlaw_sweep_csv(outdir / "sine_powerlaw_beta_sweep_metrics.csv", "beta", "beta", beta_rows)

    plot_sine_powerlaw_metric(
        outdir,
        omega_rows,
        "omega_hz",
        "dominant_frequency_hz",
        selected,
        "sine_powerlaw_omega_sweep_dominant_frequency.png",
        f"Sine+fBM input: dominant frequency vs omega (beta={args.sine_powerlaw_base_beta:g})",
        "Input sine frequency omega (Hz)",
        "Dominant output frequency (Hz)",
    )
    plot_sine_powerlaw_metric(
        outdir,
        omega_rows,
        "omega_hz",
        "slope_10_100",
        selected,
        "sine_powerlaw_omega_sweep_slope_10_100Hz.png",
        f"Sine+fBM input: 10-100 Hz PSD slope vs omega (beta={args.sine_powerlaw_base_beta:g})",
        "Input sine frequency omega (Hz)",
        "Output PSD slope, 10-100 Hz",
    )
    plot_sine_powerlaw_metric(
        outdir,
        omega_rows,
        "omega_hz",
        "slope_100_1000",
        selected,
        "sine_powerlaw_omega_sweep_slope_100_1000Hz.png",
        f"Sine+fBM input: 100-1000 Hz PSD slope vs omega (beta={args.sine_powerlaw_base_beta:g})",
        "Input sine frequency omega (Hz)",
        "Output PSD slope, 100-1000 Hz",
    )
    plot_sine_powerlaw_metric(
        outdir,
        beta_rows,
        "beta",
        "dominant_frequency_hz",
        selected,
        "sine_powerlaw_beta_sweep_dominant_frequency.png",
        f"Sine+fBM input: dominant frequency vs beta (omega={args.sine_powerlaw_base_omega:g} Hz)",
        "Input noise PSD slope beta",
        "Dominant output frequency (Hz)",
    )
    plot_sine_powerlaw_metric(
        outdir,
        beta_rows,
        "beta",
        "slope_10_100",
        selected,
        "sine_powerlaw_beta_sweep_slope_10_100Hz.png",
        f"Sine+fBM input: 10-100 Hz PSD slope vs beta (omega={args.sine_powerlaw_base_omega:g} Hz)",
        "Input noise PSD slope beta",
        "Output PSD slope, 10-100 Hz",
    )
    plot_sine_powerlaw_metric(
        outdir,
        beta_rows,
        "beta",
        "slope_100_1000",
        selected,
        "sine_powerlaw_beta_sweep_slope_100_1000Hz.png",
        f"Sine+fBM input: 100-1000 Hz PSD slope vs beta (omega={args.sine_powerlaw_base_omega:g} Hz)",
        "Input noise PSD slope beta",
        "Output PSD slope, 100-1000 Hz",
    )


def generate_three_resting_psd_conditions(outdir: Path, p: dict[str, Any], args: argparse.Namespace) -> None:
    """Generate the three requested resting-state PSD slope figures."""
    conditions = [
        {
            "stem": "psd_slopes_01_resting_all_area_weak_white",
            "title": "Resting state: all areas weak white noise (sigma=1e-5)",
            "kwargs": {"total_input": True, "stim_noise": "white", "stim_noise_sigma": 0.5},
        },
        {
            "stem": "psd_slopes_02_v1_white_sigma_0p5",
            "title": "Resting state: V1 white noise input (sigma=0.5)",
            "kwargs": {"total_input": False, "stim_noise": "white", "stim_noise_sigma": 0.5},
        },
        {
            "stem": "psd_slopes_03_v1_fbm_slope_0p5_sigma_0p5",
            "title": "Resting state: V1 power-law/fBM-like input (PSD slope=0.5, sigma=0.5)",
            "kwargs": {
                "total_input": False,
                "stim_noise": "fractional_brownian",
                "stim_noise_slope": 0.5,
                "stim_noise_sigma": 0.5,
            },
        },
    ]
    for condition in conditions:
        sim = simulate_stimulus(
            p,
            stim_area="V1",
            T=args.rest_T,
            pulse_input=False,
            seed=args.seed,
            **condition["kwargs"],
        )
        plot_psd_slope_grid(outdir, p, sim, condition["stem"], condition["title"])


def run_pipeline(args: argparse.Namespace) -> None:
    base = Path(args.base_dir).resolve()
    outdir = Path(args.outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    p_raw = load_data(base / args.datafile)
    p = apply_standard_parameters(
        p_raw,
        hier_fit_file=base / args.hier_fit_file,
        fit_hier=not args.no_fit_hier,
        diverse_hi=args.diverse_hi,
    )
    W, W_EI = build_connectivity_matrix(p)
    eig = eigen_statistics(W_EI, p["n_area"])
    corr_E, cov_E = analytical_functional_connectivity(W_EI, p["n_area"])
    pulse = simulate_stimulus(p, stim_area=args.stim_area, T=args.T, seed=args.seed)
    resting = simulate_stimulus(
        p,
        stim_area=args.stim_area,
        T=args.rest_T,
        pulse_input=False,
        total_input=args.rest_total_input,
        seed=args.seed,
        stim_noise=args.rest_stim_noise,
        stim_noise_slope=args.rest_stim_noise_slope,
        stim_noise_sigma=args.rest_stim_noise_sigma,
    )

    np.savez_compressed(
        outdir / "standard_outputs.npz",
        W=W,
        W_EI=W_EI,
        eigvals=eig["eigvals"],
        tau=eig["tau"],
        ipr=eig["ipr"],
        corr_E=corr_E,
        cov_E=cov_E,
        pulse_t=pulse["t_plot"],
        pulse_r_exc=pulse["r_exc"],
        pulse_r_inh=pulse["r_inh"],
        pulse_max_rate=pulse["max_rate"],
        pulse_decay_time=pulse["decay_time"],
        resting_t=resting["t_plot"],
        resting_r_exc=resting["r_exc"],
        areas=np.array(p["areas"]),
        hier_vals=p["hier_vals"],
    )
    write_area_table(outdir / "standard_area_stats.csv", p, pulse["max_rate"], pulse["decay_time"], eig["ipr"])

    summary = {
        "n_area": p["n_area"],
        "areas_first_5": p["areas"][:5],
        "standard_condition": {
            "MACAQUE_CASE": 0,
            "FIT_HIER": not args.no_fit_hier,
            "diverse_hi": args.diverse_hi,
            "stim_area": args.stim_area,
            "pulse_T_ms": args.T,
            "rest_T_ms": args.rest_T,
            "rest_total_input": args.rest_total_input,
            "rest_stim_noise": args.rest_stim_noise,
            "rest_stim_noise_slope": args.rest_stim_noise_slope,
            "rest_stim_noise_sigma": args.rest_stim_noise_sigma,
        },
        "parameters": {
            "beta_exc": p["beta_exc"],
            "beta_inh": p["beta_inh"],
            "tau_exc": p["tau_exc"],
            "tau_inh": p["tau_inh"],
            "wEE": p["wEE"],
            "wEI": p["wEI"],
            "wIE": p["wIE"],
            "wII": p["wII"],
            "muEE": p["muEE"],
            "muIE": p["muIE"],
            "eta": p["eta"],
            "eta_local": p["eta_local"],
            "eta_inh": p["eta_inh"],
            "eta_inh_local": p["eta_inh_local"],
        },
        "stability": {
            "max_real_eig": eig["max_real_eig"],
            "positive_real_eig_count": eig["positive_real_eig_count"],
            "note": "Analytical FC is meaningful only when all eigenvalue real parts are negative.",
        },
        "tau_top_10_ms": [float(x) for x in eig["tau"][:10]],
        "fc_summary": {
            "shape": list(corr_E.shape),
            "min": float(np.nanmin(corr_E)),
            "max": float(np.nanmax(corr_E)),
            "mean": float(np.nanmean(corr_E)),
        },
        "pulse_summary": {
            "max_response_area": p["areas"][int(np.nanargmax(pulse["max_rate"]))],
            "max_response_hz_above_baseline": float(np.nanmax(pulse["max_rate"])),
            "mean_max_response_hz_above_baseline": float(np.nanmean(pulse["max_rate"])),
            "mean_decay_time_ms": float(np.nanmean(pulse["decay_time"])),
        },
    }
    (outdir / "standard_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.figures:
        maybe_make_figures(outdir, p, eig["tau"], corr_E, pulse, resting)
    if args.three_psd_conditions:
        generate_three_resting_psd_conditions(outdir, p, args)
    if args.beta_sweep_psd:
        generate_beta_sweep_psd_slopes(outdir, p, args)
    if args.square_wave_psd:
        generate_square_wave_psd(outdir, p, args)
    if args.sine_white_psd:
        generate_sine_white_psd(outdir, p, args)
    if args.sine_powerlaw_sweeps:
        generate_sine_powerlaw_sweeps(outdir, p, args)
    if args.random_sine_powerlaw_trials:
        generate_random_sine_powerlaw_trials(outdir, p, args)
    if args.random_sine_powerlaw_input_area_sweep:
        generate_random_sine_powerlaw_input_area_sweep(outdir, p, args)

    print(json.dumps(summary, indent=2))
    print(f"\nSaved outputs to: {outdir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", default=".", help="Directory containing data files.")
    parser.add_argument("--datafile", default="subgraph_data_marmoset.pkl")
    parser.add_argument("--hier-fit-file", default="hier_vals_fit.npy")
    parser.add_argument("--outdir", default="standard_run")
    parser.add_argument("--stim-area", default="V1")
    parser.add_argument("--T", type=float, default=6000.0, help="Pulse simulation duration in ms.")
    parser.add_argument("--rest-T", type=float, default=2000.0, help="Resting/noise simulation duration in ms.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--diverse-hi", type=int, default=41)
    parser.add_argument("--no-fit-hier", action="store_true")
    parser.add_argument("--figures", action="store_true", help="Save PNG reproduction figures.")
    parser.add_argument("--rest-total-input", action="store_true", help="Drive every area with weak white noise.")
    parser.add_argument(
        "--rest-stim-noise",
        default="white",
        choices=["white", "powerlaw", "fbm", "fractional_brownian", "square", "sine_plus_white", "sine_plus_powerlaw"],
        help="Noise type for the stimulated area when --rest-total-input is not set.",
    )
    parser.add_argument("--rest-stim-noise-slope", type=float, default=0.5)
    parser.add_argument("--rest-stim-noise-sigma", type=float, default=0.5)
    parser.add_argument(
        "--three-psd-conditions",
        action="store_true",
        help="Generate the three requested resting PSD slope comparison figures.",
    )
    parser.add_argument(
        "--beta-sweep-psd",
        action="store_true",
        help="Sweep V1 power-law/fBM-like input PSD slope beta and plot output PSD slopes.",
    )
    parser.add_argument("--beta-min", type=float, default=-0.8)
    parser.add_argument("--beta-max", type=float, default=0.8)
    parser.add_argument("--beta-step", type=float, default=0.1)
    parser.add_argument("--beta-sweep-sigma", type=float, default=0.5)
    parser.add_argument(
        "--square-wave-psd",
        action="store_true",
        help="Generate PSD plots for V1 square-wave input.",
    )
    parser.add_argument("--square-frequency", type=float, default=1.0)
    parser.add_argument("--square-amplitude", type=float, default=0.5)
    parser.add_argument("--square-T", type=float, default=10000.0, help="Square-wave simulation duration in ms.")
    parser.add_argument("--square-welch-seconds", type=float, default=4.0)
    parser.add_argument("--square-peak-min-hz", type=float, default=0.5)
    parser.add_argument("--square-peak-max-hz", type=float, default=50.0)
    parser.add_argument(
        "--sine-white-psd",
        action="store_true",
        help="Generate PSD plots for V1 sine-wave plus white-noise input.",
    )
    parser.add_argument("--sine-frequency", type=float, default=1.0)
    parser.add_argument("--sine-amplitude", type=float, default=0.5)
    parser.add_argument("--sine-white-sigma", type=float, default=0.5)
    parser.add_argument("--sine-white-T", type=float, default=10000.0, help="Sine+white simulation duration in ms.")
    parser.add_argument("--sine-white-welch-seconds", type=float, default=4.0)
    parser.add_argument("--sine-white-peak-min-hz", type=float, default=0.5)
    parser.add_argument("--sine-white-peak-max-hz", type=float, default=50.0)
    parser.add_argument(
        "--sine-powerlaw-sweeps",
        action="store_true",
        help="Sweep sine frequency omega and power-law/fBM-like noise beta for V1 input.",
    )
    parser.add_argument("--sine-powerlaw-amplitude", type=float, default=0.5)
    parser.add_argument("--sine-powerlaw-sigma", type=float, default=0.5)
    parser.add_argument("--sine-powerlaw-base-omega", type=float, default=5.0)
    parser.add_argument("--sine-powerlaw-base-beta", type=float, default=0.0)
    parser.add_argument("--sine-powerlaw-omega-min", type=float, default=1.0)
    parser.add_argument("--sine-powerlaw-omega-max", type=float, default=10.0)
    parser.add_argument("--sine-powerlaw-beta-min", type=float, default=-0.8)
    parser.add_argument("--sine-powerlaw-beta-max", type=float, default=0.8)
    parser.add_argument("--sine-powerlaw-beta-step", type=float, default=0.1)
    parser.add_argument("--sine-powerlaw-T", type=float, default=10000.0)
    parser.add_argument("--sine-powerlaw-welch-seconds", type=float, default=4.0)
    parser.add_argument("--sine-powerlaw-peak-min-hz", type=float, default=0.5)
    parser.add_argument("--sine-powerlaw-peak-max-hz", type=float, default=50.0)
    parser.add_argument(
        "--random-sine-powerlaw-trials",
        action="store_true",
        help="Run random omega/beta sine+powerlaw trials and regress frequency against high-frequency PSD slope.",
    )
    parser.add_argument("--random-trials", type=int, default=100)
    parser.add_argument("--random-omega-mean", type=float, default=5.0)
    parser.add_argument("--random-omega-variance", type=float, default=2.0)
    parser.add_argument("--random-omega-min", type=float, default=1.0)
    parser.add_argument("--random-omega-max", type=float, default=10.0)
    parser.add_argument("--random-beta-mean", type=float, default=0.0)
    parser.add_argument("--random-beta-std", type=float, default=0.4)
    parser.add_argument("--random-beta-min", type=float, default=-0.8)
    parser.add_argument("--random-beta-max", type=float, default=0.8)
    parser.add_argument("--random-sine-amplitude", type=float, default=0.5)
    parser.add_argument("--random-powerlaw-sigma", type=float, default=0.5)
    parser.add_argument("--random-trial-T", type=float, default=10000.0)
    parser.add_argument("--random-welch-seconds", type=float, default=4.0)
    parser.add_argument("--random-peak-min-hz", type=float, default=0.5)
    parser.add_argument("--random-peak-max-hz", type=float, default=50.0)
    parser.add_argument(
        "--random-sine-powerlaw-input-area-sweep",
        action="store_true",
        help="Run random omega/beta regression for inputs to each of the nine selected areas.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_pipeline(parse_args())
