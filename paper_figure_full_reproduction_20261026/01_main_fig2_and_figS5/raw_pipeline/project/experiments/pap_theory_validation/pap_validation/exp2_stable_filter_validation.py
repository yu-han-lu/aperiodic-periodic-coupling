from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, signal, stats

try:
    from numba import njit
except ImportError:  # pragma: no cover
    njit = lambda f: f

from .core import save_figure, save_json
from .exp2_homotopy import _simulated_cross_spectrum
from .exp2_two_stage import _critical, _discriminant, _jacobian, _matrix, _roots, _setup
from .io import load_config, prepare_output


OUTPUT_NAME = "exp2_stable_filter_validation"
DT_S = 0.00025
ANALYSIS_S = 32.768
BURN_S = 2.048
N_BETA = 81
N_REPEATS = 3


def _dense_scan(complex_onset: float, target: float) -> np.ndarray:
    coarse = np.linspace(1.0, target, 700)
    local = np.linspace(complex_onset - 0.08, complex_onset + 0.12, 401)
    return np.unique(np.r_[coarse, local[(local >= 1.0) & (local <= target)]])


def _constant_numerator_resonance(j: np.ndarray) -> float:
    eig = np.linalg.eigvals(j)
    lam = eig[np.argmax(eig.imag)]
    gamma = -float(lam.real)
    omega = abs(float(lam.imag))
    return float(np.sqrt(max(omega * omega - gamma * gamma, 0.0)) / (2.0 * np.pi))


def _e_transfer_power(j: np.ndarray, frequencies: np.ndarray, tau_e: float) -> np.ndarray:
    out = np.empty(len(frequencies))
    b = np.array([1.0 / tau_e, 0.0])
    for pos, frequency in enumerate(frequencies):
        response = np.linalg.solve(1j * 2.0 * np.pi * frequency * np.eye(2) - j, b)
        out[pos] = abs(response[0]) ** 2
    return np.maximum(out, np.finfo(float).tiny)


def _spectral_peak(frequencies: np.ndarray, power: np.ndarray, band=(5.0, 100.0)) -> float:
    use = (frequencies >= band[0]) & (frequencies <= band[1])
    f = frequencies[use]
    p = power[use]
    peaks, _ = signal.find_peaks(np.log(np.maximum(p, 1e-300)), prominence=0.01)
    if len(peaks):
        return float(f[peaks[np.argmax(p[peaks])]])
    return float(f[np.argmax(p)])


def _scan_rows(original, tau, values, frequencies):
    rows = []
    for value in values:
        j = _jacobian(original, tau, float(value), 1.0)
        eig = _critical(j)
        transfer = _e_transfer_power(j, frequencies, float(tau[0]))
        rows.append(
            {
                "I_to_E_multiplier": value,
                "critical_real_per_s": eig.real,
                "critical_imag_per_s": abs(eig.imag),
                "eigen_frequency_hz": abs(eig.imag) / (2.0 * np.pi),
                "eigen_frequency_squared_hz2": (abs(eig.imag) / (2.0 * np.pi)) ** 2,
                "decay_time_ms": -1000.0 / eig.real,
                "quality_ratio_omega_over_gamma": abs(eig.imag) / max(-eig.real, 1e-300),
                "discriminant": _discriminant(j),
                "constant_numerator_resonance_hz": _constant_numerator_resonance(j),
                "exact_E_transfer_peak_hz": _spectral_peak(frequencies, transfer),
            }
        )
    return pd.DataFrame(rows)


@njit
def _forced_ei_batch(w, tau, forcing, dt):
    n_trials, n_steps = forcing.shape
    output = np.zeros((n_trials, n_steps), dtype=np.float32)
    for trial in range(n_trials):
        e = 0.0
        i = 0.0
        for step in range(n_steps):
            u = forcing[trial, step]
            d_e = (-e + np.tanh(w[0, 0] * e + w[0, 1] * i + u)) / tau[0]
            d_i = (-i + np.tanh(w[1, 0] * e + w[1, 1] * i)) / tau[1]
            pe = e + dt * d_e
            pi = i + dt * d_i
            p_e = (-pe + np.tanh(w[0, 0] * pe + w[0, 1] * pi + u)) / tau[0]
            p_i = (-pi + np.tanh(w[1, 0] * pe + w[1, 1] * pi)) / tau[1]
            e += 0.5 * dt * (d_e + p_e)
            i += 0.5 * dt * (d_i + p_i)
            output[trial, step] = e
    return output


def _colored_forcing(betas, n_steps, sigma, floor_sigma, seed):
    rng = np.random.default_rng(seed)
    frequencies = np.fft.rfftfreq(n_steps, d=DT_S)
    amplitudes = np.maximum(frequencies, 1.0)[None, :] ** (betas[:, None] / 2.0)
    phases = rng.normal(size=amplitudes.shape) + 1j * rng.normal(size=amplitudes.shape)
    phases[:, 0] = 0.0
    colored = np.fft.irfft(amplitudes * phases, n=n_steps, axis=1)
    colored -= colored.mean(axis=1, keepdims=True)
    colored *= sigma / np.maximum(colored.std(axis=1, keepdims=True), 1e-12)
    colored += floor_sigma * rng.normal(size=colored.shape)
    return colored.astype(np.float32)


def _features_from_paths(paths, betas):
    fs = 1.0 / DT_S
    start = int(round(BURN_S / DT_S))
    f, p = signal.welch(
        paths[:, start:], fs=fs, nperseg=int(round(4.096 / DT_S)),
        noverlap=int(round(2.048 / DT_S)), axis=-1,
    )
    peak_band = (f >= 5.0) & (f <= 100.0)
    slope_band = (f >= 100.0) & (f <= 1000.0)
    rows = []
    for trial, beta in enumerate(betas):
        peak = float(f[peak_band][np.argmax(p[trial, peak_band])])
        slope = float(stats.linregress(np.log(f[slope_band]), np.log(np.maximum(p[trial, slope_band], 1e-300))).slope)
        rows.append({"trial": trial, "beta": beta, "peak_hz": peak, "slope": slope})
    return pd.DataFrame(rows)


def _theory_features(j, tau_e, betas, sigma, floor_sigma):
    f = np.geomspace(0.5, 1200.0, 1200)
    transfer = _e_transfer_power(j, f, tau_e)
    rows = []
    for trial, beta in enumerate(betas):
        q = sigma * sigma * np.maximum(f, 1.0) ** beta + floor_sigma * floor_sigma
        psd = transfer * q
        peak = _spectral_peak(f, psd)
        use = (f >= 100.0) & (f <= 1000.0)
        slope = stats.linregress(np.log(f[use]), np.log(psd[use])).slope
        rows.append({"trial": trial, "beta": beta, "peak_hz": peak, "slope": slope})
    return pd.DataFrame(rows)


def _coupling_row(kind, s_value, repeat, features):
    beta_fit = stats.linregress(features.beta, features.peak_hz)
    slope_fit = stats.linregress(features.beta, features.slope)
    correlation = stats.pearsonr(features.peak_hz, features.slope)
    return {
        "layer": kind,
        "I_to_E_multiplier": s_value,
        "repeat": repeat,
        "m_beta_hz_per_beta": beta_fit.slope,
        "kappa_beta_per_beta": slope_fit.slope,
        "peak_beta_R2": beta_fit.rvalue ** 2,
        "slope_beta_R2": slope_fit.rvalue ** 2,
        "P_A_r": correlation.statistic,
        "P_A_p": correlation.pvalue,
    }


def _input_colour_experiment(original, tau, complex_onset, target, seed):
    scan_values = np.unique(np.r_[np.linspace(1.0, target, 13), complex_onset, 3.18])
    beta_rng = np.random.default_rng(seed + 401)
    betas = np.sort(beta_rng.uniform(-0.8, 0.8, N_BETA))
    n_steps = int(round((ANALYSIS_S + BURN_S) / DT_S))
    sigma = 0.02
    floor_sigma = 0.002
    rows = []
    examples = {}
    for repeat in range(N_REPEATS):
        forcing = _colored_forcing(betas, n_steps, sigma, floor_sigma, seed + 10000 * repeat)
        for s_value in scan_values:
            w = _matrix(original, float(s_value), 1.0)
            j = _jacobian(original, tau, float(s_value), 1.0)
            theory = _theory_features(j, float(tau[0]), betas, sigma, floor_sigma)
            simulated = _features_from_paths(_forced_ei_batch(w, tau, forcing, DT_S), betas)
            rows.append(_coupling_row("exact_resolvent", s_value, repeat, theory))
            rows.append(_coupling_row("nonlinear_simulation", s_value, repeat, simulated))
            if repeat == 0 and (abs(s_value - 3.18) < 1e-6 or abs(s_value - target) < 1e-6):
                examples[f"theory_{s_value:.6f}"] = theory
                examples[f"simulation_{s_value:.6f}"] = simulated
        print(f"stable-filter input-colour repeat {repeat + 1}/{N_REPEATS}", flush=True)
    return pd.DataFrame(rows), examples, sigma, floor_sigma


def _decay_validation(original, tau, values):
    rows = []
    times = np.arange(0.0, 0.04, DT_S)
    for value in values:
        j = _jacobian(original, tau, float(value), 1.0)
        eigvals, right = np.linalg.eig(j)
        pos = int(np.argmax(eigvals.imag))
        left = np.linalg.inv(right)[pos]
        state = right[:, pos]
        envelope = np.abs(np.exp(eigvals[pos] * times) * (left @ state))
        fit = stats.linregress(times, np.log(np.maximum(envelope, 1e-300)))
        rows.append(
            {
                "I_to_E_multiplier": value,
                "fitted_decay_rate_per_s": fit.slope,
                "eigen_real_per_s": eigvals[pos].real,
                "fitted_decay_time_ms": -1000.0 / fit.slope,
                "fit_R2": fit.rvalue ** 2,
            }
        )
    return pd.DataFrame(rows)


def _plot(out, scan, nonlinear, coupling, examples, complex_onset, resonance_onset, target):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4), constrained_layout=True)
    complex_part = scan[scan.I_to_E_multiplier >= complex_onset]
    fit = stats.linregress(
        complex_part.I_to_E_multiplier - complex_onset,
        complex_part.eigen_frequency_squared_hz2,
    )
    axes[0, 0].plot(scan.I_to_E_multiplier, scan.eigen_frequency_hz, color="#247BA0", lw=2)
    axes[0, 0].axvline(complex_onset, color="#7A5195", ls="--", label="complex onset")
    axes[0, 0].set(xlabel=r"PG I-to-E multiplier $s_{EI}$", ylabel=r"$\mathrm{Im}(\lambda)/(2\pi)$ (Hz)", title="Stable node-to-focus transition")
    axes[0, 0].legend(frameon=False)
    axes[0, 1].scatter(complex_part.I_to_E_multiplier - complex_onset, complex_part.eigen_frequency_squared_hz2, s=8, color="#D1495B")
    axes[0, 1].plot(complex_part.I_to_E_multiplier - complex_onset, fit.intercept + fit.slope * (complex_part.I_to_E_multiplier - complex_onset), color="black", lw=1.5)
    axes[0, 1].set(xlabel="S - Sc", ylabel="f_lambda^2 [Hz^2]", title=f"slope={fit.slope:.2f} Hz^2/S, R2={fit.rvalue**2:.6f}")
    axes[1, 0].plot(scan.I_to_E_multiplier, scan.critical_real_per_s, color="#2A9D8F", lw=2)
    axes[1, 0].axhline(0, color="0.2", ls="--", lw=1)
    axes[1, 0].set(xlabel=r"PG I-to-E multiplier $s_{EI}$", ylabel=r"$\mathrm{Re}(\lambda)$ (s$^{-1}$)", title="No critical slowing on this path")
    axes[1, 1].plot(scan.I_to_E_multiplier, scan.constant_numerator_resonance_hz, color="#6A4C93", lw=2, label="constant-numerator formula")
    axes[1, 1].plot(scan.I_to_E_multiplier, scan.exact_E_transfer_peak_hz, color="#F28E2B", lw=1.5, label="exact resolvent E peak")
    axes[1, 1].scatter(nonlinear.I_to_E_multiplier, nonlinear.sim_cross_spectrum_peak_hz, color="#D1495B", s=32, label="nonlinear cross-spectrum")
    axes[1, 1].axvline(resonance_onset, color="0.25", ls=":", label="resonance onset")
    axes[1, 1].set(xlabel="PG I-to-E multiplier S", ylabel="frequency [Hz]", title="Complex onset is not resonance onset")
    axes[1, 1].legend(frameon=False, fontsize=8)
    save_figure(fig, out / "figures" / "stable_node_focus_validation")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for layer, style in (("exact_resolvent", "-"), ("nonlinear_simulation", "--")):
        data = coupling[coupling.layer == layer]
        group = data.groupby("I_to_E_multiplier")
        x = np.array(sorted(group.groups))
        label = layer.replace("_", " ")
        for ax, column, ylabel in zip(axes, ("m_beta_hz_per_beta", "kappa_beta_per_beta", "P_A_r"), (r"$m_\beta$ (Hz/$\beta$)", r"$\kappa_\beta$", r"$r(P,A)$")):
            mean = group[column].mean().reindex(x)
            ax.plot(x, mean, style, lw=2, label=label)
            ax.axvline(complex_onset, color="0.5", ls=":")
            ax.set(xlabel="PG I-to-E multiplier S", ylabel=ylabel)
    axes[0].set_title("Peak sensitivity")
    axes[1].set_title("Aperiodic-slope sensitivity")
    axes[2].set_title("Input-colour P-A coupling")
    axes[2].axhline(0, color="0.2", lw=1)
    axes[2].legend(frameon=False, fontsize=8)
    save_figure(fig, out / "figures" / "stable_filter_input_colour_coupling")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    for ax, (key, frame) in zip(axes.flat, examples.items()):
        fit = stats.linregress(frame.peak_hz, frame.slope)
        ax.scatter(frame.peak_hz, frame.slope, c=frame.beta, cmap="coolwarm", s=18)
        order = np.argsort(frame.peak_hz)
        ax.plot(frame.peak_hz.iloc[order], fit.intercept + fit.slope * frame.peak_hz.iloc[order], color="black", lw=1.2)
        r, p = stats.pearsonr(frame.peak_hz, frame.slope)
        ax.set(xlabel="peak P [Hz]", ylabel="100-1000 Hz signed slope A", title=f"{key}; r={r:.3f}, p={p:.2g}")
    save_figure(fig, out / "figures" / "stable_filter_input_colour_examples")
    plt.close(fig)


def run(cfg: dict, force: bool = False) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    marker = out / "manifest.json"
    if marker.exists() and not force:
        return out
    original, tau, target, hopf_gain = _setup()
    complex_onset, _ = _roots(original, tau, target, hopf_gain)
    values = _dense_scan(complex_onset, target)
    frequencies = np.geomspace(0.2, 1200.0, 1500)
    scan = _scan_rows(original, tau, values, frequencies)
    scan.to_csv(out / "stable_node_focus_scan.csv", index=False)
    resonance_onset = optimize.brentq(
        lambda value: abs(_critical(_jacobian(original, tau, value, 1.0)).imag) + _critical(_jacobian(original, tau, value, 1.0)).real,
        complex_onset + 1e-5,
        target,
    )
    selected = sorted(set([1.5, complex_onset, 2.4, resonance_onset, 5.0, target]))
    simulations = []
    for position, value in enumerate(selected):
        row, _ = _simulated_cross_spectrum(
            _matrix(original, value, 1.0), tau, DT_S, 80.0, 10.0, 0.002,
            range(8), value, int(cfg["seed"]) + 2410000 + position * 100,
        )
        row["I_to_E_multiplier"] = value
        simulations.append(row)
        print(f"stable node-focus nonlinear simulation S={value:.6f}", flush=True)
    nonlinear = pd.DataFrame(simulations)
    nonlinear.to_csv(out / "nonlinear_cross_spectrum_selected.csv", index=False)
    decay = _decay_validation(original, tau, [2.4, resonance_onset, 5.0, target])
    decay.to_csv(out / "impulse_decay_validation.csv", index=False)
    coupling, examples, input_sigma, floor_sigma = _input_colour_experiment(
        original, tau, complex_onset, target, int(cfg["seed"]) + 2420000
    )
    coupling.to_csv(out / "input_colour_coupling.csv", index=False)
    _plot(out, scan, nonlinear, coupling, examples, complex_onset, resonance_onset, target)
    complex_part = scan[scan.I_to_E_multiplier >= complex_onset]
    square_fit = stats.linregress(
        complex_part.I_to_E_multiplier - complex_onset,
        complex_part.eigen_frequency_squared_hz2,
    )
    save_json(
        marker,
        {
            "experiment": "stable_node_focus_and_stable_filter_input_colour_validation",
            "complex_onset_S": complex_onset,
            "constant_numerator_resonance_onset_S": resonance_onset,
            "target_S": target,
            "frequency_squared_slope_hz2_per_S": square_fit.slope,
            "frequency_squared_intercept_hz2": square_fit.intercept,
            "frequency_squared_R2": square_fit.rvalue ** 2,
            "critical_real_per_s": float(_critical(_jacobian(original, tau, target, 1.0)).real),
            "decay_time_ms": float(-1000.0 / _critical(_jacobian(original, tau, target, 1.0)).real),
            "input_colour": {
                "beta_range": [-0.8, 0.8], "n_beta_trials": N_BETA,
                "repeats": N_REPEATS, "analysis_s": ANALYSIS_S, "burn_s": BURN_S,
                "welch_segment_s": 4.096, "input_sigma": input_sigma,
                "white_floor_sigma": floor_sigma, "peak_band_hz": [5.0, 100.0],
                "slope_band_hz": [100.0, 1000.0],
            },
            "runtime_s": time.time() - started,
        },
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(load_config(args.config), force=args.force))


if __name__ == "__main__":
    main()
