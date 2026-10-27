from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, signal

from .core import save_figure, save_json
from .exp2 import _ei_path, _local_ei_setup
from .io import load_config, prepare_output


OUTPUT_NAME = "exp2_kappa_homotopy"


def _path_setup():
    tuned, tau, hopf_gain, kappa_new = _local_ei_setup()
    original = tuned.copy()
    original[0, 1] /= kappa_new
    gain_new = hopf_gain + 0.02
    return original, tau, float(kappa_new), float(gain_new), float(hopf_gain)


def _matrix_at(s: float, original: np.ndarray, kappa_new: float, gain_new: float):
    w = original.copy()
    w[0, 1] *= 1.0 + s * (kappa_new - 1.0)
    w[0, 0] *= 1.0 + s * (gain_new - 1.0)
    return w


def _jacobian_at(s: float, original, tau, kappa_new, gain_new):
    w = _matrix_at(s, original, kappa_new, gain_new)
    return (w - np.eye(2)) / tau[:, None]


def _linear_cross_spectrum(j: np.ndarray, freq_hz: np.ndarray):
    eye = np.eye(2)
    see = np.empty(len(freq_hz))
    sii = np.empty(len(freq_hz))
    sei = np.empty(len(freq_hz), dtype=complex)
    for pos, freq in enumerate(freq_hz):
        r = np.linalg.inv(1j * 2 * np.pi * freq * eye - j)
        spectrum = r @ r.conj().T
        see[pos] = spectrum[0, 0].real
        sii[pos] = spectrum[1, 1].real
        sei[pos] = spectrum[0, 1]
    coherency = sei / np.sqrt(np.maximum(see * sii, 1e-300))
    return see, sii, sei, coherency


def _spectrum_metrics(j: np.ndarray, freq_hz: np.ndarray):
    see, sii, sei, coherency = _linear_cross_spectrum(j, freq_hz)
    band = (freq_hz >= 1.0) & (freq_hz <= 60.0)
    idxs = np.flatnonzero(band)
    local = int(np.argmax(np.abs(np.imag(coherency[idxs]))))
    idx = int(idxs[local])
    peak = int(idxs[np.argmax(np.abs(sei[idxs]))])
    return {
        "linear_peak_imag_coherency": float(abs(np.imag(coherency[idx]))),
        "linear_imag_coherency_freq_hz": float(freq_hz[idx]),
        "linear_peak_cross_spectrum": float(abs(sei[peak])),
        "linear_cross_spectrum_peak_hz": float(freq_hz[peak]),
        "linear_E_peak_hz": float(freq_hz[idxs[np.argmax(see[idxs])]]),
    }


def _crossing_points(original, tau, kappa_new, gain_new):
    def discriminant(s):
        j = _jacobian_at(s, original, tau, kappa_new, gain_new)
        return float(np.trace(j) ** 2 - 4.0 * np.linalg.det(j))

    def max_real(s):
        return float(np.max(np.linalg.eigvals(_jacobian_at(s, original, tau, kappa_new, gain_new)).real))

    s_dense = np.linspace(0.0, 1.0, 2001)

    def first_root(fun):
        values = np.array([fun(s) for s in s_dense])
        for left, right, yl, yr in zip(s_dense[:-1], s_dense[1:], values[:-1], values[1:]):
            if yl == 0 or yl * yr < 0:
                return float(optimize.brentq(fun, left, right))
        return np.nan

    return first_root(discriminant), first_root(max_real)


def _simulated_cross_spectrum(w, tau, dt, duration, burn, noise, seeds, s_value, base_seed):
    fs = 1.0 / dt
    n = int(round(duration / dt))
    start = int(round(burn / dt))
    spectra = []
    for seed in seeds:
        x = _ei_path(w, tau, noise, dt, n, base_seed + seed + int(round(s_value * 1e5)))
        xx = x[start:]
        nperseg = min(16384, len(xx))
        freq, see = signal.welch(xx[:, 0], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        _, sii = signal.welch(xx[:, 1], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        _, sei = signal.csd(xx[:, 0], xx[:, 1], fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
        spectra.append((see, sii, sei))
    see = np.mean([item[0] for item in spectra], axis=0)
    sii = np.mean([item[1] for item in spectra], axis=0)
    sei = np.mean([item[2] for item in spectra], axis=0)
    coherency = sei / np.sqrt(np.maximum(see * sii, 1e-300))
    band = (freq >= 1.0) & (freq <= 60.0)
    idxs = np.flatnonzero(band)
    idx = int(idxs[np.argmax(abs(np.imag(coherency[idxs])))])
    peak = int(idxs[np.argmax(abs(sei[idxs]))])
    row = {
        "s": s_value,
        "sim_peak_imag_coherency": float(abs(np.imag(coherency[idx]))),
        "sim_imag_coherency_freq_hz": float(freq[idx]),
        "sim_peak_cross_spectrum": float(abs(sei[peak])),
        "sim_cross_spectrum_peak_hz": float(freq[peak]),
    }
    keep = (freq >= 0.5) & (freq <= 80.0)
    return row, (freq[keep], see[keep], sei[keep], coherency[keep])


def run(cfg: dict) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    original, tau, kappa_new, gain_new, hopf_gain_at_tuned_kappa = _path_setup()
    s_complex, s_hopf = _crossing_points(original, tau, kappa_new, gain_new)
    freq = np.geomspace(0.1, 100.0, 600)
    rows = []
    for s_value in np.linspace(0.0, 1.0, 501):
        w = _matrix_at(s_value, original, kappa_new, gain_new)
        j = (w - np.eye(2)) / tau[:, None]
        eigenvalues = np.linalg.eigvals(j)
        critical = eigenvalues[np.argmax(eigenvalues.real)]
        rows.append(
            {
                "s": s_value,
                "kappa_I_multiplier": 1.0 + s_value * (kappa_new - 1.0),
                "E_E_gain": 1.0 + s_value * (gain_new - 1.0),
                "critical_real_per_s": float(critical.real),
                "critical_imag_per_s": float(abs(critical.imag)),
                "critical_frequency_hz": float(abs(critical.imag) / (2 * np.pi)),
                "discriminant": float(np.trace(j) ** 2 - 4 * np.linalg.det(j)),
                "stable": bool(np.max(eigenvalues.real) < 0),
                "complex_pair": bool(np.max(abs(eigenvalues.imag)) > 1e-8),
                **_spectrum_metrics(j, freq),
            }
        )
    eigen_scan = pd.DataFrame(rows)
    eigen_scan.to_csv(out / "homotopy_eigen_cross_spectrum.csv", index=False)

    ecfg = cfg["exp2"]
    sim_s = sorted(set([0.0, 0.35, s_complex + 0.03, 0.70, 0.85, 0.95, s_hopf - 0.01, 1.0]))
    sim_rows = []
    examples = {}
    for s_value in sim_s:
        w = _matrix_at(s_value, original, kappa_new, gain_new)
        row, spectra = _simulated_cross_spectrum(
            w,
            tau,
            float(ecfg["dt_s"]),
            float(ecfg["duration_s"]),
            float(ecfg["burn_s"]),
            0.002,
            range(8),
            s_value,
            int(cfg["seed"]) + 950000,
        )
        sim_rows.append(row)
        examples[f"s_{s_value:.5f}"] = spectra
        print(f"Experiment 2 homotopy simulation s={s_value:.5f}", flush=True)
    simulations = pd.DataFrame(sim_rows)
    simulations.to_csv(out / "nonlinear_cross_spectrum_validation.csv", index=False)
    np.savez(
        out / "cross_spectrum_examples.npz",
        **{
            f"{key}_{name}": value[pos]
            for key, value in examples.items()
            for pos, name in enumerate(("freq_hz", "E_psd", "cross_spectrum", "coherency"))
        },
    )
    _plot(out, eigen_scan, simulations, examples, s_complex, s_hopf)
    save_json(
        out / "manifest.json",
        {
            "experiment": "continuous_PG_local_EI_homotopy",
            "homotopy": "kappa_I(s)=1+s(kappa_new-1), g_EE(s)=1+s(g_new-1)",
            "kappa_I_original": 1.0,
            "kappa_I_new": kappa_new,
            "E_E_gain_original": 1.0,
            "E_E_gain_new": gain_new,
            "hopf_gain_at_tuned_kappa": hopf_gain_at_tuned_kappa,
            "complex_eigenvalue_onset_s": s_complex,
            "hopf_crossing_s": s_hopf,
            "nonlinear_noise": 0.002,
            "nonlinear_seeds": 8,
            "duration_s": ecfg["duration_s"],
            "burn_s": ecfg["burn_s"],
            "dt_s": ecfg["dt_s"],
            "runtime_s": time.time() - started,
        },
    )
    return out


def _plot(out: Path, scan: pd.DataFrame, simulations: pd.DataFrame, examples, s_complex, s_hopf):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.3), constrained_layout=True)
    axes[0, 0].plot(scan.s, scan.critical_real_per_s, color="#247BA0", lw=2)
    axes[0, 0].axhline(0, color="0.2", ls="--", lw=1)
    axes[0, 0].set(xlabel=r"homotopy $s$", ylabel=r"critical $\mathrm{Re}(\lambda)$ (s$^{-1}$)", title="Stability boundary")
    axes[0, 1].plot(scan.s, scan.critical_frequency_hz, color="#D1495B", lw=2)
    axes[0, 1].set(xlabel=r"homotopy $s$", ylabel=r"$|\mathrm{Im}(\lambda)|/(2\pi)$ (Hz)", title="Complex-mode frequency")
    axes[1, 0].plot(scan.s, scan.linear_peak_imag_coherency, color="#6A4C93", lw=2, label="linear theory")
    axes[1, 0].scatter(
        simulations.s, simulations.sim_peak_imag_coherency, color="#F28E2B", s=35,
        edgecolor="white", lw=0.6, label="nonlinear simulation",
    )
    axes[1, 0].set(
        xlabel="homotopy s", ylabel="peak |Im coherency E-I|",
        title="Lagged E-I cross-spectrum coupling",
    )
    axes[1, 0].legend(frameon=False)
    axes[1, 1].plot(scan.s, scan.linear_cross_spectrum_peak_hz, color="#2A9D8F", lw=2, label="linear theory")
    axes[1, 1].scatter(
        simulations.s, simulations.sim_cross_spectrum_peak_hz, color="#F28E2B", s=35,
        edgecolor="white", lw=0.6, label="nonlinear simulation",
    )
    axes[1, 1].set(
        xlabel="homotopy s", ylabel="cross-spectrum peak [Hz]",
        title="Cross-spectrum characteristic frequency",
    )
    for ax in axes.flat:
        ax.axvline(s_complex, color="#7A5195", ls=":", lw=1.5, label="complex onset")
        ax.axvline(s_hopf, color="#EF5675", ls="--", lw=1.5, label="Hopf")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    axes[0, 0].legend(handles, labels, frameon=False, fontsize=8)
    fig.suptitle("Experiment 2: continuous path from original PG block to tuned near-Hopf block", fontsize=13)
    save_figure(fig, out / "figures" / "exp2_homotopy_eigen_cross_spectrum")
    plt.close(fig)

    chosen = [min(examples, key=lambda key: abs(float(key.split("_")[1]) - target)) for target in (0.35, s_complex + 0.03, 0.95, 1.0)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.8), constrained_layout=True)
    for ax, key in zip(axes.flat, chosen):
        freq, _, cross_spectrum, _ = examples[key]
        magnitude = abs(cross_spectrum)
        magnitude /= max(float(magnitude.max()), np.finfo(float).tiny)
        peak_hz = float(freq[np.argmax(magnitude)])
        ax.plot(freq, magnitude, color="#6A4C93", lw=1.8)
        ax.axvline(peak_hz, color="#F28E2B", ls="--", lw=1.2)
        s_value = float(key.split("_")[1])
        regime = "real stable" if s_value < s_complex else ("complex stable" if s_value < s_hopf else "post-Hopf")
        ax.set(
            xlim=(0, 60), ylim=(0, 1.05), xlabel="frequency [Hz]",
            ylabel=r"normalized $|S_{EI}(f)|$", title=rf"$s={s_value:.3f}$: {regime}; peak={peak_hz:.1f} Hz",
        )
    fig.suptitle("Nonlinear E-I cross-spectrum magnitude along the homotopy", fontsize=13)
    save_figure(fig, out / "figures" / "exp2_homotopy_cross_spectrum_examples")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
