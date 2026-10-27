from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize

from .core import save_figure, save_json
from .exp2 import _local_ei_setup
from .exp2_homotopy import _linear_cross_spectrum, _simulated_cross_spectrum
from .io import load_config, prepare_output


OUTPUT_NAME = "exp2_two_stage_scan"


def _setup():
    tuned, tau, hopf_gain, kappa_target = _local_ei_setup()
    original = tuned.copy()
    original[0, 1] /= kappa_target
    return original, tau, float(kappa_target), float(hopf_gain)


def _matrix(original: np.ndarray, kappa_i: float, gain_ee: float) -> np.ndarray:
    w = original.copy()
    w[0, 1] *= kappa_i
    w[0, 0] *= gain_ee
    return w


def _jacobian(original, tau, kappa_i, gain_ee):
    return (_matrix(original, kappa_i, gain_ee) - np.eye(2)) / tau[:, None]


def _discriminant(j: np.ndarray) -> float:
    return float(np.trace(j) ** 2 - 4.0 * np.linalg.det(j))


def _critical(j: np.ndarray) -> complex:
    eigenvalues = np.linalg.eigvals(j)
    return complex(eigenvalues[np.argmax(eigenvalues.real)])


def _cross_metrics(j: np.ndarray, frequencies: np.ndarray) -> dict[str, float]:
    _, _, cross, _ = _linear_cross_spectrum(j, frequencies)
    band = (frequencies >= 0.5) & (frequencies <= 80.0)
    f = frequencies[band]
    magnitude = abs(cross[band])
    peak = int(np.argmax(magnitude))
    reference = int(np.argmin(abs(f - 1.0)))
    return {
        "cross_spectrum_peak_hz": float(f[peak]),
        "cross_spectrum_peak_to_1hz": float(magnitude[peak] / max(magnitude[reference], 1e-300)),
    }


def _stage_rows(stage, values, original, tau, kappa_target, frequencies):
    rows = []
    for value in values:
        kappa_i = float(value if stage == "I_to_E" else kappa_target)
        gain_ee = float(1.0 if stage == "I_to_E" else value)
        j = _jacobian(original, tau, kappa_i, gain_ee)
        critical = _critical(j)
        rows.append(
            {
                "stage": stage,
                "scan_value": float(value),
                "kappa_I_multiplier": kappa_i,
                "E_E_gain": gain_ee,
                "critical_real_per_s": critical.real,
                "critical_imag_per_s": abs(critical.imag),
                "critical_frequency_hz": abs(critical.imag) / (2 * np.pi),
                "discriminant": _discriminant(j),
                "stable": bool(critical.real < 0),
                "complex_pair": bool(abs(critical.imag) > 1e-8),
                **_cross_metrics(j, frequencies),
            }
        )
    return rows


def _roots(original, tau, kappa_target, hopf_gain):
    complex_kappa = optimize.brentq(
        lambda kappa: _discriminant(_jacobian(original, tau, kappa, 1.0)),
        1.0,
        kappa_target,
    )
    hopf_ee = optimize.brentq(
        lambda gain: _critical(_jacobian(original, tau, kappa_target, gain)).real,
        1.0,
        hopf_gain + 0.2,
    )
    return float(complex_kappa), float(hopf_ee)


def _selected_conditions(complex_kappa, kappa_target, hopf_gain):
    return [
        ("I_to_E", 1.0),
        ("I_to_E", max(1.0, complex_kappa - 0.15)),
        ("I_to_E", min(kappa_target, complex_kappa + 0.20)),
        ("I_to_E", kappa_target),
        ("E_to_E", 1.0),
        ("E_to_E", 0.5 * (1.0 + hopf_gain)),
        ("E_to_E", hopf_gain - 0.05),
        ("E_to_E", hopf_gain + 0.02),
    ]


def run(cfg: dict) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    original, tau, kappa_target, hopf_gain = _setup()
    complex_kappa, hopf_ee = _roots(original, tau, kappa_target, hopf_gain)
    frequencies = np.geomspace(0.1, 100.0, 600)
    rows = _stage_rows(
        "I_to_E", np.linspace(1.0, kappa_target, 501), original, tau, kappa_target, frequencies
    )
    rows += _stage_rows(
        "E_to_E", np.linspace(1.0, hopf_gain + 0.08, 501), original, tau, kappa_target, frequencies
    )
    scan = pd.DataFrame(rows)
    scan.to_csv(out / "two_stage_eigen_cross_spectrum.csv", index=False)

    ecfg = cfg["exp2"]
    simulations = []
    examples = {}
    for position, (stage, value) in enumerate(_selected_conditions(complex_kappa, kappa_target, hopf_gain)):
        kappa_i = value if stage == "I_to_E" else kappa_target
        gain_ee = 1.0 if stage == "I_to_E" else value
        w = _matrix(original, kappa_i, gain_ee)
        row, spectrum = _simulated_cross_spectrum(
            w,
            tau,
            float(ecfg["dt_s"]),
            float(ecfg["duration_s"]),
            float(ecfg["burn_s"]),
            0.002,
            range(8),
            float(position + 1),
            int(cfg["seed"]) + 1250000,
        )
        freq, _, cross, _ = spectrum
        band = (freq >= 0.5) & (freq <= 80.0)
        row["sim_cross_spectrum_peak_hz"] = float(freq[band][np.argmax(abs(cross[band]))])
        row.update({"stage": stage, "scan_value": value, "kappa_I_multiplier": kappa_i, "E_E_gain": gain_ee})
        simulations.append(row)
        examples[f"{stage}_{value:.6f}"] = spectrum
        print(f"Experiment 2 two-stage simulation {stage}={value:.6f}", flush=True)
    simulation_frame = pd.DataFrame(simulations)
    simulation_frame.to_csv(out / "nonlinear_cross_spectrum_validation.csv", index=False)
    np.savez(
        out / "cross_spectrum_examples.npz",
        **{
            f"condition_{pos}_{name}": value[item]
            for pos, value in enumerate(examples.values())
            for item, name in enumerate(("freq_hz", "E_psd", "cross_spectrum", "coherency"))
        },
    )
    _plot(out, scan, simulation_frame, examples, complex_kappa, hopf_ee)
    save_json(
        out / "manifest.json",
        {
            "experiment": "two_stage_PG_local_EI_scan",
            "stage_1": "vary only I-to-E multiplier with E-to-E gain fixed at 1",
            "stage_2": "fix I-to-E at target and vary only E-to-E gain",
            "kappa_I_original": 1.0,
            "kappa_I_target": kappa_target,
            "complex_eigenvalue_onset_kappa_I": complex_kappa,
            "E_E_gain_original": 1.0,
            "hopf_E_E_gain": hopf_ee,
            "nonlinear_noise": 0.002,
            "nonlinear_seeds": 8,
            "duration_s": ecfg["duration_s"],
            "burn_s": ecfg["burn_s"],
            "dt_s": ecfg["dt_s"],
            "runtime_s": time.time() - started,
        },
    )
    return out


def _plot(out: Path, scan, simulations, examples, complex_kappa, hopf_ee):
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for row, (stage, boundary, boundary_label) in enumerate(
        (("I_to_E", complex_kappa, "complex onset"), ("E_to_E", hopf_ee, "Hopf"))
    ):
        data = scan[scan.stage == stage]
        sim = simulations[simulations.stage == stage]
        label = "I-to-E multiplier" if stage == "I_to_E" else "E-to-E gain"
        axes[row, 0].plot(data.scan_value, data.critical_real_per_s, color="#247BA0", lw=2)
        axes[row, 0].axhline(0, color="0.2", ls="--", lw=1)
        axes[row, 0].set(xlabel=label, ylabel=r"critical $\mathrm{Re}(\lambda)$ (s$^{-1}$)", title=f"Stage {row + 1}: stability")
        axes[row, 1].plot(data.scan_value, data.critical_frequency_hz, color="#D1495B", lw=2)
        axes[row, 1].set(xlabel=label, ylabel=r"$|\mathrm{Im}(\lambda)|/(2\pi)$ (Hz)", title="Critical-mode frequency")
        axes[row, 2].plot(data.scan_value, data.cross_spectrum_peak_hz, color="#2A9D8F", lw=2, label="linear theory")
        axes[row, 2].scatter(sim.scan_value, sim.sim_cross_spectrum_peak_hz, color="#F28E2B", s=32, label="nonlinear simulation")
        axes[row, 2].set(xlabel=label, ylabel="cross-spectrum peak [Hz]", title="E-I cross-spectrum peak")
        for ax in axes[row]:
            ax.axvline(boundary, color="#7A5195" if row == 0 else "#EF5675", ls=":" if row == 0 else "--", lw=1.6, label=boundary_label)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 2: separate complex-mode formation from Hopf destabilization", fontsize=13)
    save_figure(fig, out / "figures" / "exp2_two_stage_eigen_cross_spectrum")
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(13, 6.5), constrained_layout=True)
    for ax, (key, spectrum) in zip(axes.flat, examples.items()):
        freq, _, cross, _ = spectrum
        magnitude = abs(cross)
        magnitude /= max(float(magnitude.max()), np.finfo(float).tiny)
        peak = float(freq[np.argmax(magnitude)])
        ax.plot(freq, magnitude, color="#6A4C93", lw=1.5)
        ax.axvline(peak, color="#F28E2B", ls="--", lw=1)
        stage, value = key.rsplit("_", 1)
        stage_label = r"$s_{EI}$" if stage == "I_to_E" else r"$s_{EE}$"
        ax.set(xlim=(0, 80), ylim=(0, 1.05), xlabel="frequency (Hz)", ylabel=r"normalized $|S_{EI}|$", title=rf"{stage_label}={float(value):.3f}; peak={peak:.1f} Hz")
    fig.suptitle("Nonlinear E-I cross spectra in the two-stage scan", fontsize=13)
    save_figure(fig, out / "figures" / "exp2_two_stage_cross_spectrum_examples")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
