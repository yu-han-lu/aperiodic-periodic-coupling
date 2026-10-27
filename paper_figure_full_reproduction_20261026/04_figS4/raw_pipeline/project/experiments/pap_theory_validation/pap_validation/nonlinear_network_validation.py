from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    def njit(*args, **kwargs):
        return lambda f: f
    prange = range

from .core import (
    AREAS9,
    central_moments,
    feature_from_spectrum,
    frequency_grid,
    input_psd,
    load_model,
    network_features,
    prediction_metrics,
    save_figure,
    save_json,
    transfer_power,
)
from .exp1 import Z0, ZHI, ZLO, ZSCALE, _draw_design, _feature_fun
from .exp3 import _direct_stats, _edge_kernel, _evaluate_delta, _input_trials
from .io import load_config, prepare_output


@njit(parallel=True)
def _simulate_threshold_linear_batch(
    inputs,
    local_ee,
    local_ei,
    local_ie,
    local_ii,
    edge_target,
    edge_source,
    edge_e,
    edge_i,
    background_e,
    background_i,
    dt_ms,
    tau_e_ms,
    tau_i_ms,
    burn_steps,
    output_idx,
    input_idx,
):
    n_trial, n_time = inputs.shape
    n_area = len(local_ee)
    n_analysis = n_time - burn_steps
    output = np.empty((n_trial, n_analysis), dtype=np.float64)
    rectified = np.zeros(n_trial, dtype=np.int64)
    min_rate_e = np.empty(n_trial, dtype=np.float64)
    min_rate_i = np.empty(n_trial, dtype=np.float64)
    min_drive_e = np.empty(n_trial, dtype=np.float64)
    min_drive_i = np.empty(n_trial, dtype=np.float64)

    for trial in prange(n_trial):
        rate_e = np.full(n_area, 10.0)
        rate_i = np.full(n_area, 35.0)
        next_e = np.empty(n_area)
        next_i = np.empty(n_area)
        long_e = np.empty(n_area)
        long_i = np.empty(n_area)
        trial_min_e = 10.0
        trial_min_i = 35.0
        trial_min_drive_e = 1e300
        trial_min_drive_i = 1e300
        count = 0
        out_pos = 0
        for t in range(1, n_time):
            long_e[:] = 0.0
            long_i[:] = 0.0
            for edge in range(len(edge_target)):
                target = edge_target[edge]
                source = edge_source[edge]
                source_rate = rate_e[source]
                long_e[target] += edge_e[edge] * source_rate
                long_i[target] += edge_i[edge] * source_rate
            for area in range(n_area):
                drive_e = local_ee[area] * rate_e[area] + local_ei[area] * rate_i[area] + long_e[area] + background_e[area]
                if area == input_idx:
                    drive_e += inputs[trial, t]
                drive_i = local_ie[area] * rate_e[area] + local_ii[area] * rate_i[area] + long_i[area] + background_i[area]
                if drive_e < trial_min_drive_e:
                    trial_min_drive_e = drive_e
                if drive_i < trial_min_drive_i:
                    trial_min_drive_i = drive_i
                if drive_e < 0.0:
                    drive_e = 0.0
                    count += 1
                if drive_i < 0.0:
                    drive_i = 0.0
                    count += 1
                next_e[area] = rate_e[area] + (-rate_e[area] + drive_e) * dt_ms / tau_e_ms
                next_i[area] = rate_i[area] + (-rate_i[area] + drive_i) * dt_ms / tau_i_ms
            rate_e, next_e = next_e, rate_e
            rate_i, next_i = next_i, rate_i
            for area in range(n_area):
                if rate_e[area] < trial_min_e:
                    trial_min_e = rate_e[area]
                if rate_i[area] < trial_min_i:
                    trial_min_i = rate_i[area]
            if t >= burn_steps:
                output[trial, out_pos] = rate_e[output_idx] - 10.0
                out_pos += 1
        rectified[trial] = count
        min_rate_e[trial] = trial_min_e
        min_rate_i[trial] = trial_min_i
        min_drive_e[trial] = trial_min_drive_e
        min_drive_i[trial] = trial_min_drive_i
    return output, rectified, min_rate_e, min_rate_i, min_drive_e, min_drive_i


@njit(parallel=True)
def _simulate_threshold_linear_multioutput_batch(
    inputs,
    local_ee,
    local_ei,
    local_ie,
    local_ii,
    edge_target,
    edge_source,
    edge_e,
    edge_i,
    background_e,
    background_i,
    dt_ms,
    tau_e_ms,
    tau_i_ms,
    burn_steps,
    output_indices,
    input_idx,
):
    """Integrate the nonlinear network once and record several E populations."""
    n_trial, n_time = inputs.shape
    n_area = len(local_ee)
    n_output = len(output_indices)
    n_analysis = n_time - burn_steps
    output = np.empty((n_trial, n_output, n_analysis), dtype=np.float32)
    rectified = np.zeros(n_trial, dtype=np.int64)
    min_rate_e = np.empty(n_trial, dtype=np.float64)
    min_rate_i = np.empty(n_trial, dtype=np.float64)
    min_drive_e = np.empty(n_trial, dtype=np.float64)
    min_drive_i = np.empty(n_trial, dtype=np.float64)

    for trial in prange(n_trial):
        rate_e = np.full(n_area, 10.0)
        rate_i = np.full(n_area, 35.0)
        next_e = np.empty(n_area)
        next_i = np.empty(n_area)
        long_e = np.empty(n_area)
        long_i = np.empty(n_area)
        trial_min_e = 10.0
        trial_min_i = 35.0
        trial_min_drive_e = 1e300
        trial_min_drive_i = 1e300
        count = 0
        out_pos = 0
        for t in range(1, n_time):
            long_e[:] = 0.0
            long_i[:] = 0.0
            for edge in range(len(edge_target)):
                target = edge_target[edge]
                source = edge_source[edge]
                source_rate = rate_e[source]
                long_e[target] += edge_e[edge] * source_rate
                long_i[target] += edge_i[edge] * source_rate
            for area in range(n_area):
                drive_e = local_ee[area] * rate_e[area] + local_ei[area] * rate_i[area] + long_e[area] + background_e[area]
                if area == input_idx:
                    drive_e += inputs[trial, t]
                drive_i = local_ie[area] * rate_e[area] + local_ii[area] * rate_i[area] + long_i[area] + background_i[area]
                if drive_e < trial_min_drive_e:
                    trial_min_drive_e = drive_e
                if drive_i < trial_min_drive_i:
                    trial_min_drive_i = drive_i
                if drive_e < 0.0:
                    drive_e = 0.0
                    count += 1
                if drive_i < 0.0:
                    drive_i = 0.0
                    count += 1
                next_e[area] = rate_e[area] + (-rate_e[area] + drive_e) * dt_ms / tau_e_ms
                next_i[area] = rate_i[area] + (-rate_i[area] + drive_i) * dt_ms / tau_i_ms
            rate_e, next_e = next_e, rate_e
            rate_i, next_i = next_i, rate_i
            for area in range(n_area):
                if rate_e[area] < trial_min_e:
                    trial_min_e = rate_e[area]
                if rate_i[area] < trial_min_i:
                    trial_min_i = rate_i[area]
            if t >= burn_steps:
                for output_pos in range(n_output):
                    output[trial, output_pos, out_pos] = rate_e[output_indices[output_pos]] - 10.0
                out_pos += 1
        rectified[trial] = count
        min_rate_e[trial] = trial_min_e
        min_rate_i[trial] = trial_min_i
        min_drive_e[trial] = trial_min_drive_e
        min_drive_i[trial] = trial_min_drive_i
    return output, rectified, min_rate_e, min_rate_i, min_drive_e, min_drive_i


def _simulation_arrays(params: dict) -> tuple[np.ndarray, ...]:
    n = int(params["n_area"])
    local_ee = params["beta_exc"] * params["wEE"] * np.asarray(params["local_exc_scale"])
    local_ei = -params["beta_exc"] * params["wEI"] * np.ones(n)
    local_ie = params["beta_inh"] * params["wIE"] * np.asarray(params["local_inh_scale"])
    local_ii = -params["beta_inh"] * params["wII"] * np.ones(n)
    targets, sources = np.nonzero(params["fln_mat"])
    weights = np.asarray(params["fln_mat"])[targets, sources]
    edge_e = params["beta_exc"] * params["muEE"] * np.asarray(params["exc_scale"])[targets] * weights
    edge_i = params["beta_inh"] * params["muIE"] * np.asarray(params["inh_scale"])[targets] * weights
    long_e0 = np.zeros(n)
    long_i0 = np.zeros(n)
    np.add.at(long_e0, targets, edge_e * 10.0)
    np.add.at(long_i0, targets, edge_i * 10.0)
    background_e = 10.0 - (local_ee * 10.0 + local_ei * 35.0 + long_e0)
    background_i = 35.0 - (local_ie * 10.0 + local_ii * 35.0 + long_i0)
    return (
        local_ee.astype(float), local_ei.astype(float), local_ie.astype(float), local_ii.astype(float),
        targets.astype(np.int64), sources.astype(np.int64), edge_e.astype(float), edge_i.astype(float),
        background_e.astype(float), background_i.astype(float),
    )


def _fourier_inputs(zs: np.ndarray, dt_ms: float, analysis_steps: int, burn_steps: int, sigma: float, seed: int) -> np.ndarray:
    fs = 1000.0 / dt_ms
    freqs = np.fft.rfftfreq(analysis_steps, d=1.0 / fs)
    reference = np.sqrt(input_psd(freqs, Z0)).astype(np.complex128)
    reference[0] = 0.0
    reference_signal = np.fft.irfft(reference, n=analysis_steps)
    global_scale = sigma / (np.std(reference_signal) + 1e-15)
    rng = np.random.default_rng(seed)
    result = np.empty((len(zs), analysis_steps + burn_steps))
    for trial, z in enumerate(zs):
        q = input_psd(freqs, z)
        phase = rng.uniform(0.0, 2.0 * np.pi, len(freqs))
        spectrum = np.sqrt(q) * np.exp(1j * phase)
        spectrum[0] = 0.0
        if analysis_steps % 2 == 0:
            spectrum[-1] = spectrum[-1].real
        x = np.fft.irfft(spectrum, n=analysis_steps) * global_scale
        x -= x.mean()
        result[trial, :burn_steps] = x[-burn_steps:]
        result[trial, burn_steps:] = x
    return result


def _run_batch(
    params: dict,
    inputs: np.ndarray,
    dt_ms: float,
    burn_steps: int,
    output_idx: int,
    input_idx: int = 0,
):
    arrays = _simulation_arrays(params)
    return _simulate_threshold_linear_batch(
        inputs, *arrays, dt_ms, float(params["tau_exc"]), float(params["tau_inh"]), burn_steps,
        output_idx, input_idx,
    )


def _run_multioutput_batch(
    params: dict,
    inputs: np.ndarray,
    dt_ms: float,
    burn_steps: int,
    output_indices: np.ndarray,
    input_idx: int = 0,
):
    arrays = _simulation_arrays(params)
    return _simulate_threshold_linear_multioutput_batch(
        inputs, *arrays, dt_ms, float(params["tau_exc"]), float(params["tau_inh"]), burn_steps,
        np.asarray(output_indices, dtype=np.int64), input_idx,
    )


def _timeseries_features(
    outputs: np.ndarray,
    dt_ms: float,
    spectral_config: dict | None = None,
) -> tuple[pd.DataFrame, list[tuple[np.ndarray, np.ndarray]]]:
    spectral_config = spectral_config or {}
    fs = 1000.0 / dt_ms
    mode = spectral_config.get("mode", "hann_welch")
    segment_s = float(spectral_config.get("segment_s", 2.048))
    smoothing_sigma_hz = spectral_config.get("smoothing_sigma_hz")
    rows = []
    spectra = []
    for trial, trace in enumerate(outputs):
        if mode == "hann_welch":
            nperseg = min(len(trace), int(round(segment_s * fs)))
            freq, psd = signal.welch(
                trace, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
                detrend="constant", scaling="density",
            )
        elif mode == "boxcar_full":
            nperseg = len(trace)
            freq, psd = signal.welch(
                trace, fs=fs, window="boxcar", nperseg=nperseg, noverlap=0,
                detrend="constant", scaling="density",
            )
        else:
            raise ValueError(f"Unknown spectral estimator mode: {mode}")
        keep = (freq >= 0.5) & (freq <= 1200.0)
        if smoothing_sigma_hz is None:
            smoothing_sigma_bins = 2.0
        else:
            smoothing_sigma_bins = float(smoothing_sigma_hz) / float(freq[1] - freq[0])
        log_psd = ndimage.gaussian_filter1d(
            np.log(np.maximum(psd[keep], 1e-300)), smoothing_sigma_bins
        )
        smooth_psd = np.exp(log_psd)
        feature = feature_from_spectrum(freq[keep], smooth_psd)
        rows.append({"trial_id": trial, **feature})
        spectra.append((freq[keep], smooth_psd))
    return pd.DataFrame(rows), spectra


def _fit_annotation(ax, predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    predicted = np.asarray(predicted, float)
    observed = np.asarray(observed, float)
    ok = np.isfinite(predicted) & np.isfinite(observed)
    x, y = predicted[ok], observed[ok]
    fit = stats.linregress(x, y)
    xx = np.linspace(x.min(), x.max(), 100)
    ax.plot(xx, fit.intercept + fit.slope * xx, color="#1f6f8b", lw=2, label="OLS")
    lo = min(x.min(), y.min())
    hi = max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
    p_text = f"{fit.pvalue:.2e}" if fit.pvalue < 0.001 else f"{fit.pvalue:.3f}"
    r2_identity = 1.0 - np.sum((x - y) ** 2) / np.sum((y - y.mean()) ** 2)
    ax.text(
        0.04, 0.96,
        f"b={fit.slope:.3g}, p={p_text}\nPearson r={fit.rvalue:.3f}\nidentity R^2={r2_identity:.3f}\nn={len(x)}",
        transform=ax.transAxes, va="top", fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.92},
    )
    return {"slope": float(fit.slope), "p_value": float(fit.pvalue), "pearson_r": float(fit.rvalue), "identity_R2": float(r2_identity), "n": int(len(x))}


def _exp1_conditions(cfg: dict) -> list[tuple[str, float, float]]:
    eps = float(cfg["exp1_nonlinear"]["epsilon"])
    designs = ["independent_gaussian", "rank_one", "uniform", "fixed_total", "skewed", "one_factor_mixture", "finite_sample"]
    return [(name, eps, 0.0) for name in designs] + [("correlated_gaussian", eps, float(rho)) for rho in cfg["exp1"]["rhos"]]


def run_exp1(
    cfg: dict,
    output_name: str = "exp1_nonlinear_full55",
    spectral_config: dict | None = None,
) -> Path:
    started = time.time()
    out = prepare_output(cfg, output_name)
    ecfg = cfg["exp1_nonlinear"]
    model = load_model(Path(cfg["model_dir"]))
    dt = float(ecfg["dt_ms"])
    analysis_steps = int(round(float(ecfg["analysis_s"]) * 1000.0 / dt))
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    n_trials = int(ecfg["n_trials"])
    input_idx, output_idx = model.areas.index("V1"), model.areas.index("PG")
    if input_idx != 0:
        raise ValueError("The batch integrator currently expects V1 at index 0.")
    freqs = frequency_grid()
    transfer = transfer_power(model.J, input_idx, output_idx, freqs)
    feature_fun = _feature_fun(transfer, freqs, tuple(cfg["peak_band_hz"]), tuple(cfg["slope_band_hz"]))
    h = 2e-3
    eye = np.eye(4)
    grad = np.stack([(feature_fun(h * e) - feature_fun(-h * e)) / (2 * h) for e in eye], axis=1)
    condition_rows = []
    trial_rows = []
    representative = None

    for ci, (design, eps, rho) in enumerate(_exp1_conditions(cfg)):
        rng = np.random.default_rng(int(cfg["seed"]) + 1000 + ci)
        u = _draw_design(design, n_trials, eps, rng, rho)
        raw = Z0 + u * ZSCALE
        z = np.clip(raw, ZLO, ZHI)
        clipped = np.any(raw != z, axis=1)
        inputs = _fourier_inputs(z, dt, analysis_steps, burn_steps, float(ecfg["input_sigma"]), int(cfg["seed"]) + 120000 + ci)
        outputs, rect, min_e, min_i, min_de, min_di = _run_batch(model.params, inputs, dt, burn_steps, output_idx)
        nonlinear, spectra = _timeseries_features(outputs, dt, spectral_config)
        analytic_values = np.empty((n_trials, 2))
        analytic_valid = np.zeros(n_trials, bool)
        for trial, zz in enumerate(z):
            feature, _ = network_features(model.J, zz, input_idx, output_idx, freqs, transfer)
            analytic_values[trial] = feature["peak_omega"], feature["exponent"]
            analytic_valid[trial] = feature["peak_valid"]
        valid = nonlinear.peak_valid.to_numpy(bool) & analytic_valid & ~clipped
        accepted_u = (z[valid] - Z0) / ZSCALE
        sigma, _ = central_moments(accepted_u)
        theory_cov = float(grad[0] @ sigma @ grad[1])
        surrogate = accepted_u @ grad.T
        theory_r = float(stats.pearsonr(surrogate[:, 0], surrogate[:, 1]).statistic)
        nl_values = nonlinear.loc[valid, ["peak_omega", "exponent"]].to_numpy()
        direct_values = analytic_values[valid]
        nl_cov = float(np.cov(nl_values.T, ddof=1)[0, 1])
        nl_r = float(stats.pearsonr(nl_values[:, 0], nl_values[:, 1]).statistic)
        direct_cov = float(np.cov(direct_values.T, ddof=1)[0, 1])
        direct_r = float(stats.pearsonr(direct_values[:, 0], direct_values[:, 1]).statistic)
        condition = f"{design}_eps{eps:g}_rho{rho:g}"
        condition_rows.append({
            "condition": condition, "design": design, "epsilon": eps, "rho": rho,
            "n_trials": n_trials, "n_valid": int(valid.sum()), "n_excluded": int(n_trials - valid.sum()),
            "nonlinear_covariance": nl_cov, "theory_C49": theory_cov, "analytic_direct_covariance": direct_cov,
            "nonlinear_r": nl_r, "theory_linear_surrogate_r": theory_r, "analytic_direct_r": direct_r,
            "rectification_fraction": float(rect.sum() / (2 * len(model.areas) * inputs.shape[1] * n_trials)),
            "min_rate_E": float(min_e.min()), "min_rate_I": float(min_i.min()),
            "min_drive_E": float(min_de.min()), "min_drive_I": float(min_di.min()),
        })
        for trial in range(n_trials):
            trial_rows.append({
                "condition": condition, "trial_id": trial,
                "bump_center_hz": z[trial, 0], "bump_amplitude": z[trial, 1], "beta": z[trial, 2], "aperiodic_amplitude": z[trial, 3],
                "clipped": bool(clipped[trial]), "peak_hz": nonlinear.loc[trial, "peak_hz"],
                "peak_omega": nonlinear.loc[trial, "peak_omega"], "exponent": nonlinear.loc[trial, "exponent"],
                "peak_valid": bool(nonlinear.loc[trial, "peak_valid"]), "rectified_steps": int(rect[trial]),
                "min_rate_E": min_e[trial], "min_rate_I": min_i[trial],
            })
        if representative is None:
            representative = (inputs[0, burn_steps:], outputs[0], spectra[0][0], spectra[0][1], z[0])
        print(f"exp1 nonlinear {ci + 1}/{len(_exp1_conditions(cfg))}: {condition}", flush=True)

    summary = pd.DataFrame(condition_rows)
    trials = pd.DataFrame(trial_rows)
    summary.to_csv(out / "condition_summary.csv", index=False)
    trials.to_csv(out / "trials.csv.gz", index=False, compression="gzip")
    covariance_metrics = prediction_metrics(summary.nonlinear_covariance.values, summary.theory_C49.values)
    correlation_metrics = prediction_metrics(summary.nonlinear_r.values, summary.theory_linear_surrogate_r.values)
    direct_cov_metrics = prediction_metrics(summary.nonlinear_covariance.values, summary.analytic_direct_covariance.values)
    direct_r_metrics = prediction_metrics(summary.nonlinear_r.values, summary.analytic_direct_r.values)
    metrics = {
        "nonlinear_vs_C49_covariance": covariance_metrics,
        "nonlinear_vs_linear_surrogate_correlation": correlation_metrics,
        "nonlinear_vs_analytic_direct_covariance": direct_cov_metrics,
        "nonlinear_vs_analytic_direct_correlation": direct_r_metrics,
        "rectification_fraction": float(summary.rectification_fraction.mean()),
        "runtime_s": time.time() - started,
    }
    metrics["nonlinear_vs_C49_covariance"]["regression_p_value"] = float(
        stats.linregress(summary.theory_C49, summary.nonlinear_covariance).pvalue
    )
    save_json(out / "metrics.json", metrics)
    save_json(out / "manifest.json", {
        "experiment": "full55_threshold_nonlinear_input_sampling",
        "areas": len(model.areas), "input": "V1 E", "output": "PG E", "n_conditions": len(summary),
        "n_trials_per_condition": n_trials, "dt_ms": dt, "burn_s": ecfg["burn_s"], "analysis_s": ecfg["analysis_s"],
        "input_generation": "fixed Fourier amplitudes from Q(f;z), random phases; periodic burn-in",
        "spectral_estimator": spectral_config or {"mode": "hann_welch", "segment_s": 2.048, "smoothing_sigma_bins": 2.0},
    })
    if representative is not None:
        np.savez(out / "representative_timeseries_psd.npz", input=representative[0], output=representative[1], freq_hz=representative[2], output_psd=representative[3], z=representative[4], dt_ms=dt)
    _figure_exp1(out, summary)
    return out


def _figure_exp1(out: Path, summary: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    comparisons = [
        ("theory_C49", "nonlinear_covariance", "Eq.49 covariance", "nonlinear covariance"),
        ("theory_linear_surrogate_r", "nonlinear_r", "linear-surrogate r", "nonlinear r"),
        ("analytic_direct_covariance", "nonlinear_covariance", "analytic direct covariance", "nonlinear covariance"),
        ("analytic_direct_r", "nonlinear_r", "analytic direct r", "nonlinear r"),
    ]
    for ax, (x, y, xlabel, ylabel) in zip(axes.flat, comparisons):
        ax.scatter(summary[x], summary[y], c=summary.epsilon, cmap="viridis", s=42, alpha=0.82)
        _fit_annotation(ax, summary[x].to_numpy(), summary[y].to_numpy())
        ax.set(xlabel=xlabel, ylabel=ylabel)
    fig.suptitle("Full 55-area threshold-nonlinear simulation: theory versus time-domain readout", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, out / "figures" / "full55_nonlinear_vs_theory")
    plt.close(fig)


def run_exp3(
    cfg: dict,
    output_name: str = "exp3_nonlinear_subnetwork",
    spectral_config: dict | None = None,
) -> Path:
    started = time.time()
    out = prepare_output(cfg, output_name)
    ecfg = cfg["exp3_nonlinear"]
    model = load_model(Path(cfg["model_dir"]), AREAS9)
    dt = float(ecfg["dt_ms"])
    analysis_steps = int(round(float(ecfg["analysis_s"]) * 1000.0 / dt))
    burn_steps = int(round(float(ecfg["burn_s"]) * 1000.0 / dt))
    n_trials = int(ecfg["n_input_trials"])
    zs, u = _input_trials(n_trials, int(cfg["seed"]) + 300)
    inputs = _fourier_inputs(zs, dt, analysis_steps, burn_steps, float(ecfg["input_sigma"]), int(cfg["seed"]) + 220000)
    baseline_output, base_rect, base_min_e, base_min_i, base_min_de, base_min_di = _run_batch(model.params, inputs, dt, burn_steps, 5)
    baseline_features, baseline_spectra = _timeseries_features(baseline_output, dt, spectral_config)
    source_trials = pd.read_csv(Path(cfg["output_dir"]) / "exp3_structural_kernel" / "trials.csv.gz")
    selected = source_trials[(source_trials.kind == "random15_delete") & (source_trials.split == "heldout")].head(int(ecfg["n_masks"]))
    simulations = []
    representative = None
    for position, source_row in enumerate(selected.itertuples(index=False), 1):
        delta_w = np.zeros_like(model.fln)
        for source, target, weight in json.loads(source_row.edges_json):
            delta_w[model.areas.index(target), model.areas.index(source)] = -float(weight)
        params = model.params.copy()
        params["fln_mat"] = model.fln + delta_w
        pert_output, rect, min_e, min_i, min_de, min_di = _run_batch(params, inputs, dt, burn_steps, 5)
        pert_features, pert_spectra = _timeseries_features(pert_output, dt, spectral_config)
        simulations.append({
            "source_row": source_row, "delta_w": delta_w, "features": pert_features,
            "rect": rect, "min_e": min_e, "min_i": min_i, "min_de": min_de, "min_di": min_di,
        })
        if representative is None:
            representative = (baseline_output[0], pert_output[0], baseline_spectra[0], pert_spectra[0], source_row.perturbation_id)
        if position % 10 == 0 or position == len(selected):
            print(f"exp3 nonlinear {position}/{len(selected)} masks", flush=True)

    global_valid = baseline_features.peak_valid.to_numpy(bool).copy()
    global_valid &= np.isfinite(baseline_features[["peak_omega", "exponent"]]).all(axis=1).to_numpy()
    for record in simulations:
        features = record["features"]
        global_valid &= features.peak_valid.to_numpy(bool)
        global_valid &= np.isfinite(features[["peak_omega", "exponent"]]).all(axis=1).to_numpy()
    if global_valid.sum() < 10:
        raise RuntimeError(f"Only {global_valid.sum()} trials are valid across all structural masks.")

    zs_common = zs[global_valid]
    u_common = u[global_valid]
    sigma = np.cov(u_common, rowvar=False, ddof=1)
    freqs = frequency_grid()
    base_linear, coefficients = _edge_kernel(model, sigma, freqs, 0, 5, list(zip(*np.nonzero(model.fln))))
    analytic_baseline, _, _ = _direct_stats(model.J, zs_common, freqs, 0, 5)
    base_values = baseline_features.loc[global_valid, ["peak_omega", "exponent"]].to_numpy()
    base_r = float(stats.pearsonr(base_values[:, 0], base_values[:, 1]).statistic)
    rows = []
    feature_rows = []
    for trial, feature in baseline_features.iterrows():
        feature_rows.append({
            "network": "baseline", "perturbation_id": "baseline", "trial_id": trial,
            "global_valid": bool(global_valid[trial]), "peak_hz": feature.peak_hz,
            "peak_omega": feature.peak_omega, "exponent": feature.exponent,
            "peak_valid": bool(feature.peak_valid),
        })
    for record in simulations:
        source_row = record["source_row"]
        pert_features = record["features"]
        pert_values = pert_features.loc[global_valid, ["peak_omega", "exponent"]].to_numpy()
        pert_r = float(stats.pearsonr(pert_values[:, 0], pert_values[:, 1]).statistic)
        theory = _evaluate_delta(
            model, record["delta_w"], coefficients, "random15_delete", source_row.perturbation_id,
            zs_common, sigma, freqs, 0, 5, analytic_baseline, base_linear,
        )
        pre_common_valid = baseline_features.peak_valid.to_numpy(bool) & pert_features.peak_valid.to_numpy(bool)
        rows.append({
            "perturbation_id": source_row.perturbation_id, "n_edges": int(source_row.n_edges),
            "n_input_trials": n_trials, "n_valid_before_global_intersection": int(pre_common_valid.sum()),
            "n_valid_common": int(global_valid.sum()),
            "baseline_nonlinear_r": base_r, "perturbed_nonlinear_r": pert_r,
            "delta_rho_nonlinear": pert_r - base_r,
            "delta_rho_analytic_direct": theory["delta_rho_observed"],
            "delta_rho_first": theory["delta_rho_first"], "delta_rho_second": theory["delta_rho_second"],
            "rectification_fraction": float(record["rect"].sum() / (2 * len(model.areas) * inputs.shape[1] * n_trials)),
            "min_rate_E": float(record["min_e"].min()), "min_rate_I": float(record["min_i"].min()),
            "min_drive_E": float(record["min_de"].min()), "min_drive_I": float(record["min_di"].min()),
            "edges_json": source_row.edges_json,
        })
        for trial, feature in pert_features.iterrows():
            feature_rows.append({
                "network": "perturbed", "perturbation_id": source_row.perturbation_id, "trial_id": trial,
                "global_valid": bool(global_valid[trial]), "peak_hz": feature.peak_hz,
                "peak_omega": feature.peak_omega, "exponent": feature.exponent,
                "peak_valid": bool(feature.peak_valid),
            })

    results = pd.DataFrame(rows)
    results.to_csv(out / "delta_correlation_results.csv", index=False)
    pd.DataFrame(feature_rows).to_csv(out / "trial_features.csv.gz", index=False, compression="gzip")
    metrics = {}
    for key in ("delta_rho_analytic_direct", "delta_rho_first", "delta_rho_second"):
        metrics[key] = prediction_metrics(results.delta_rho_nonlinear.values, results[key].values)
        fit = stats.linregress(results[key], results.delta_rho_nonlinear)
        metrics[key]["regression_p_value"] = float(fit.pvalue)
    metrics.update({
        "baseline_nonlinear_r_common_trials": base_r,
        "n_common_valid_trials": int(global_valid.sum()),
        "baseline_rectification_fraction": float(base_rect.sum() / (2 * len(model.areas) * inputs.shape[1] * n_trials)),
        "runtime_s": time.time() - started,
    })
    save_json(out / "metrics.json", metrics)
    save_json(out / "manifest.json", {
        "experiment": "nine_area_threshold_nonlinear_structural_validation",
        "areas": AREAS9, "input": "V1 E", "output": "PG E", "n_masks": len(results),
        "mask_selection": "first n pre-existing held-out random15_delete masks",
        "n_input_trials": n_trials, "common_random_number_inputs": True,
        "validity_rule": "fixed intersection valid in baseline and every selected mask",
        "dt_ms": dt, "burn_s": ecfg["burn_s"], "analysis_s": ecfg["analysis_s"],
        "spectral_estimator": spectral_config or {"mode": "hann_welch", "segment_s": 2.048, "smoothing_sigma_bins": 2.0},
    })
    if representative is not None:
        np.savez(
            out / "representative_timeseries_psd.npz",
            baseline_output=representative[0], perturbed_output=representative[1],
            baseline_freq=representative[2][0], baseline_psd=representative[2][1],
            perturbed_freq=representative[3][0], perturbed_psd=representative[3][1],
            perturbation_id=representative[4], dt_ms=dt,
        )
    _figure_exp3(out, results)
    return out


def _figure_exp3(out: Path, results: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    comparisons = [
        ("delta_rho_analytic_direct", "Direct analytic"),
        ("delta_rho_first", "First-order kernel"),
        ("delta_rho_second", "Second-order expansion"),
    ]
    for ax, (column, title) in zip(axes, comparisons):
        ax.scatter(results[column], results.delta_rho_nonlinear, s=24, alpha=0.7, color="#4c72b0")
        _fit_annotation(ax, results[column].to_numpy(), results.delta_rho_nonlinear.to_numpy())
        ax.set(xlabel=f"Predicted Delta correlation: {title}", ylabel="Nonlinear time-domain Delta correlation", title=title)
    fig.suptitle("Nine-area threshold-nonlinear network: structural Delta correlation", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_figure(fig, out / "figures" / "nonlinear_delta_correlation_predictions")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--experiment", choices=["exp1", "exp3", "all"], default="all")
    args = parser.parse_args()
    cfg = load_config(args.config)
    outputs = []
    if args.experiment in ("exp1", "all"):
        outputs.append(run_exp1(cfg))
    if args.experiment in ("exp3", "all"):
        outputs.append(run_exp3(cfg))
    print("\n".join(map(str, outputs)))


if __name__ == "__main__":
    main()
