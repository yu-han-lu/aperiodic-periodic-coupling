from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import interpolate, ndimage, signal, stats

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = lambda *args, **kwargs: (lambda f: f)
    prange = range

from .core import feature_from_spectrum, prediction_metrics, save_figure, save_json
from .exp2_two_stage import _critical, _jacobian, _matrix, _setup
from .io import load_config, prepare_output


OUTPUT_NAME = "exp2_pap_correlation_validation"
DT_S = 0.00025
BURN_S = 2.048
ANALYSIS_S = 32.768
N_TRIALS = 256
N_CALIBRATION_PHASES = 3
N_HELDOUT_PHASES = 5
S_VALUES = (1.0, 1.5, 2.0815631208601286, 2.4, 3.1774487915806437, 4.0, 5.0, 6.0, 6.693999317940512)
SIGMA_BETA_VALUES = (0.05, 0.10, 0.20, 0.40, 0.60)
FLOOR_RATIOS = (0.0, 0.025, 0.05, 0.10, 0.20, 0.50)
MAIN_INPUT_SIGMA = 0.02
AMPLITUDE_CONTROLS = (0.005, 0.05)
F0_HZ = 10.0
PEAK_BAND = (5.0, 100.0)
SLOPE_BAND = (100.0, 1000.0)


@njit(parallel=True)
def _forced_ei_batch_parallel(w, tau, forcing, dt, burn_steps):
    n_trials, n_steps = forcing.shape
    n_analysis = n_steps - burn_steps
    output = np.empty((n_trials, n_analysis), dtype=np.float32)
    for trial in prange(n_trials):
        e = 0.0
        i = 0.0
        out_pos = 0
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
            if step >= burn_steps:
                output[trial, out_pos] = e
                out_pos += 1
    return output


def _beta_trials(sigma_beta, seed):
    rng = np.random.default_rng(seed)
    lower, upper = -0.8 / sigma_beta, 0.8 / sigma_beta
    return stats.truncnorm.rvs(lower, upper, loc=0.0, scale=sigma_beta, size=N_TRIALS, random_state=rng)


def _normalized_input_shape(frequencies, beta):
    f = np.maximum(frequencies, 0.5)
    raw = (f / F0_HZ) ** beta
    # Match the time-domain generator, whose FFT covers the full 0--Nyquist
    # interval.  Normalizing on the analysis slice itself would make the
    # relative white-floor level depend on which frequencies the caller kept.
    nyquist = 0.5 / DT_S
    floor_hz = 0.5
    integral = (
        floor_hz * (floor_hz / F0_HZ) ** beta
        + F0_HZ ** (-beta)
        * (nyquist ** (beta + 1.0) - floor_hz ** (beta + 1.0))
        / (beta + 1.0)
    )
    return raw / (integral / nyquist)


def _forcing(betas, input_sigma, floor_ratio, seed):
    analysis_steps = int(round(ANALYSIS_S / DT_S))
    burn_steps = int(round(BURN_S / DT_S))
    frequencies = np.fft.rfftfreq(analysis_steps, d=DT_S)
    rng = np.random.default_rng(seed)
    result = np.empty((len(betas), analysis_steps + burn_steps), dtype=np.float32)
    for trial, beta in enumerate(betas):
        q = _normalized_input_shape(frequencies, beta)
        phase = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
        spectrum = np.sqrt(q) * np.exp(1j * phase)
        spectrum[0] = 0.0
        spectrum[-1] = spectrum[-1].real
        colored = np.fft.irfft(spectrum, n=analysis_steps)
        colored *= input_sigma / max(colored.std(), 1e-15)
        floor = input_sigma * floor_ratio * rng.normal(size=analysis_steps)
        x = colored + floor
        x -= x.mean()
        result[trial, :burn_steps] = x[-burn_steps:]
        result[trial, burn_steps:] = x
    return result


def _linear_heun_outputs(j, tau_e, forcing):
    analysis_steps = int(round(ANALYSIS_S / DT_S))
    burn_steps = int(round(BURN_S / DT_S))
    u = forcing[:, burn_steps:]
    a = np.eye(2) + DT_S * j + 0.5 * DT_S**2 * (j @ j)
    b = np.array([1.0 / tau_e, 0.0])
    bd = DT_S * b + 0.5 * DT_S**2 * (j @ b)
    frequencies = np.fft.rfftfreq(analysis_steps, d=DT_S)
    theta = 2.0 * np.pi * frequencies * DT_S
    transfer = np.empty(len(frequencies), dtype=complex)
    for pos, angle in enumerate(theta):
        transfer[pos] = np.linalg.solve(np.eye(2) - a * np.exp(-1j * angle), bd)[0]
    return np.fft.irfft(np.fft.rfft(u, axis=1) * transfer[None, :], n=analysis_steps, axis=1).astype(np.float32)


def _extract_features(paths):
    fs = 1.0 / DT_S
    nperseg = int(round(4.096 / DT_S))
    frequencies, psd = signal.welch(
        paths, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2,
        detrend="constant", scaling="density", axis=-1,
    )
    keep = (frequencies >= 0.5) & (frequencies <= 1200.0)
    f = frequencies[keep]
    sigma_bins = 0.9765625 / (f[1] - f[0])
    smooth = np.exp(ndimage.gaussian_filter1d(np.log(np.maximum(psd[:, keep], 1e-300)), sigma_bins, axis=-1))
    values = np.empty((len(paths), 2))
    valid = np.empty(len(paths), dtype=bool)
    reasons = []
    for trial in range(len(paths)):
        feature = feature_from_spectrum(f, smooth[trial], peak_band=PEAK_BAND, slope_band=SLOPE_BAND)
        values[trial] = feature["peak_hz"], -feature["exponent"]
        # Welch spectra contain harmless local ripples.  The registered feature
        # is the highest interior maximum; competing small maxima are retained
        # and reported as QC rather than deleting the whole trial.
        valid[trial] = bool(not feature["peak_boundary"] and "no_interior_peak" not in feature["feature_failure_reason"])
        reasons.append(feature["feature_failure_reason"])
    return values, valid, reasons


def _transfer_power(j, tau_e, frequencies):
    b = np.array([1.0 / tau_e, 0.0])
    result = np.empty(len(frequencies))
    for pos, frequency in enumerate(frequencies):
        response = np.linalg.solve(1j * 2.0 * np.pi * frequency * np.eye(2) - j, b)
        result[pos] = abs(response[0]) ** 2
    return np.maximum(result, 1e-300)


def _input_q(frequencies, beta, input_sigma, floor_ratio):
    return input_sigma**2 * (
        _normalized_input_shape(frequencies, beta) + floor_ratio**2
    )


def _theory_feature(transfer, frequencies, beta, input_sigma, floor_ratio):
    feature = feature_from_spectrum(
        frequencies, transfer * _input_q(frequencies, beta, input_sigma, floor_ratio),
        peak_band=PEAK_BAND, slope_band=SLOPE_BAND,
    )
    return np.array([feature["peak_hz"], -feature["exponent"]]), bool(feature["peak_valid"]), feature["feature_failure_reason"]


def _theory_derivatives(j, tau_e, input_sigma, floor_ratio):
    frequencies = np.geomspace(0.5, 1200.0, 1600)
    transfer = _transfer_power(j, tau_e, frequencies)
    h = 1e-3
    minus, _, _ = _theory_feature(transfer, frequencies, -h, input_sigma, floor_ratio)
    base, base_valid, _ = _theory_feature(transfer, frequencies, 0.0, input_sigma, floor_ratio)
    plus, _, _ = _theory_feature(transfer, frequencies, h, input_sigma, floor_ratio)
    finite = (plus - minus) / (2.0 * h)

    z = np.log(frequencies)
    q0 = _input_q(frequencies, 0.0, input_sigma, floor_ratio)
    dh = 1e-4
    dlogq = (
        np.log(_input_q(frequencies, dh, input_sigma, floor_ratio))
        - np.log(_input_q(frequencies, -dh, input_sigma, floor_ratio))
    ) / (2.0 * dh)
    log_psd = np.log(transfer * q0)
    spline = interpolate.CubicSpline(z, log_psd)
    peak_z = np.log(base[0])
    lzz = float(spline.derivative(2)(peak_z))
    dzbeta = interpolate.CubicSpline(z, dlogq).derivative()(peak_z)
    m_implicit = float(-base[0] * dzbeta / lzz) if base_valid else np.nan
    slope_use = (frequencies >= SLOPE_BAND[0]) & (frequencies <= SLOPE_BAND[1])
    k_analytic = float(stats.linregress(z[slope_use], dlogq[slope_use]).slope)
    return {
        "base_peak_hz": base[0], "base_slope": base[1], "base_peak_valid": base_valid,
        "m_beta_implicit": m_implicit, "m_beta_finite": finite[0],
        "kappa_beta_analytic": k_analytic, "kappa_beta_finite": finite[1],
        "m_relative_error": abs(m_implicit - finite[0]) / max(abs(finite[0]), 1e-12) if base_valid else np.nan,
        "kappa_relative_error": abs(k_analytic - finite[1]) / max(abs(finite[1]), 1e-12),
    }


def _condition_grid():
    rows = []
    for s_value in S_VALUES:
        for sigma_beta in SIGMA_BETA_VALUES:
            for floor_ratio in FLOOR_RATIOS:
                rows.append((s_value, sigma_beta, floor_ratio, MAIN_INPUT_SIGMA, "main_grid"))
        for input_sigma in AMPLITUDE_CONTROLS:
            rows.append((s_value, 0.40, 0.10, input_sigma, "amplitude_control"))
    return rows


def _simulate_condition(cfg, out, original, tau, condition_id, condition, force=False):
    s_value, sigma_beta, floor_ratio, input_sigma, family = condition
    path = out / "condition_features" / f"condition_{condition_id:04d}.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return np.load(path, allow_pickle=True)
    betas = _beta_trials(sigma_beta, int(cfg["seed"]) + 6100000 + int(round(sigma_beta * 1000)))
    w = _matrix(original, s_value, 1.0)
    j = _jacobian(original, tau, s_value, 1.0)
    theory_frequencies = np.geomspace(0.5, 1200.0, 1600)
    theory_transfer = _transfer_power(j, float(tau[0]), theory_frequencies)
    theory_valid_count = sum(
        _theory_feature(theory_transfer, theory_frequencies, beta, input_sigma, floor_ratio)[1]
        for beta in betas
    )
    if theory_valid_count < 80:
        n_phases = N_CALIBRATION_PHASES + N_HELDOUT_PHASES
        empty = np.full((n_phases, N_TRIALS, 2), np.nan)
        empty_valid = np.zeros((n_phases, N_TRIALS), dtype=bool)
        np.savez_compressed(
            path, betas=betas, nonlinear=empty, linear=empty.copy(),
            nonlinear_valid=empty_valid, linear_valid=empty_valid.copy(),
            nonlinear_reasons=np.full((n_phases, N_TRIALS), "theory_peak_not_identifiable", dtype="U64"),
            condition=np.asarray([s_value, sigma_beta, floor_ratio, input_sigma], dtype=float),
            family=np.asarray(family),
        )
        return np.load(path, allow_pickle=True)
    nonlinear = np.empty((N_CALIBRATION_PHASES + N_HELDOUT_PHASES, N_TRIALS, 2))
    linear = np.full_like(nonlinear, np.nan)
    nonlinear_valid = np.empty((len(nonlinear), N_TRIALS), dtype=bool)
    linear_valid = np.zeros_like(nonlinear_valid)
    run_linear_control = np.isclose(sigma_beta, 0.40) and np.isclose(floor_ratio, 0.10)
    nonlinear_reasons = []
    for phase in range(len(nonlinear)):
        forcing = _forcing(
            betas, input_sigma, floor_ratio,
            int(cfg["seed"]) + 6200000 + condition_id * 1000 + phase * 100000,
        )
        nonlinear_paths = _forced_ei_batch_parallel(
            w, tau, forcing, DT_S, int(round(BURN_S / DT_S))
        )
        nonlinear[phase], nonlinear_valid[phase], reasons = _extract_features(nonlinear_paths)
        if run_linear_control:
            linear_paths = _linear_heun_outputs(j, float(tau[0]), forcing)
            linear[phase], linear_valid[phase], _ = _extract_features(linear_paths)
        nonlinear_reasons.append(reasons)
    np.savez_compressed(
        path, betas=betas, nonlinear=nonlinear, linear=linear,
        nonlinear_valid=nonlinear_valid, linear_valid=linear_valid,
        nonlinear_reasons=np.asarray(nonlinear_reasons, dtype="U64"),
        condition=np.asarray([s_value, sigma_beta, floor_ratio, input_sigma], dtype=float),
        family=np.asarray(family),
    )
    return np.load(path, allow_pickle=True)


def _pearson(values):
    if len(values) < 8 or np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0:
        return np.nan, np.nan
    result = stats.pearsonr(values[:, 0], values[:, 1])
    return float(result.statistic), float(result.pvalue)


def _residual_prediction(theory_derivative, beta_variance, residual_covariance):
    m = theory_derivative["m_beta_implicit"]
    k = theory_derivative["kappa_beta_analytic"]
    vp = m * m * beta_variance + residual_covariance[0, 0]
    va = k * k * beta_variance + residual_covariance[1, 1]
    cov = m * k * beta_variance + residual_covariance[0, 1]
    return float(cov / np.sqrt(max(vp * va, 1e-30)))


def _analyze_condition(original, tau, condition_id, condition, cached):
    s_value, sigma_beta, floor_ratio, input_sigma, family = condition
    betas = cached["betas"]
    nonlinear = cached["nonlinear"]
    linear = cached["linear"]
    nonlinear_valid = cached["nonlinear_valid"]
    linear_valid = cached["linear_valid"]
    j = _jacobian(original, tau, s_value, 1.0)
    derivative = _theory_derivatives(j, float(tau[0]), input_sigma, floor_ratio)
    frequencies = np.geomspace(0.5, 1200.0, 1600)
    transfer = _transfer_power(j, float(tau[0]), frequencies)
    theory_values = np.empty((N_TRIALS, 2))
    theory_valid = np.empty(N_TRIALS, dtype=bool)
    theory_reasons = []
    for trial, beta in enumerate(betas):
        theory_values[trial], theory_valid[trial], reason = _theory_feature(
            transfer, frequencies, beta, input_sigma, floor_ratio
        )
        theory_reasons.append(reason)
    common = theory_valid & nonlinear_valid.all(axis=0)
    registered_linear_control = np.isclose(sigma_beta, 0.40) and np.isclose(floor_ratio, 0.10)
    linear_common = common & linear_valid.all(axis=0) if registered_linear_control else np.zeros_like(common)
    n_valid = int(common.sum())
    failure = ""
    if n_valid < 80:
        failure = "insufficient_common_peak_valid"
    cal = nonlinear[:N_CALIBRATION_PHASES, common]
    cal_centered = cal - cal.mean(axis=0, keepdims=True)
    residual_cov = np.cov(cal_centered.reshape(-1, 2), rowvar=False, ddof=1) if n_valid >= 2 else np.full((2, 2), np.nan)
    linear_cal = linear[:N_CALIBRATION_PHASES, linear_common]
    linear_centered = linear_cal - linear_cal.mean(axis=0, keepdims=True) if linear_common.any() else linear_cal
    linear_residual_cov = np.cov(linear_centered.reshape(-1, 2), rowvar=False, ddof=1) if linear_common.sum() >= 2 else np.full((2, 2), np.nan)
    beta_variance = float(np.var(betas[common], ddof=1)) if n_valid >= 2 else np.nan
    exact_r, _ = _pearson(theory_values[common])
    local_values = np.column_stack(
        [
            derivative["base_peak_hz"] + derivative["m_beta_implicit"] * betas[common],
            derivative["base_slope"] + derivative["kappa_beta_analytic"] * betas[common],
        ]
    )
    local_r, _ = _pearson(local_values)
    corrected_r = _residual_prediction(derivative, beta_variance, residual_cov) if n_valid >= 2 else np.nan
    linear_corrected_r = _residual_prediction(derivative, beta_variance, linear_residual_cov) if n_valid >= 2 else np.nan
    heldout_nonlinear_r = []
    heldout_nonlinear_p = []
    heldout_linear_r = []
    for phase in range(N_CALIBRATION_PHASES, N_CALIBRATION_PHASES + N_HELDOUT_PHASES):
        value, pvalue = _pearson(nonlinear[phase, common])
        heldout_nonlinear_r.append(value)
        heldout_nonlinear_p.append(pvalue)
        heldout_linear_r.append(_pearson(linear[phase, linear_common])[0])
    ensemble_nonlinear = nonlinear[N_CALIBRATION_PHASES:, common].mean(axis=0)
    ensemble_linear = linear[N_CALIBRATION_PHASES:, linear_common].mean(axis=0)
    ensemble_nonlinear_r, ensemble_nonlinear_p = _pearson(ensemble_nonlinear)
    ensemble_linear_r, _ = _pearson(ensemble_linear)
    nonlinear_peak_fit = stats.linregress(betas[common], ensemble_nonlinear[:, 0]) if n_valid >= 3 else None
    nonlinear_slope_fit = stats.linregress(betas[common], ensemble_nonlinear[:, 1]) if n_valid >= 3 else None
    finite_phase = np.asarray(heldout_nonlinear_r)[np.isfinite(heldout_nonlinear_r)]
    phase_mean = float(np.mean(finite_phase)) if len(finite_phase) else np.nan
    phase_sd = float(np.std(finite_phase, ddof=1)) if len(finite_phase) >= 2 else np.nan
    phase_sem = phase_sd / np.sqrt(N_HELDOUT_PHASES)
    tcrit = stats.t.ppf(0.975, N_HELDOUT_PHASES - 1)
    reasons = cached["nonlinear_reasons"].reshape(-1)
    return {
        "condition_id": condition_id, "family": family, "S": s_value,
        "sigma_beta": sigma_beta, "floor_ratio": floor_ratio, "input_sigma": input_sigma,
        "critical_real_per_s": _critical(j).real,
        "critical_frequency_hz": abs(_critical(j).imag) / (2 * np.pi),
        **derivative, "beta_variance_valid": beta_variance,
        "n_trials": N_TRIALS, "n_valid_common": n_valid,
        "valid_fraction": n_valid / N_TRIALS, "failure_reason": failure,
        "boundary_failure_fraction": float(np.mean(np.char.find(reasons.astype(str), "boundary") >= 0)),
        "multiple_peak_failure_fraction": float(np.mean(np.char.find(reasons.astype(str), "multiple") >= 0)),
        "residual_var_P": residual_cov[0, 0], "residual_var_A": residual_cov[1, 1],
        "residual_cov_PA": residual_cov[0, 1],
        "exact_mapping_r": exact_r, "local_first_order_r": local_r,
        "residual_corrected_r": corrected_r, "linear_residual_corrected_r": linear_corrected_r,
        "linear_heldout_r_mean": float(np.mean(np.asarray(heldout_linear_r)[np.isfinite(heldout_linear_r)])) if np.any(np.isfinite(heldout_linear_r)) else np.nan,
        "linear_phase_ensemble_r": ensemble_linear_r,
        "nonlinear_heldout_r_mean": phase_mean, "nonlinear_heldout_r_sd": phase_sd,
        "nonlinear_heldout_r_ci_low": phase_mean - tcrit * phase_sem,
        "nonlinear_heldout_r_ci_high": phase_mean + tcrit * phase_sem,
        "nonlinear_phase_ensemble_r": ensemble_nonlinear_r,
        "nonlinear_phase_ensemble_p": ensemble_nonlinear_p,
        "nonlinear_m_beta": nonlinear_peak_fit.slope if nonlinear_peak_fit else np.nan,
        "nonlinear_m_beta_ci_low": nonlinear_peak_fit.slope - 1.96 * nonlinear_peak_fit.stderr if nonlinear_peak_fit else np.nan,
        "nonlinear_m_beta_ci_high": nonlinear_peak_fit.slope + 1.96 * nonlinear_peak_fit.stderr if nonlinear_peak_fit else np.nan,
        "nonlinear_kappa_beta": nonlinear_slope_fit.slope if nonlinear_slope_fit else np.nan,
        "nonlinear_kappa_beta_ci_low": nonlinear_slope_fit.slope - 1.96 * nonlinear_slope_fit.stderr if nonlinear_slope_fit else np.nan,
        "nonlinear_kappa_beta_ci_high": nonlinear_slope_fit.slope + 1.96 * nonlinear_slope_fit.stderr if nonlinear_slope_fit else np.nan,
        "theory_in_nonlinear_ci": bool(
            np.isfinite(exact_r) and phase_mean - tcrit * phase_sem <= exact_r <= phase_mean + tcrit * phase_sem
        ),
        "heldout_phase_r_json": json.dumps(heldout_nonlinear_r),
        "heldout_phase_p_json": json.dumps(heldout_nonlinear_p),
    }, theory_values, theory_valid, common


def _bootstrap_regression(frame, predicted, actual, seed, n_boot=1000):
    use = frame[[predicted, actual]].dropna()
    x, y = use[predicted].to_numpy(), use[actual].to_numpy()
    if len(use) < 3 or np.std(y) < 1e-12:
        return {
            "prediction": predicted, "n": int(len(use)), "slope": np.nan,
            "intercept": np.nan, "R2": np.nan, "identity_R2": np.nan,
            "rmse": np.nan, "mae": np.nan, "regression_p": np.nan,
            "slope_ci_low": np.nan, "slope_ci_high": np.nan,
            "intercept_ci_low": np.nan, "intercept_ci_high": np.nan,
            "coverage": np.nan,
        }
    if np.std(x) < 1e-10:
        error = x - y
        identity_r2 = 1.0 - np.sum(error**2) / np.sum((y - y.mean()) ** 2)
        return {
            "prediction": predicted, "n": int(len(use)), "pearson_r": np.nan,
            "spearman_r": np.nan, "slope": np.nan, "intercept": np.nan,
            "R2": float(identity_r2), "MAE": float(np.mean(abs(error))),
            "RMSE": float(np.sqrt(np.mean(error**2))),
            "NRMSE": float(np.sqrt(np.mean(error**2)) / np.std(y)),
            "sign_accuracy": float(np.mean(np.sign(x) == np.sign(y))),
            "regression_p": np.nan, "slope_ci_low": np.nan, "slope_ci_high": np.nan,
            "intercept_ci_low": np.nan, "intercept_ci_high": np.nan, "coverage": np.nan,
        }
    fit = stats.linregress(x, y)
    rng = np.random.default_rng(seed)
    slopes = []
    intercepts = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(x), len(x))
        if np.std(x[idx]) == 0:
            continue
        item = stats.linregress(x[idx], y[idx])
        slopes.append(item.slope)
        intercepts.append(item.intercept)
    metrics = prediction_metrics(y, x)
    return {
        "prediction": predicted, **metrics,
        "regression_p": fit.pvalue,
        "slope_ci_low": float(np.quantile(slopes, 0.025)),
        "slope_ci_high": float(np.quantile(slopes, 0.975)),
        "intercept_ci_low": float(np.quantile(intercepts, 0.025)),
        "intercept_ci_high": float(np.quantile(intercepts, 0.975)),
        "coverage": float(frame.loc[use.index, "theory_in_nonlinear_ci"].mean()) if predicted == "exact_mapping_r" else np.nan,
    }


def _summaries(results, seed):
    valid = results[(results.n_valid_common >= 80) & (results.family == "main_grid")].copy()
    rows = []
    for predicted in ("exact_mapping_r", "local_first_order_r", "residual_corrected_r"):
        rows.append(_bootstrap_regression(valid, predicted, "nonlinear_heldout_r_mean", seed + len(rows)))
    rows.append(_bootstrap_regression(valid, "exact_mapping_r", "linear_heldout_r_mean", seed + 10))
    rows[-1]["prediction"] = "exact_mapping_vs_linear_time_domain"
    return pd.DataFrame(rows)


def _plots(out, results, metrics):
    plt.style.use("seaborn-v0_8-whitegrid")
    main = results[results.family == "main_grid"]
    reference = main[(main.sigma_beta == 0.40) & (main.floor_ratio == 0.10) & (main.input_sigma == MAIN_INPUT_SIGMA)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), constrained_layout=True)
    axes[0, 0].plot(reference.S, reference.critical_real_per_s, marker="o")
    axes[0, 0].axhline(0, color="0.2", ls=":")
    axes[0, 0].set(xlabel=r"I-to-E multiplier $s_{EI}$", ylabel=r"$\mathrm{Re}(\lambda)$ (s$^{-1}$)", title="Stable path")
    axes[0, 1].plot(reference.S, reference.critical_frequency_hz, marker="o", color="#D1495B")
    axes[0, 1].set(xlabel="S", ylabel="mode frequency [Hz]", title="Complex-mode frequency")
    axes[0, 2].plot(reference.S, reference.m_beta_implicit, marker="o", label="exact implicit")
    axes[0, 2].plot(reference.S, reference.nonlinear_m_beta, marker="s", label="nonlinear")
    axes[0, 2].set(xlabel=r"$s_{EI}$", ylabel=r"$m_\beta$ (Hz/$\beta$)", title="Peak sensitivity")
    axes[0, 2].legend(frameon=False, fontsize=8)
    axes[1, 0].plot(reference.S, reference.kappa_beta_analytic, marker="o", label="exact analytic")
    axes[1, 0].plot(reference.S, reference.nonlinear_kappa_beta, marker="s", label="nonlinear")
    axes[1, 0].set(xlabel=r"$s_{EI}$", ylabel=r"$\kappa_\beta$", title="Aperiodic sensitivity")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(reference.S, reference.exact_mapping_r, marker="o", label="exact mapping")
    axes[1, 1].plot(reference.S, reference.residual_corrected_r, marker="^", label="residual corrected")
    axes[1, 1].errorbar(reference.S, reference.nonlinear_heldout_r_mean, yerr=reference.nonlinear_heldout_r_sd, marker="s", label="nonlinear")
    axes[1, 1].set(xlabel=r"$s_{EI}$", ylabel=r"$r(P,A)$", title="P-A coupling along stable path")
    axes[1, 1].legend(frameon=False, fontsize=7)
    axes[1, 2].plot(reference.S, reference.valid_fraction, marker="o", label="common valid")
    axes[1, 2].plot(reference.S, reference.boundary_failure_fraction, marker="s", label="boundary")
    axes[1, 2].plot(reference.S, reference.multiple_peak_failure_fraction, marker="^", label="multiple peak")
    axes[1, 2].set(xlabel="S", ylabel="fraction", ylim=(0, 1.05), title="Peak identifiability")
    axes[1, 2].legend(frameon=False, fontsize=7)
    save_figure(fig, out / "figures" / "exp2_pap_path_validation")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    sigma_curve = main[(np.isclose(main.S, 3.1774487915806437)) & (main.floor_ratio == 0.10)]
    axes[0].plot(sigma_curve.sigma_beta, sigma_curve.exact_mapping_r, marker="o", label="exact mapping")
    axes[0].plot(sigma_curve.sigma_beta, sigma_curve.residual_corrected_r, marker="^", label="residual corrected")
    axes[0].errorbar(sigma_curve.sigma_beta, sigma_curve.nonlinear_heldout_r_mean, yerr=sigma_curve.nonlinear_heldout_r_sd, marker="s", label="nonlinear")
    axes[0].set(xlabel=r"$\sigma_\beta$", ylabel=r"$r(P,A)$", title="Input-colour variance attenuation")
    axes[0].legend(frameon=False, fontsize=8)
    floor_curve = main[(np.isclose(main.S, 3.1774487915806437)) & (main.sigma_beta == 0.40)]
    axes[1].plot(floor_curve.floor_ratio, floor_curve.kappa_beta_analytic, marker="o", label="kappa theory")
    axes[1].plot(floor_curve.floor_ratio, floor_curve.nonlinear_kappa_beta, marker="s", label="kappa nonlinear")
    axes[1].set(xlabel=r"$\sigma_{\mathrm{floor}}/\sigma_{\mathrm{colored}}$", ylabel=r"$\kappa_\beta$", title="White-floor attenuation")
    axes[1].legend(frameon=False, fontsize=8)
    save_figure(fig, out / "figures" / "exp2_pap_variance_floor")
    plt.close(fig)

    valid = main[main.n_valid_common >= 80]
    if valid.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.1), constrained_layout=True)
    for ax, column, title in zip(
        axes,
        ("exact_mapping_r", "local_first_order_r", "residual_corrected_r"),
        ("Exact beta mapping", "Local first order", "Residual corrected"),
    ):
        x, y = valid[column], valid.nonlinear_heldout_r_mean
        ax.scatter(x, y, s=18, alpha=0.7)
        lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        row = metrics[metrics.prediction == column].iloc[0]
        annotation = (
            f"slope={row.slope:.2f}, p={row.regression_p:.2g}, identity R2={row.R2:.2f}"
            if np.isfinite(row.slope)
            else f"single-latent prediction is constant; identity R2={row.R2:.2f}"
        )
        ax.set(xlabel="theory r", ylabel="nonlinear held-out r", title=f"{title}\n{annotation}")
    save_figure(fig, out / "figures" / "exp2_pap_theory_simulation_identity")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)
    axes[0].scatter(valid.exact_mapping_r, valid.linear_heldout_r_mean, s=18, label="linear time-domain")
    axes[0].scatter(valid.exact_mapping_r, valid.nonlinear_heldout_r_mean, s=18, label="nonlinear time-domain")
    lo = min(valid.exact_mapping_r.min(), valid.linear_heldout_r_mean.min(), valid.nonlinear_heldout_r_mean.min())
    hi = max(valid.exact_mapping_r.max(), valid.linear_heldout_r_mean.max(), valid.nonlinear_heldout_r_mean.max())
    axes[0].plot([lo, hi], [lo, hi], "k--", lw=1)
    axes[0].set(xlabel="exact ideal-spectrum r", ylabel="time-domain r", title="Observation and nonlinearity error")
    axes[0].legend(frameon=False)
    errors = pd.DataFrame(
        {
            "linear finite window": abs(valid.linear_heldout_r_mean - valid.exact_mapping_r),
            "nonlinear finite window": abs(valid.nonlinear_heldout_r_mean - valid.exact_mapping_r),
            "nonlinear phase ensemble": abs(valid.nonlinear_phase_ensemble_r - valid.exact_mapping_r),
        }
    )
    axes[1].boxplot([errors[column] for column in errors], tick_labels=list(errors.columns), showfliers=False)
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].set(ylabel="absolute r error", title="Error decomposition")
    save_figure(fig, out / "figures" / "exp2_pap_error_decomposition")
    plt.close(fig)


def _representative_plot(out, original, tau, conditions, results):
    candidates = results[
        (results.family == "main_grid")
        & np.isclose(results.S, 6.0)
        & np.isclose(results.sigma_beta, 0.40)
        & np.isclose(results.floor_ratio, 0.10)
        & (results.n_valid_common >= 80)
    ]
    if candidates.empty:
        return
    row = candidates.iloc[0]
    condition_id = int(row.condition_id)
    cached = np.load(out / "condition_features" / f"condition_{condition_id:04d}.npz", allow_pickle=True)
    condition = conditions[condition_id]
    _, theory, _, common = _analyze_condition(original, tau, condition_id, condition, cached)
    betas = cached["betas"][common]
    nonlinear = cached["nonlinear"][N_CALIBRATION_PHASES:, common].mean(axis=0)
    theory = theory[common]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), constrained_layout=True)
    panels = (
        (betas, nonlinear[:, 0], theory[:, 0], "Input colour beta", "Peak P [Hz]"),
        (betas, nonlinear[:, 1], theory[:, 1], "Input colour beta", "Aperiodic slope A"),
        (nonlinear[:, 0], nonlinear[:, 1], None, "Peak P [Hz]", "Aperiodic slope A"),
    )
    for ax, (x, y, theory_y, xlabel, ylabel) in zip(axes, panels):
        ax.scatter(x, y, s=18, alpha=0.55, label="nonlinear held-out mean")
        if theory_y is not None:
            order = np.argsort(x)
            ax.plot(x[order], theory_y[order], color="#D1495B", lw=1.8, label="exact spectrum")
        else:
            fit = stats.linregress(x, y)
            xx = np.linspace(x.min(), x.max(), 100)
            ax.plot(xx, fit.intercept + fit.slope * xx, color="#D1495B", lw=1.8)
            association = stats.pearsonr(x, y)
            ax.text(
                0.04, 0.96, f"r={association.statistic:.3f}, p={association.pvalue:.2g}",
                transform=ax.transAxes, va="top",
            )
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.legend(frameon=False, fontsize=7, loc="best") if theory_y is not None else None
    fig.suptitle(r"Representative stable-filter trials: $s_{EI}=6.0$, $\sigma_\beta=0.4$, floor=0.1")
    save_figure(fig, out / "figures" / "exp2_pap_representative_trials")
    plt.close(fig)


def run(cfg, force=False, max_conditions=None):
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    original, tau, _, _ = _setup()
    conditions = _condition_grid()
    if max_conditions is not None:
        conditions = conditions[:max_conditions]
    rows = []
    for condition_id, condition in enumerate(conditions):
        cached = _simulate_condition(cfg, out, original, tau, condition_id, condition, force=force)
        row, _, _, _ = _analyze_condition(original, tau, condition_id, condition, cached)
        rows.append(row)
        if (condition_id + 1) % 3 == 0 or condition_id + 1 == len(conditions):
            pd.DataFrame(rows).to_csv(out / "condition_results_partial.csv", index=False)
            print(f"Experiment 2 P-A condition {condition_id + 1}/{len(conditions)}", flush=True)
    results = pd.DataFrame(rows)
    results.to_csv(out / "condition_results.csv", index=False)
    metrics = _summaries(results, int(cfg["seed"]) + 6900000)
    metrics.to_csv(out / "prediction_metrics.csv", index=False)
    _plots(out, results, metrics)
    _representative_plot(out, original, tau, _condition_grid(), results)
    no_floor = results[(results.family == "main_grid") & (results.floor_ratio == 0.0)]
    save_json(
        out / "manifest.json",
        {
            "experiment": "quantitative_stable_filter_P_aperiodic_correlation_validation",
            "S_values": S_VALUES, "sigma_beta_values": SIGMA_BETA_VALUES,
            "floor_ratios": FLOOR_RATIOS, "main_input_sigma": MAIN_INPUT_SIGMA,
            "amplitude_controls": AMPLITUDE_CONTROLS, "n_trials": N_TRIALS,
            "calibration_phases": N_CALIBRATION_PHASES, "heldout_phases": N_HELDOUT_PHASES,
            "linear_time_domain_controls": "sigma_beta=0.40 and floor_ratio=0.10 at every S and input amplitude",
            "dt_s": DT_S, "burn_s": BURN_S, "analysis_s": ANALYSIS_S,
            "welch_segment_s": 4.096, "peak_band_hz": PEAK_BAND,
            "slope_band_hz": SLOPE_BAND,
            "peak_validity": "highest interior peak; competing Welch ripples retained and reported as QC",
            "median_no_floor_kappa_beta": float(no_floor.kappa_beta_analytic.median()),
            "max_analytic_finite_m_relative_error": float(results.m_relative_error.max()),
            "max_analytic_finite_kappa_relative_error": float(results.kappa_relative_error.max()),
            "runtime_s": time.time() - started,
        },
    )
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-conditions", type=int)
    args = parser.parse_args()
    print(run(load_config(args.config), force=args.force, max_conditions=args.max_conditions))


if __name__ == "__main__":
    main()
