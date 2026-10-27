from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats

from .core import feature_from_spectrum, prediction_metrics, save_figure, save_json
from .exp2_pap_correlation_validation import (
    BURN_S,
    DT_S,
    MAIN_INPUT_SIGMA,
    N_CALIBRATION_PHASES,
    N_HELDOUT_PHASES,
    N_TRIALS,
    PEAK_BAND,
    SLOPE_BAND,
    _beta_trials,
    _condition_grid,
    _forced_ei_batch_parallel,
    _input_q,
    _jacobian,
    _matrix,
    _normalized_input_shape,
    _setup,
    _theory_feature,
    _transfer_power,
)
from .io import load_config, prepare_output


OUTPUT_NAME = "exp2_window_second_order_validation"
SOURCE_NAME = "exp2_pap_correlation_validation"
REPRESENTATIVE_CONDITION_ID = 245
WINDOW_PHASES = 3
WINDOW_CHUNK_SIZE = 32
MAX_ANALYSIS_S = 131.072
WINDOW_SPECS = (
    ("T32_L4", 32.768, 4.096),
    ("T65_L4", 65.536, 4.096),
    ("T131_L4", 131.072, 4.096),
    ("T65_L8", 65.536, 8.192),
    ("T131_L8", 131.072, 8.192),
    ("T131_L16", 131.072, 16.384),
)


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def _pearson(values: np.ndarray) -> tuple[float, float]:
    if len(values) < 8 or np.std(values[:, 0]) == 0 or np.std(values[:, 1]) == 0:
        return np.nan, np.nan
    result = stats.pearsonr(values[:, 0], values[:, 1])
    return float(result.statistic), float(result.pvalue)


def _pooled_within_trial_covariance(values: np.ndarray) -> np.ndarray:
    """Unbiased covariance of phase residuals after a separate mean per trial."""
    n_phases, n_trials, _ = values.shape
    centered = values - values.mean(axis=0, keepdims=True)
    return np.einsum("pti,ptj->ij", centered, centered) / (n_trials * (n_phases - 1))


def _correlation_with_residual(signal_values: np.ndarray, residual_covariance: np.ndarray) -> float:
    total = np.cov(signal_values, rowvar=False, ddof=1) + residual_covariance
    return float(total[0, 1] / np.sqrt(max(total[0, 0] * total[1, 1], 1e-30)))


def _second_order_features(j, tau_e, input_sigma, floor_ratio, betas):
    frequencies = np.geomspace(0.5, 1200.0, 1600)
    transfer = _transfer_power(j, tau_e, frequencies)
    h = 0.01
    minus, minus_valid, _ = _theory_feature(transfer, frequencies, -h, input_sigma, floor_ratio)
    base, base_valid, _ = _theory_feature(transfer, frequencies, 0.0, input_sigma, floor_ratio)
    plus, plus_valid, _ = _theory_feature(transfer, frequencies, h, input_sigma, floor_ratio)
    first = (plus - minus) / (2.0 * h)
    second = (plus - 2.0 * base + minus) / h**2
    first_values = base[None, :] + betas[:, None] * first[None, :]
    second_values = first_values + 0.5 * betas[:, None] ** 2 * second[None, :]
    exact_values = np.empty((len(betas), 2))
    exact_valid = np.empty(len(betas), dtype=bool)
    for trial, beta in enumerate(betas):
        exact_values[trial], exact_valid[trial], _ = _theory_feature(
            transfer, frequencies, beta, input_sigma, floor_ratio
        )
    return {
        "base": base,
        "first_derivative": first,
        "second_derivative": second,
        "derivative_valid": bool(minus_valid and base_valid and plus_valid),
        "first_values": first_values,
        "second_values": second_values,
        "exact_values": exact_values,
        "exact_valid": exact_valid,
    }


def _second_order_grid(cfg, out, force=False):
    cache = out / "second_order_condition_results.csv"
    representative_cache = out / "second_order_representative.npz"
    if cache.exists() and representative_cache.exists() and not force:
        return pd.read_csv(cache), np.load(representative_cache)
    original, tau, _, _ = _setup()
    source = _rooted(cfg["output_dir"]) / SOURCE_NAME
    rows = []
    representative = {}
    conditions = _condition_grid()
    for condition_id, condition in enumerate(conditions):
        s_value, sigma_beta, floor_ratio, input_sigma, family = condition
        path = source / "condition_features" / f"condition_{condition_id:04d}.npz"
        cached = np.load(path, allow_pickle=True)
        betas = cached["betas"]
        nonlinear = cached["nonlinear"]
        nonlinear_valid = cached["nonlinear_valid"]
        j = _jacobian(original, tau, s_value, 1.0)
        theory = _second_order_features(j, float(tau[0]), input_sigma, floor_ratio, betas)
        common = theory["exact_valid"] & nonlinear_valid.all(axis=0)
        n_valid = int(common.sum())
        if n_valid < 80 or not theory["derivative_valid"]:
            rows.append(
                {
                    "condition_id": condition_id, "family": family, "S": s_value,
                    "sigma_beta": sigma_beta, "floor_ratio": floor_ratio,
                    "input_sigma": input_sigma, "n_valid_common": n_valid,
                    "failure_reason": "insufficient_common_valid_or_derivative",
                }
            )
            continue
        calibration = nonlinear[:N_CALIBRATION_PHASES, common]
        residual_cov = _pooled_within_trial_covariance(calibration)
        beta = betas[common]
        exact = theory["exact_values"][common]
        first = theory["first_values"][common]
        second = theory["second_values"][common]
        heldout_rs = [
            _pearson(nonlinear[phase, common])[0]
            for phase in range(N_CALIBRATION_PHASES, N_CALIBRATION_PHASES + N_HELDOUT_PHASES)
        ]
        heldout = nonlinear[N_CALIBRATION_PHASES:, common].mean(axis=0)
        exact_r, _ = _pearson(exact)
        first_r, _ = _pearson(first)
        second_r, _ = _pearson(second)
        first_corrected = _correlation_with_residual(first, residual_cov)
        second_corrected = _correlation_with_residual(second, residual_cov)
        noise_only = float(
            residual_cov[0, 1]
            / np.sqrt(max(residual_cov[0, 0] * residual_cov[1, 1], 1e-30))
        )
        row = {
            "condition_id": condition_id, "family": family, "S": s_value,
            "sigma_beta": sigma_beta, "floor_ratio": floor_ratio,
            "input_sigma": input_sigma, "n_valid_common": n_valid, "failure_reason": "",
            "P_first_derivative": theory["first_derivative"][0],
            "A_first_derivative": theory["first_derivative"][1],
            "P_second_derivative": theory["second_derivative"][0],
            "A_second_derivative": theory["second_derivative"][1],
            "residual_var_P_unbiased": residual_cov[0, 0],
            "residual_var_A_unbiased": residual_cov[1, 1],
            "residual_cov_PA_unbiased": residual_cov[0, 1],
            "noise_only_r": noise_only, "exact_mapping_r": exact_r,
            "first_order_r": first_r, "second_order_r": second_r,
            "first_order_residual_r": first_corrected,
            "second_order_residual_r": second_corrected,
            "nonlinear_heldout_r_mean": float(np.nanmean(heldout_rs)),
            "nonlinear_heldout_r_sd": float(np.nanstd(heldout_rs, ddof=1)),
            "nonlinear_phase_ensemble_r": _pearson(heldout)[0],
            "heldout_phase_r_json": json.dumps(heldout_rs),
        }
        rows.append(row)
        if condition_id == REPRESENTATIVE_CONDITION_ID:
            representative = {
                "betas": beta, "simulation": heldout, "exact": exact,
                "first": first, "second": second, "residual_cov": residual_cov,
            }
    result = pd.DataFrame(rows)
    result.to_csv(cache, index=False)
    np.savez_compressed(representative_cache, **representative)
    return result, np.load(representative_cache)


def _variable_forcing(betas, analysis_s, input_sigma, floor_ratio, seed):
    analysis_steps = int(round(analysis_s / DT_S))
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


def _extract_window_features(paths, analysis_s, segment_s):
    steps = int(round(analysis_s / DT_S))
    nperseg = int(round(segment_s / DT_S))
    frequencies, psd = signal.welch(
        paths[:, :steps], fs=1.0 / DT_S, window="hann", nperseg=nperseg,
        noverlap=nperseg // 2, detrend="constant", scaling="density", axis=-1,
    )
    keep = (frequencies >= 0.5) & (frequencies <= 1200.0)
    f = frequencies[keep]
    sigma_bins = 0.9765625 / (f[1] - f[0])
    smooth = np.exp(
        ndimage.gaussian_filter1d(
            np.log(np.maximum(psd[:, keep], 1e-300)), sigma_bins, axis=-1
        )
    )
    values = np.empty((len(paths), 2))
    valid = np.empty(len(paths), dtype=bool)
    for trial in range(len(paths)):
        feature = feature_from_spectrum(
            f, smooth[trial], peak_band=PEAK_BAND, slope_band=SLOPE_BAND
        )
        values[trial] = feature["peak_hz"], -feature["exponent"]
        valid[trial] = bool(
            not feature["peak_boundary"]
            and "no_interior_peak" not in feature["feature_failure_reason"]
        )
    return values, valid


def _window_scan(cfg, out, representative, force=False):
    cache = out / "window_features.npz"
    metric_cache = out / "window_length_results.csv"
    if cache.exists() and metric_cache.exists() and not force:
        return np.load(cache), pd.read_csv(metric_cache)
    original, tau, _, _ = _setup()
    condition = _condition_grid()[REPRESENTATIVE_CONDITION_ID]
    s_value, sigma_beta, floor_ratio, input_sigma, _ = condition
    betas = _beta_trials(
        sigma_beta, int(cfg["seed"]) + 6100000 + int(round(sigma_beta * 1000))
    )
    w = _matrix(original, s_value, 1.0)
    n_specs = len(WINDOW_SPECS)
    features = np.empty((n_specs, WINDOW_PHASES, N_TRIALS, 2))
    valid = np.empty((n_specs, WINDOW_PHASES, N_TRIALS), dtype=bool)
    for phase in range(WINDOW_PHASES):
        for start in range(0, N_TRIALS, WINDOW_CHUNK_SIZE):
            stop = min(start + WINDOW_CHUNK_SIZE, N_TRIALS)
            forcing = _variable_forcing(
                betas[start:stop], MAX_ANALYSIS_S, input_sigma, floor_ratio,
                int(cfg["seed"]) + 7300000 + phase * 100000 + start,
            )
            paths = _forced_ei_batch_parallel(
                w, tau, forcing, DT_S, int(round(BURN_S / DT_S))
            )
            for spec_pos, (_, analysis_s, segment_s) in enumerate(WINDOW_SPECS):
                values, is_valid = _extract_window_features(paths, analysis_s, segment_s)
                features[spec_pos, phase, start:stop] = values
                valid[spec_pos, phase, start:stop] = is_valid
        print(f"Experiment 2 window scan phase {phase + 1}/{WINDOW_PHASES}", flush=True)
    np.savez_compressed(
        cache, features=features, valid=valid, betas=betas,
        spec_names=np.asarray([item[0] for item in WINDOW_SPECS]),
    )
    exact = representative["exact"]
    # The representative cache uses the same beta seed and all trials are valid.
    rows = []
    for spec_pos, (name, analysis_s, segment_s) in enumerate(WINDOW_SPECS):
        common = valid[spec_pos].all(axis=0)
        common_features = features[spec_pos][:, common, :]
        averaged = common_features.mean(axis=0)
        beta = betas[common]
        exact_common = exact[common]
        residual_cov = _pooled_within_trial_covariance(common_features)
        phase_rs = [_pearson(common_features[phase])[0] for phase in range(WINDOW_PHASES)]
        rows.append(
            {
                "spec": name, "analysis_s": analysis_s, "welch_segment_s": segment_s,
                "n_welch_segments": int(1 + (analysis_s - segment_s) / (segment_s / 2.0)),
                "n_valid_common": int(common.sum()),
                "phase_ensemble_r_PA": _pearson(averaged)[0],
                "phase_mean_r_PA": float(np.mean(phase_rs)),
                "phase_sd_r_PA": float(np.std(phase_rs, ddof=1)),
                "r_beta_P": float(stats.pearsonr(beta, averaged[:, 0]).statistic),
                "r_beta_A": float(stats.pearsonr(beta, averaged[:, 1]).statistic),
                "P_theory_simulation_RMSE_hz": float(np.sqrt(np.mean((averaged[:, 0] - exact_common[:, 0]) ** 2))),
                "P_theory_simulation_bias_hz": float(np.mean(averaged[:, 0] - exact_common[:, 0])),
                "A_theory_simulation_RMSE": float(np.sqrt(np.mean((averaged[:, 1] - exact_common[:, 1]) ** 2))),
                "residual_sd_P_hz": float(np.sqrt(residual_cov[0, 0])),
                "residual_sd_A": float(np.sqrt(residual_cov[1, 1])),
                "residual_cov_PA": float(residual_cov[0, 1]),
                "exact_r_same_trials": _pearson(exact_common)[0],
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(metric_cache, index=False)
    return np.load(cache), metrics


def _identity_metrics(results):
    valid = results[(results.family == "main_grid") & (results.n_valid_common >= 80)].copy()
    rows = []
    target = valid.nonlinear_heldout_r_mean.to_numpy()
    for prediction in (
        "exact_mapping_r", "first_order_r", "second_order_r", "noise_only_r",
        "first_order_residual_r", "second_order_residual_r",
    ):
        use = np.isfinite(valid[prediction]) & np.isfinite(target)
        x, y = valid.loc[use, prediction].to_numpy(), target[use]
        if np.std(x) > 1e-12:
            metrics = prediction_metrics(y, x)
            fit = stats.linregress(x, y)
            slope, intercept, pvalue = fit.slope, fit.intercept, fit.pvalue
        else:
            error = x - y
            rmse = float(np.sqrt(np.mean(error**2)))
            metrics = {
                "pearson_r": np.nan, "spearman_r": np.nan,
                "R2": float(1.0 - np.sum(error**2) / np.sum((y - y.mean()) ** 2)),
                "slope": np.nan, "intercept": np.nan,
                "MAE": float(np.mean(np.abs(error))), "RMSE": rmse,
                "NRMSE": float(rmse / np.std(y)),
                "sign_accuracy": float(np.mean(np.sign(y) == np.sign(x))),
                "n": int(len(y)),
            }
            slope = intercept = pvalue = np.nan
        rows.append(
            {
                "prediction": prediction, "n": int(use.sum()), **metrics,
                "slope": slope, "intercept": intercept, "regression_p": pvalue,
            }
        )
    return pd.DataFrame(rows)


def _plot_second_order(out, results, representative, metrics):
    plt.style.use("seaborn-v0_8-whitegrid")
    beta = representative["betas"]
    order = np.argsort(beta)
    simulation = representative["simulation"]
    exact = representative["exact"]
    first = representative["first"]
    second = representative["second"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), constrained_layout=True)
    for values, label, color, style in (
        (exact, "exact spectrum", "#D1495B", "-"),
        (first, "first order", "#6A4C93", "--"),
        (second, "second order", "#F28E2B", "-."),
    ):
        axes[0].plot(beta[order], values[order, 0], color=color, ls=style, lw=1.8, label=label)
        axes[1].plot(beta[order], values[order, 1], color=color, ls=style, lw=1.8, label=label)
        axes[2].plot(values[order, 0], values[order, 1], color=color, ls=style, lw=1.8, label=label)
    axes[0].scatter(beta, simulation[:, 0], s=16, alpha=0.45, color="#247BA0", label="nonlinear held-out mean")
    axes[1].scatter(beta, simulation[:, 1], s=16, alpha=0.45, color="#247BA0")
    axes[2].scatter(simulation[:, 0], simulation[:, 1], s=16, alpha=0.45, color="#247BA0")
    axes[0].set(xlabel="input colour beta", ylabel="Peak P [Hz]", title="beta -> P")
    axes[1].set(xlabel="input colour beta", ylabel="Aperiodic slope A", title="beta -> A")
    axes[2].set(xlabel="Peak P [Hz]", ylabel="Aperiodic slope A", title=f"P-A: simulation r={_pearson(simulation)[0]:.3f}")
    axes[0].legend(frameon=False, fontsize=7)
    axes[2].legend(frameon=False, fontsize=7)
    fig.suptitle("S31-style second-order comparison: S=6.0, sigma_beta=0.4, floor=0.1")
    save_figure(fig, out / "figures" / "exp2_second_order_representative")
    plt.close(fig)

    valid = results[(results.family == "main_grid") & (results.n_valid_common >= 80)]
    columns = (
        ("exact_mapping_r", "Exact mapping"),
        ("first_order_r", "First order"),
        ("second_order_r", "Second order"),
        ("first_order_residual_r", "First order + residual"),
        ("second_order_residual_r", "Second order + residual"),
        ("noise_only_r", "Residual only"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 8.0), constrained_layout=True)
    for ax, (column, title) in zip(axes.flat, columns):
        x = valid[column].to_numpy()
        y = valid.nonlinear_heldout_r_mean.to_numpy()
        use = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[use], y[use], s=14, alpha=0.55)
        lo, hi = min(x[use].min(), y[use].min()), max(x[use].max(), y[use].max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        row = metrics[metrics.prediction == column].iloc[0]
        annotation = (
            f"n={int(row.n)}, r={row.pearson_r:.2f}, id R2={row.R2:.2f}\n"
            f"slope={row.slope:.2f}, p={row.regression_p:.2g}"
            if np.isfinite(row.slope)
            else f"n={int(row.n)}, constant prediction\nid R2={row.R2:.2f}"
        )
        ax.set(xlabel="prediction r", ylabel="nonlinear held-out r", title=f"{title}\n{annotation}")
    fig.suptitle("S32-style held-out comparison across Experiment 2 conditions")
    save_figure(fig, out / "figures" / "exp2_second_order_identity")
    plt.close(fig)


def _plot_windows(out, window_data, metrics, representative):
    plt.style.use("seaborn-v0_8-whitegrid")
    features = window_data["features"]
    valid = window_data["valid"]
    betas = window_data["betas"]
    exact = representative["exact"]
    chosen = [0, 1, 2]
    fig, axes = plt.subplots(3, 3, figsize=(13, 11), constrained_layout=True)
    for row_pos, spec_pos in enumerate(chosen):
        name, analysis_s, segment_s = WINDOW_SPECS[spec_pos]
        common = valid[spec_pos].all(axis=0)
        averaged = features[spec_pos][:, common, :].mean(axis=0)
        beta = betas[common]
        theory = exact[common]
        order = np.argsort(beta)
        axes[row_pos, 0].scatter(beta, averaged[:, 0], s=12, alpha=0.45)
        axes[row_pos, 0].plot(beta[order], theory[order, 0], color="#D1495B", lw=1.8)
        axes[row_pos, 1].scatter(beta, averaged[:, 1], s=12, alpha=0.45)
        axes[row_pos, 1].plot(beta[order], theory[order, 1], color="#D1495B", lw=1.8)
        axes[row_pos, 2].scatter(averaged[:, 0], averaged[:, 1], s=12, alpha=0.45)
        fit = stats.linregress(averaged[:, 0], averaged[:, 1])
        xx = np.linspace(averaged[:, 0].min(), averaged[:, 0].max(), 100)
        axes[row_pos, 2].plot(xx, fit.intercept + fit.slope * xx, color="#D1495B", lw=1.8)
        axes[row_pos, 0].set(ylabel=f"P [Hz]\nT={analysis_s:.3f}s", title="beta -> P")
        axes[row_pos, 1].set(ylabel="A", title="beta -> A")
        axes[row_pos, 2].set(ylabel="A", title=f"P-A: r={fit.rvalue:.3f}, p={fit.pvalue:.2g}")
        for col in range(3):
            axes[row_pos, col].set_xlabel("beta" if col < 2 else "P [Hz]")
    fig.suptitle("S31-style window-length comparison (Welch segment=4.096 s; 3-phase mean)")
    save_figure(fig, out / "figures" / "exp2_window_length_representative")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6), constrained_layout=True)
    for segment_s, group in metrics.groupby("welch_segment_s"):
        group = group.sort_values("analysis_s")
        label = f"Welch L={segment_s:g}s"
        axes[0, 0].plot(group.analysis_s, group.phase_ensemble_r_PA, marker="o", label=label)
        axes[0, 1].plot(group.analysis_s, group.residual_sd_P_hz, marker="o", label=label)
        axes[1, 0].plot(group.analysis_s, group.P_theory_simulation_RMSE_hz, marker="o", label=label)
        axes[1, 1].plot(group.analysis_s, group.r_beta_P, marker="o", label=label)
    axes[0, 0].axhline(metrics.exact_r_same_trials.iloc[0], color="#D1495B", ls="--", label="exact r")
    axes[0, 0].set(xlabel="analysis length [s]", ylabel="r(P,A)", title="Correlation convergence")
    axes[0, 1].set(xlabel="analysis length [s]", ylabel="residual SD(P) [Hz]", title="Finite-window peak noise")
    axes[1, 0].set(xlabel="analysis length [s]", ylabel="P RMSE vs exact [Hz]", title="Theory-simulation peak error")
    axes[1, 1].set(xlabel="analysis length [s]", ylabel="r(beta,P)", title="Input-to-peak reliability")
    for ax in axes.flat:
        ax.legend(frameon=False, fontsize=8)
    save_figure(fig, out / "figures" / "exp2_window_length_summary")
    plt.close(fig)


def run(cfg, force=False):
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    second_order, representative = _second_order_grid(cfg, out, force=force)
    metrics = _identity_metrics(second_order)
    metrics.to_csv(out / "second_order_prediction_metrics.csv", index=False)
    window_data, window_metrics = _window_scan(cfg, out, representative, force=force)
    _plot_second_order(out, second_order, representative, metrics)
    _plot_windows(out, window_data, window_metrics, representative)
    save_json(
        out / "manifest.json",
        {
            "experiment": "Experiment 2 window-length and beta-second-order validation",
            "representative_condition_id": REPRESENTATIVE_CONDITION_ID,
            "window_phases": WINDOW_PHASES,
            "window_specs": [
                {"name": name, "analysis_s": analysis_s, "welch_segment_s": segment_s}
                for name, analysis_s, segment_s in WINDOW_SPECS
            ],
            "second_order": "Taylor expansion in beta around beta=0 using h=0.01",
            "residual_covariance": "condition-specific unbiased pooled within-trial covariance from three calibration phases",
            "heldout_target": "mean of five independent phase-wise nonlinear correlations",
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
