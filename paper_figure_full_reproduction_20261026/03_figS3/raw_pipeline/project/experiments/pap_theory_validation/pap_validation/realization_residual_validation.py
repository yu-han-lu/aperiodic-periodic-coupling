from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import (
    AREAS9,
    apply_fln_delta,
    finite_derivatives,
    frequency_grid,
    load_model,
    network_features,
    prediction_metrics,
    save_figure,
    save_json,
    transfer_power,
)
from .exp1 import Z0, ZHI, ZLO, ZSCALE, _draw_design, _feature_fun
from .exp3_emergent_correlation_regime import (
    EPSILON,
    INPUT_AREA,
    INPUT_CORRELATION,
    N_TRIALS,
    OUTPUT_AREA,
    _frozen_mask,
    _trial_design,
)
from .exp3_large_delta_screen import _continuous_transfer, _discrete_transfer, _exact_trial_features
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .nonlinear_network_validation import _exp1_conditions, _run_multioutput_batch
from .pairwise_nonlinear_window_validation import _fit_metrics, _rooted, _taylor_values
from .exp3_dt_convergence import _matched_bandlimited_inputs


OUTPUT_NAME = "realization_residual_validation"
CALIBRATION_PHASES = 3
HELDOUT_PHASES = 5
TOTAL_PHASES = CALIBRATION_PHASES + HELDOUT_PHASES
EXP1_TRIALS = 64
EXP1_ANALYSIS_S = 32.768
EXP3_ANALYSIS_S = 65.536
WELCH_SEGMENT_S = 4.096
SMOOTHING_SIGMA_HZ = 0.9765625
DT_MS = 0.25
CHUNK_SIZE_EXP1 = 16
CHUNK_SIZE_EXP3 = 32


def _spectral_spec(analysis_s: float) -> dict:
    return {
        "long": {
            "analysis_s": analysis_s,
            "spectral": {
                "mode": "hann_welch",
                "segment_s": WELCH_SEGMENT_S,
                "smoothing_sigma_hz": SMOOTHING_SIGMA_HZ,
            },
        }
    }


def _corr(values: np.ndarray) -> float:
    values = np.asarray(values, float)
    if len(values) < 5 or not np.isfinite(values).all() or np.min(np.std(values, axis=0)) == 0:
        return np.nan
    return float(stats.pearsonr(values[:, 0], values[:, 1]).statistic)


def _pooled_within_trial_cov(values: np.ndarray) -> np.ndarray:
    """Unbiased covariance of realization error after centering within each trial."""
    values = np.asarray(values, float)
    q, n, p = values.shape
    if q < 2 or n < 2:
        return np.full((p, p), np.nan)
    centered = values - values.mean(axis=0, keepdims=True)
    return np.einsum("qni,qnj->ij", centered, centered) / (n * (q - 1))


def _cov_to_corr(cov: np.ndarray) -> float:
    if not np.isfinite(cov).all() or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return np.nan
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def _corrected_corr(signal: np.ndarray, residual_cov: np.ndarray) -> tuple[float, float, np.ndarray]:
    signal_cov = np.cov(np.asarray(signal, float), rowvar=False, ddof=1)
    total_cov = signal_cov + residual_cov
    return _cov_to_corr(signal_cov), _cov_to_corr(total_cov), signal_cov


def _identity_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    result = _fit_metrics(np.asarray(observed, float), np.asarray(predicted, float))
    return {
        "n": int(result["n"]),
        "identity_R2": float(result["R2"]),
        "ols_R2": float(result["ols_R2"]),
        "slope": float(result["slope"]),
        "intercept": float(result["intercept"]),
        "RMSE": float(result["RMSE"]),
        "pearson_r": float(result["pearson_r"]),
        "regression_p": float(result["regression_p_value"]),
    }


def _exp1_designs(cfg: dict) -> tuple[list[dict], np.ndarray, np.ndarray]:
    conditions = []
    all_u = np.empty((len(_exp1_conditions(cfg)), EXP1_TRIALS, 4), dtype=float)
    all_z = np.empty_like(all_u)
    for condition_pos, (design, eps, rho) in enumerate(_exp1_conditions(cfg)):
        rng = np.random.default_rng(int(cfg["seed"]) + 1000 + condition_pos)
        u = _draw_design(design, EXP1_TRIALS, eps, rng, rho)
        raw = Z0 + u * ZSCALE
        z = np.clip(raw, ZLO, ZHI)
        all_u[condition_pos] = (z - Z0) / ZSCALE
        all_z[condition_pos] = z
        conditions.append(
            {
                "condition_pos": condition_pos,
                "condition": f"{design}_eps{eps:g}_rho{rho:g}",
                "design": design,
                "epsilon": float(eps),
                "rho": float(rho),
                "n_clipped": int(np.any(raw != z, axis=1).sum()),
            }
        )
    return conditions, all_u, all_z


def _simulate_exp1_phase_input(
    cfg: dict,
    model,
    transfers: np.ndarray,
    all_z: np.ndarray,
    phase: int,
    input_pos: int,
    cache: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        data = np.load(cache)
        return data["features"], data["valid"]
    specs = _spectral_spec(EXP1_ANALYSIS_S)
    n_steps = int(round(EXP1_ANALYSIS_S * 1000.0 / DT_MS))
    burn_s = float(cfg["exp1_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    sigma = float(cfg["exp1_nonlinear"]["input_sigma"])
    features = np.empty((len(all_z), len(AREAS9), EXP1_TRIALS, 2), dtype=np.float32)
    valid = np.empty((len(all_z), len(AREAS9), EXP1_TRIALS), dtype=bool)
    for condition_pos, z in enumerate(all_z):
        for start in range(0, EXP1_TRIALS, CHUNK_SIZE_EXP1):
            stop = min(start + CHUNK_SIZE_EXP1, EXP1_TRIALS)
            inputs = _matched_bandlimited_inputs(
                z[start:stop], DT_MS, EXP1_ANALYSIS_S, burn_s, sigma,
                int(cfg["seed"]) + 8100000 + phase * 100000 + input_pos * 10000 + condition_pos,
                trial_offset=start,
            )
            input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
            outputs = np.fft.irfft(
                input_fft[:, None, :] * transfers[input_pos][None, :, :],
                n=n_steps,
                axis=-1,
            ).astype(np.float32)
            values, is_valid = _feature_arrays_vectorized(outputs, DT_MS, specs)["long"]
            features[condition_pos, :, start:stop] = values
            valid[condition_pos, :, start:stop] = is_valid
        print(
            f"Experiment 1 realization phase {phase + 1}/{TOTAL_PHASES}, "
            f"input {AREAS9[input_pos]}: condition {condition_pos + 1}/{len(all_z)}",
            flush=True,
        )
    np.savez_compressed(cache, features=features, valid=valid)
    return features, valid


def _exp1_theory(cfg: dict, model, all_u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cache = _rooted(cfg["output_dir"]) / OUTPUT_NAME / "exp1_theory_maps.npz"
    if cache.exists():
        data = np.load(cache)
        return data["first"], data["second"]
    freqs = frequency_grid()
    first = np.empty((9, len(all_u), 9, EXP1_TRIALS, 2), dtype=float)
    second = np.empty_like(first)
    for input_pos, input_area in enumerate(AREAS9):
        input_idx = model.areas.index(input_area)
        for output_pos, output_area in enumerate(AREAS9):
            output_idx = model.areas.index(output_area)
            transfer = transfer_power(model.J, input_idx, output_idx, freqs)
            fun = _feature_fun(
                transfer, freqs, tuple(cfg["peak_band_hz"]), tuple(cfg["slope_band_hz"])
            )
            grad, hess, _ = finite_derivatives(fun, np.zeros(4), h=2e-3)
            for condition_pos, u in enumerate(all_u):
                first[input_pos, condition_pos, output_pos], second[
                    input_pos, condition_pos, output_pos
                ] = _taylor_values(fun, grad, hess, u)
    np.savez_compressed(cache, first=first, second=second)
    return first, second


def _run_exp1(cfg: dict, model, out: Path, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_dir = out / "exp1_phase_cache"
    phase_dir.mkdir(parents=True, exist_ok=True)
    conditions, all_u, all_z = _exp1_designs(cfg)
    condition_frame = pd.DataFrame(conditions)
    condition_frame.to_csv(out / "exp1_conditions.csv", index=False)
    n_steps = int(round(EXP1_ANALYSIS_S * 1000.0 / DT_MS))
    tau_e = float(model.params["tau_exc"])
    transfers = np.empty((9, 9, n_steps // 2 + 1), dtype=np.complex128)
    for input_pos, input_area in enumerate(AREAS9):
        input_idx = model.areas.index(input_area)
        for output_pos, output_area in enumerate(AREAS9):
            transfers[input_pos, output_pos] = _discrete_transfer(
                model.J, input_idx, model.areas.index(output_area), n_steps, tau_e
            )
    phase_features = np.empty(
        (TOTAL_PHASES, 9, len(conditions), 9, EXP1_TRIALS, 2), dtype=np.float32
    )
    phase_valid = np.empty(
        (TOTAL_PHASES, 9, len(conditions), 9, EXP1_TRIALS), dtype=bool
    )
    for phase in range(TOTAL_PHASES):
        for input_pos, input_area in enumerate(AREAS9):
            cache = phase_dir / f"phase{phase:02d}_input_{input_area}.npz"
            if force and cache.exists():
                cache.unlink()
            features, valid = _simulate_exp1_phase_input(
                cfg, model, transfers, all_z, phase, input_pos, cache
            )
            phase_features[phase, input_pos] = features
            phase_valid[phase, input_pos] = valid
    first, second = _exp1_theory(cfg, model, all_u)

    rows = []
    for input_pos, input_area in enumerate(AREAS9):
        for condition_pos, condition in enumerate(conditions):
            for output_pos, output_area in enumerate(AREAS9):
                valid = phase_valid[:, input_pos, condition_pos, output_pos]
                theoretical_valid = np.isfinite(first[input_pos, condition_pos, output_pos]).all(axis=1)
                theoretical_valid &= np.isfinite(second[input_pos, condition_pos, output_pos]).all(axis=1)
                use = valid.all(axis=0) & theoretical_valid
                if use.sum() < 16:
                    continue
                observed = phase_features[:, input_pos, condition_pos, output_pos, use].astype(float)
                residual_cov = _pooled_within_trial_cov(observed[:CALIBRATION_PHASES])
                heldout_rs = np.asarray([_corr(values) for values in observed[CALIBRATION_PHASES:]])
                heldout_mean_features = observed[CALIBRATION_PHASES:].mean(axis=0)
                row = {
                    **condition,
                    "input_area": input_area,
                    "output_area": output_area,
                    "n_valid_common": int(use.sum()),
                    "residual_var_P": residual_cov[0, 0],
                    "residual_var_A": residual_cov[1, 1],
                    "residual_cov_PA": residual_cov[0, 1],
                    "residual_only_r": _cov_to_corr(residual_cov),
                    "heldout_r_mean": float(np.nanmean(heldout_rs)),
                    "heldout_r_sd": float(np.nanstd(heldout_rs, ddof=1)),
                    "heldout_phase_ensemble_r": _corr(heldout_mean_features),
                    "heldout_phase_r_json": json.dumps(heldout_rs.tolist()),
                }
                for order, maps in (("first", first), ("second", second)):
                    signal = maps[input_pos, condition_pos, output_pos, use]
                    latent_r, corrected_r, signal_cov = _corrected_corr(signal, residual_cov)
                    row[f"{order}_latent_r"] = latent_r
                    row[f"{order}_residual_corrected_r"] = corrected_r
                    row[f"{order}_signal_var_P"] = signal_cov[0, 0]
                    row[f"{order}_signal_var_A"] = signal_cov[1, 1]
                    row[f"{order}_signal_cov_PA"] = signal_cov[0, 1]
                rows.append(row)
    results = pd.DataFrame(rows)
    results.to_csv(out / "exp1_condition_results.csv.gz", index=False, compression="gzip")

    metric_rows = []
    for (input_area, output_area), group in results.groupby(["input_area", "output_area"]):
        for prediction in (
            "first_latent_r",
            "second_latent_r",
            "first_residual_corrected_r",
            "second_residual_corrected_r",
        ):
            metric_rows.append(
                {
                    "input_area": input_area,
                    "output_area": output_area,
                    "prediction": prediction,
                    **_identity_metrics(group["heldout_r_mean"], group[prediction]),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(out / "exp1_9x9_metrics.csv", index=False)
    _plot_exp1(out, results, metrics)
    _explicit_qc_exp1(cfg, model, all_z, transfers, out)
    return results, metrics


def _plot_exp1(out: Path, results: pd.DataFrame, metrics: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.4), constrained_layout=True)
    pair = results[(results.input_area == "V1") & (results.output_area == "PG")]
    methods = [
        ("first_latent_r", "First order: latent only"),
        ("first_residual_corrected_r", "First order + realization"),
        ("second_latent_r", "Second order: latent only"),
        ("second_residual_corrected_r", "Second order + realization"),
    ]
    for ax, (column, title) in zip(axes.flat[:4], methods):
        x = pair[column].to_numpy()
        y = pair.heldout_r_mean.to_numpy()
        fit = stats.linregress(x, y)
        lo = min(np.nanmin(x), np.nanmin(y)); hi = max(np.nanmax(x), np.nanmax(y))
        ax.scatter(x, y, c=pair.epsilon, cmap="viridis", s=42, edgecolor="white", linewidth=.4)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
        grid = np.linspace(lo, hi, 100)
        ax.plot(grid, fit.intercept + fit.slope * grid, color="#D1495B", lw=1.5, label="OLS")
        metric = _identity_metrics(y, x)
        ax.set(
            xlabel="theory r", ylabel="held-out simulation r", title=title,
        )
        ax.text(
            .03, .97,
            f"identity R2={metric['identity_R2']:.3f}\nOLS R2={metric['ols_R2']:.3f}\nslope={metric['slope']:.3f}",
            transform=ax.transAxes, va="top", fontsize=8,
        )
    for ax, prediction, title in zip(
        axes.flat[4:],
        ("first_residual_corrected_r", "second_residual_corrected_r"),
        ("9 x 9 first-order corrected", "9 x 9 second-order corrected"),
    ):
        pivot = metrics[metrics.prediction == prediction].pivot(
            index="output_area", columns="input_area", values="identity_R2"
        ).reindex(index=AREAS9, columns=AREAS9)
        image = ax.imshow(pivot, cmap="RdYlBu", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(9), AREAS9, rotation=45, ha="right")
        ax.set_yticks(range(9), AREAS9)
        ax.set(xlabel="input area", ylabel="output area", title=title)
        for row in range(9):
            for col in range(9):
                ax.text(col, row, f"{pivot.iloc[row, col]:.2f}", ha="center", va="center", fontsize=6)
        fig.colorbar(image, ax=ax, shrink=.75, label="identity R2")
    fig.suptitle("Experiment 1: adding condition-specific realization covariance", fontsize=15)
    save_figure(fig, out / "figures" / "exp1_realization_residual_validation")
    plt.close(fig)


def _explicit_qc_exp1(cfg: dict, model, all_z: np.ndarray, transfers: np.ndarray, out: Path) -> None:
    burn_s = float(cfg["exp1_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(EXP1_ANALYSIS_S * 1000.0 / DT_MS))
    input_pos = AREAS9.index("V1")
    input_idx = model.areas.index("V1")
    output_indices = np.asarray([model.areas.index(area) for area in AREAS9])
    inputs = _matched_bandlimited_inputs(
        all_z[0, :2], DT_MS, EXP1_ANALYSIS_S, burn_s,
        float(cfg["exp1_nonlinear"]["input_sigma"]), int(cfg["seed"]) + 8980000,
    )
    explicit, rect, *_ = _run_multioutput_batch(
        model.params, inputs, DT_MS, burn_steps, output_indices, input_idx=input_idx
    )
    input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
    accelerated = np.fft.irfft(
        input_fft[:, None, :] * transfers[input_pos][None, :, :], n=n_steps, axis=-1
    )
    rows = []
    for output_pos, output_area in enumerate(AREAS9):
        error = explicit[:, output_pos] - accelerated[:, output_pos]
        rows.append(
            {
                "input_area": "V1",
                "output_area": output_area,
                "n_trials": 2,
                "rectification_count_all_states": int(rect.sum()),
                "timeseries_r": float(
                    np.corrcoef(explicit[:, output_pos].ravel(), accelerated[:, output_pos].ravel())[0, 1]
                ),
                "relative_rmse": float(
                    np.sqrt(np.mean(error**2)) / max(np.std(explicit[:, output_pos]), 1e-15)
                ),
                "max_abs_error": float(np.max(np.abs(error))),
            }
        )
    pd.DataFrame(rows).to_csv(out / "exp1_explicit_recurrence_qc.csv", index=False)


def _exp3_theory_maps(cfg: dict, model, perturbed_j: np.ndarray, z: np.ndarray, u: np.ndarray):
    frequencies = frequency_grid()
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    center = u.mean(axis=0)
    centered = u - center
    maps = {}
    for name, j in (("baseline", model.J), ("deleted", perturbed_j)):
        transfer = _continuous_transfer(j, input_idx, output_idx)
        exact, exact_valid = _exact_trial_features(transfer, z)

        def fun(dx):
            design = Z0 + ZSCALE * (center + np.asarray(dx))
            feature, _ = network_features(
                j, design, input_idx, output_idx, frequencies, transfer,
                tuple(cfg["peak_band_hz"]), tuple(cfg["slope_band_hz"]),
            )
            return np.asarray([feature["peak_omega"], feature["exponent"]], float)

        grad, hess, _ = finite_derivatives(fun, np.zeros(4), h=2e-3)
        first, second = _taylor_values(fun, grad, hess, centered)
        maps[name] = {"first": first, "second": second, "exact": exact, "valid": exact_valid}
    return maps


def _simulate_exp3_phase(
    cfg: dict,
    model,
    perturbed_j: np.ndarray,
    z: np.ndarray,
    phase: int,
    cache: Path,
) -> tuple[np.ndarray, np.ndarray]:
    if cache.exists():
        data = np.load(cache)
        return data["features"], data["valid"]
    specs = _spectral_spec(EXP3_ANALYSIS_S)
    n_steps = int(round(EXP3_ANALYSIS_S * 1000.0 / DT_MS))
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    transfers = np.asarray(
        [
            _discrete_transfer(model.J, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
            _discrete_transfer(perturbed_j, input_idx, output_idx, n_steps, float(model.params["tau_exc"])),
        ]
    )
    features = np.empty((2, N_TRIALS, 2), dtype=np.float32)
    valid = np.empty((2, N_TRIALS), dtype=bool)
    for start in range(0, N_TRIALS, CHUNK_SIZE_EXP3):
        stop = min(start + CHUNK_SIZE_EXP3, N_TRIALS)
        inputs = _matched_bandlimited_inputs(
            z[start:stop], DT_MS, EXP3_ANALYSIS_S, burn_s,
            float(cfg["exp3_nonlinear"]["input_sigma"]),
            int(cfg["seed"]) + 8300000 + phase * 100000,
            trial_offset=start,
        )
        input_fft = np.fft.rfft(inputs[:, burn_steps:], axis=-1)
        outputs = np.empty((stop - start, 2, n_steps), dtype=np.float32)
        for network_pos in range(2):
            outputs[:, network_pos] = np.fft.irfft(
                input_fft * transfers[network_pos][None, :], n=n_steps, axis=-1
            ).astype(np.float32)
        values, is_valid = _feature_arrays_vectorized(outputs, DT_MS, specs)["long"]
        features[:, start:stop] = values
        valid[:, start:stop] = is_valid
        if start % (CHUNK_SIZE_EXP3 * 16) == 0:
            print(
                f"Experiment 3 realization phase {phase + 1}/{TOTAL_PHASES}: "
                f"trials {start + 1}-{stop}/{N_TRIALS}", flush=True,
            )
    np.savez_compressed(cache, features=features, valid=valid)
    return features, valid


def _run_exp3(cfg: dict, model, out: Path, force: bool) -> pd.DataFrame:
    delta_w, edges = _frozen_mask(model)
    perturbed_j = apply_fln_delta(model, delta_w)
    edges.to_csv(out / "exp3_deleted_edges.csv", index=False)
    z, u = _trial_design(int(cfg["seed"]) + 6100000)
    maps = _exp3_theory_maps(cfg, model, perturbed_j, z, u)
    phase_dir = out / "exp3_phase_cache"
    phase_dir.mkdir(parents=True, exist_ok=True)
    features = np.empty((TOTAL_PHASES, 2, N_TRIALS, 2), dtype=np.float32)
    valid = np.empty((TOTAL_PHASES, 2, N_TRIALS), dtype=bool)
    for phase in range(TOTAL_PHASES):
        cache = phase_dir / f"phase{phase:02d}.npz"
        if force and cache.exists():
            cache.unlink()
        features[phase], valid[phase] = _simulate_exp3_phase(
            cfg, model, perturbed_j, z, phase, cache
        )
    theory_valid = maps["baseline"]["valid"] & maps["deleted"]["valid"]
    for network in ("baseline", "deleted"):
        for order in ("first", "second", "exact"):
            theory_valid &= np.isfinite(maps[network][order]).all(axis=1)
    use = theory_valid & valid.all(axis=(0, 1))
    if use.sum() < 256:
        raise RuntimeError(f"Only {use.sum()} common valid trials in Experiment 3")

    rows = []
    residual_covariances = {}
    heldout_by_network = {}
    predictions = {}
    for network_pos, network in enumerate(("baseline", "deleted")):
        observed = features[:, network_pos, use].astype(float)
        residual_cov = _pooled_within_trial_cov(observed[:CALIBRATION_PHASES])
        residual_covariances[network] = residual_cov
        heldout_rs = np.asarray([_corr(values) for values in observed[CALIBRATION_PHASES:]])
        heldout_by_network[network] = heldout_rs
        predictions[network] = {}
        for order in ("first", "second", "exact"):
            signal = maps[network][order][use]
            latent_r, corrected_r, signal_cov = _corrected_corr(signal, residual_cov)
            predictions[network][order] = {"latent": latent_r, "corrected": corrected_r}
            rows.append(
                {
                    "network": network,
                    "order": order,
                    "n_valid_common": int(use.sum()),
                    "latent_r": latent_r,
                    "residual_corrected_r": corrected_r,
                    "heldout_r_mean": float(heldout_rs.mean()),
                    "heldout_r_sd": float(heldout_rs.std(ddof=1)),
                    "heldout_phase_ensemble_r": _corr(observed[CALIBRATION_PHASES:].mean(axis=0)),
                    "residual_var_P": residual_cov[0, 0],
                    "residual_var_A": residual_cov[1, 1],
                    "residual_cov_PA": residual_cov[0, 1],
                    "residual_only_r": _cov_to_corr(residual_cov),
                    "signal_var_P": signal_cov[0, 0],
                    "signal_var_A": signal_cov[1, 1],
                    "signal_cov_PA": signal_cov[0, 1],
                    "heldout_phase_r_json": json.dumps(heldout_rs.tolist()),
                }
            )
    for order in ("first", "second", "exact"):
        rows.append(
            {
                "network": "deleted_minus_baseline",
                "order": order,
                "n_valid_common": int(use.sum()),
                "latent_r": predictions["deleted"][order]["latent"] - predictions["baseline"][order]["latent"],
                "residual_corrected_r": predictions["deleted"][order]["corrected"] - predictions["baseline"][order]["corrected"],
                "heldout_r_mean": float(np.mean(heldout_by_network["deleted"] - heldout_by_network["baseline"])),
                "heldout_r_sd": float(np.std(heldout_by_network["deleted"] - heldout_by_network["baseline"], ddof=1)),
                "heldout_phase_ensemble_r": np.nan,
                "residual_var_P": np.nan,
                "residual_var_A": np.nan,
                "residual_cov_PA": np.nan,
                "residual_only_r": np.nan,
                "signal_var_P": np.nan,
                "signal_var_A": np.nan,
                "signal_cov_PA": np.nan,
                "heldout_phase_r_json": json.dumps(
                    (heldout_by_network["deleted"] - heldout_by_network["baseline"]).tolist()
                ),
            }
        )
    results = pd.DataFrame(rows)
    results.to_csv(out / "exp3_68edge_long_results.csv", index=False)
    np.savez_compressed(
        out / "exp3_68edge_long_features.npz",
        features=features, valid=valid, use=use, z=z, u=u,
        baseline_first=maps["baseline"]["first"], baseline_second=maps["baseline"]["second"],
        baseline_exact=maps["baseline"]["exact"], deleted_first=maps["deleted"]["first"],
        deleted_second=maps["deleted"]["second"], deleted_exact=maps["deleted"]["exact"],
    )
    _plot_exp3(out, results, heldout_by_network)
    _explicit_qc_exp3(cfg, model, perturbed_j, z, out)
    return results


def _explicit_qc_exp3(cfg: dict, model, perturbed_j: np.ndarray, z: np.ndarray, out: Path) -> None:
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / DT_MS))
    n_steps = int(round(EXP3_ANALYSIS_S * 1000.0 / DT_MS))
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    inputs = _matched_bandlimited_inputs(
        z[:2], DT_MS, EXP3_ANALYSIS_S, burn_s,
        float(cfg["exp3_nonlinear"]["input_sigma"]), int(cfg["seed"]) + 8990000,
    )
    rows = []
    for network, j in (("baseline", model.J), ("deleted", perturbed_j)):
        transfer = _discrete_transfer(j, input_idx, output_idx, n_steps, float(model.params["tau_exc"]))
        accelerated = np.fft.irfft(
            np.fft.rfft(inputs[:, burn_steps:], axis=-1) * transfer[None, :], n=n_steps, axis=-1
        )
        if network == "baseline":
            params = model.params
        else:
            params = dict(model.params)
            delta_w, _ = _frozen_mask(model)
            params["fln_mat"] = model.fln + delta_w
        explicit, rect, *_ = _run_multioutput_batch(
            params, inputs, DT_MS, burn_steps, np.asarray([output_idx]), input_idx=input_idx
        )
        error = explicit[:, 0] - accelerated
        rows.append(
            {
                "network": network,
                "n_trials": 2,
                "rectification_count": int(rect.sum()),
                "timeseries_r": float(np.corrcoef(explicit[:, 0].ravel(), accelerated.ravel())[0, 1]),
                "relative_rmse": float(np.sqrt(np.mean(error**2)) / max(np.std(explicit[:, 0]), 1e-15)),
                "max_abs_error": float(np.max(np.abs(error))),
            }
        )
    pd.DataFrame(rows).to_csv(out / "exp3_explicit_recurrence_qc.csv", index=False)


def _plot_exp3(out: Path, results: pd.DataFrame, heldout_by_network: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.4), constrained_layout=True)
    colors = {"first": "#4C78A8", "second": "#F28E2B", "exact": "#6A4C93"}
    for ax, network, title in zip(
        axes.flat[:3],
        ("baseline", "deleted", "deleted_minus_baseline"),
        ("Baseline r", "68-edge deletion r", "Delta r: deleted - baseline"),
    ):
        subset = results[results.network == network].set_index("order")
        x = np.arange(3)
        latent = [subset.loc[o, "latent_r"] for o in ("first", "second", "exact")]
        corrected = [subset.loc[o, "residual_corrected_r"] for o in ("first", "second", "exact")]
        observed = subset.loc["exact", "heldout_r_mean"]
        observed_sd = subset.loc["exact", "heldout_r_sd"]
        ax.bar(x - .2, latent, .2, color=[colors[o] for o in ("first", "second", "exact")], alpha=.45, label="latent")
        ax.bar(x, corrected, .2, color=[colors[o] for o in ("first", "second", "exact")], label="+ realization")
        ax.axhline(observed, color="#2A9D8F", lw=2, label="held-out mean")
        ax.axhspan(observed - observed_sd, observed + observed_sd, color="#2A9D8F", alpha=.14)
        ax.set_xticks(x, ["First", "Second", "Exact"])
        ax.set(ylabel="P-A correlation r" if network != "deleted_minus_baseline" else "Delta r", title=title)
        ax.axhline(0, color=".25", lw=.8)
    axes[0, 0].legend(frameon=False, fontsize=8)
    phase_delta = heldout_by_network["deleted"] - heldout_by_network["baseline"]
    axes[1, 1].plot(np.arange(1, HELDOUT_PHASES + 1), heldout_by_network["baseline"], "o-", label="baseline")
    axes[1, 1].plot(np.arange(1, HELDOUT_PHASES + 1), heldout_by_network["deleted"], "s-", label="deleted")
    axes[1, 1].plot(np.arange(1, HELDOUT_PHASES + 1), phase_delta, "D-", label="delta")
    axes[1, 1].set(
        xlabel="independent held-out realization", ylabel="r or Delta r",
        title="Held-out realization stability", xticks=np.arange(1, HELDOUT_PHASES + 1),
    )
    axes[1, 1].legend(frameon=False)
    fig.suptitle("Experiment 3: 65.536 s window with realization covariance", fontsize=15)
    save_figure(fig, out / "figures" / "exp3_realization_residual_validation")
    plt.close(fig)


def run(cfg: dict, force: bool = False) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(_rooted(cfg["model_dir"]))
    exp1_results, exp1_metrics = _run_exp1(cfg, model, out, force)
    exp3_results = _run_exp3(cfg, model, out, force)
    exp2_source = _rooted(cfg["output_dir"]) / "exp2_window_second_order_validation"
    save_json(
        out / "manifest.json",
        {
            "experiment": "condition_specific_realization_covariance_validation",
            "observation_model": "F_observed = F_latent + epsilon_realization",
            "total_covariance": "Sigma_total = Sigma_signal + Sigma_realization",
            "residual_estimator": "within-trial centered pooled covariance with denominator N*(Q-1)",
            "calibration_phases": CALIBRATION_PHASES,
            "heldout_phases": HELDOUT_PHASES,
            "exp1": {
                "model": "complete 55-area threshold-linear network, discrete transfer validated against explicit recurrence",
                "input_areas": AREAS9,
                "output_areas": AREAS9,
                "conditions": int(exp1_results.condition.nunique()),
                "trials_per_condition": EXP1_TRIALS,
                "analysis_s": EXP1_ANALYSIS_S,
                "dt_ms": DT_MS,
            },
            "exp2": {
                "source": str(exp2_source),
                "status": "copied without rerun because it already uses 3 calibration and 5 held-out phases",
            },
            "exp3": {
                "input_area": INPUT_AREA,
                "output_area": OUTPUT_AREA,
                "n_trials": N_TRIALS,
                "epsilon": EPSILON,
                "input_correlation": INPUT_CORRELATION.tolist(),
                "analysis_s": EXP3_ANALYSIS_S,
                "dt_ms": DT_MS,
                "n_deleted_edges": 68,
            },
            "runtime_s": time.time() - started,
        },
    )
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
