from __future__ import annotations

"""Rebuild the final A--F figure with only the FOOOF pipeline changed.

The layout, generative spectra, SPLIT estimator, panel meanings, and plotting
code are inherited from ``fooof_split_attached_code_theory``.  The FOOOF
branch is replaced by the other researcher's full-range knee-mode sequential
procedure: robust AP -> Gaussian peak -> AP refit.  Its covariance prediction
uses the corresponding active-set delta-method influence functions.
"""

import json
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import fooof_split_attached_code_theory as base
import reproduce_gamma_artifact as researcher


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "fooof_split_researcher_pipeline_matched_2026-09-01"
SEED = 20260901 + 4100
N_CAL = 1600
N_SWEEP = 1500
N_NULL = 1600


def _ap_design(freq: np.ndarray, pars: np.ndarray) -> np.ndarray:
    """Jacobian of offset-log10(knee+f**exponent) wrt its parameters."""
    _, exponent, log10_knee = pars
    knee = 10.0 ** log10_knee
    fp = freq ** exponent
    den = knee + fp
    return np.column_stack(
        [
            np.ones(len(freq)),
            -(fp * np.log(freq)) / (np.log(10.0) * den),
            -knee / den,
        ]
    )


def _peak_design(freq: np.ndarray, pars: np.ndarray) -> np.ndarray:
    center, height, sigma = pars
    g = np.exp(-0.5 * ((freq - center) / sigma) ** 2)
    return np.column_stack(
        [
            height * g * (freq - center) / sigma**2,
            g,
            height * g * (freq - center) ** 2 / sigma**3,
        ]
    )


def sequential_stages(
    y: np.ndarray, peak_range: tuple[float, float]
) -> dict[str, np.ndarray]:
    freq = base.FREQ
    first = researcher.fit_ap_knee(y, freq)
    flat_positive = np.maximum(y - researcher.ap_knee(freq, first), 0.0)
    baseline = flat_positive <= np.percentile(flat_positive, 2.5)
    if baseline.sum() < 8:
        order = np.argsort(flat_positive)
        baseline = np.zeros(len(freq), dtype=bool)
        baseline[order[: max(8, int(0.025 * len(freq)))]] = True
    robust = researcher.fit_ap_knee(y[baseline], freq[baseline], first)
    flattened = y - researcher.ap_knee(freq, robust)
    peak_mask = (freq >= peak_range[0]) & (freq <= peak_range[1])
    center0 = float(freq[peak_mask][np.argmax(flattened[peak_mask])])
    height0 = float(np.clip(max(flattened[peak_mask].max(), 0.60), 0.01, 2.95))
    peak = researcher.optimize.least_squares(
        lambda x: flattened - researcher.gaussian(freq, x[0], x[1], x[2]),
        np.array([center0, height0, 1.5]),
        bounds=(
            np.array([peak_range[0], 0.0, 0.75]),
            np.array([peak_range[1], 3.0, 3.0]),
        ),
        max_nfev=180,
    ).x
    peak_curve = researcher.gaussian(freq, *peak)
    final = researcher.fit_ap_knee(y - peak_curve, freq, robust)
    return {
        "first": first,
        "baseline": baseline,
        "robust": robust,
        "flattened": flattened,
        "peak": peak,
        "peak_curve": peak_curve,
        "final": final,
    }


def fit_fooof_researcher(
    y_original: np.ndarray, fit_mask: np.ndarray | None = None
) -> tuple[float, float]:
    mask = base.PEAK_MASK if fit_mask is None else fit_mask
    peak_range = (float(base.FREQ[mask][0]), float(base.FREQ[mask][-1]))
    try:
        return researcher.fit_sequential_knee_fooof(
            y_original, base.FREQ, peak_range
        )
    except (ValueError, RuntimeError, FloatingPointError):
        return np.nan, np.nan


def theory_covariances_researcher(
    params: dict[str, float],
    noise: dict[str, float],
    fit_mask: np.ndarray | None = None,
    covariance_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Active-set delta covariance for the researcher's sequential FOOOF."""
    cov_o, cov_f, cov_of = (
        base.covariance_blocks(noise)
        if covariance_cache is None
        else covariance_cache
    )
    use_mask = base.PEAK_MASK if fit_mask is None else fit_mask
    peak_range = (float(base.FREQ[use_mask][0]), float(base.FREQ[use_mask][-1]))
    mean_o, _ = base.mean_log_spectra(params)
    stages = sequential_stages(mean_o, peak_range)

    n = len(base.FREQ)
    selection = np.flatnonzero(stages["baseline"])
    d_robust_subset = _ap_design(base.FREQ[selection], stages["robust"])
    robust_map = np.zeros((3, n))
    robust_map[:, selection] = np.linalg.pinv(d_robust_subset)
    d_robust_full = _ap_design(base.FREQ, stages["robust"])

    d_peak = _peak_design(base.FREQ, stages["peak"])
    peak_inverse = np.linalg.pinv(d_peak)
    # Algebraically identical to multiplying by the full N x N active-set
    # maps, without ever materialising those dense identity matrices.
    peak_map = peak_inverse - (peak_inverse @ d_robust_full) @ robust_map
    d_final = _ap_design(base.FREQ, stages["final"])
    final_inverse = np.linalg.pinv(d_final)
    final_map = final_inverse - (final_inverse @ d_peak) @ peak_map
    j_fooof = np.vstack([peak_map[0], final_map[1]])
    v_fooof = j_fooof @ cov_o @ j_fooof.T

    # SPLIT is identical to the original figure.
    center, height, sigma = (
        params["center"], params["height"], params["sigma"]
    )
    split_freq = base.FREQ[use_mask]
    g = np.exp(-0.5 * ((split_freq - center) / sigma) ** 2)
    d_split_peak = np.column_stack(
        [
            height * g * (split_freq - center) / sigma**2,
            g,
            height * g * (split_freq - center) ** 2 / sigma**3,
        ]
    )
    j_center = np.linalg.pinv(d_split_peak)[0]
    d_ap = np.column_stack([np.ones(n), -np.log10(base.FREQ)])
    j_exponent = np.linalg.pinv(d_ap)[1]
    peak_idx = np.flatnonzero(use_mask)
    cov_residual = cov_o + cov_f - cov_of - cov_of.T
    cov_residual_fractal = cov_of - cov_f
    v_split = np.array(
        [
            [
                j_center
                @ cov_residual[np.ix_(peak_idx, peak_idx)]
                @ j_center,
                j_center
                @ cov_residual_fractal[np.ix_(peak_idx, np.arange(n))]
                @ j_exponent,
            ],
            [
                j_exponent
                @ cov_residual_fractal[np.ix_(peak_idx, np.arange(n))].T
                @ j_center,
                j_exponent @ cov_f @ j_exponent,
            ],
        ]
    )
    return v_fooof, v_split, {
        "fooof_center": j_fooof[0],
        "fooof_exponent": j_fooof[1],
        "split_center": j_center,
        "split_exponent": j_exponent,
    }


def _normalise(values: np.ndarray) -> np.ndarray:
    scale = float(np.max(np.abs(values)))
    return values / scale if scale else values


def _scientific(value: float) -> str:
    if value == 0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    return rf"{value / 10.0**exponent:.1f}\times10^{{{exponent}}}"


def draw_influence_mechanism_external_researcher(axes: list[plt.Axes]) -> None:
    """Same B1--B4 construction, with the sequential FOOOF Jacobian."""
    params = dict(base.BASE)
    params["center"] = float(base.BANDS["alpha"]["representative"])
    mask = base.band_mask("alpha")
    vf, vs, weights = theory_covariances_researcher(params, base.BASE_NOISE, mask)
    cov_o, cov_f, cov_of = base.covariance_blocks(base.BASE_NOISE)

    fooof_jp = np.asarray(weights["fooof_center"])
    fooof_ja = np.asarray(weights["fooof_exponent"])
    split_jp = np.zeros_like(base.FREQ)
    split_jp[mask] = np.asarray(weights["split_center"])
    split_ja = np.asarray(weights["split_exponent"])
    fooof_contribution = fooof_jp * (cov_o @ fooof_ja)
    split_contribution = split_jp * ((cov_of - cov_f) @ split_ja)

    top = [
        (fooof_jp, fooof_ja, "B1", "FOOOF influence functions", "robust AP / peak / AP refit"),
        (split_jp, split_ja, "B2", "SPLIT influence functions", "residual P / fractal AP"),
    ]
    for ax, (jp, ja, label, title, subtitle) in zip(axes[:2], top):
        ax.plot(base.FREQ, _normalise(jp), color="#247BA0", lw=3.0, label=r"$J_P(f)$")
        ax.plot(base.FREQ, _normalise(ja), color="#D1495B", lw=3.0, label=r"$J_{AP}(f)$")
        ax.axhline(0, color="0.45", lw=1.2)
        ax.set(xscale="log", xlim=(0.2, 200), ylim=(-1.08, 1.08),
               xlabel="Frequency (Hz)", ylabel="Normalized influence")
        ax.set_xticks([0.2, 1, 10, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(title, fontsize=17, fontweight="bold", pad=18,
                     color="#247BA0" if "FOOOF" in title else "#2A9D8F")
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=12.5, color="0.30")
        ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=19,
                fontweight="bold", ha="left", va="top")
        ax.legend(frameon=False, fontsize=13, loc="lower left", ncol=2,
                  handlelength=1.5, columnspacing=0.9)

    lower = [
        (fooof_contribution, vf, "B3", "FOOOF covariance contribution", "#247BA0"),
        (split_contribution, vs, "B4", "SPLIT covariance contribution", "#D1495B"),
    ]
    for ax, (contribution, cov, label, title, color) in zip(axes[2:], lower):
        values = _normalise(contribution)
        ax.fill_between(base.FREQ, 0, values, where=values >= 0,
                        color="#6A4C93", alpha=0.78)
        ax.fill_between(base.FREQ, 0, values, where=values < 0,
                        color="#E76F51", alpha=0.78)
        ax.plot(base.FREQ, values, color=color, lw=2.4)
        ax.axhline(0, color="0.35", lw=1.2)
        ax.set(xscale="log", xlim=(0.2, 200), ylim=(-1.08, 1.08),
               xlabel="Frequency (Hz)", ylabel="Normalized signed contribution")
        ax.set_xticks([0.2, 1, 10, 100])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(title, fontsize=17, fontweight="bold", pad=12,
                     color="#247BA0" if "FOOOF" in title else "#2A9D8F")
        ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=19,
                fontweight="bold", ha="left", va="top")
        ax.text(0.97, 0.94,
                rf"$\sum_f c(f)=V_{{P,AP}}={_scientific(float(cov[0,1]))}$" + "\n" +
                rf"$\rho_{{art}}={base.corr_from_cov(cov):.2f}$",
                transform=ax.transAxes, ha="right", va="top", fontsize=13)


def install_variant() -> None:
    base.OUT = OUT
    base.fit_fooof_local = fit_fooof_researcher
    base.theory_covariances = theory_covariances_researcher
    base.draw_influence_mechanism_external = draw_influence_mechanism_external_researcher


def _cal_worker(item: tuple[int, dict[str, object]]) -> dict[str, object]:
    index, chosen = item
    rng = np.random.default_rng(SEED + 10000 + index)
    params, noise, mask = chosen["params"], chosen["noise"], chosen["fit_mask"]
    train_o, train_f = base.noisy_log_spectra(rng, params, noise, 260)
    estimated_noise = base._estimate_noise_from_training(train_o, train_f)
    held_o, held_f = base.noisy_log_spectra(rng, params, noise, N_CAL)
    fooof, split = base.fit_many(held_o, held_f, mask)
    vf, vs = chosen["v_fooof"], chosen["v_split"]
    vf_est, vs_est, _ = theory_covariances_researcher(params, estimated_noise, mask)
    return {
        "condition": index,
        "target_r": chosen["target_r"],
        "band": chosen["band"],
        "center": params["center"],
        "fit_lo": float(base.FREQ[mask][0]),
        "fit_hi": float(base.FREQ[mask][-1]),
        "fooof_pred": base.corr_from_cov(vf),
        "fooof_artifact_r": base.corr_from_cov(vf),
        "fooof_heldout": base.empirical_corr(fooof),
        "fooof_corrected": base._corrected_corr(fooof, vf_est),
        "split_pred": base.corr_from_cov(vs),
        "split_artifact_r": base.corr_from_cov(vs),
        "split_heldout": base.empirical_corr(split),
        "split_corrected": base._corrected_corr(split, vs_est),
        "fooof_valid": np.isfinite(fooof).all(axis=1).mean(),
        "split_valid": np.isfinite(split).all(axis=1).mean(),
        "n_train": 260,
        "n_heldout": N_CAL,
    }


def candidate_conditions(n_candidates: int = 720) -> list[dict[str, object]]:
    """Theory-only candidate bank spanning approximately -0.08 to +0.50."""
    rng = np.random.default_rng(SEED + 777)
    candidates: list[dict[str, object]] = []
    names = list(base.BANDS)
    for _ in range(n_candidates):
        band = str(rng.choice(names, p=[0.55, 0.20, 0.15, 0.10]))
        spec = base.BANDS[band]
        params = dict(base.BASE)
        params["center"] = float(rng.uniform(*spec["center_range"]))
        params["sigma"] = float(rng.uniform(0.75, 3.0))
        params["height"] = float(rng.uniform(0.60, 1.50))
        params["exponent"] = float(rng.uniform(0.80, 2.80))
        noise = dict(base.BASE_NOISE)
        # Log-uniform distance from one samples both ordinary and very smooth
        # errors; selection below is performed before any held-out simulation.
        noise["ar1"] = float(1.0 - 10.0 ** rng.uniform(-3.10, -0.25))
        covariance = base.covariance_blocks(noise)
        mask = base.band_mask(band)
        vf, vs, _ = theory_covariances_researcher(
            params, noise, mask, covariance_cache=covariance
        )
        value = base.corr_from_cov(vf)
        if np.isfinite(value) and -0.10 <= value <= 0.65:
            candidates.append({"band": band, "params": params, "noise": noise,
                               "fit_mask": mask, "v_fooof": vf, "v_split": vs,
                               "value": value})
    return candidates


def select_targeted_conditions(
    candidates: list[dict[str, object]], n_conditions: int,
    lower: float, upper: float,
) -> list[dict[str, object]]:
    targets = np.linspace(lower, upper, n_conditions)
    selected: list[dict[str, object]] = []
    available = list(candidates)
    for target in targets:
        idx = min(range(len(available)),
                  key=lambda i: abs(float(available[i]["value"]) - target))
        row = available.pop(idx)
        row["target_r"] = float(target)
        selected.append(row)
    return selected


def null_conditions_from_candidates(
    candidates: list[dict[str, object]], n_conditions: int = 120
) -> list[dict[str, object]]:
    selected = select_targeted_conditions(candidates, n_conditions, -0.08, 0.50)
    latent_scale = 0.25
    for row in selected:
        error_r = base.corr_from_cov(row["v_fooof"])
        row["theory_error_r"] = error_r
        row["expected_observed_r"] = base.spearman_from_pearson(
            error_r / (1.0 + latent_scale**2)
        )
        row["latent_scale"] = latent_scale
    return selected


def _sweep_worker(item: tuple[int, float]) -> dict[str, object]:
    index, center = item
    rng = np.random.default_rng(SEED + 20000 + index)
    band = base.band_for_center(center)
    mask = base.band_mask(band)
    params = dict(base.BASE)
    params["center"] = float(center)
    y_o, y_f = base.noisy_log_spectra(rng, params, base.BASE_NOISE, N_SWEEP)
    fooof, split = base.fit_many(y_o, y_f, mask)
    vf, vs, _ = theory_covariances_researcher(params, base.BASE_NOISE, mask)
    fr, sr = base.empirical_corr(fooof), base.empirical_corr(split)
    nf = int(np.isfinite(fooof).all(axis=1).sum())
    ns = int(np.isfinite(split).all(axis=1).sum())
    flo, fhi = base._fisher_interval(fr, nf)
    slo, shi = base._fisher_interval(sr, ns)
    return {"center": center, "band": band, "fooof_pred": base.corr_from_cov(vf),
            "fooof_emp": fr, "fooof_lo": flo, "fooof_hi": fhi,
            "split_pred": base.corr_from_cov(vs), "split_emp": sr,
            "split_lo": slo, "split_hi": shi, "n_fooof": nf, "n_split": ns}


def _null_worker(item: tuple[int, dict[str, object]]) -> dict[str, object]:
    index, condition = item
    rng = np.random.default_rng(SEED + 30000 + index)
    return base.simulate_targeted_null_condition(
        rng, condition, index, n_trials=N_NULL
    )


def main() -> None:
    install_variant()
    base.style()
    OUT.mkdir(parents=True, exist_ok=True)

    cond = base.conditional_experiment(np.random.default_rng(SEED), n=1400)
    candidates = candidate_conditions(720)
    conditions = select_targeted_conditions(candidates, 25, 0.00, 0.50)
    with mp.get_context("fork").Pool(4) as pool:
        cal = pd.DataFrame(pool.map(_cal_worker, list(enumerate(conditions))))
    cal = cal.sort_values("fooof_pred").reset_index(drop=True)

    centers = np.array(
        [8, 9, 10, 11, 12, 14, 18, 22, 26, 30, 32, 39, 46, 53, 60,
         70, 85, 105, 127, 150], dtype=float
    )
    with mp.get_context("fork").Pool(4) as pool:
        sweep = pd.DataFrame(pool.map(_sweep_worker, list(enumerate(centers))))
    sweep = sweep.sort_values("center").reset_index(drop=True)

    null_conditions = null_conditions_from_candidates(candidates, 120)
    with mp.get_context("fork").Pool(4) as pool:
        null = pd.DataFrame(pool.map(_null_worker, list(enumerate(null_conditions))))
    null = null.sort_values("dataset").reset_index(drop=True)

    factors = pd.DataFrame()
    path = base.plot_six_panel(cond, cal, sweep, null, factors)
    # The inherited plot names the output as in the original; place it in a
    # separate directory and also give the new pipeline an explicit filename.
    explicit = OUT / "figure2_theory_validation_nine_panels_researcher_fooof.svg"
    (OUT / "figure2_theory_validation_nine_panels.svg").replace(explicit)
    preview_src = OUT / "figure2_theory_validation_nine_panels_preview.png"
    preview = OUT / "figure2_theory_validation_nine_panels_researcher_fooof_preview.png"
    preview_src.replace(preview)

    cal.to_csv(OUT / "uniform_theory_calibration_conditions.csv", index=False)
    sweep.to_csv(OUT / "center_sweep.csv", index=False)
    null.to_csv(OUT / "null_dataset_results.csv", index=False)
    fit = stats.linregress(cal.fooof_pred, cal.fooof_heldout)
    metrics = {
        "figure": str(explicit),
        "fooof_pipeline": "full-range knee; robust AP -> peak -> AP refit",
        "theory": "active-set sequential delta-method Jacobian",
        "layout": "same 3x3 A-F figure; B is the same grouped 2x2 panel",
        "conditional_fooof_r": base.empirical_corr(cond["fooof"]),
        "conditional_split_r": base.empirical_corr(cond["split"]),
        "panel_d_pearson_r": float(fit.rvalue),
        "panel_d_ols_slope": float(fit.slope),
        "panel_d_r2": float(fit.rvalue**2),
        "n_calibration_per_condition": N_CAL,
        "n_sweep_per_center": N_SWEEP,
        "n_null_per_condition": N_NULL,
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
