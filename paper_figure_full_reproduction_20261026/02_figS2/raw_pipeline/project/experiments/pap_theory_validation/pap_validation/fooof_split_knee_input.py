from __future__ import annotations

import json
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, stats

import fooof_split_attached_code_theory as base


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "fooof_split_knee15_input_2026-08-31"
KNEE_FREQUENCY_HZ = 15.0
SEED = base.SEED + 1000


def knee_log_aperiodic(offset: float, exponent: np.ndarray | float) -> np.ndarray:
    exponent_array = np.asarray(exponent)
    if exponent_array.ndim == 0:
        knee = KNEE_FREQUENCY_HZ ** float(exponent_array)
        return offset - np.log10(knee + base.FREQ ** float(exponent_array))
    knee = KNEE_FREQUENCY_HZ ** exponent_array
    return offset - np.log10(
        knee[:, None] + base.FREQ[None, :] ** exponent_array[:, None]
    )


def mean_log_spectra_knee(params: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    ap = knee_log_aperiodic(params["offset"], params["exponent"])
    osc = base.gaussian(base.FREQ, params["center"], params["height"], params["sigma"])
    return ap + osc, ap


def noisy_log_spectra_with_latents_knee(
    rng: np.random.Generator,
    params: dict[str, float],
    noise: dict[str, float],
    centers: np.ndarray,
    exponents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n = len(centers)
    ap = knee_log_aperiodic(params["offset"], exponents)
    osc = params["height"] * np.exp(
        -0.5 * ((base.FREQ[None, :] - centers[:, None]) / params["sigma"]) ** 2
    )
    shared = base.ar1_draws(rng, n, len(base.FREQ), noise["ar1"])
    only_o = base.ar1_draws(rng, n, len(base.FREQ), noise["ar1"])
    only_f = base.ar1_draws(rng, n, len(base.FREQ), noise["ar1"])
    q = noise["shared_fraction"]
    eo = noise["sd_original"] * (
        np.sqrt(q) * shared + np.sqrt(1.0 - q) * only_o
    )
    ef = noise["sd_fractal"] * (
        np.sqrt(q) * shared + np.sqrt(1.0 - q) * only_f
    )
    return ap + osc + eo, ap + ef


def fit_fooof_knee_full(
    y_original: np.ndarray,
    use_mask: np.ndarray,
    enforce_threshold: bool = True,
) -> np.ndarray:
    """Fit offset, exponent, knee, center, height and width jointly.

    The fitted knee is parameterized as log10(knee); no true-knee value is
    supplied to the optimizer or to the covariance prediction.
    """
    # FOOOF knee mode needs frequencies on both sides of the bend.  The full
    # 0.2--200 Hz spectrum identifies the aperiodic knee, while use_mask only
    # constrains which oscillatory peak is returned.
    y = y_original
    fit_freq = base.FREQ
    peak_freq = base.FREQ[use_mask]
    off0, exp0 = base._ap_guess(y, fit_freq)
    flat = y - (off0 - exp0 * np.log10(fit_freq))
    c0 = float(peak_freq[np.argmax(flat[use_mask])])
    x0 = np.array(
        [
            np.clip(off0, -7.95, 7.95),
            np.clip(exp0, 0.05, 4.5),
            np.clip(np.clip(exp0, 0.05, 4.5) * np.log10(10.0), -3.9, 4.9),
            np.clip(c0, peak_freq[0] + 1e-6, peak_freq[-1] - 1e-6),
            np.clip(max(float(flat.max()), 0.65), 1e-4, 2.95),
            1.5,
        ]
    )

    def residual(x: np.ndarray) -> np.ndarray:
        off, exponent, log10_knee, center, height, sigma = x
        model = (
            off
            - np.log10(10.0**log10_knee + fit_freq**exponent)
            + base.gaussian(fit_freq, center, height, sigma)
        )
        return y - model

    def residual_jacobian(x: np.ndarray) -> np.ndarray:
        _, exponent, log10_knee, center, height, sigma = x
        knee = 10.0**log10_knee
        frequency_power = fit_freq**exponent
        denominator = knee + frequency_power
        g = np.exp(-0.5 * ((fit_freq - center) / sigma) ** 2)
        model_jacobian = np.column_stack(
            [
                np.ones(len(fit_freq)),
                -(frequency_power * np.log(fit_freq))
                / (np.log(10.0) * denominator),
                -knee / denominator,
                height * g * (fit_freq - center) / sigma**2,
                g,
                height * g * (fit_freq - center) ** 2 / sigma**3,
            ]
        )
        return -model_jacobian

    lo = np.array([-8.0, 0.05, -4.0, peak_freq[0], 0.0, 0.75])
    hi = np.array([8.0, 5.0, 5.0, peak_freq[-1], 3.0, 3.0])
    try:
        fit = optimize.least_squares(
            residual,
            x0,
            jac=residual_jacobian,
            bounds=(lo, hi),
            max_nfev=220,
        )
        off, exponent, log10_knee, center, height, sigma = fit.x
        if enforce_threshold:
            aperiodic = off - np.log10(10.0**log10_knee + fit_freq**exponent)
            flat_final = y - aperiodic
            if height < 0.60 or height <= 2.0 * np.std(
                flat_final - base.gaussian(fit_freq, center, height, sigma)
            ):
                return np.full(6, np.nan)
        return fit.x
    except (ValueError, RuntimeError, FloatingPointError):
        return np.full(6, np.nan)


def fit_fooof_knee_local(
    y_original: np.ndarray, fit_mask: np.ndarray | None = None
) -> tuple[float, float]:
    use_mask = base.PEAK_MASK if fit_mask is None else fit_mask
    fit = fit_fooof_knee_full(y_original, use_mask, enforce_threshold=True)
    if not np.isfinite(fit).all():
        return np.nan, np.nan
    return float(fit[3]), float(fit[1])


def pseudo_knee_mode_fit(params: dict[str, float], use_mask: np.ndarray) -> np.ndarray:
    """Pseudo-true FOOOF fit used for the delta-method Jacobian."""
    return fit_fooof_knee_full(
        mean_log_spectra_knee(params)[0], use_mask, enforce_threshold=False
    )


def theory_covariances_knee(
    params: dict[str, float],
    noise: dict[str, float],
    fit_mask: np.ndarray | None = None,
    covariance_cache: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    cov_o, cov_f, cov_of = (
        base.covariance_blocks(noise) if covariance_cache is None else covariance_cache
    )
    use_mask = base.PEAK_MASK if fit_mask is None else fit_mask
    fit_freq = base.FREQ
    off, exponent, log10_knee, c, h, s = pseudo_knee_mode_fit(params, use_mask)
    if not np.isfinite([off, exponent, log10_knee, c, h, s]).all():
        return np.full((2, 2), np.nan), np.full((2, 2), np.nan), {}
    knee = 10.0**log10_knee
    denominator = knee + fit_freq**exponent
    g = np.exp(-0.5 * ((fit_freq - c) / s) ** 2)
    d_fooof = np.column_stack(
        [
            np.ones(len(fit_freq)),
            -(fit_freq**exponent * np.log(fit_freq)) / (np.log(10.0) * denominator),
            -knee / denominator,
            h * g * (fit_freq - c) / s**2,
            g,
            h * g * (fit_freq - c) ** 2 / s**3,
        ]
    )
    influence_fooof = np.linalg.pinv(d_fooof)
    j_fooof = np.vstack([influence_fooof[3], influence_fooof[1]])
    v_fooof = j_fooof @ cov_o @ j_fooof.T

    # SPLIT's oscillatory residual cancels the knee exactly; its AP estimate
    # remains the same full-range linear functional of the fractal spectrum.
    c0, h0, s0 = params["center"], params["height"], params["sigma"]
    split_freq = base.FREQ[use_mask]
    g0 = np.exp(-0.5 * ((split_freq - c0) / s0) ** 2)
    d_peak = np.column_stack(
        [h0 * g0 * (split_freq - c0) / s0**2, g0,
         h0 * g0 * (split_freq - c0) ** 2 / s0**3]
    )
    j_center = np.linalg.pinv(d_peak)[0]
    d_ap = np.column_stack([np.ones(len(base.FREQ)), -np.log10(base.FREQ)])
    j_exponent = np.linalg.pinv(d_ap)[1]
    peak_idx = np.flatnonzero(use_mask)
    cov_residual = cov_o + cov_f - cov_of - cov_of.T
    cov_residual_fractal = cov_of - cov_f
    v_split = np.array(
        [
            [
                j_center @ cov_residual[np.ix_(peak_idx, peak_idx)] @ j_center,
                j_center
                @ cov_residual_fractal[np.ix_(peak_idx, np.arange(len(base.FREQ)))]
                @ j_exponent,
            ],
            [
                j_exponent
                @ cov_residual_fractal[np.ix_(peak_idx, np.arange(len(base.FREQ)))].T
                @ j_center,
                j_exponent @ cov_f @ j_exponent,
            ],
        ]
    )
    weights = {
        "fooof_center": influence_fooof[3],
        "fooof_exponent": influence_fooof[1],
        "split_center": j_center,
        "split_exponent": j_exponent,
    }
    return v_fooof, v_split, weights


def draw_influence_mechanism_external_knee(axes: list[plt.Axes]) -> None:
    """B1--B4 for knee-mode FOOOF, with matched full-frequency axes."""
    params = dict(base.BASE)
    params["center"] = float(base.BANDS["alpha"]["representative"])
    use_mask = base.band_mask("alpha")
    vf, vs, weights = theory_covariances_knee(params, base.BASE_NOISE, use_mask)
    cov_o, cov_f, cov_of = base.covariance_blocks(base.BASE_NOISE)

    fooof_jp = np.asarray(weights["fooof_center"])
    fooof_ja = np.asarray(weights["fooof_exponent"])
    split_jp = np.zeros_like(base.FREQ)
    split_jp[use_mask] = np.asarray(weights["split_center"])
    split_ja = np.asarray(weights["split_exponent"])
    fooof_contribution = fooof_jp * (cov_o @ fooof_ja)
    split_contribution = split_jp * ((cov_of - cov_f) @ split_ja)

    def normalize(values: np.ndarray) -> np.ndarray:
        scale = float(np.max(np.abs(values)))
        return values / scale if scale > 0 else values

    def scientific_math(value: float) -> str:
        exponent = int(np.floor(np.log10(abs(value))))
        return rf"{value / 10.0**exponent:.1f}\times10^{{{exponent}}}"

    top_specs = [
        (fooof_jp, fooof_ja, "B1", "FOOOF influence functions", "jointly fitted knee"),
        (split_jp, split_ja, "B2", "SPLIT influence functions", "residual P / fractal AP"),
    ]
    for ax, (jp, ja, label, title, subtitle) in zip(axes[:2], top_specs):
        ax.plot(base.FREQ, normalize(jp), color="#247BA0", lw=3.0, label=r"$J_P(f)$")
        ax.plot(base.FREQ, normalize(ja), color="#D1495B", lw=3.0, label=r"$J_{AP}(f)$")
        ax.axhline(0, color="0.45", lw=1.2)
        ax.set(xscale="log", xlim=(0.2, 200.0), ylim=(-1.08, 1.08),
               xlabel="Frequency (Hz)", ylabel="Normalized influence")
        ax.set_xticks([0.2, 1, 10, 100, 200])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(title, fontsize=17, fontweight="bold", pad=18,
                     color="#247BA0" if "FOOOF" in title else "#2A9D8F")
        ax.text(0.5, 1.015, subtitle, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12.5, color="0.30")
        ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=19,
                fontweight="bold", ha="left", va="top")
        ax.legend(frameon=False, fontsize=13, loc="lower left", ncol=2,
                  handlelength=1.5, columnspacing=0.9)

    bottom_specs = [
        (fooof_contribution, vf, "B3", "FOOOF covariance contribution", "#247BA0"),
        (split_contribution, vs, "B4", "SPLIT covariance contribution", "#D1495B"),
    ]
    for ax, (contribution, cov, label, title, color) in zip(axes[2:], bottom_specs):
        cn = normalize(contribution)
        ax.fill_between(base.FREQ, 0, cn, where=cn >= 0, color="#6A4C93", alpha=0.78)
        ax.fill_between(base.FREQ, 0, cn, where=cn < 0, color="#E76F51", alpha=0.78)
        ax.plot(base.FREQ, cn, color=color, lw=2.4)
        ax.axhline(0, color="0.35", lw=1.2)
        ax.set(xscale="log", xlim=(0.2, 200.0), ylim=(-1.08, 1.08),
               xlabel="Frequency (Hz)", ylabel="Normalized signed contribution")
        ax.set_xticks([0.2, 1, 10, 100, 200])
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_title(title, fontsize=17, fontweight="bold", pad=12,
                     color="#247BA0" if "FOOOF" in title else "#2A9D8F")
        ax.text(-0.13, 1.10, label, transform=ax.transAxes, fontsize=19,
                fontweight="bold", ha="left", va="top")
        ax.text(
            0.97, 0.94,
            rf"$\sum_f c(f)=V_{{P,AP}}={scientific_math(float(cov[0,1]))}$" + "\n" +
            rf"$\rho_{{art}}={base.corr_from_cov(cov):.2f}$",
            transform=ax.transAxes, ha="right", va="top", fontsize=13,
        )


def install_knee_variant() -> None:
    base.OUT = OUT
    base.mean_log_spectra = mean_log_spectra_knee
    base.noisy_log_spectra_with_latents = noisy_log_spectra_with_latents_knee
    base.theory_covariances = theory_covariances_knee
    base.fit_fooof_local = fit_fooof_knee_local
    base.draw_influence_mechanism_external = draw_influence_mechanism_external_knee


def _null_worker(args: tuple[int, dict[str, object]]) -> dict:
    dataset, condition = args
    rng = np.random.default_rng(SEED + 9000 + dataset)
    return base.simulate_targeted_null_condition(
        rng, condition, dataset, n_trials=1600
    )


def uniform_knee_calibration_experiment(
    rng: np.random.Generator,
    n_conditions: int = 25,
    n_train: int = 260,
    n_heldout: int = 1600,
) -> pd.DataFrame:
    """Held-out validation spanning the stable range of the knee-mode theory."""
    candidates: list[dict[str, object]] = []
    for ar1 in np.linspace(0.30, 0.99, 120):
        noise = dict(base.BASE_NOISE)
        noise["ar1"] = float(ar1)
        cached_covariance = base.covariance_blocks(noise)
        for band_name, spec in base.BANDS.items():
            center = float(spec["representative"])
            params = dict(base.BASE)
            params["center"] = center
            use_mask = base.band_mask(band_name)
            vf, vs, _ = theory_covariances_knee(
                params, noise, use_mask, covariance_cache=cached_covariance
            )
            if not np.isfinite(vf).all():
                continue
            prediction = base.corr_from_cov(vf)
            if -0.22 <= prediction <= 0.10:
                candidates.append(
                    {
                        "band": band_name,
                        "center": center,
                        "fit_lo": float(base.FREQ[use_mask][0]),
                        "fit_hi": float(base.FREQ[use_mask][-1]),
                        "fit_mask": use_mask,
                        "params": dict(params),
                        "noise": noise,
                        "v_fooof": vf,
                        "v_split": vs,
                        "fooof_r": prediction,
                    }
                )
    candidates.sort(key=lambda item: float(item["fooof_r"]))
    targets = np.linspace(
        max(-0.20, float(candidates[0]["fooof_r"])),
        min(0.08, float(candidates[-1]["fooof_r"])),
        n_conditions,
    )
    chosen_rows: list[dict[str, object]] = []
    available = list(candidates)
    for target in targets:
        selected_index = min(
            range(len(available)),
            key=lambda index: abs(float(available[index]["fooof_r"]) - target),
        )
        chosen_rows.append(available.pop(selected_index))

    rows: list[dict[str, float | str]] = []
    for condition, (target, chosen) in enumerate(zip(targets, chosen_rows)):
        params = chosen["params"]
        noise = chosen["noise"]
        use_mask = chosen["fit_mask"]
        train_o, train_f = noisy_log_spectra_with_latents_knee(
            rng,
            params,
            noise,
            np.full(n_train, float(params["center"])),
            np.full(n_train, float(params["exponent"])),
        )
        estimated_noise = base._estimate_noise_from_training(train_o, train_f)
        held_o, held_f = noisy_log_spectra_with_latents_knee(
            rng,
            params,
            noise,
            np.full(n_heldout, float(params["center"])),
            np.full(n_heldout, float(params["exponent"])),
        )
        fooof, split = base.fit_many(held_o, held_f, use_mask)
        vf, vs = chosen["v_fooof"], chosen["v_split"]
        vf_est, vs_est, _ = theory_covariances_knee(params, estimated_noise, use_mask)
        rows.append(
            {
                "condition": condition,
                "target_r": float(target),
                "band": str(chosen["band"]),
                "center": float(chosen["center"]),
                "fit_lo": float(chosen["fit_lo"]),
                "fit_hi": float(chosen["fit_hi"]),
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
                "n_train": n_train,
                "n_heldout": n_heldout,
            }
        )
    return pd.DataFrame(rows).sort_values("fooof_pred").reset_index(drop=True)


def plot_knee_six_panel(
    cond: dict,
    cal: pd.DataFrame,
    sweep: pd.DataFrame,
    null: pd.DataFrame,
) -> Path:
    fig, axes = plt.subplots(
        2, 3, figsize=(19.0, 10.5), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.0, 1.34, 1.0]},
    )
    mean_o, mean_f = mean_log_spectra_knee(cond["params"])

    ax = axes[0, 0]
    ax.plot(base.FREQ, mean_o, color="#333333", label="Original log PSD")
    ax.plot(base.FREQ, mean_f, color="#D1495B", label="IRASA fractal")
    ax.plot(base.FREQ, mean_o - mean_f, color="#247BA0", label="Oscillatory residual")
    lo, hi = base.BANDS["alpha"]["fit_range"]
    ax.axvspan(lo, hi, color="#247BA0", alpha=0.10)
    ax.set(xscale="log", xlabel="Frequency (Hz)", ylabel="log10 power")
    ax.set_xticks([0.2, 1, 10, 30, 100])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, loc="lower left")
    base.add_panel_label(ax, "A")

    ax = axes[0, 1]
    base.draw_influence_mechanism_panel(ax)
    base.add_panel_label(ax, "B")

    ax = axes[0, 2]
    fooof, split = cond["fooof"], cond["split"]
    gf, gs = np.isfinite(fooof).all(axis=1), np.isfinite(split).all(axis=1)
    ax.scatter(fooof[gf, 0], fooof[gf, 1], s=15, alpha=0.16,
               color="#247BA0", label="FOOOF")
    ax.scatter(split[gs, 0], split[gs, 1], s=15, alpha=0.16,
               color="#2A9D8F", label="SPLIT")
    base.ellipse_from_cov(ax, np.nanmean(fooof, axis=0), cond["v_fooof"], "#247BA0")
    base.ellipse_from_cov(ax, np.nanmean(split, axis=0), cond["v_split"], "#2A9D8F")
    ax.set(xlabel="Estimated peak center (Hz)", ylabel="Estimated exponent")
    ax.legend(frameon=False, loc="upper left")
    ax.text(0.04, 0.13, f"FOOOF: r = {base.empirical_corr(fooof):.2f}",
            transform=ax.transAxes, color="#247BA0", fontsize=14, fontweight="bold")
    ax.text(0.04, 0.06, f"SPLIT: r = {base.empirical_corr(split):.2f}",
            transform=ax.transAxes, color="#2A9D8F", fontsize=14, fontweight="bold")
    base.add_panel_label(ax, "C")

    ax = axes[1, 0]
    ax.scatter(cal.fooof_pred, cal.fooof_heldout, s=54, marker="o",
               color="#247BA0", edgecolor="black", linewidth=0.55,
               alpha=0.84, label="FOOOF")
    values = np.r_[cal.fooof_pred, cal.fooof_heldout]
    lower, upper = np.nanmin(values) - 0.06, np.nanmax(values) + 0.06
    ax.plot([lower, upper], [lower, upper], color="0.25", ls="--", lw=2.0)
    ax.axhline(0, color="0.45", lw=1.3)
    ax.set(xlim=(lower, upper), ylim=(lower, upper),
           xlabel="Theory-predicted error r", ylabel="Held-out error r")
    ax.legend(frameon=False, loc="upper left", fontsize=12.5)
    fit = stats.linregress(cal.fooof_pred, cal.fooof_heldout)
    ax.text(0.97, 0.05,
            f"FOOOF: Pearson r = {fit.rvalue:.2f}\n"
            f"OLS slope = {fit.slope:.2f}; $R^2$ = {fit.rvalue**2:.2f}",
            transform=ax.transAxes, ha="right", va="bottom", color="#247BA0",
            fontsize=12.5, fontweight="bold")
    base.add_panel_label(ax, "D")

    ax = axes[1, 1]
    for band_name, specification in base.BANDS.items():
        band_lo, band_hi = specification["center_range"]
        ax.axvspan(band_lo, band_hi, color=base.BAND_COLORS[band_name], alpha=0.075, lw=0)
        ax.text(np.sqrt(band_lo * band_hi), 0.98, band_name,
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                fontsize=10.5, color=base.BAND_COLORS[band_name], fontweight="bold")
    ax.fill_between(sweep.center, sweep.fooof_lo, sweep.fooof_hi,
                    color="#247BA0", alpha=0.13, linewidth=0)
    ax.fill_between(sweep.center, sweep.split_lo, sweep.split_hi,
                    color="#2A9D8F", alpha=0.13, linewidth=0)
    ax.plot(sweep.center, sweep.fooof_emp, "o-", color="#247BA0", label="FOOOF empirical")
    ax.plot(sweep.center, sweep.fooof_pred, "--", color="#247BA0", label="FOOOF theory")
    ax.plot(sweep.center, sweep.split_emp, "o-", color="#2A9D8F", label="SPLIT empirical")
    ax.plot(sweep.center, sweep.split_pred, "--", color="#2A9D8F", label="SPLIT theory")
    ax.axhline(0, color="0.35", lw=1.5)
    ax.set(xscale="log", xlabel="True peak center (Hz)", ylabel="Conditional error r")
    ax.set_xticks([8, 12, 30, 60, 100, 150])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.legend(frameon=False, ncol=1, loc="best")
    base.add_panel_label(ax, "E")

    ax = axes[1, 2]
    series = [null.fooof_r.dropna(), null.fooof_corrected_r.dropna(), null.split_r.dropna()]
    violin = ax.violinplot(series, positions=[0, 1, 2], widths=0.72,
                           showmeans=True, showextrema=False)
    for body, color in zip(violin["bodies"], ["#247BA0", "#D1495B", "#2A9D8F"]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_alpha(0.78)
    violin["cmeans"].set_color("black")
    violin["cmeans"].set_linewidth(3)
    ax.axhline(0, color="0.25", lw=1.8)
    ax.set(xticks=[0, 1, 2], xticklabels=["FOOOF", "FOOOF\ncorrected", "SPLIT"],
           ylabel="Null-dataset Spearman rho")
    ax.text(0.98, 0.97, f"{len(null)} conditions; n = 1600 trials/condition",
            transform=ax.transAxes, ha="right", va="top", fontsize=10.5)
    base.add_panel_label(ax, "F")

    for axis in axes.ravel():
        base.thicken(axis)
    path = OUT / "fooof_split_six_panels_knee_input.svg"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / "fooof_split_six_panels_knee_input_preview.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main() -> None:
    install_knee_variant()
    base.style()
    OUT.mkdir(parents=True, exist_ok=True)
    cond = base.conditional_experiment(np.random.default_rng(SEED))
    factors = base.covariance_factor_experiment(np.random.default_rng(SEED + 1))
    cal = uniform_knee_calibration_experiment(np.random.default_rng(SEED + 2))
    cal.to_csv(OUT / "uniform_theory_calibration_conditions.csv", index=False)
    sweep = base.center_sweep(np.random.default_rng(SEED + 3))
    sweep.to_csv(OUT / "center_sweep.csv", index=False)
    factors.to_csv(OUT / "covariance_factor_decomposition.csv", index=False)
    conditions = base.targeted_null_conditions(120)
    with mp.get_context("fork").Pool(4) as pool:
        null_rows = pool.map(_null_worker, list(enumerate(conditions)))
    null = pd.DataFrame(null_rows).sort_values("dataset").reset_index(drop=True)
    null.to_csv(OUT / "null_dataset_results.csv", index=False)

    path = base.plot_six_panel(cond, cal, sweep, null, factors)
    # Report fitted knee frequencies from spectra not used to set the model's
    # true knee.  This is an output of the six-parameter FOOOF fit.
    fitted_full = np.array(
        [fit_fooof_knee_full(row, cond["fit_mask"], enforce_threshold=True)
         for row in cond["y_original"][:400]]
    )
    valid_full = np.isfinite(fitted_full).all(axis=1) & (fitted_full[:, 1] > 0)
    fitted_knee_hz = 10.0 ** (
        fitted_full[valid_full, 2] / fitted_full[valid_full, 1]
    )
    summary = {
        "knee_frequency_hz": KNEE_FREQUENCY_HZ,
        "aperiodic_model": "offset - log10(knee + f^exponent)",
        "fooof_knee_mode": "offset, exponent, and knee estimated jointly for every spectrum",
        "fooof_initial_knee_hz": 10.0,
        "fooof_estimated_knee_median_hz": float(np.median(fitted_knee_hz)),
        "fooof_estimated_knee_iqr_hz": [
            float(np.quantile(fitted_knee_hz, 0.25)),
            float(np.quantile(fitted_knee_hz, 0.75)),
        ],
        "all_other_parameters": "unchanged from the no-knee figure",
        "figure": str(path),
        "conditional_fooof_r": base.empirical_corr(cond["fooof"]),
        "conditional_split_r": base.empirical_corr(cond["split"]),
        "n_null_conditions": int(len(null)),
        "n_trials_per_null_condition": 1600,
        "layout": "3x3 nine-panel mode (B1-B4 grouped as panel B)",
    }
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
