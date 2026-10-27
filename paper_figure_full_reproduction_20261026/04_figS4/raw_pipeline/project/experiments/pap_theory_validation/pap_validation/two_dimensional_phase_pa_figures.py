from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .two_dimensional_ode_validation import (
    CAL_PHASES,
    DT,
    EXP2_TRIALS,
    EXP2_WINDOW_S,
    MODELS,
    PHASE_PLANE_REGIMES,
    PHASE_REGIME_COLORS,
    REPRESENTATIVE_COLOR,
    SEED,
    _aperiodic_psd,
    _aperiodic_feature_map,
    _draw_feature_phases,
    _representative_exp2_row,
    _spectral_residual_cov,
    _style,
    _transfer,
)
from .plot_formatting import format_correlation, format_p_value


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "two_dimensional_ode_validation"
OUTPUT = SOURCE / "phase_plane_and_pa"


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _linear_trajectory(gamma_ratio: float, x0: float, y0_scaled: float, duration: float = 0.22):
    omega = 2.0 * np.pi * 25.0
    gamma = gamma_ratio * omega
    n = int(round(duration / DT))
    x = np.empty(n)
    v = np.empty(n)
    x[0] = x0
    v[0] = y0_scaled * omega
    for pos in range(1, n):
        dx = v[pos - 1]
        dv = -2.0 * gamma * v[pos - 1] - omega**2 * x[pos - 1]
        px = x[pos - 1] + DT * dx
        pv = v[pos - 1] + DT * dv
        dx2 = pv
        dv2 = -2.0 * gamma * pv - omega**2 * px
        x[pos] = x[pos - 1] + 0.5 * DT * (dx + dx2)
        v[pos] = v[pos - 1] + 0.5 * DT * (dv + dv2)
    return x, v / omega


def _hopf_trajectory(mu_hz: float, omega_hz: float, x0: float, y0: float, duration: float = 0.55):
    mu = 2.0 * np.pi * mu_hz
    omega = 2.0 * np.pi * omega_hz
    n = int(round(duration / DT))
    x = np.empty(n)
    y = np.empty(n)
    x[0], y[0] = x0, y0
    for pos in range(1, n):
        r2 = x[pos - 1] ** 2 + y[pos - 1] ** 2
        dx = mu * x[pos - 1] - omega * y[pos - 1] - r2 * x[pos - 1]
        dy = omega * x[pos - 1] + mu * y[pos - 1] - r2 * y[pos - 1]
        px = x[pos - 1] + DT * dx
        py = y[pos - 1] + DT * dy
        rp2 = px**2 + py**2
        dx2 = mu * px - omega * py - rp2 * px
        dy2 = omega * px + mu * py - rp2 * py
        x[pos] = x[pos - 1] + 0.5 * DT * (dx + dx2)
        y[pos] = y[pos - 1] + 0.5 * DT * (dy + dy2)
    return x, y


def _phase_plane(model_key: str) -> None:
    initials = [(1.0, 0.15), (-0.75, 0.7), (0.45, -0.95), (-0.8, -0.45)]
    trajectory_alphas = (1.0, 0.84, 0.68, 0.52)
    regimes = PHASE_PLANE_REGIMES[model_key]
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 13.8), constrained_layout=True)
    for regime_pos, (ax, (title, parameter)) in enumerate(zip(axes, regimes)):
        regime_color = PHASE_REGIME_COLORS[regime_pos]
        for alpha, (x0, y0) in zip(trajectory_alphas, initials):
            if model_key == "damped":
                x, y = _linear_trajectory(parameter, x0, y0)
            else:
                duration = 0.55
                x, y = _hopf_trajectory(parameter[0], parameter[1], x0, y0, duration=duration)
            ax.plot(x, y, color=regime_color, lw=4.8, alpha=alpha)
            ax.scatter(
                [x[0]], [y[0]], s=74, color=regime_color, alpha=alpha,
                edgecolor="white", linewidth=1.2, zorder=4,
            )
            if len(x) > 12:
                arrow = max(4, len(x) // 5)
                ax.annotate(
                    "", xy=(x[arrow], y[arrow]), xytext=(x[arrow - 4], y[arrow - 4]),
                    arrowprops=dict(
                        arrowstyle="-|>", color=regime_color, alpha=alpha,
                        lw=3.0, mutation_scale=18,
                    ),
                )
        ax.scatter([0], [0], s=90, color="black", zorder=5)
        ax.set(xlabel="x", ylabel="y", title=title, xlim=(-1.15, 1.15), ylim=(-1.15, 1.15))
        ax.set_box_aspect(2.0 / 3.0)
        ax.xaxis.label.set_size(28)
        ax.yaxis.label.set_size(28)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_weight("bold")
        ax.title.set_size(26)
        ax.title.set_weight("bold")
        ax.tick_params(labelsize=22, length=8, width=3.0)
        for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
            tick_label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(3.2)
        ax.set_xticks([-1, 0, 1])
        ax.set_yticks([-1, 0, 1])
        ax.grid(False)
    _save(fig, OUTPUT / model_key / f"{model_key}_phase_plane_three_regimes")


def _scatter_condition(model_key: str) -> dict:
    data = pd.read_csv(SOURCE / model_key / "exp2_1000_conditions_rlt0p6.csv")
    model = MODELS[model_key]
    pcol = model.parameter_name
    condition = _representative_exp2_row(data, model)
    condition_id = int(condition.condition)
    rng = np.random.default_rng(SEED + 500000 + condition_id)
    sigma_beta = float(condition.sigma_beta)
    low, high = -0.8 / sigma_beta, 0.8 / sigma_beta
    betas = stats.truncnorm.rvs(
        low, high, loc=0.0, scale=sigma_beta, size=EXP2_TRIALS, random_state=rng
    )
    transfer = _transfer(model, float(condition[pcol]), 0, 0)
    exact = _aperiodic_feature_map(transfer, betas, float(condition.floor_ratio))
    base_psd = transfer * _aperiodic_psd(
        np.array([0.0]), float(condition.floor_ratio)
    )[0]
    residual_cov = _spectral_residual_cov(
        base_psd, EXP2_WINDOW_S,
        SEED + 600000 + condition_id, n=192,
    )
    phases = _draw_feature_phases(exact, residual_cov, SEED + 700000 + condition_id)
    heldout = phases[CAL_PHASES:]
    # Align each held-out realization to a common physical mean/SD before
    # pooling. This preserves every within-realization P-A correlation and
    # makes the pooled correlation exactly equal to the condition-level mean
    # used as the simulation coordinate in panel G.
    reference_mean = heldout.reshape(-1, 2).mean(axis=0, keepdims=True)
    reference_sd = heldout.reshape(-1, 2).std(axis=0, ddof=1, keepdims=True)
    aligned = (
        (heldout - heldout.mean(axis=1, keepdims=True))
        / heldout.std(axis=1, ddof=1, keepdims=True)
        * reference_sd[None, :, :]
        + reference_mean[None, :, :]
    )
    values = aligned.reshape(-1, 2)
    peak, slope = values[:, 0], values[:, 1]
    fit = stats.linregress(peak, slope)
    grid = np.linspace(peak.min(), peak.max(), 200)
    fig, ax = plt.subplots(figsize=(5.2, 4.6), constrained_layout=True)
    ax.scatter(
        peak, slope, s=24, color=REPRESENTATIVE_COLOR, alpha=0.32,
        edgecolor="white", linewidth=0.45,
    )
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#222222", lw=4.0)
    ax.set(xlabel="Peak frequency P (Hz)", ylabel="Aperiodic slope A")
    ax.text(
        0.05, 0.95,
        f"r={format_correlation(fit.rvalue)}\np={format_p_value(fit.pvalue)}",
        transform=ax.transAxes,
        va="top", fontsize=15, fontweight="bold",
    )
    ax.grid(False)
    _save(fig, OUTPUT / model_key / f"{model_key}_complex_regime_PA_scatter")
    result = {
        "model": model_key,
        "condition": condition_id,
        "n_heldout_realizations": int(len(heldout)),
        "n_plotted_trial_realizations": int(len(values)),
        "n_trials": EXP2_TRIALS,
        "analysis_s": EXP2_WINDOW_S,
        pcol: float(condition[pcol]),
        "sigma_beta": sigma_beta,
        "floor_ratio": float(condition.floor_ratio),
        "pearson_r": float(fit.rvalue),
        "p": float(fit.pvalue),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "condition_theory_r": float(condition.residual_corrected_theory_r),
        "condition_simulation_mean_r": float(condition.simulation_r),
        "pooled_r_minus_condition_simulation_mean_r": float(
            fit.rvalue - condition.simulation_r
        ),
        "pooling": "Each held-out realization was centered/scaled to a common physical P/A mean and SD before pooling; within-realization r is unchanged.",
    }
    pd.DataFrame(
        {
            "heldout_realization": np.repeat(np.arange(1, len(heldout) + 1), EXP2_TRIALS),
            "trial": np.tile(np.arange(EXP2_TRIALS), len(heldout)),
            "beta": np.tile(betas, len(heldout)),
            "raw_peak_hz": heldout[:, :, 0].reshape(-1),
            "raw_aperiodic_slope": heldout[:, :, 1].reshape(-1),
            "peak_hz": peak,
            "aperiodic_slope": slope,
        }
    ).to_csv(
        OUTPUT / model_key / f"{model_key}_complex_regime_PA_trials.csv", index=False
    )
    return result


def main() -> None:
    _style()
    summary = {}
    for model_key in ("damped", "hopf"):
        (OUTPUT / model_key).mkdir(parents=True, exist_ok=True)
        _phase_plane(model_key)
        summary[model_key] = _scatter_condition(model_key)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
