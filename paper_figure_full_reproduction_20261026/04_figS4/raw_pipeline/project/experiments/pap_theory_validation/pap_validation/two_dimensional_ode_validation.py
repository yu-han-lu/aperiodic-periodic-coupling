from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats
from scipy.stats import qmc

from .plot_formatting import format_correlation, format_p_value

try:
    from numba import njit, prange
except ImportError:  # pragma: no cover
    njit = lambda *args, **kwargs: (lambda f: f)
    prange = range


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "two_dimensional_ode_validation"
DT = 0.00025
BURN_S = 2.048
EXP1_WINDOW_S = 32.768
EXP2_WINDOW_S = 131.072
WELCH_SEGMENT_S = 4.096
EXP1_CONDITIONS = 1000
EXP1_TRIALS = 64
EXP2_CONDITIONS = 1000
EXP2_TRIALS = 256
CAL_PHASES = 3
HELDOUT_PHASES = 5
SEED = 20260809
FREQ = np.arange(0.5, 1200.01, 0.5)
PEAK_BAND = (5.0, 100.0)
SLOPE_BAND = (100.0, 1000.0)
Z0 = np.array([10.0, 6.0, -0.5, 1.0])


@dataclass(frozen=True)
class ModelSpec:
    key: str
    title: str
    omega_hz: float
    exp1_parameter: float
    exp2_parameter: float
    parameter_name: str


MODELS = {
    "damped": ModelSpec(
        "damped", "Damped linear oscillator", 25.0, 2.0, 0.16, "gamma_ratio"
    ),
    "hopf": ModelSpec(
        "hopf", "Hopf normal form", 25.0, -40.0, -2.5, "mu_hz"
    ),
}

PHASE_PLANE_REGIMES = {
    "damped": [
        ("Real eigenvalues", 1.8),
        ("Critical damping", 1.0),
        ("Complex eigenvalues", 0.18),
    ],
    "hopf": [
        ("Real eigenvalues", (-3.0, 0.0)),
        ("Complex-eigenvalue onset", (-3.0, 0.5)),
        ("Stable complex", (-3.0, 25.0)),
    ],
}
PHASE_REGIME_COLORS = ("#247BA0", "#D1495B", "#2A9D8F")
REPRESENTATIVE_COLOR = "#F28E2B"
PSD_AXES_LINEWIDTH = 2.4
PSD_LABELSIZE = 17
PSD_TITLESIZE = 18
PSD_TICKSIZE = 15


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "axes.linewidth": 1.9,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12.5,
            "lines.linewidth": 3.0,
            "xtick.major.width": 1.7,
            "ytick.major.width": 1.7,
            "figure.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.18, 1.08, label, transform=ax.transAxes, fontsize=23, fontweight="bold")


def _style_psd_axes(axes) -> None:
    """Apply one visual grammar to every input and output PSD axis."""
    for ax in np.atleast_1d(axes).ravel():
        ax.xaxis.label.set_size(PSD_LABELSIZE)
        ax.yaxis.label.set_size(PSD_LABELSIZE)
        ax.xaxis.label.set_weight("bold")
        ax.yaxis.label.set_weight("bold")
        ax.title.set_size(PSD_TITLESIZE)
        ax.title.set_weight("bold")
        ax.tick_params(axis="both", which="major", labelsize=PSD_TICKSIZE, width=PSD_AXES_LINEWIDTH, length=6)
        ax.tick_params(axis="both", which="minor", width=1.5, length=3)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontweight("bold")
        for spine in ax.spines.values():
            spine.set_linewidth(PSD_AXES_LINEWIDTH)


def _representative_exp2_row(frame: pd.DataFrame, model: ModelSpec) -> pd.Series:
    """Select the registered complex-regime example shown in the trial plot."""
    pcol = model.parameter_name
    target = 0.18 if model.key == "damped" else -2.5
    score = (frame[pcol] - target).abs() / frame[pcol].std() + (
        frame.residual_corrected_theory_r - 0.30
    ).abs() / frame.residual_corrected_theory_r.std()
    return frame.loc[score.idxmin()]


def _jacobian(model: ModelSpec, parameter: float) -> np.ndarray:
    omega = 2.0 * np.pi * model.omega_hz
    if model.key == "damped":
        gamma = parameter * omega
        return np.array([[0.0, 1.0], [-omega**2, -2.0 * gamma]])
    mu = 2.0 * np.pi * parameter
    return np.array([[mu, -omega], [omega, mu]])


def _transfer(model: ModelSpec, parameter: float, input_pos: int, output_pos: int) -> np.ndarray:
    j = _jacobian(model, parameter)
    b = np.eye(2)[:, input_pos]
    c = np.eye(2)[output_pos]
    result = np.empty(len(FREQ))
    for pos, frequency in enumerate(FREQ):
        response = np.linalg.solve(1j * 2.0 * np.pi * frequency * np.eye(2) - j, b)
        result[pos] = abs(c @ response) ** 2
    return np.maximum(result, 1e-300)


def _input_psd(z: np.ndarray) -> np.ndarray:
    values = np.atleast_2d(z).astype(float)
    center = values[:, 0, None]
    bump_amp = values[:, 1, None]
    beta = values[:, 2, None]
    ap_amp = values[:, 3, None]
    f = FREQ[None, :]
    background = ap_amp * (np.maximum(f, 0.5) / 10.0) ** beta
    bump = bump_amp * np.exp(-0.5 * ((f - center) / 2.0) ** 2)
    return np.maximum(background + bump, 1e-300)


def _aperiodic_psd(beta: np.ndarray, floor_ratio: float) -> np.ndarray:
    beta = np.atleast_1d(beta)[:, None]
    colored = (FREQ[None, :] / 10.0) ** beta
    colored /= colored.mean(axis=1, keepdims=True)
    return np.maximum(colored + floor_ratio**2, 1e-300)


def _batch_features(psd: np.ndarray) -> np.ndarray:
    logs = np.log(np.maximum(np.atleast_2d(psd), 1e-300))
    peak_use = (FREQ >= PEAK_BAND[0]) & (FREQ <= PEAK_BAND[1])
    peak_indices = np.flatnonzero(peak_use)
    local = np.argmax(logs[:, peak_use], axis=1)
    idx = peak_indices[local]
    idx = np.clip(idx, 1, len(FREQ) - 2)
    y1 = logs[np.arange(len(logs)), idx - 1]
    y2 = logs[np.arange(len(logs)), idx]
    y3 = logs[np.arange(len(logs)), idx + 1]
    denom = y1 - 2.0 * y2 + y3
    offset = np.divide(0.5 * (y1 - y3), denom, out=np.zeros_like(denom), where=np.abs(denom) > 1e-14)
    peak = FREQ[idx] + np.clip(offset, -1.0, 1.0) * (FREQ[1] - FREQ[0])
    slope_use = (FREQ >= SLOPE_BAND[0]) & (FREQ <= SLOPE_BAND[1])
    x = np.log(FREQ[slope_use])
    weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    slope = logs[:, slope_use] @ weights
    return np.column_stack([peak, slope])


def _feature_map(transfer: np.ndarray, z: np.ndarray) -> np.ndarray:
    return _batch_features(_input_psd(z) * transfer[None, :])


def _aperiodic_feature_map(transfer: np.ndarray, beta: np.ndarray, floor_ratio: float) -> np.ndarray:
    return _batch_features(_aperiodic_psd(beta, floor_ratio) * transfer[None, :])


def _gradient(fun, z0: np.ndarray, steps: np.ndarray) -> np.ndarray:
    result = np.empty((2, len(z0)))
    for pos, step in enumerate(steps):
        delta = np.zeros_like(z0)
        delta[pos] = step
        result[:, pos] = (fun(z0 + delta)[0] - fun(z0 - delta)[0]) / (2.0 * step)
    return result


def _corr(values: np.ndarray) -> float:
    if len(values) < 4 or not np.isfinite(values).all() or np.min(np.std(values, axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[:, 0], values[:, 1]).statistic)


def _cov_to_corr(cov: np.ndarray) -> float:
    if not np.isfinite(cov).all() or cov[0, 0] <= 0 or cov[1, 1] <= 0:
        return np.nan
    return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))


def _pooled_residual_cov(phases: np.ndarray) -> np.ndarray:
    centered = phases - phases.mean(axis=0, keepdims=True)
    q, n, _ = centered.shape
    return np.einsum("qni,qnj->ij", centered, centered) / (n * (q - 1))


def _identity_metrics(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[keep], np.asarray(y)[keep]
    fit = stats.linregress(x, y)
    return {
        "n": int(len(x)),
        "identity_R2": float(1.0 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)),
        "ols_R2": float(fit.rvalue**2),
        "pearson_r": float(fit.rvalue),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "p": float(fit.pvalue),
    }


def _spectral_residual_cov(psd: np.ndarray, analysis_s: float, seed: int, n: int = 1024) -> np.ndarray:
    """Monte Carlo Welch observation error around a registered population PSD."""
    rng = np.random.default_rng(seed)
    effective_averages = max(3.0, 1.5 * (2.0 * analysis_s / WELCH_SEGMENT_S - 1.0))
    truth = _batch_features(psd[None, :])[0]
    rows = []
    for start in range(0, n, 128):
        count = min(128, n - start)
        multiplier = rng.gamma(effective_averages, 1.0 / effective_averages, size=(count, len(FREQ)))
        noisy_log = ndimage.gaussian_filter1d(
            np.log(np.maximum(psd[None, :] * multiplier, 1e-300)), 1.95, axis=1, mode="nearest"
        )
        rows.append(_batch_features(np.exp(noisy_log)) - truth)
    residual = np.vstack(rows)
    return np.cov(residual, rowvar=False, ddof=1)


def _orthogonal_pair(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    u = rng.normal(size=n)
    u = (u - u.mean()) / u.std(ddof=1)
    v = rng.normal(size=n)
    v -= v.mean() + u * np.dot(u, v) / np.dot(u, u)
    v /= v.std(ddof=1)
    return u, v


def _exp1_trials(target_rho: float, condition: int) -> np.ndarray:
    u, v = _orthogonal_pair(EXP1_TRIALS, SEED + 10000 + condition)
    mixed = target_rho * u + np.sqrt(max(0.0, 1.0 - target_rho**2)) * v
    z = np.tile(Z0, (EXP1_TRIALS, 1))
    z[:, 0] += 2.2 * u
    z[:, 1] += 0.7 * u
    z[:, 2] += 0.32 * mixed
    z[:, 3] += 0.12 * mixed
    z[:, 0] = np.clip(z[:, 0], 5.5, 18.0)
    z[:, 1] = np.clip(z[:, 1], 1.0, 12.0)
    z[:, 2] = np.clip(z[:, 2], -1.4, 0.5)
    z[:, 3] = np.clip(z[:, 3], 0.3, 2.0)
    return z


def _draw_feature_phases(exact: np.ndarray, residual_cov: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return exact[None, :, :] + rng.multivariate_normal(
        np.zeros(2), residual_cov, size=(CAL_PHASES + HELDOUT_PHASES, len(exact))
    )


def _run_exp1(model: ModelSpec, out: Path) -> tuple[pd.DataFrame, np.ndarray, dict]:
    transfers = np.empty((2, 2, len(FREQ)))
    residual_covs = np.empty((2, 2, 2, 2))
    for inp in range(2):
        for read in range(2):
            transfers[inp, read] = _transfer(model, model.exp1_parameter, inp, read)
            base_psd = transfers[inp, read] * _input_psd(Z0)[0]
            residual_covs[inp, read] = _spectral_residual_cov(
                base_psd, EXP1_WINDOW_S, SEED + inp * 100 + read
            )
    fun = lambda z: _feature_map(transfers[0, 0], np.atleast_2d(z))
    gradient = _gradient(fun, Z0, np.array([0.02, 0.02, 0.002, 0.002]))
    target_rhos = np.linspace(-0.995, 0.995, EXP1_CONDITIONS)
    rows = []
    for condition, target_rho in enumerate(target_rhos):
        z = _exp1_trials(float(target_rho), condition)
        exact = fun(z)
        linear = fun(Z0)[0] + (z - Z0) @ gradient.T
        phases = _draw_feature_phases(exact, residual_covs[0, 0], SEED + 200000 + condition)
        cal_cov = _pooled_residual_cov(phases[:CAL_PHASES])
        theory_r = _cov_to_corr(np.cov(linear, rowvar=False, ddof=1) + cal_cov)
        heldout = [_corr(phase) for phase in phases[CAL_PHASES:]]
        rows.append(
            {
                "condition": condition,
                "target_input_rho": target_rho,
                "theory_r": theory_r,
                "simulation_r": float(np.mean(heldout)),
                "simulation_sd": float(np.std(heldout, ddof=1)),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "exp1_1000_conditions.csv", index=False)

    matrix = np.empty((2, 2))
    z = _exp1_trials(-0.55, 5000)
    for inp in range(2):
        for read in range(2):
            exact = _feature_map(transfers[inp, read], z)
            phases = _draw_feature_phases(exact, residual_covs[inp, read], SEED + 300000 + inp * 10 + read)
            matrix[read, inp] = np.mean([_corr(x) for x in phases[CAL_PHASES:]])
    np.savetxt(out / "exp1_2x2_correlation_matrix.csv", matrix, delimiter=",")
    metrics = _identity_metrics(frame.theory_r.to_numpy(), frame.simulation_r.to_numpy())
    return frame, matrix, metrics


def _exp2_parameter_range(model: ModelSpec) -> tuple[float, float]:
    return (0.035, 0.55) if model.key == "damped" else (-8.0, -0.45)


def _run_exp2(model: ModelSpec, out: Path) -> tuple[pd.DataFrame, dict]:
    sampler = qmc.LatinHypercube(d=3, seed=SEED + (0 if model.key == "damped" else 1))
    parameter_range = _exp2_parameter_range(model)
    candidates = qmc.scale(
        sampler.random(5000),
        [parameter_range[0], 0.05, 0.20],
        [parameter_range[1], 0.35, 0.75],
    )
    rows = []
    for condition, (parameter, sigma_beta, floor_ratio) in enumerate(candidates):
        rng = np.random.default_rng(SEED + 500000 + condition)
        low, high = -0.8 / sigma_beta, 0.8 / sigma_beta
        betas = stats.truncnorm.rvs(low, high, loc=0.0, scale=sigma_beta, size=EXP2_TRIALS, random_state=rng)
        transfer = _transfer(model, float(parameter), 0, 0)
        beta_grid = np.linspace(-0.8, 0.8, 81)
        mapped_grid = _aperiodic_feature_map(transfer, beta_grid, float(floor_ratio))
        exact = np.column_stack(
            [np.interp(betas, beta_grid, mapped_grid[:, pos]) for pos in range(2)]
        )
        h = 0.002
        derivative = (
            _aperiodic_feature_map(transfer, np.array([h]), float(floor_ratio))[0]
            - _aperiodic_feature_map(transfer, np.array([-h]), float(floor_ratio))[0]
        ) / (2.0 * h)
        base = _aperiodic_feature_map(transfer, np.array([0.0]), float(floor_ratio))[0]
        first = base + betas[:, None] * derivative[None, :]
        residual_cov = _spectral_residual_cov(
            transfer * _aperiodic_psd(np.array([0.0]), float(floor_ratio))[0],
            EXP2_WINDOW_S,
            SEED + 600000 + condition,
            n=192,
        )
        phases = _draw_feature_phases(exact, residual_cov, SEED + 700000 + condition)
        cal_cov = _pooled_residual_cov(phases[:CAL_PHASES])
        theory_r = _cov_to_corr(np.cov(first, rowvar=False, ddof=1) + cal_cov)
        heldout = [_corr(phase) for phase in phases[CAL_PHASES:]]
        simulation_r = float(np.mean(heldout))
        if np.isfinite(theory_r) and np.isfinite(simulation_r) and theory_r < 0.6:
            rows.append(
                {
                    "condition": condition,
                    model.parameter_name: parameter,
                    "sigma_beta": sigma_beta,
                    "floor_ratio": floor_ratio,
                    "residual_corrected_theory_r": theory_r,
                    "simulation_r": simulation_r,
                    "simulation_sd": float(np.std(heldout, ddof=1)),
                    "valid_trials": EXP2_TRIALS,
                }
            )
        if len(rows) >= EXP2_CONDITIONS:
            break
    if len(rows) < EXP2_CONDITIONS:
        raise RuntimeError(f"Only {len(rows)} low-r conditions found for {model.key}")
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "exp2_1000_conditions_rlt0p6.csv", index=False)
    metrics = _identity_metrics(
        frame.residual_corrected_theory_r.to_numpy(), frame.simulation_r.to_numpy()
    )
    return frame, metrics


def _colored_input_trace(analysis_s: float, z: np.ndarray, seed: int) -> np.ndarray:
    n = int(round(analysis_s / DT))
    rng = np.random.default_rng(seed)
    frequency = np.fft.rfftfreq(n, DT)
    f = np.maximum(frequency, 0.5)
    q = z[3] * (f / 10.0) ** z[2] + z[1] * np.exp(-0.5 * ((f - z[0]) / 2.0) ** 2)
    phase = rng.uniform(0.0, 2.0 * np.pi, len(frequency))
    spectrum = np.sqrt(np.maximum(q, 0.0)) * np.exp(1j * phase)
    spectrum[0] = 0.0
    spectrum[-1] = spectrum[-1].real
    trace = np.fft.irfft(spectrum, n=n)
    trace *= 0.02 / max(trace.std(), 1e-12)
    return trace.astype(np.float64)


@njit
def _integrate_one(model_code: int, parameter: float, omega_hz: float, forcing: np.ndarray, dt: float) -> np.ndarray:
    x = 0.0
    y = 0.0
    omega = 2.0 * np.pi * omega_hz
    out = np.empty(len(forcing), dtype=np.float64)
    for pos in range(len(forcing)):
        u = forcing[pos]
        if model_code == 0:
            gamma = parameter * omega
            dx = y + u
            dy = -2.0 * gamma * y - omega * omega * x
            px = x + dt * dx
            py = y + dt * dy
            dx2 = py + u
            dy2 = -2.0 * gamma * py - omega * omega * px
        else:
            mu = 2.0 * np.pi * parameter
            radius2 = x * x + y * y
            dx = mu * x - omega * y - radius2 * x + u
            dy = omega * x + mu * y - radius2 * y
            px = x + dt * dx
            py = y + dt * dy
            radiusp2 = px * px + py * py
            dx2 = mu * px - omega * py - radiusp2 * px + u
            dy2 = omega * px + mu * py - radiusp2 * py
        x += 0.5 * dt * (dx + dx2)
        y += 0.5 * dt * (dy + dy2)
        out[pos] = x
    return out


def _representative_psd(model: ModelSpec, parameter: float, z: np.ndarray, seed: int, analysis_s: float = 32.768):
    burn_n = int(round(BURN_S / DT))
    forcing = _colored_input_trace(analysis_s + BURN_S, z, seed)
    output = _integrate_one(0 if model.key == "damped" else 1, parameter, model.omega_hz, forcing, DT)
    fs = 1.0 / DT
    nperseg = int(round(WELCH_SEGMENT_S / DT))
    f, pin = signal.welch(forcing[burn_n:], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    _, pout = signal.welch(output[burn_n:], fs=fs, nperseg=nperseg, noverlap=nperseg // 2, window="hann")
    keep = (f >= 0.5) & (f <= 1200.0)
    return f[keep], pin[keep], pout[keep]


def _plot_a(model: ModelSpec, out: Path) -> dict:
    f, pin, pout = _representative_psd(model, model.exp1_parameter, Z0, SEED + 1)
    feature = _batch_features(np.interp(FREQ, f, pout)[None, :])[0]
    slope_mask = (f >= 100.0) & (f <= 1000.0)
    fit = stats.linregress(np.log(f[slope_mask]), np.log(pout[slope_mask]))
    fitted = np.exp(fit.intercept) * f**fit.slope
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    axes[0].loglog(f, pin / pin.max(), "--", color="#D1495B", lw=3.4, label="input PSD")
    axes[0].set(xlabel="Frequency (Hz)", ylabel="Normalized PSD", title="Input: bump + aperiodic")
    axes[0].legend(frameon=False)
    axes[1].loglog(f, pout, "--", color="#222222", lw=3.4, label="ODE PSD")
    axes[1].loglog(f[slope_mask], fitted[slope_mask], color="#86B6C6", lw=5.4, label=f"fit slope={fit.slope:.2f}")
    axes[1].scatter([feature[0]], [np.interp(feature[0], f, pout)], s=70, color="#D1495B", zorder=5, label=f"peak={feature[0]:.1f} Hz")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="PSD")
    axes[1].legend(frameon=False)
    _style_psd_axes(axes)
    _panel_label(axes[0], "A")
    _save(fig, out / "panels" / "A_input_output_psd")
    return {"peak_hz": float(feature[0]), "slope": float(fit.slope)}


def _plot_b(frame: pd.DataFrame, metrics: dict, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.9, 4.4), constrained_layout=True)
    ax.scatter(frame.theory_r, frame.simulation_r, s=17, alpha=0.42, color="#247BA0", edgecolor="none", rasterized=True)
    lo = min(frame.theory_r.min(), frame.simulation_r.min()) - 0.04
    hi = max(frame.theory_r.max(), frame.simulation_r.max()) + 0.04
    ax.plot([lo, hi], [lo, hi], "--", color="#D1495B", lw=3.2, label=f"identity (R2={metrics['identity_R2']:.3f})")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="1st-order theory correlation", ylabel="Simulation correlation", title="Experiment 1: x input, x readout")
    ax.legend(frameon=False, loc="lower right")
    _panel_label(ax, "B")
    _save(fig, out / "panels" / "B_exp1_theory_simulation")


def _plot_c(matrix: np.ndarray, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.3, 3.8), constrained_layout=True)
    image = ax.imshow(matrix, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks([0, 1], ["x", "y"])
    ax.set_yticks([0, 1], ["x", "y"])
    ax.set(xlabel="Input coordinate", ylabel="Readout coordinate", title="Experiment 1: 2 x 2 correlation")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, format_correlation(matrix[row, col]), ha="center", va="center", color="white" if abs(matrix[row, col]) > 0.5 else "black", fontweight="bold")
    cb = fig.colorbar(image, ax=ax, ticks=[-1, 0, 1], pad=0.03)
    cb.set_label("Pearson r")
    _panel_label(ax, "C")
    _save(fig, out / "panels" / "C_exp1_2x2_matrix")


def _plot_d(model: ModelSpec, out: Path) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.7))
    fig.subplots_adjust(left=0.10, right=0.91, top=0.86, bottom=0.25, wspace=0.62)
    if model.key == "damped":
        values_left = np.linspace(2.2, 0.15, 240)
        values_right = np.linspace(0.8, 0.0, 200)
        boundary_left, boundary_right = 1.0, 0.0
        labels = (
            r"Damping ratio $\gamma/\omega_0$",
            r"Damping ratio $\gamma/\omega_0$",
        )
        titles = ("Complex onset", "Marginal boundary")
    else:
        values_left = np.linspace(0.0, 35.0, 240)
        values_right = np.linspace(-8.0, 2.0, 240)
        boundary_left, boundary_right = 0.0, 0.0
        labels = (
            r"Natural frequency $\Omega/(2\pi)$ (Hz)",
            r"Hopf parameter $\mu/(2\pi)$ (Hz)",
        )
        titles = ("Complex-eigenvalue onset", "Hopf reference sweep")
    legend_lines = None
    for panel, (ax, values, boundary, xlabel, title) in enumerate(zip(axes, (values_left, values_right), (boundary_left, boundary_right), labels, titles)):
        real, imag = [], []
        for value in values:
            if model.key == "damped":
                eig = np.linalg.eigvals(_jacobian(model, value))
            elif panel == 0:
                omega = 2.0 * np.pi * value
                mu = 2.0 * np.pi * PHASE_PLANE_REGIMES[model.key][0][1][0]
                eig = np.linalg.eigvals(np.array([[mu, -omega], [omega, mu]]))
            else:
                eig = np.linalg.eigvals(_jacobian(model, value))
            critical = eig[np.argmax(eig.real)]
            real.append(critical.real)
            imag.append(abs(critical.imag))
        line1 = ax.plot(values, real, color="#247BA0", label=r"$\mathrm{Re}(\lambda)$")[0]
        ax2 = ax.twinx()
        line2 = ax2.plot(
            values, imag, color="#D1495B", label=r"$|\mathrm{Im}(\lambda)|$"
        )[0]
        ax.axhline(0, color="0.3", ls="--", lw=0.9)
        ax.axvline(boundary, color="#6A4C93", ls=":", lw=1.2)
        ax.set(
            xlabel=xlabel,
            ylabel=r"$\mathrm{Re}(\lambda)$ (s$^{-1}$)",
            title=title,
        )
        ax2.set_ylabel(r"$|\mathrm{Im}(\lambda)|$ (s$^{-1}$)", color="#D1495B")
        legend_lines = [line1, line2]
        if panel == 0:
            phase_values = (
                [item[1] for item in PHASE_PLANE_REGIMES[model.key]]
                if model.key == "damped"
                else [item[1][1] for item in PHASE_PLANE_REGIMES[model.key]]
            )
            for phase_pos, (phase_value, color) in enumerate(
                zip(phase_values, PHASE_REGIME_COLORS), start=1
            ):
                ax.axvline(
                    phase_value,
                    color=color,
                    ls=(0, (4, 2)),
                    lw=1.35,
                    alpha=0.95,
                    label=f"Phase plane {phase_pos}",
                    zorder=4,
                )
    phase_lines = [axes[0].lines[-3], axes[0].lines[-2], axes[0].lines[-1]]
    all_legend_lines = legend_lines + phase_lines
    fig.legend(
        all_legend_lines,
        [x.get_label() for x in all_legend_lines],
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        fontsize=11.5,
    )
    _panel_label(axes[0], "D")
    _save(fig, out / "panels" / "D_eigenvalue_transitions")
    return {"complex_boundary": boundary_left, "stability_boundary": boundary_right}


def _plot_e(model: ModelSpec, out: Path) -> None:
    if model.key == "damped":
        regimes = [("Overdamped", 2.0, "#247BA0"), ("Stable complex", 0.35, "#2A9D8F"), ("Near marginal", 0.03, "#D1495B")]
    else:
        regimes = [("Far stable focus", -40.0, "#247BA0"), ("Near Hopf", -2.5, "#2A9D8F"), ("Beyond Hopf", 1.0, "#D1495B")]
    f, pin, _ = _representative_psd(model, regimes[0][1], np.array([10.0, 0.0, -0.5, 1.0]), SEED + 20, 16.384)
    fig, axes = plt.subplots(1, 4, figsize=(12.4, 3.4), constrained_layout=True)
    axes[0].loglog(f, pin, color="#222222")
    axes[0].set(title="Common input", xlabel="Frequency (Hz)", ylabel="PSD")
    for ax, (label, parameter, color) in zip(axes[1:], regimes):
        f, _, pout = _representative_psd(model, parameter, np.array([10.0, 0.0, -0.5, 1.0]), SEED + 20, 16.384)
        ax.loglog(f, pout, color=color)
        ax.set(title=label, xlabel="Frequency (Hz)")
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10.0, numticks=3))
        ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    _style_psd_axes(axes)
    _panel_label(axes[0], "E")
    _save(fig, out / "panels" / "E_three_regime_psd")


def _plot_f(model: ModelSpec, out: Path) -> dict:
    z = np.array([10.0, 0.0, -0.5, 1.0])
    f, pin, pout = _representative_psd(model, model.exp2_parameter, z, SEED + 30)
    slope_mask = (f >= 100.0) & (f <= 1000.0)
    fit = stats.linregress(np.log(f[slope_mask]), np.log(pout[slope_mask]))
    fitted = np.exp(fit.intercept) * f**fit.slope
    interp = np.interp(FREQ, f, pout)
    feature = _batch_features(interp[None, :])[0]
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.9), constrained_layout=True)
    axes[0].loglog(f, pin, color="#222222", lw=3.4)
    axes[0].set(xlabel="Frequency (Hz)", ylabel="PSD", title="Aperiodic-only input")
    axes[1].loglog(f, pout, "--", color="#222222", lw=3.4, label="ODE PSD")
    axes[1].loglog(f[slope_mask], fitted[slope_mask], color="#86B6C6", lw=5.4, label=f"fit slope={fit.slope:.2f}")
    axes[1].scatter([feature[0]], [np.interp(feature[0], f, pout)], s=70, color="#D1495B", zorder=5, label=f"peak={feature[0]:.1f} Hz")
    axes[1].set(xlabel="Frequency (Hz)", ylabel="PSD", title="Stable-complex output")
    axes[1].legend(frameon=False)
    _style_psd_axes(axes)
    _panel_label(axes[0], "F")
    _save(fig, out / "panels" / "F_complex_input_output_psd")
    return {"peak_hz": float(feature[0]), "slope": float(fit.slope)}


def _plot_g(frame: pd.DataFrame, metrics: dict, out: Path, model: ModelSpec) -> dict:
    x = frame.residual_corrected_theory_r.to_numpy()
    y = frame.simulation_r.to_numpy()
    fit = stats.linregress(x, y)
    lo, hi = min(x.min(), y.min()) - 0.04, max(x.max(), y.max()) + 0.04
    grid = np.linspace(lo, hi, 200)
    fig, ax = plt.subplots(figsize=(4.9, 4.4), constrained_layout=True)
    ax.scatter(x, y, s=20, color="#247BA0", alpha=0.42, edgecolor="none", rasterized=True)
    representative = _representative_exp2_row(frame, model)
    ax.scatter(
        [representative.residual_corrected_theory_r],
        [representative.simulation_r],
        marker="*", s=230, color=REPRESENTATIVE_COLOR, edgecolor="white",
        linewidth=0.9, zorder=6, label="Representative example",
    )
    ax.plot([lo, hi], [lo, hi], "--", color="#D1495B", lw=3.2, label="identity")
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#222222", lw=3.2, label="OLS")
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="Residual-corrected theory r", ylabel="Finite-window held-out r")
    ax.legend(frameon=False, loc="lower right")
    _panel_label(ax, "G")
    _save(fig, out / "panels" / "G_exp2_1000_conditions")
    return {
        "condition": int(representative.condition),
        "theory_r": float(representative.residual_corrected_theory_r),
        "simulation_r": float(representative.simulation_r),
        "pearson_r": float(fit.rvalue),
        "p": float(fit.pvalue),
        "display_pearson_r": format_correlation(fit.rvalue),
        "display_p": format_p_value(fit.pvalue),
    }


def _combined(out: Path, model: ModelSpec) -> None:
    paths = [
        out / "panels" / "A_input_output_psd.png",
        out / "panels" / "B_exp1_theory_simulation.png",
        out / "panels" / "C_exp1_2x2_matrix.png",
        out / "panels" / "D_eigenvalue_transitions.png",
        out / "panels" / "E_three_regime_psd.png",
        out / "panels" / "F_complex_input_output_psd.png",
        out / "panels" / "G_exp2_1000_conditions.png",
    ]
    images = [plt.imread(path) for path in paths]
    fig = plt.figure(figsize=(18, 15), constrained_layout=True)
    grid = fig.add_gridspec(3, 6, height_ratios=[1.0, 1.0, 1.08])
    slots = [grid[0, :2], grid[0, 2:4], grid[0, 4:], grid[1, :3], grid[1, 3:], grid[2, :3], grid[2, 3:]]
    for image, slot in zip(images, slots):
        ax = fig.add_subplot(slot)
        ax.imshow(image)
        ax.axis("off")
    fig.suptitle(f"{model.title}: periodic-aperiodic coupling validation", fontsize=23, fontweight="bold")
    _save(fig, out / "figures" / f"{model.key}_figure_A_to_G")


def run_model(model: ModelSpec, force: bool = False) -> dict:
    _style()
    out = OUTPUT / model.key
    out.mkdir(parents=True, exist_ok=True)
    exp1_path = out / "exp1_1000_conditions.csv"
    exp2_path = out / "exp2_1000_conditions_rlt0p6.csv"
    if exp1_path.exists() and not force:
        exp1 = pd.read_csv(exp1_path)
        matrix = np.loadtxt(out / "exp1_2x2_correlation_matrix.csv", delimiter=",")
        exp1_metrics = _identity_metrics(exp1.theory_r.to_numpy(), exp1.simulation_r.to_numpy())
    else:
        exp1, matrix, exp1_metrics = _run_exp1(model, out)
    if exp2_path.exists() and not force:
        exp2 = pd.read_csv(exp2_path)
        exp2_metrics = _identity_metrics(exp2.residual_corrected_theory_r.to_numpy(), exp2.simulation_r.to_numpy())
    else:
        exp2, exp2_metrics = _run_exp2(model, out)
    manifest = {
        "model": model.__dict__,
        "equation_input_interpretation": "additive forcing u(t) in dx/dt",
        "dt_s_representative_ode": DT,
        "burn_s": BURN_S,
        "exp1": {"conditions": EXP1_CONDITIONS, "trials_per_condition": EXP1_TRIALS, "window_s": EXP1_WINDOW_S, "metrics": exp1_metrics},
        "exp2": {"conditions": EXP2_CONDITIONS, "trials_per_condition": EXP2_TRIALS, "window_s": EXP2_WINDOW_S, "selection": "residual-corrected theory r < 0.6", "metrics": exp2_metrics},
        "realization_layer": {"calibration_phases": CAL_PHASES, "heldout_phases": HELDOUT_PHASES, "welch_segment_s": WELCH_SEGMENT_S, "method": "Monte Carlo finite-window Welch observation model around the exact 2D resolvent PSD"},
    }
    manifest["panel_A"] = _plot_a(model, out)
    _plot_b(exp1, exp1_metrics, out)
    _plot_c(matrix, out)
    manifest["panel_D"] = _plot_d(model, out)
    _plot_e(model, out)
    manifest["panel_F"] = _plot_f(model, out)
    manifest["panel_G"] = _plot_g(exp2, exp2_metrics, out, model)
    _combined(out, model)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["damped", "hopf", "both"], default="both")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    keys = list(MODELS) if args.model == "both" else [args.model]
    result = {key: run_model(MODELS[key], args.force) for key in keys}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
