from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats

from .core import finite_derivatives, input_psd, load_model, transfer_power
from .exp1 import Z0, ZHI, ZLO, ZSCALE, _feature_fun
from .exp2_pap_correlation_validation import (
    ANALYSIS_S as EXP2_ANALYSIS_S,
    BURN_S as EXP2_BURN_S,
    DT_S as EXP2_DT_S,
    _extract_features as exp2_extract_features,
    _forced_ei_batch_parallel,
    _forcing as exp2_forcing,
    _theory_feature as exp2_theory_feature,
    _transfer_power as exp2_transfer_power,
)
from .exp2_two_stage import _jacobian as exp2_jacobian
from .exp2_two_stage import _matrix as exp2_matrix
from .exp2_two_stage import _setup as exp2_setup
from .exp3_large_delta_screen import _discrete_transfer
from .io import load_config
from .paper_figure_node_deletion import make_combined
from .plot_formatting import format_p_value
from .realization_residual_validation import _corr, _cov_to_corr, _pooled_within_trial_cov


EXP1_CONDITIONS = 1000
EXP1_TRIALS = 64
EXP1_EPSILON = 0.20
EXP1_ANALYSIS_S = 32.768
DT_MS = 0.25
WELCH_SEGMENT_S = 4.096
SMOOTHING_SIGMA_HZ = 0.9765625

EXP2_CONDITIONS = 20
EXP2_TRIALS = 100
EXP2_BETA_MIN = -0.50
EXP2_BETA_MAX = 0.70
EXP2_BETA_SD = 0.08
EXP2_S = 6.0
EXP2_INPUT_SIGMA = 0.02
EXP2_FLOOR_RATIO = 0.10
EXP2_CALIBRATION_PHASES = 3
EXP2_HELDOUT_PHASES = 5

SEED = 20260819


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def _paths(cfg: dict) -> tuple[Path, Path]:
    output_root = _rooted(cfg["output_dir"])
    dense = output_root / "paper_figure_dense_revision"
    paper = output_root / "paper_figure_node_deletion"
    dense.mkdir(parents=True, exist_ok=True)
    (dense / "exp2_beta_mean_cache").mkdir(exist_ok=True)
    return dense, paper


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 1.2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "savefig.dpi": 300,
        }
    )


def _sample_exact_correlation(n: int, rho: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = rng.normal(size=n)
    x = (x - x.mean()) / x.std(ddof=1)
    y -= y.mean()
    y -= x * np.dot(x, y) / np.dot(x, x)
    y /= y.std(ddof=1)
    return x, rho * x + np.sqrt(max(1.0 - rho * rho, 0.0)) * y


def _exp1_residual_covariance(output_root: Path) -> np.ndarray:
    phase_dir = output_root / "realization_residual_validation" / "exp1_phase_cache"
    pg_pos = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"].index("PG")
    phases = []
    valids = []
    for phase in range(3):
        data = np.load(phase_dir / f"phase{phase:02d}_input_V1.npz")
        phases.append(data["features"][:, pg_pos])
        valids.append(data["valid"][:, pg_pos])
    values = np.asarray(phases, float)
    valid = np.all(np.asarray(valids), axis=0)
    flattened = np.stack([values[p][valid] for p in range(3)])
    return _pooled_within_trial_cov(flattened)


def _welch_features(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fs = 1000.0 / DT_MS
    nperseg = int(round(WELCH_SEGMENT_S * fs))
    freq, psd = signal.welch(
        paths,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    keep = (freq >= 0.5) & (freq <= 1200.0)
    f = freq[keep]
    sigma_bins = SMOOTHING_SIGMA_HZ / (f[1] - f[0])
    smooth = np.exp(
        ndimage.gaussian_filter1d(
            np.log(np.maximum(psd[:, keep], 1e-300)), sigma_bins, axis=-1
        )
    )

    peak_band = (f >= 5.0) & (f <= 20.0)
    peak_ids = np.flatnonzero(peak_band)
    logp = np.log(np.maximum(smooth, 1e-300))
    band_logp = logp[:, peak_band]
    local_maxima = (band_logp[:, 1:-1] > band_logp[:, :-2]) & (
        band_logp[:, 1:-1] > band_logp[:, 2:]
    )
    scores = np.where(local_maxima, band_logp[:, 1:-1], -np.inf)
    local = np.argmax(scores, axis=1) + 1
    ids = peak_ids[local]
    valid = local_maxima.any(axis=1)
    rows = np.arange(len(paths))
    y0 = logp[rows, np.maximum(ids - 1, 0)]
    y1 = logp[rows, ids]
    y2 = logp[rows, np.minimum(ids + 1, len(f) - 1)]
    denom = y0 - 2.0 * y1 + y2
    offset = np.divide(0.5 * (y0 - y2), denom, out=np.zeros_like(denom), where=np.abs(denom) > 1e-12)
    offset = np.clip(offset, -1.0, 1.0)
    peak = f[ids] + offset * (f[1] - f[0])

    slope_band = (f >= 100.0) & (f <= 1000.0)
    x = np.log(f[slope_band])
    xc = x - x.mean()
    slopes = np.sum(logp[:, slope_band] * xc[None, :], axis=1) / np.sum(xc * xc)
    values = np.column_stack([2.0 * np.pi * peak, -slopes])
    valid &= np.isfinite(values).all(axis=1)
    return values, valid


def _exp1_simulation_features(z: np.ndarray, discrete_transfer: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n_steps = int(round(EXP1_ANALYSIS_S * 1000.0 / DT_MS))
    freq = np.fft.rfftfreq(n_steps, d=DT_MS / 1000.0)
    active = (freq < 2000.0) & (freq > 0.0)
    f = np.maximum(freq, 0.5)[None, :]
    center = z[:, 0, None]
    amp = z[:, 1, None]
    beta = z[:, 2, None]
    background_amp = z[:, 3, None]
    q = background_amp * (f / 10.0) ** beta
    q += amp * np.exp(-0.5 * ((f - center) / 2.0) ** 2)
    q[:, ~active] = 0.0
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, q.shape)
    spectra = np.sqrt(np.maximum(q, 0.0)) * np.exp(1j * phase)
    spectra[:, ~active] = 0.0
    paths = np.fft.irfft(spectra * discrete_transfer[None, :], n=n_steps, axis=-1).astype(np.float32)
    return _welch_features(paths)


def run_exp1_dense(cfg: dict, dense: Path, paper: Path, force: bool = False, limit: int | None = None) -> pd.DataFrame:
    result_path = dense / "exp1_dense_conditions.csv"
    existing = pd.DataFrame()
    if result_path.exists() and not force:
        existing = pd.read_csv(result_path)
    completed = set(existing.condition_id.astype(int)) if len(existing) else set()

    model = load_model(_rooted(cfg["model_dir"]))
    input_idx = model.areas.index("V1")
    output_idx = model.areas.index("PG")
    n_steps = int(round(EXP1_ANALYSIS_S * 1000.0 / DT_MS))
    discrete = _discrete_transfer(model.J, input_idx, output_idx, n_steps, float(model.params["tau_exc"]))
    f_theory = np.unique(np.r_[np.geomspace(0.5, 40.0, 180), np.geomspace(40.1, 1200.0, 220)])
    power = transfer_power(model.J, input_idx, output_idx, f_theory)
    fun = _feature_fun(power, f_theory, tuple(cfg["peak_band_hz"]), tuple(cfg["slope_band_hz"]))
    grad, _, _ = finite_derivatives(fun, np.zeros(4), h=2e-3)
    base = fun(np.zeros(4))
    residual_cov = _exp1_residual_covariance(_rooted(cfg["output_dir"]))

    n_conditions = EXP1_CONDITIONS if limit is None else min(EXP1_CONDITIONS, limit)
    targets = np.linspace(-0.995, 0.995, EXP1_CONDITIONS)
    rows = existing.to_dict("records") if len(existing) else []
    for condition_id in range(n_conditions):
        if condition_id in completed:
            continue
        rho = float(targets[condition_id])
        x, y = _sample_exact_correlation(EXP1_TRIALS, rho, SEED + condition_id * 17)
        u = np.zeros((EXP1_TRIALS, 4))
        u[:, 0] = EXP1_EPSILON * x
        u[:, 2] = EXP1_EPSILON * y
        raw = Z0 + u * ZSCALE
        z = np.clip(raw, ZLO, ZHI)
        u = (z - Z0) / ZSCALE
        first = base + u @ grad.T
        signal_cov = np.cov(first, rowvar=False, ddof=1)
        predicted = _cov_to_corr(signal_cov + residual_cov)
        simulated, valid = _exp1_simulation_features(z, discrete, SEED + 100000 + condition_id * 31)
        use = valid & np.isfinite(first).all(axis=1)
        rows.append(
            {
                "condition_id": condition_id,
                "target_input_r": rho,
                "sample_input_r": float(stats.pearsonr(u[:, 0], u[:, 2]).statistic),
                "n_trials": EXP1_TRIALS,
                "n_valid": int(use.sum()),
                "first_latent_r": _corr(first[use]),
                "first_residual_corrected_r": predicted,
                "simulation_r": _corr(simulated[use]),
            }
        )
        if (condition_id + 1) % 10 == 0 or condition_id + 1 == n_conditions:
            frame = pd.DataFrame(rows).drop_duplicates("condition_id", keep="last").sort_values("condition_id")
            frame.to_csv(result_path, index=False)
            print(f"Experiment 1 dense conditions {condition_id + 1}/{n_conditions}", flush=True)

    result = pd.read_csv(result_path)
    if len(result) >= EXP1_CONDITIONS:
        plot_exp1_dense(result, paper)
    np.savez_compressed(dense / "exp1_dense_metadata.npz", residual_covariance=residual_cov, gradient=grad)
    return result


def plot_exp1_dense(data: pd.DataFrame, paper: Path) -> dict:
    data = data.dropna(subset=["first_residual_corrected_r", "simulation_r"])
    x = data.first_residual_corrected_r.to_numpy()
    y = data.simulation_r.to_numpy()
    fit = stats.linregress(x, y)
    identity_r2 = 1.0 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
    fig, ax = plt.subplots(figsize=(4.5, 4.0), constrained_layout=True)
    ax.scatter(x, y, s=18, color="#247BA0", alpha=.42, edgecolor="none", rasterized=True)
    ax.plot([-1, 1], [-1, 1], color="#D1495B", ls="--", lw=3.0,
            label=f"identity (R2={identity_r2:.3f})")
    ax.set(
        xlim=(-1.02, 1.02),
        ylim=(-1.02, 1.02),
        xlabel="1st-order theory correlation",
        ylabel="Simulation correlation",
        title="Experiment 1: V1 input, PG readout",
    )
    ax.set_xticks([-1, -.5, 0, .5, 1])
    ax.set_yticks([-1, -.5, 0, .5, 1])
    ax.legend(frameon=False, loc="lower right", fontsize=12.5)
    ax.tick_params(labelsize=13, width=1.7)
    ax.xaxis.label.set_size(14)
    ax.yaxis.label.set_size(14)
    ax.title.set_size(15)
    for spine in ax.spines.values():
        spine.set_linewidth(1.9)
    ax.text(-.18, 1.18, "B", transform=ax.transAxes, fontsize=23, fontweight="bold")
    _save(fig, paper / "panels" / "B_exp1_V1_PG_theory_simulation")
    return {"identity_R2": float(identity_r2), "ols_R2": float(fit.rvalue**2), "p": float(fit.pvalue)}


def _truncated_beta_trials(mean: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = (-0.8 - mean) / EXP2_BETA_SD
    b = (0.8 - mean) / EXP2_BETA_SD
    return stats.truncnorm.rvs(a, b, loc=mean, scale=EXP2_BETA_SD, size=EXP2_TRIALS, random_state=rng)


def run_exp2_beta_means(cfg: dict, dense: Path, paper: Path, force: bool = False, limit: int | None = None) -> pd.DataFrame:
    result_path = dense / "exp2_beta_mean_conditions.csv"
    original, tau, _, _ = exp2_setup()
    w = exp2_matrix(original, EXP2_S, 1.0)
    j = exp2_jacobian(original, tau, EXP2_S, 1.0)
    f_theory = np.geomspace(0.5, 1200.0, 1600)
    transfer = exp2_transfer_power(j, float(tau[0]), f_theory)
    beta_means = np.linspace(EXP2_BETA_MIN, EXP2_BETA_MAX, EXP2_CONDITIONS)
    n_conditions = EXP2_CONDITIONS if limit is None else min(EXP2_CONDITIONS, limit)
    rows = []
    for condition_id, beta_mean in enumerate(beta_means[:n_conditions]):
        cache = dense / "exp2_beta_mean_cache" / f"condition_{condition_id:02d}.npz"
        betas = _truncated_beta_trials(float(beta_mean), SEED + 500000 + condition_id)
        if cache.exists() and not force:
            saved = np.load(cache)
            features = saved["features"]
            valid = saved["valid"]
            betas = saved["betas"]
        else:
            n_phases = EXP2_CALIBRATION_PHASES + EXP2_HELDOUT_PHASES
            features = np.empty((n_phases, EXP2_TRIALS, 2), dtype=float)
            valid = np.empty((n_phases, EXP2_TRIALS), dtype=bool)
            for phase in range(n_phases):
                forcing = exp2_forcing(
                    betas,
                    EXP2_INPUT_SIGMA,
                    EXP2_FLOOR_RATIO,
                    SEED + 600000 + condition_id * 1000 + phase * 100000,
                )
                paths = _forced_ei_batch_parallel(
                    w, tau, forcing, EXP2_DT_S, int(round(EXP2_BURN_S / EXP2_DT_S))
                )
                features[phase], valid[phase], _ = exp2_extract_features(paths)
                print(
                    f"Experiment 2 beta mean {condition_id + 1}/{n_conditions}, phase {phase + 1}/{n_phases}",
                    flush=True,
                )
            np.savez_compressed(cache, betas=betas, features=features, valid=valid)

        theory = np.asarray(
            [
                exp2_theory_feature(
                    transfer, f_theory, beta, EXP2_INPUT_SIGMA, EXP2_FLOOR_RATIO
                )[0]
                for beta in betas
            ]
        )
        use = valid.all(axis=0) & np.isfinite(theory).all(axis=1)
        observed = features[:, use]
        residual_cov = _pooled_within_trial_cov(observed[:EXP2_CALIBRATION_PHASES])
        signal_cov = np.cov(theory[use], rowvar=False, ddof=1)
        heldout = np.asarray([_corr(phase) for phase in observed[EXP2_CALIBRATION_PHASES:]])
        rows.append(
            {
                "condition_id": condition_id,
                "beta_mean": float(beta_mean),
                "beta_sd": EXP2_BETA_SD,
                "n_trials": EXP2_TRIALS,
                "n_valid": int(use.sum()),
                "exact_mapping_r": _corr(theory[use]),
                "theory_plus_realization_r": _cov_to_corr(signal_cov + residual_cov),
                "simulation_r": float(np.nanmean(heldout)),
                "simulation_r_sd": float(np.nanstd(heldout, ddof=1)),
                "heldout_phase_r_json": json.dumps(heldout.tolist()),
            }
        )
        pd.DataFrame(rows).to_csv(result_path, index=False)
    result = pd.read_csv(result_path)
    if len(result) >= EXP2_CONDITIONS:
        plot_exp2_beta_means(result, paper)
    return result


def plot_exp2_beta_means(data: pd.DataFrame, paper: Path) -> dict:
    data = data.dropna(subset=["theory_plus_realization_r", "simulation_r"])
    x = data.theory_plus_realization_r.to_numpy()
    y = data.simulation_r.to_numpy()
    fit = stats.linregress(x, y)
    identity_r2 = 1.0 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
    lo = min(x.min(), y.min()) - .05
    hi = max(x.max(), y.max()) + .05
    fig, ax = plt.subplots(figsize=(4.5, 3.9), constrained_layout=True)
    points = ax.scatter(
        x,
        y,
        c=data.beta_mean,
        cmap="coolwarm",
        vmin=EXP2_BETA_MIN,
        vmax=EXP2_BETA_MAX,
        s=52,
        edgecolor="white",
        linewidth=.5,
        zorder=3,
    )
    ax.errorbar(x, y, yerr=data.simulation_r_sd, fmt="none", ecolor="0.4", lw=1, capsize=2, zorder=2)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.5, label="identity")
    grid = np.linspace(lo, hi, 100)
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#D1495B", lw=1.8, label="OLS")
    ax.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="Theory + realization correlation",
        ylabel="Nonlinear simulation correlation",
        title="Experiment 2: 20 beta-mean conditions",
    )
    ax.text(
        .96,
        .96,
        f"100 trials/condition\nidentity R2={identity_r2:.3f}\nOLS R2={fit.rvalue**2:.3f}\np={format_p_value(fit.pvalue)}",
        transform=ax.transAxes,
        va="top",
        ha="right",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .86, "pad": 2.5},
    )
    ax.legend(frameon=False, loc="lower right")
    cb = fig.colorbar(points, ax=ax, shrink=.82, pad=.02)
    cb.set_label("Mean beta")
    cb.set_ticks([EXP2_BETA_MIN, 0.0, EXP2_BETA_MAX])
    ax.text(-.18, 1.06, "G", transform=ax.transAxes, fontsize=20, fontweight="bold")
    _save(fig, paper / "panels" / "G_exp2_1000trial_theory_simulation")
    return {"identity_R2": float(identity_r2), "ols_R2": float(fit.rvalue**2), "p": float(fit.pvalue)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--part", choices=("all", "exp1", "exp2", "assemble"), default="all")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    dense, paper = _paths(cfg)
    _style()
    if args.part in ("all", "exp1"):
        run_exp1_dense(cfg, dense, paper, force=args.force, limit=args.limit)
    if args.part in ("all", "exp2"):
        run_exp2_beta_means(cfg, dense, paper, force=args.force, limit=args.limit)
    if args.part in ("all", "assemble"):
        make_combined(paper)
    print(dense)


if __name__ == "__main__":
    main()
