from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy import interpolate, stats


ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "Marmoset_Cortex_Model_code"
if str(MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(MODEL_DIR))

from standard_marmoset_simulation import (  # noqa: E402
    apply_standard_parameters,
    build_connectivity_matrix,
    load_data,
)


AREAS9 = ["V1", "V2", "V4", "MST", "LIP", "PG", "A23a", "A6DR", "A6Va"]


@dataclass
class Model:
    params: dict[str, Any]
    J: np.ndarray
    fln: np.ndarray
    areas: list[str]
    full_indices: np.ndarray


def load_model(model_dir: Path | str = MODEL_DIR, areas: list[str] | None = None) -> Model:
    model_dir = Path(model_dir)
    p = apply_standard_parameters(
        load_data(model_dir / "subgraph_data_marmoset.pkl"),
        model_dir / "hier_vals_fit.npy",
    )
    _, j_full = build_connectivity_matrix(p)
    if areas is None:
        idx = np.arange(p["n_area"])
        return Model(p, j_full, p["fln_mat"].copy(), list(p["areas"]), idx)
    idx = np.array([p["areas"].index(a) for a in areas], dtype=int)
    state_idx = np.r_[idx, idx + p["n_area"]]
    p_sub = p.copy()
    p_sub["n_area"] = len(idx)
    p_sub["areas"] = list(areas)
    for key in ("hier_vals", "hier_vals_inh", "exc_scale", "inh_scale", "local_exc_scale", "local_inh_scale"):
        p_sub[key] = np.asarray(p[key])[idx]
    p_sub["fln_mat"] = p["fln_mat"][np.ix_(idx, idx)].copy()
    p_sub["sln_mat"] = p["sln_mat"][np.ix_(idx, idx)].copy()
    _, j_sub = build_connectivity_matrix(p_sub)
    # Rebuilding, rather than slicing J, preserves each selected area's local block
    # while restricting long-range coupling to the induced anatomical subnetwork.
    return Model(p_sub, j_sub, p_sub["fln_mat"].copy(), list(areas), idx)


def edge_to_delta_j(model: Model, target: int, source: int, delta_w: float) -> np.ndarray:
    """Pull an anatomical FLN edge perturbation into block-ordered J."""
    p, n = model.params, len(model.areas)
    d = np.zeros_like(model.J)
    d[target, source] = p["beta_exc"] * p["muEE"] * p["exc_scale"][target] * delta_w / p["tau_exc"]
    d[n + target, source] = p["beta_inh"] * p["muIE"] * p["inh_scale"][target] * delta_w / p["tau_inh"]
    return d


def apply_fln_delta(model: Model, delta_w: np.ndarray) -> np.ndarray:
    j = model.J.copy()
    for target, source in zip(*np.nonzero(delta_w)):
        j += edge_to_delta_j(model, int(target), int(source), float(delta_w[target, source]))
    return j


def is_stable(j: np.ndarray, tol: float = -1e-10) -> bool:
    return bool(np.max(np.linalg.eigvals(j).real) < tol)


def frequency_grid() -> np.ndarray:
    low = np.geomspace(0.5, 40.0, 180)
    high = np.geomspace(40.1, 1200.0, 220)
    return np.unique(np.r_[low, high])


def transfer_power(
    j: np.ndarray,
    input_idx: int,
    output_idx: int,
    freqs_hz: np.ndarray,
) -> np.ndarray:
    """Scalar |c'R(i omega)b|^2; J uses inverse milliseconds."""
    n = j.shape[0]
    b = np.zeros(n)
    b[input_idx] = 1.0
    eye = np.eye(n)
    out = np.empty(len(freqs_hz))
    for k, f in enumerate(freqs_hz):
        r_b = np.linalg.solve(1j * 2 * np.pi * f / 1000.0 * eye - j, b)
        out[k] = abs(r_b[output_idx]) ** 2
    return np.maximum(out, np.finfo(float).tiny)


def input_psd(freqs_hz: np.ndarray, z: np.ndarray, width_hz: float = 2.0) -> np.ndarray:
    """Input z=(bump center Hz, bump amplitude, beta, background amplitude)."""
    center, amp, beta, noise_amp = np.asarray(z, dtype=float)
    f = np.maximum(freqs_hz, 0.5)
    background = noise_amp * (f / 10.0) ** beta
    bump = amp * np.exp(-0.5 * ((f - center) / width_hz) ** 2)
    return np.maximum(background + bump, np.finfo(float).tiny)


def feature_from_spectrum(
    freqs_hz: np.ndarray,
    psd: np.ndarray,
    peak_band: tuple[float, float] = (5.0, 20.0),
    slope_band: tuple[float, float] = (100.0, 1000.0),
) -> dict[str, Any]:
    logf = np.log(freqs_hz)
    logs = np.log(np.maximum(psd, np.finfo(float).tiny))
    peak_mask = (freqs_hz >= peak_band[0]) & (freqs_hz <= peak_band[1])
    idxs = np.flatnonzero(peak_mask)
    # A differentiable cubic spectral interpolant is essential: optimizing a
    # piecewise-linear curve pins the peak to a grid point and destroys dP/dz.
    spline = interpolate.CubicSpline(freqs_hz, logs)
    roots = spline.derivative().roots(extrapolate=False)
    maxima = roots[(roots > peak_band[0]) & (roots < peak_band[1])]
    maxima = maxima[spline.derivative(2)(maxima) < 0]
    if len(maxima) > 1:
        heights=np.sort(spline(maxima))[::-1]
        # Identity switching requires a genuinely competing peak, not an
        # arbitrarily small Welch ripple elsewhere in the registered band.
        peak_switch=bool(heights[0]-heights[1] < np.log(1.20))
    else:
        peak_switch=False
    peak_valid = len(maxima) >= 1 and not peak_switch
    if len(maxima):
        peak = float(maxima[np.argmax(spline(maxima))])
    else:
        peak = float(freqs_hz[idxs[np.argmax(logs[idxs])]])
    boundary = bool(peak <= peak_band[0] * 1.005 or peak >= peak_band[1] * 0.995)
    slope_mask = (freqs_hz >= slope_band[0]) & (freqs_hz <= slope_band[1])
    slope = stats.linregress(logf[slope_mask], logs[slope_mask]).slope
    reason = ""
    if not peak_valid:
        reason = "multiple_peaks" if peak_switch else "no_interior_peak"
    if boundary:
        reason = (reason + ";" if reason else "") + "peak_boundary"
    return {
        "peak_hz": peak,
        "peak_omega": 2 * np.pi * peak,
        "exponent": -float(slope),
        "peak_valid": bool(peak_valid and not boundary),
        "peak_boundary": boundary,
        "peak_switch": peak_switch,
        "feature_failure_reason": reason,
    }


def network_features(
    j: np.ndarray,
    z: np.ndarray,
    input_idx: int,
    output_idx: int,
    freqs_hz: np.ndarray,
    transfer: np.ndarray | None = None,
    peak_band: tuple[float, float] = (5.0, 20.0),
    slope_band: tuple[float, float] = (100.0, 1000.0),
) -> tuple[dict[str, Any], np.ndarray]:
    if transfer is None:
        transfer = transfer_power(j, input_idx, output_idx, freqs_hz)
    psd = transfer * input_psd(freqs_hz, z)
    return feature_from_spectrum(freqs_hz, psd, peak_band, slope_band), psd


def finite_derivatives(fun: Callable[[np.ndarray], np.ndarray], x: np.ndarray, h: float = 2e-3):
    """Central finite gradient/Hessian/third tensor for a small input vector."""
    x = np.asarray(x, float)
    q = len(x)
    y0 = np.atleast_1d(fun(x))
    m = len(y0)
    grad = np.zeros((m, q))
    hess = np.zeros((m, q, q))
    third = np.zeros((m, q, q, q))
    eye = np.eye(q)
    for i in range(q):
        fp, fm = np.atleast_1d(fun(x + h * eye[i])), np.atleast_1d(fun(x - h * eye[i]))
        grad[:, i] = (fp - fm) / (2 * h)
        hess[:, i, i] = (fp - 2 * y0 + fm) / h**2
        f2p, f2m = np.atleast_1d(fun(x + 2*h*eye[i])), np.atleast_1d(fun(x - 2*h*eye[i]))
        third[:, i, i, i] = (f2p - 2*fp + 2*fm - f2m) / (2*h**3)
    for i in range(q):
        for j in range(i + 1, q):
            fpp = np.atleast_1d(fun(x + h*eye[i] + h*eye[j]))
            fpm = np.atleast_1d(fun(x + h*eye[i] - h*eye[j]))
            fmp = np.atleast_1d(fun(x - h*eye[i] + h*eye[j]))
            fmm = np.atleast_1d(fun(x - h*eye[i] - h*eye[j]))
            hess[:, i, j] = hess[:, j, i] = (fpp - fpm - fmp + fmm) / (4*h**2)
    # Third mixed derivatives are derivatives of the Hessian. This is slower but
    # robust for q=3 and avoids differentiating a discrete peak index.
    def hessian_at(xx):
        hh = np.zeros((m, q, q))
        yy = np.atleast_1d(fun(xx))
        for a in range(q):
            fp, fm = np.atleast_1d(fun(xx+h*eye[a])), np.atleast_1d(fun(xx-h*eye[a]))
            hh[:, a, a] = (fp - 2*yy + fm)/h**2
            for b in range(a+1, q):
                vals = [np.atleast_1d(fun(xx+s*h*eye[a]+t*h*eye[b])) for s,t in ((1,1),(1,-1),(-1,1),(-1,-1))]
                hh[:, a, b] = hh[:, b, a] = (vals[0]-vals[1]-vals[2]+vals[3])/(4*h**2)
        return hh
    for k in range(q):
        third[:, :, :, k] = (hessian_at(x+h*eye[k])-hessian_at(x-h*eye[k]))/(2*h)
    # Numerical noise can break tensor symmetry; the theory uses the symmetric part.
    for out in range(m):
        t = third[out]
        third[out] = sum(t.transpose(p) for p in ((0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0))) / 6
    return grad, hess, third


def central_moments(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    xc = x - x.mean(axis=0)
    return np.cov(xc, rowvar=False, ddof=1), np.einsum("ni,nj,nk->ijk", xc, xc, xc) / len(xc)


def theory_covariances(
    grad: np.ndarray, hess: np.ndarray, third: np.ndarray, sigma: np.ndarray, m3: np.ndarray
) -> dict[str, float]:
    p, a = grad
    hp, ha = hess
    tp, ta = third
    c49 = float(p @ sigma @ a)
    c3 = 0.5 * np.einsum("ijk,ijk", m3, np.einsum("i,jk->ijk", p, ha) + np.einsum("i,jk->ijk", a, hp))
    cq = c49 + 0.5 * np.trace(hp @ sigma @ ha @ sigma)
    ta_sigma = np.einsum("ijk,jk->i", ta, sigma)
    tp_sigma = np.einsum("ijk,jk->i", tp, sigma)
    clc = 0.5 * (p @ sigma @ ta_sigma + a @ sigma @ tp_sigma)
    return {"C49": c49, "C48": c49 + float(c3), "Cquad": float(cq), "C64": float(cq + clc), "third_term": float(c3), "linear_cubic": float(clc)}


def correlation_summary(p: np.ndarray, a: np.ndarray, bootstrap: int, rng: np.random.Generator) -> dict[str, float]:
    ok = np.isfinite(p) & np.isfinite(a)
    p, a = p[ok], a[ok]
    cov = float(np.cov(p, a, ddof=1)[0, 1])
    pearson = float(stats.pearsonr(p, a).statistic) if len(p) > 2 else np.nan
    spearman = float(stats.spearmanr(p, a).statistic) if len(p) > 2 else np.nan
    boots = np.empty(bootstrap)
    for i in range(bootstrap):
        ii = rng.integers(0, len(p), len(p))
        boots[i] = stats.pearsonr(p[ii], a[ii]).statistic
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    return {"covariance": cov, "pearson_r": pearson, "spearman_r": spearman, "pearson_ci_low": float(lo), "pearson_ci_high": float(hi), "n_valid": int(len(p))}


def prediction_metrics(obs: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = np.asarray(obs)[ok], np.asarray(pred)[ok]
    reg = stats.linregress(pred, obs) if len(obs) > 2 and np.std(pred) > 0 else None
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err**2)))
    return {
        "pearson_r": float(stats.pearsonr(obs, pred).statistic) if len(obs) > 2 and np.std(pred) > 0 else np.nan,
        "spearman_r": float(stats.spearmanr(obs, pred).statistic) if len(obs) > 2 and np.std(pred) > 0 else np.nan,
        "R2": float(1 - np.sum(err**2) / np.sum((obs-obs.mean())**2)) if np.var(obs) > 0 else np.nan,
        "slope": float(reg.slope) if reg else np.nan,
        "intercept": float(reg.intercept) if reg else np.nan,
        "MAE": float(np.mean(abs(err))), "RMSE": rmse,
        "NRMSE": float(rmse/(np.std(obs)+1e-15)),
        "sign_accuracy": float(np.mean(np.sign(obs) == np.sign(pred))),
        "n": int(len(obs)),
    }


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=lambda x: float(x) if np.isscalar(x) else str(x)))


def save_figure(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
