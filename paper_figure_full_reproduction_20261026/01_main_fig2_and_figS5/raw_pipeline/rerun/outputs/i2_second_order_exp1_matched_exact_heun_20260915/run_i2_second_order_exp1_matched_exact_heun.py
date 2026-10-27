from __future__ import annotations

"""Figure 2i2 with a second-order input-parameter feature expansion.

Only the Taylor order changes relative to the corrected first-order analysis.
The frozen design, complete nonlinear Heun simulations, calibration residual
estimator, common-valid masks, held-out statistic, and delta definition are
reused without alteration.
"""

import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIRST_DIR = ROOT / "outputs/i2_exp1_matched_exact_heun_20260914"
FIRST_SCRIPT = FIRST_DIR / "run_i2_exp1_matched_exact_heun.py"
FIRST_RESULTS = FIRST_DIR / "i2_exp1_matched_results.csv"
FIRST_METRICS = FIRST_DIR / "i2_exp1_matched_metrics.json"
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(PACKAGE), str(FIGURE_SKILL / "scripts")]

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from pap_validation.core import finite_derivatives  # noqa: E402


H = 0.002
H_AUDIT = (0.004, 0.001)
PROTOCOL = "i2_exp1_matched_second_order_complete_nonlinear_heun_v1"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


first = load_module("i2_first_order_reference", FIRST_SCRIPT)


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hessian_only(fun, x: np.ndarray, h: float) -> np.ndarray:
    """The Hessian stencil used by pap_validation.core.finite_derivatives."""
    x = np.asarray(x, float)
    q = len(x)
    y0 = np.atleast_1d(fun(x))
    hess = np.zeros((len(y0), q, q), float)
    eye = np.eye(q)
    for i in range(q):
        plus = np.atleast_1d(fun(x + h * eye[i]))
        minus = np.atleast_1d(fun(x - h * eye[i]))
        hess[:, i, i] = (plus - 2 * y0 + minus) / h**2
    for i in range(q):
        for j in range(i + 1, q):
            fpp = np.atleast_1d(fun(x + h * eye[i] + h * eye[j]))
            fpm = np.atleast_1d(fun(x + h * eye[i] - h * eye[j]))
            fmp = np.atleast_1d(fun(x - h * eye[i] + h * eye[j]))
            fmm = np.atleast_1d(fun(x - h * eye[i] - h * eye[j]))
            mixed = (fpp - fpm - fmp + fmm) / (4 * h**2)
            hess[:, i, j] = hess[:, j, i] = mixed
    return hess


def second_order_map(jacobian: np.ndarray, input_idx: int, output_idx: int, source) -> dict:
    transfer = source._all_transfer_power(
        jacobian, [input_idx, output_idx], first.THEORY_FREQUENCIES
    )[0, 1]
    fun = lambda u: first.feature_map(transfer, u)
    zero = np.zeros(4)
    base = fun(zero)
    gradient, hessian, _unused_third = finite_derivatives(fun, zero, h=H)
    audited = {step: hessian_only(fun, zero, step) for step in H_AUDIT}
    return {
        "base": base,
        "gradient": gradient,
        "hessian": hessian,
        "h004_vs_h002": float(
            np.linalg.norm(audited[0.004] - hessian)
            / (np.linalg.norm(hessian) + 1e-15)
        ),
        "h002_vs_h001": float(
            np.linalg.norm(hessian - audited[0.001])
            / (np.linalg.norm(audited[0.001]) + 1e-15)
        ),
    }


def taylor2(base: np.ndarray, gradient: np.ndarray, hessian: np.ndarray, u: np.ndarray) -> np.ndarray:
    return (
        base[None, :]
        + u @ gradient.T
        + 0.5 * np.einsum("ni,kij,nj->nk", u, hessian, u)
    )


def assemble() -> tuple[pd.DataFrame, dict]:
    runner = first.load_module("i2_second_order_source", first.I2_RUNNER)
    model, patterns, z, maps, design = runner.build_context()
    u = np.asarray(design["u"], float)
    if len(patterns) != 82 or len(z) != 128:
        raise RuntimeError("Registered patterns or frozen trial count changed")
    if not np.allclose((z - first.Z0) / first.ZSCALE, u, rtol=0, atol=2e-15):
        raise RuntimeError("Frozen z/u parameterization changed")

    input_idx = model.areas.index(runner.source.INPUT_AREA)
    output_idx = model.areas.index(runner.source.OUTPUT_AREA)
    maps2 = []
    derivative_rows = []
    for network, jacobian in enumerate(maps["jacobians"]):
        item = second_order_map(jacobian, input_idx, output_idx, runner.source)
        maps2.append(item)
        derivative_rows.append({
            "network_position": network,
            "fixedpoint_residual": float(maps["fixedpoint_residual"][network]),
            "active_E": int(maps["active_e"][network]),
            "active_I": int(maps["active_i"][network]),
            "hessian_h004_vs_h002": item["h004_vs_h002"],
            "hessian_h002_vs_h001": item["h002_vs_h001"],
        })
        if network % 10 == 0 or network == 82:
            print(f"second-order derivatives: {network + 1}/83", flush=True)

    bases = np.asarray([item["base"] for item in maps2])
    gradients = np.asarray([item["gradient"] for item in maps2])
    hessians = np.asarray([item["hessian"] for item in maps2])
    np.savez_compressed(
        HERE / "i2_exp1_second_order_maps.npz",
        base=bases,
        gradient_u=gradients,
        hessian_u=hessians,
        gradient_theta=gradients / first.ZSCALE[None, None, :],
        hessian_theta=(
            hessians
            / first.ZSCALE[None, None, :, None]
            / first.ZSCALE[None, None, None, :]
        ),
        theta0=first.Z0,
        theta_scale=first.ZSCALE,
        h=np.asarray(H),
    )
    with np.load(FIRST_DIR / "i2_exp1_first_order_maps.npz") as first_maps:
        base_max_abs_difference = float(np.max(np.abs(bases - first_maps["base"])))
        gradient_max_abs_difference = float(
            np.max(np.abs(gradients - first_maps["gradient_u"]))
        )
    if base_max_abs_difference != 0 or gradient_max_abs_difference != 0:
        raise RuntimeError("The retained zeroth/first-order map changed")
    derivatives = pd.DataFrame(derivative_rows)
    derivatives.to_csv(HERE / "i2_exp1_second_order_derivative_diagnostics.csv", index=False)

    # Read-only reuse of the corrected complete nonlinear Heun feature cache.
    features, feature_valid, rectification = first.load_exact_features()
    reference = pd.read_csv(FIRST_RESULTS).set_index("network_position")
    rows = []
    exclusions = []
    residual_diffs = []
    simulation_diffs = []
    common_n_diffs = []
    for position, pattern in enumerate(patterns.itertuples(index=False), 1):
        theoretical_intact = taylor2(bases[0], gradients[0], hessians[0], u)
        theoretical_deleted = taylor2(
            bases[position], gradients[position], hessians[position], u
        )
        common = (
            np.isfinite(theoretical_intact).all(axis=1)
            & np.isfinite(theoretical_deleted).all(axis=1)
            & feature_valid[0].all(axis=0)
            & feature_valid[position].all(axis=0)
        )
        if common.sum() < 8:
            exclusions.append({
                "network_position": position,
                "deleted_nodes": pattern.deleted_nodes,
                "reason": f"fewer than 8 common trials ({int(common.sum())})",
            })
            continue

        intact_signal_cov = np.cov(theoretical_intact[common], rowvar=False, ddof=1)
        deleted_signal_cov = np.cov(theoretical_deleted[common], rowvar=False, ddof=1)
        intact_residual, intact_trials, intact_degrees = first.pooled_within_trial_cov(
            features[0, : first.N_CALIBRATION],
            feature_valid[0, : first.N_CALIBRATION],
            common,
        )
        deleted_residual, deleted_trials, deleted_degrees = first.pooled_within_trial_cov(
            features[position, : first.N_CALIBRATION],
            feature_valid[position, : first.N_CALIBRATION],
            common,
        )
        theory_intact = first.cov_to_corr(intact_signal_cov + intact_residual)
        theory_deleted = first.cov_to_corr(deleted_signal_cov + deleted_residual)
        simulation_intact = first.corr(features[0, first.N_CALIBRATION], common)
        simulation_deleted = first.corr(features[position, first.N_CALIBRATION], common)

        prior = reference.loc[position]
        common_n_diffs.append(abs(int(common.sum()) - int(prior.common_n)))
        simulation_diffs.extend([
            abs(simulation_intact - prior.simulation_intact_r),
            abs(simulation_deleted - prior.simulation_deleted_r),
        ])
        residual_diffs.extend([
            abs(intact_residual[0, 0] - prior.intact_residual_var_P),
            abs(intact_residual[1, 1] - prior.intact_residual_var_A),
            abs(intact_residual[0, 1] - prior.intact_residual_cov_PA),
            abs(deleted_residual[0, 0] - prior.deleted_residual_var_P),
            abs(deleted_residual[1, 1] - prior.deleted_residual_var_A),
            abs(deleted_residual[0, 1] - prior.deleted_residual_cov_PA),
        ])

        row = pattern._asdict()
        row.update({
            "network_position": position,
            "common_n": int(common.sum()),
            "theory_intact_r": theory_intact,
            "theory_deleted_r": theory_deleted,
            "theory_delta_r": theory_intact - theory_deleted,
            "simulation_intact_r": simulation_intact,
            "simulation_deleted_r": simulation_deleted,
            "simulation_delta_r": simulation_intact - simulation_deleted,
            "theory_latent_intact_r": first.cov_to_corr(intact_signal_cov),
            "theory_latent_deleted_r": first.cov_to_corr(deleted_signal_cov),
            "intact_residual_var_P": float(intact_residual[0, 0]),
            "intact_residual_var_A": float(intact_residual[1, 1]),
            "intact_residual_cov_PA": float(intact_residual[0, 1]),
            "deleted_residual_var_P": float(deleted_residual[0, 0]),
            "deleted_residual_var_A": float(deleted_residual[1, 1]),
            "deleted_residual_cov_PA": float(deleted_residual[0, 1]),
            "intact_residual_contributing_trials": int(intact_trials),
            "deleted_residual_contributing_trials": int(deleted_trials),
            "intact_residual_degrees": int(intact_degrees),
            "deleted_residual_degrees": int(deleted_degrees),
            "intact_rectification_count_all_phases": int(rectification[0].sum()),
            "deleted_rectification_count_all_phases": int(rectification[position].sum()),
            "fixedpoint_residual_recomputed": float(maps["fixedpoint_residual"][position]),
            "active_E_recomputed": int(maps["active_e"][position]),
            "active_I_recomputed": int(maps["active_i"][position]),
        })
        rows.append(row)

    result = pd.DataFrame(rows)
    result.to_csv(HERE / "i2_exp1_second_order_matched_results.csv", index=False)
    pd.DataFrame(exclusions).to_csv(HERE / "i2_exp1_second_order_exclusions.csv", index=False)

    first_positions = set(reference.index.astype(int))
    second_positions = set(result.network_position.astype(int))
    invariance = {
        "analyzed_position_sets_identical": first_positions == second_positions,
        "common_n_max_abs_difference": int(max(common_n_diffs, default=0)),
        "simulation_r_max_abs_difference": float(max(simulation_diffs, default=0.0)),
        "calibration_residual_component_max_abs_difference": float(max(residual_diffs, default=0.0)),
        "simulation_cache_files": 83,
        "simulation_cache_reused_read_only": True,
        "simulation_rerun": False,
        "base_map_max_abs_difference": base_max_abs_difference,
        "gradient_map_max_abs_difference": gradient_max_abs_difference,
    }
    if not invariance["analyzed_position_sets_identical"]:
        raise RuntimeError("Analyzed deletion patterns changed")
    if invariance["common_n_max_abs_difference"] != 0:
        raise RuntimeError("Common-valid trial counts changed")
    if invariance["simulation_r_max_abs_difference"] > 1e-12:
        raise RuntimeError("Held-out simulation statistics changed")
    if invariance["calibration_residual_component_max_abs_difference"] > 1e-10:
        raise RuntimeError("Calibration residual statistics changed")

    first_metrics = json.loads(FIRST_METRICS.read_text(encoding="utf-8"))
    metrics = {
        "protocol": {
            "name": PROTOCOL,
            "only_change_from_corrected_first_order": (
                "input-parameter feature map order: first to second"
            ),
            "theory_formula": "m(u)=m0+G u+0.5 u^T H u",
            "derivative_coordinates": "four standardized input parameters u",
            "derivative_center": "u=0 (theta=theta0)",
            "derivative_step": H,
            "derivative_implementation": "pap_validation.core.finite_derivatives; central finite differences",
            "network_specific_expansion": (
                "intact and each deleted network use their own fixed point, active set, Jacobian, gradient, and Hessian"
            ),
            "simulation": "reused complete nonlinear 110-state explicit-Heun phase features",
            "calibration_phases": first.N_CALIBRATION,
            "heldout_phases_in_statistic": first.N_HELDOUT,
            "residual_estimator": "unbiased pooled within-trial phase-centered feature covariance",
            "residual_scaling": "single-phase covariance added directly; no division",
            "heldout_statistic": "Pearson correlation of per-trial features from one independent phase",
            "delta_definition": "intact r - deleted r",
        },
        "intact": first.fit_metrics(result.theory_intact_r, result.simulation_intact_r),
        "deleted": first.fit_metrics(result.theory_deleted_r, result.simulation_deleted_r),
        "delta": first.fit_metrics(result.theory_delta_r, result.simulation_delta_r),
        "delta_by_n_deleted": {
            str(int(n_deleted)): first.fit_metrics(local.theory_delta_r, local.simulation_delta_r)
            for n_deleted, local in result.groupby("n_deleted")
        },
        "first_order_reference": {
            "intact": first_metrics["intact"],
            "deleted": first_metrics["deleted"],
            "delta": first_metrics["delta"],
            "delta_by_n_deleted": first_metrics["delta_by_n_deleted"],
        },
        "registered_patterns": 82,
        "analyzed_patterns": int(len(result)),
        "excluded_patterns": int(len(exclusions)),
        "common_n_min": int(result.common_n.min()),
        "common_n_max": int(result.common_n.max()),
        "fixedpoint_residual_max": float(np.max(maps["fixedpoint_residual"])),
        "hessian_h004_vs_h002_max": float(derivatives.hessian_h004_vs_h002.max()),
        "hessian_h002_vs_h001_max": float(derivatives.hessian_h002_vs_h001.max()),
        "hessian_h004_vs_h002_median": float(derivatives.hessian_h004_vs_h002.median()),
        "hessian_h002_vs_h001_median": float(derivatives.hessian_h002_vs_h001.median()),
        "invariance_checks": invariance,
        "exclusions": exclusions,
        "input_fingerprints": {
            "first_order_program": sha256(FIRST_SCRIPT),
            "first_order_results": sha256(FIRST_RESULTS),
            "frozen_design": sha256(first.I2_ROOT / "design.npz"),
            "simulation_cache_manifest": hashlib.sha256(
                "".join(
                    sha256(FIRST_DIR / "exact_heun_phase_features" / f"network_{i:03d}.npz")
                    for i in range(83)
                ).encode("ascii")
            ).hexdigest(),
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "matplotlib": mpl.__version__,
            "platform": platform.platform(),
        },
    }
    write_json(HERE / "i2_exp1_second_order_matched_metrics.json", metrics)
    return result, metrics


def draw(result: pd.DataFrame) -> Path:
    x = result.theory_delta_r.to_numpy(float)
    y = result.simulation_delta_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y)
    fit = stats.linregress(x[use], y[use])
    low, high = first.limits(x[use], y[use])
    grid = np.linspace(low, high, 300)

    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 13,
        "axes.linewidth": 1.7,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10.5,
    })
    fig, ax = plt.subplots(figsize=(5.0, 4.25))
    fig.subplots_adjust(left=0.15, right=0.96, bottom=0.17, top=0.84)
    handles, labels = [], []
    for n_deleted in sorted(result.n_deleted.unique()):
        color, marker, label = first.STYLES[int(n_deleted)]
        take = use & (result.n_deleted.to_numpy(int) == int(n_deleted))
        handles.append(ax.scatter(
            x[take], y[take], s=42, marker=marker, color=color,
            edgecolor="white", linewidth=0.5, alpha=0.88, zorder=3,
        ))
        labels.append(label)
    ax.plot([low, high], [low, high], color="#777777", ls="--", lw=1.7)
    ax.plot(grid, fit.intercept + fit.slope * grid, color=first.BLACK, lw=2.4)
    ax.axhline(0, color="0.82", lw=1.0, zorder=0)
    ax.axvline(0, color="0.82", lw=1.0, zorder=0)
    ax.set(
        xlim=(low, high), ylim=(low, high),
        xlabel="Second-order theory Δr (intact − deleted)",
        ylabel="Simulation Δr (intact − deleted)",
    )
    ax.set_aspect("equal", adjustable="box")
    ax.text(
        0.04, 0.96,
        f"n = {int(use.sum())}\nr = {fit.rvalue:.2f}\nslope = {fit.slope:.2f}",
        transform=ax.transAxes, ha="left", va="top", fontsize=12.5,
    )
    ax.legend(
        handles, labels, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.015), ncol=4, columnspacing=1.2,
        handletextpad=0.7, borderaxespad=0.0,
    )
    ax.tick_params(width=1.5, length=5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.7)

    stem = HERE / "Figure2i2_second_order_Exp1_matched_complete_Heun"
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)
    return stem


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    result, metrics = assemble()
    draw(result)
    summary = {
        "second_order": {
            key: metrics[key] for key in ("intact", "deleted", "delta", "delta_by_n_deleted")
        },
        "first_order": metrics["first_order_reference"],
        "invariance_checks": metrics["invariance_checks"],
    }
    write_json(HERE / "first_vs_second_order_comparison.json", summary)
    print(json.dumps(json_ready(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
