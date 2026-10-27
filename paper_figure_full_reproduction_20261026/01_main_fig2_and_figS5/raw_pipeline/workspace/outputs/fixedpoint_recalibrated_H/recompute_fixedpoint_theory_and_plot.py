from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import integrate, optimize, stats

from audit_panel_alignment import require_matplotlib_panel_alignment


DT_MS = 0.25
ANALYSIS_S = 65.536
INPUT_AREA = "A6Va"
OUTPUT_AREA = "V2"
FIXED_POINT_TOL = 1e-7


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 13,
        "axes.labelsize": 18,
        "axes.linewidth": 1.5,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11.5,
        "lines.linewidth": 2.4,
    }
)


def _identity_r2(predicted: np.ndarray, observed: np.ndarray) -> float:
    denominator = np.sum((observed - observed.mean()) ** 2)
    return float(1.0 - np.sum((observed - predicted) ** 2) / denominator)


def _drive_matrix(model, fln, simulation_arrays, edge_arrays) -> tuple[np.ndarray, np.ndarray]:
    n = len(model.areas)
    local_ee, local_ei, local_ie, local_ii = simulation_arrays[:4]
    background_e, background_i = simulation_arrays[-2:]
    matrix = np.zeros((2 * n, 2 * n), dtype=float)
    ids = np.arange(n)
    matrix[ids, ids] = local_ee
    matrix[ids, n + ids] = local_ei
    matrix[n + ids, ids] = local_ie
    matrix[n + ids, n + ids] = local_ii
    target, source, edge_e, edge_i = edge_arrays(model.params, fln)
    np.add.at(matrix, (target, source), edge_e)
    np.add.at(matrix, (n + target, source), edge_i)
    background = np.r_[background_e, background_i]
    return matrix, background


def _fixed_point(matrix: np.ndarray, background: np.ndarray, tau: np.ndarray, initial: np.ndarray):
    identity = np.eye(len(initial))

    def residual(rate):
        drive = matrix @ rate + background
        return rate - np.maximum(drive, 0.0)

    def residual_jac(rate):
        drive = matrix @ rate + background
        gain = (drive > 0.0).astype(float)
        return identity - gain[:, None] * matrix

    candidate = optimize.root(
        residual,
        initial,
        jac=residual_jac,
        method="hybr",
        options={"xtol": 1e-11, "maxfev": 2500},
    ).x

    if np.max(np.abs(residual(candidate))) > FIXED_POINT_TOL or np.min(candidate) < -FIXED_POINT_TOL:
        def ode(_, rate):
            return -residual(rate) / tau

        def ode_jac(_, rate):
            return -residual_jac(rate) / tau[:, None]

        trajectory = integrate.solve_ivp(
            ode,
            (0.0, 30000.0),
            initial,
            method="BDF",
            jac=ode_jac,
            rtol=1e-9,
            atol=1e-11,
        )
        seed = np.maximum(trajectory.y[:, -1], 0.0)
        refined = optimize.root(
            residual,
            seed,
            jac=residual_jac,
            method="hybr",
            options={"xtol": 1e-11, "maxfev": 2500},
        ).x
        if np.max(np.abs(residual(refined))) < np.max(np.abs(residual(candidate))):
            candidate = refined

    fixed_residual = float(np.max(np.abs(residual(candidate))))
    candidate[np.abs(candidate) < 1e-10] = 0.0
    drive = matrix @ candidate + background
    gain = (drive > 1e-8).astype(float)
    marginal = np.abs(drive) <= 1e-8
    return candidate, drive, gain, marginal, fixed_residual


def _effective_jacobian(matrix: np.ndarray, gain: np.ndarray, tau: np.ndarray) -> np.ndarray:
    return (-np.eye(len(gain)) + gain[:, None] * matrix) / tau[:, None]


def _nodes(row: pd.Series) -> list[str]:
    value = row.get("deleted_nodes")
    if not isinstance(value, str) or not value:
        value = row.get("deleted_node")
    return str(value).split("|")


def _recompute(project_root: Path, output_dir: Path) -> pd.DataFrame:
    package_root = project_root / "experiments/pap_theory_validation"
    sys.path.insert(0, str(package_root))

    from pap_validation.core import input_psd, load_model
    from pap_validation.exp3_large_delta_screen import _all_transfer_power, _discrete_transfer
    from pap_validation.io import load_config
    from pap_validation.nonlinear_network_validation import _simulation_arrays
    from pap_validation.nonlinear_resimulations import _deleted_fln, _edge_arrays
    from pap_validation.paper_figure_node_deletion import (
        _cov_to_corr,
        _features_from_averaged_psd,
        _full_error_layer,
        _observed_values,
    )
    from pap_validation.single_node_deletion_all_nodes import _full_error_prediction, _network_features

    source_root = package_root / "outputs"
    exact_path = source_root / "nonlinear_resimulations_exact/wholebrain_H/node_deletion_nonlinear_exact.csv"
    source_results_path = source_root / "single_node_deletion_all_nodes/combined_single_multinode_deletion_results.csv"
    source_features_path = source_root / "single_node_deletion_all_nodes/all_single_node_deletion_features.npz"
    shared_inputs_path = source_root / "single_node_deletion_all_nodes/shared_longwindow_input_psd_means.npz"
    cfg = load_config(package_root / "configs/full.yaml")
    model = load_model(project_root / cfg["model_dir"])

    exact_raw = pd.read_csv(exact_path)
    source_results = pd.read_csv(source_results_path).reset_index(names="row_id")
    simulation_columns = [
        "row_id", "simulation_baseline_r", "simulation_deleted_r", "simulation_delta_r",
        "n_valid", "baseline_rectification_count", "deleted_rectification_count",
    ]
    source_clean = source_results.drop(
        columns=[c for c in simulation_columns if c != "row_id" and c in source_results.columns]
    )
    exact = source_clean.merge(
        exact_raw[simulation_columns], on="row_id", how="left", validate="one_to_one"
    )

    result_path = output_dir / "fixedpoint_recalibrated_node_deletion.csv"
    cached_result = pd.DataFrame()
    if result_path.exists():
        cached_result = pd.read_csv(result_path)
        for name in simulation_columns[1:]:
            if name + "_y" in cached_result.columns:
                cached_result[name] = cached_result[name + "_y"]
            cached_result = cached_result.drop(
                columns=[c for c in (name + "_x", name + "_y") if c in cached_result.columns]
            )
        if (
            "fixedpoint_status" in cached_result.columns
            and set(exact.row_id) == set(cached_result.row_id)
        ):
            cached_result = cached_result.sort_values("row_id")
            cached_result.to_csv(result_path, index=False)
            return cached_result

    with np.load(source_features_path) as cached, np.load(shared_inputs_path) as shared:
        frequencies = cached["frequencies"].astype(float)
        frequency_ids = cached["frequency_ids"].astype(int)
        z = cached["z"].astype(float)
        baseline_theory = cached["baseline_theory"].astype(float)
        baseline_theory_valid = cached["baseline_theory_valid"].astype(bool)
        baseline_calibration = cached["baseline_calibration"].astype(float)
        baseline_calibration_valid = cached["baseline_calibration_valid"].astype(bool)
        baseline_heldout_valid = cached["baseline_heldout_valid"].astype(bool)
        calibration_input_psd = shared["calibration_input_psd"].astype(float)
        heldout_input_psd = shared["heldout_input_psd"].astype(float)
    special_features_path = source_root / "paper_figure_node_deletion/exp3_node_deletion_features_robust_v3.npz"
    with np.load(special_features_path) as special:
        special_z = special["z"].astype(float)
        special_observed = special["features"].astype(float)
        special_valid = special["valid"].astype(bool)
        special_use = special["use"].astype(bool)
        special_baseline_theory = special["baseline_theory"].astype(float)

    n = len(model.areas)
    tau = np.r_[
        np.full(n, float(model.params["tau_exc"])),
        np.full(n, float(model.params["tau_inh"])),
    ]
    initial = np.r_[np.full(n, 10.0), np.full(n, 35.0)]
    simulation_arrays = _simulation_arrays(model.params)
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    n_steps = int(round(ANALYSIS_S * 1000.0 / DT_MS))

    rows = []
    cached_by_row = (
        cached_result.set_index("row_id", drop=False)
        if len(cached_result) and "fixedpoint_status" in cached_result.columns
        else None
    )
    for position, item in exact.sort_values("row_id").iterrows():
        if cached_by_row is not None and int(item.row_id) in cached_by_row.index:
            cached_row = cached_by_row.loc[int(item.row_id)]
            if isinstance(cached_row, pd.DataFrame):
                cached_row = cached_row.iloc[-1]
            result = item.to_dict()
            for key, value in cached_row.items():
                if key not in exact.columns or key.startswith("fixedpoint_") or key in {
                    "active_E_count", "active_I_count", "input_active", "output_active",
                    "marginal_gain_count", "effective_max_real_eigenvalue_per_ms",
                }:
                    result[key] = value
            rows.append(result)
            continue
        nodes = _nodes(item)
        result = item.to_dict()
        result.update(
            {
                "fixedpoint_status": "not_computed",
                "fixedpoint_residual": np.nan,
                "effective_max_real_eigenvalue_per_ms": np.nan,
                "active_E_count": np.nan,
                "active_I_count": np.nan,
                "input_active": False,
                "output_active": False,
                "marginal_gain_count": np.nan,
                "fixedpoint_distance_from_intact": np.nan,
                "fixedpoint_theory_baseline_r": np.nan,
                "fixedpoint_theory_deleted_r": np.nan,
                "fixedpoint_theory_delta_r": np.nan,
                "fixedpoint_latent_baseline_r": np.nan,
                "fixedpoint_latent_deleted_r": np.nan,
                "fixedpoint_latent_delta_r": np.nan,
                "fixedpoint_theory_n_valid": 0,
            }
        )
        deleted_fln = _deleted_fln(model, nodes)
        matrix, background = _drive_matrix(model, deleted_fln, simulation_arrays, _edge_arrays)
        rate, drive, gain, marginal, fp_residual = _fixed_point(matrix, background, tau, initial)
        effective_j = _effective_jacobian(matrix, gain, tau)
        max_real = float(np.max(np.linalg.eigvals(effective_j).real))
        result.update(
            {
                "fixedpoint_residual": fp_residual,
                "effective_max_real_eigenvalue_per_ms": max_real,
                "active_E_count": int(gain[:n].sum()),
                "active_I_count": int(gain[n:].sum()),
                "input_active": bool(gain[input_idx]),
                "output_active": bool(gain[output_idx]),
                "marginal_gain_count": int(marginal.sum()),
                "fixedpoint_distance_from_intact": float(np.linalg.norm(rate - initial)),
            }
        )
        if fp_residual > FIXED_POINT_TOL:
            result["fixedpoint_status"] = "fixedpoint_failed"
            rows.append(result)
            continue
        if max_real >= -1e-10:
            result["fixedpoint_status"] = "unstable_linearization"
            rows.append(result)
            continue
        if not gain[input_idx]:
            result["fixedpoint_status"] = "input_inactive"
            rows.append(result)
            continue

        if int(item.row_id) == 221:
            continuous_power = _all_transfer_power(
                effective_j, [input_idx, output_idx], frequencies
            )[0, 1]
            deleted_theory, deleted_theory_valid = _observed_values(
                continuous_power, frequencies, special_z
            )
            use = special_use & deleted_theory_valid & np.isfinite(deleted_theory).all(axis=1)
            if use.sum() < 8:
                result["fixedpoint_status"] = "insufficient_theory_features"
                rows.append(result)
                continue
            _, _, _, baseline_total = _full_error_layer(
                special_baseline_theory,
                special_observed[:3, 0],
                special_valid[:3, 0],
                use,
            )
            _, _, _, deleted_total = _full_error_layer(
                deleted_theory,
                special_observed[:3, 1],
                special_valid[:3, 1],
                use,
            )
            theory_baseline_r = _cov_to_corr(baseline_total)
            theory_deleted_r = _cov_to_corr(deleted_total)
            latent_baseline_r = float(
                stats.pearsonr(special_baseline_theory[use, 0], special_baseline_theory[use, 1]).statistic
            )
            latent_deleted_r = float(
                stats.pearsonr(deleted_theory[use, 0], deleted_theory[use, 1]).statistic
            )
            result.update(
                {
                    "fixedpoint_status": "ok",
                    "fixedpoint_theory_baseline_r": theory_baseline_r,
                    "fixedpoint_theory_deleted_r": theory_deleted_r,
                    "fixedpoint_theory_delta_r": theory_deleted_r - theory_baseline_r,
                    "fixedpoint_latent_baseline_r": latent_baseline_r,
                    "fixedpoint_latent_deleted_r": latent_deleted_r,
                    "fixedpoint_latent_delta_r": latent_deleted_r - latent_baseline_r,
                    "fixedpoint_theory_n_valid": int(use.sum()),
                }
            )
            rows.append(result)
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(
                f"fixed-point theory {len(rows)}/{len(exact)}: row {int(item.row_id)}, "
                f"status={result['fixedpoint_status']}",
                flush=True,
            )
            continue

        continuous_power = _all_transfer_power(
            effective_j, [input_idx, output_idx], frequencies
        )[0, 1]
        deleted_theory, deleted_theory_valid = _observed_values(
            continuous_power, frequencies, z
        )
        discrete_power = np.abs(
            _discrete_transfer(
                effective_j,
                input_idx,
                output_idx,
                n_steps,
                float(model.params["tau_exc"]),
            )
        ) ** 2
        deleted_cal, deleted_cal_valid, _, deleted_hold_valid = _network_features(
            discrete_power[frequency_ids],
            frequencies,
            calibration_input_psd,
            heldout_input_psd,
        )
        use = (
            baseline_theory_valid
            & deleted_theory_valid
            & baseline_calibration_valid
            & deleted_cal_valid
            & baseline_heldout_valid
            & deleted_hold_valid
            & np.isfinite(baseline_theory).all(axis=1)
            & np.isfinite(deleted_theory).all(axis=1)
        )
        if use.sum() < 8:
            result["fixedpoint_status"] = "insufficient_theory_features"
            rows.append(result)
            continue

        theory_baseline_r = _full_error_prediction(
            baseline_theory[use], baseline_calibration[use]
        )
        theory_deleted_r = _full_error_prediction(deleted_theory[use], deleted_cal[use])
        latent_baseline_r = float(stats.pearsonr(baseline_theory[use, 0], baseline_theory[use, 1]).statistic)
        latent_deleted_r = float(stats.pearsonr(deleted_theory[use, 0], deleted_theory[use, 1]).statistic)
        result.update(
            {
                "fixedpoint_status": "ok",
                "fixedpoint_theory_baseline_r": theory_baseline_r,
                "fixedpoint_theory_deleted_r": theory_deleted_r,
                "fixedpoint_theory_delta_r": theory_deleted_r - theory_baseline_r,
                "fixedpoint_latent_baseline_r": latent_baseline_r,
                "fixedpoint_latent_deleted_r": latent_deleted_r,
                "fixedpoint_latent_delta_r": latent_deleted_r - latent_baseline_r,
                "fixedpoint_theory_n_valid": int(use.sum()),
            }
        )
        rows.append(result)
        pd.DataFrame(rows).to_csv(result_path, index=False)
        print(
            f"fixed-point theory {len(rows)}/{len(exact)}: row {int(item.row_id)}, "
            f"status={result['fixedpoint_status']}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values("row_id")
    frame.to_csv(result_path, index=False)
    return frame


def _plot(frame: pd.DataFrame, output_dir: Path) -> dict:
    data = frame[
        np.isfinite(frame.fixedpoint_theory_delta_r)
        & np.isfinite(frame.simulation_delta_r)
    ].copy()
    x = data.fixedpoint_theory_delta_r.to_numpy(float)
    y = data.simulation_delta_r.to_numpy(float)
    fit = stats.linregress(x, y)
    pearson = stats.pearsonr(x, y)
    r2_identity = _identity_r2(x, y)
    old_x = data.theory_delta_r.to_numpy(float)
    old_fit = stats.linregress(old_x, y)
    old_pearson = stats.pearsonr(old_x, y)
    low = float(min(x.min(), y.min()))
    high = float(max(x.max(), y.max()))
    padding = max((high - low) * 0.10, 0.005)
    low -= padding
    high += padding
    grid = np.linspace(low, high, 200)

    fig, ax = plt.subplots(figsize=(6.35, 5.45), constrained_layout=True)
    styles = {
        1: ("#247BA0", "o", "1 node"),
        3: ("#F4A261", "^", "3 nodes"),
        4: ("#2A9D8F", "s", "4 nodes"),
        5: ("#8E5EA2", "D", "5 nodes"),
        12: ("#D1495B", "*", "12 nodes"),
    }
    for n_deleted, (color, marker, label) in styles.items():
        subset = data[data.n_deleted == n_deleted]
        ax.scatter(
            subset.fixedpoint_theory_delta_r,
            subset.simulation_delta_r,
            s=145 if n_deleted == 12 else 76,
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.90,
            label=label,
            zorder=3,
        )
    ax.plot([low, high], [low, high], color="#D1495B", ls="--", lw=3.2, label="Identity")
    fitted = fit.intercept + fit.slope * grid
    fit_margin = 0.025 * (high - low)
    fit_visible = (fitted >= low + fit_margin) & (fitted <= high - fit_margin)
    ax.plot(grid[fit_visible], fitted[fit_visible], color="#202020", lw=3.2, label="Linear fit")
    ax.axhline(0, xmin=0.01, color="0.55", lw=1.4, zorder=0)
    ax.axvline(0, ymin=0.05, color="0.55", lw=1.4, zorder=0)
    ax.set(
        xlim=(low, high),
        ylim=(low, high),
        xlabel=r"Theory $\Delta r$",
        ylabel=r"Simulation $\Delta r$",
    )
    ax.text(
        0.24,
        0.96,
        f"n={len(data)} deletion patterns\n"
        f"Pearson r={pearson.statistic:.3f}\n"
        f"OLS $R^2$={fit.rvalue ** 2:.3f}\n"
        f"Identity $R^2$={r2_identity:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13,
    )
    max_pos = int(np.argmax(np.maximum(x, y)))
    ax.scatter(
        [x[max_pos]],
        [y[max_pos]],
        s=190,
        facecolor="none",
        edgecolor="#D1495B",
        linewidth=2.4,
        zorder=4,
    )
    ax.annotate(
        rf"max $\Delta r$={max(x[max_pos], y[max_pos]):.3f}",
        (x[max_pos], y[max_pos]),
        xytext=(0.55, 0.82),
        textcoords="axes fraction",
        ha="left",
        va="top",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.15),
        fontsize=11.5,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    ax.text(-0.15, 1.06, "H", transform=ax.transAxes, fontsize=23, fontweight="bold")
    ax.tick_params(labelsize=12, length=5, width=1.5, pad=8)
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)

    stem = output_dir / "H_fixedpoint_recalibrated_theory_vs_exact_simulation"
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(stem) + ".alignment.json",
        overlay_svg=str(stem) + ".alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        strict=True,
    )
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {
        "rows_available": int(len(frame)),
        "rows_with_finite_simulation": int(np.isfinite(frame.simulation_delta_r).sum()),
        "rows_fixedpoint_theory_ok": int((frame.fixedpoint_status == "ok").sum()),
        "rows_plotted": int(len(data)),
        "pearson_r": float(pearson.statistic),
        "p_value": float(pearson.pvalue),
        "ols_slope": float(fit.slope),
        "ols_intercept": float(fit.intercept),
        "ols_R2": float(fit.rvalue ** 2),
        "identity_R2": float(r2_identity),
        "same_rows_old_theory_pearson_r": float(old_pearson.statistic),
        "same_rows_old_theory_ols_slope": float(old_fit.slope),
        "same_rows_old_theory_ols_R2": float(old_fit.rvalue ** 2),
        "same_rows_old_theory_identity_R2": float(_identity_r2(old_x, y)),
        "median_abs_fixedpoint_theory_delta_r": float(np.median(np.abs(x))),
        "median_abs_simulation_delta_r": float(np.median(np.abs(y))),
        "axis_limits": [low, high],
        "status_counts": frame.fixedpoint_status.value_counts(dropna=False).to_dict(),
        "status_counts_among_finite_simulation": frame[
            np.isfinite(frame.simulation_delta_r)
        ].fixedpoint_status.value_counts(dropna=False).to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = _recompute(args.project_root.resolve(), output_dir)
    metrics = _plot(frame, output_dir)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
