from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
import traceback
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

for _name in (
    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
TASK = HERE.parents[1]
HISTORY = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "workspace"
OLD_OUT = HISTORY / "outputs"
PROJECT = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project"
PACKAGE = PROJECT / "experiments/pap_theory_validation"
FIGURE_SKILL = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure"
sys.path[:0] = [str(PACKAGE), str(OLD_OUT), str(FIGURE_SKILL / "scripts")]

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
from pap_validation.core import feature_from_spectrum, input_psd  # noqa: E402
from pap_validation.exp1 import Z0, ZSCALE  # noqa: E402


EXP2_ROOT = OLD_OUT / "full_network_heun_20260908_retry"
EXP2_SOURCE_ROOT = OLD_OUT / "full_network_exact_vector_20260907"
EXP2_SOURCE_SCRIPT = EXP2_SOURCE_ROOT / "run_exp2_theory_c_1000conditions.py"
EXP2_CONDITIONS = EXP2_ROOT / "conditions_input_1000.csv"
EXP2_HEUN_RESULTS = EXP2_ROOT / "results_full.csv"
EXP2_OLD_RESULTS = (
    EXP2_SOURCE_ROOT / "exp2_theory_c_1000conditions/results_condition_level.csv"
)
EXP2_CACHE = EXP2_SOURCE_ROOT / "exp2_theory_c_1000conditions/condition_cache"
ZERO_INPUT_SOURCE = (
    OLD_OUT / "revised_core_and_zero_input_boxplots_20260908/"
    "wholebrain_zero_input_boxplot_values.csv"
)

I1_ROOT = OLD_OUT / "deletion_count_direct_nondirect_20260910"
I1_SCRIPT = I1_ROOT / "run_full55_count_direct_nondirect.py"
I1_STATES = I1_ROOT / "full55_states_512"
I1_DESIGN = OLD_OUT / "deletion_bands_robustness_512_heun_20260908/design_512.npz"

I2_ROOT = OLD_OUT / "wholebrain_H_heun_20260908"
I2_SCRIPT = I2_ROOT / "run_H_heun.py"
I2_LINEAR = I2_ROOT / "linear_calibration_features_heun.npz"
I2_NONLINEAR = I2_ROOT / "nonlinear_states"

PROTOCOL = "figure2_first_order_rerun_v1"
EXP2_H = 0.002
EXP3_H = 0.002
H_AUDIT = (0.004, 0.002, 0.001)
THEORY_F = np.unique(
    np.r_[np.geomspace(0.5, 40.0, 180), np.geomspace(40.1, 1200.0, 220)]
)
PEAK_BAND_EXP3 = (5.0, 20.0)
SLOPE_BAND = (100.0, 1000.0)
BLUE = "#247BA0"
ORANGE = "#F28E2B"
BLACK = "#222222"
I1_COLORS = {
    ("Theory", True): "#D1495B",
    ("Theory", False): "#247BA0",
    ("Simulation", True): "#F28E2B",
    ("Simulation", False): "#2A9D8F",
}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_ready(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return None if not np.isfinite(value) else value
    return value


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp.{os.getpid()}")
    temp.write_text(value, encoding="utf-8")
    temp.replace(path)


def atomic_json(path: Path, value: dict) -> None:
    atomic_text(path, json.dumps(json_ready(value), indent=2) + "\n")


def atomic_csv(path: Path, frame: pd.DataFrame, compression=None) -> None:
    suffix = ".tmp.gz" if compression == "gzip" else ".tmp"
    temp = path.with_name(path.name + f".{os.getpid()}{suffix}")
    frame.to_csv(temp, index=False, compression=compression)
    temp.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_digest(root: Path, pattern: str = "*.npz") -> dict:
    digest = hashlib.sha256()
    paths = sorted(root.glob(pattern))
    for path in paths:
        digest.update(path.name.encode())
        digest.update(sha256(path).encode())
    return {"root": str(root), "files": len(paths), "sha256": digest.hexdigest()}


def corr_from_cov(cov: np.ndarray) -> float:
    denom = float(cov[0, 0] * cov[1, 1])
    return float(cov[0, 1] / np.sqrt(max(denom, 1e-30)))


def corr_feature(values: np.ndarray, use: np.ndarray) -> float:
    good = np.asarray(use, bool) & np.isfinite(values).all(axis=1)
    if good.sum() < 8 or np.min(np.std(values[good], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[good, 0], values[good, 1]).statistic)


def pooled_phase_cov(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    return np.cov(centered.reshape(-1, 2), rowvar=False, ddof=1)


def fit_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    use = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x, float)[use]
    y = np.asarray(y, float)[use]
    fit = stats.linregress(x, y)
    delta = y - x
    return {
        "n": int(len(x)), "pearson_r": float(fit.rvalue),
        "ols_slope": float(fit.slope), "ols_intercept": float(fit.intercept),
        "R2": float(fit.rvalue**2),
        "identity_R2": float(1 - np.sum(delta**2) / np.sum((y-y.mean())**2)),
        "MAE": float(np.mean(np.abs(delta))),
        "RMSE": float(np.sqrt(np.mean(delta**2))),
        "mean_bias": float(np.mean(delta)),
    }


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a-b) / (np.linalg.norm(b) + 1e-15))


_EXP2_SOURCE = None
_EXP2_OLD = None
_EXP2_BASE = None
_EXP2_MODEL = None


def exp2_worker_init() -> None:
    global _EXP2_SOURCE, _EXP2_OLD, _EXP2_BASE, _EXP2_MODEL
    sys.path.insert(0, str(EXP2_SOURCE_ROOT))
    _EXP2_SOURCE = load_module("exp2_first_order_source", EXP2_SOURCE_SCRIPT)
    _EXP2_OLD = sys.modules.get("run_exp2_oldprotocol_uniform")
    _EXP2_BASE = sys.modules.get("run_full_network_exact")
    if _EXP2_OLD is None or _EXP2_BASE is None:
        raise RuntimeError("Experiment 2 helper imports failed")
    _EXP2_MODEL = _EXP2_BASE.load_wholebrain()


def exp2_worker(payload: dict) -> dict:
    condition = pd.Series(payload)
    cid = int(condition.condition_id)
    try:
        solved = _EXP2_SOURCE._assemble_and_solve_condition(_EXP2_MODEL, condition)
        probe = np.asarray([0.0, -0.004, 0.004, -0.002, 0.002, -0.001, 0.001])
        values, valid = _EXP2_SOURCE._convolved_theory_features(
            _EXP2_MODEL, condition, probe, solved["jacobian"],
            solved["input_idx"], solved["output_idx"],
        )
        lookup = {float(v): pos for pos, v in enumerate(probe)}
        gradients = {}
        for h in H_AUDIT:
            gradients[h] = (
                values[lookup[float(h)]] - values[lookup[float(-h)]]
            ) / (2*h)
        with np.load(EXP2_CACHE / f"condition_{cid:04d}.npz") as saved:
            betas = saved["betas"].astype(float)
        return {
            "status": "success", "condition_id": cid,
            "m0_P": float(values[0, 0]), "m0_A": float(values[0, 1]),
            "g_P": float(gradients[EXP2_H][0]),
            "g_A": float(gradients[EXP2_H][1]),
            "gradient_h0p004_P": float(gradients[0.004][0]),
            "gradient_h0p004_A": float(gradients[0.004][1]),
            "gradient_h0p001_P": float(gradients[0.001][0]),
            "gradient_h0p001_A": float(gradients[0.001][1]),
            "gradient_rel_h004_vs_h002": rel_error(gradients[0.004], gradients[0.002]),
            "gradient_rel_h002_vs_h001": rel_error(gradients[0.002], gradients[0.001]),
            "derivative_probe_all_valid": bool(valid.all()),
            "beta_mean": float(betas.mean()),
            "beta_variance_sample": float(np.var(betas, ddof=1)),
            **solved["diagnostics"],
        }
    except Exception as exc:
        return {
            "status": "failure", "condition_id": cid, "reason": repr(exc),
            "traceback": traceback.format_exc(),
        }


def run_exp2(workers: int) -> pd.DataFrame:
    conditions = pd.read_csv(EXP2_CONDITIONS).sort_values("condition_id")
    heun = pd.read_csv(EXP2_HEUN_RESULTS).sort_values("condition_id")
    old = pd.read_csv(EXP2_OLD_RESULTS).sort_values("condition_id")
    if len(conditions) != 1000 or conditions.condition_id.nunique() != 1000:
        raise RuntimeError("Frozen Experiment 2 table is not 1000 unique conditions")
    if len(heun) != 1000 or not heun.n_common.eq(256).all():
        raise RuntimeError("Heun result table is incomplete or changed")
    row_dir = HERE / "exp2_derivative_rows"
    row_dir.mkdir(exist_ok=True)
    done: dict[int, dict] = {}
    pending = []
    for item in conditions.to_dict("records"):
        cid = int(item["condition_id"])
        path = row_dir / f"condition_{cid:04d}.json"
        if path.exists():
            row = json.loads(path.read_text())
            if row.get("status") == "success":
                done[cid] = row
                continue
        pending.append(item)
    with ProcessPoolExecutor(max_workers=workers, initializer=exp2_worker_init) as pool:
        futures = {pool.submit(exp2_worker, item): int(item["condition_id"]) for item in pending}
        for future in as_completed(futures):
            cid = futures[future]
            row = future.result()
            atomic_json(row_dir / f"condition_{cid:04d}.json", row)
            if row["status"] == "success":
                done[cid] = row
            atomic_json(HERE / "exp2_progress.json", {
                "completed": len(done), "total": 1000,
                "failures": 1000-len(done)-len([f for f in futures if not f.done()]),
            })
    derivative = pd.DataFrame(done.values()).sort_values("condition_id")
    failures = sorted(set(conditions.condition_id.astype(int)) - set(done))
    if failures:
        raise RuntimeError(f"Experiment 2 derivative failures: {failures[:20]}")
    condition_columns = [
        "condition_id", "family", "S", "sigma_beta", "floor_ratio",
        "input_sigma", "critical_real_per_s", "critical_frequency_hz",
    ]
    frame = conditions[condition_columns].merge(
        derivative, on="condition_id", validate="one_to_one"
    )
    frame = frame.merge(
        heun[["condition_id", "theory_r", "simulation_r", "residual_var_P", "residual_var_A",
              "residual_cov_PA", "n_common"]].rename(
                  columns={"theory_r": "theory_full_mapping_heun_calibration_r"}
              ), on="condition_id", validate="one_to_one",
    )
    frame = frame.merge(
        old[["condition_id", "theory_C_corrected_r"]].rename(
            columns={"theory_C_corrected_r": "theory_full_mapping_legacy_calibration_r"}
        ), on="condition_id", validate="one_to_one",
    )
    theory_r = []
    cov_identity = []
    for row in frame.itertuples(index=False):
        g = np.asarray([row.g_P, row.g_A])
        signal = np.outer(g, g) * float(row.beta_variance_sample)
        residual = np.asarray([
            [row.residual_var_P, row.residual_cov_PA],
            [row.residual_cov_PA, row.residual_var_A],
        ])
        theory_r.append(corr_from_cov(signal + residual))
        with np.load(EXP2_CACHE / f"condition_{int(row.condition_id):04d}.npz") as saved:
            beta = saved["betas"].astype(float)
        linear = np.c_[row.m0_P + row.g_P*beta, row.m0_A + row.g_A*beta]
        cov_identity.append(float(np.max(np.abs(np.cov(linear, rowvar=False, ddof=1)-signal))))
    frame["theory_first_order_r"] = theory_r
    frame["linear_covariance_identity_max_abs"] = cov_identity
    atomic_csv(HERE / "experiment2_first_order_results.csv", frame)
    metrics = {
        "first_order_vs_heun": fit_metrics(
            frame.theory_first_order_r.to_numpy(float), frame.simulation_r.to_numpy(float)
        ),
        "old_full_mapping_audit_vs_heun": fit_metrics(
            frame.theory_full_mapping_heun_calibration_r.to_numpy(float),
            frame.simulation_r.to_numpy(float)
        ),
        "legacy_full_mapping_and_legacy_calibration_audit_vs_heun": fit_metrics(
            frame.theory_full_mapping_legacy_calibration_r.to_numpy(float),
            frame.simulation_r.to_numpy(float)
        ),
        "gradient_convergence": {
            "h004_vs_h002_max": float(frame.gradient_rel_h004_vs_h002.max()),
            "h004_vs_h002_median": float(frame.gradient_rel_h004_vs_h002.median()),
            "h002_vs_h001_max": float(frame.gradient_rel_h002_vs_h001.max()),
            "h002_vs_h001_median": float(frame.gradient_rel_h002_vs_h001.median()),
            "all_probe_features_valid": bool(frame.derivative_probe_all_valid.all()),
        },
        "linear_covariance_identity_max_abs": float(frame.linear_covariance_identity_max_abs.max()),
    }
    atomic_json(HERE / "experiment2_metrics.json", metrics)
    return frame


def feature_gradients(
    jacobian: np.ndarray, input_idx: int, output_indices: np.ndarray, transfer_fun,
) -> tuple[np.ndarray, dict[float, np.ndarray], np.ndarray]:
    transfer = transfer_fun(jacobian, input_idx, output_indices, THEORY_F)
    n_out = len(output_indices)
    base = np.full((n_out, 2), np.nan)
    gradients = {h: np.full((n_out, 2, 4), np.nan) for h in H_AUDIT}
    valid = np.ones(n_out, bool)
    probes = {0.0: Z0.copy()}
    for h in H_AUDIT:
        for q in range(4):
            probes[(h, q, 1)] = Z0 + ZSCALE*h*np.eye(4)[q]
            probes[(h, q, -1)] = Z0 - ZSCALE*h*np.eye(4)[q]
    for out in range(n_out):
        f0 = feature_from_spectrum(
            THEORY_F, transfer[out] * input_psd(THEORY_F, Z0),
            PEAK_BAND_EXP3, SLOPE_BAND,
        )
        base[out] = f0["peak_omega"], f0["exponent"]
        valid[out] &= bool(f0["peak_valid"])
        for h in H_AUDIT:
            for q in range(4):
                vals = []
                for sign in (1, -1):
                    z = probes[(h, q, sign)]
                    feat = feature_from_spectrum(
                        THEORY_F, transfer[out] * input_psd(THEORY_F, z),
                        PEAK_BAND_EXP3, SLOPE_BAND,
                    )
                    vals.append(np.asarray([feat["peak_omega"], feat["exponent"]]))
                    valid[out] &= bool(feat["peak_valid"])
                gradients[h][out, :, q] = (vals[0]-vals[1])/(2*h)
    return base, gradients, valid


def exp3_derivative_diagnostic(gradients: dict[float, np.ndarray], valid: np.ndarray) -> dict:
    a = gradients[0.004][valid]
    b = gradients[0.002][valid]
    c = gradients[0.001][valid]
    row_ab = np.linalg.norm(a-b, axis=(1, 2)) / (np.linalg.norm(b, axis=(1, 2))+1e-15)
    row_bc = np.linalg.norm(b-c, axis=(1, 2)) / (np.linalg.norm(c, axis=(1, 2))+1e-15)
    return {
        "n_valid_outputs": int(valid.sum()),
        "h004_vs_h002_max": float(row_ab.max()) if len(row_ab) else None,
        "h004_vs_h002_median": float(np.median(row_ab)) if len(row_ab) else None,
        "h002_vs_h001_max": float(row_bc.max()) if len(row_bc) else None,
        "h002_vs_h001_median": float(np.median(row_bc)) if len(row_bc) else None,
    }


def run_i1() -> pd.DataFrame:
    count = load_module("count_first_order_source", I1_SCRIPT)
    model, selected, z, networks, input_idx, outputs, _, _, _ = count.build_context()
    if len(selected) != 60 or not selected.groupby("n_deleted").size().eq(12).all():
        raise RuntimeError("i1 deletion pattern table changed")
    with np.load(I1_DESIGN) as saved:
        frozen_z = saved["z"].astype(float)
        frozen_u = saved["u"].astype(float)
    if not np.array_equal(z, frozen_z):
        raise RuntimeError("i1 frozen z design changed")
    if not np.allclose((z-Z0)/ZSCALE, frozen_u, rtol=0, atol=2e-15):
        raise RuntimeError("i1 z/u parameterization changed")
    states = [count.load_state(I1_STATES / f"network_{p:03d}.npz") for p in range(61)]
    bases, grads, grad_valid, diagnostics = [], [], [], []
    for pos, network in enumerate(networks):
        _, _, gain, jacobian, residual = network
        base, all_grad, valid = feature_gradients(
            jacobian, input_idx, outputs, count.base.continuous_transfer_power_all
        )
        bases.append(base); grads.append(all_grad[EXP3_H]); grad_valid.append(valid)
        diagnostics.append({
            "network_position": pos, "fixedpoint_residual": float(residual),
            "active_E": int(gain[:55].sum()), "active_I": int(gain[55:].sum()),
            **exp3_derivative_diagnostic(all_grad, valid),
        })
    bases = np.asarray(bases); grads = np.asarray(grads); grad_valid = np.asarray(grad_valid)
    np.savez_compressed(
        HERE / "i1_first_order_maps.npz", base=bases, gradient_u=grads,
        gradient_theta=grads/ZSCALE[None, None, None, :], valid=grad_valid,
        theta0=Z0, theta_scale=ZSCALE, h=np.asarray(EXP3_H),
    )
    atomic_csv(HERE / "i1_derivative_diagnostics.csv", pd.DataFrame(diagnostics))
    rows = []
    covariance_errors = []
    skipped = []
    for position, item in enumerate(selected.itertuples(index=False), 1):
        deleted_nodes = str(item.deleted_nodes).split("|")
        deleted_indices = [model.areas.index(node) for node in deleted_nodes]
        gain = networks[position][2]
        if not bool(gain[input_idx]):
            skipped.append({"position": position, "reason": "input inactive"})
            continue
        for output_idx, output_area in enumerate(model.areas):
            if output_idx in deleted_indices or not bool(gain[output_idx]):
                continue
            if not (grad_valid[0, output_idx] and grad_valid[position, output_idx]):
                skipped.append({
                    "position": position, "output_area": output_area,
                    "reason": "feature invalid at theta0 or predetermined derivative probe",
                })
                continue
            common = (
                states[0]["simulation_valid"][output_idx]
                & states[position]["simulation_valid"][output_idx]
                & states[0]["calibration_valid"][:, output_idx].all(axis=0)
                & states[position]["calibration_valid"][:, output_idx].all(axis=0)
            )
            if common.sum() < 8:
                skipped.append({
                    "position": position, "output_area": output_area,
                    "reason": f"fewer than 8 common trials ({int(common.sum())})",
                })
                continue
            sigma = np.cov(frozen_u[common], rowvar=False, ddof=1)
            base_signal = grads[0, output_idx] @ sigma @ grads[0, output_idx].T
            del_signal = grads[position, output_idx] @ sigma @ grads[position, output_idx].T
            base_linear = bases[0, output_idx] + frozen_u[common] @ grads[0, output_idx].T
            del_linear = bases[position, output_idx] + frozen_u[common] @ grads[position, output_idx].T
            covariance_errors.extend([
                float(np.max(np.abs(np.cov(base_linear, rowvar=False, ddof=1)-base_signal))),
                float(np.max(np.abs(np.cov(del_linear, rowvar=False, ddof=1)-del_signal))),
            ])
            base_residual = pooled_phase_cov(states[0]["calibration"][:, output_idx, common])
            del_residual = pooled_phase_cov(states[position]["calibration"][:, output_idx, common])
            theory_base = corr_from_cov(base_signal + base_residual/count.HELD_PHASES)
            theory_del = corr_from_cov(del_signal + del_residual/count.HELD_PHASES)
            sim_base = corr_feature(states[0]["simulation"][output_idx], common)
            sim_del = corr_feature(states[position]["simulation"][output_idx], common)
            rows.append({
                "deleted_nodes": item.deleted_nodes, "n_deleted": int(item.n_deleted),
                "output_area": output_area,
                "direct": bool(any(model.fln[output_idx, d] > 0 for d in deleted_indices)),
                "n_valid": int(common.sum()),
                "theory_baseline_r": theory_base, "theory_deleted_r": theory_del,
                "theory_delta_r": theory_del-theory_base,
                "simulation_baseline_r": sim_base, "simulation_deleted_r": sim_del,
                "simulation_delta_r": sim_del-sim_base,
            })
    frame = pd.DataFrame(rows)
    atomic_csv(HERE / "i1_first_order_raw_units.csv.gz", frame, compression="gzip")
    atomic_csv(HERE / "i1_exclusions.csv", pd.DataFrame(skipped))
    atomic_json(HERE / "i1_computation_audit.json", {
        "networks": 61, "deletion_patterns": 60, "raw_units": len(frame),
        "excluded_structure_output_pairs": len(skipped),
        "linear_covariance_identity_max_abs": max(covariance_errors),
        "input_active_all_structures": bool(all(n[2][input_idx] for n in networks)),
    })
    return frame


def load_i2_nonlinear(source, maps: dict, heldout: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.full((83, 128, 2), np.nan)
    valid = np.zeros((83, 128), bool)
    for network in range(83):
        with np.load(I2_NONLINEAR / f"network_{network:03d}.npz") as saved:
            completed = saved["completed"].astype(bool)
            if len(completed) != heldout or not completed.all():
                raise RuntimeError(f"i2 nonlinear cache incomplete for network {network}")
            psd = saved["psd_sum"].astype(float) / heldout
        local, local_valid = source._features_from_averaged_psd(
            maps["frequencies"], psd[None, :, :]
        )
        values[network] = local[0]
        valid[network] = local_valid[0]
    return values, valid


def run_i2() -> pd.DataFrame:
    runner = load_module("i2_first_order_source", I2_SCRIPT)
    model, patterns, z, maps, design = runner.build_context()
    if len(patterns) != 82 or len(z) != 128:
        raise RuntimeError("i2 registered pattern/design count changed")
    with np.load(I2_LINEAR) as saved:
        calibration = saved["features"].astype(float)
        calibration_valid = saved["valid"].astype(bool)
    if calibration.shape != (3, 83, 128, 2):
        raise RuntimeError("i2 calibration cache shape changed")
    nonlinear, nonlinear_valid = load_i2_nonlinear(runner.source, maps, runner.N_HELDOUT)
    input_idx = model.areas.index(runner.source.INPUT_AREA)
    output_idx = model.areas.index(runner.source.OUTPUT_AREA)
    output_indices = np.asarray([output_idx], int)
    bases, grads, valid_maps, diagnostics = [], [], [], []
    for pos, jacobian in enumerate(maps["jacobians"]):
        base, all_grad, valid = feature_gradients(
            jacobian, input_idx, output_indices,
            lambda j, inp, outs, frequencies: runner.source._all_transfer_power(
                j, [inp, *np.asarray(outs, int).tolist()], frequencies
            )[0, 1:],
        )
        bases.append(base[0]); grads.append(all_grad[EXP3_H][0]); valid_maps.append(valid[0])
        diagnostics.append({
            "network_position": pos,
            "fixedpoint_residual": float(maps["fixedpoint_residual"][pos]),
            "active_E": int(maps["active_e"][pos]), "active_I": int(maps["active_i"][pos]),
            **exp3_derivative_diagnostic(all_grad, valid),
        })
    bases = np.asarray(bases); grads = np.asarray(grads); valid_maps = np.asarray(valid_maps)
    np.savez_compressed(
        HERE / "i2_first_order_maps.npz", base=bases, gradient_u=grads,
        gradient_theta=grads/ZSCALE[None, None, :], valid=valid_maps,
        theta0=Z0, theta_scale=ZSCALE, h=np.asarray(EXP3_H),
    )
    atomic_csv(HERE / "i2_derivative_diagnostics.csv", pd.DataFrame(diagnostics))
    u = np.asarray(design["u"], float)
    if not np.allclose((z-Z0)/ZSCALE, u, rtol=0, atol=2e-15):
        raise RuntimeError("i2 z/u parameterization changed")
    rows, skipped, covariance_errors = [], [], []
    for position, item in enumerate(patterns.itertuples(index=False), 1):
        if hasattr(item, "input_active") and not bool(item.input_active):
            skipped.append({"position": position, "reason": "input inactive"}); continue
        if hasattr(item, "output_active") and not bool(item.output_active):
            skipped.append({"position": position, "reason": "output inactive"}); continue
        if not (valid_maps[0] and valid_maps[position]):
            skipped.append({"position": position, "reason": "feature invalid at derivative probe"}); continue
        common = (
            nonlinear_valid[0] & nonlinear_valid[position]
            & calibration_valid[:, 0].all(axis=0)
            & calibration_valid[:, position].all(axis=0)
        )
        if common.sum() < 8:
            skipped.append({"position": position, "reason": f"fewer than 8 common trials ({common.sum()})"}); continue
        sigma = np.cov(u[common], rowvar=False, ddof=1)
        base_signal = grads[0] @ sigma @ grads[0].T
        del_signal = grads[position] @ sigma @ grads[position].T
        base_linear = bases[0] + u[common] @ grads[0].T
        del_linear = bases[position] + u[common] @ grads[position].T
        covariance_errors.extend([
            float(np.max(np.abs(np.cov(base_linear, rowvar=False, ddof=1)-base_signal))),
            float(np.max(np.abs(np.cov(del_linear, rowvar=False, ddof=1)-del_signal))),
        ])
        base_resid = pooled_phase_cov(calibration[:, 0, common])
        del_resid = pooled_phase_cov(calibration[:, position, common])
        theory_base = corr_from_cov(base_signal + base_resid/runner.N_HELDOUT)
        theory_del = corr_from_cov(del_signal + del_resid/runner.N_HELDOUT)
        sim_base = corr_feature(nonlinear[0], common)
        sim_del = corr_feature(nonlinear[position], common)
        row = item._asdict()
        row.update({
            "theory_baseline_r": theory_base, "theory_deleted_r": theory_del,
            "theory_delta_r": theory_del-theory_base,
            "simulation_baseline_r": sim_base, "simulation_deleted_r": sim_del,
            "simulation_delta_r": sim_del-sim_base, "common_n": int(common.sum()),
        })
        rows.append(row)
    frame = pd.DataFrame(rows)
    atomic_csv(HERE / "i2_first_order_results.csv", frame)
    atomic_csv(HERE / "i2_exclusions.csv", pd.DataFrame(skipped))
    atomic_json(HERE / "i2_computation_audit.json", {
        "registered_patterns": 82, "analyzed_patterns": len(frame),
        "excluded_patterns": len(skipped),
        "linear_covariance_identity_max_abs": max(covariance_errors),
        "fit": fit_metrics(frame.theory_delta_r.to_numpy(), frame.simulation_delta_r.to_numpy()),
    })
    return frame


def style_scatter() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 15, "xtick.labelsize": 13, "ytick.labelsize": 13,
        "axes.linewidth": 1.7, "axes.spines.right": False, "axes.spines.top": False,
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig, json_out=stem.with_suffix(".alignment.json"),
        overlay_svg=stem.with_suffix(".alignment.svg"),
        tolerance_pt=1.5, gutter_tolerance_pt=1.5, strict=True,
    )
    fig.savefig(stem.with_suffix(".svg"), facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white")
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white")
    plt.close(fig)


def draw_f(frame: pd.DataFrame) -> Path:
    style_scatter()
    x = frame.theory_first_order_r.to_numpy(float)
    y = frame.simulation_r.to_numpy(float)
    fit = stats.linregress(x, y)
    lo, hi = float(min(x.min(), y.min())), float(max(x.max(), y.max()))
    pad = 0.035*max(hi-lo, 0.1)
    grid = np.linspace(x.min(), x.max(), 300)
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.subplots_adjust(left=.14, right=.95, bottom=.16, top=.96)
    scatter = ax.scatter(x, y, s=19, alpha=.46, color=BLUE, edgecolor="none", rasterized=False)
    if len(scatter.get_offsets()) != 1000:
        raise RuntimeError("f does not contain 1000 points")
    ax.plot(grid, fit.intercept+fit.slope*grid, color=BLACK, lw=2.8)
    ax.set(xlim=(lo-pad, hi+pad), ylim=(lo-pad, hi+pad), xlabel="", ylabel="", title="")
    ax.tick_params(width=1.7, length=6)
    stem = HERE / "Figure2f_first_order_theory_vs_nonlinear_Heun"
    save_figure(fig, stem)
    return stem


def draw_g(frame: pd.DataFrame) -> Path:
    source_values = pd.read_csv(ZERO_INPUT_SOURCE)
    stable_node = source_values[source_values.regime == "stable_node"].copy()
    stable_complex = pd.DataFrame({
        "regime": "stable_complex", "condition_id": frame.condition_id,
        "theory_r": frame.theory_first_order_r, "simulation_r": frame.simulation_r,
    })
    atomic_csv(HERE / "Figure2g_source_values.csv", pd.concat([stable_node, stable_complex]))
    style_scatter()
    mpl.rcParams.update({"font.size": 17, "xtick.labelsize": 17, "ytick.labelsize": 17, "axes.linewidth": 2.8})
    groups = [stable_complex.theory_r.to_numpy(), stable_complex.simulation_r.to_numpy()]
    fig, ax = plt.subplots(figsize=(5.4, 4.6), constrained_layout=True)
    boxes = ax.boxplot(
        groups, positions=[0, 1], widths=.52, patch_artist=True, showfliers=False,
        medianprops={"color": BLACK, "linewidth": 3},
        whiskerprops={"linewidth": 2.2}, capprops={"linewidth": 2.2},
    )
    for patch, color in zip(boxes["boxes"], (BLUE, ORANGE)):
        patch.set_facecolor(color); patch.set_alpha(.28); patch.set_linewidth(2.6)
    rng = np.random.default_rng(2026082901)
    for pos, (values, color) in enumerate(zip(groups, (BLUE, ORANGE))):
        ax.scatter(pos+rng.normal(0, .075, len(values)), values, s=12, color=color,
                   alpha=.24, edgecolor="none", zorder=2, rasterized=False)
    ax.axhline(0, color="0.45", lw=2, zorder=0)
    ax.set_xticks([0, 1], ["Theory", "Simulation"])
    ax.set(xlabel="", ylabel="", title="", xlim=(-.65, 1.65))
    stem = HERE / "Figure2g_first_order_theory_vs_nonlinear_Heun_boxplot"
    save_figure(fig, stem)
    return stem


def sign_flip_p(values: np.ndarray, seed: int, permutations: int = 100000) -> float:
    values = np.asarray(values, float)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    extreme = 0
    chunk = 10000
    for start in range(0, permutations, chunk):
        n = min(chunk, permutations-start)
        signs = rng.choice((-1.0, 1.0), size=(n, len(values)))
        extreme += int(np.sum(np.abs((signs*values).mean(axis=1)) >= observed))
    return float((extreme+1)/(permutations+1))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order]
    q = np.minimum.accumulate((ranked*len(p)/np.arange(1, len(p)+1))[::-1])[::-1]
    result = np.empty_like(q); result[order] = np.minimum(q, 1.0)
    return result


def stars(q: float) -> str:
    if q < 1e-4: return "****"
    if q < 1e-3: return "***"
    if q < 1e-2: return "**"
    if q < 0.05: return "*"
    return "ns"


def draw_i1(raw: pd.DataFrame) -> Path:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 12, "axes.linewidth": 1.7, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 9.5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    units = []
    for source_name, column in (("Theory", "theory_delta_r"), ("Simulation", "simulation_delta_r")):
        local = raw[["n_deleted", "deleted_nodes", "output_area", "direct", column]].copy()
        local["source"] = source_name; local["value"] = local[column].abs()
        units.append(local[["n_deleted", "deleted_nodes", "output_area", "direct", "source", "value"]])
    units = pd.concat(units, ignore_index=True)
    zero_count = int((units.value == 0).sum())
    units = units[np.isfinite(units.value) & (units.value > 0)].copy()
    atomic_csv(HERE / "i1_plot_units_absolute_delta_r.csv.gz", units, compression="gzip")
    network_units = units.groupby(
        ["n_deleted", "deleted_nodes", "source", "direct"], as_index=False
    ).value.mean()
    atomic_csv(HERE / "i1_network_level_means.csv", network_units)
    statistics = []
    for count in range(1, 6):
        for source_index, source_name in enumerate(("Theory", "Simulation")):
            local = network_units[(network_units.n_deleted == count) & (network_units.source == source_name)]
            paired = local.pivot(index="deleted_nodes", columns="direct", values="value").dropna()
            difference = paired[True].to_numpy()-paired[False].to_numpy()
            statistics.append({
                "n_deleted": count, "source": source_name,
                "n_deletion_networks": len(paired),
                "direct_mean": float(paired[True].mean()),
                "non_direct_mean": float(paired[False].mean()),
                "mean_paired_difference": float(difference.mean()),
                "sign_flip_p": sign_flip_p(difference, 73421+7000+count*10+source_index),
                "fraction_positive": float(np.mean(difference > 0)),
            })
    stat_frame = pd.DataFrame(statistics)
    stat_frame["sign_flip_fdr_q"] = bh(stat_frame.sign_flip_p.to_numpy())
    atomic_csv(HERE / "i1_significance.csv", stat_frame)
    summary = units.groupby(["n_deleted", "source", "direct"]).value.agg(
        ["count", "mean", "median", "std", "min", "max"]
    ).reset_index()
    atomic_csv(HERE / "i1_group_summary.csv", summary)
    fig, ax = plt.subplots(figsize=(183/25.4, 91.5/25.4))
    fig.subplots_adjust(left=.075, right=.985, bottom=.16, top=.84)
    offsets = {("Theory", True): -.30, ("Theory", False): -.10,
               ("Simulation", True): .10, ("Simulation", False): .30}
    rng = np.random.default_rng(20260910+21)
    annotation_max = 0.0
    for count in range(1, 6):
        for source_name in ("Theory", "Simulation"):
            for direct in (True, False):
                values = units.loc[
                    (units.n_deleted == count) & (units.source == source_name) & (units.direct == direct), "value"
                ].to_numpy()
                position = count-1+offsets[(source_name, direct)]
                color = I1_COLORS[(source_name, direct)]
                box = ax.boxplot([values], positions=[position], widths=.16, patch_artist=True,
                    showfliers=False, medianprops={"color": "#1A1A1A", "linewidth": 2.4},
                    whiskerprops={"color": color, "linewidth": 1.7}, capprops={"color": color, "linewidth": 1.7})
                box["boxes"][0].set_facecolor(color); box["boxes"][0].set_edgecolor(color)
                box["boxes"][0].set_alpha(.34); box["boxes"][0].set_linewidth(1.9)
                ax.scatter(np.full(len(values), position)+rng.uniform(-.055, .055, len(values)), values,
                           s=5.5, color=color, edgecolor="none", alpha=.28, rasterized=False, zorder=2)
            source_values = units.loc[(units.n_deleted == count) & (units.source == source_name), "value"].to_numpy()
            top = float(source_values.max()*1.18)
            left = count-1+offsets[(source_name, True)]; right = count-1+offsets[(source_name, False)]
            ax.plot([left, left, right, right], [top/1.13, top, top, top/1.13], color="#1A1A1A", lw=1.35, clip_on=False)
            q = float(stat_frame.loc[(stat_frame.n_deleted == count) & (stat_frame.source == source_name), "sign_flip_fdr_q"].iloc[0])
            ax.text((left+right)/2, top*1.08, stars(q), ha="center", va="bottom", fontsize=12, fontweight="bold")
            annotation_max = max(annotation_max, top*1.42)
    ax.set_yscale("log"); ax.set_xlim(-.55, 4.55)
    ax.set_xticks(np.arange(5), ["1", "2", "3", "4", "5"])
    ax.set(xlabel="", ylabel="", title="")
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values(): spine.set_linewidth(1.7)
    positive = units.value.to_numpy(); ax.set_ylim(float(positive.min()*.70), annotation_max*8)
    order = [("Theory", True, "Theory direct"), ("Theory", False, "Theory non-direct"),
             ("Simulation", True, "Simulation direct"), ("Simulation", False, "Simulation non-direct")]
    handles = [Line2D([0], [0], marker="s", color="none", markerfacecolor=I1_COLORS[(s, d)],
                      markeredgecolor="none", alpha=.72, markersize=8.5, label=label) for s, d, label in order]
    fig.legend(handles=handles, frameon=False, loc="upper center", ncol=4, handletextpad=.45, columnspacing=1.4)
    stem = HERE / "Figure2i1_first_order_direct_nondirect_by_deleted_count"
    save_figure(fig, stem)
    atomic_json(HERE / "i1_plot_audit.json", {"zero_absolute_effects_omitted_from_log_plot": zero_count})
    return stem


def draw_i2(frame: pd.DataFrame) -> Path:
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 13, "axes.linewidth": 1.7, "xtick.labelsize": 13,
        "ytick.labelsize": 13, "legend.fontsize": 11.5,
        "svg.fonttype": "none", "pdf.fonttype": 42,
    })
    x = frame.theory_delta_r.to_numpy(float); y = frame.simulation_delta_r.to_numpy(float)
    use = np.isfinite(x) & np.isfinite(y); fit = fit_metrics(x, y)
    low = float(min(x[use].min(), y[use].min())); high = float(max(x[use].max(), y[use].max()))
    pad = .04*max(high-low, .1)
    fig, ax = plt.subplots(figsize=(5, 4)); fig.subplots_adjust(left=.12, right=.94, bottom=.12, top=.96)
    colors = {1: BLUE, 3: "#F4A261", 4: "#2A9D8F", 5: "#8E5EA2"}
    markers = {1: "o", 3: "^", 4: "s", 5: "D"}
    for count in sorted(frame.n_deleted.unique()):
        take = use & (frame.n_deleted.to_numpy(int) == int(count))
        ax.scatter(x[take], y[take], s=28, marker=markers[int(count)], color=colors[int(count)],
                   alpha=.85, label=f"{int(count)} nodes", rasterized=False)
    ax.plot([low-pad, high+pad], [low-pad, high+pad], color="#555555", ls="--", lw=1.7)
    ax.set_xlim(low-pad, high+pad); ax.set_ylim(low-pad, high+pad); ax.set_aspect("equal", adjustable="box")
    ax.set(xlabel="", ylabel="", title=""); ax.legend(frameon=False, loc="lower right")
    ax.text(.05, .94, f"r = {fit['pearson_r']:.2f}\nslope = {fit['ols_slope']:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=18, fontweight="bold")
    ax.tick_params(width=1.7, length=6)
    for spine in ax.spines.values(): spine.set_linewidth(1.7)
    stem = HERE / "Figure2i2_first_order_deletion_effect_theory_vs_Heun"
    save_figure(fig, stem)
    atomic_json(HERE / "i2_metrics.json", fit)
    return stem


def svg_audit(stem: Path) -> dict:
    root = ET.parse(stem.with_suffix(".svg")).getroot()
    images = [node for node in root.iter() if node.tag.endswith("}image")]
    text_nodes = [node for node in root.iter() if node.tag.endswith("}text")]
    return {
        "svg_parses": True, "embedded_raster_images": len(images),
        "editable_text_nodes": len(text_nodes),
    }


def qa(stems: list[Path]) -> dict:
    reports = {}
    validator = FIGURE_SKILL / "scripts/validate_figure.py"
    run = subprocess.run([sys.executable, str(validator), str(Path(__file__).resolve()), "--json"],
                         text=True, capture_output=True)
    atomic_text(HERE / "source-validation.json", run.stdout or run.stderr)
    reports["source_validation"] = {"returncode": run.returncode}
    if run.returncode != 0:
        raise RuntimeError(f"Figure source validation failed: {run.stdout} {run.stderr}")
    for stem in stems:
        text_run = subprocess.run([
            sys.executable, str(FIGURE_SKILL/"scripts/audit_pdf_text.py"),
            str(stem.with_suffix(".pdf")), "--min-pt", "5", "--json",
        ], text=True, capture_output=True)
        atomic_text(stem.with_suffix(".pdf-text-audit.json"), text_run.stdout or text_run.stderr)
        collision = subprocess.run([
            sys.executable, str(FIGURE_SKILL/"scripts/audit_figure_collisions.py"),
            str(stem.with_suffix(".pdf")), "--json-out", str(stem.with_suffix(".collision-audit.json")),
            "--overlay-pdf", str(stem.with_suffix(".collision-audit.pdf")),
        ], text=True, capture_output=True)
        reports[stem.name] = {
            "pdf_text_returncode": text_run.returncode,
            "collision_returncode": collision.returncode,
            "svg": svg_audit(stem),
        }
        if text_run.returncode != 0 or collision.returncode not in (0,):
            raise RuntimeError(f"QA failed for {stem.name}: {text_run.stdout} {collision.stdout}")
    atomic_json(HERE / "qa_summary.json", reports)
    return reports


def manifest() -> dict:
    files = [
        Path(__file__).resolve(), HERE/"protocol_locked.md", EXP2_CONDITIONS,
        EXP2_HEUN_RESULTS, EXP2_OLD_RESULTS, EXP2_SOURCE_SCRIPT, ZERO_INPUT_SOURCE,
        I1_SCRIPT, I1_ROOT/"full55_pattern_selection.csv", I1_DESIGN,
        I2_SCRIPT, I2_LINEAR, I2_ROOT/"pattern_results_heun.csv",
        PACKAGE/"pap_validation/exp1.py", PACKAGE/"pap_validation/core.py",
    ]
    return {
        "protocol": PROTOCOL, "created_at_unix": time.time(),
        "source_fingerprints": {str(p): sha256(p) for p in files},
        "state_fingerprints": {
            "i1": tree_digest(I1_STATES), "i2": tree_digest(I2_NONLINEAR),
            "exp2_theory_cache": tree_digest(EXP2_CACHE),
        },
        "software": {
            "python": sys.version, "numpy": np.__version__, "pandas": pd.__version__,
            "matplotlib": mpl.__version__, "platform": platform.platform(),
        },
        "methods": {
            "exp2_beta0": 0.0, "exp2_derivative_h": EXP2_H,
            "exp3_theta0": Z0, "exp3_theta_scale": ZSCALE,
            "exp3_standardized_derivative_h": EXP3_H,
            "diagnostic_steps": H_AUDIT,
        },
    }


def write_report(exp2: pd.DataFrame, i1: pd.DataFrame, i2: pd.DataFrame) -> None:
    e2 = json.loads((HERE/"experiment2_metrics.json").read_text())
    i1a = json.loads((HERE/"i1_computation_audit.json").read_text())
    i2a = json.loads((HERE/"i2_computation_audit.json").read_text())
    sig = pd.read_csv(HERE/"i1_significance.csv")
    i1_fit = fit_metrics(
        i1.theory_delta_r.to_numpy(float), i1.simulation_delta_r.to_numpy(float)
    )
    i1_abs_fit = fit_metrics(
        i1.theory_delta_r.abs().to_numpy(float), i1.simulation_delta_r.abs().to_numpy(float)
    )
    g_theory = exp2.theory_first_order_r.to_numpy(float)
    g_simulation = exp2.simulation_r.to_numpy(float)
    atomic_json(HERE/"i1_metrics.json", {
        "signed_delta_r": i1_fit, "absolute_delta_r": i1_abs_fit,
    })
    atomic_json(HERE/"g_distribution_metrics.json", {
        "n_conditions": len(exp2),
        "theory_mean": float(g_theory.mean()), "theory_median": float(np.median(g_theory)),
        "theory_sd": float(g_theory.std(ddof=1)),
        "simulation_mean": float(g_simulation.mean()),
        "simulation_median": float(np.median(g_simulation)),
        "simulation_sd": float(g_simulation.std(ddof=1)),
    })
    report = f"""# Figure 2 一阶理论重算审计报告

## 结论

本次主理论已从逐 trial 完整特征映射协方差改为锁定的一阶输入参数理论。所有结果均保留，无按拟合优劣筛选、无斜率校准、无更换展开中心或输入分布。

- Experiment 2：{len(exp2)} 个稳定复特征条件全部完成。一阶理论对完整非线性 Heun 的 Pearson r = {e2['first_order_vs_heun']['pearson_r']:.6f}，OLS slope = {e2['first_order_vs_heun']['ols_slope']:.6f}，R² = {e2['first_order_vs_heun']['R2']:.6f}，MAE = {e2['first_order_vs_heun']['MAE']:.6f}，RMSE = {e2['first_order_vs_heun']['RMSE']:.6f}。旧逐 trial 完整映射仅作审计对照：r = {e2['old_full_mapping_audit_vs_heun']['pearson_r']:.6f}，slope = {e2['old_full_mapping_audit_vs_heun']['ols_slope']:.6f}，R² = {e2['old_full_mapping_audit_vs_heun']['R2']:.6f}。
- g：一阶理论分布 mean={g_theory.mean():.6f}、median={np.median(g_theory):.6f}、SD={g_theory.std(ddof=1):.6f}；完整非线性 Heun 分布 mean={g_simulation.mean():.6f}、median={np.median(g_simulation):.6f}、SD={g_simulation.std(ddof=1):.6f}。
- i1：{len(i1)} 个结构－输出单位进入正式结果；结构－输出排除 {i1a['excluded_structure_output_pairs']} 个，理由逐条保存在排除表。signed Delta r 的理论－模拟 Pearson r={i1_fit['pearson_r']:.6f}、OLS slope={i1_fit['ols_slope']:.6f}、R²={i1_fit['R2']:.6f}；|Delta r| 的 Pearson r={i1_abs_fit['pearson_r']:.6f}、OLS slope={i1_abs_fit['ols_slope']:.6f}、R²={i1_abs_fit['R2']:.6f}。显著性全部由新一阶结果重新计算，未沿用旧 P 值。
- i2：82 个冻结删除结构中完成 {len(i2)} 个、排除 {i2a['excluded_patterns']} 个。一阶理论删除效应对完整非线性 Heun 的 Pearson r = {i2a['fit']['pearson_r']:.6f}，OLS slope = {i2a['fit']['ols_slope']:.6f}，R² = {i2a['fit']['R2']:.6f}，MAE = {i2a['fit']['MAE']:.6f}，RMSE = {i2a['fit']['RMSE']:.6f}。

## 实际方法与条件

Experiment 2 对每个既有 S 条件独立求完整 55 区域、110 状态网络的固定点、活跃集合与 Jacobian。以 beta0=0 为中心，对“连续全网络传递 PSD + 4.096 s periodic-Hann 谱窗卷积 + 登记特征提取”定义的 m=(P,A) 作中心有限差分，正式步长 h=0.002；信号协方差严格为 g Var_sample(beta) g^T。realization-error covariance 复用三个独立 calibration phase：每个 trial 的 beta 在 phase 间固定，先按 trial 做 phase 中心化再汇总，因此确定性的完整映射及其高阶项在误差估计中抵消。模拟值复用已经完整性验证的 110 状态两阶段 ReLU Heun：256 trial/condition，5 held-out phase，dt=0.25 ms，burn-in=2.048 s，分析窗=131.072 s。1000 条件无删除、无新增、无重新筛选。

Experiment 3 严格按当前 Experiment 1 参数化：theta=(峰中心、峰幅、beta、aperiodic 幅度)，theta0=(10,6,-0.5,1)，尺度=(3,4,0.8,0.7)，在标准化 u 坐标以 h=0.002 求 G=dm/du；四个分量全部进入 G 和样本协方差。intact 与每个 deleted structure 分别重求固定点、活跃集合、Jacobian 和 G。信号协方差为 G Cov_sample(u_common) G^T，校准协方差为独立 phase 的 trial 内残差协方差，并按 held-out PSD 平均相位数除以 2（i1）或 5（i2）。

i1 使用每个删除数 1–5 各 12 个冻结 deletion pattern、512 个冻结 trial、A6Va 输入、所有合格输出、16.384 s 分析窗、2 calibration + 2 held-out phase；直接输出定义为完整网络中接收任一被删除节点正 FLN 边的输出。统计单位是 deletion network：先在网络内分别平均 direct/nondirect 输出的 |Delta r|，再作配对双侧 100,000 次 sign-flip 检验，并对 5×2 个比较做 BH-FDR。新显著性表共 {len(sig)} 行。

i2 使用冻结的 82 个删除结构、kappa=100 输入协方差、128 trial、A6Va 输入、V2 输出、32.768 s 分析窗、3 calibration + 5 held-out phase；每个结构从其新固定点初始化完整 110 状态 Heun。删除集合、trial、seed、输入幅度和筛选规则均未改变。

## 导数与协方差诊断

- Experiment 2 预定步长比较 h=0.004/0.002/0.001；h=0.002 对 h=0.001 的最大相对差 {e2['gradient_convergence']['h002_vs_h001_max']:.6g}，中位数 {e2['gradient_convergence']['h002_vs_h001_median']:.6g}。
- g Var(beta) g^T 与逐 trial 一阶预测样本协方差的最大绝对差为 {e2['linear_covariance_identity_max_abs']:.6g}。
- i1 的 G Cov(u) G^T 恒等式最大绝对差为 {i1a['linear_covariance_identity_max_abs']:.6g}；i2 为 {i2a['linear_covariance_identity_max_abs']:.6g}。

## 数据复用与重算边界

- 复用：冻结输入设计；experiment2 的完整非线性 Heun held-out correlation 和独立 calibration covariance；i1 的完整 110 状态 Heun simulation/calibration 特征；i2 的完整 110 状态 Heun PSD 累积缓存和 Heun calibration 特征。
- 重算：所有一阶导数、所有一阶信号协方差、全部理论 r/Delta r、f/g/i1/i2 数据表、OLS 指标、i1 显著性，以及全部图形。
- 未重跑完整非线性积分，因为现有缓存的模型、输入、Heun 求解器、窗口、trial/phase 和来源指纹与锁定协议兼容；本次没有把固定活跃分支线性卷积冒充新运行的非线性仿真。
- 未修改任何 Methods 或 figure legend Word 文件。

## 每张图支持的理论层次

- f：beta 输入参数在 beta0=0 处的一阶特征理论，跨 1000 个完整网络条件验证。
- g：与 f 相同的一阶 beta 理论和完整非线性 Heun 结果的分布比较；stable-node 行仅保留在来源表中作追踪，不进入该 stable-complex 箱线图的两组统计。
- i1：四维输入参数一阶特征理论下，删除 1–5 个节点后 direct/nondirect 输出的 |Delta r| 分布和网络级配对统计。
- i2：四维输入参数一阶特征理论下，82 个 intact-versus-deleted 配对结构的 Delta r 与完整非线性 Heun 对应关系。

## 完整性与 QA

来源文件和缓存树的 SHA-256 已锁定在 manifest.json。四张 SVG 均由 Matplotlib 直接导出、保留可编辑文本、未嵌入栅格图像；相同最终版同时输出 PDF/TIFF/PNG。单 panel alignment、PDF 字号与 rendered collision 审计结果见 qa_summary.json 及各图旁的审计文件。
"""
    atomic_text(HERE/"audit_report_zh.md", report)


def write_integrity_summary(stems: list[Path]) -> None:
    tracked = [
        HERE/"protocol_locked.md", HERE/"manifest.json", HERE/"audit_report_zh.md",
        HERE/"experiment2_first_order_results.csv", HERE/"i1_first_order_raw_units.csv.gz",
        HERE/"i2_first_order_results.csv", HERE/"i1_significance.csv",
        HERE/"qa_summary.json", HERE/"qa_warning_disposition.md",
    ]
    for stem in stems:
        tracked.extend([stem.with_suffix(".svg"), stem.with_suffix(".pdf")])
    atomic_json(HERE/"integrity_summary.json", {
        "status": "PASS",
        "tracked_file_count": len(tracked),
        "sha256": {path.name: sha256(path) for path in tracked},
        "conditions": {
            "experiment2": 1000, "i1_raw_units": 2991, "i2_patterns": 82,
            "methods_or_legends_modified": False,
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--exp2", action="store_true")
    parser.add_argument("--i1", action="store_true")
    parser.add_argument("--i2", action="store_true")
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    atomic_json(HERE/"manifest.json", manifest())
    exp2 = run_exp2(args.workers) if (args.all or args.exp2) else None
    i1 = run_i1() if (args.all or args.i1) else None
    i2 = run_i2() if (args.all or args.i2) else None
    if args.all or args.figures:
        if exp2 is None:
            exp2 = pd.read_csv(HERE/"experiment2_first_order_results.csv")
        if i1 is None:
            i1 = pd.read_csv(HERE/"i1_first_order_raw_units.csv.gz")
        if i2 is None:
            i2 = pd.read_csv(HERE/"i2_first_order_results.csv")
        stems = [draw_f(exp2), draw_g(exp2), draw_i1(i1), draw_i2(i2)]
        qa(stems)
        write_report(exp2, i1, i2)
        write_integrity_summary(stems)


if __name__ == "__main__":
    main()
