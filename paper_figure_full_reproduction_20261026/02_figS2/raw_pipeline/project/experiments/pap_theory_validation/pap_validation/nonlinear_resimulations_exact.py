from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from .core import AREAS9, load_model
from .exp1 import Z0, ZHI, ZLO, ZSCALE
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config
from .nonlinear_network_validation import (
    _exp1_conditions,
    _fourier_inputs,
    _run_multioutput_batch,
)
from .paper_figure_dense_revision import (
    EXP1_EPSILON,
    EXP1_TRIALS,
    SEED as DENSE_SEED,
    _sample_exact_correlation,
)
from .realization_residual_validation import (
    CALIBRATION_PHASES,
    EXP1_ANALYSIS_S,
    HELDOUT_PHASES,
    _exp1_designs,
    _spectral_spec,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    ROOT
    / "experiments"
    / "pap_theory_validation"
    / "outputs"
    / "nonlinear_resimulations_exact"
)
CONFIG = ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml"
DT_MS = 0.25
BURN_S = 2.048
WELCH_SEGMENT_S = 4.096
SMOOTHING_SIGMA_HZ = 0.9765625
INPUT_SIGMA = 0.5
TOTAL_PHASES = CALIBRATION_PHASES + HELDOUT_PHASES
CHUNK = 16

SOURCE_FILES = [
    Path(__file__),
    ROOT / "experiments" / "pap_theory_validation" / "pap_validation" / "paper_figure_dense_revision.py",
    ROOT / "experiments" / "pap_theory_validation" / "pap_validation" / "realization_residual_validation.py",
    ROOT / "experiments" / "pap_theory_validation" / "pap_validation" / "exp3_dt_convergence.py",
    ROOT / "experiments" / "pap_theory_validation" / "outputs" / "paper_figure_node_deletion" / "manifest.json",
    ROOT / "experiments" / "pap_theory_validation" / "outputs" / "two_dimensional_ode_validation" / "hopf" / "manifest.json",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _lock_payload() -> dict:
    payload = {
        "schema": 1,
        "constraint": "Only replace transfer/resolvent simulation with direct nonlinear integration; all other source parameters are frozen.",
        "panels": {
            "hopf_B": {
                "analysis_s": 32.768,
                "trials": 64,
                "conditions": 1000,
                "calibration_phases": 3,
                "heldout_phases": 5,
                "dt_s": 0.00025,
                "burn_s": 2.048,
                "welch_segment_s": 4.096,
                "input_rms": 0.02,
                "phase_mapping": "pending_user_confirmation",
            },
            "hopf_F": {
                "analysis_s": 131.072,
                "trials": 256,
                "conditions": 1000,
                "calibration_phases": 3,
                "heldout_phases": 5,
                "dt_s": 0.00025,
                "burn_s": 2.048,
                "welch_segment_s": 4.096,
                "input_rms": 0.02,
                "phase_mapping": "pending_user_confirmation",
            },
            "wholebrain_C": {
                "analysis_s": 32.768,
                "trials": 64,
                "conditions": 1000,
                "dt_ms": 0.25,
                "burn_s": 2.048,
                "welch_segment_s": 4.096,
                "input_sigma": 0.5,
                "input_area": "V1",
                "output_area": "PG",
                "time_phase_seed": "20260819 + 100000 + condition_id*31",
            },
            "wholebrain_9x9": {
                "analysis_s": 32.768,
                "trials": 64,
                "condition_pos": 1,
                "calibration_phases": 3,
                "heldout_phases": 5,
                "dt_ms": 0.25,
                "burn_s": 2.048,
                "welch_segment_s": 4.096,
                "input_sigma": 0.5,
                "time_phase_seed": "cfg.seed + 8100000 + phase*100000 + input_pos*10000 + condition_pos",
            },
            "wholebrain_F": {
                "action": "unchanged_existing_nonlinear_tanh_integration",
                "analysis_s": 131.072,
                "trials": 256,
                "conditions": 1000,
                "calibration_phases": 3,
                "heldout_phases": 5,
            },
            "wholebrain_H": {
                "analysis_s": 65.536,
                "trials": 512,
                "patterns": 222,
                "calibration_phases": 3,
                "heldout_phases": 5,
                "dt_ms": 0.25,
                "burn_s": 2.048,
                "welch_segment_s": 4.096,
                "input_sigma": 0.5,
                "input_area": "A6Va",
                "output_area": "V2",
                "phase_mapping": "pending_user_confirmation",
            },
        },
        "source_sha256": {
            str(path.relative_to(ROOT)): _sha256(path)
            for path in SOURCE_FILES
            if path.exists()
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    temporary.write_text(text)
    temporary.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def _prepare() -> tuple[dict, str]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    lock = _lock_payload()
    lock_path = OUTPUT / "parameter_lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text())
        if previous.get("fingerprint") != lock["fingerprint"]:
            raise RuntimeError(
                "Parameter/source fingerprint changed. Move the old exact-output directory aside before rerunning."
            )
    else:
        _atomic_text(lock_path, json.dumps(lock, indent=2) + "\n")
    return lock, str(lock["fingerprint"])


def _cache_valid(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    with np.load(path) as data:
        return str(data["fingerprint"].item()) == fingerprint


def _corr(values: np.ndarray, valid: np.ndarray) -> float:
    use = np.asarray(valid, bool) & np.isfinite(values).all(axis=1)
    if use.sum() < 8 or np.min(np.std(values[use], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[use, 0], values[use, 1]).statistic)


def _progress(panel: str, completed: int, total: int, fingerprint: str) -> None:
    _atomic_text(
        OUTPUT / "progress.json",
        json.dumps(
            {
                "panel": panel,
                "completed_items": completed,
                "total_items": total,
                "fingerprint": fingerprint,
                "updated_unix_s": time.time(),
            },
            indent=2,
        )
        + "\n",
    )


def run_wb_c(max_items: int | None, walltime_minutes: float | None) -> None:
    _, fingerprint = _prepare()
    cfg = load_config(CONFIG)
    model = load_model(ROOT / cfg["model_dir"])
    source_path = (
        ROOT
        / "experiments"
        / "pap_theory_validation"
        / "outputs"
        / "paper_figure_dense_revision"
        / "exp1_dense_conditions.csv"
    )
    source = pd.read_csv(source_path)
    cache_dir = OUTPUT / "wholebrain_C" / "conditions"
    output_indices = np.asarray([model.areas.index("PG")], dtype=np.int64)
    input_idx = model.areas.index("V1")
    analysis_steps = int(round(EXP1_ANALYSIS_S * 1000.0 / DT_MS))
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    specs = _spectral_spec(EXP1_ANALYSIS_S)
    started = time.monotonic()
    made = 0
    for source_row, item in source.iterrows():
        condition_id = int(item["condition_id"] if "condition_id" in item else source_row)
        cache = cache_dir / f"condition_{condition_id:04d}.npz"
        if _cache_valid(cache, fingerprint):
            continue
        if max_items is not None and made >= max_items:
            break
        if walltime_minutes is not None and made and time.monotonic() - started >= 60.0 * walltime_minutes:
            break
        x, y = _sample_exact_correlation(
            EXP1_TRIALS,
            float(item.target_input_r),
            DENSE_SEED + condition_id * 17,
        )
        u = np.zeros((EXP1_TRIALS, 4))
        u[:, 0] = EXP1_EPSILON * x
        u[:, 2] = EXP1_EPSILON * y
        z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
        inputs = _fourier_inputs(
            z,
            DT_MS,
            analysis_steps,
            burn_steps,
            INPUT_SIGMA,
            DENSE_SEED + 100000 + condition_id * 31,
        )
        outputs, rect, min_e, min_i, min_de, min_di = _run_multioutput_batch(
            model.params,
            inputs,
            DT_MS,
            burn_steps,
            output_indices,
            input_idx=input_idx,
        )
        features, valid = _feature_arrays_vectorized(outputs, DT_MS, specs)["long"]
        _atomic_npz(
            cache,
            fingerprint=np.asarray(fingerprint),
            condition_id=np.asarray(condition_id),
            features=features[0].astype(np.float32),
            valid=valid[0],
            rectification=rect,
            min_rate_e=min_e,
            min_rate_i=min_i,
            min_drive_e=min_de,
            min_drive_i=min_di,
        )
        made += 1
        completed = sum(_cache_valid(cache_dir / f"condition_{int(row.condition_id):04d}.npz", fingerprint) for row in source.itertuples())
        _progress("wholebrain_C", completed, len(source), fingerprint)
        print(f"wholebrain_C exact nonlinear: {completed}/{len(source)}", flush=True)
    aggregate_wb_c(source, cache_dir, fingerprint)


def aggregate_wb_c(source: pd.DataFrame, cache_dir: Path, fingerprint: str) -> None:
    rows = []
    for source_row, item in source.iterrows():
        condition_id = int(item["condition_id"] if "condition_id" in item else source_row)
        cache = cache_dir / f"condition_{condition_id:04d}.npz"
        if not _cache_valid(cache, fingerprint):
            continue
        with np.load(cache) as data:
            features = data["features"].astype(float)
            valid = data["valid"]
            rect = data["rectification"]
            rows.append(
                {
                    "condition_id": condition_id,
                    "target_input_r": float(item.target_input_r),
                    "theory_r": float(item.first_residual_corrected_r),
                    "simulation_r": _corr(features, valid),
                    "n_trials": EXP1_TRIALS,
                    "n_valid": int(valid.sum()),
                    "rectification_count": int(rect.sum()),
                    "analysis_s": EXP1_ANALYSIS_S,
                }
            )
    _atomic_csv(OUTPUT / "wholebrain_C" / "exp1_nonlinear_full55_exact.csv", pd.DataFrame(rows))


def run_wb_9x9(max_items: int | None, walltime_minutes: float | None) -> None:
    _, fingerprint = _prepare()
    cfg = load_config(CONFIG)
    model = load_model(ROOT / cfg["model_dir"])
    conditions, _, all_z = _exp1_designs(cfg)
    condition_pos = 1
    if conditions[condition_pos]["condition"] != "rank_one_eps0.2_rho0":
        raise RuntimeError("The registered 9x9 condition at position 1 changed")
    z = all_z[condition_pos]
    cache_dir = OUTPUT / "wholebrain_9x9" / "phase_input"
    output_indices = np.asarray([model.areas.index(area) for area in AREAS9], dtype=np.int64)
    burn_steps = int(round(BURN_S * 1000.0 / DT_MS))
    specs = _spectral_spec(EXP1_ANALYSIS_S)
    items = [(phase, input_pos) for phase in range(TOTAL_PHASES) for input_pos in range(len(AREAS9))]
    started = time.monotonic()
    made = 0
    for phase, input_pos in items:
        cache = cache_dir / f"phase_{phase:02d}_input_{input_pos:02d}.npz"
        if _cache_valid(cache, fingerprint):
            continue
        if max_items is not None and made >= max_items:
            break
        if walltime_minutes is not None and made and time.monotonic() - started >= 60.0 * walltime_minutes:
            break
        features = np.empty((len(AREAS9), EXP1_TRIALS, 2), dtype=np.float32)
        valid = np.empty((len(AREAS9), EXP1_TRIALS), dtype=bool)
        rectification = np.empty(EXP1_TRIALS, dtype=np.int64)
        for start in range(0, EXP1_TRIALS, CHUNK):
            stop = min(start + CHUNK, EXP1_TRIALS)
            inputs = _matched_bandlimited_inputs(
                z[start:stop],
                DT_MS,
                EXP1_ANALYSIS_S,
                BURN_S,
                INPUT_SIGMA,
                int(cfg["seed"]) + 8100000 + phase * 100000 + input_pos * 10000 + condition_pos,
                trial_offset=start,
            )
            outputs, rect, *_ = _run_multioutput_batch(
                model.params,
                inputs,
                DT_MS,
                burn_steps,
                output_indices,
                input_idx=model.areas.index(AREAS9[input_pos]),
            )
            values, is_valid = _feature_arrays_vectorized(outputs, DT_MS, specs)["long"]
            features[:, start:stop] = values
            valid[:, start:stop] = is_valid
            rectification[start:stop] = rect
        _atomic_npz(
            cache,
            fingerprint=np.asarray(fingerprint),
            phase=np.asarray(phase),
            input_pos=np.asarray(input_pos),
            features=features,
            valid=valid,
            rectification=rectification,
        )
        made += 1
        completed = sum(
            _cache_valid(cache_dir / f"phase_{p:02d}_input_{i:02d}.npz", fingerprint)
            for p, i in items
        )
        _progress("wholebrain_9x9", completed, len(items), fingerprint)
        print(f"wholebrain_9x9 exact nonlinear: {completed}/{len(items)}", flush=True)
    aggregate_wb_9x9(cache_dir, fingerprint)


def aggregate_wb_9x9(cache_dir: Path, fingerprint: str) -> None:
    paths = [
        cache_dir / f"phase_{phase:02d}_input_{input_pos:02d}.npz"
        for phase in range(TOTAL_PHASES)
        for input_pos in range(len(AREAS9))
    ]
    if not all(_cache_valid(path, fingerprint) for path in paths):
        return
    features = np.empty((TOTAL_PHASES, len(AREAS9), len(AREAS9), EXP1_TRIALS, 2), dtype=np.float32)
    valid = np.empty((TOTAL_PHASES, len(AREAS9), len(AREAS9), EXP1_TRIALS), dtype=bool)
    for phase in range(TOTAL_PHASES):
        for input_pos in range(len(AREAS9)):
            with np.load(cache_dir / f"phase_{phase:02d}_input_{input_pos:02d}.npz") as data:
                features[phase, input_pos] = data["features"]
                valid[phase, input_pos] = data["valid"]
    matrix = np.full((len(AREAS9), len(AREAS9)), np.nan)
    common_counts = np.zeros_like(matrix, dtype=int)
    phase_sd = np.full_like(matrix, np.nan)
    for input_pos in range(len(AREAS9)):
        for output_pos in range(len(AREAS9)):
            use = valid[:, input_pos, output_pos].all(axis=0)
            rs = np.asarray(
                [
                    _corr(features[phase, input_pos, output_pos], use)
                    for phase in range(CALIBRATION_PHASES, TOTAL_PHASES)
                ]
            )
            matrix[output_pos, input_pos] = float(np.nanmean(rs))
            phase_sd[output_pos, input_pos] = float(np.nanstd(rs, ddof=1))
            common_counts[output_pos, input_pos] = int(use.sum())
    matrix_frame = pd.DataFrame(matrix, index=AREAS9, columns=AREAS9)
    counts_frame = pd.DataFrame(common_counts, index=AREAS9, columns=AREAS9)
    sd_frame = pd.DataFrame(phase_sd, index=AREAS9, columns=AREAS9)
    matrix_frame.to_csv(OUTPUT / "wholebrain_9x9" / "heldout_mean_correlation_exact.csv")
    counts_frame.to_csv(OUTPUT / "wholebrain_9x9" / "common_valid_counts.csv")
    sd_frame.to_csv(OUTPUT / "wholebrain_9x9" / "heldout_phase_sd.csv")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("wb_c", "wb_9x9"), required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--walltime-minutes", type=float)
    args = parser.parse_args()
    if args.panel == "wb_c":
        run_wb_c(args.max_items, args.walltime_minutes)
    else:
        run_wb_9x9(args.max_items, args.walltime_minutes)


if __name__ == "__main__":
    main()
