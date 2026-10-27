from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal, stats

from .core import load_model
from .exp3_dt_convergence import _matched_bandlimited_inputs
from .io import load_config
from .nonlinear_resimulations import _deleted_fln, _network_run
from .paper_figure_node_deletion import (
    CALIBRATION_PHASES,
    DT_MS,
    EXP3_ANALYSIS_S,
    EXP3_TRIALS,
    HELDOUT_PHASES,
    SEED,
    WELCH_SEGMENT_S,
    _features_from_averaged_psd,
)


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "experiments" / "pap_theory_validation" / "outputs"
SOURCE_DIR = BASE / "single_node_deletion_all_nodes"
SOURCE_RESULTS = SOURCE_DIR / "combined_single_multinode_deletion_results.csv"
SOURCE_SHARED = SOURCE_DIR / "all_single_node_deletion_features.npz"
SOURCE_SPECIAL = BASE / "paper_figure_node_deletion" / "exp3_node_deletion_features_robust_v3.npz"
CONFIRMATION = BASE / "nonlinear_resimulations_exact" / "phase_mapping_confirmation.json"
OUTPUT = BASE / "nonlinear_resimulations_exact" / "wholebrain_H"
CONFIG = ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml"
INPUT_SIGMA = 0.5
INPUT_AREA = "A6Va"
OUTPUT_AREA = "V2"
TOTAL_PHASES = CALIBRATION_PHASES + HELDOUT_PHASES
PHASE_SEED_STRIDE = 100000
CHUNK = 16
SPECIAL_ROW = 221


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _payload() -> dict:
    payload = {
        "schema": 1,
        "confirmed_phase_rule": json.loads(CONFIRMATION.read_text()),
        "system": "55-area, 110-state threshold-nonlinear E-I ODE",
        "integrator": "forward Euler with positive-part nonlinearity evaluated at every step",
        "analysis_s": EXP3_ANALYSIS_S,
        "trials": EXP3_TRIALS,
        "patterns": 222,
        "calibration_phases": CALIBRATION_PHASES,
        "heldout_phases": HELDOUT_PHASES,
        "dt_ms": DT_MS,
        "burn_s": 2.048,
        "welch_segment_s": WELCH_SEGMENT_S,
        "welch_overlap": 0.5,
        "input_sigma": INPUT_SIGMA,
        "input_area": INPUT_AREA,
        "output_area": OUTPUT_AREA,
        "external_background": "frozen at baseline; no post-deletion recentering",
        "shared_group_phase_seed": "20260807 + 1010000 + phase*100000; trial stride is 1000003 inside matched input generator",
        "registered_12node_phase_seed": "20260807 + 610000 + phase*100000; trial stride is 1000003 inside matched input generator",
        "aggregation": "average Welch PSD within 3 calibration and 5 held-out phases before feature extraction, matching the source H panel",
        "source_sha256": {
            "results": _sha256(SOURCE_RESULTS),
            "shared_design": _sha256(SOURCE_SHARED),
            "registered_12node_design": _sha256(SOURCE_SPECIAL),
            "confirmation": _sha256(CONFIRMATION),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["fingerprint"] = hashlib.sha256(canonical).hexdigest()
    return payload


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _prepare() -> str:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = _payload()
    lock = OUTPUT / "parameter_lock.json"
    if lock.exists():
        old = json.loads(lock.read_text())
        if old.get("fingerprint") != payload["fingerprint"]:
            raise RuntimeError("H parameter/source fingerprint changed")
    else:
        _atomic_text(lock, json.dumps(payload, indent=2) + "\n")
    return str(payload["fingerprint"])


def _load_designs() -> tuple[np.ndarray, np.ndarray]:
    with np.load(SOURCE_SHARED) as data:
        shared = data["z"].astype(float)
    with np.load(SOURCE_SPECIAL) as data:
        special = data["z"].astype(float)
    if shared.shape != (EXP3_TRIALS, 4) or special.shape != (EXP3_TRIALS, 4):
        raise RuntimeError("Registered H trial design shape changed")
    return shared, special


def _empty_state(fingerprint: str, frequencies: np.ndarray) -> dict:
    shape = (EXP3_TRIALS, len(frequencies))
    return {
        "fingerprint": np.asarray(fingerprint),
        "frequencies": frequencies,
        "completed": np.zeros(TOTAL_PHASES, dtype=bool),
        "calibration_sum": np.zeros(shape, dtype=np.float32),
        "heldout_sum": np.zeros(shape, dtype=np.float32),
        "rectification": np.zeros(TOTAL_PHASES, dtype=np.int64),
    }


def _load_state(path: Path, fingerprint: str, frequencies: np.ndarray) -> dict:
    if not path.exists():
        return _empty_state(fingerprint, frequencies)
    with np.load(path) as data:
        if str(data["fingerprint"].item()) != fingerprint:
            raise RuntimeError(f"Fingerprint mismatch: {path}")
        if not np.array_equal(data["frequencies"], frequencies):
            raise RuntimeError(f"Frequency grid mismatch: {path}")
        return {key: data[key].copy() for key in data.files}


def _state_complete(path: Path, fingerprint: str, frequencies: np.ndarray) -> bool:
    if not path.exists():
        return False
    state = _load_state(path, fingerprint, frequencies)
    return bool(state["completed"].all())


def _frequency_grid() -> tuple[np.ndarray, np.ndarray]:
    fs = 1000.0 / DT_MS
    nperseg = int(round(WELCH_SEGMENT_S * fs))
    frequencies = np.fft.rfftfreq(nperseg, d=1.0 / fs)
    keep = (frequencies >= 0.5) & (frequencies <= 1200.0)
    return frequencies[keep], keep


def _nodes(item: pd.Series) -> list[str]:
    value = item.deleted_nodes
    if not isinstance(value, str) or not value:
        value = item.deleted_node
    return str(value).split("|")


def _phase_psd(model, z: np.ndarray, phase: int, seed_root: int, fln: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int]:
    kept_frequencies, keep = _frequency_grid()
    fs = 1000.0 / DT_MS
    nperseg = int(round(WELCH_SEGMENT_S * fs))
    psd = np.empty((EXP3_TRIALS, len(kept_frequencies)), dtype=np.float32)
    rectification = 0
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    for start in range(0, EXP3_TRIALS, CHUNK):
        stop = min(start + CHUNK, EXP3_TRIALS)
        inputs = _matched_bandlimited_inputs(
            z[start:stop],
            DT_MS,
            EXP3_ANALYSIS_S,
            2.048,
            INPUT_SIGMA,
            seed_root + phase * PHASE_SEED_STRIDE,
            trial_offset=start,
        )
        paths, rect = _network_run(
            model,
            inputs,
            np.asarray([output_idx], dtype=np.int64),
            input_idx,
            fln,
        )
        _, power = signal.welch(
            paths[:, 0],
            fs=fs,
            window="hann",
            nperseg=nperseg,
            noverlap=nperseg // 2,
            detrend="constant",
            scaling="density",
            axis=-1,
        )
        psd[start:stop] = power[:, keep]
        rectification += int(rect.sum())
        print(
            f"H phase {phase + 1}/{TOTAL_PHASES}: trials {start + 1}-{stop}/{EXP3_TRIALS}",
            flush=True,
        )
    return kept_frequencies, psd, rectification


def _advance_state(
    state_path: Path,
    model,
    z: np.ndarray,
    seed_root: int,
    fln: np.ndarray | None,
    fingerprint: str,
    frequencies: np.ndarray,
) -> int | None:
    state = _load_state(state_path, fingerprint, frequencies)
    missing = np.flatnonzero(~state["completed"])
    if not len(missing):
        return None
    phase = int(missing[0])
    actual_frequencies, psd, rectification = _phase_psd(
        model, z, phase, seed_root, fln
    )
    if not np.array_equal(actual_frequencies, frequencies):
        raise RuntimeError("Welch frequency grid changed")
    if phase < CALIBRATION_PHASES:
        state["calibration_sum"] += psd
    else:
        state["heldout_sum"] += psd
    state["rectification"][phase] = rectification
    state["completed"][phase] = True
    _atomic_npz(state_path, **state)
    return phase


def _feature_pair(state: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    frequencies = state["frequencies"]
    calibration_psd = state["calibration_sum"] / CALIBRATION_PHASES
    heldout_psd = state["heldout_sum"] / HELDOUT_PHASES
    calibration, calibration_valid = _features_from_averaged_psd(
        frequencies, calibration_psd[None, :, :]
    )
    heldout, heldout_valid = _features_from_averaged_psd(
        frequencies, heldout_psd[None, :, :]
    )
    return calibration[0], calibration_valid[0], heldout[0], heldout_valid[0]


def _corr(values: np.ndarray, use: np.ndarray) -> float:
    values = np.asarray(values, float)
    use = np.asarray(use, bool) & np.isfinite(values).all(axis=1)
    if use.sum() < 8 or np.min(np.std(values[use], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[use, 0], values[use, 1]).statistic)


def aggregate(source: pd.DataFrame, fingerprint: str, frequencies: np.ndarray) -> None:
    shared_baseline_path = OUTPUT / "states" / "baseline_shared.npz"
    special_baseline_path = OUTPUT / "states" / "baseline_registered_12node.npz"
    rows = []
    for row_id, item in source.iterrows():
        pattern_path = OUTPUT / "states" / f"pattern_{row_id:03d}.npz"
        baseline_path = special_baseline_path if row_id == SPECIAL_ROW else shared_baseline_path
        if not _state_complete(pattern_path, fingerprint, frequencies):
            continue
        if not _state_complete(baseline_path, fingerprint, frequencies):
            continue
        baseline_state = _load_state(baseline_path, fingerprint, frequencies)
        pattern_state = _load_state(pattern_path, fingerprint, frequencies)
        _, base_cal_valid, base_hold, base_hold_valid = _feature_pair(baseline_state)
        _, deleted_cal_valid, deleted_hold, deleted_hold_valid = _feature_pair(pattern_state)
        use = base_cal_valid & deleted_cal_valid & base_hold_valid & deleted_hold_valid
        baseline_r = _corr(base_hold, use)
        deleted_r = _corr(deleted_hold, use)
        result = item.to_dict()
        result.update(
            {
                "row_id": int(row_id),
                "simulation_baseline_r": baseline_r,
                "simulation_deleted_r": deleted_r,
                "simulation_delta_r": deleted_r - baseline_r,
                "n_valid": int(use.sum()),
                "baseline_rectification_count": int(baseline_state["rectification"].sum()),
                "deleted_rectification_count": int(pattern_state["rectification"].sum()),
                "analysis_s": EXP3_ANALYSIS_S,
            }
        )
        rows.append(result)
    _atomic_csv(OUTPUT / "node_deletion_nonlinear_exact.csv", pd.DataFrame(rows))


def run(max_items: int | None, walltime_minutes: float | None) -> None:
    fingerprint = _prepare()
    cfg = load_config(CONFIG)
    model = load_model(ROOT / cfg["model_dir"])
    source = pd.read_csv(SOURCE_RESULTS)
    if len(source) != 222:
        raise RuntimeError("Registered H pattern count changed")
    shared_z, special_z = _load_designs()
    frequencies, _ = _frequency_grid()
    states = OUTPUT / "states"
    work = [
        ("baseline_shared", states / "baseline_shared.npz", shared_z, SEED + 1010000, None),
        ("baseline_registered_12node", states / "baseline_registered_12node.npz", special_z, SEED + 610000, None),
    ]
    for row_id, item in source.iterrows():
        z = special_z if row_id == SPECIAL_ROW else shared_z
        seed_root = SEED + 610000 if row_id == SPECIAL_ROW else SEED + 1010000
        work.append(
            (
                f"pattern_{row_id:03d}",
                states / f"pattern_{row_id:03d}.npz",
                z,
                seed_root,
                _deleted_fln(model, _nodes(item)),
            )
        )
    started = time.monotonic()
    made = 0
    while True:
        pending = [item for item in work if not _state_complete(item[1], fingerprint, frequencies)]
        if not pending:
            break
        if max_items is not None and made >= max_items:
            break
        if walltime_minutes is not None and made and time.monotonic() - started >= walltime_minutes * 60.0:
            break
        label, state_path, z, seed_root, fln = pending[0]
        phase = _advance_state(
            state_path, model, z, seed_root, fln, fingerprint, frequencies
        )
        made += 1
        completed = sum(
            int(_load_state(path, fingerprint, frequencies)["completed"].sum())
            for _, path, *_ in work
            if path.exists()
        )
        total = len(work) * TOTAL_PHASES
        _atomic_text(
            OUTPUT / "progress.json",
            json.dumps(
                {
                    "last_item": label,
                    "last_phase": phase,
                    "completed_pattern_phases": completed,
                    "total_pattern_phases": total,
                    "fingerprint": fingerprint,
                    "updated_unix_s": time.time(),
                },
                indent=2,
            )
            + "\n",
        )
        print(f"wholebrain_H exact nonlinear: {completed}/{total} pattern-phases", flush=True)
        aggregate(source, fingerprint, frequencies)
    aggregate(source, fingerprint, frequencies)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--walltime-minutes", type=float)
    args = parser.parse_args()
    run(args.max_items, args.walltime_minutes)


if __name__ == "__main__":
    main()
