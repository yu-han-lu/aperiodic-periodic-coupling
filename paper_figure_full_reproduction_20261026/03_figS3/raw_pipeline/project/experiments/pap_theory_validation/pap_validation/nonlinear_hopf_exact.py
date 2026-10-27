from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, signal, stats

from .core import feature_from_spectrum
from .nonlinear_resimulations import _hopf_heun_batch
from .two_dimensional_ode_validation import (
    BURN_S,
    CAL_PHASES,
    DT,
    EXP1_CONDITIONS,
    EXP1_TRIALS,
    EXP1_WINDOW_S,
    EXP2_CONDITIONS,
    EXP2_TRIALS,
    EXP2_WINDOW_S,
    HELDOUT_PHASES,
    MODELS,
    PEAK_BAND,
    SEED,
    SLOPE_BAND,
    WELCH_SEGMENT_S,
    _exp1_trials,
)


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "two_dimensional_ode_validation" / "hopf"
OUTPUT = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "nonlinear_resimulations_exact" / "hopf"
CONFIRMATION = OUTPUT.parent / "phase_mapping_confirmation.json"
INPUT_RMS = 0.02
SMOOTHING_SIGMA_HZ = 0.9765625
TOTAL_PHASES = CAL_PHASES + HELDOUT_PHASES
TRIAL_SEED_STRIDE = 1000003
PHASE_SEED_STRIDE = 100000
CHUNK = 32


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _payload() -> dict:
    payload = {
        "schema": 1,
        "confirmed_phase_rule": json.loads(CONFIRMATION.read_text()),
        "phase_derivation": "phase_seed = original_seed_root + phase*100000; trial_seed = phase_seed + trial*1000003",
        "panels": {
            "B": {
                "analysis_s": EXP1_WINDOW_S,
                "conditions": EXP1_CONDITIONS,
                "trials": EXP1_TRIALS,
                "calibration_phases": CAL_PHASES,
                "heldout_phases": HELDOUT_PHASES,
                "seed_root": "20260809 + 200000 + condition",
            },
            "F": {
                "analysis_s": EXP2_WINDOW_S,
                "conditions": EXP2_CONDITIONS,
                "trials": EXP2_TRIALS,
                "calibration_phases": CAL_PHASES,
                "heldout_phases": HELDOUT_PHASES,
                "seed_root": "20260809 + 700000 + source candidate condition",
            },
        },
        "shared": {
            "system": "full cubic Hopf normal form",
            "integrator": "explicit Heun",
            "dt_s": DT,
            "burn_s": BURN_S,
            "welch_segment_s": WELCH_SEGMENT_S,
            "welch_overlap": 0.5,
            "smoothing_sigma_hz": SMOOTHING_SIGMA_HZ,
            "peak_band_hz": list(PEAK_BAND),
            "slope_band_hz": list(SLOPE_BAND),
            "input_rms": INPUT_RMS,
        },
        "source_sha256": {
            "exp1": _sha256(SOURCE / "exp1_1000_conditions.csv"),
            "exp2": _sha256(SOURCE / "exp2_1000_conditions_rlt0p6.csv"),
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
            raise RuntimeError("Hopf parameter/source fingerprint changed")
    else:
        _atomic_text(lock, json.dumps(payload, indent=2) + "\n")
    return str(payload["fingerprint"])


def _cache_valid(path: Path, fingerprint: str) -> bool:
    if not path.exists():
        return False
    with np.load(path) as data:
        return str(data["fingerprint"].item()) == fingerprint


def _normalized_shape(frequencies: np.ndarray, beta: float) -> np.ndarray:
    f = np.maximum(frequencies, 0.5)
    raw = (f / 10.0) ** beta
    return raw / raw.mean()


def _forcing_trial_exp1(z: np.ndarray, analysis_s: float, seed: int) -> np.ndarray:
    analysis_steps = int(round(analysis_s / DT))
    burn_steps = int(round(BURN_S / DT))
    frequencies = np.fft.rfftfreq(analysis_steps, DT)
    f = np.maximum(frequencies, 0.5)
    center, bump_amp, beta, ap_amp = np.asarray(z, float)
    q = ap_amp * (f / 10.0) ** beta
    q += bump_amp * np.exp(-0.5 * ((f - center) / 2.0) ** 2)
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
    spectrum = np.sqrt(np.maximum(q, 0.0)) * np.exp(1j * phase)
    spectrum[0] = 0.0
    if analysis_steps % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    x = np.fft.irfft(spectrum, n=analysis_steps)
    x *= INPUT_RMS / max(x.std(), 1e-15)
    x -= x.mean()
    return np.concatenate((x[-burn_steps:], x)).astype(np.float32)


def _forcing_trial_exp2(beta: float, floor_ratio: float, analysis_s: float, seed: int) -> np.ndarray:
    analysis_steps = int(round(analysis_s / DT))
    burn_steps = int(round(BURN_S / DT))
    frequencies = np.fft.rfftfreq(analysis_steps, DT)
    rng = np.random.default_rng(seed)
    q = _normalized_shape(frequencies, float(beta))
    phase = rng.uniform(0.0, 2.0 * np.pi, len(frequencies))
    spectrum = np.sqrt(q) * np.exp(1j * phase)
    spectrum[0] = 0.0
    if analysis_steps % 2 == 0:
        spectrum[-1] = spectrum[-1].real
    colored = np.fft.irfft(spectrum, n=analysis_steps)
    colored *= INPUT_RMS / max(colored.std(), 1e-15)
    floor = INPUT_RMS * floor_ratio * rng.normal(size=analysis_steps)
    x = colored + floor
    x -= x.mean()
    return np.concatenate((x[-burn_steps:], x)).astype(np.float32)


def _extract(paths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fs = 1.0 / DT
    nperseg = int(round(WELCH_SEGMENT_S / DT))
    frequencies, psd = signal.welch(
        paths,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=nperseg // 2,
        detrend="constant",
        scaling="density",
        axis=-1,
    )
    keep = (frequencies >= 0.5) & (frequencies <= 1200.0)
    f = frequencies[keep]
    sigma_bins = SMOOTHING_SIGMA_HZ / (f[1] - f[0])
    smooth = np.exp(
        ndimage.gaussian_filter1d(
            np.log(np.maximum(psd[:, keep], 1e-300)), sigma_bins, axis=-1
        )
    )
    values = np.empty((len(paths), 2), dtype=np.float32)
    valid = np.empty(len(paths), dtype=bool)
    for trial in range(len(paths)):
        feature = feature_from_spectrum(
            f, smooth[trial], peak_band=PEAK_BAND, slope_band=SLOPE_BAND
        )
        values[trial] = feature["peak_hz"], -feature["exponent"]
        valid[trial] = bool(
            not feature["peak_boundary"]
            and "no_interior_peak" not in feature["feature_failure_reason"]
            and np.isfinite(values[trial]).all()
        )
    return values, valid


def _corr(values: np.ndarray, valid: np.ndarray) -> float:
    use = np.asarray(valid, bool) & np.isfinite(values).all(axis=1)
    if use.sum() < 8 or np.min(np.std(values[use], axis=0)) <= 0:
        return np.nan
    return float(stats.pearsonr(values[use, 0], values[use, 1]).statistic)


def _run_phase(
    panel: str,
    source_id: int,
    phase: int,
    parameter: float,
    trial_values: np.ndarray,
    floor_ratio: float,
    analysis_s: float,
    seed_root: int,
    fingerprint: str,
    cache: Path,
) -> None:
    n_trials = len(trial_values)
    features = np.empty((n_trials, 2), dtype=np.float32)
    valid = np.empty(n_trials, dtype=bool)
    phase_seed = seed_root + phase * PHASE_SEED_STRIDE
    burn_steps = int(round(BURN_S / DT))
    for start in range(0, n_trials, CHUNK):
        stop = min(start + CHUNK, n_trials)
        forcing = np.asarray(
            [
                _forcing_trial_exp1(trial_values[trial], analysis_s, phase_seed + trial * TRIAL_SEED_STRIDE)
                if panel == "B"
                else _forcing_trial_exp2(float(trial_values[trial]), floor_ratio, analysis_s, phase_seed + trial * TRIAL_SEED_STRIDE)
                for trial in range(start, stop)
            ]
        )
        paths = _hopf_heun_batch(
            forcing, parameter, MODELS["hopf"].omega_hz, DT, burn_steps
        )
        values, is_valid = _extract(paths)
        features[start:stop] = values
        valid[start:stop] = is_valid
    _atomic_npz(
        cache,
        fingerprint=np.asarray(fingerprint),
        panel=np.asarray(panel),
        source_id=np.asarray(source_id),
        phase=np.asarray(phase),
        features=features,
        valid=valid,
    )


def _progress(panel: str, completed: int, total: int, fingerprint: str) -> None:
    _atomic_text(
        OUTPUT / f"progress_{panel}.json",
        json.dumps(
            {
                "panel": panel,
                "completed_condition_phases": completed,
                "total_condition_phases": total,
                "fingerprint": fingerprint,
                "updated_unix_s": time.time(),
            },
            indent=2,
        )
        + "\n",
    )


def run_panel(panel: str, max_items: int | None, walltime_minutes: float | None) -> None:
    fingerprint = _prepare()
    if panel == "B":
        source = pd.read_csv(SOURCE / "exp1_1000_conditions.csv")
        analysis_s = EXP1_WINDOW_S
        trials = EXP1_TRIALS
    else:
        source = pd.read_csv(SOURCE / "exp2_1000_conditions_rlt0p6.csv")
        analysis_s = EXP2_WINDOW_S
        trials = EXP2_TRIALS
    cache_dir = OUTPUT / panel / "condition_phase"
    items = [(source_id, phase) for source_id in range(len(source)) for phase in range(TOTAL_PHASES)]
    completed = sum(
        _cache_valid(path, fingerprint)
        for path in cache_dir.glob("condition_*_phase_*.npz")
    )
    started = time.monotonic()
    made = 0
    for source_id, phase in items:
        cache = cache_dir / f"condition_{source_id:04d}_phase_{phase:02d}.npz"
        if _cache_valid(cache, fingerprint):
            continue
        if max_items is not None and made >= max_items:
            break
        if walltime_minutes is not None and made and time.monotonic() - started >= walltime_minutes * 60.0:
            break
        item = source.iloc[source_id]
        if panel == "B":
            trial_values = _exp1_trials(float(item.target_input_rho), source_id)
            parameter = MODELS["hopf"].exp1_parameter
            floor_ratio = 0.0
            seed_root = SEED + 200000 + source_id
        else:
            condition = int(item.condition)
            sigma_beta = float(item.sigma_beta)
            rng = np.random.default_rng(SEED + 500000 + condition)
            trial_values = stats.truncnorm.rvs(
                -0.8 / sigma_beta,
                0.8 / sigma_beta,
                loc=0.0,
                scale=sigma_beta,
                size=trials,
                random_state=rng,
            )
            parameter = float(item.mu_hz)
            floor_ratio = float(item.floor_ratio)
            seed_root = SEED + 700000 + condition
        if len(trial_values) != trials:
            raise RuntimeError("Registered trial count changed")
        _run_phase(
            panel,
            source_id,
            phase,
            parameter,
            trial_values,
            floor_ratio,
            analysis_s,
            seed_root,
            fingerprint,
            cache,
        )
        made += 1
        completed += 1
        _progress(panel, completed, len(items), fingerprint)
        print(f"Hopf {panel} exact nonlinear: {completed}/{len(items)}", flush=True)
    aggregate(panel, source, cache_dir, fingerprint)


def aggregate(panel: str, source: pd.DataFrame, cache_dir: Path, fingerprint: str) -> None:
    rows = []
    representative_rows = []
    representative_source = None
    if panel == "F":
        score = (
            (source.mu_hz + 2.5).abs() / source.mu_hz.std()
            + (source.residual_corrected_theory_r - 0.30).abs()
            / source.residual_corrected_theory_r.std()
        )
        representative_source = int(score.idxmin())
    for source_id, item in source.iterrows():
        paths = [cache_dir / f"condition_{source_id:04d}_phase_{phase:02d}.npz" for phase in range(TOTAL_PHASES)]
        if not all(_cache_valid(path, fingerprint) for path in paths):
            continue
        phase_features = []
        phase_valid = []
        for path in paths:
            with np.load(path) as data:
                phase_features.append(data["features"].astype(float))
                phase_valid.append(data["valid"])
        phase_features = np.asarray(phase_features)
        phase_valid = np.asarray(phase_valid)
        common = phase_valid.all(axis=0)
        heldout_r = np.asarray(
            [_corr(phase_features[phase], common) for phase in range(CAL_PHASES, TOTAL_PHASES)]
        )
        row = item.to_dict()
        row.update(
            {
                "source_condition": int(source_id),
                "simulation_r": float(np.nanmean(heldout_r)),
                "simulation_sd": float(np.nanstd(heldout_r, ddof=1)),
                "n_valid_common": int(common.sum()),
                "heldout_phase_r_json": json.dumps(heldout_r.tolist()),
                "analysis_s": EXP1_WINDOW_S if panel == "B" else EXP2_WINDOW_S,
            }
        )
        rows.append(row)
        if panel == "F" and source_id == representative_source:
            heldout = phase_features[CAL_PHASES:, common]
            reference_mean = heldout.reshape(-1, 2).mean(axis=0, keepdims=True)
            reference_sd = heldout.reshape(-1, 2).std(axis=0, ddof=1, keepdims=True)
            aligned = (
                (heldout - heldout.mean(axis=1, keepdims=True))
                / heldout.std(axis=1, ddof=1, keepdims=True)
                * reference_sd[None, :, :]
                + reference_mean[None, :, :]
            )
            condition = int(item.condition)
            sigma_beta = float(item.sigma_beta)
            rng = np.random.default_rng(SEED + 500000 + condition)
            betas = stats.truncnorm.rvs(
                -0.8 / sigma_beta, 0.8 / sigma_beta, loc=0.0, scale=sigma_beta,
                size=EXP2_TRIALS, random_state=rng,
            )[common]
            for phase_pos in range(HELDOUT_PHASES):
                for trial_pos, original_trial in enumerate(np.flatnonzero(common)):
                    representative_rows.append(
                        {
                            "heldout_realization": phase_pos + 1,
                            "trial": int(original_trial),
                            "beta": float(betas[trial_pos]),
                            "peak_hz": float(aligned[phase_pos, trial_pos, 0]),
                            "aperiodic_slope": float(aligned[phase_pos, trial_pos, 1]),
                        }
                    )
    _atomic_csv(OUTPUT / panel / "condition_results_exact.csv", pd.DataFrame(rows))
    if representative_rows:
        _atomic_csv(OUTPUT / panel / "representative_heldout_trials_exact.csv", pd.DataFrame(representative_rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel", choices=("B", "F"), required=True)
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--walltime-minutes", type=float)
    args = parser.parse_args()
    run_panel(args.panel, args.max_items, args.walltime_minutes)


if __name__ == "__main__":
    main()
