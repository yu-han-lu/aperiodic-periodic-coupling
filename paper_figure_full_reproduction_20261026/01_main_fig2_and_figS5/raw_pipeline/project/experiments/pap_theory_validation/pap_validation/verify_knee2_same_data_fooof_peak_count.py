from __future__ import annotations

"""Strict same-data check of the knee=2 Hz alpha discrepancy.

The custom sequential one-peak approximation and official FOOOFGroup fits are
applied to exactly the same train/held-out spectra used by the knee=2 six-panel
figure.  This isolates peak-model choice from OU-realisation and trial noise.
"""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import fooof_split_attached_code_theory as base
import fooof_split_researcher_e_gamma_fix as fixed
import fooof_split_researcher_ou_knee2_six_panel as knee2
import fooof_split_researcher_ou_six_panel as six
import fooof_split_researcher_pipeline_matched as matched
import reproduce_gamma_artifact as researcher
import rerun_researcher_official_fooof_bands as official


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "knee2_same_data_fooof_peak_count_check_2026-09-01"


def rho(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if good.sum() < 8:
        return np.nan
    return float(stats.spearmanr(values[good, 0], values[good, 1]).statistic)


def worker(repeat: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for band_index, display_band in enumerate(researcher.BANDS):
        band = display_band.lower()
        peak_range = researcher.BANDS[display_band][2]
        train_rng = np.random.default_rng(six.SEED + repeat * 100 + band_index * 10)
        held_rng = np.random.default_rng(six.SEED + repeat * 100 + band_index * 10 + 1)
        train_o, train_f, _ = fixed.generate_batch(train_rng, display_band, six.N_TRAIN)
        held_o, held_f, _ = fixed.generate_batch(held_rng, display_band, six.N_HELDOUT)

        for split_name, original, fractal in [
            ("train", train_o, train_f),
            ("heldout", held_o, held_f),
        ]:
            current = six.fit_fooof_batch(original, peak_range)
            official_one = official.official_fooof(original, display_band, max_n_peaks=1)
            official_three = official.official_fooof(original, display_band, max_n_peaks=3)
            split = six.fit_split_batch(original, fractal, band)
            rows.append(
                {
                    "repeat": repeat,
                    "band": band,
                    "sample": split_name,
                    "custom_sequential_onepeak_rho": rho(current),
                    "official_max1_rho": rho(official_one),
                    "official_max3_rho": rho(official_three),
                    "split_rho": rho(split),
                    "official_max1_valid": float(np.isfinite(official_one).all(axis=1).mean()),
                    "official_max3_valid": float(np.isfinite(official_three).all(axis=1).mean()),
                    "n_trials": len(original),
                }
            )
    return rows


def summarise(groups: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "custom_sequential_onepeak_rho",
        "official_max1_rho",
        "official_max3_rho",
        "split_rho",
    ]
    rows: list[dict[str, float | int | str]] = []
    for (band, sample), data in groups.groupby(["band", "sample"], sort=False):
        row: dict[str, float | int | str] = {
            "band": band,
            "sample": sample,
            "n_groups": len(data),
            "trials_per_group": int(data.n_trials.iloc[0]),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(data[metric].mean())
            row[f"{metric}_q025"] = float(data[metric].quantile(0.025))
            row[f"{metric}_q975"] = float(data[metric].quantile(0.975))
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    knee_metrics = knee2.replay_ou_at_knee2()
    matched.install_variant()
    official._fooof_group_class()
    with mp.get_context("fork").Pool(4) as pool:
        nested = pool.map(worker, range(six.N_GROUPS))
    groups = pd.DataFrame([row for part in nested for row in part])
    summary = summarise(groups)
    groups.to_csv(OUT / "same_data_group_results.csv", index=False)
    summary.to_csv(OUT / "same_data_summary.csv", index=False)
    result = {
        "question": "Does the knee=2 alpha discrepancy persist on identical spectra?",
        "controlled_factors": "same OU innovations/base, same group/trial seeds, same spectra, same Welch, same bands",
        "changed_factor": "FOOOF peak model only: custom sequential one peak vs official max_n_peaks=1/3",
        "knee_replay": knee_metrics,
        "summary": summary.to_dict(orient="records"),
    }
    (OUT / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
