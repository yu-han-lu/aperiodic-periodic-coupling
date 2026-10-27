from __future__ import annotations

"""Compare active-model Jacobian anchors for the corrected official-FOOOF run."""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import fooof_split_researcher_e_gamma_fix as fixed
import fooof_split_researcher_ou_six_panel as six
import fooof_split_knee2_corrected_official3_six_panel as corrected
import reproduce_gamma_artifact as researcher


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "knee2_official3_reference_jacobian_check_2026-09-01"
EMPIRICAL = (
    ROOT
    / "output"
    / "fooof_split_knee2_corrected_official3_six_panel_2026-09-01"
    / "all_group_results.csv"
)


def worker(repeat: int) -> list[dict[str, float | int | str]]:
    corrected.install_independent_ou(repeat)
    rows = []
    for band_index, display_band in enumerate(researcher.BANDS):
        rng = np.random.default_rng(six.SEED + repeat * 100 + band_index * 10)
        original, _, true_center = fixed.generate_batch(rng, display_band, six.N_TRAIN)
        mid = np.mean(researcher.BANDS[display_band][:2])
        reference_index = int(np.argmin(np.abs(true_center - mid)))
        anchors = {
            "mean_spectrum": original.mean(axis=0),
            "median_spectrum": np.median(original, axis=0),
            "midcenter_trial": original[reference_index],
        }
        for anchor, spectrum in anchors.items():
            influence = corrected.official_multipeak_influence(spectrum, display_band)
            projected = (original - original.mean(axis=0)) @ influence.T
            rows.append(
                {
                    "repeat": repeat,
                    "band": display_band.lower(),
                    "anchor": anchor,
                    "pred": six.corr(projected),
                }
            )
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fixed.load_sources()
    corrected.matched.install_variant()
    corrected.official._fooof_group_class()
    with mp.get_context("fork").Pool(4) as pool:
        nested = pool.map(worker, range(six.N_GROUPS))
    prediction = pd.DataFrame([row for part in nested for row in part])
    empirical = pd.read_csv(EMPIRICAL)[["repeat", "band", "fooof_emp"]]
    joined = prediction.merge(empirical, on=["repeat", "band"])
    rows = []
    for anchor, data in joined.groupby("anchor"):
        good = np.isfinite(data.pred) & np.isfinite(data.fooof_emp)
        fit = stats.linregress(data.loc[good, "pred"], data.loc[good, "fooof_emp"])
        rows.append(
            {
                "anchor": anchor,
                "valid_n": int(good.sum()),
                "pearson_r": float(fit.rvalue),
                "slope": float(fit.slope),
                "r2": float(fit.rvalue**2),
            }
        )
    result = pd.DataFrame(rows)
    joined.to_csv(OUT / "group_predictions.csv", index=False)
    result.to_csv(OUT / "summary.csv", index=False)
    print(json.dumps(result.to_dict(orient="records"), indent=2))


if __name__ == "__main__":
    main()
