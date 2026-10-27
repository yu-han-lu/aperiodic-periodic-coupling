from __future__ import annotations

"""Rebuild the unified A--F figure after moving the OU knee from 5 to 2 Hz.

The innovation sequence is recovered from the provided 5-Hz OU realization
and replayed through the Euler OU recursion at 2 Hz.  This changes only the
aperiodic time constant while matching the underlying stochastic innovations.
All downstream trial seeds, oscillation parameters, Welch settings, sample
sizes, estimators, panel definitions, and plotting code are inherited from the
final 5-Hz six-panel implementation.
"""

import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

import fooof_split_attached_code_theory as base
import fooof_split_researcher_e_gamma_fix as fixed
import fooof_split_researcher_ou_six_panel as six
import fooof_split_researcher_pipeline_matched as matched
import reproduce_gamma_artifact as researcher


ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "output" / "fooof_split_researcher_ou_knee2_six_panel_2026-09-01"
TRUE_KNEE_HZ = 2.0
SOURCE_KNEE_HZ = 5.0


def replay_ou_at_knee2() -> dict[str, float]:
    """Load residual library, then replace the base OU signal with a 2-Hz replay."""
    fixed.load_sources()
    assert fixed.BASE_SIGNAL is not None
    source = np.asarray(fixed.BASE_SIGNAL, dtype=float)
    dt = 1.0 / researcher.FS
    source_tau = 1.0 / (2.0 * np.pi * SOURCE_KNEE_HZ)
    target_tau = 1.0 / (2.0 * np.pi * TRUE_KNEE_HZ)
    source_coefficient = 1.0 - dt / source_tau
    target_coefficient = 1.0 - dt / target_tau

    innovations = (source[1:] - source_coefficient * source[:-1]) / np.sqrt(dt)
    target_tail = signal.lfilter(
        [np.sqrt(dt)], [1.0, -target_coefficient], innovations
    )
    target = np.r_[source[0], target_tail]
    frequency, psd = signal.welch(
        target,
        fs=researcher.FS,
        window="hamming",
        nperseg=len(target),
        noverlap=len(target) // 2,
        nfft=32768,
        detrend="constant",
    )
    fixed.BASE_SIGNAL = target
    fixed.BASE_FREQ = frequency
    fixed.BASE_PSD = psd
    fixed.BASE_LOG_PSD = np.interp(base.FREQ, frequency, np.log10(psd))

    ap = researcher.fit_ap_knee(fixed.BASE_LOG_PSD, base.FREQ)
    fitted_knee_hz = float(10.0 ** (ap[2] / ap[1]))
    return {
        "source_knee_hz": SOURCE_KNEE_HZ,
        "true_target_knee_hz": TRUE_KNEE_HZ,
        "base_psd_fitted_knee_hz": fitted_knee_hz,
        "reused_innovation_correlation": 1.0,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    knee_metrics = replay_ou_at_knee2()
    matched.install_variant()
    six.OUT = OUT

    with mp.get_context("fork").Pool(4) as pool:
        nested = pool.map(six.worker, range(six.N_GROUPS))
    groups = pd.DataFrame([row for part in nested for row in part])
    selected_repeat = six.choose_c_group(groups)
    representative = six.reconstruct_group(selected_repeat, "High gamma")
    original_svg, metrics = six.make_figure(groups, representative)

    explicit_svg = OUT / "figure2_fooof_split_ou_welch_knee2_six_panels.svg"
    explicit_png = OUT / "figure2_fooof_split_ou_welch_knee2_six_panels_preview.png"
    original_png = OUT / "figure2_fooof_split_ou_welch_six_panels_preview.png"
    original_svg.replace(explicit_svg)
    original_png.replace(explicit_png)
    metrics["figure"] = str(explicit_svg)
    metrics["preview"] = str(explicit_png)
    metrics["knee_replay"] = knee_metrics
    metrics["all_other_conditions_match_knee5_figure"] = True

    groups.to_csv(OUT / "all_ou_welch_knee2_group_results.csv", index=False)
    six.aggregate(groups).to_csv(OUT / "cross_band_knee2_summary.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
