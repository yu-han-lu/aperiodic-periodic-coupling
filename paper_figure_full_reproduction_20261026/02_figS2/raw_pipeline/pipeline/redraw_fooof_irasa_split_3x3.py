from __future__ import annotations

"""Redraw the frozen FOOOF/SPLIT 3x3 comparison using one shared s(t).

The FOOOF branch receives the Welch PSD of the composite time series.  The
IRASA--SPLIT branch receives the aperiodic spectrum estimated by applying
standard irregular resampling directly to that same composite time series.
All sampling, seeds, signal generation, fitting, covariance propagation and
figure-design settings are inherited from the frozen 2026-09-01 analysis.
"""

import argparse
import fractions
import json
import multiprocessing as mp
from pathlib import Path
import sys
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy import signal, stats


PROJECT_CODE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project/experiments/pap_theory_validation/pap_validation"
if str(PROJECT_CODE) not in sys.path:
    sys.path.insert(0, str(PROJECT_CODE))

import fooof_split_official3_two_knee_sixpanels as frozen  # noqa: E402
import fooof_split_researcher_e_gamma_fix as fixed  # noqa: E402
import fooof_split_researcher_ou_six_panel as six  # noqa: E402
import fooof_split_researcher_pipeline_matched as matched  # noqa: E402
import knee_position_official_fooof3_band_sweep as sweep  # noqa: E402
import reproduce_gamma_artifact as researcher  # noqa: E402


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_shared_signal_3x3_2026-09-12"
)
OLD_RESULTS = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_split_official3_two_knee_sixpanels_2026-09-01/empirical/"
    "all_group_results.csv"
)
STEM = "fooof_irasa_split_shared_signal_nature_neuroscience_3x3"
HSET = np.round(np.arange(1.1, 1.95, 0.05), 4)


mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6.2,
        "axes.labelsize": 6.4,
        "legend.fontsize": 5.3,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def composite_trial(
    rng: np.random.Generator,
    center: float,
    width: float,
    power_fraction: float,
    knee_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate the frozen OU-plus-bandpass time series and its Welch PSD."""
    innovations = rng.normal(size=sweep.N)
    background = sweep.ou_from_innovations(innovations, knee_hz)
    base_frequency, base_psd = sweep.base_welch(background)

    low = max(center - width / 2.0, 0.1)
    high = min(center + width / 2.0, researcher.FS / 2.0 - 0.1)
    sos = signal.butter(
        2, [low, high], btype="bandpass", fs=researcher.FS, output="sos"
    )
    oscillation = signal.sosfiltfilt(sos, rng.normal(size=len(background)))
    oscillation /= np.sqrt(np.mean(oscillation**2))

    _, unit_psd = signal.welch(
        background + oscillation,
        fs=researcher.FS,
        window="hamming",
        nperseg=len(background),
        noverlap=len(background) // 2,
        nfft=32768,
        detrend="constant",
    )
    local_base = float(np.interp(center, base_frequency, base_psd))
    unit_bump = float(np.interp(center, base_frequency, unit_psd) - local_base)
    amplitude = np.sqrt(
        max(power_fraction * local_base, 1e-30) / max(unit_bump, 1e-30)
    )
    composite = background + amplitude * oscillation
    frequency, psd = signal.welch(
        composite,
        fs=researcher.FS,
        window="hamming",
        nperseg=len(composite),
        noverlap=len(composite) // 2,
        nfft=32768,
        detrend="constant",
    )
    keep = (frequency >= 0.2) & (frequency <= 200.0)
    return composite, frequency[keep], np.log10(
        np.maximum(psd[keep], np.finfo(float).tiny)
    )


def irasa_aperiodic_batch(signals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the standard YASA/NeuroDSP IRASA calculation to a signal batch."""
    welch_kwargs = {
        "window": "hamming",
        "nperseg": sweep.N,
        "noverlap": sweep.N // 2,
        "nfft": 32768,
        "detrend": "constant",
        "average": "mean",
        "axis": -1,
    }
    frequency, psd = signal.welch(signals, fs=sweep.FS, **welch_kwargs)
    resampled_geometric = np.empty((len(HSET), *psd.shape), dtype=float)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="nperseg=.* is greater than signal length.*"
        )
        for index, h_value in enumerate(HSET):
            ratio = fractions.Fraction(str(float(h_value)))
            up, down = ratio.numerator, ratio.denominator
            signal_up = signal.resample_poly(signals, up, down, axis=-1)
            signal_down = signal.resample_poly(signals, down, up, axis=-1)
            _, psd_up = signal.welch(
                signal_up, fs=float(h_value) * sweep.FS, **welch_kwargs
            )
            _, psd_down = signal.welch(
                signal_down, fs=sweep.FS / float(h_value), **welch_kwargs
            )
            resampled_geometric[index] = np.sqrt(psd_up * psd_down)
    aperiodic = np.median(resampled_geometric, axis=0)
    return frequency, aperiodic


def shared_signal_batch(
    rng: np.random.Generator,
    display_band: str,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate s(t), then derive both the FOOOF and IRASA--SPLIT inputs."""
    assert fixed.FRACTAL_RESIDUALS is not None
    low, high, _ = researcher.BANDS[display_band]
    centers = rng.uniform(low, high, n_trials)
    widths = rng.uniform(1.5, 6.0, n_trials)
    fractions_power = rng.uniform(2.8, 3.8, n_trials)
    knees = frozen.sample_empirical_knees_independent_of_centers(rng, centers)

    # Preserve the frozen random-number stream exactly.  The previous analysis
    # sampled these indices before time-series generation; they are intentionally
    # consumed here even though the near-oracle residual is no longer used.
    rng.integers(0, len(fixed.FRACTAL_RESIDUALS), n_trials)

    signals: list[np.ndarray] = []
    original: list[np.ndarray] = []
    for center, width, fraction_power, knee in zip(
        centers, widths, fractions_power, knees
    ):
        composite, frequency, log_psd = composite_trial(
            rng,
            float(center),
            float(width),
            float(fraction_power),
            float(knee),
        )
        signals.append(composite)
        assert np.all(np.diff(frequency) > 0)
        original.append(np.interp(frozen.base.FREQ, frequency, log_psd))

    irasa_frequency, irasa_aperiodic = irasa_aperiodic_batch(np.asarray(signals))
    irasa_log = np.log10(np.maximum(irasa_aperiodic, np.finfo(float).tiny))
    assert np.all(np.diff(irasa_frequency) > 0)
    fractal = np.asarray(
        [np.interp(frozen.base.FREQ, irasa_frequency, row) for row in irasa_log]
    )
    return np.asarray(original), fractal, centers, knees


def worker(repeat: int) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    scenario = "empirical"
    seed = frozen.BASE_SEED + 4_000_000
    for band_index, display_band in enumerate(researcher.BANDS):
        band = display_band.lower()
        train_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10)
        held_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10 + 1)
        mc_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10 + 2)
        train_o, train_f, train_c, train_k = shared_signal_batch(
            train_rng, display_band, frozen.N_TRAIN
        )
        held_o, held_f, held_c, held_k = shared_signal_batch(
            held_rng, display_band, frozen.N_HELDOUT
        )
        mc_o, mc_f = frozen.covariance_mc_spectra(mc_rng, train_o, train_f)
        fooof_mc = frozen.official_fit(mc_o, display_band)
        split_mc = six.fit_split_batch(mc_o, mc_f, band)
        fooof_held = frozen.official_fit(held_o, display_band)
        split_held = six.fit_split_batch(held_o, held_f, band)
        fooof_prediction = six.corr(fooof_mc)
        rows.append(
            {
                "scenario": scenario,
                "repeat": repeat,
                "band": band,
                "fooof_pred": fooof_prediction,
                "fooof_emp": six.corr(fooof_held),
                "fooof_corrected": six.corrected_corr(
                    fooof_held, fooof_prediction
                ),
                "split_pred": six.corr(split_mc),
                "split_emp": six.corr(split_held),
                "fooof_valid": float(
                    np.isfinite(fooof_held).all(axis=1).mean()
                ),
                "split_valid": float(
                    np.isfinite(split_held).all(axis=1).mean()
                ),
                "true_center_knee_rho_train": float(
                    stats.spearmanr(train_c, train_k).statistic
                ),
                "true_center_knee_rho_held": float(
                    stats.spearmanr(held_c, held_k).statistic
                ),
                "train_knee_mean": float(np.mean(train_k)),
                "held_knee_mean": float(np.mean(held_k)),
                "train_knee_min": float(np.min(train_k)),
                "train_knee_max": float(np.max(train_k)),
                "fitted_base_knee_hz": np.nan,
                "n_train": frozen.N_TRAIN,
                "n_heldout": frozen.N_HELDOUT,
                "n_mc": frozen.N_MC,
            }
        )
    return rows


def reconstruct_group(repeat: int, display_band: str) -> dict[str, object]:
    band_index = list(researcher.BANDS).index(display_band)
    band = display_band.lower()
    seed = frozen.BASE_SEED + 4_000_000
    train_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10)
    held_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10 + 1)
    mc_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10 + 2)
    train_o, train_f, train_c, train_k = shared_signal_batch(
        train_rng, display_band, frozen.N_TRAIN
    )
    held_o, held_f, held_c, held_k = shared_signal_batch(
        held_rng, display_band, frozen.N_HELDOUT
    )
    mc_o, mc_f = frozen.covariance_mc_spectra(mc_rng, train_o, train_f)
    fooof_mc = frozen.official_fit(mc_o, display_band)
    split_mc = six.fit_split_batch(mc_o, mc_f, band)
    fooof_held = frozen.official_fit(held_o, display_band)
    split_held = six.fit_split_batch(held_o, held_f, band)
    influence = frozen.local_influence(train_o, train_c, display_band)
    both_good = np.isfinite(fooof_held).all(axis=1) & np.isfinite(split_held).all(
        axis=1
    )
    if not np.any(both_good):
        raise RuntimeError("No held-out trial was valid for both estimators")
    target = np.nanmedian(fooof_held[both_good, 0])
    valid_indices = np.flatnonzero(both_good)
    component_index = int(
        valid_indices[
            np.argmin(np.abs(fooof_held[both_good, 0] - target))
        ]
    )
    components = frozen.official_fit_components(held_o[component_index])
    fooof_good = np.isfinite(fooof_mc).all(axis=1)
    split_good = np.isfinite(split_mc).all(axis=1)
    return {
        "repeat": repeat,
        "band": band,
        "train_o": train_o,
        "train_f": train_f,
        "train_c": train_c,
        "train_knee": train_k,
        "held_o": held_o,
        "held_f": held_f,
        "held_c": held_c,
        "held_knee": held_k,
        "influence": influence,
        "fooof_projected": fooof_mc,
        "split_projected": split_mc,
        "fooof": fooof_held,
        "split": split_held,
        "component_trial_index": component_index,
        "fooof_component_frequency": components["frequency"],
        "fooof_source_psd": components["source"],
        "fooof_aperiodic_fit": components["aperiodic"],
        "fooof_periodic_fit": components["periodic"],
        "v_fooof": np.cov(fooof_mc[fooof_good].T, ddof=1),
        "v_split": np.cov(split_mc[split_good].T, ddof=1),
    }


def verify_frozen_fooof(groups: pd.DataFrame) -> dict[str, float]:
    """Prove that the only estimator-input change is the SPLIT fractal source."""
    old = pd.read_csv(OLD_RESULTS).sort_values(["repeat", "band"])
    new = groups.sort_values(["repeat", "band"])
    checks = {}
    for column in [
        "fooof_pred",
        "fooof_emp",
        "fooof_corrected",
        "true_center_knee_rho_train",
        "true_center_knee_rho_held",
        "train_knee_mean",
        "held_knee_mean",
    ]:
        difference = np.nanmax(
            np.abs(old[column].to_numpy() - new[column].to_numpy())
        )
        checks[column] = float(difference)
    if max(checks.values()) > 1e-10:
        raise AssertionError(f"Frozen FOOOF/input stream changed: {checks}")
    return checks


def render(groups: pd.DataFrame) -> tuple[Path, dict[str, object]]:
    representative_repeat = frozen.choose_representative(groups)
    representative = reconstruct_group(representative_repeat, "High gamma")
    six.OUT = OUTPUT

    captured: dict[str, plt.Figure] = {}
    original_close = six.plt.close

    def capture_close(figure: plt.Figure | None = None) -> None:
        captured["figure"] = plt.gcf() if figure is None else figure

    six.plt.close = capture_close
    try:
        _, metrics = six.make_figure(
            groups,
            representative,
            panel_f_absolute=False,
            panel_f_band="high gamma",
            panel_a_fractal_dashed=True,
            panel_b_frequency_xlim=(60.0, 150.0),
            panel_b_average_summary=None,
            panel_a_fooof_components=True,
            panel_d_ylim=None,
            panel_e_ylim=(-0.3, 0.3),
            panel_f_ylim=(-0.5, 0.5),
            nature_style=True,
            nature_grid_3x3=True,
            label_frequency_bands=True,
            output_stem=STEM,
        )
    finally:
        six.plt.close = original_close

    figure = captured["figure"]
    for artist in figure.findobj(match=lambda obj: isinstance(obj, Text)):
        if artist.get_text() == "FOOOF theory":
            artist.set_text("FOOOF covariance theory")
        elif artist.get_text() == "SPLIT theory":
            artist.set_text("SPLIT covariance theory")
        elif (
            "SPLIT covariance contribution" in artist.get_text()
            and "nan" in artist.get_text()
        ):
            artist.set_text("SPLIT covariance: not estimable")

    svg = OUTPUT / f"{STEM}.svg"
    pdf = OUTPUT / f"{STEM}.pdf"
    png = OUTPUT / f"{STEM}_preview.png"
    tiff = OUTPUT / f"{STEM}.tiff"
    figure.canvas.draw()
    figure.savefig(svg, facecolor="white")
    figure.savefig(pdf, facecolor="white")
    figure.savefig(png, dpi=300, facecolor="white")
    figure.savefig(tiff, dpi=600, facecolor="white")

    alignment_error = None
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            figure,
            json_out=OUTPUT / f"{STEM}.alignment-layout.json",
            overlay_svg=OUTPUT / f"{STEM}.alignment-overlay.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            # The standard parser only accepts one-letter panel tags; this
            # frozen layout deliberately uses b1--b4. Geometry remains strict.
            require_panel_labels=False,
            strict=True,
        )
    except Exception as exc:  # preserved in QA; never silently reported as pass
        alignment_error = repr(exc)
    original_close(figure)

    metrics.update(
        {
            "figure": str(svg),
            "pdf": str(pdf),
            "preview": str(png),
            "tiff": str(tiff),
            "comparison": "FOOOF versus IRASA-SPLIT end-to-end",
            "shared_input": "the identical composite OU-plus-oscillation time series s(t)",
            "irasa": {
                "implementation": "standard irregular-resampling geometric-mean median",
                "hset": HSET.tolist(),
                "welch": {
                    "fs_hz": sweep.FS,
                    "window": "hamming",
                    "nperseg": sweep.N,
                    "noverlap": sweep.N // 2,
                    "nfft": 32768,
                    "detrend": "constant",
                    "average": "mean",
                },
            },
            "alignment_error": alignment_error,
        }
    )
    return svg, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--verify-generator", action="store_true")
    parser.add_argument("--render-existing", action="store_true")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    fixed.load_sources()
    matched.install_variant()
    frozen.official._fooof_group_class()
    frozen.load_empirical_knees()

    if args.verify_generator:
        seed = frozen.BASE_SEED + 4_000_000
        rng_old = np.random.default_rng(seed)
        rng_new = np.random.default_rng(seed)
        old_o, _, old_c, old_k = frozen.variable_knee_batch(
            rng_old, "Alpha", 4, scenario="empirical"
        )
        new_o, _, new_c, new_k = shared_signal_batch(rng_new, "Alpha", 4)
        report = {
            "max_abs_original_log_psd_difference": float(
                np.max(np.abs(old_o - new_o))
            ),
            "max_abs_center_difference": float(np.max(np.abs(old_c - new_c))),
            "max_abs_knee_difference": float(np.max(np.abs(old_k - new_k))),
        }
        print(json.dumps(report, indent=2))
        return

    if args.render_existing:
        groups = pd.read_csv(OUTPUT / "all_group_results.csv")
        frozen_checks = verify_frozen_fooof(groups)
        svg, metrics = render(groups)
        metrics["frozen_fooof_max_abs_differences"] = frozen_checks
        metrics["n_groups"] = frozen.N_GROUPS
        metrics["n_train_per_group_band"] = frozen.N_TRAIN
        metrics["n_heldout_per_group_band"] = frozen.N_HELDOUT
        metrics["n_covariance_mc_per_group_band"] = frozen.N_MC
        (OUTPUT / f"{STEM}_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
        print(json.dumps(metrics, indent=2))
        print(svg)
        return

    rows: list[dict[str, float | int | str]] = []
    context = mp.get_context("fork")
    with context.Pool(args.processes) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(worker, range(frozen.N_GROUPS)), start=1
        ):
            rows.extend(result)
            checkpoint = pd.DataFrame(rows).sort_values(["repeat", "band"])
            checkpoint.to_csv(OUTPUT / "all_group_results_checkpoint.csv", index=False)
            print(
                f"completed repeats: {completed}/{frozen.N_GROUPS}",
                flush=True,
            )
    groups = pd.DataFrame(rows).sort_values(["repeat", "band"]).reset_index(
        drop=True
    )
    frozen_checks = verify_frozen_fooof(groups)
    groups.to_csv(OUTPUT / "all_group_results.csv", index=False)
    six.aggregate(groups).to_csv(OUTPUT / "cross_band_summary.csv", index=False)
    svg, metrics = render(groups)
    metrics["frozen_fooof_max_abs_differences"] = frozen_checks
    metrics["n_groups"] = frozen.N_GROUPS
    metrics["n_train_per_group_band"] = frozen.N_TRAIN
    metrics["n_heldout_per_group_band"] = frozen.N_HELDOUT
    metrics["n_covariance_mc_per_group_band"] = frozen.N_MC
    metrics_path = OUTPUT / f"{STEM}_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(svg)


if __name__ == "__main__":
    main()
