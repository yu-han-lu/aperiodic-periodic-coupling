from __future__ import annotations

"""Full 3x3 FOOOF versus IRASA--SPLIT figure with direct covariance theory.

Every calibration spectrum comes from an independently sampled, physically
generated s(t).  The sample covariance of those real Welch/IRASA spectra enters
the local Jacobian theory directly; no post-IRASA Gaussian spectrum mixing is
used.  SPLIT Jacobians are evaluated at a valid real local peak nearest the
centre of each input band, with a strict 105 +/- 5 Hz target for high gamma.
"""

import json
import multiprocessing as mp
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd
from scipy import stats


WORKSPACE = Path(__file__).resolve().parent
PROJECT_CODE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project/experiments/pap_theory_validation/pap_validation"
for source_path in [WORKSPACE, PROJECT_CODE]:
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

import redraw_fooof_irasa_split_3x3 as shared  # noqa: E402


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_direct_covariance_3x3_2026-09-13"
)
STEM = "fooof_irasa_split_direct_covariance_nature_neuroscience_3x3"
N_GROUPS = 20
N_TRAIN = 120
N_HELDOUT = 120
BASE_SEED = shared.frozen.BASE_SEED + 4_000_000
SUPPLEMENT_OFFSET = 50_000_000
HIGH_GAMMA_TOLERANCE_HZ = 5.0
SUPPLEMENT_BATCH = 48
MAX_SUPPLEMENT_BATCHES = 10


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


def band_target(display_band: str) -> float:
    low, high, _ = shared.researcher.BANDS[display_band]
    return float((low + high) / 2.0)


def choose_split_reference(
    original: np.ndarray,
    fractal: np.ndarray,
    centers: np.ndarray,
    display_band: str,
    supplement_seed: int,
    fitted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool], np.ndarray]:
    """Choose a valid local SPLIT peak nearest the band midpoint."""
    band = display_band.lower()
    target = band_target(display_band)
    valid = np.isfinite(fitted).all(axis=1)
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        nearest = None
    else:
        nearest = int(
            valid_indices[np.argmin(np.abs(centers[valid_indices] - target))]
        )

    needs_supplement = (
        display_band == "High gamma"
        and (
            nearest is None
            or abs(float(centers[nearest]) - target) > HIGH_GAMMA_TOLERANCE_HZ
        )
    )
    if nearest is not None and not needs_supplement:
        return (
            original[nearest],
            fractal[nearest],
            {
                "reference_from_core": True,
                "reference_index": nearest,
                "reference_true_center_hz": float(centers[nearest]),
                "reference_distance_hz": float(abs(centers[nearest] - target)),
                "supplement_generated": 0,
            },
            fitted,
        )

    if display_band != "High gamma":
        raise RuntimeError(f"No valid SPLIT reference in {display_band}")

    supplement_rng = np.random.default_rng(supplement_seed)
    generated = 0
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    for _ in range(MAX_SUPPLEMENT_BATCHES):
        extra_o, extra_f, extra_c, _ = shared.shared_signal_batch(
            supplement_rng, display_band, SUPPLEMENT_BATCH
        )
        generated += SUPPLEMENT_BATCH
        extra_fit = shared.six.fit_split_batch(extra_o, extra_f, band)
        extra_valid = np.flatnonzero(np.isfinite(extra_fit).all(axis=1))
        for index in extra_valid:
            distance = abs(float(extra_c[index]) - target)
            if best is None or distance < best[0]:
                best = (distance, extra_o[index], extra_f[index], float(extra_c[index]))
        if best is not None and best[0] <= HIGH_GAMMA_TOLERANCE_HZ:
            break
    if best is None or best[0] > HIGH_GAMMA_TOLERANCE_HZ:
        raise RuntimeError("Could not find a valid SPLIT reference within 105 +/- 5 Hz")
    return (
        best[1],
        best[2],
        {
            "reference_from_core": False,
            "reference_index": -1,
            "reference_true_center_hz": best[3],
            "reference_distance_hz": best[0],
            "supplement_generated": generated,
        },
        fitted,
    )


def direct_projection(
    original: np.ndarray,
    fractal: np.ndarray,
    display_band: str,
    centers: np.ndarray,
    supplement_seed: int,
    fooof_fitted: np.ndarray,
    split_fitted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int | bool], np.ndarray]:
    """Compute FOOOF/SPLIT first-order projections from real sample covariance."""
    fooof_influence = shared.frozen.local_influence(
        original, centers, display_band
    )
    reference_o, reference_f, reference, split_fitted = choose_split_reference(
        original,
        fractal,
        centers,
        display_band,
        supplement_seed,
        split_fitted,
    )
    split_influence = np.asarray(
        shared.six.split_weights(
            reference_o[None, :], reference_f[None, :], display_band.lower()
        )
    )

    x_original = original - original.mean(axis=0, keepdims=True)
    residual = original - fractal
    x_residual = residual - residual.mean(axis=0, keepdims=True)
    x_fractal = fractal - fractal.mean(axis=0, keepdims=True)
    fooof_projected = x_original @ fooof_influence.T
    split_projected = np.column_stack(
        [
            x_residual @ split_influence[0],
            x_fractal @ split_influence[1],
        ]
    )

    valid_split = np.isfinite(split_fitted).all(axis=1)
    valid_fooof = np.isfinite(fooof_fitted).all(axis=1)
    reference.update(
        {
            "target_hz": band_target(display_band),
            "train_split_valid_n": int(valid_split.sum()),
            "train_fooof_valid_n": int(valid_fooof.sum()),
            "split_theory_all_rho": shared.six.corr(split_projected),
            "fooof_theory_all_rho": shared.six.corr(fooof_projected),
            "split_theory_valid_conditioned_rho": (
                shared.six.corr(split_projected[valid_split])
                if valid_split.sum() >= 8
                else np.nan
            ),
            "fooof_theory_valid_conditioned_rho": (
                shared.six.corr(fooof_projected[valid_fooof])
                if valid_fooof.sum() >= 8
                else np.nan
            ),
        }
    )
    return (
        fooof_projected,
        split_projected,
        fooof_influence,
        reference,
        split_influence,
    )


def worker(repeat: int) -> list[dict[str, float | int | str | bool]]:
    rows: list[dict[str, float | int | str | bool]] = []
    for band_index, display_band in enumerate(shared.researcher.BANDS):
        band = display_band.lower()
        train_seed = BASE_SEED + repeat * 100 + band_index * 10
        held_seed = train_seed + 1
        supplement_seed = (
            BASE_SEED + SUPPLEMENT_OFFSET + repeat * 100 + band_index * 10
        )
        train_o, train_f, train_c, train_k = shared.shared_signal_batch(
            np.random.default_rng(train_seed), display_band, N_TRAIN
        )
        held_o, held_f, held_c, held_k = shared.shared_signal_batch(
            np.random.default_rng(held_seed), display_band, N_HELDOUT
        )
        fooof_train = shared.frozen.official_fit(train_o, display_band)
        split_train = shared.six.fit_split_batch(train_o, train_f, band)
        (
            fooof_projected,
            split_projected,
            _,
            reference,
            _,
        ) = direct_projection(
            train_o,
            train_f,
            display_band,
            train_c,
            supplement_seed,
            fooof_train,
            split_train,
        )
        fooof_held = shared.frozen.official_fit(held_o, display_band)
        split_held = shared.six.fit_split_batch(held_o, held_f, band)
        fooof_prediction = shared.six.corr(fooof_projected)
        rows.append(
            {
                "scenario": "empirical",
                "repeat": repeat,
                "band": band,
                "fooof_pred": fooof_prediction,
                "fooof_emp": shared.six.corr(fooof_held),
                "fooof_corrected": shared.six.corrected_corr(
                    fooof_held, fooof_prediction
                ),
                "split_pred": shared.six.corr(split_projected),
                "split_pred_valid_conditioned": reference[
                    "split_theory_valid_conditioned_rho"
                ],
                "fooof_pred_valid_conditioned": reference[
                    "fooof_theory_valid_conditioned_rho"
                ],
                "split_emp": shared.six.corr(split_held),
                "fooof_train_valid_n": int(
                    np.isfinite(fooof_train).all(axis=1).sum()
                ),
                "split_train_valid_n": int(
                    np.isfinite(split_train).all(axis=1).sum()
                ),
                "fooof_held_valid_n": int(
                    np.isfinite(fooof_held).all(axis=1).sum()
                ),
                "split_held_valid_n": int(
                    np.isfinite(split_held).all(axis=1).sum()
                ),
                "fooof_valid": float(
                    np.isfinite(fooof_held).all(axis=1).mean()
                ),
                "split_valid": float(
                    np.isfinite(split_held).all(axis=1).mean()
                ),
                "reference_true_center_hz": reference[
                    "reference_true_center_hz"
                ],
                "reference_distance_hz": reference["reference_distance_hz"],
                "reference_from_core": reference["reference_from_core"],
                "reference_supplement_generated": reference[
                    "supplement_generated"
                ],
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
                "n_train": N_TRAIN,
                "n_heldout": N_HELDOUT,
                "n_mc": 0,
            }
        )
    return rows


def reconstruct_group(repeat: int, display_band: str) -> dict[str, object]:
    band_index = list(shared.researcher.BANDS).index(display_band)
    band = display_band.lower()
    train_seed = BASE_SEED + repeat * 100 + band_index * 10
    held_seed = train_seed + 1
    supplement_seed = (
        BASE_SEED + SUPPLEMENT_OFFSET + repeat * 100 + band_index * 10
    )
    train_o, train_f, train_c, train_k = shared.shared_signal_batch(
        np.random.default_rng(train_seed), display_band, N_TRAIN
    )
    held_o, held_f, held_c, held_k = shared.shared_signal_batch(
        np.random.default_rng(held_seed), display_band, N_HELDOUT
    )
    fooof_train = shared.frozen.official_fit(train_o, display_band)
    split_train = shared.six.fit_split_batch(train_o, train_f, band)
    (
        fooof_projected,
        split_projected,
        fooof_influence,
        reference,
        split_influence,
    ) = direct_projection(
        train_o,
        train_f,
        display_band,
        train_c,
        supplement_seed,
        fooof_train,
        split_train,
    )
    fooof = shared.frozen.official_fit(held_o, display_band)
    split = shared.six.fit_split_batch(held_o, held_f, band)
    both_good = np.isfinite(fooof).all(axis=1) & np.isfinite(split).all(axis=1)
    if not np.any(both_good):
        raise RuntimeError("No held-out condition was valid for both estimators")
    target = np.nanmedian(fooof[both_good, 0])
    valid_indices = np.flatnonzero(both_good)
    component_index = int(
        valid_indices[np.argmin(np.abs(fooof[both_good, 0] - target))]
    )
    components = shared.frozen.official_fit_components(held_o[component_index])
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
        "influence": fooof_influence,
        "split_influence": split_influence,
        "split_reference": reference,
        "fooof_projected": fooof_projected,
        "split_projected": split_projected,
        "fooof": fooof,
        "split": split,
        "component_trial_index": component_index,
        "fooof_component_frequency": components["frequency"],
        "fooof_source_psd": components["source"],
        "fooof_aperiodic_fit": components["aperiodic"],
        "fooof_periodic_fit": components["periodic"],
        "v_fooof": np.cov(fooof_projected.T, ddof=1),
        "v_split": np.cov(split_projected.T, ddof=1),
    }


def detection_summary(groups: pd.DataFrame) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for band in ["alpha", "beta", "low gamma", "high gamma"]:
        subset = groups[groups.band == band]
        for split_name, suffix, n_per_group in [
            ("calibration", "train", N_TRAIN),
            ("held-out", "held", N_HELDOUT),
        ]:
            total = len(subset) * n_per_group
            for method in ["fooof", "split"]:
                valid = int(subset[f"{method}_{suffix}_valid_n"].sum())
                rows.append(
                    {
                        "band": band,
                        "split": split_name,
                        "method": method.upper() if method == "fooof" else "IRASA-SPLIT",
                        "valid_n": valid,
                        "total_n": total,
                        "detection_rate": valid / total,
                    }
                )
    return rows


def render(groups: pd.DataFrame) -> tuple[Path, dict[str, object]]:
    representative_repeat = shared.frozen.choose_representative(groups)
    representative = reconstruct_group(representative_repeat, "High gamma")
    shared.six.OUT = OUTPUT

    captured: dict[str, plt.Figure] = {}
    original_close = shared.six.plt.close
    original_split_weights = shared.six.split_weights

    def capture_close(figure: plt.Figure | None = None) -> None:
        captured["figure"] = plt.gcf() if figure is None else figure

    def fixed_split_weights(
        original: np.ndarray, fractal: np.ndarray, band: str
    ) -> tuple[np.ndarray, np.ndarray]:
        influence = np.asarray(representative["split_influence"])
        return influence[0], influence[1]

    shared.six.plt.close = capture_close
    shared.six.split_weights = fixed_split_weights
    try:
        _, metrics = shared.six.make_figure(
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
        shared.six.plt.close = original_close
        shared.six.split_weights = original_split_weights

    figure = captured["figure"]
    # Preserve the established 3x3 layout while adding a small publication-safe
    # outer inset.  The source template placed the top panel label and the
    # right-column 150-Hz tick fractionally beyond the PDF media box.
    source_left, source_right = 0.075, 0.992
    source_bottom, source_top = 0.065, 0.978
    target_left, target_right = 0.075, 0.970
    target_bottom, target_top = 0.065, 0.960
    for axis in figure.axes:
        position = axis.get_position()
        x0 = target_left + (position.x0 - source_left) * (
            (target_right - target_left) / (source_right - source_left)
        )
        x1 = target_left + (position.x1 - source_left) * (
            (target_right - target_left) / (source_right - source_left)
        )
        y0 = target_bottom + (position.y0 - source_bottom) * (
            (target_top - target_bottom) / (source_top - source_bottom)
        )
        y1 = target_bottom + (position.y1 - source_bottom) * (
            (target_top - target_bottom) / (source_top - source_bottom)
        )
        axis.set_position([x0, y0, x1 - x0, y1 - y0])
    for artist in figure.findobj(match=lambda obj: isinstance(obj, Text)):
        if artist.get_text() == "FOOOF theory":
            artist.set_text("FOOOF covariance theory")
        elif artist.get_text() == "SPLIT theory":
            artist.set_text("SPLIT covariance theory")

    # Keep both closely spaced high-frequency tick labels readable in panel a.
    figure.canvas.draw()
    for tick_label in figure.axes[0].get_xticklabels():
        if tick_label.get_text() in {"100", "100.0"}:
            tick_label.set_horizontalalignment("right")
        elif tick_label.get_text() in {"200", "200.0"}:
            tick_label.set_horizontalalignment("left")

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
            json_out=OUTPUT / f"{STEM}.alignment.json",
            overlay_svg=OUTPUT / f"{STEM}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    except Exception as exc:
        alignment_error = repr(exc)
    original_close(figure)
    metrics.update(
        {
            "figure": str(svg),
            "pdf": str(pdf),
            "preview": str(png),
            "tiff": str(tiff),
            "theory": "local Jacobian plus direct sample covariance of independently generated calibration Welch/IRASA spectra; no Gaussian spectral mixing",
            "displayed_theory_population": "all real calibration inputs",
            "representative_split_reference": representative["split_reference"],
            "alignment_error": alignment_error,
        }
    )
    return svg, metrics


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shared.fixed.load_sources()
    shared.matched.install_variant()
    shared.frozen.official._fooof_group_class()
    shared.frozen.load_empirical_knees()
    rows: list[dict[str, float | int | str | bool]] = []
    with mp.get_context("fork").Pool(4) as pool:
        for completed, result in enumerate(
            pool.imap_unordered(worker, range(N_GROUPS)), start=1
        ):
            rows.extend(result)
            pd.DataFrame(rows).sort_values(["repeat", "band"]).to_csv(
                OUTPUT / "all_group_results_checkpoint.csv", index=False
            )
            print(f"completed repeats: {completed}/{N_GROUPS}", flush=True)
    groups = pd.DataFrame(rows).sort_values(["repeat", "band"]).reset_index(
        drop=True
    )
    groups.to_csv(OUTPUT / "all_group_results.csv", index=False)
    shared.six.aggregate(groups).to_csv(
        OUTPUT / "cross_band_summary.csv", index=False
    )
    detection = detection_summary(groups)
    pd.DataFrame(detection).to_csv(OUTPUT / "detection_summary.csv", index=False)
    svg, metrics = render(groups)
    metrics["detection_summary"] = detection
    metrics["n_groups"] = N_GROUPS
    metrics["n_calibration_per_group_band"] = N_TRAIN
    metrics["n_heldout_per_group_band"] = N_HELDOUT
    (OUTPUT / f"{STEM}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(svg)


if __name__ == "__main__":
    main()
