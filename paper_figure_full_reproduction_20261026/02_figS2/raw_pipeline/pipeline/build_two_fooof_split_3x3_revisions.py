from __future__ import annotations

"""Build two requested 3x3 FOOOF versus IRASA--SPLIT revisions."""

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

import redraw_fooof_irasa_split_3x3 as gaussian_old  # noqa: E402
import redraw_fooof_irasa_split_direct_covariance_3x3 as direct_old  # noqa: E402
import scan_high_gamma_peak_strength_gaussian_mixing as scan  # noqa: E402


OLD_GROUPS = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_shared_signal_3x3_2026-09-12/all_group_results.csv"
)
DIRECT_GROUPS = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_direct_covariance_3x3_2026-09-13/all_group_results.csv"
)
STRENGTH_OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_high_gamma_strength_28_34_gaussian_3x3_2026-09-13"
)
HYBRID_OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_hybrid_gaussian_fooof_direct_split_3x3_2026-09-13"
)
STRENGTH_STEM = "fooof_split_high_gamma_strength_28_34_gaussian_3x3"
HYBRID_STEM = "fooof_gaussian_split_direct_old_sampling_3x3"
DISPLAY_BAND = "High gamma"
BAND = "high gamma"
STRENGTH_LOW = 28.0
STRENGTH_HIGH = 34.0
N_GROUPS = gaussian_old.frozen.N_GROUPS
N_TRAIN = gaussian_old.frozen.N_TRAIN
N_HELDOUT = gaussian_old.frozen.N_HELDOUT
N_MC = gaussian_old.frozen.N_MC
BASE_SEED = gaussian_old.frozen.BASE_SEED + 4_000_000
HIGH_GAMMA_INDEX = list(gaussian_old.researcher.BANDS).index(DISPLAY_BAND)


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


def regression_influence(design: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Low-rank ridge regression derivative on the Gaussian sample cloud."""
    x = design - design.mean(axis=0, keepdims=True)
    y = target - target.mean()
    gram = x @ x.T
    ridge = max(float(np.trace(gram) / len(gram)) * 1e-8, 1e-12)
    coefficients = np.linalg.solve(gram + ridge * np.eye(len(gram)), y)
    return x.T @ coefficients


def high_gamma_worker(repeat: int) -> dict[str, float | int | str]:
    seed = BASE_SEED + repeat * 100 + HIGH_GAMMA_INDEX * 10
    train_o, train_f, train_c, train_k, _ = scan.high_gamma_batch(
        np.random.default_rng(seed), N_TRAIN, STRENGTH_LOW, STRENGTH_HIGH
    )
    held_o, held_f, held_c, held_k, _ = scan.high_gamma_batch(
        np.random.default_rng(seed + 1),
        N_HELDOUT,
        STRENGTH_LOW,
        STRENGTH_HIGH,
    )
    mc_o, mc_f = gaussian_old.frozen.covariance_mc_spectra(
        np.random.default_rng(seed + 2), train_o, train_f, n_mc=N_MC
    )
    fooof_mc = gaussian_old.frozen.official_fit(mc_o, DISPLAY_BAND)
    split_mc = gaussian_old.six.fit_split_batch(mc_o, mc_f, BAND)
    fooof_held = gaussian_old.frozen.official_fit(held_o, DISPLAY_BAND)
    split_held = gaussian_old.six.fit_split_batch(held_o, held_f, BAND)
    valid = {
        "fooof_mc": np.isfinite(fooof_mc).all(axis=1),
        "split_mc": np.isfinite(split_mc).all(axis=1),
        "fooof_held": np.isfinite(fooof_held).all(axis=1),
        "split_held": np.isfinite(split_held).all(axis=1),
    }
    fooof_prediction = gaussian_old.six.corr(fooof_mc)
    return {
        "scenario": "empirical_high_gamma_strength_28_34",
        "repeat": repeat,
        "band": BAND,
        "fooof_pred": fooof_prediction,
        "fooof_emp": gaussian_old.six.corr(fooof_held),
        "fooof_corrected": gaussian_old.six.corrected_corr(
            fooof_held, fooof_prediction
        ),
        "split_pred": gaussian_old.six.corr(split_mc),
        "split_emp": gaussian_old.six.corr(split_held),
        "fooof_valid": float(valid["fooof_held"].mean()),
        "split_valid": float(valid["split_held"].mean()),
        "fooof_mc_valid_n": int(valid["fooof_mc"].sum()),
        "split_mc_valid_n": int(valid["split_mc"].sum()),
        "fooof_held_valid_n": int(valid["fooof_held"].sum()),
        "split_held_valid_n": int(valid["split_held"].sum()),
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
        "n_train": N_TRAIN,
        "n_heldout": N_HELDOUT,
        "n_mc": N_MC,
    }


def reconstruct_strength_group(repeat: int) -> dict[str, object]:
    seed = BASE_SEED + repeat * 100 + HIGH_GAMMA_INDEX * 10
    train_o, train_f, train_c, train_k, train_strength = scan.high_gamma_batch(
        np.random.default_rng(seed), N_TRAIN, STRENGTH_LOW, STRENGTH_HIGH
    )
    held_o, held_f, held_c, held_k, held_strength = scan.high_gamma_batch(
        np.random.default_rng(seed + 1),
        N_HELDOUT,
        STRENGTH_LOW,
        STRENGTH_HIGH,
    )
    mc_o, mc_f = gaussian_old.frozen.covariance_mc_spectra(
        np.random.default_rng(seed + 2), train_o, train_f, n_mc=N_MC
    )
    fooof_mc = gaussian_old.frozen.official_fit(mc_o, DISPLAY_BAND)
    split_mc = gaussian_old.six.fit_split_batch(mc_o, mc_f, BAND)
    fooof = gaussian_old.frozen.official_fit(held_o, DISPLAY_BAND)
    split = gaussian_old.six.fit_split_batch(held_o, held_f, BAND)
    fooof_mc_good = np.isfinite(fooof_mc).all(axis=1)
    split_mc_good = np.isfinite(split_mc).all(axis=1)
    if fooof_mc_good.sum() < 8 or split_mc_good.sum() < 8:
        raise RuntimeError("Insufficient valid Gaussian samples for representative")

    # Panels b1--b4 are the local mechanism view, whereas panels d/e use the
    # full estimator-level Gaussian propagation.  Use a detected real local
    # peak for the mechanism view so finite-sample high-dimensional regression
    # noise is not mistaken for a physiological frequency response.
    fooof_influence = gaussian_old.frozen.local_influence(
        train_o, train_c, DISPLAY_BAND
    )
    split_train = gaussian_old.six.fit_split_batch(train_o, train_f, BAND)
    reference_o, reference_f, _, _ = direct_old.choose_split_reference(
        train_o,
        train_f,
        train_c,
        DISPLAY_BAND,
        BASE_SEED + 90_000_000 + repeat,
        split_train,
    )
    split_influence = np.asarray(
        gaussian_old.six.split_weights(
            reference_o[None, :], reference_f[None, :], BAND
        )
    )

    both_good = np.isfinite(fooof).all(axis=1) & np.isfinite(split).all(axis=1)
    if not np.any(both_good):
        raise RuntimeError("No held-out sample valid for both estimators")
    target = np.nanmedian(fooof[both_good, 0])
    valid_indices = np.flatnonzero(both_good)
    component_index = int(
        valid_indices[np.argmin(np.abs(fooof[both_good, 0] - target))]
    )
    components = gaussian_old.frozen.official_fit_components(
        held_o[component_index]
    )
    return {
        "repeat": repeat,
        "band": BAND,
        "train_o": train_o,
        "train_f": train_f,
        "train_c": train_c,
        "train_knee": train_k,
        "train_strength": train_strength,
        "held_o": held_o,
        "held_f": held_f,
        "held_c": held_c,
        "held_knee": held_k,
        "held_strength": held_strength,
        "influence": fooof_influence,
        "split_influence": split_influence,
        "fooof_projected": fooof_mc,
        "split_projected": split_mc,
        "fooof": fooof,
        "split": split,
        "component_trial_index": component_index,
        "fooof_component_frequency": components["frequency"],
        "fooof_source_psd": components["source"],
        "fooof_aperiodic_fit": components["aperiodic"],
        "fooof_periodic_fit": components["periodic"],
        "v_fooof": np.cov(fooof_mc[fooof_mc_good].T, ddof=1),
        "v_split": np.cov(split_mc[split_mc_good].T, ddof=1),
    }


def hybrid_groups() -> pd.DataFrame:
    gaussian = pd.read_csv(OLD_GROUPS).sort_values(["repeat", "band"])
    direct = pd.read_csv(DIRECT_GROUPS).sort_values(["repeat", "band"])
    merged = gaussian.copy()
    for column in ["split_pred", "split_emp", "split_valid"]:
        merged[column] = direct[column].to_numpy()
    merged["scenario"] = "fooof_gaussian_split_direct"
    return merged.reset_index(drop=True)


def reconstruct_hybrid_group(repeat: int) -> dict[str, object]:
    gaussian = gaussian_old.reconstruct_group(repeat, DISPLAY_BAND)
    direct = direct_old.reconstruct_group(repeat, DISPLAY_BAND)
    representative = dict(gaussian)
    representative["split_projected"] = direct["split_projected"]
    representative["split_influence"] = direct["split_influence"]
    representative["v_split"] = direct["v_split"]
    # Held-out inputs and fits are identical by construction; retain one copy.
    if not np.allclose(
        np.asarray(gaussian["held_o"]), np.asarray(direct["held_o"])
    ):
        raise AssertionError("Hybrid branches do not share identical held-out inputs")
    return representative


def inset_template_layout(figure: plt.Figure) -> None:
    """Keep the established grid but place all outer text inside the media box."""
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
    figure.canvas.draw()
    for tick_label in figure.axes[0].get_xticklabels():
        if tick_label.get_text() in {"100", "100.0"}:
            tick_label.set_horizontalalignment("right")
        elif tick_label.get_text() in {"200", "200.0"}:
            tick_label.set_horizontalalignment("left")


def render_3x3(
    groups: pd.DataFrame,
    representative: dict[str, object],
    output: Path,
    stem: str,
    fooof_theory_label: str,
    split_theory_label: str,
    rasterize_pdf_linework: bool = False,
) -> tuple[Path, dict[str, object]]:
    output.mkdir(parents=True, exist_ok=True)
    gaussian_old.six.OUT = output
    captured: dict[str, plt.Figure] = {}
    original_close = gaussian_old.six.plt.close
    original_split_weights = gaussian_old.six.split_weights

    def capture_close(figure: plt.Figure | None = None) -> None:
        captured["figure"] = plt.gcf() if figure is None else figure

    def fixed_split_weights(
        original: np.ndarray, fractal: np.ndarray, band: str
    ) -> tuple[np.ndarray, np.ndarray]:
        influence = np.asarray(representative["split_influence"])
        return influence[0], influence[1]

    gaussian_old.six.plt.close = capture_close
    gaussian_old.six.split_weights = fixed_split_weights
    try:
        _, metrics = gaussian_old.six.make_figure(
            groups,
            representative,
            panel_f_absolute=False,
            panel_f_band=BAND,
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
            output_stem=stem,
        )
    finally:
        gaussian_old.six.plt.close = original_close
        gaussian_old.six.split_weights = original_split_weights

    figure = captured["figure"]
    for artist in figure.findobj(match=lambda obj: isinstance(obj, Text)):
        text_value = artist.get_text()
        if text_value == "FOOOF theory":
            artist.set_text(fooof_theory_label)
        elif text_value == "SPLIT theory":
            artist.set_text(split_theory_label)
        else:
            artist.set_text(
                text_value.replace(r"$\log_{10}$", "log10")
                .replace(r"$J_P$", "J_P")
                .replace(r"$J_{AP}$", "J_AP")
                .replace(r"$\rho$", "ρ")
                .replace(r"$R^2$", "R2")
            )
        if artist.get_fontsize() < 5.4:
            artist.set_fontsize(5.4)
        if "\n" in artist.get_text():
            artist.set_linespacing(1.25)
    inset_template_layout(figure)

    svg = output / f"{stem}.svg"
    pdf = output / f"{stem}.pdf"
    preview = output / f"{stem}_preview.png"
    tiff = output / f"{stem}.tiff"
    figure.canvas.draw()
    alignment_error = None
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment

        require_matplotlib_panel_alignment(
            figure,
            json_out=output / f"{stem}.alignment.json",
            overlay_svg=output / f"{stem}.alignment.svg",
            tolerance_pt=1.5,
            gutter_tolerance_pt=1.5,
            require_panel_labels=False,
            strict=True,
        )
    except Exception as exc:
        alignment_error = repr(exc)
    if alignment_error is not None:
        original_close(figure)
        raise RuntimeError(f"Panel alignment failed: {alignment_error}")
    # Keep the requested SVG fully vector.  The optional PDF-only rasterization
    # is useful for rendered-geometry QA of dense spectra whose clipped vector
    # paths are otherwise interpreted as crossing neighbouring text by generic
    # PDF collision scanners.
    figure.savefig(svg, facecolor="white")
    if rasterize_pdf_linework:
        for axis in figure.axes:
            for artist in [*axis.lines, *axis.collections, *axis.patches]:
                artist.set_rasterized(True)
            for spine in axis.spines.values():
                spine.set_rasterized(True)
            for tick in [*axis.xaxis.get_major_ticks(), *axis.yaxis.get_major_ticks()]:
                tick.tick1line.set_rasterized(True)
                tick.tick2line.set_rasterized(True)
    figure.savefig(pdf, facecolor="white")
    figure.savefig(preview, dpi=300, facecolor="white")
    figure.savefig(tiff, dpi=600, facecolor="white")
    original_close(figure)
    metrics.update(
        {
            "figure": str(svg),
            "pdf": str(pdf),
            "preview": str(preview),
            "tiff": str(tiff),
            "alignment_error": alignment_error,
        }
    )
    return svg, metrics


def run_strength_version() -> tuple[Path, dict[str, object]]:
    rows: list[dict[str, float | int | str]] = []
    with mp.get_context("fork").Pool(4) as pool:
        for completed, row in enumerate(
            pool.imap_unordered(high_gamma_worker, range(N_GROUPS)), start=1
        ):
            rows.append(row)
            pd.DataFrame(rows).sort_values("repeat").to_csv(
                STRENGTH_OUTPUT / "high_gamma_checkpoint.csv", index=False
            )
            print(f"completed high-gamma groups: {completed}/{N_GROUPS}", flush=True)
    high = pd.DataFrame(rows).sort_values("repeat").reset_index(drop=True)
    old = pd.read_csv(OLD_GROUPS)
    combined = pd.concat(
        [old[old.band != BAND], high], ignore_index=True, sort=False
    ).sort_values(["repeat", "band"]).reset_index(drop=True)
    high.to_csv(STRENGTH_OUTPUT / "high_gamma_results.csv", index=False)
    combined.to_csv(STRENGTH_OUTPUT / "combined_group_results.csv", index=False)
    gaussian_old.six.aggregate(combined).to_csv(
        STRENGTH_OUTPUT / "cross_band_summary.csv", index=False
    )
    selected_repeat = gaussian_old.frozen.choose_representative(combined)
    representative = reconstruct_strength_group(selected_repeat)
    svg, metrics = render_3x3(
        combined,
        representative,
        STRENGTH_OUTPUT,
        STRENGTH_STEM,
        "FOOOF Gaussian theory",
        "SPLIT Gaussian theory",
    )
    metrics.update(
        {
            "selected_peak_strength_distribution": [
                STRENGTH_LOW,
                STRENGTH_HIGH,
            ],
            "peak_strength_definition": (
                "oscillatory-to-local-background power ratio at the sampled "
                "center frequency"
            ),
            "selection_reason": (
                "weakest scanned interval with finite SPLIT Gaussian theory in "
                "every pilot repeat and mean Gaussian-SPLIT detection above 10%, "
                "while retaining empirical FOOOF-SPLIT separation"
            ),
            "high_gamma_detection": {
                "fooof_gaussian_mc_valid_n": int(high.fooof_mc_valid_n.sum()),
                "fooof_gaussian_mc_total_n": N_GROUPS * N_MC,
                "split_gaussian_mc_valid_n": int(high.split_mc_valid_n.sum()),
                "split_gaussian_mc_total_n": N_GROUPS * N_MC,
                "fooof_held_valid_n": int(high.fooof_held_valid_n.sum()),
                "fooof_held_total_n": N_GROUPS * N_HELDOUT,
                "split_held_valid_n": int(high.split_held_valid_n.sum()),
                "split_held_total_n": N_GROUPS * N_HELDOUT,
            },
        }
    )
    (STRENGTH_OUTPUT / f"{STRENGTH_STEM}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return svg, metrics


def run_hybrid_version() -> tuple[Path, dict[str, object]]:
    groups = hybrid_groups()
    groups.to_csv(HYBRID_OUTPUT / "combined_group_results.csv", index=False)
    gaussian_old.six.aggregate(groups).to_csv(
        HYBRID_OUTPUT / "cross_band_summary.csv", index=False
    )
    selected_repeat = gaussian_old.frozen.choose_representative(groups)
    representative = reconstruct_hybrid_group(selected_repeat)
    svg, metrics = render_3x3(
        groups,
        representative,
        HYBRID_OUTPUT,
        HYBRID_STEM,
        "FOOOF Gaussian theory",
        "SPLIT direct theory",
    )
    metrics.update(
        {
            "sampling_distribution": "frozen original peak-strength U(2.8, 3.8)",
            "fooof_theory": "Gaussian covariance propagation through the full estimator",
            "split_theory": "local Jacobian projection of real calibration covariance",
            "held_out_identity_check": "FOOOF and SPLIT use identical held-out spectra",
        }
    )
    (HYBRID_OUTPUT / f"{HYBRID_STEM}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    return svg, metrics


def main() -> None:
    STRENGTH_OUTPUT.mkdir(parents=True, exist_ok=True)
    HYBRID_OUTPUT.mkdir(parents=True, exist_ok=True)
    gaussian_old.fixed.load_sources()
    gaussian_old.matched.install_variant()
    gaussian_old.frozen.official._fooof_group_class()
    gaussian_old.frozen.load_empirical_knees()
    strength_svg, strength_metrics = run_strength_version()
    hybrid_svg, hybrid_metrics = run_hybrid_version()
    print(
        json.dumps(
            {
                "strength_optimized": strength_metrics,
                "hybrid": hybrid_metrics,
            },
            indent=2,
        )
    )
    print(strength_svg)
    print(hybrid_svg)


if __name__ == "__main__":
    main()
