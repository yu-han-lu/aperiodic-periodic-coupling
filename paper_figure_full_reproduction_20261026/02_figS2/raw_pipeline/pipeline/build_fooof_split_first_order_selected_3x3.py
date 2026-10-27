from __future__ import annotations

"""Rebuild the selected FOOOF--IRASA/SPLIT 3x3 figure with delta theory.

The input and held-out estimator pipelines are frozen to the selected version:
alpha, beta and low gamma use the original peak-strength distribution, whereas
high gamma uses U(28, 34).  The only analysis change is that panels d--f use
local estimator Jacobians and the directly observed calibration spectral
covariance, rather than Gaussian covariance resampling through the estimators.
"""

import json
import multiprocessing as mp
from pathlib import Path
import sys

import matplotlib as mpl
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
PROJECT_CODE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project/experiments/pap_theory_validation/pap_validation"
FIGURE_SKILL_SCRIPTS = Path(
    "/PORTABLE_HOME/.codex/skills/nature-figure/scripts"
)
for source_path in (HERE, PROJECT_CODE, FIGURE_SKILL_SCRIPTS):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

import build_two_fooof_split_3x3_revisions as selected  # noqa: E402
import redraw_fooof_irasa_split_3x3 as shared  # noqa: E402
import redraw_fooof_irasa_split_direct_covariance_3x3 as delta  # noqa: E402
import scan_high_gamma_peak_strength_gaussian_mixing as scan  # noqa: E402


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_first_order_selected_3x3_2026-09-13"
)
STEM = "fooof_irasa_split_first_order_selected_nature_neuroscience_3x3"
DIRECT_GROUPS = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_direct_covariance_3x3_2026-09-13/all_group_results.csv"
)
N_GROUPS = shared.frozen.N_GROUPS
N_TRAIN = shared.frozen.N_TRAIN
N_HELDOUT = shared.frozen.N_HELDOUT
BASE_SEED = shared.frozen.BASE_SEED + 4_000_000
DISPLAY_BAND = "High gamma"
BAND = "high gamma"
HIGH_GAMMA_INDEX = list(shared.researcher.BANDS).index(DISPLAY_BAND)
STRENGTH_LOW = 28.0
STRENGTH_HIGH = 34.0

# Publication/export contract.  The inherited renderer writes the matching
# figure.svg, figure.pdf and figure.tiff assets at 600 dpi for the raster copy.
mpl.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 6.2,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
EXPORT_CONTRACT = ("figure.svg", "figure.pdf", "figure.tiff")
RASTER_DPI = 600


def _corr(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if int(good.sum()) < 8:
        return np.nan
    return float(stats.spearmanr(values[good, 0], values[good, 1]).statistic)


def _first_order_projection(
    original: np.ndarray,
    fractal: np.ndarray,
    centers: np.ndarray,
    fooof_fit: np.ndarray,
    split_fit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Return J delta-L for FOOOF and J(delta-R, delta-F) for SPLIT."""
    fooof_j = shared.frozen.local_influence(original, centers, DISPLAY_BAND)
    if not np.isfinite(fooof_j).all():
        raise RuntimeError("No valid local FOOOF active-model Jacobian")

    target = 0.5 * sum(shared.researcher.BANDS[DISPLAY_BAND][:2])
    split_good = np.isfinite(split_fit).all(axis=1)
    candidates = np.flatnonzero(split_good)
    if len(candidates) == 0:
        raise RuntimeError("No valid SPLIT peak in high-gamma calibration set")
    reference_index = int(candidates[np.argmin(np.abs(centers[candidates] - target))])
    split_j = np.asarray(
        shared.six.split_weights(
            original[reference_index][None, :],
            fractal[reference_index][None, :],
            BAND,
        )
    )
    if not np.isfinite(split_j).all():
        raise RuntimeError("Non-finite local SPLIT Jacobian")

    d_original = original - original.mean(axis=0, keepdims=True)
    residual = original - fractal
    d_residual = residual - residual.mean(axis=0, keepdims=True)
    d_fractal = fractal - fractal.mean(axis=0, keepdims=True)
    fooof_linear = d_original @ fooof_j.T
    split_linear = np.column_stack(
        (d_residual @ split_j[0], d_fractal @ split_j[1])
    )
    fooof_good = np.isfinite(fooof_fit).all(axis=1)
    diagnostics = {
        "fooof_train_valid_n": int(fooof_good.sum()),
        "split_train_valid_n": int(split_good.sum()),
        "split_reference_index": reference_index,
        "split_reference_center_hz": float(centers[reference_index]),
        "fooof_pred_all": _corr(fooof_linear),
        "split_pred_all": _corr(split_linear),
        "fooof_pred_valid_conditioned": _corr(fooof_linear[fooof_good]),
        "split_pred_valid_conditioned": _corr(split_linear[split_good]),
    }
    return fooof_linear, split_linear, fooof_j, split_j, diagnostics


def high_gamma_worker(repeat: int) -> dict[str, float | int | str]:
    seed = BASE_SEED + repeat * 100 + HIGH_GAMMA_INDEX * 10
    train_o, train_f, train_c, train_k, _ = scan.high_gamma_batch(
        np.random.default_rng(seed), N_TRAIN, STRENGTH_LOW, STRENGTH_HIGH
    )
    held_o, held_f, held_c, held_k, _ = scan.high_gamma_batch(
        np.random.default_rng(seed + 1), N_HELDOUT, STRENGTH_LOW, STRENGTH_HIGH
    )
    fooof_train = shared.frozen.official_fit(train_o, DISPLAY_BAND)
    split_train = shared.six.fit_split_batch(train_o, train_f, BAND)
    fooof_linear, split_linear, _, _, diagnostic = _first_order_projection(
        train_o, train_f, train_c, fooof_train, split_train
    )
    fooof_held = shared.frozen.official_fit(held_o, DISPLAY_BAND)
    split_held = shared.six.fit_split_batch(held_o, held_f, BAND)

    # Primary prediction is the strict first-order covariance over all real
    # calibration spectra.  Valid-conditioned values are retained separately
    # to expose the effect of non-differentiable detection/model selection.
    fooof_pred = _corr(fooof_linear)
    split_pred = _corr(split_linear)
    return {
        "scenario": "empirical_high_gamma_strength_28_34_first_order",
        "repeat": repeat,
        "band": BAND,
        "fooof_pred": fooof_pred,
        "fooof_emp": shared.six.corr(fooof_held),
        "fooof_corrected": shared.six.corrected_corr(fooof_held, fooof_pred),
        "split_pred": split_pred,
        "split_emp": shared.six.corr(split_held),
        "fooof_pred_valid_conditioned": diagnostic["fooof_pred_valid_conditioned"],
        "split_pred_valid_conditioned": diagnostic["split_pred_valid_conditioned"],
        "fooof_train_valid_n": diagnostic["fooof_train_valid_n"],
        "split_train_valid_n": diagnostic["split_train_valid_n"],
        "fooof_held_valid_n": int(np.isfinite(fooof_held).all(axis=1).sum()),
        "split_held_valid_n": int(np.isfinite(split_held).all(axis=1).sum()),
        "fooof_valid": float(np.isfinite(fooof_held).all(axis=1).mean()),
        "split_valid": float(np.isfinite(split_held).all(axis=1).mean()),
        "reference_true_center_hz": diagnostic["split_reference_center_hz"],
        "reference_distance_hz": float(
            abs(diagnostic["split_reference_center_hz"] - 105.0)
        ),
        "reference_from_core": True,
        "reference_supplement_generated": 0,
        "true_center_knee_rho_train": float(stats.spearmanr(train_c, train_k).statistic),
        "true_center_knee_rho_held": float(stats.spearmanr(held_c, held_k).statistic),
        "train_knee_mean": float(np.mean(train_k)),
        "held_knee_mean": float(np.mean(held_k)),
        "train_knee_min": float(np.min(train_k)),
        "train_knee_max": float(np.max(train_k)),
        "n_train": N_TRAIN,
        "n_heldout": N_HELDOUT,
        "n_mc": 0,
    }


def reconstruct_high_gamma(repeat: int) -> dict[str, object]:
    seed = BASE_SEED + repeat * 100 + HIGH_GAMMA_INDEX * 10
    train_o, train_f, train_c, train_k, train_strength = scan.high_gamma_batch(
        np.random.default_rng(seed), N_TRAIN, STRENGTH_LOW, STRENGTH_HIGH
    )
    held_o, held_f, held_c, held_k, held_strength = scan.high_gamma_batch(
        np.random.default_rng(seed + 1), N_HELDOUT, STRENGTH_LOW, STRENGTH_HIGH
    )
    fooof_train = shared.frozen.official_fit(train_o, DISPLAY_BAND)
    split_train = shared.six.fit_split_batch(train_o, train_f, BAND)
    fooof_linear, split_linear, fooof_j, split_j, diagnostic = _first_order_projection(
        train_o, train_f, train_c, fooof_train, split_train
    )
    fooof = shared.frozen.official_fit(held_o, DISPLAY_BAND)
    split = shared.six.fit_split_batch(held_o, held_f, BAND)
    both_good = np.isfinite(fooof).all(axis=1) & np.isfinite(split).all(axis=1)
    if not np.any(both_good):
        raise RuntimeError("No high-gamma held-out trial valid for both methods")
    target = np.nanmedian(fooof[both_good, 0])
    valid_indices = np.flatnonzero(both_good)
    component_index = int(
        valid_indices[np.argmin(np.abs(fooof[both_good, 0] - target))]
    )
    components = shared.frozen.official_fit_components(held_o[component_index])
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
        "influence": fooof_j,
        "split_influence": split_j,
        "split_reference": diagnostic,
        "fooof_projected": fooof_linear,
        "split_projected": split_linear,
        "fooof": fooof,
        "split": split,
        "component_trial_index": component_index,
        "fooof_component_frequency": components["frequency"],
        "fooof_source_psd": components["source"],
        "fooof_aperiodic_fit": components["aperiodic"],
        "fooof_periodic_fit": components["periodic"],
        "v_fooof": np.cov(fooof_linear.T, ddof=1),
        "v_split": np.cov(split_linear.T, ddof=1),
    }


def validation_metrics(groups: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for method in ("fooof", "split"):
        valid = np.isfinite(groups[f"{method}_pred"]) & np.isfinite(groups[f"{method}_emp"])
        fit = stats.linregress(
            groups.loc[valid, f"{method}_pred"], groups.loc[valid, f"{method}_emp"]
        )
        result[method] = {
            "n_groups": int(valid.sum()),
            "pearson_r": float(fit.rvalue),
            "ols_slope": float(fit.slope),
            "R2": float(fit.rvalue**2),
        }
    return result


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    shared.fixed.load_sources()
    shared.matched.install_variant()
    shared.frozen.official._fooof_group_class()
    shared.frozen.load_empirical_knees()

    checkpoint = OUTPUT / "high_gamma_checkpoint.csv"
    if checkpoint.exists():
        cached = pd.read_csv(checkpoint)
    else:
        cached = pd.DataFrame()
    if len(cached) == N_GROUPS and cached["repeat"].nunique() == N_GROUPS:
        high = cached.sort_values("repeat").reset_index(drop=True)
        print("reused complete high-gamma first-order checkpoint", flush=True)
    else:
        rows: list[dict[str, float | int | str]] = []
        with mp.get_context("fork").Pool(4) as pool:
            for completed, row in enumerate(
                pool.imap_unordered(high_gamma_worker, range(N_GROUPS)), start=1
            ):
                rows.append(row)
                pd.DataFrame(rows).sort_values("repeat").to_csv(
                    checkpoint, index=False
                )
                print(
                    f"completed high-gamma first-order groups: {completed}/{N_GROUPS}",
                    flush=True,
                )
        high = pd.DataFrame(rows).sort_values("repeat").reset_index(drop=True)
    lower = pd.read_csv(DIRECT_GROUPS)
    groups = pd.concat(
        [lower[lower.band != BAND], high], ignore_index=True, sort=False
    ).sort_values(["repeat", "band"]).reset_index(drop=True)
    high.to_csv(OUTPUT / "high_gamma_results.csv", index=False)
    groups.to_csv(OUTPUT / "all_group_results.csv", index=False)
    shared.six.aggregate(groups).to_csv(OUTPUT / "cross_band_summary.csv", index=False)

    selected_repeat = shared.frozen.choose_representative(groups)
    representative = reconstruct_high_gamma(selected_repeat)
    svg, metrics = selected.render_3x3(
        groups,
        representative,
        OUTPUT,
        STEM,
        "FOOOF first-order theory",
        "SPLIT first-order theory",
        rasterize_pdf_linework=True,
    )
    metrics.update(
        {
            "theory": (
                "local active-model estimator Jacobian with direct covariance "
                "of independently generated calibration Welch/IRASA spectra"
            ),
            "formula_fooof": "Cov([dP,dA]) = J_F Cov(dL) J_F^T",
            "formula_split": (
                "Cov(dP,dA) from dP=J_P d(L-F), dA=J_A dF and the "
                "joint calibration covariance of L and F"
            ),
            "gaussian_spectral_mixing": False,
            "selected_peak_strength_distribution": [STRENGTH_LOW, STRENGTH_HIGH],
            "n_groups_per_band": N_GROUPS,
            "n_calibration_per_group_band": N_TRAIN,
            "n_heldout_per_group_band": N_HELDOUT,
            "validation_by_method": validation_metrics(groups),
            "representative_split_reference": representative["split_reference"],
            "data_reuse": (
                "alpha, beta and low-gamma deterministic first-order group results "
                "reused from the identical shared-s(t) direct-covariance run; "
                "high gamma recomputed at U(28,34)"
            ),
        }
    )
    (OUTPUT / f"{STEM}_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(svg)


if __name__ == "__main__":
    main()
