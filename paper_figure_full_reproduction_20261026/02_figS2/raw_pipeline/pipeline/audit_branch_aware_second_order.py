from __future__ import annotations

"""Pilot audit of first/second-order and branch-aware estimator theory.

The data generator and FOOOF/IRASA--SPLIT settings are frozen to the selected
high-gamma peak-strength version.  Calibration and held-out Gaussian spectral
clouds are independent.  Polynomial response surfaces are local Taylor models
in calibration-PCA coordinates: degree 1 estimates the Jacobian, and degree 2
adds the Hessian terms.  Branch-specific fits expose non-smooth peak-selection
effects separately from within-branch curvature.
"""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


HERE = Path(__file__).resolve().parent
PROJECT_CODE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "project/experiments/pap_theory_validation/pap_validation"
for source_path in (HERE, PROJECT_CODE):
    if str(source_path) not in sys.path:
        sys.path.insert(0, str(source_path))

import build_two_fooof_split_3x3_revisions as selected  # noqa: E402


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_split_branch_aware_second_order_audit_2026-09-13"
)
DISPLAY_BAND = "High gamma"
BAND = "high gamma"
REPEAT = 2
N_CALIBRATION = 1500
N_HELDOUT = 1500
N_COMPONENTS = 32
N_BRANCH_COMPONENTS = 12
MIN_BRANCH = 120
ALPHAS = np.logspace(-5, 4, 28)


def fit_fooof_with_branches(spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return official FOOOF outputs plus active peak-selection labels."""
    frozen = selected.gaussian_old.frozen
    researcher = selected.gaussian_old.researcher
    FOOOFGroup = frozen.official._fooof_group_class()
    fit_mask = (frozen.base.FREQ >= 0.2) & (frozen.base.FREQ <= 197.0)
    fg = FOOOFGroup(
        aperiodic_mode="knee",
        max_n_peaks=3,
        peak_width_limits=[1.5, 6.0],
        min_peak_height=0.6,
        peak_threshold=2.0,
        verbose=False,
    )
    fg.fit(
        frozen.base.FREQ[fit_mask],
        10.0 ** spectra[:, fit_mask],
        [0.2, 197.0],
    )
    exponent = np.asarray(
        fg.get_params("aperiodic_params", "exponent"), dtype=float
    )
    lo, hi, _ = researcher.BANDS[DISPLAY_BAND]
    center = np.full(len(spectra), np.nan)
    labels: list[str] = []
    details: list[dict[str, float | int | str]] = []
    for index in range(len(spectra)):
        model = fg.get_fooof(index, regenerate=True)
        peaks = np.asarray(model.peak_params_, dtype=float)
        if peaks.size == 0:
            peaks = np.empty((0, 3), dtype=float)
        else:
            peaks = np.atleast_2d(peaks)
        candidates = peaks[(peaks[:, 0] >= lo) & (peaks[:, 0] <= hi)]
        if len(candidates) == 0:
            label = f"K{len(peaks)}_NONE"
            target_rank = -1
            boundary = "NA"
        else:
            selected_index = int(np.argmax(candidates[:, 1]))
            chosen = candidates[selected_index]
            center[index] = chosen[0]
            target_rank = int(np.sum(peaks[:, 0] < chosen[0]))
            if chosen[2] <= 1.55:
                boundary = "L"
            elif chosen[2] >= 5.95:
                boundary = "U"
            else:
                boundary = "I"
            label = (
                f"K{len(peaks)}_C{len(candidates)}_"
                f"T{target_rank}_B{boundary}"
            )
        # The model-selection branch used by the Taylor theory is deliberately
        # coarse: which fitted peak becomes the reported high-gamma peak.  Peak
        # count and bound state remain in the audit table but would fragment the
        # calibration sample too strongly for a stable Hessian estimate.
        theory_label = "NONE" if target_rank < 0 else f"TARGET_RANK_{target_rank}"
        labels.append(theory_label)
        details.append(
            {
                "n_peaks": int(len(peaks)),
                "n_candidates": int(len(candidates)),
                "target_rank": target_rank,
                "bandwidth_state": boundary,
                "label": label,
            }
        )
    return (
        np.column_stack([center, exponent]),
        np.asarray(labels, dtype=object),
        pd.DataFrame(details),
    )


def split_branches(values: np.ndarray) -> np.ndarray:
    """SPLIT has one peak branch; retain validity and center-boundary states."""
    labels = np.full(len(values), "INVALID", dtype=object)
    valid = np.isfinite(values).all(axis=1)
    centers = values[:, 0]
    labels[valid & (centers < 61.0)] = "VALID_LOWER"
    labels[valid & (centers > 149.0)] = "VALID_UPPER"
    labels[valid & (centers >= 61.0) & (centers <= 149.0)] = "VALID_INTERIOR"
    return labels


def spearman_pair(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if int(good.sum()) < 8:
        return np.nan
    return float(stats.spearmanr(values[good, 0], values[good, 1]).statistic)


def pearson_pair(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if int(good.sum()) < 8:
        return np.nan
    return float(stats.pearsonr(values[good, 0], values[good, 1]).statistic)


def cross_covariance(values: np.ndarray) -> float:
    good = np.isfinite(values).all(axis=1)
    if int(good.sum()) < 8:
        return np.nan
    return float(np.cov(values[good].T, ddof=1)[0, 1])


def build_taylor_model(order: int):
    """Fit a regularized local Taylor response surface of requested order."""
    return make_pipeline(
        PolynomialFeatures(degree=order, include_bias=False),
        StandardScaler(),
        RidgeCV(alphas=ALPHAS),
    )


def fit_global_models(
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    x_held: np.ndarray,
) -> dict[int, np.ndarray]:
    good = np.isfinite(y_cal).all(axis=1)
    predictions: dict[int, np.ndarray] = {}
    for order in (1, 2):
        model = build_taylor_model(order)
        model.fit(x_cal[good], y_cal[good])
        predictions[order] = model.predict(x_held)
    return predictions


def retained_labels(labels: np.ndarray, valid: np.ndarray) -> list[str]:
    counts = pd.Series(labels[valid]).value_counts()
    return [str(label) for label, count in counts.items() if int(count) >= MIN_BRANCH]


def fit_branch_models(
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    labels_cal: np.ndarray,
    x_held: np.ndarray,
    labels_held: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], np.ndarray, dict[str, int]]:
    """Return deployable classified and oracle branch predictions."""
    valid_cal = np.isfinite(y_cal).all(axis=1)
    keep = retained_labels(labels_cal, valid_cal)
    fallback = max(keep, key=lambda item: int(np.sum(labels_cal == item)))
    accepted_cal = valid_cal & np.isin(labels_cal, keep)

    classifier = RandomForestClassifier(
        n_estimators=500,
        min_samples_leaf=12,
        class_weight="balanced_subsample",
        random_state=20260913,
        n_jobs=-1,
    )
    classifier.fit(x_cal[accepted_cal], labels_cal[accepted_cal])
    predicted_labels = classifier.predict(x_held)

    branch_models: dict[tuple[str, int], object] = {}
    for label in keep:
        subset = valid_cal & (labels_cal == label)
        for order in (1, 2):
            model = build_taylor_model(order)
            model.fit(x_cal[subset, :N_BRANCH_COMPONENTS], y_cal[subset])
            branch_models[(label, order)] = model

    predicted: dict[int, np.ndarray] = {}
    oracle: dict[int, np.ndarray] = {}
    for order in (1, 2):
        classified = np.full((len(x_held), 2), np.nan)
        oracle_values = np.full((len(x_held), 2), np.nan)
        for label in keep:
            selected = predicted_labels == label
            if np.any(selected):
                classified[selected] = branch_models[(label, order)].predict(
                    x_held[selected, :N_BRANCH_COMPONENTS]
                )
            oracle_selected = labels_held == label
            if np.any(oracle_selected):
                oracle_values[oracle_selected] = branch_models[(label, order)].predict(
                    x_held[oracle_selected, :N_BRANCH_COMPONENTS]
                )
        # An unseen predicted class is impossible, but keep an explicit fallback.
        missing = ~np.isfinite(classified).all(axis=1)
        if np.any(missing):
            classified[missing] = branch_models[(fallback, order)].predict(
                x_held[missing, :N_BRANCH_COMPONENTS]
            )
        predicted[order] = classified
        oracle[order] = oracle_values
    counts = {label: int(np.sum(labels_cal == label)) for label in keep}
    return predicted, oracle, predicted_labels, counts


def metric_row(
    estimator: str,
    theory: str,
    prediction: np.ndarray,
    observed: np.ndarray,
    mask: np.ndarray | None = None,
) -> dict[str, float | int | str]:
    use = np.ones(len(observed), dtype=bool) if mask is None else mask.copy()
    use &= np.isfinite(prediction).all(axis=1) & np.isfinite(observed).all(axis=1)
    predicted = prediction[use]
    actual = observed[use]
    if len(actual) < 8:
        return {"estimator": estimator, "theory": theory, "n": int(len(actual))}
    return {
        "estimator": estimator,
        "theory": theory,
        "n": int(len(actual)),
        "predicted_spearman": spearman_pair(predicted),
        "observed_spearman": spearman_pair(actual),
        "abs_spearman_error": abs(spearman_pair(predicted) - spearman_pair(actual)),
        "predicted_pearson": pearson_pair(predicted),
        "observed_pearson": pearson_pair(actual),
        "predicted_cross_cov": cross_covariance(predicted),
        "observed_cross_cov": cross_covariance(actual),
        "r2_peak": float(r2_score(actual[:, 0], predicted[:, 0])),
        "r2_aperiodic": float(r2_score(actual[:, 1], predicted[:, 1])),
    }


def estimator_audit(
    name: str,
    x_cal: np.ndarray,
    y_cal: np.ndarray,
    labels_cal: np.ndarray,
    x_held: np.ndarray,
    y_held: np.ndarray,
    labels_held: np.ndarray,
) -> tuple[list[dict[str, float | int | str]], dict[str, object]]:
    global_prediction = fit_global_models(x_cal, y_cal, x_held)
    classified, oracle, predicted_labels, counts = fit_branch_models(
        x_cal, y_cal, labels_cal, x_held, labels_held
    )
    rows: list[dict[str, float | int | str]] = []
    for order in (1, 2):
        rows.append(
            metric_row(name, f"global_order{order}", global_prediction[order], y_held)
        )
        rows.append(
            metric_row(name, f"branch_predicted_order{order}", classified[order], y_held)
        )
        oracle_mask = np.isfinite(oracle[order]).all(axis=1)
        rows.append(
            metric_row(
                name,
                f"branch_oracle_order{order}",
                oracle[order],
                y_held,
                oracle_mask,
            )
        )
    valid_held = np.isfinite(y_held).all(axis=1)
    classifier_accuracy = float(
        np.mean(predicted_labels[valid_held] == labels_held[valid_held])
    )
    diagnostics: dict[str, object] = {
        "retained_calibration_branches": counts,
        "classifier_accuracy_on_valid_heldout": classifier_accuracy,
        "calibration_valid_n": int(np.isfinite(y_cal).all(axis=1).sum()),
        "heldout_valid_n": int(valid_held.sum()),
        "heldout_branch_counts": {
            str(k): int(v) for k, v in pd.Series(labels_held).value_counts().items()
        },
    }
    return rows, diagnostics


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected.gaussian_old.fixed.load_sources()
    selected.gaussian_old.matched.install_variant()

    seed = selected.BASE_SEED + REPEAT * 100 + selected.HIGH_GAMMA_INDEX * 10
    train_o, train_f, _, _, _ = selected.scan.high_gamma_batch(
        np.random.default_rng(seed),
        selected.N_TRAIN,
        selected.STRENGTH_LOW,
        selected.STRENGTH_HIGH,
    )
    cal_o, cal_f = selected.gaussian_old.frozen.covariance_mc_spectra(
        np.random.default_rng(seed + 20), train_o, train_f, n_mc=N_CALIBRATION
    )
    held_o, held_f = selected.gaussian_old.frozen.covariance_mc_spectra(
        np.random.default_rng(seed + 21), train_o, train_f, n_mc=N_HELDOUT
    )

    fooof_cal, fooof_cal_branch, fooof_cal_details = fit_fooof_with_branches(cal_o)
    fooof_held, fooof_held_branch, fooof_held_details = fit_fooof_with_branches(held_o)
    split_cal = selected.gaussian_old.six.fit_split_batch(cal_o, cal_f, BAND)
    split_held = selected.gaussian_old.six.fit_split_batch(held_o, held_f, BAND)
    split_cal_branch = split_branches(split_cal)
    split_held_branch = split_branches(split_held)

    pca_fooof = PCA(n_components=N_COMPONENTS, whiten=False, random_state=20260913)
    x_fooof_cal = pca_fooof.fit_transform(cal_o)
    x_fooof_held = pca_fooof.transform(held_o)
    split_cal_input = np.column_stack([cal_o - cal_f, cal_f])
    split_held_input = np.column_stack([held_o - held_f, held_f])
    pca_split = PCA(n_components=N_COMPONENTS, whiten=False, random_state=20260913)
    x_split_cal = pca_split.fit_transform(split_cal_input)
    x_split_held = pca_split.transform(split_held_input)

    rows: list[dict[str, float | int | str]] = []
    fooof_rows, fooof_diagnostics = estimator_audit(
        "FOOOF",
        x_fooof_cal,
        fooof_cal,
        fooof_cal_branch,
        x_fooof_held,
        fooof_held,
        fooof_held_branch,
    )
    rows.extend(fooof_rows)
    split_rows, split_diagnostics = estimator_audit(
        "SPLIT",
        x_split_cal,
        split_cal,
        split_cal_branch,
        x_split_held,
        split_held,
        split_held_branch,
    )
    rows.extend(split_rows)

    comparison = pd.DataFrame(rows)
    comparison.to_csv(OUTPUT / "theory_comparison.csv", index=False)
    fooof_cal_details.assign(dataset="calibration").to_csv(
        OUTPUT / "fooof_calibration_branch_details.csv", index=False
    )
    fooof_held_details.assign(dataset="heldout").to_csv(
        OUTPUT / "fooof_heldout_branch_details.csv", index=False
    )
    diagnostics = {
        "configuration": {
            "repeat": REPEAT,
            "n_calibration": N_CALIBRATION,
            "n_heldout": N_HELDOUT,
            "n_pca_components": N_COMPONENTS,
            "n_branch_pca_components": N_BRANCH_COMPONENTS,
            "min_branch": MIN_BRANCH,
            "strength_range": [selected.STRENGTH_LOW, selected.STRENGTH_HIGH],
        },
        "pca_variance_explained": {
            "FOOOF": float(pca_fooof.explained_variance_ratio_.sum()),
            "SPLIT": float(pca_split.explained_variance_ratio_.sum()),
        },
        "FOOOF": fooof_diagnostics,
        "SPLIT": split_diagnostics,
    }
    (OUTPUT / "audit_metrics.json").write_text(
        json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(comparison.to_string(index=False))
    print(json.dumps(diagnostics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
