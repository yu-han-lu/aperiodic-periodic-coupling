from __future__ import annotations

"""Recompute Supplementary Fig. 2d-f with matched equal setting weights.

Only the evaluation weighting criticized in manuscript Comment 80 is changed.
All signal generation, estimator fitting, second-order response surfaces,
calibration residuals, seeds, validity definitions and display styling remain
those of the frozen broad-distribution analysis.
"""

import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


HERE = Path(__file__).resolve().parent
FIGURE_SCRIPTS = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/nature_figure/scripts"
for source in (HERE, FIGURE_SCRIPTS):
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))

from audit_panel_alignment import require_matplotlib_panel_alignment  # noqa: E402
import build_broad_total_covariance_3x3 as frozen  # noqa: E402


SOURCE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "outputs/broad_total_covariance"
SOURCE_CSV = SOURCE / "all_group_results.csv"
OUTPUT = HERE / "outputs" / "figs2_def_equal_setting_weights_20261025"
FOOOF_PACKAGE = Path(__import__("os").environ["PAP_RAW_ROOT"]) / "vendor/fooof_1_1_1"
N_RESAMPLES = 50_000

FOOOF = "#247BA0"
SPLIT = "#2A9D8F"
CORRECTED = "#D1495B"

BAND_DISPLAY = {
    "alpha": "Alpha",
    "beta": "Beta",
    "low gamma": "Low gamma",
    "high gamma": "High gamma",
}
BAND_CENTRE = {
    "alpha": 10.0,
    "beta": 21.5,
    "low gamma": 45.0,
    "high gamma": 105.0,
}
BAND_ORDER = ["alpha", "beta", "low gamma", "high gamma"]


def spearman(values: np.ndarray) -> float:
    if len(values) < 8 or not np.isfinite(values).all():
        return np.nan
    return float(stats.spearmanr(values[:, 0], values[:, 1]).statistic)


def matched_equal_setting_correlations(
    held_values: np.ndarray,
    held_index: np.ndarray,
    conditional: np.ndarray,
    residuals: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Match prediction and held-out mixtures using identical setting draws.

    Each evaluable input setting has the same selection probability.  After a
    setting is selected, one of its valid held-out realizations is selected
    uniformly for the empirical mixture.  The prediction uses that same
    setting draw and adds one complete calibration residual pair sampled
    uniformly, exactly as in the frozen prediction.
    """

    valid_pair = np.isfinite(held_values).all(axis=1)
    rows_by_setting: dict[int, np.ndarray] = {}
    for setting in range(len(conditional)):
        rows = np.flatnonzero((held_index == setting) & valid_pair)
        if len(rows):
            rows_by_setting[setting] = rows
    valid_settings = np.asarray(sorted(rows_by_setting), dtype=int)
    if len(valid_settings) < 8:
        raise RuntimeError("Fewer than eight evaluable held-out settings")
    if not np.isfinite(conditional[valid_settings]).all():
        raise RuntimeError("Non-finite response-surface prediction")
    if len(residuals) < 8 or not np.isfinite(residuals).all():
        raise RuntimeError("Insufficient finite calibration residual pairs")

    # The same finite-sample setting draw is used on both sides, so the only
    # difference is fitted-map-plus-calibration-error versus held-out outcome.
    setting_position = rng.integers(0, len(valid_settings), N_RESAMPLES)
    setting_draw = valid_settings[setting_position]

    empirical_rows = np.empty(N_RESAMPLES, dtype=int)
    for setting in valid_settings:
        where = np.flatnonzero(setting_draw == setting)
        candidates = rows_by_setting[int(setting)]
        empirical_rows[where] = candidates[
            rng.integers(0, len(candidates), len(where))
        ]
    empirical_values = held_values[empirical_rows]

    residual_draw = rng.integers(0, len(residuals), N_RESAMPLES)
    predicted_values = conditional[setting_draw] + residuals[residual_draw]

    counts = np.asarray([len(rows_by_setting[int(s)]) for s in valid_settings])
    return {
        "emp": spearman(empirical_values),
        "pred": spearman(predicted_values),
        "valid_settings_n": int(len(valid_settings)),
        "zero_valid_settings_n": int(len(conditional) - len(valid_settings)),
        "valid_pairs_n": int(valid_pair.sum()),
        "min_valid_realizations_per_retained_setting": int(counts.min()),
        "max_valid_realizations_per_retained_setting": int(counts.max()),
        "residual_pairs_n": int(len(residuals)),
    }


def run_one(task: tuple[int, str]) -> dict[str, float | int | str]:
    repeat, band = task
    display_band = BAND_DISPLAY[band]
    original_row, state = frozen.run_group(repeat, display_band, return_state=True)
    if state is None:
        raise RuntimeError("Frozen reconstruction did not return state")

    calibration_parameters = np.asarray(state["calibration_parameters"])
    held_parameters = np.asarray(state["held_parameters"])
    location = calibration_parameters.mean(axis=0)
    scale = calibration_parameters.std(axis=0, ddof=1)
    held_standardized = (held_parameters - location) / scale
    held_index = np.asarray(state["held_index"], dtype=int)

    out: dict[str, float | int | str] = {
        "repeat": repeat,
        "band": band,
        "n_calibration_settings": frozen.N_CALIBRATION_CONDITIONS,
        "n_calibration_realizations": frozen.N_CALIBRATION_REALIZATIONS,
        "n_heldout_settings_attempted": frozen.N_HELDOUT_CONDITIONS,
        "n_heldout_realizations_per_setting": frozen.N_HELDOUT_REALIZATIONS,
        "n_equal_weight_resamples": N_RESAMPLES,
        "original_fooof_emp": float(original_row["fooof_emp"]),
        "original_fooof_pred": float(original_row["fooof_pred"]),
        "original_split_emp": float(original_row["split_emp"]),
        "original_split_pred": float(original_row["split_pred"]),
    }

    seed = (
        frozen.BASE_SEED
        + repeat * 100_000
        + list(frozen.local.shared.researcher.BANDS).index(display_band) * 10_000
    )
    for estimator, offset in (("FOOOF", 71_001), ("SPLIT", 72_001)):
        polynomial, model = state["estimator_state"][estimator]["order2"]
        conditional = model.predict(polynomial.transform(held_standardized))
        result = matched_equal_setting_correlations(
            np.asarray(state["held_fit"][estimator]),
            held_index,
            conditional,
            np.asarray(state["estimator_state"][estimator]["residuals"]),
            np.random.default_rng(seed + offset),
        )
        key = estimator.lower()
        for name, value in result.items():
            out[f"{key}_{name}"] = value

    out["fooof_observed_minus_predicted"] = (
        float(out["fooof_emp"]) - float(out["fooof_pred"])
    )
    return out


def reproduce_groups() -> pd.DataFrame:
    checkpoint = OUTPUT / "equal_setting_group_results.csv"
    if checkpoint.exists():
        return pd.read_csv(checkpoint).sort_values(["repeat", "band"]).reset_index(drop=True)

    tasks = [(repeat, band) for repeat in range(frozen.N_GROUPS) for band in BAND_ORDER]
    rows: list[dict[str, float | int | str]] = []
    with mp.get_context("fork").Pool(4) as pool:
        for completed, row in enumerate(pool.imap_unordered(run_one, tasks), 1):
            rows.append(row)
            pd.DataFrame(rows).sort_values(["repeat", "band"]).to_csv(
                OUTPUT / "equal_setting_checkpoint.csv", index=False
            )
            print(f"equal-setting groups {completed}/{len(tasks)}", flush=True)
    groups = pd.DataFrame(rows).sort_values(["repeat", "band"]).reset_index(drop=True)
    groups.to_csv(checkpoint, index=False)
    return groups


def verify_frozen_reconstruction(groups: pd.DataFrame) -> None:
    source = pd.read_csv(SOURCE_CSV).sort_values(["repeat", "band"]).reset_index(drop=True)
    check = groups.sort_values(["repeat", "band"]).reset_index(drop=True)
    for estimator in ("fooof", "split"):
        for statistic in ("emp", "pred"):
            frozen_values = source[f"{estimator}_{statistic}"].to_numpy()
            replay_values = check[f"original_{estimator}_{statistic}"].to_numpy()
            maximum = float(np.nanmax(np.abs(frozen_values - replay_values)))
            if maximum > 1e-12:
                raise AssertionError(
                    f"Frozen replay mismatch for {estimator}_{statistic}: {maximum}"
                )


def aggregate(groups: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for band in BAND_ORDER:
        part = groups[groups.band == band]
        rows.append(
            {
                "band": band,
                "center": BAND_CENTRE[band],
                "fooof_pred": float(part.fooof_pred.mean()),
                "fooof_emp": float(part.fooof_emp.mean()),
                "fooof_lo": float(part.fooof_emp.quantile(0.025)),
                "fooof_hi": float(part.fooof_emp.quantile(0.975)),
                "split_pred": float(part.split_pred.mean()),
                "split_emp": float(part.split_emp.mean()),
                "split_lo": float(part.split_emp.quantile(0.025)),
                "split_hi": float(part.split_emp.quantile(0.975)),
            }
        )
    return pd.DataFrame(rows)


def style_axis(ax: plt.Axes) -> None:
    ax.tick_params(width=0.6, length=2.5, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["bottom", "left"]].set_linewidth(0.65)


def new_panel(label: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(2.4, 2.176), constrained_layout=False)
    fig.subplots_adjust(left=0.24, right=0.96, bottom=0.22, top=0.89)
    ax.text(
        -0.205,
        1.035,
        label,
        transform=ax.transAxes,
        fontsize=8,
        fontweight="bold",
        ha="left",
        va="bottom",
    )
    style_axis(ax)
    return fig, ax


def save_panel(fig: plt.Figure, stem: str) -> None:
    prefix = OUTPUT / stem
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=prefix.with_suffix(".alignment.json"),
        overlay_svg=prefix.with_suffix(".alignment.svg"),
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=False,
        strict=True,
    )
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), facecolor="white")
    fig.savefig(prefix.with_suffix(".tiff"), dpi=600, facecolor="white")
    fig.savefig(prefix.with_name(prefix.name + "_preview.png"), dpi=300, facecolor="white")
    plt.close(fig)


def render_d(groups: pd.DataFrame) -> dict[str, float | int]:
    x = groups.fooof_pred.to_numpy()
    y = groups.fooof_emp.to_numpy()
    fit = stats.linregress(x, y)
    bounds = [min(x.min(), y.min()) - 0.08, max(x.max(), y.max()) + 0.08]
    fig, ax = new_panel("d")
    ax.scatter(x, y, s=11, color=FOOOF, alpha=0.80, edgecolor="black", linewidth=0.25)
    ax.plot(bounds, bounds, ls="--", color="0.35", lw=0.7)
    ax.set(
        xlim=bounds,
        ylim=bounds,
        xlabel="Theory-predicted error ρ",
        ylabel="Held-out error ρ",
    )
    ax.text(
        0.96,
        0.06,
        f"Pearson r = {fit.rvalue:.2f}\nOLS slope = {fit.slope:.2f}; R² = {fit.rvalue**2:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.5,
        color="black",
    )
    save_panel(fig, "FigS2d_equal_setting_weights")
    return {
        "n_groups": int(len(groups)),
        "pearson_r": float(fit.rvalue),
        "pearson_p": float(fit.pvalue),
        "ols_slope": float(fit.slope),
        "ols_intercept": float(fit.intercept),
        "R2": float(fit.rvalue**2),
    }


def render_e(groups: pd.DataFrame) -> list[dict[str, float | str]]:
    summary = aggregate(groups)
    fig, ax = new_panel("e")
    for display_band, (lo, hi, _) in frozen.local.shared.researcher.BANDS.items():
        band = display_band.lower()
        ax.axvspan(lo, hi, color=frozen.local.shared.six.base.BAND_COLORS[band], alpha=0.075, lw=0)
    ax.fill_between(summary.center, summary.fooof_lo, summary.fooof_hi, color=FOOOF, alpha=0.14)
    ax.fill_between(summary.center, summary.split_lo, summary.split_hi, color=CORRECTED, alpha=0.14)
    ax.plot(summary.center, summary.fooof_emp, "o-", color=FOOOF, label="FOOOF empirical", lw=0.8, ms=2.6)
    ax.plot(summary.center, summary.fooof_pred, "--", color=FOOOF, label="FOOOF theory", lw=0.8)
    ax.plot(summary.center, summary.split_emp, "o-", color=CORRECTED, label="SPLIT empirical", lw=0.8, ms=2.6)
    ax.plot(summary.center, summary.split_pred, "--", color=CORRECTED, label="SPLIT theory", lw=0.8)
    ax.axhline(0, color="0.4", lw=0.55)
    ax.set(
        xscale="log",
        xlabel="Oscillation band center (Hz)",
        ylabel="Estimated p–ap Spearman ρ",
        ylim=(-0.3, 0.3),
    )
    ax.set_xticks([8, 12, 30, 60, 100, 150])
    ax.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    for label, label_x in [
        ("alpha", 9.2),
        ("beta", 21.5),
        ("low gamma", 45.0),
        ("high gamma", 105.0),
    ]:
        ax.text(
            label_x,
            1.015,
            label,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="bottom",
            fontsize=5.0,
            fontweight="bold",
            color=frozen.local.shared.six.base.BAND_COLORS[label],
            clip_on=False,
        )
    ax.legend(frameon=False, fontsize=5.0, loc="lower right", handlelength=1.8, labelspacing=0.25, borderaxespad=0.3)
    save_panel(fig, "FigS2e_equal_setting_weights")
    return summary.to_dict(orient="records")


def exact_wilcoxon(values: np.ndarray) -> float:
    return float(stats.wilcoxon(values, alternative="two-sided", method="exact").pvalue)


def render_f(groups: pd.DataFrame) -> dict[str, object]:
    high = groups[groups.band == "high gamma"].sort_values("repeat")
    arrays = [
        high.fooof_emp.to_numpy(),
        high.fooof_observed_minus_predicted.to_numpy(),
        high.split_emp.to_numpy(),
    ]
    fig, ax = new_panel("f")
    violin = ax.violinplot(arrays, positions=[0, 1, 2], widths=0.76, showmeans=True, showextrema=False)
    for body, color in zip(violin["bodies"], [FOOOF, SPLIT, CORRECTED]):
        body.set_facecolor(color)
        body.set_edgecolor("black")
        body.set_linewidth(0.6)
        body.set_alpha(0.80)
    violin["cmeans"].set_color("black")
    violin["cmeans"].set_linewidth(1.0)
    ax.axhline(0, color="0.35", lw=0.6)
    ax.set(
        xticks=[0, 1, 2],
        xticklabels=["FOOOF", "FOOOF\nobserved − predicted", "SPLIT"],
        ylabel="Spearman ρ or Δρ",
        ylim=(-0.5, 0.5),
    )
    for label in ax.get_xticklabels():
        label.set_linespacing(1.25)
    save_panel(fig, "FigS2f_equal_setting_weights")

    names = ["FOOOF", "FOOOF observed minus predicted", "SPLIT"]
    results = {}
    for name, values in zip(names, arrays):
        results[name] = {
            "n_groups": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p_against_zero": exact_wilcoxon(values),
        }
    results["corrected_vs_SPLIT_p"] = float(
        stats.wilcoxon(arrays[1], arrays[2], alternative="two-sided", method="exact").pvalue
    )
    return results


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mplconfig_figs2")
    frozen.local.shared.frozen.official.FOOOF_PACKAGE = FOOOF_PACKAGE
    frozen.local.initialize()
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.2,
            "axes.labelsize": 6.4,
            "axes.linewidth": 0.65,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    groups = reproduce_groups()
    verify_frozen_reconstruction(groups)
    panel_d = render_d(groups)
    panel_e = render_e(groups)
    panel_f = render_f(groups)

    high = groups[groups.band == "high gamma"].copy()
    high.to_csv(OUTPUT / "FigS2f_source_data.csv", index=False)
    aggregate(groups).to_csv(OUTPUT / "FigS2e_source_data.csv", index=False)
    groups[
        ["repeat", "band", "fooof_pred", "fooof_emp"]
    ].to_csv(OUTPUT / "FigS2d_source_data.csv", index=False)

    provenance = {
        "change_requested": "Manuscript Comment 80: match prediction and held-out correlation using equal input-setting weights.",
        "what_changed": (
            "For each estimator and group, the theory and held-out mixtures use the same 50,000 uniformly sampled evaluable-setting indices. "
            "Within a selected setting, one valid held-out realization is sampled uniformly for the empirical mixture; the theory adds one complete calibration residual pair to the second-order conditional mean."
        ),
        "what_did_not_change": [
            "four input-generating parameter distributions",
            "64 calibration settings and eight realizations per setting",
            "128 held-out settings and four realizations per setting",
            "30-s, 1,000-Hz OU-plus-bandpass signal generation and Welch PSD",
            "FOOOF 1.1.1 knee-mode fitting and IRASA-SPLIT fitting",
            "second-order calibration response surface",
            "pooled complete two-coordinate calibration residual pairs",
            "estimator-specific validity definition",
            "20 groups in each of four frequency bands",
            "direct Spearman correction rho_observed minus rho_predicted in panel f",
        ],
        "validity_support": (
            "Estimator-specific settings with at least one valid held-out realization are retained; theory is evaluated on that identical setting set. "
            "FOOOF and SPLIT are not forced onto a cross-estimator common mask."
        ),
        "resampling": {
            "n": N_RESAMPLES,
            "setting_probability": "uniform across retained settings",
            "within_setting_realization_probability": "uniform across valid held-out realizations",
            "calibration_residual_probability": "uniform across pooled complete residual pairs",
            "same_setting_draw_used_for_prediction_and_heldout": True,
            "seed_rule": "BASE_SEED + repeat*100000 + band_index*10000 + 71001 (FOOOF) or 72001 (SPLIT)",
        },
        "frozen_replay_check": "All 320 original group-level empirical/predicted values matched the frozen CSV within 1e-12 before reweighting.",
        "source_csv_sha256": hashlib.sha256(SOURCE_CSV.read_bytes()).hexdigest(),
        "package": "FOOOF 1.1.1 from existing local package cache",
        "panel_d": panel_d,
        "panel_e": panel_e,
        "panel_f": panel_f,
        "validity_summary": {
            estimator: {
                "retained_settings_min": int(groups[f"{estimator}_valid_settings_n"].min()),
                "retained_settings_median": float(groups[f"{estimator}_valid_settings_n"].median()),
                "retained_settings_max": int(groups[f"{estimator}_valid_settings_n"].max()),
                "zero_valid_settings_total": int(groups[f"{estimator}_zero_valid_settings_n"].sum()),
                "valid_pairs_total": int(groups[f"{estimator}_valid_pairs_n"].sum()),
            }
            for estimator in ("fooof", "split")
        },
    }
    (OUTPUT / "statistics_and_provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUTPUT / "QA_notes.md").write_text(
        "# QA notes\n\n"
        "- The frozen analysis was replayed before reweighting; all 320 original empirical/predicted group values matched within 1e-12.\n"
        "- The only inferential change is matched equal-probability sampling of evaluable held-out settings for prediction and empirical evaluation.\n"
        "- Panels d and f passed the rendered collision audit without findings.\n"
        "- Panel e has no collision failures. Its four text-fill-edge warnings arise because the legend is intentionally placed over the pale frequency-band background spans; the legend remains unobscured at final size.\n"
        "- Standalone panel width is 61 mm to preserve the original Supplementary Fig. 2 panel scale; the SVG and PDF retain editable text.\n",
        encoding="utf-8",
    )
    print(json.dumps(provenance, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
