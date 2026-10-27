from __future__ import annotations

"""Build two official-three-peak FOOOF/SPLIT A--F figures.

Scenarios
---------
fixed2
    The true OU knee is fixed at 2 Hz. Each group has an independent 30-s OU
    realization, avoiding reuse of one atypical finite record.
lognormal40
    Trial-level knees follow a 10--100 Hz truncated log-normal distribution
    with E[knee] = 40 Hz. Knee and true peak center are independently assigned.

For the non-smooth official FOOOF max-three estimator, theoretical error
correlations are obtained by covariance propagation: the joint training PSD
covariance is represented exactly in its low-rank sample basis, Gaussian draws
are propagated through the official estimator, and independent held-out raw
time-domain simulations test the prediction. This includes peak detection and
model-selection transitions that a single active-set Jacobian cannot capture.
Panel B retains the local active-model J_P/J_AP mechanism for interpretation.
"""

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import io, optimize, signal, stats

import fooof_split_attached_code_theory as base
import fooof_split_researcher_e_gamma_fix as fixed
import fooof_split_researcher_ou_six_panel as six
import fooof_split_researcher_pipeline_matched as matched
import knee_position_official_fooof3_band_sweep as sweep
import reproduce_gamma_artifact as researcher
import rerun_researcher_official_fooof_bands as official
from fooof_split_knee2_corrected_official3_six_panel import (
    official_multipeak_influence,
)


ROOT = Path(__file__).resolve().parents[3]
OUT_ROOT = ROOT / "output" / "fooof_split_official3_two_knee_sixpanels_2026-09-01"
N_GROUPS = 20
N_TRAIN = 120
N_HELDOUT = 120
N_MC = 240
BASE_SEED = 20260901 + 91000
LOG_KNEE_SIGMA = 0.45
KNEE_MIN = 10.0
KNEE_MAX = 100.0
TARGET_KNEE_MEAN = 40.0
EMPIRICAL_KNEE_FILE = Path(
    "/PORTABLE_HOME/Library/Containers/com.tencent.xinWeChat/Data/Documents/"
    "xwechat_files/wxid_rhp0l7gp8pzq22_9b7d/msg/file/2026-09/single_knee_data.mat"
)
EMPIRICAL_KNEES: np.ndarray | None = None


def truncated_lognormal_mean(mu: float, sigma: float) -> float:
    za = (np.log(KNEE_MIN) - mu) / sigma
    zb = (np.log(KNEE_MAX) - mu) / sigma
    numerator = np.exp(mu + sigma**2 / 2.0) * (
        stats.norm.cdf(zb - sigma) - stats.norm.cdf(za - sigma)
    )
    denominator = stats.norm.cdf(zb) - stats.norm.cdf(za)
    return float(numerator / denominator)


LOG_KNEE_MU = float(
    optimize.brentq(
        lambda value: truncated_lognormal_mean(value, LOG_KNEE_SIGMA)
        - TARGET_KNEE_MEAN,
        np.log(15.0),
        np.log(70.0),
    )
)


def install_fixed_base(repeat: int, knee_hz: float = 2.0) -> float:
    rng = np.random.default_rng(BASE_SEED + repeat * 10000 + 777)
    innovations = rng.normal(size=sweep.N)
    x = sweep.ou_from_innovations(innovations, knee_hz)
    frequency, psd = sweep.base_welch(x)
    fixed.BASE_SIGNAL = x
    fixed.BASE_FREQ = frequency
    fixed.BASE_PSD = psd
    fixed.BASE_LOG_PSD = np.interp(base.FREQ, frequency, np.log10(psd))
    ap = researcher.fit_ap_knee(fixed.BASE_LOG_PSD, base.FREQ)
    return float(10.0 ** (ap[2] / ap[1]))


def sample_knees_independent_of_centers(
    rng: np.random.Generator, centers: np.ndarray
) -> np.ndarray:
    """Stratified truncated-normal log knees, permuted to near-zero rank rho."""
    n = len(centers)
    a = (np.log(KNEE_MIN) - LOG_KNEE_MU) / LOG_KNEE_SIGMA
    b = (np.log(KNEE_MAX) - LOG_KNEE_MU) / LOG_KNEE_SIGMA
    probabilities = (np.arange(n) + rng.random(n)) / n
    logs = stats.truncnorm.ppf(
        probabilities, a, b, loc=LOG_KNEE_MU, scale=LOG_KNEE_SIGMA
    )
    knees = np.exp(logs)
    best = knees.copy()
    best_rho = np.inf
    for _ in range(32):
        candidate = rng.permutation(knees)
        rho = abs(float(stats.spearmanr(centers, candidate).statistic))
        if rho < best_rho:
            best, best_rho = candidate, rho
    return best


def load_empirical_knees() -> np.ndarray:
    global EMPIRICAL_KNEES
    values = np.asarray(
        io.loadmat(EMPIRICAL_KNEE_FILE, squeeze_me=True)["single_knee_data"],
        dtype=float,
    ).ravel()
    values = np.sort(values[np.isfinite(values) & (values > 0)])
    if len(values) < 100:
        raise ValueError("single_knee_data must contain at least 100 positive values")
    EMPIRICAL_KNEES = values
    return values


def sample_empirical_knees_independent_of_centers(
    rng: np.random.Generator, centers: np.ndarray
) -> np.ndarray:
    """Stratified inverse-ECDF sampling from the supplied experimental knees."""
    if EMPIRICAL_KNEES is None:
        load_empirical_knees()
    assert EMPIRICAL_KNEES is not None
    n = len(centers)
    probabilities = (np.arange(n) + rng.random(n)) / n
    knees = np.quantile(EMPIRICAL_KNEES, probabilities, method="linear")
    best = knees.copy()
    best_rho = np.inf
    for _ in range(32):
        candidate = rng.permutation(knees)
        rho = abs(float(stats.spearmanr(centers, candidate).statistic))
        if rho < best_rho:
            best, best_rho = candidate, rho
    return best


def variable_knee_batch(
    rng: np.random.Generator,
    display_band: str,
    n_trials: int,
    scenario: str = "lognormal40",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate time-domain OU+bump trials with one independent knee per trial."""
    assert fixed.FRACTAL_RESIDUALS is not None
    lo, hi, _ = researcher.BANDS[display_band]
    centers = rng.uniform(lo, hi, n_trials)
    widths = rng.uniform(1.5, 6.0, n_trials)
    fractions = rng.uniform(2.8, 3.8, n_trials)
    knees = (
        sample_knees_independent_of_centers(rng, centers)
        if scenario == "lognormal40"
        else sample_empirical_knees_independent_of_centers(rng, centers)
    )
    original: list[np.ndarray] = []
    fractal: list[np.ndarray] = []
    residual_indices = rng.integers(0, len(fixed.FRACTAL_RESIDUALS), n_trials)
    for index, (center, width, fraction, knee) in enumerate(
        zip(centers, widths, fractions, knees)
    ):
        innovations = rng.normal(size=sweep.N)
        base_signal = sweep.ou_from_innovations(innovations, float(knee))
        base_frequency, base_psd = sweep.base_welch(base_signal)
        frequency, log_psd = researcher.generate_trial_psd(
            rng,
            base_signal,
            base_frequency,
            base_psd,
            float(center),
            float(width),
            float(fraction),
        )
        original.append(np.interp(base.FREQ, frequency, log_psd))
        base_log = np.interp(base.FREQ, base_frequency, np.log10(base_psd))
        fractal.append(
            base_log
            + fixed.FRACTAL_NOISE_SCALE * fixed.FRACTAL_RESIDUALS[residual_indices[index]]
        )
    return np.asarray(original), np.asarray(fractal), centers, knees


def fixed_knee_batch(
    rng: np.random.Generator, display_band: str, n_trials: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    original, fractal, centers = fixed.generate_batch(rng, display_band, n_trials)
    return original, fractal, centers, np.full(n_trials, 2.0)


def generate_batch(
    scenario: str,
    rng: np.random.Generator,
    display_band: str,
    n_trials: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if scenario == "fixed2":
        return fixed_knee_batch(rng, display_band, n_trials)
    if scenario in {"lognormal40", "empirical"}:
        return variable_knee_batch(rng, display_band, n_trials, scenario=scenario)
    raise ValueError(scenario)


def official_fit(original: np.ndarray, display_band: str) -> np.ndarray:
    return official.official_fooof(original, display_band, max_n_peaks=3)


def official_fit_components(log_psd: np.ndarray) -> dict[str, np.ndarray]:
    """Return the exact official three-peak FOOOF decomposition for one PSD."""
    FOOOFGroup = official._fooof_group_class()
    fit = (base.FREQ >= 0.2) & (base.FREQ <= 197.0)
    fg = FOOOFGroup(
        aperiodic_mode="knee",
        max_n_peaks=3,
        peak_width_limits=[1.5, 6.0],
        min_peak_height=0.6,
        peak_threshold=2.0,
        verbose=False,
    )
    fg.fit(
        base.FREQ[fit],
        (10.0 ** np.asarray(log_psd)[fit])[None, :],
        [0.2, 197.0],
    )
    model = fg.get_fooof(0, regenerate=True)
    return {
        "frequency": np.asarray(model.freqs, dtype=float),
        "source": np.asarray(log_psd, dtype=float)[fit],
        "aperiodic": np.asarray(model.get_model("aperiodic", "log"), dtype=float),
        "periodic": np.asarray(model.get_model("peak", "log"), dtype=float),
    }


def covariance_mc_spectra(
    rng: np.random.Generator,
    original: np.ndarray,
    fractal: np.ndarray,
    n_mc: int = N_MC,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian low-rank propagation preserving the training joint covariance."""
    centered_o = original - original.mean(axis=0, keepdims=True)
    centered_f = fractal - fractal.mean(axis=0, keepdims=True)
    weights = rng.normal(size=(n_mc, len(original))) / np.sqrt(len(original) - 1.0)
    mc_o = original.mean(axis=0, keepdims=True) + weights @ centered_o
    mc_f = fractal.mean(axis=0, keepdims=True) + weights @ centered_f
    return mc_o, mc_f


def local_influence(
    original: np.ndarray, centers: np.ndarray, display_band: str
) -> np.ndarray:
    """Choose a detected representative trial for the local B-panel Jacobian."""
    midpoint = np.mean(researcher.BANDS[display_band][:2])
    order = np.argsort(np.abs(centers - midpoint))
    for index in order:
        influence = official_multipeak_influence(original[index], display_band)
        if np.isfinite(influence).all():
            return influence
    return np.zeros((2, len(base.FREQ)))


def worker(task: tuple[str, int]) -> list[dict[str, float | int | str]]:
    scenario, repeat = task
    fitted_base_knee = install_fixed_base(repeat) if scenario == "fixed2" else np.nan
    rows: list[dict[str, float | int | str]] = []
    for band_index, display_band in enumerate(researcher.BANDS):
        band = display_band.lower()
        seed_offset = {"fixed2": 0, "lognormal40": 2_000_000, "empirical": 4_000_000}
        seed = BASE_SEED + seed_offset[scenario]
        train_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10)
        held_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10 + 1)
        mc_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10 + 2)
        train_o, train_f, train_c, train_k = generate_batch(
            scenario, train_rng, display_band, N_TRAIN
        )
        held_o, held_f, held_c, held_k = generate_batch(
            scenario, held_rng, display_band, N_HELDOUT
        )
        mc_o, mc_f = covariance_mc_spectra(mc_rng, train_o, train_f)
        fooof_mc = official_fit(mc_o, display_band)
        split_mc = six.fit_split_batch(mc_o, mc_f, band)
        fooof_held = official_fit(held_o, display_band)
        split_held = six.fit_split_batch(held_o, held_f, band)
        fooof_pred = six.corr(fooof_mc)
        rows.append(
            {
                "scenario": scenario,
                "repeat": repeat,
                "band": band,
                "fooof_pred": fooof_pred,
                "fooof_emp": six.corr(fooof_held),
                "fooof_corrected": six.corrected_corr(fooof_held, fooof_pred),
                "split_pred": six.corr(split_mc),
                "split_emp": six.corr(split_held),
                "fooof_valid": float(np.isfinite(fooof_held).all(axis=1).mean()),
                "split_valid": float(np.isfinite(split_held).all(axis=1).mean()),
                "true_center_knee_rho_train": float(stats.spearmanr(train_c, train_k).statistic)
                if np.std(train_k) > 0 else 0.0,
                "true_center_knee_rho_held": float(stats.spearmanr(held_c, held_k).statistic)
                if np.std(held_k) > 0 else 0.0,
                "train_knee_mean": float(np.mean(train_k)),
                "held_knee_mean": float(np.mean(held_k)),
                "train_knee_min": float(np.min(train_k)),
                "train_knee_max": float(np.max(train_k)),
                "fitted_base_knee_hz": fitted_base_knee,
                "n_train": N_TRAIN,
                "n_heldout": N_HELDOUT,
                "n_mc": N_MC,
            }
        )
    return rows


def reconstruct_group(
    scenario: str, repeat: int, display_band: str
) -> dict[str, object]:
    if scenario == "fixed2":
        install_fixed_base(repeat)
    band_index = list(researcher.BANDS).index(display_band)
    band = display_band.lower()
    seed_offset = {"fixed2": 0, "lognormal40": 2_000_000, "empirical": 4_000_000}
    seed = BASE_SEED + seed_offset[scenario]
    train_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10)
    held_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10 + 1)
    mc_rng = np.random.default_rng(seed + repeat * 100 + band_index * 10 + 2)
    train_o, train_f, train_c, train_k = generate_batch(
        scenario, train_rng, display_band, N_TRAIN
    )
    held_o, held_f, held_c, held_k = generate_batch(
        scenario, held_rng, display_band, N_HELDOUT
    )
    mc_o, mc_f = covariance_mc_spectra(mc_rng, train_o, train_f)
    fooof_mc = official_fit(mc_o, display_band)
    split_mc = six.fit_split_batch(mc_o, mc_f, band)
    fooof = official_fit(held_o, display_band)
    split = six.fit_split_batch(held_o, held_f, band)
    influence = local_influence(train_o, train_c, display_band)
    both_good = np.isfinite(fooof).all(axis=1) & np.isfinite(split).all(axis=1)
    target = np.nanmedian(fooof[both_good, 0])
    valid_indices = np.flatnonzero(both_good)
    component_trial_index = int(
        valid_indices[np.argmin(np.abs(fooof[both_good, 0] - target))]
    )
    fit_components = official_fit_components(held_o[component_trial_index])
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
        "fooof": fooof,
        "split": split,
        "component_trial_index": component_trial_index,
        "fooof_component_frequency": fit_components["frequency"],
        "fooof_source_psd": fit_components["source"],
        "fooof_aperiodic_fit": fit_components["aperiodic"],
        "fooof_periodic_fit": fit_components["periodic"],
        "v_fooof": np.cov(fooof_mc[np.isfinite(fooof_mc).all(axis=1)].T, ddof=1),
        "v_split": np.cov(split_mc[np.isfinite(split_mc).all(axis=1)].T, ddof=1),
    }


def choose_representative(groups: pd.DataFrame) -> int:
    """Choose the high-gamma medoid without optimizing either method's result."""
    high = groups[groups.band == "high gamma"].copy()
    columns = ["fooof_pred", "fooof_emp", "split_pred", "split_emp"]
    values = high[columns]
    center = values.median(axis=0)
    scale = (values - center).abs().median(axis=0).replace(0.0, 1.0)
    high["medoid_distance"] = np.sqrt(
        (((values - center) / scale) ** 2).sum(axis=1)
    )
    return int(high.sort_values("medoid_distance").iloc[0]["repeat"])


def panel_b_group_worker(task: tuple[str, int]) -> dict[str, object]:
    """Return one group's model-selection-aware Monte Carlo transfer terms."""
    scenario, repeat = task
    band_index = list(researcher.BANDS).index("High gamma")
    seed_offset = {"fixed2": 0, "lognormal40": 2_000_000, "empirical": 4_000_000}
    rng = np.random.default_rng(
        BASE_SEED + seed_offset[scenario] + repeat * 100 + band_index * 10
    )
    original, fractal, _, _ = generate_batch(
        scenario, rng, "High gamma", N_TRAIN
    )
    mc_rng = np.random.default_rng(
        BASE_SEED + seed_offset[scenario] + repeat * 100 + band_index * 10 + 2
    )
    mc_o, mc_f = covariance_mc_spectra(mc_rng, original, fractal)
    fooof_values = official_fit(mc_o, "High gamma")
    split_values = six.fit_split_batch(mc_o, mc_f, "high gamma")
    visible = (base.FREQ >= 60.0) & (base.FREQ <= 150.0)

    def normalise_visible(values: np.ndarray) -> np.ndarray:
        scale = np.nanmax(np.abs(values[visible]))
        return values / scale if np.isfinite(scale) and scale > 0 else values

    def regression_influence(design: np.ndarray, target: np.ndarray) -> np.ndarray:
        x = design - design.mean(axis=0, keepdims=True)
        y = target - target.mean()
        gram = x @ x.T
        ridge = max(float(np.trace(gram) / len(gram)) * 1e-8, 1e-12)
        coefficients = np.linalg.solve(gram + ridge * np.eye(len(gram)), y)
        return x.T @ coefficients

    fooof_good = np.isfinite(fooof_values).all(axis=1)
    xo = mc_o[fooof_good] - mc_o[fooof_good].mean(axis=0, keepdims=True)
    fooof_p = fooof_values[fooof_good, 0]
    fooof_ap = fooof_values[fooof_good, 1]
    fooof_jp = regression_influence(xo, fooof_p)
    fooof_ja = regression_influence(xo, fooof_ap)
    fooof_p_hat = xo @ fooof_jp
    fooof_ap_hat = xo @ fooof_ja
    fooof_transfer = xo.T @ fooof_ap_hat / (len(xo) - 1)
    fooof_scale = np.std(fooof_p_hat, ddof=1) * np.std(fooof_ap_hat, ddof=1)

    split_good = np.isfinite(split_values).all(axis=1)
    residual = mc_o[split_good] - mc_f[split_good]
    xr = residual - residual.mean(axis=0, keepdims=True)
    xf = mc_f[split_good] - mc_f[split_good].mean(axis=0, keepdims=True)
    split_p = split_values[split_good, 0]
    split_ap = split_values[split_good, 1]
    split_jp = regression_influence(xr, split_p)
    split_ja = regression_influence(xf, split_ap)
    split_p_hat = xr @ split_jp
    split_ap_hat = xf @ split_ja
    split_transfer = xr.T @ split_ap_hat / (len(xr) - 1)
    split_scale = np.std(split_p_hat, ddof=1) * np.std(split_ap_hat, ddof=1)
    fooof_contribution = fooof_jp * fooof_transfer / max(fooof_scale, 1e-30)
    split_contribution = split_jp * split_transfer / max(split_scale, 1e-30)
    return {
        "repeat": repeat,
        "fooof_jp": normalise_visible(fooof_jp),
        "fooof_ja": normalise_visible(fooof_ja),
        "split_jp": normalise_visible(split_jp),
        "split_ja": normalise_visible(split_ja),
        "fooof_contribution": fooof_contribution,
        "split_contribution": split_contribution,
        "fooof_local_pearson": float(np.corrcoef(fooof_p, fooof_ap)[0, 1]),
        "split_local_pearson": float(np.corrcoef(split_p, split_ap)[0, 1]),
    }


def build_panel_b_average(scenario: str, groups: pd.DataFrame) -> dict[str, object]:
    """Aggregate all 20 high-gamma groups without representative selection."""
    tasks = [(scenario, repeat) for repeat in range(N_GROUPS)]
    with mp.get_context("fork").Pool(4) as pool:
        rows = pool.map(panel_b_group_worker, tasks)
    result: dict[str, object] = {}
    critical = float(stats.t.ppf(0.975, len(rows) - 1))
    for prefix in ["fooof_jp", "fooof_ja", "split_jp", "split_ja",
                   "fooof_contribution", "split_contribution"]:
        values = np.asarray([row[prefix] for row in rows], dtype=float)
        mean = values.mean(axis=0)
        half_width = critical * values.std(axis=0, ddof=1) / np.sqrt(len(values))
        result[f"{prefix}_mean"] = mean
        result[f"{prefix}_lo"] = mean - half_width
        result[f"{prefix}_hi"] = mean + half_width
    high = groups[groups.band == "high gamma"]
    result.update(
        {
            "n_groups": len(rows),
            "fooof_mean_rho": float(high.fooof_pred.mean()),
            "split_mean_rho": float(high.split_pred.mean()),
            "fooof_mean_local_pearson": float(
                np.mean([row["fooof_local_pearson"] for row in rows])
            ),
            "split_mean_local_pearson": float(
                np.mean([row["split_local_pearson"] for row in rows])
            ),
        }
    )
    return result


def relabel_mc_theory(svg_path: Path) -> None:
    """Make the estimator-aware numerical theory explicit in the SVG legend."""
    text = svg_path.read_text(encoding="utf-8")
    text = text.replace("FOOOF theory", "FOOOF covariance theory")
    text = text.replace("SPLIT theory", "SPLIT covariance theory")
    text = text.replace("Theory-predicted error", "Covariance-theory error")
    svg_path.write_text(text, encoding="utf-8")


def save_scenario(scenario: str, groups: pd.DataFrame) -> dict[str, object]:
    out = OUT_ROOT / scenario
    out.mkdir(parents=True, exist_ok=True)
    selected = choose_representative(groups)
    representative = reconstruct_group(scenario, selected, "High gamma")
    panel_b_average = None
    six.OUT = out
    original_svg, metrics = six.make_figure(
        groups,
        representative,
        panel_f_absolute=False,
        panel_f_band=("high gamma" if scenario != "fixed2" else None),
        panel_a_fractal_dashed=(scenario != "fixed2"),
        panel_b_frequency_xlim=((60.0, 150.0) if scenario != "fixed2" else None),
        panel_b_average_summary=panel_b_average,
        panel_a_fooof_components=(scenario == "empirical"),
        panel_d_ylim=None,
        panel_e_ylim=((-0.3, 0.3) if scenario == "empirical" else
                      ((-0.5, 0.5) if scenario == "lognormal40" else (-1.08, 1.10))),
        panel_f_ylim=((-0.5, 0.5) if scenario == "empirical" else None),
    )
    stems = {
        "fixed2": "figure2_official3_knee_fixed_2hz_six_panels",
        "lognormal40": "figure2_official3_knee_lognormal_10_100_mean40_six_panels",
        "empirical": "figure2_official3_empirical_knee_distribution_six_panels",
    }
    stem = stems[scenario]
    svg = out / f"{stem}.svg"
    png = out / f"{stem}_preview.png"
    original_png = out / "figure2_fooof_split_ou_welch_six_panels_preview.png"
    original_svg.replace(svg)
    original_png.replace(png)
    relabel_mc_theory(svg)
    metrics.update(
        {
            "figure": str(svg),
            "preview": str(png),
            "scenario": scenario,
            "fooof": "FOOOF 1.1.1; aperiodic_mode=knee; max_n_peaks=3; peak_width_limits=1.5-6 Hz; min_peak_height=0.6; peak_threshold=2; fit=0.2-197 Hz",
            "theory": "Gaussian propagation of training joint PSD covariance through the official max-three FOOOF/SPLIT estimators; held-out raw time-domain trials are independent",
            "n_groups": N_GROUPS,
            "n_train": N_TRAIN,
            "n_heldout": N_HELDOUT,
            "n_covariance_mc": N_MC,
            "representative_selection": (
                "high-gamma medoid: minimum MAD-standardized Euclidean distance "
                "to the population median of fooof_pred, fooof_emp, split_pred, split_emp"
            ),
            "knee_distribution": (
                {"fixed_hz": 2.0}
                if scenario == "fixed2"
                else ({
                    "family": "log-normal truncated to 10-100 Hz",
                    "log_mu": LOG_KNEE_MU,
                    "log_sigma": LOG_KNEE_SIGMA,
                    "target_truncated_mean_hz": TARGET_KNEE_MEAN,
                    "observed_train_mean_hz": float(groups.train_knee_mean.mean()),
                    "observed_min_hz": float(groups.train_knee_min.min()),
                    "observed_max_hz": float(groups.train_knee_max.max()),
                    "max_abs_true_center_knee_rho": float(
                        groups[["true_center_knee_rho_train", "true_center_knee_rho_held"]]
                        .abs()
                        .to_numpy()
                        .max()
                    ),
                } if scenario == "lognormal40" else {
                    "family": "empirical inverse-ECDF sampling",
                    "source": str(EMPIRICAL_KNEE_FILE),
                    "source_n": int(len(EMPIRICAL_KNEES)) if EMPIRICAL_KNEES is not None else None,
                    "source_mean_hz": float(np.mean(EMPIRICAL_KNEES)) if EMPIRICAL_KNEES is not None else None,
                    "source_median_hz": float(np.median(EMPIRICAL_KNEES)) if EMPIRICAL_KNEES is not None else None,
                    "source_min_hz": float(np.min(EMPIRICAL_KNEES)) if EMPIRICAL_KNEES is not None else None,
                    "source_max_hz": float(np.max(EMPIRICAL_KNEES)) if EMPIRICAL_KNEES is not None else None,
                    "observed_train_mean_hz": float(groups.train_knee_mean.mean()),
                    "observed_min_hz": float(groups.train_knee_min.min()),
                    "observed_max_hz": float(groups.train_knee_max.max()),
                    "max_abs_true_center_knee_rho": float(
                        groups[["true_center_knee_rho_train", "true_center_knee_rho_held"]]
                        .abs().to_numpy().max()
                    ),
                })
            ),
        }
    )
    groups.to_csv(out / "all_group_results.csv", index=False)
    six.aggregate(groups).to_csv(out / "cross_band_summary.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=["fixed2", "lognormal40", "empirical", "all"],
        default="all",
    )
    args = parser.parse_args()
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    fixed.load_sources()
    matched.install_variant()
    official._fooof_group_class()
    if args.scenario in {"empirical", "all"}:
        load_empirical_knees()
    all_metrics: dict[str, object] = {}
    scenarios = (
        ["fixed2", "lognormal40", "empirical"]
        if args.scenario == "all"
        else [args.scenario]
    )
    for scenario in scenarios:
        tasks = [(scenario, repeat) for repeat in range(N_GROUPS)]
        with mp.get_context("fork").Pool(4) as pool:
            nested = pool.map(worker, tasks)
        groups = pd.DataFrame([row for part in nested for row in part])
        checkpoint = OUT_ROOT / scenario
        checkpoint.mkdir(parents=True, exist_ok=True)
        groups.to_csv(checkpoint / "all_group_results_checkpoint.csv", index=False)
        all_metrics[scenario] = save_scenario(scenario, groups)
    (OUT_ROOT / "metrics_both.json").write_text(
        json.dumps(all_metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
