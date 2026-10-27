from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
from scipy import ndimage, stats

from .core import AREAS9, input_psd, load_model
from .io import load_config
from .plot_formatting import format_p_value
from .paper_figure_node_deletion import (
    DT_MS,
    SMOOTHING_SIGMA_HZ,
    _node_deleted_j,
    _style,
)
from .single_node_deletion_all_nodes import _shared_inputs
from .single_node_deletion_revision import _frequency_grid


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "single_node_deletion_revision"
OUTPUT = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "single_node_neighbor_delta"
DEFAULT_DELETED_NODE = "V1"
REFERENCE_OUTPUT_AREA = "V2"
MIN_VALID_TRIALS = 300
PERMUTATIONS = 100_000


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _multioutput_discrete_transfer_power(
    j: np.ndarray,
    input_indices: list[int],
    output_indices: list[int],
    frequencies: np.ndarray,
    tau_e: float,
) -> np.ndarray:
    """Return |H|^2 with shape (input, output, frequency).

    This is the multi-input/multi-output form of the established discrete
    transfer calculation.  One eigendecomposition is reused for every pair.
    """
    eigenvalues, vectors = np.linalg.eig(j)
    discrete = 1.0 + DT_MS * eigenvalues
    fs = 1000.0 / DT_MS
    theta = 2.0 * np.pi * np.asarray(frequencies) / fs
    denominator = 1.0 / (
        1.0 - discrete[None, :] * np.exp(-1j * theta[:, None])
    )
    forcing = np.zeros((len(j), len(input_indices)))
    for column, index in enumerate(input_indices):
        forcing[index, column] = DT_MS / tau_e
    coefficients = np.linalg.solve(vectors, forcing)
    readout = vectors[np.asarray(output_indices), :]
    power = np.empty(
        (len(input_indices), len(output_indices), len(frequencies)), dtype=np.float32
    )
    for input_pos in range(len(input_indices)):
        response = (denominator * coefficients[:, input_pos][None, :]) @ readout.T
        power[input_pos] = np.maximum(
            np.abs(response.T) ** 2, np.finfo(np.float32).tiny
        ).astype(np.float32)
    return power


def _fast_features_from_psd_components(
    input_psds: np.ndarray,
    transfer_power: np.ndarray,
    frequencies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract P and A for all input/output/trial combinations.

    The established pipeline smooths log(input PSD * transfer power).  Because
    smoothing and the log-slope fit are linear in log power, input and transfer
    components can be processed separately.  Peak locations use local-grid
    maxima with quadratic sub-bin interpolation, avoiding a several-gigabyte
    materialized PSD tensor while preserving the same feature definition.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    spacing = np.diff(frequencies)
    if not np.allclose(spacing, spacing[0], rtol=1e-7, atol=1e-10):
        raise ValueError("The fast feature extractor requires a uniform frequency grid.")
    sigma_bins = SMOOTHING_SIGMA_HZ / spacing[0]
    positive_floor = np.finfo(np.float32).tiny
    smooth_input = ndimage.gaussian_filter1d(
        np.log(np.maximum(input_psds, positive_floor)), sigma_bins, axis=-1
    )
    smooth_transfer = ndimage.gaussian_filter1d(
        np.log(np.maximum(transfer_power, positive_floor)), sigma_bins, axis=-1
    )

    slope_mask = (frequencies >= 100.0) & (frequencies <= 1000.0)
    x = np.log(frequencies[slope_mask])
    weights = (x - x.mean()) / np.sum((x - x.mean()) ** 2)
    input_slope = smooth_input[:, slope_mask] @ weights
    transfer_slope = np.tensordot(
        smooth_transfer[..., slope_mask], weights, axes=([-1], [0])
    )

    peak_mask = (frequencies >= 5.0) & (frequencies <= 20.0)
    peak_f = frequencies[peak_mask]
    input_peak = smooth_input[:, peak_mask]
    transfer_peak = smooth_transfer[..., peak_mask]
    n_inputs, n_outputs = transfer_power.shape[:2]
    n_trials = len(input_psds)
    features = np.empty((n_inputs, n_outputs, n_trials, 2), dtype=np.float32)
    valid = np.zeros((n_inputs, n_outputs, n_trials), dtype=bool)

    for input_pos in range(n_inputs):
        combined = input_peak[:, None, :] + transfer_peak[input_pos][None, :, :]
        local = (combined[..., 1:-1] > combined[..., :-2]) & (
            combined[..., 1:-1] > combined[..., 2:]
        )
        candidates = np.where(local, combined[..., 1:-1], -np.inf)
        best_rel = np.argmax(candidates, axis=-1)
        has_local = np.any(local, axis=-1)
        best_index = best_rel + 1
        gather = best_index[..., None]
        y_mid = np.take_along_axis(combined, gather, axis=-1)[..., 0]
        y_left = np.take_along_axis(combined, gather - 1, axis=-1)[..., 0]
        y_right = np.take_along_axis(combined, gather + 1, axis=-1)[..., 0]
        curvature = y_left - 2.0 * y_mid + y_right
        offset = np.zeros_like(y_mid)
        curved = np.abs(curvature) > 1e-12
        offset[curved] = 0.5 * (y_left[curved] - y_right[curved]) / curvature[curved]
        offset = np.clip(offset, -1.0, 1.0)
        peak_hz = peak_f[best_index] + offset * spacing[0]

        if candidates.shape[-1] >= 2:
            second = np.partition(candidates, -2, axis=-1)[..., -2]
        else:
            second = np.full_like(y_mid, -np.inf)
        competing_peak = np.isfinite(second) & ((y_mid - second) < np.log(1.20))
        is_valid = has_local & ~competing_peak & np.isfinite(peak_hz)

        features[input_pos, :, :, 0] = (2.0 * np.pi * peak_hz).T
        features[input_pos, :, :, 1] = -(
            input_slope[None, :] + transfer_slope[input_pos, :, None]
        )
        valid[input_pos] = is_valid.T & np.isfinite(features[input_pos]).all(axis=-1)
    return features, valid


def _corr(features: np.ndarray) -> float:
    if len(features) < 3:
        return np.nan
    return float(stats.pearsonr(features[:, 0], features[:, 1]).statistic)


def _adjacency_fields(model, deleted_idx: int, output_idx: int) -> dict[str, object]:
    deleted_to_output = bool(model.fln[output_idx, deleted_idx] > 0)
    output_to_deleted = bool(model.fln[deleted_idx, output_idx] > 0)
    return {
        "deleted_to_output": deleted_to_output,
        "output_to_deleted": output_to_deleted,
        "any_direct": deleted_to_output or output_to_deleted,
        "deleted_to_output_fln": float(model.fln[output_idx, deleted_idx]),
        "output_to_deleted_fln": float(model.fln[deleted_idx, output_idx]),
    }


def _condition_rows(
    model,
    deleted_node: str,
    input_areas: list[str],
    output_areas: list[str],
    feature_sets: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    deleted_idx = model.areas.index(deleted_node)
    rows: list[dict[str, object]] = []
    for input_pos, input_area in enumerate(input_areas):
        for output_pos, output_area in enumerate(output_areas):
            row: dict[str, object] = {
                "deleted_node": deleted_node,
                "input_area": input_area,
                "output_area": output_area,
                "output_is_input_area": output_area == input_area,
                **_adjacency_fields(model, deleted_idx, model.areas.index(output_area)),
            }
            for source, (baseline, baseline_valid, deleted, deleted_valid) in feature_sets.items():
                use = baseline_valid[input_pos, output_pos] & deleted_valid[input_pos, output_pos]
                use &= np.isfinite(baseline[input_pos, output_pos]).all(axis=1)
                use &= np.isfinite(deleted[input_pos, output_pos]).all(axis=1)
                n_valid = int(use.sum())
                prefix = "simulation" if source == "heldout" else source
                row[f"{prefix}_n_valid"] = n_valid
                if n_valid >= MIN_VALID_TRIALS:
                    baseline_r = _corr(baseline[input_pos, output_pos, use])
                    deleted_r = _corr(deleted[input_pos, output_pos, use])
                    row[f"{prefix}_baseline_r"] = baseline_r
                    row[f"{prefix}_deleted_r"] = deleted_r
                    row[f"{prefix}_delta_r"] = deleted_r - baseline_r
                else:
                    row[f"{prefix}_baseline_r"] = np.nan
                    row[f"{prefix}_deleted_r"] = np.nan
                    row[f"{prefix}_delta_r"] = np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def _permutation_pvalue(
    direct: np.ndarray, indirect: np.ndarray, rng: np.random.Generator
) -> float:
    values = np.r_[direct, indirect]
    n_direct = len(direct)
    observed = float(np.mean(direct) - np.mean(indirect))
    exceed = 0
    for _ in range(PERMUTATIONS):
        order = rng.permutation(len(values))
        difference = values[order[:n_direct]].mean() - values[order[n_direct:]].mean()
        exceed += abs(difference) >= abs(observed)
    return float((exceed + 1) / (PERMUTATIONS + 1))


def _bootstrap_difference(
    direct: np.ndarray, indirect: np.ndarray, rng: np.random.Generator, n: int = 20_000
) -> tuple[float, float]:
    differences = np.empty(n)
    for pos in range(n):
        differences[pos] = (
            rng.choice(direct, len(direct), replace=True).mean()
            - rng.choice(indirect, len(indirect), replace=True).mean()
        )
    low, high = np.percentile(differences, [2.5, 97.5])
    return float(low), float(high)


def _exact_sign_flip_pvalue(values: np.ndarray) -> float:
    observed = abs(float(np.mean(values)))
    statistics = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(values)):
        statistics.append(abs(float(np.mean(values * np.asarray(signs)))))
    statistics = np.asarray(statistics)
    return float(np.mean(statistics >= observed - 1e-15))


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def group_statistics(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2026081901)
    statistic_rows: list[dict[str, object]] = []
    block_rows: list[dict[str, object]] = []
    adjacency_columns = ("deleted_to_output", "any_direct", "output_to_deleted")
    sources = ("deterministic", "calibration", "simulation")
    scopes = {
        "all_outputs": results,
        "exclude_input_readout": results[~results.output_is_input_area],
        # The input covariance was originally selected using A6Va -> V2.  This
        # sensitivity analysis removes V2 entirely, preventing that reference
        # readout from driving the direct-versus-indirect result.
        "exclude_original_selected_output": results[
            results.output_area != REFERENCE_OUTPUT_AREA
        ],
    }

    for scope, scoped in scopes.items():
        for adjacency in adjacency_columns:
            for source in sources:
                delta_column = f"{source}_delta_r"
                for outcome in ("absolute", "signed"):
                    data = scoped.dropna(subset=[delta_column]).copy()
                    data["outcome"] = (
                        data[delta_column].abs() if outcome == "absolute" else data[delta_column]
                    )
                    output_level = data.groupby("output_area", as_index=False).agg(
                        outcome=("outcome", "mean"), direct=(adjacency, "first")
                    )
                    direct = output_level.loc[output_level.direct, "outcome"].to_numpy(float)
                    indirect = output_level.loc[~output_level.direct, "outcome"].to_numpy(float)
                    if len(direct) < 3 or len(indirect) < 3:
                        continue
                    difference = float(direct.mean() - indirect.mean())
                    ci_low, ci_high = _bootstrap_difference(direct, indirect, rng)
                    welch = stats.ttest_ind(direct, indirect, equal_var=False)
                    mann_whitney = stats.mannwhitneyu(direct, indirect, alternative="two-sided")

                    input_differences = []
                    for input_area, part in data.groupby("input_area"):
                        group_direct = part.loc[part[adjacency], "outcome"].to_numpy(float)
                        group_indirect = part.loc[~part[adjacency], "outcome"].to_numpy(float)
                        if len(group_direct) and len(group_indirect):
                            input_difference = float(group_direct.mean() - group_indirect.mean())
                            input_differences.append(input_difference)
                            block_rows.append(
                                {
                                    "scope": scope,
                                    "adjacency": adjacency,
                                    "source": source,
                                    "outcome": outcome,
                                    "input_area": input_area,
                                    "direct_mean": float(group_direct.mean()),
                                    "indirect_mean": float(group_indirect.mean()),
                                    "difference": input_difference,
                                    "n_direct": int(len(group_direct)),
                                    "n_indirect": int(len(group_indirect)),
                                }
                            )
                    input_differences = np.asarray(input_differences)
                    sign_flip_p = (
                        _exact_sign_flip_pvalue(input_differences)
                        if len(input_differences) >= 3
                        else np.nan
                    )
                    statistic_rows.append(
                        {
                            "scope": scope,
                            "adjacency": adjacency,
                            "source": source,
                            "outcome": outcome,
                            "n_output_units_direct": int(len(direct)),
                            "n_output_units_indirect": int(len(indirect)),
                            "direct_mean": float(direct.mean()),
                            "indirect_mean": float(indirect.mean()),
                            "mean_difference": difference,
                            "mean_ratio": float(direct.mean() / indirect.mean())
                            if indirect.mean() != 0
                            else np.nan,
                            "difference_bootstrap_ci_low": ci_low,
                            "difference_bootstrap_ci_high": ci_high,
                            "output_label_permutation_p": _permutation_pvalue(
                                direct, indirect, rng
                            ),
                            "welch_t_p": float(welch.pvalue),
                            "mann_whitney_p": float(mann_whitney.pvalue),
                            "n_input_blocks": int(len(input_differences)),
                            "input_block_mean_difference": float(input_differences.mean()),
                            "input_block_sign_flip_p": sign_flip_p,
                        }
                    )
    statistics_frame = pd.DataFrame(statistic_rows)
    statistics_frame["permutation_fdr_q"] = _benjamini_hochberg(
        statistics_frame.output_label_permutation_p.to_numpy(float)
    )
    return statistics_frame, pd.DataFrame(block_rows)


def make_figure(
    results: pd.DataFrame, statistics_frame: pd.DataFrame, block_frame: pd.DataFrame, output: Path
) -> None:
    deleted_node = str(results.deleted_node.iloc[0])
    primary = statistics_frame[
        (statistics_frame.scope == "all_outputs")
        & (statistics_frame.adjacency == "deleted_to_output")
        & (statistics_frame.source == "simulation")
        & (statistics_frame.outcome == "absolute")
    ].iloc[0]
    valid = results.dropna(subset=["simulation_delta_r"]).copy()
    output_level = valid.groupby("output_area", as_index=False).agg(
        mean_abs_delta_r=("simulation_delta_r", lambda x: float(np.mean(np.abs(x)))),
        direct=("deleted_to_output", "first"),
    )
    direct = output_level[output_level.direct]
    indirect = output_level[~output_level.direct]

    fig = plt.figure(figsize=(18.0, 12.0), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 1.2))
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    groups = [direct.mean_abs_delta_r.to_numpy(), indirect.mean_abs_delta_r.to_numpy()]
    boxes = ax_a.boxplot(groups, positions=[0, 1], widths=0.52, patch_artist=True, showfliers=False)
    for patch, color in zip(boxes["boxes"], ("#D1495B", "#247BA0")):
        patch.set_facecolor(color)
        patch.set_alpha(0.28)
        patch.set_linewidth(2.0)
    jitter_rng = np.random.default_rng(11)
    for pos, (values, color) in enumerate(zip(groups, ("#D1495B", "#247BA0"))):
        ax_a.scatter(
            pos + jitter_rng.normal(0, 0.055, len(values)), values,
            s=60, color=color, edgecolor="white", linewidth=0.7, alpha=0.9, zorder=3,
        )
    ax_a.set_xticks([0, 1], ["Direct receiver", "Not direct"])
    ax_a.set_ylabel(r"Mean across inputs, $|\Delta r|$")
    ax_a.set_title("Output-area-level comparison")
    ax_a.text(
        0.03, 0.97,
        (
            f"mean difference={primary.mean_difference:.4f}\n"
            f"95% CI [{primary.difference_bootstrap_ci_low:.4f}, "
            f"{primary.difference_bootstrap_ci_high:.4f}]\n"
            f"permutation p={format_p_value(primary.output_label_permutation_p)}"
        ),
        transform=ax_a.transAxes, va="top",
    )

    primary_blocks = block_frame[
        (block_frame.scope == "all_outputs")
        & (block_frame.adjacency == "deleted_to_output")
        & (block_frame.source == "simulation")
        & (block_frame.outcome == "absolute")
    ].sort_values("difference")
    y_positions = np.arange(len(primary_blocks))
    colors = np.where(primary_blocks.difference >= 0, "#D1495B", "#247BA0")
    ax_b.barh(
        y_positions, primary_blocks.difference, color=colors, alpha=0.86,
        edgecolor="white", linewidth=0.8,
    )
    ax_b.axvline(0, color="0.2", lw=1.5)
    ax_b.set_yticks(y_positions, primary_blocks.input_area)
    ax_b.set_xlabel(r"Direct minus non-direct mean $|\Delta r|$")
    ax_b.set_ylabel("Input area")
    ax_b.set_title("Input-area-blocked differences")
    ax_b.text(
        0.03, 0.97,
        f"exact sign-flip p={format_p_value(primary.input_block_sign_flip_p)}",
        transform=ax_b.transAxes, va="top",
    )

    output_order = output_level.sort_values(
        ["direct", "mean_abs_delta_r"], ascending=[False, False]
    ).output_area.tolist()
    input_order = [area for area in AREAS9 if area in valid.input_area.unique()]
    matrix = valid.pivot(index="input_area", columns="output_area", values="simulation_delta_r").reindex(
        index=input_order, columns=output_order
    )
    limit = float(np.nanpercentile(np.abs(matrix.to_numpy()), 98))
    image = ax_c.imshow(
        matrix.to_numpy(), aspect="auto", cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit), interpolation="nearest",
    )
    n_direct = int(output_level.direct.sum())
    ax_c.axvline(n_direct - 0.5, color="black", lw=2.0)
    ax_c.set_xticks(np.arange(len(output_order)), output_order, rotation=90, fontsize=8)
    ax_c.set_yticks(np.arange(len(input_order)), input_order)
    ax_c.set_xlabel(f"Output area (direct {deleted_node} receivers first)")
    ax_c.set_ylabel("Input area")
    ax_c.set_title(r"Held-out simulation $\Delta r = r_{deleted}-r_{baseline}$")
    ax_c.text((n_direct - 1) / 2, -1.0, "Direct receivers", ha="center", va="bottom")
    ax_c.text((n_direct + len(output_order) - 1) / 2, -1.0, "Not direct", ha="center", va="bottom")
    colorbar = fig.colorbar(image, ax=ax_c, shrink=0.72, pad=0.015)
    colorbar.set_label(r"Simulation $\Delta r$")

    for label, ax in zip("ABC", (ax_a, ax_b, ax_c)):
        ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontsize=22, fontweight="bold")
    _save(fig, output / "direct_vs_indirect_output_delta_r")


def run(deleted_node: str, input_areas: list[str], force: bool = False) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml")
    model = load_model(ROOT / cfg["model_dir"])
    if deleted_node not in model.areas:
        raise ValueError(f"Unknown deleted node: {deleted_node}")
    input_areas = [area for area in input_areas if area != deleted_node]
    unknown = sorted(set(input_areas) - set(model.areas))
    if unknown:
        raise ValueError(f"Unknown input areas: {unknown}")
    if not input_areas:
        raise ValueError("At least one input area distinct from the deleted node is required.")
    output_areas = [area for area in model.areas if area != deleted_node]

    selected = json.loads((SOURCE / "selected_single_node_observation_matched.json").read_text())
    frequencies, _, _ = _frequency_grid()
    z, _, calibration_input, heldout_input = _shared_inputs(selected, frequencies)
    shared_cache = (
        ROOT / "experiments" / "pap_theory_validation" / "outputs"
        / "single_node_deletion_all_nodes" / "shared_longwindow_input_psd_means.npz"
    )
    if shared_cache.exists():
        cached_frequencies = np.load(shared_cache)["frequencies"]
        if not np.array_equal(cached_frequencies, frequencies):
            raise RuntimeError("Shared input-PSD cache uses a different frequency grid.")
    deterministic_input = np.asarray(
        [input_psd(frequencies, trial) for trial in z], dtype=np.float32
    )

    cache_tag = f"delete_{deleted_node}_inputs_{'-'.join(input_areas)}"
    transfer_cache = OUTPUT / f"{cache_tag}_transfer_power.npz"
    if transfer_cache.exists() and not force:
        cached = np.load(transfer_cache)
        baseline_transfer = cached["baseline"]
        deleted_transfer = cached["deleted"]
    else:
        input_indices = [model.areas.index(area) for area in input_areas]
        output_indices = [model.areas.index(area) for area in output_areas]
        tau_e = float(model.params["tau_exc"])
        deleted_j, _ = _node_deleted_j(model, [deleted_node])
        baseline_transfer = _multioutput_discrete_transfer_power(
            model.J, input_indices, output_indices, frequencies, tau_e
        )
        deleted_transfer = _multioutput_discrete_transfer_power(
            deleted_j, input_indices, output_indices, frequencies, tau_e
        )
        np.savez_compressed(
            transfer_cache,
            frequencies=frequencies,
            input_areas=np.asarray(input_areas),
            output_areas=np.asarray(output_areas),
            baseline=baseline_transfer,
            deleted=deleted_transfer,
        )

    feature_cache = OUTPUT / f"{cache_tag}_features.npz"
    if feature_cache.exists() and not force:
        cached = np.load(feature_cache)
        feature_sets = {
            name: (
                cached[f"{name}_baseline"], cached[f"{name}_baseline_valid"],
                cached[f"{name}_deleted"], cached[f"{name}_deleted_valid"],
            )
            for name in ("deterministic", "calibration", "heldout")
        }
    else:
        feature_sets = {}
        for name, input_values in (
            ("deterministic", deterministic_input),
            ("calibration", calibration_input),
            ("heldout", heldout_input),
        ):
            baseline_features, baseline_valid = _fast_features_from_psd_components(
                input_values, baseline_transfer, frequencies
            )
            deleted_features, deleted_valid = _fast_features_from_psd_components(
                input_values, deleted_transfer, frequencies
            )
            feature_sets[name] = (
                baseline_features, baseline_valid, deleted_features, deleted_valid
            )
        payload = {}
        for name, values in feature_sets.items():
            payload[f"{name}_baseline"] = values[0]
            payload[f"{name}_baseline_valid"] = values[1]
            payload[f"{name}_deleted"] = values[2]
            payload[f"{name}_deleted_valid"] = values[3]
        np.savez_compressed(feature_cache, **payload)

    results = _condition_rows(
        model, deleted_node, input_areas, output_areas, feature_sets
    )
    results.to_csv(OUTPUT / "input_output_delta_r_results.csv", index=False)
    statistics_frame, block_frame = group_statistics(results)
    statistics_frame.to_csv(OUTPUT / "direct_vs_indirect_statistics.csv", index=False)
    block_frame.to_csv(OUTPUT / "input_block_differences.csv", index=False)
    make_figure(results, statistics_frame, block_frame, OUTPUT)

    primary = statistics_frame[
        (statistics_frame.scope == "all_outputs")
        & (statistics_frame.adjacency == "deleted_to_output")
        & (statistics_frame.source == "simulation")
        & (statistics_frame.outcome == "absolute")
    ].iloc[0].to_dict()
    reference_output_sensitivity = statistics_frame[
        (statistics_frame.scope == "exclude_original_selected_output")
        & (statistics_frame.adjacency == "deleted_to_output")
        & (statistics_frame.source == "simulation")
        & (statistics_frame.outcome == "absolute")
    ].iloc[0].to_dict()
    summary = {
        "setup": {
            "deleted_node": deleted_node,
            "input_areas": input_areas,
            "output_areas": output_areas,
            "n_input_areas": len(input_areas),
            "n_output_areas": len(output_areas),
            "n_input_output_rows": int(len(results)),
            "n_input_conditions": int(len(z)),
            "primary_adjacency": "deleted_to_output: FLN[output, deleted] > 0",
            "primary_outcome": "absolute held-out simulation delta r",
            "statistical_unit": "output area after averaging across input areas",
            "minimum_common_valid_trials": MIN_VALID_TRIALS,
            "permutations": PERMUTATIONS,
        },
        "primary_result": primary,
        "robustness_excluding_original_selected_output": {
            "excluded_output": REFERENCE_OUTPUT_AREA,
            **reference_output_sensitivity,
        },
        "files": {
            "condition_results": str((OUTPUT / "input_output_delta_r_results.csv").resolve()),
            "statistics": str((OUTPUT / "direct_vs_indirect_statistics.csv").resolve()),
            "input_blocks": str((OUTPUT / "input_block_differences.csv").resolve()),
            "figure_png": str((OUTPUT / "direct_vs_indirect_output_delta_r.png").resolve()),
            "figure_pdf": str((OUTPUT / "direct_vs_indirect_output_delta_r.pdf").resolve()),
        },
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare output delta-r for areas directly versus indirectly connected to one deleted node."
    )
    parser.add_argument("--deleted-node", default=DEFAULT_DELETED_NODE)
    parser.add_argument(
        "--input-areas",
        default=",".join(AREAS9),
        help="Comma-separated input areas; the deleted node is automatically omitted.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _style()
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 16, "axes.titlesize": 17})
    input_areas = [item.strip() for item in args.input_areas.split(",") if item.strip()]
    summary = run(args.deleted_node, input_areas, force=args.force)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
