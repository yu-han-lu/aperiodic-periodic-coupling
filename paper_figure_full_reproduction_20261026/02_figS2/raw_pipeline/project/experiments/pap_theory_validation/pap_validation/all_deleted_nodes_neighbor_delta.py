from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import AREAS9, input_psd, load_model
from .io import load_config
from .paper_figure_node_deletion import _node_deleted_j, _style
from .single_node_deletion_all_nodes import _shared_inputs
from .single_node_deletion_revision import _frequency_grid
from .single_node_neighbor_delta import (
    MIN_VALID_TRIALS,
    REFERENCE_OUTPUT_AREA,
    SOURCE,
    _benjamini_hochberg,
    _condition_rows,
    _fast_features_from_psd_components,
    _multioutput_discrete_transfer_power,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    ROOT / "experiments" / "pap_theory_validation" / "outputs"
    / "all_deleted_nodes_neighbor_delta"
)
PERMUTATIONS = 100_000
BOOTSTRAPS = 50_000


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _feature_payload(
    input_psds: dict[str, np.ndarray],
    transfer: np.ndarray,
    frequencies: np.ndarray,
) -> dict[str, np.ndarray]:
    payload: dict[str, np.ndarray] = {}
    for source, values in input_psds.items():
        features, valid = _fast_features_from_psd_components(values, transfer, frequencies)
        payload[f"{source}_features"] = features
        payload[f"{source}_valid"] = valid
    return payload


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    cached = np.load(path)
    return {name: cached[name] for name in cached.files}


def run_all_deletions(model, selected: dict, force: bool = False) -> pd.DataFrame:
    result_path = OUTPUT / "all_deleted_nodes_input_output_delta_r.csv.gz"
    cache_dir = OUTPUT / "feature_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    frequencies, _, _ = _frequency_grid()
    z, _, calibration_input, heldout_input = _shared_inputs(selected, frequencies)
    input_psds = {
        "deterministic": np.asarray(
            [input_psd(frequencies, trial) for trial in z], dtype=np.float32
        ),
        "calibration": calibration_input,
        "heldout": heldout_input,
    }
    all_input_indices = [model.areas.index(area) for area in AREAS9]
    all_output_indices = list(range(len(model.areas)))
    tau_e = float(model.params["tau_exc"])

    baseline_cache = cache_dir / "baseline_all_inputs_all_outputs_features.npz"
    if baseline_cache.exists() and not force:
        baseline_payload = _load_payload(baseline_cache)
    else:
        print("building shared baseline transfer/features", flush=True)
        baseline_transfer = _multioutput_discrete_transfer_power(
            model.J, all_input_indices, all_output_indices, frequencies, tau_e
        )
        baseline_payload = _feature_payload(input_psds, baseline_transfer, frequencies)
        np.savez_compressed(baseline_cache, **baseline_payload)

    completed_rows: list[pd.DataFrame] = []
    for deleted_pos, deleted_node in enumerate(model.areas):
        deleted_cache = cache_dir / f"delete_{deleted_node}_all_inputs_all_outputs_features.npz"
        if deleted_cache.exists() and not force:
            deleted_payload = _load_payload(deleted_cache)
        else:
            deleted_j, _ = _node_deleted_j(model, [deleted_node])
            deleted_transfer = _multioutput_discrete_transfer_power(
                deleted_j, all_input_indices, all_output_indices, frequencies, tau_e
            )
            deleted_payload = _feature_payload(input_psds, deleted_transfer, frequencies)
            np.savez_compressed(deleted_cache, **deleted_payload)

        input_areas = [area for area in AREAS9 if area != deleted_node]
        output_areas = [area for area in model.areas if area != deleted_node]
        input_positions = np.asarray([AREAS9.index(area) for area in input_areas])
        output_positions = np.asarray([model.areas.index(area) for area in output_areas])
        feature_sets = {}
        for source in ("deterministic", "calibration", "heldout"):
            baseline_features = baseline_payload[f"{source}_features"][input_positions][
                :, output_positions
            ]
            baseline_valid = baseline_payload[f"{source}_valid"][input_positions][
                :, output_positions
            ]
            deleted_features = deleted_payload[f"{source}_features"][input_positions][
                :, output_positions
            ]
            deleted_valid = deleted_payload[f"{source}_valid"][input_positions][
                :, output_positions
            ]
            feature_sets[source] = (
                baseline_features, baseline_valid, deleted_features, deleted_valid
            )
        completed_rows.append(
            _condition_rows(
                model, deleted_node, input_areas, output_areas, feature_sets
            )
        )
        combined = pd.concat(completed_rows, ignore_index=True)
        combined.to_csv(result_path, index=False, compression="gzip")
        print(
            f"all-node neighbor simulation {deleted_pos + 1}/{len(model.areas)}: "
            f"delete {deleted_node}; rows={len(combined)}",
            flush=True,
        )
    return pd.concat(completed_rows, ignore_index=True)


def _sign_flip_pvalue(values: np.ndarray, rng: np.random.Generator) -> float:
    values = np.asarray(values, dtype=float)
    observed = abs(float(values.mean()))
    exceed = 0
    chunk_size = 2_000
    completed = 0
    while completed < PERMUTATIONS:
        size = min(chunk_size, PERMUTATIONS - completed)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        permuted = np.abs(signs @ values / len(values))
        exceed += int(np.sum(permuted >= observed))
        completed += size
    return float((exceed + 1) / (PERMUTATIONS + 1))


def _bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    means = np.empty(BOOTSTRAPS)
    chunk_size = 2_000
    completed = 0
    while completed < BOOTSTRAPS:
        size = min(chunk_size, BOOTSTRAPS - completed)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        means[completed : completed + size] = values[indices].mean(axis=1)
        completed += size
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def pooled_statistics(
    results: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(2026081902)
    statistic_rows: list[dict[str, object]] = []
    node_rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    scopes = {
        "all_outputs": results,
        "exclude_original_selected_output": results[
            results.output_area != REFERENCE_OUTPUT_AREA
        ],
    }

    for scope, scoped in scopes.items():
        for adjacency in ("deleted_to_output", "any_direct", "output_to_deleted"):
            for source in ("deterministic", "calibration", "simulation"):
                delta_column = f"{source}_delta_r"
                for outcome in ("absolute", "signed"):
                    data = scoped.dropna(subset=[delta_column]).copy()
                    data["outcome"] = (
                        data[delta_column].abs()
                        if outcome == "absolute"
                        else data[delta_column]
                    )
                    scenario_node_rows = []
                    for deleted_node, deletion in data.groupby("deleted_node", sort=False):
                        output_level = deletion.groupby("output_area", as_index=False).agg(
                            value=("outcome", "mean"), direct=(adjacency, "first")
                        )
                        direct = output_level.loc[output_level.direct, "value"].to_numpy(float)
                        indirect = output_level.loc[~output_level.direct, "value"].to_numpy(float)
                        if not len(direct) or not len(indirect):
                            continue
                        row = {
                            "scope": scope,
                            "adjacency": adjacency,
                            "source": source,
                            "outcome": outcome,
                            "deleted_node": deleted_node,
                            "n_direct_outputs": int(len(direct)),
                            "n_indirect_outputs": int(len(indirect)),
                            "direct_mean": float(direct.mean()),
                            "indirect_mean": float(indirect.mean()),
                            "difference": float(direct.mean() - indirect.mean()),
                            "ratio": float(direct.mean() / indirect.mean())
                            if indirect.mean() != 0
                            else np.nan,
                        }
                        scenario_node_rows.append(row)
                        node_rows.append(row)

                        for input_area, input_part in deletion.groupby("input_area"):
                            direct_input = input_part.loc[
                                input_part[adjacency], "outcome"
                            ].to_numpy(float)
                            indirect_input = input_part.loc[
                                ~input_part[adjacency], "outcome"
                            ].to_numpy(float)
                            if len(direct_input) and len(indirect_input):
                                input_rows.append(
                                    {
                                        "scope": scope,
                                        "adjacency": adjacency,
                                        "source": source,
                                        "outcome": outcome,
                                        "deleted_node": deleted_node,
                                        "input_area": input_area,
                                        "direct_mean": float(direct_input.mean()),
                                        "indirect_mean": float(indirect_input.mean()),
                                        "difference": float(
                                            direct_input.mean() - indirect_input.mean()
                                        ),
                                    }
                                )

                    node_frame = pd.DataFrame(scenario_node_rows)
                    differences = node_frame.difference.to_numpy(float)
                    if len(differences) < 3:
                        continue
                    ci_low, ci_high = _bootstrap_mean_ci(differences, rng)
                    wilcoxon = stats.wilcoxon(differences, alternative="two-sided")
                    one_sample_t = stats.ttest_1samp(differences, 0.0)
                    positive = int(np.sum(differences > 0))
                    binomial = stats.binomtest(
                        positive, len(differences), p=0.5, alternative="two-sided"
                    )
                    statistic_rows.append(
                        {
                            "scope": scope,
                            "adjacency": adjacency,
                            "source": source,
                            "outcome": outcome,
                            "n_deleted_node_units": int(len(node_frame)),
                            "direct_mean_equal_node_weight": float(
                                node_frame.direct_mean.mean()
                            ),
                            "indirect_mean_equal_node_weight": float(
                                node_frame.indirect_mean.mean()
                            ),
                            "mean_difference": float(differences.mean()),
                            "median_difference": float(np.median(differences)),
                            "mean_difference_bootstrap_ci_low": ci_low,
                            "mean_difference_bootstrap_ci_high": ci_high,
                            "mean_ratio_equal_node_weight": float(
                                node_frame.direct_mean.mean()
                                / node_frame.indirect_mean.mean()
                            ),
                            "sign_flip_p": _sign_flip_pvalue(differences, rng),
                            "wilcoxon_p": float(wilcoxon.pvalue),
                            "one_sample_t_p": float(one_sample_t.pvalue),
                            "n_positive_deleted_nodes": positive,
                            "fraction_positive_deleted_nodes": float(
                                positive / len(differences)
                            ),
                            "binomial_sign_p": float(binomial.pvalue),
                        }
                    )
    statistics_frame = pd.DataFrame(statistic_rows)
    statistics_frame["sign_flip_fdr_q"] = _benjamini_hochberg(
        statistics_frame.sign_flip_p.to_numpy(float)
    )
    return statistics_frame, pd.DataFrame(node_rows), pd.DataFrame(input_rows)


def _scientific_3(value: float) -> str:
    """Format a value in scientific notation with three significant digits."""
    mantissa, exponent = f"{float(value):.2e}".split("e")
    return rf"${mantissa}\times10^{{{int(exponent)}}}$"


def make_figure(results: pd.DataFrame, statistics_frame: pd.DataFrame) -> None:
    selector = (
        (statistics_frame.scope == "all_outputs")
        & (statistics_frame.adjacency == "deleted_to_output")
        & (statistics_frame.source == "simulation")
        & (statistics_frame.outcome == "absolute")
    )
    primary = statistics_frame[selector].iloc[0]
    valid = results.dropna(subset=["simulation_delta_r"]).copy()
    valid["abs_delta_r"] = valid.simulation_delta_r.abs()
    output_units = valid.groupby(["deleted_node", "output_area"], as_index=False).agg(
        mean_abs_delta_r=("abs_delta_r", "mean"),
        direct=("deleted_to_output", "first"),
    )
    direct = output_units.loc[output_units.direct, "mean_abs_delta_r"].to_numpy(float)
    indirect = output_units.loc[~output_units.direct, "mean_abs_delta_r"].to_numpy(float)

    fig, ax = plt.subplots(figsize=(9.2, 8.2), constrained_layout=True)

    groups = [direct[direct > 0], indirect[indirect > 0]]
    boxes = ax.boxplot(
        groups,
        positions=[0, 1],
        widths=0.54,
        patch_artist=True,
        showfliers=False,
        medianprops={"linewidth": 3.2, "color": "#F28E2B"},
        whiskerprops={"linewidth": 2.5, "color": "black"},
        capprops={"linewidth": 2.5, "color": "black"},
    )
    for patch, color in zip(boxes["boxes"], ("#D1495B", "#247BA0")):
        patch.set_facecolor(color)
        patch.set_alpha(0.30)
        patch.set_linewidth(2.8)
    jitter_rng = np.random.default_rng(12)
    for pos, (values, color) in enumerate(zip(groups, ("#D1495B", "#247BA0"))):
        ax.scatter(
            pos + jitter_rng.normal(0, 0.07, len(values)), values,
            s=17, color=color, alpha=0.25, linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xticks([0, 1], ["Direct receiver", "Not direct"])
    ax.set_ylabel(r"Across-input mean $|\Delta r|$ (log scale)", fontsize=21, fontweight="bold")
    ax.set_title(
        "Direct vs. non-direct outputs across all node deletions",
        fontsize=22,
        fontweight="bold",
        pad=16,
    )
    ax.text(
        0.035, 0.965,
        (
            f"Deleted nodes: {primary.n_deleted_node_units:.0f}\n"
            f"Direct mean: {_scientific_3(primary.direct_mean_equal_node_weight)}\n"
            f"Non-direct mean: {_scientific_3(primary.indirect_mean_equal_node_weight)}\n"
            f"Difference: {_scientific_3(primary.mean_difference)}\n"
            f"Sign-flip $p$: {_scientific_3(primary.sign_flip_p)}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=17,
        fontweight="bold",
        linespacing=1.35,
    )
    ax.tick_params(axis="both", which="major", labelsize=18, width=2.5, length=8)
    ax.tick_params(axis="y", which="minor", width=1.8, length=4)
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontweight("bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2.8)
    _save(fig, OUTPUT / "all_deleted_nodes_direct_vs_indirect_delta_r_panel_A")


def run(force: bool = False) -> dict:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(
        ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml"
    )
    model = load_model(ROOT / cfg["model_dir"])
    selected = json.loads(
        (SOURCE / "selected_single_node_observation_matched.json").read_text()
    )
    results = run_all_deletions(model, selected, force=force)
    statistics_frame, node_frame, input_frame = pooled_statistics(results)
    statistics_frame.to_csv(OUTPUT / "pooled_statistics.csv", index=False)
    node_frame.to_csv(OUTPUT / "deleted_node_group_differences.csv", index=False)
    input_frame.to_csv(OUTPUT / "deleted_node_input_group_differences.csv", index=False)
    make_figure(results, statistics_frame)

    primary = statistics_frame[
        (statistics_frame.scope == "all_outputs")
        & (statistics_frame.adjacency == "deleted_to_output")
        & (statistics_frame.source == "simulation")
        & (statistics_frame.outcome == "absolute")
    ].iloc[0].to_dict()
    any_direction = statistics_frame[
        (statistics_frame.scope == "all_outputs")
        & (statistics_frame.adjacency == "any_direct")
        & (statistics_frame.source == "simulation")
        & (statistics_frame.outcome == "absolute")
    ].iloc[0].to_dict()
    summary = {
        "setup": {
            "deleted_nodes": model.areas,
            "n_deleted_nodes": len(model.areas),
            "candidate_input_areas": AREAS9,
            "input_rule": "use AREAS9 except the deleted node",
            "output_rule": "all 55 areas except the deleted node",
            "n_input_conditions": 512,
            "primary_adjacency": "deleted_to_output: FLN[output, deleted] > 0",
            "primary_outcome": "absolute held-out simulation delta r",
            "primary_statistical_unit": "deleted node; direct and indirect output means paired within node",
            "minimum_common_valid_trials": MIN_VALID_TRIALS,
            "sign_flip_permutations": PERMUTATIONS,
            "bootstrap_samples": BOOTSTRAPS,
        },
        "primary_result": primary,
        "sensitivity_any_anatomical_direction": any_direction,
        "files": {
            "all_conditions": str(
                (OUTPUT / "all_deleted_nodes_input_output_delta_r.csv.gz").resolve()
            ),
            "statistics": str((OUTPUT / "pooled_statistics.csv").resolve()),
            "deleted_node_differences": str(
                (OUTPUT / "deleted_node_group_differences.csv").resolve()
            ),
            "deleted_node_input_differences": str(
                (OUTPUT / "deleted_node_input_group_differences.csv").resolve()
            ),
            "figure_png": str(
                (OUTPUT / "all_deleted_nodes_direct_vs_indirect_delta_r_panel_A.png").resolve()
            ),
            "figure_pdf": str(
                (OUTPUT / "all_deleted_nodes_direct_vs_indirect_delta_r_panel_A.pdf").resolve()
            ),
        },
    }
    (OUTPUT / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete every whole-brain node in turn and pool direct-versus-indirect output delta-r."
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    _style()
    plt.rcParams.update(
        {
            "font.size": 18,
            "font.weight": "bold",
            "axes.labelsize": 21,
            "axes.labelweight": "bold",
            "axes.titlesize": 22,
            "axes.titleweight": "bold",
        }
    )
    print(json.dumps(run(force=args.force), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
