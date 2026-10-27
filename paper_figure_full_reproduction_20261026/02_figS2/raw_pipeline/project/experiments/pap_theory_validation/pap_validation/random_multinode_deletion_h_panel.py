from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import load_model
from .exp3_large_delta_screen import _discrete_transfer
from .io import load_config
from .paper_figure_node_deletion import (
    _all_transfer_power,
    _corr,
    _node_deleted_j,
    _observed_values,
    _style,
)
from .single_node_deletion_all_nodes import (
    OUTPUT,
    SOURCE,
    _full_error_prediction,
    _identity_r2,
    _network_features,
)
from .single_node_deletion_revision import INPUT_AREA, OUTPUT_AREA


ROOT = Path(__file__).resolve().parents[3]
SEED = 20260818
POOL_PER_SIZE = 20000
TOP_PER_SIZE = 40
RANDOM_PER_SIZE = 10
TARGET_MAX_DELTA = 0.10
POOL_PATH = OUTPUT / "random_multinode_candidate_pool.csv.gz"
MULTI_PATH = OUTPUT / "random_multinode_deletion_results.csv"
COMBINED_PATH = OUTPUT / "combined_single_multinode_deletion_results.csv"
FIGURE_PATH = OUTPUT / "H_single_multinode_theory_delta_vs_simulation_delta"
LARGE_EXISTING_ROOT = (
    ROOT / "experiments" / "pap_theory_validation" / "outputs" / "paper_figure_node_deletion"
)


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _candidate_pool(nodes: list[str], singles: pd.DataFrame) -> pd.DataFrame:
    if POOL_PATH.exists():
        return pd.read_csv(POOL_PATH)
    effect = singles.set_index("deleted_node").theory_delta_r.to_dict()
    rng = np.random.default_rng(SEED)
    rows: list[dict] = []
    seen: set[tuple[str, ...]] = set()
    for n_deleted in (3, 4, 5):
        count = 0
        while count < POOL_PER_SIZE:
            combo = tuple(sorted(rng.choice(nodes, size=n_deleted, replace=False).tolist()))
            if combo in seen:
                continue
            seen.add(combo)
            rows.append(
                {
                    "combo_id": "|".join(combo),
                    "nodes": "|".join(combo),
                    "n_deleted": n_deleted,
                    "additive_screen_delta_r": float(sum(effect[node] for node in combo)),
                }
            )
            count += 1
    pool = pd.DataFrame(rows)
    pool.to_csv(POOL_PATH, index=False, compression="gzip")
    return pool


def _selected_candidates(pool: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED + 1)
    selected = []
    for n_deleted in (3, 4, 5):
        subset = pool[pool.n_deleted == n_deleted].sort_values(
            "additive_screen_delta_r", ascending=False
        )
        selected.append(subset.head(TOP_PER_SIZE))
        remainder = subset.iloc[TOP_PER_SIZE:].copy()
        remainder["screen_bin"] = pd.qcut(
            remainder.additive_screen_delta_r, RANDOM_PER_SIZE,
            labels=False, duplicates="drop",
        )
        sampled = (
            remainder.groupby("screen_bin", group_keys=False)
            .sample(n=1, random_state=int(rng.integers(0, 2**31 - 1)))
        )
        selected.append(sampled)
    return pd.concat(selected, ignore_index=True).drop_duplicates("combo_id")


def _evaluate(model, row: pd.Series, cache: np.lib.npyio.NpzFile) -> dict:
    nodes = str(row.nodes).split("|")
    frequencies = cache["frequencies"]
    frequency_ids = cache["frequency_ids"]
    z = cache["z"]
    calibration_input = cache["calibration_input_psd"] if "calibration_input_psd" in cache.files else None
    heldout_input = cache["heldout_input_psd"] if "heldout_input_psd" in cache.files else None
    if calibration_input is None or heldout_input is None:
        shared = np.load(OUTPUT / "shared_longwindow_input_psd_means.npz")
        calibration_input = shared["calibration_input_psd"]
        heldout_input = shared["heldout_input_psd"]

    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    tau_e = float(model.params["tau_exc"])
    n_steps = int((len(cache["baseline_transfer_power"]) - 1) * 2)
    # The cached frequency ids come from the original full FFT grid and retain
    # the exact long-window discretization used for all existing single-node points.
    n_steps = int(round(65.536 * 1000.0 / 0.25))

    deleted_j, _ = _node_deleted_j(model, nodes)
    continuous = _all_transfer_power(deleted_j, [input_idx, output_idx], frequencies)[0, 1]
    theory, theory_valid = _observed_values(continuous, frequencies, z)
    discrete = np.abs(
        _discrete_transfer(deleted_j, input_idx, output_idx, n_steps, tau_e)
    ) ** 2
    calibration, calibration_valid, heldout, heldout_valid = _network_features(
        discrete[frequency_ids], frequencies, calibration_input, heldout_input
    )

    baseline_theory = cache["baseline_theory"]
    baseline_theory_valid = cache["baseline_theory_valid"]
    baseline_cal = cache["baseline_calibration"]
    baseline_cal_valid = cache["baseline_calibration_valid"]
    baseline_hold = cache["baseline_heldout"]
    baseline_hold_valid = cache["baseline_heldout_valid"]
    use = (
        baseline_theory_valid
        & theory_valid
        & baseline_cal_valid
        & calibration_valid
        & baseline_hold_valid
        & heldout_valid
        & np.isfinite(baseline_theory).all(axis=1)
        & np.isfinite(theory).all(axis=1)
    )
    if use.sum() < 300:
        raise RuntimeError(f"Only {use.sum()} common valid trials for {row.combo_id}")

    theory_baseline_r = _full_error_prediction(baseline_theory[use], baseline_cal[use])
    theory_deleted_r = _full_error_prediction(theory[use], calibration[use])
    simulation_baseline_r = _corr(baseline_hold[use])
    simulation_deleted_r = _corr(heldout[use])
    return {
        "combo_id": row.combo_id,
        "deleted_nodes": row.nodes,
        "n_deleted": int(row.n_deleted),
        "n_valid": int(use.sum()),
        "additive_screen_delta_r": float(row.additive_screen_delta_r),
        "theory_baseline_r": theory_baseline_r,
        "theory_deleted_r": theory_deleted_r,
        "theory_delta_r": theory_deleted_r - theory_baseline_r,
        "simulation_baseline_r": simulation_baseline_r,
        "simulation_deleted_r": simulation_deleted_r,
        "simulation_delta_r": simulation_deleted_r - simulation_baseline_r,
    }


def run_multinode(model, candidates: pd.DataFrame) -> pd.DataFrame:
    existing = pd.read_csv(MULTI_PATH) if MULTI_PATH.exists() else pd.DataFrame()
    completed = set(existing.combo_id) if not existing.empty else set()
    rows = existing.to_dict("records")
    cache = np.load(OUTPUT / "all_single_node_deletion_features.npz")
    queue = candidates[~candidates.combo_id.isin(completed)]
    for pos, (_, candidate) in enumerate(queue.iterrows(), start=1):
        result = _evaluate(model, candidate, cache)
        rows.append(result)
        pd.DataFrame(rows).to_csv(MULTI_PATH, index=False)
        print(
            f"random multi-node deletion {pos}/{len(queue)}: "
            f"n={result['n_deleted']}, delta={result['simulation_delta_r']:.4f}",
            flush=True,
        )
    return pd.DataFrame(rows)


def _existing_large_delta_point() -> pd.DataFrame:
    results = pd.read_csv(
        LARGE_EXISTING_ROOT / "exp3_node_deletion_long_results_robust_v3_corrected.csv"
    ).set_index("network")
    selected = json.loads(
        (LARGE_EXISTING_ROOT / "exp3_selected_candidate_robust_v3.json").read_text()
    )
    delta = results.loc["delta"]
    baseline = results.loc["baseline"]
    deleted = results.loc["node_deleted"]
    return pd.DataFrame(
        [
            {
                "deleted_node": "|".join(selected["nodes"]),
                "deleted_nodes": "|".join(selected["nodes"]),
                "n_deleted": len(selected["nodes"]),
                "n_valid": int(delta.n_valid),
                "theory_baseline_r": float(baseline.theory_plus_realization_r),
                "theory_deleted_r": float(deleted.theory_plus_realization_r),
                "theory_delta_r": float(delta.theory_plus_realization_r),
                "simulation_baseline_r": float(baseline.simulation_r),
                "simulation_deleted_r": float(deleted.simulation_r),
                "simulation_delta_r": float(delta.simulation_r),
                "source": "existing 12-node condition",
            }
        ]
    )


def _plot(combined: pd.DataFrame) -> dict:
    x = combined.theory_delta_r.to_numpy(float)
    y = combined.simulation_delta_r.to_numpy(float)
    fit = stats.linregress(x, y)
    pearson = stats.pearsonr(x, y)
    identity_r2 = _identity_r2(x, y)
    lo = float(min(x.min(), y.min()))
    hi = float(max(x.max(), y.max()))
    pad = max((hi - lo) * 0.10, 0.005)
    lo -= pad
    hi += pad
    grid = np.linspace(lo, hi, 200)

    fig, ax = plt.subplots(figsize=(7.2, 6.3), constrained_layout=True)
    styles = {
        1: ("#247BA0", "o", "1 node"),
        3: ("#F4A261", "^", "3 nodes"),
        4: ("#2A9D8F", "s", "4 nodes"),
        5: ("#8E5EA2", "D", "5 nodes"),
        12: ("#D1495B", "*", "12 nodes"),
    }
    for n_deleted, (color, marker, label) in styles.items():
        subset = combined[combined.n_deleted == n_deleted]
        ax.scatter(
            subset.theory_delta_r,
            subset.simulation_delta_r,
            s=92 if n_deleted == 1 else (220 if n_deleted == 12 else 112),
            color=color,
            marker=marker,
            edgecolor="white",
            linewidth=1.0,
            alpha=0.92,
            label=label,
            zorder=3,
        )
    ax.plot([lo, hi], [lo, hi], color="#D1495B", ls="--", lw=3.2, label="Identity")
    ax.plot(grid, fit.intercept + fit.slope * grid, color="#202020", lw=4.0, label="Linear fit")
    ax.axhline(0, color="0.55", lw=1.6, zorder=0)
    ax.axvline(0, color="0.55", lw=1.6, zorder=0)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel=r"Theory $\Delta r$", ylabel=r"Simulation $\Delta r$")
    ax.xaxis.label.set_size(24)
    ax.yaxis.label.set_size(24)
    ax.tick_params(labelsize=16, length=7, width=2.2)
    ax.text(
        0.04,
        0.96,
        (
            f"n={len(combined)} deletion patterns\n"
            f"Pearson r={pearson.statistic:.3f}\n"
            f"OLS R$^2$={fit.rvalue ** 2:.3f}\n"
            f"Identity R$^2$={identity_r2:.3f}"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=16,
    )
    max_pos = int(np.argmax(np.maximum(x, y)))
    ax.scatter([x[max_pos]], [y[max_pos]], s=240, facecolor="none", edgecolor="#D1495B", linewidth=2.7, zorder=4)
    ax.annotate(
        rf"max $\Delta r$={max(x[max_pos], y[max_pos]):.3f}",
        (x[max_pos], y[max_pos]),
        xytext=(-8, 13),
        textcoords="offset points",
        ha="right",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=13,
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.5,
    )
    ax.text(-0.15, 1.06, "H", transform=ax.transAxes, fontsize=28, fontweight="bold")
    for spine in ax.spines.values():
        spine.set_linewidth(2.4)
    _save(fig, FIGURE_PATH)
    return {
        "n_patterns": int(len(combined)),
        "n_multinode": int((combined.n_deleted > 1).sum()),
        "max_theory_delta_r": float(x.max()),
        "max_simulation_delta_r": float(y.max()),
        "pearson_r": float(pearson.statistic),
        "ols_R2": float(fit.rvalue**2),
        "identity_R2": float(identity_r2),
    }


def main() -> None:
    _style()
    plt.rcParams.update(
        {
            "font.size": 17,
            "axes.labelsize": 24,
            "axes.linewidth": 2.4,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 15,
            "lines.linewidth": 3.2,
        }
    )
    cfg = load_config(ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml")
    model = load_model(ROOT / cfg["model_dir"])
    singles = pd.read_csv(OUTPUT / "all_single_node_deletion_results.csv").copy()
    singles["deleted_nodes"] = singles.deleted_node
    singles["n_deleted"] = 1
    singles["source"] = "existing single-node condition"
    nodes = [area for area in model.areas if area not in (INPUT_AREA, OUTPUT_AREA)]
    pool = _candidate_pool(nodes, singles)
    candidates = _selected_candidates(pool)
    multi = run_multinode(model, candidates)
    multi["source"] = "random 3-5-node condition"
    large_existing = _existing_large_delta_point()
    combined = pd.concat([singles, multi, large_existing], ignore_index=True, sort=False)
    combined.to_csv(COMBINED_PATH, index=False)
    metrics = _plot(combined)
    if max(metrics["max_theory_delta_r"], metrics["max_simulation_delta_r"]) <= TARGET_MAX_DELTA:
        raise RuntimeError(
            f"Maximum delta r {max(metrics['max_theory_delta_r'], metrics['max_simulation_delta_r']):.4f} "
            f"did not exceed {TARGET_MAX_DELTA:.2f}; expand the candidate screen."
        )
    summary = {
        "setup": {
            "input_area": INPUT_AREA,
            "output_area": OUTPUT_AREA,
            "random_seed": SEED,
            "random_pool_per_size": POOL_PER_SIZE,
            "deleted_node_counts": [3, 4, 5],
            "selected_top_per_size": TOP_PER_SIZE,
            "selected_random_strata_per_size": RANDOM_PER_SIZE,
            "analysis_s": 65.536,
            "n_input_conditions": 512,
            "calibration_realizations": 3,
            "heldout_realizations": 5,
            "existing_large_delta_condition": (
                "saved 12-node A6Va-input/V2-readout validation; included unchanged"
            ),
        },
        "metrics": metrics,
        "files": {
            "candidate_pool": str(POOL_PATH.resolve()),
            "multinode_results": str(MULTI_PATH.resolve()),
            "combined_results": str(COMBINED_PATH.resolve()),
            "figure": str(FIGURE_PATH.with_suffix(".png").resolve()),
        },
    }
    (OUTPUT / "random_multinode_h_panel_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
