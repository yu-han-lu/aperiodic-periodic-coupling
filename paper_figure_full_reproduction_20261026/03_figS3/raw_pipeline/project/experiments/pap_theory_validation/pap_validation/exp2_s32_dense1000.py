from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import qmc

from .exp2_pap_correlation_validation import (
    MAIN_INPUT_SIGMA,
    N_CALIBRATION_PHASES,
    N_HELDOUT_PHASES,
    N_TRIALS,
    _analyze_condition,
    _simulate_condition,
)
from .exp2_two_stage import _setup
from .io import load_config
from .paper_figure_node_deletion import make_combined


OUTPUT_NAME = "exp2_pap_dense1000_validation"
TARGET_VALID_CONDITIONS = 1000
CANDIDATE_CONDITIONS = 2200
S_RANGE = (2.0815631208601286, 6.693999317940512)
SIGMA_BETA_RANGE = (0.05, 0.60)
FLOOR_RATIO_RANGE = (0.0, 0.50)
SEED = 20260820


def _evaluate_candidate(payload: tuple[dict, str, dict, bool]) -> dict:
    cfg, out_text, candidate, force = payload
    try:
        import numba

        numba.set_num_threads(1)
    except (ImportError, ValueError):
        pass
    out = Path(out_text)
    original, tau, _, _ = _setup()
    condition_id = int(candidate["condition_id"])
    condition = (
        float(candidate["S"]),
        float(candidate["sigma_beta"]),
        float(candidate["floor_ratio"]),
        MAIN_INPUT_SIGMA,
        "dense_main",
    )
    cached = _simulate_condition(cfg, out, original, tau, condition_id, condition, force=force)
    row, _, _, _ = _analyze_condition(original, tau, condition_id, condition, cached)
    return row


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def _candidate_table(out: Path) -> pd.DataFrame:
    path = out / "candidate_conditions.csv"
    if path.exists():
        return pd.read_csv(path)
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    unit = sampler.random(CANDIDATE_CONDITIONS)
    values = qmc.scale(
        unit,
        [S_RANGE[0], SIGMA_BETA_RANGE[0], FLOOR_RATIO_RANGE[0]],
        [S_RANGE[1], SIGMA_BETA_RANGE[1], FLOOR_RATIO_RANGE[1]],
    )
    frame = pd.DataFrame(values, columns=["S", "sigma_beta", "floor_ratio"])
    frame.insert(0, "condition_id", np.arange(len(frame), dtype=int))
    frame["input_sigma"] = MAIN_INPUT_SIGMA
    frame.to_csv(path, index=False)
    return frame


def _metrics(data: pd.DataFrame) -> dict[str, float]:
    x = data.residual_corrected_r.to_numpy(float)
    y = data.nonlinear_heldout_r_mean.to_numpy(float)
    fit = stats.linregress(x, y)
    identity_r2 = 1.0 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "n": int(len(data)),
        "identity_R2": float(identity_r2),
        "ols_R2": float(fit.rvalue**2),
        "pearson_r": float(fit.rvalue),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "p": float(fit.pvalue),
    }


def _save_figure(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot(data: pd.DataFrame, out: Path, paper: Path) -> dict[str, float]:
    metric = _metrics(data)
    x = data.residual_corrected_r.to_numpy(float)
    y = data.nonlinear_heldout_r_mean.to_numpy(float)
    lo = min(x.min(), y.min()) - 0.04
    hi = max(x.max(), y.max()) + 0.04
    grid = np.linspace(lo, hi, 200)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.scatter(x, y, s=18, color="#247BA0", alpha=.42, edgecolor="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", lw=2.4, label="identity")
    ax.plot(grid, metric["intercept"] + metric["slope"] * grid, color="#D1495B", lw=2.7, label="OLS")
    ax.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="Residual-corrected theory r",
        ylabel="Nonlinear held-out r",
        title="Experiment 2: S32 method, 1000 conditions",
    )
    ax.text(
        .04,
        .96,
        f"slope={metric['slope']:.2f}\nPearson r={metric['pearson_r']:.2f}\n"
        f"identity R2={metric['identity_R2']:.2f}\np={metric['p']:.2g}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 2.5},
    )
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    ax.tick_params(labelsize=11, width=1.4)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(13)
    ax.text(-.17, 1.06, "G", transform=ax.transAxes, fontsize=21, fontweight="bold")
    _save_figure(fig, out / "figures" / "exp2_s32_residual_corrected_dense1000")
    _save_figure_from_data(data, paper / "panels" / "G_exp2_1000trial_theory_simulation", metric)
    return metric


def _save_figure_from_data(data: pd.DataFrame, base: Path, metric: dict[str, float]) -> None:
    x = data.residual_corrected_r.to_numpy(float)
    y = data.nonlinear_heldout_r_mean.to_numpy(float)
    lo = min(x.min(), y.min()) - 0.04
    hi = max(x.max(), y.max()) + 0.04
    grid = np.linspace(lo, hi, 200)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    ax.scatter(x, y, s=18, color="#247BA0", alpha=.42, edgecolor="none", rasterized=True)
    ax.plot([lo, hi], [lo, hi], "k--", lw=2.4, label="identity")
    ax.plot(grid, metric["intercept"] + metric["slope"] * grid, color="#D1495B", lw=2.7, label="OLS")
    ax.set(
        xlim=(lo, hi),
        ylim=(lo, hi),
        xlabel="Residual-corrected theory r",
        ylabel="Nonlinear held-out r",
        title="Experiment 2: S32 method, 1000 conditions",
    )
    ax.text(
        .04,
        .96,
        f"slope={metric['slope']:.2f}\nPearson r={metric['pearson_r']:.2f}\n"
        f"identity R2={metric['identity_R2']:.2f}\np={metric['p']:.2g}",
        transform=ax.transAxes,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .82, "pad": 2.5},
    )
    ax.legend(frameon=False, loc="lower right", fontsize=10.5)
    ax.tick_params(labelsize=11, width=1.4)
    ax.xaxis.label.set_size(12)
    ax.yaxis.label.set_size(12)
    ax.title.set_size(13)
    ax.text(-.17, 1.06, "G", transform=ax.transAxes, fontsize=21, fontweight="bold")
    _save_figure(fig, base)


def run(cfg: dict, force: bool = False, jobs: int = 1) -> Path:
    output_root = _rooted(cfg["output_dir"])
    out = output_root / OUTPUT_NAME
    paper = output_root / "paper_figure_node_deletion"
    (out / "condition_features").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    candidates = _candidate_table(out)
    result_path = out / "all_condition_results.csv"
    previous = pd.read_csv(result_path) if result_path.exists() and not force else pd.DataFrame()
    previous_by_id = {int(row.condition_id): row._asdict() for row in previous.itertuples(index=False)}
    rows: list[dict] = []
    accepted = 0
    candidate_records = candidates.to_dict("records")
    original, tau, _, _ = _setup()

    def register(row: dict) -> None:
        nonlocal accepted
        row = dict(row)
        valid_point = bool(
            int(row["n_valid_common"]) >= 80
            and np.isfinite(row["residual_corrected_r"])
            and np.isfinite(row["nonlinear_heldout_r_mean"])
        )
        row["accepted_dense1000"] = valid_point
        rows.append(row)
        accepted += int(valid_point)

    cursor = 0
    while accepted < TARGET_VALID_CONDITIONS and cursor < len(candidate_records):
        batch = candidate_records[cursor:cursor + max(40, jobs * 8)]
        cursor += len(batch)
        pending = []
        for candidate in batch:
            condition_id = int(candidate["condition_id"])
            if condition_id in previous_by_id:
                register(previous_by_id[condition_id])
            else:
                pending.append(candidate)
        if pending:
            if jobs == 1:
                evaluated = [
                    _evaluate_candidate((cfg, str(out), candidate, force))
                    for candidate in pending
                ]
            else:
                payloads = [(cfg, str(out), candidate, force) for candidate in pending]
                with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
                    evaluated = list(pool.map(_evaluate_candidate, payloads))
            for row in evaluated:
                register(row)
        rows.sort(key=lambda item: int(item["condition_id"]))
        pd.DataFrame(rows).to_csv(result_path, index=False)
        print(
            f"Experiment 2 S32 dense: candidates {len(rows)}, valid {accepted}/{TARGET_VALID_CONDITIONS}",
            flush=True,
        )
    all_results = pd.DataFrame(rows)
    all_results.to_csv(result_path, index=False)
    selected = all_results[all_results.accepted_dense1000.astype(bool)].head(TARGET_VALID_CONDITIONS).copy()
    if len(selected) != TARGET_VALID_CONDITIONS:
        raise RuntimeError(f"Only {len(selected)} valid conditions were obtained")
    selected.to_csv(out / "selected_1000_condition_results.csv", index=False)
    metric = _plot(selected, out, paper)
    (out / "metrics.json").write_text(json.dumps(metric, indent=2))

    manifest_path = paper / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["panel_G"] = metric | {
        "method": "S32 residual-corrected condition-level validation",
        "conditions": TARGET_VALID_CONDITIONS,
        "trials_per_condition": N_TRIALS,
        "calibration_phases": N_CALIBRATION_PHASES,
        "heldout_phases": N_HELDOUT_PHASES,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2))
    make_combined(paper)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(run(cfg, force=args.force, jobs=max(1, args.jobs)))


if __name__ == "__main__":
    main()
