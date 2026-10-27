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

from . import exp2_pap_correlation_validation as exp2
from .exp2_two_stage import _setup
from .io import load_config
from .paper_figure_node_deletion import make_combined


TARGET_DEFAULT = 100
CANDIDATE_CONDITIONS = 2200
S_RANGE = (2.0815631208601286, 6.693999317940512)
SIGMA_BETA_RANGE = (0.05, 0.60)
FLOOR_RATIO_RANGE = (0.0, 0.50)
SEED = 20260820
ENRICHED_CANDIDATES = 3000
ENRICHED_S_RANGE = (4.5, 6.693999317940512)
ENRICHED_SIGMA_BETA_RANGE = (0.05, 0.25)
ENRICHED_FLOOR_RATIO_RANGE = (0.30, 0.50)


def _rooted(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else Path(__file__).resolve().parents[3] / path


def _tag(analysis_s: float) -> str:
    return f"{analysis_s:.3f}".replace(".", "p")


def _candidate_table(out: Path) -> pd.DataFrame:
    path = out / "candidate_conditions.csv"
    if path.exists():
        return pd.read_csv(path)
    sampler = qmc.LatinHypercube(d=3, seed=SEED)
    values = qmc.scale(
        sampler.random(CANDIDATE_CONDITIONS),
        [S_RANGE[0], SIGMA_BETA_RANGE[0], FLOOR_RATIO_RANGE[0]],
        [S_RANGE[1], SIGMA_BETA_RANGE[1], FLOOR_RATIO_RANGE[1]],
    )
    frame = pd.DataFrame(values, columns=["S", "sigma_beta", "floor_ratio"])
    frame.insert(0, "condition_id", np.arange(len(frame), dtype=int))
    frame["input_sigma"] = exp2.MAIN_INPUT_SIGMA
    frame.to_csv(path, index=False)
    return frame


def _threshold_candidate_table(out: Path, base: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    path = out / "candidate_conditions_rlt0p6_enriched.csv"
    if path.exists():
        return pd.read_csv(path)
    sampler = qmc.LatinHypercube(d=3, seed=SEED + 1)
    values = qmc.scale(
        sampler.random(ENRICHED_CANDIDATES),
        [ENRICHED_S_RANGE[0], ENRICHED_SIGMA_BETA_RANGE[0], ENRICHED_FLOOR_RATIO_RANGE[0]],
        [ENRICHED_S_RANGE[1], ENRICHED_SIGMA_BETA_RANGE[1], ENRICHED_FLOOR_RATIO_RANGE[1]],
    )
    enriched = pd.DataFrame(values, columns=["S", "sigma_beta", "floor_ratio"])
    enriched.insert(0, "condition_id", np.arange(len(base), len(base) + len(enriched), dtype=int))
    enriched["input_sigma"] = exp2.MAIN_INPUT_SIGMA
    completed = set(previous.condition_id.astype(int)) if len(previous) else set()
    base = base.copy()
    base["priority"] = [0 if int(value) in completed else 2 for value in base.condition_id]
    enriched["priority"] = 1
    combined = pd.concat([base, enriched], ignore_index=True).sort_values(["priority", "condition_id"])
    combined.to_csv(path, index=False)
    return combined


def _evaluate(payload: tuple[dict, str, dict, float, bool]) -> dict:
    cfg, out_text, candidate, analysis_s, force = payload
    exp2.ANALYSIS_S = float(analysis_s)
    try:
        import numba

        numba.set_num_threads(1)
    except (ImportError, ValueError):
        pass
    original, tau, _, _ = _setup()
    condition_id = int(candidate["condition_id"])
    condition = (
        float(candidate["S"]),
        float(candidate["sigma_beta"]),
        float(candidate["floor_ratio"]),
        exp2.MAIN_INPUT_SIGMA,
        f"window_{_tag(analysis_s)}",
    )
    out = Path(out_text)
    cached = exp2._simulate_condition(cfg, out, original, tau, condition_id, condition, force=force)
    row, _, _, _ = exp2._analyze_condition(original, tau, condition_id, condition, cached)
    row["analysis_s"] = float(analysis_s)
    return row


def _metrics(data: pd.DataFrame) -> dict[str, float]:
    x = data.residual_corrected_r.to_numpy(float)
    y = data.nonlinear_heldout_r_mean.to_numpy(float)
    fit = stats.linregress(x, y)
    identity_r2 = 1.0 - np.sum((y - x) ** 2) / np.sum((y - y.mean()) ** 2)
    return {
        "n_conditions": int(len(data)),
        "identity_R2": float(identity_r2),
        "ols_R2": float(fit.rvalue**2),
        "pearson_r": float(fit.rvalue),
        "slope": float(fit.slope),
        "intercept": float(fit.intercept),
        "p": float(fit.pvalue),
    }


def _save(fig: plt.Figure, base: Path) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot(data: pd.DataFrame, metric: dict, base: Path, panel_label: bool) -> None:
    x = data.residual_corrected_r.to_numpy(float)
    y = data.nonlinear_heldout_r_mean.to_numpy(float)
    lo = min(x.min(), y.min()) - .04
    hi = max(x.max(), y.max()) + .04
    grid = np.linspace(lo, hi, 200)
    with plt.rc_context(
        {
            "font.size": 14,
            "axes.labelsize": 14,
            "axes.titlesize": 15,
            "axes.linewidth": 1.9,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12.5,
            "xtick.major.width": 1.7,
            "ytick.major.width": 1.7,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.9, 4.25), constrained_layout=True)
        ax.scatter(x, y, s=20, color="#247BA0", alpha=.42, edgecolor="none", rasterized=True)
        ax.plot([lo, hi], [lo, hi], color="#D1495B", ls="--", lw=3.0, label="identity")
        ax.plot(grid, metric["intercept"] + metric["slope"] * grid, color="#222222", lw=3.0, label="OLS")
        ax.set(
            xlim=(lo, hi),
            ylim=(lo, hi),
            xlabel="Residual-corrected theory r",
            ylabel="Nonlinear held-out r",
        )
        ax.legend(frameon=False, loc="lower right")
        if panel_label:
            ax.text(-.18, 1.06, "G", transform=ax.transAxes, fontsize=23, fontweight="bold")
        _save(fig, base)


def run(
    cfg: dict,
    analysis_s: float,
    target: int,
    jobs: int,
    force: bool,
    install_panel: bool,
    max_theory_r: float | None,
) -> Path:
    exp2.ANALYSIS_S = float(analysis_s)
    output_root = _rooted(cfg["output_dir"])
    out = output_root / f"exp2_pap_s32_window_{_tag(analysis_s)}"
    paper = output_root / "paper_figure_node_deletion"
    (out / "condition_features").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    candidates = _candidate_table(out)
    result_path = out / "all_condition_results.csv"
    previous = pd.read_csv(result_path) if result_path.exists() and not force else pd.DataFrame()
    if max_theory_r is not None:
        candidates = _threshold_candidate_table(out, candidates, previous)
    previous_by_id = {int(row.condition_id): row._asdict() for row in previous.itertuples(index=False)}
    records = candidates.to_dict("records")
    rows: list[dict] = []
    accepted = 0

    def register(row: dict) -> None:
        nonlocal accepted
        row = dict(row)
        valid = bool(
            int(row["n_valid_common"]) >= 80
            and np.isfinite(row["residual_corrected_r"])
            and np.isfinite(row["nonlinear_heldout_r_mean"])
            and (max_theory_r is None or float(row["residual_corrected_r"]) < max_theory_r)
        )
        row["accepted"] = valid
        rows.append(row)
        accepted += int(valid)

    cursor = 0
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=jobs) if jobs > 1 else None
    try:
        while accepted < target and cursor < len(records):
            batch = records[cursor:cursor + max(20, jobs * 4)]
            cursor += len(batch)
            pending = []
            for candidate in batch:
                condition_id = int(candidate["condition_id"])
                if condition_id in previous_by_id:
                    register(previous_by_id[condition_id])
                else:
                    pending.append(candidate)
            if pending:
                payloads = [(cfg, str(out), item, analysis_s, force) for item in pending]
                evaluated = [_evaluate(payload) for payload in payloads] if pool is None else list(pool.map(_evaluate, payloads))
                for row in evaluated:
                    register(row)
            rows.sort(key=lambda item: int(item["condition_id"]))
            pd.DataFrame(rows).to_csv(result_path, index=False)
            print(f"window={analysis_s:.3f}s candidates={len(rows)} valid={accepted}/{target}", flush=True)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
    all_results = pd.DataFrame(rows)
    all_results.to_csv(result_path, index=False)
    selected = all_results[all_results.accepted.astype(bool)].head(target).copy()
    if len(selected) != target:
        raise RuntimeError(f"Only {len(selected)} valid conditions for {analysis_s:.3f} s")
    suffix = "" if max_theory_r is None else f"_rlt{str(max_theory_r).replace('.', 'p')}"
    selected_path = out / f"selected_{target}_condition_results{suffix}.csv"
    selected.to_csv(selected_path, index=False)
    metric = _metrics(selected)
    metric.update(
        {
            "analysis_s": float(analysis_s),
            "trials_per_condition": exp2.N_TRIALS,
            "calibration_phases": exp2.N_CALIBRATION_PHASES,
            "heldout_phases": exp2.N_HELDOUT_PHASES,
        }
    )
    metric["max_residual_corrected_r"] = max_theory_r
    (out / f"metrics_{target}{suffix}.json").write_text(json.dumps(metric, indent=2))
    _plot(selected, metric, out / "figures" / f"s32_window_{_tag(analysis_s)}_{target}{suffix}", False)
    if install_panel:
        _plot(selected, metric, paper / "panels" / "G_exp2_1000trial_theory_simulation", True)
        manifest_path = paper / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["panel_G"] = metric | {"method": "S32 residual-corrected long-window validation"}
        manifest_path.write_text(json.dumps(manifest, indent=2))
        make_combined(paper)
    (out / "run_metadata.json").write_text(
        json.dumps(
            {
                "analysis_s": analysis_s,
                "candidate_seed": SEED,
                "candidate_ranges": {
                    "S": S_RANGE,
                    "sigma_beta": SIGMA_BETA_RANGE,
                    "floor_ratio": FLOOR_RATIO_RANGE,
                },
                "input_sigma": exp2.MAIN_INPUT_SIGMA,
                "target_valid_conditions": target,
                "max_residual_corrected_r": max_theory_r,
                "enriched_candidate_ranges": {
                    "S": ENRICHED_S_RANGE,
                    "sigma_beta": ENRICHED_SIGMA_BETA_RANGE,
                    "floor_ratio": ENRICHED_FLOOR_RATIO_RANGE,
                } if max_theory_r is not None else None,
            },
            indent=2,
        )
    )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    parser.add_argument("--analysis-s", type=float, required=True)
    parser.add_argument("--target", type=int, default=TARGET_DEFAULT)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--install-panel", action="store_true")
    parser.add_argument("--max-theory-r", type=float)
    args = parser.parse_args()
    cfg = load_config(args.config)
    print(run(
        cfg, args.analysis_s, args.target, max(1, args.jobs), args.force,
        args.install_panel, args.max_theory_r,
    ))


if __name__ == "__main__":
    main()
