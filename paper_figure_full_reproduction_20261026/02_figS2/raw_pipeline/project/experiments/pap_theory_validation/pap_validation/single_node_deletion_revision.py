from __future__ import annotations

import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

from .core import AREAS9, frequency_grid, input_psd, load_model
from .exp1 import Z0, ZHI, ZLO, ZSCALE
from .exp3_large_delta_screen import _continuous_transfer, _discrete_transfer, _exact_trial_features
from .io import load_config
from .paper_figure_node_deletion import (
    CALIBRATION_PHASES,
    DT_MS,
    EXP3_ANALYSIS_S,
    EXP3_TRIALS,
    HELDOUT_PHASES,
    SEED,
    TOTAL_PHASES,
    _all_transfer_power,
    _corr,
    _cov_to_corr,
    _features_from_averaged_psd,
    _node_deleted_j,
    _observed_values,
    _pair_gradient,
    _style,
)


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "single_node_deletion_revision"
INPUT_AREA = "A6Va"
OUTPUT_AREA = "V2"
SCREEN_TRIALS = 1024
FINAL_TRIALS = EXP3_TRIALS
COARSE_BIN_COUNT = 16


def _save(fig: plt.Figure, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _sigma_from_baseline(model, variance_ratio: float, target_r: float) -> np.ndarray:
    transfer = _continuous_transfer(
        model.J, model.areas.index(INPUT_AREA), model.areas.index(OUTPUT_AREA)
    )
    gradient = _pair_gradient(transfer, frequency_grid())
    right_inverse = np.linalg.pinv(gradient)
    target = np.array(
        [
            [variance_ratio, target_r * np.sqrt(variance_ratio)],
            [target_r * np.sqrt(variance_ratio), 1.0],
        ]
    )
    sigma = right_inverse @ target @ right_inverse.T
    sigma = (sigma + sigma.T) / 2.0
    sigma /= max(float(np.max(np.diag(sigma))), 1e-15)
    return sigma


def _design(sigma: np.ndarray, epsilon: float, n_trials: int, seed: int):
    eigenvalues, eigenvectors = np.linalg.eigh((sigma + sigma.T) / 2.0)
    root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 0.0)))
    u = np.random.default_rng(seed).normal(size=(n_trials, 4)) @ root.T * epsilon
    z = np.clip(Z0 + u * ZSCALE, ZLO, ZHI)
    return z, u


def _screen(model) -> tuple[pd.DataFrame, dict]:
    cache = OUTPUT / "single_node_exact_screen.csv"
    selected_path = OUTPUT / "selected_single_node.json"
    if cache.exists() and selected_path.exists():
        return pd.read_csv(cache), json.loads(selected_path.read_text())
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    baseline_transfer = _continuous_transfer(model.J, input_idx, output_idx)
    rows = []
    sigma_cache = {}
    for variance_ratio in (0.02, 0.032614, 0.053183, 0.086725, 0.141421):
        for target_r in (-0.15, 0.0, 0.15):
            sigma_cache[(variance_ratio, target_r)] = _sigma_from_baseline(
                model, variance_ratio, target_r
            )
    for node_pos, node in enumerate(model.areas):
        if node in (INPUT_AREA, OUTPUT_AREA):
            continue
        deleted_j, _ = _node_deleted_j(model, [node])
        deleted_transfer = _continuous_transfer(deleted_j, input_idx, output_idx)
        for (variance_ratio, target_r), sigma in sigma_cache.items():
            for epsilon in (0.20, 0.30, 0.40, 0.50, 0.60):
                z, _ = _design(
                    sigma, epsilon, SCREEN_TRIALS,
                    SEED + 910000 + node_pos * 1000 + int(round(epsilon * 100)),
                )
                baseline, valid_b = _exact_trial_features(baseline_transfer, z)
                deleted, valid_d = _exact_trial_features(deleted_transfer, z)
                use = valid_b & valid_d & np.isfinite(baseline).all(axis=1) & np.isfinite(deleted).all(axis=1)
                if use.sum() < 850:
                    continue
                rb = _corr(baseline[use])
                rd = _corr(deleted[use])
                rows.append(
                    {
                        "node": node,
                        "variance_ratio": variance_ratio,
                        "target_r": target_r,
                        "epsilon": epsilon,
                        "sigma_json": json.dumps(sigma.tolist()),
                        "exact_baseline_r": rb,
                        "exact_deleted_r": rd,
                        "exact_delta_r": rd - rb,
                        "n_valid": int(use.sum()),
                        "score": abs(rd - rb) - 0.35 * abs(rb),
                    }
                )
        print(f"single-node exact screen {node_pos + 1}/{len(model.areas)}: {node}", flush=True)
    screen = pd.DataFrame(rows).sort_values("score", ascending=False)
    screen.to_csv(cache, index=False)
    eligible = screen[(screen.exact_baseline_r.abs() <= 0.20)].copy()
    if eligible.empty:
        eligible = screen
    row = eligible.iloc[0]
    selected = {
        "node": row.node,
        "nodes": [row.node],
        "input_area": INPUT_AREA,
        "output_area": OUTPUT_AREA,
        "variance_ratio": float(row.variance_ratio),
        "target_r": float(row.target_r),
        "epsilon": float(row.epsilon),
        "sigma": json.loads(row.sigma_json),
        "screen_exact_baseline_r": float(row.exact_baseline_r),
        "screen_exact_deleted_r": float(row.exact_deleted_r),
        "screen_exact_delta_r": float(row.exact_delta_r),
        "screen_n_valid": int(row.n_valid),
    }
    selected_path.write_text(json.dumps(selected, indent=2))
    return screen, selected


def _frequency_grid():
    n_steps = int(round(EXP3_ANALYSIS_S * 1000.0 / DT_MS))
    full_f = np.fft.rfftfreq(n_steps, d=DT_MS / 1000.0)
    start = int(np.ceil(0.5 / (full_f[1] - full_f[0])))
    stop = int(np.floor(1200.0 / (full_f[1] - full_f[0]))) + 1
    ids = np.arange(start, stop, COARSE_BIN_COUNT)
    return full_f[ids], ids, n_steps


def _observation_refine(model, screen: pd.DataFrame) -> dict:
    path = OUTPUT / "single_node_observation_screen.csv"
    selected_path = OUTPUT / "selected_single_node_observation_matched.json"
    if path.exists() and selected_path.exists():
        return json.loads(selected_path.read_text())
    frequencies, frequency_ids, n_steps = _frequency_grid()
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    tau_e = float(model.params["tau_exc"])
    baseline_continuous = _all_transfer_power(
        model.J, [input_idx, output_idx], frequencies
    )[0, 1]
    baseline_discrete = np.abs(
        _discrete_transfer(model.J, input_idx, output_idx, n_steps, tau_e)
    ) ** 2
    shortlist = (
        screen.sort_values("score", ascending=False)
        .groupby("node", as_index=False, group_keys=False)
        .head(4)
    )
    rows = []
    for candidate_pos, item in enumerate(shortlist.itertuples(index=False)):
        selected = {
            "node": item.node,
            "nodes": [item.node],
            "input_area": INPUT_AREA,
            "output_area": OUTPUT_AREA,
            "variance_ratio": float(item.variance_ratio),
            "target_r": float(item.target_r),
            "epsilon": float(item.epsilon),
            "sigma": json.loads(item.sigma_json),
        }
        deleted_j, _ = _node_deleted_j(model, [item.node])
        deleted_continuous = _all_transfer_power(
            deleted_j, [input_idx, output_idx], frequencies
        )[0, 1]
        deleted_discrete = np.abs(
            _discrete_transfer(deleted_j, input_idx, output_idx, n_steps, tau_e)
        ) ** 2
        z, _ = _design(
            np.asarray(selected["sigma"]), selected["epsilon"], 256,
            SEED + 1110000 + candidate_pos * 100,
        )
        theory = []
        theory_valid = np.ones(len(z), dtype=bool)
        for transfer in (baseline_continuous, deleted_continuous):
            values, valid = _observed_values(transfer, frequencies, z)
            theory.append(values)
            theory_valid &= valid
        observed = []
        observed_valid = []
        for repeat in range(2):
            rng = np.random.default_rng(SEED + 1120000 + candidate_pos * 100 + repeat)
            psd = np.empty((2, len(z), len(frequencies)), dtype=np.float32)
            shape = COARSE_BIN_COUNT * (CALIBRATION_PHASES if repeat == 0 else HELDOUT_PHASES)
            for trial in range(len(z)):
                realization = rng.gamma(shape, 1.0 / shape, size=len(frequencies))
                input_power = input_psd(frequencies, z[trial]) * realization
                psd[0, trial] = input_power * baseline_discrete[frequency_ids]
                psd[1, trial] = input_power * deleted_discrete[frequency_ids]
            values, valid = _features_from_averaged_psd(frequencies, psd)
            observed.append(values)
            observed_valid.append(valid)
        use = theory_valid & observed_valid[0].all(axis=0) & observed_valid[1].all(axis=0)
        if use.sum() < 180:
            continue
        predicted, simulation = [], []
        for network_pos in range(2):
            signal_values = theory[network_pos][use]
            error = observed[0][network_pos, use] - signal_values
            sc = signal_values - signal_values.mean(axis=0, keepdims=True)
            ec = error - error.mean(axis=0, keepdims=True)
            signal_cov = np.cov(signal_values, rowvar=False, ddof=1)
            error_cov = np.cov(error, rowvar=False, ddof=1)
            cross = sc.T @ ec / (len(signal_values) - 1)
            predicted.append(_cov_to_corr(signal_cov + error_cov + cross + cross.T))
            simulation.append(_corr(observed[1][network_pos, use]))
        delta_theory = predicted[1] - predicted[0]
        delta_sim = simulation[1] - simulation[0]
        rows.append(
            selected
            | {
                "sigma_json": json.dumps(selected["sigma"]),
                "n_valid": int(use.sum()),
                "theory_baseline_r": predicted[0],
                "theory_deleted_r": predicted[1],
                "theory_delta_r": delta_theory,
                "simulation_baseline_r": simulation[0],
                "simulation_deleted_r": simulation[1],
                "simulation_delta_r": delta_sim,
                "score": abs(delta_sim) - 1.5 * abs(delta_sim - delta_theory),
            }
        )
        if (candidate_pos + 1) % 20 == 0:
            print(f"single-node observation screen {candidate_pos + 1}/{len(shortlist)}", flush=True)
    refined = pd.DataFrame(rows).sort_values("score", ascending=False)
    refined.to_csv(path, index=False)
    row = refined.iloc[0]
    selected = {
        "node": row.node,
        "nodes": [row.node],
        "input_area": INPUT_AREA,
        "output_area": OUTPUT_AREA,
        "variance_ratio": float(row.variance_ratio),
        "target_r": float(row.target_r),
        "epsilon": float(row.epsilon),
        "sigma": json.loads(row.sigma_json),
        "screen_theory_baseline_r": float(row.theory_baseline_r),
        "screen_theory_deleted_r": float(row.theory_deleted_r),
        "screen_theory_delta_r": float(row.theory_delta_r),
        "screen_simulation_baseline_r": float(row.simulation_baseline_r),
        "screen_simulation_deleted_r": float(row.simulation_deleted_r),
        "screen_simulation_delta_r": float(row.simulation_delta_r),
        "screen_n_valid": int(row.n_valid),
    }
    selected_path.write_text(json.dumps(selected, indent=2))
    return selected


def _simulate(model, selected: dict) -> tuple[pd.DataFrame, dict]:
    result_path = OUTPUT / "single_node_observation_selected_longwindow_results.csv"
    detail_path = OUTPUT / "single_node_observation_selected_longwindow_features.npz"
    if result_path.exists() and detail_path.exists():
        return pd.read_csv(result_path), selected
    deleted_j, _ = _node_deleted_j(model, [selected["node"]])
    frequencies, frequency_ids, n_steps = _frequency_grid()
    input_idx = model.areas.index(INPUT_AREA)
    output_idx = model.areas.index(OUTPUT_AREA)
    z, u = _design(
        np.asarray(selected["sigma"]), float(selected["epsilon"]), FINAL_TRIALS, SEED + 990000
    )
    continuous = {
        "baseline": _all_transfer_power(model.J, [input_idx, output_idx], frequencies)[0, 1],
        "node_deleted": _all_transfer_power(deleted_j, [input_idx, output_idx], frequencies)[0, 1],
    }
    theory = {}
    theory_valid = np.ones(FINAL_TRIALS, dtype=bool)
    for name in ("baseline", "node_deleted"):
        theory[name], valid = _observed_values(continuous[name], frequencies, z)
        theory_valid &= valid
    tau_e = float(model.params["tau_exc"])
    discrete = [
        np.abs(_discrete_transfer(j, input_idx, output_idx, n_steps, tau_e)) ** 2
        for j in (model.J, deleted_j)
    ]
    phase_psds = np.empty((TOTAL_PHASES, 2, FINAL_TRIALS, len(frequencies)), dtype=np.float32)
    for phase in range(TOTAL_PHASES):
        rng = np.random.default_rng(SEED + 1010000 + phase * 100000)
        for trial in range(FINAL_TRIALS):
            realization = rng.gamma(COARSE_BIN_COUNT, 1.0 / COARSE_BIN_COUNT, size=len(frequencies))
            input_power = input_psd(frequencies, z[trial]) * realization
            for network_pos in range(2):
                phase_psds[phase, network_pos, trial] = input_power * discrete[network_pos][frequency_ids]
        print(f"single-node held-out PSD phase {phase + 1}/{TOTAL_PHASES}", flush=True)
    calibration, calibration_valid = _features_from_averaged_psd(
        frequencies, phase_psds[:CALIBRATION_PHASES].mean(axis=0)
    )
    heldout, heldout_valid = _features_from_averaged_psd(
        frequencies, phase_psds[CALIBRATION_PHASES:].mean(axis=0)
    )
    use = theory_valid & calibration_valid.all(axis=0) & heldout_valid.all(axis=0)
    if use.sum() < 350:
        raise RuntimeError(f"Only {use.sum()} common valid single-node trials")
    combo_rs = {"baseline": [], "node_deleted": []}
    for combo in itertools.combinations(range(HELDOUT_PHASES), CALIBRATION_PHASES):
        features, valid = _features_from_averaged_psd(
            frequencies, phase_psds[CALIBRATION_PHASES + np.asarray(combo)].mean(axis=0)
        )
        for network_pos, name in enumerate(("baseline", "node_deleted")):
            combo_use = use & valid[network_pos]
            combo_rs[name].append(_corr(features[network_pos, combo_use]))
    rows = []
    values = {}
    for network_pos, name in enumerate(("baseline", "node_deleted")):
        signal_values = theory[name][use]
        error = calibration[network_pos, use] - signal_values
        sc = signal_values - signal_values.mean(axis=0, keepdims=True)
        ec = error - error.mean(axis=0, keepdims=True)
        signal_cov = np.cov(signal_values, rowvar=False, ddof=1)
        error_cov = np.cov(error, rowvar=False, ddof=1)
        cross = sc.T @ ec / (len(signal_values) - 1)
        predicted = _cov_to_corr(signal_cov + error_cov + cross + cross.T)
        observed = _corr(heldout[network_pos, use])
        repeats = np.asarray(combo_rs[name])
        values[name] = (predicted, observed, repeats)
        rows.append(
            {
                "network": name,
                "n_valid": int(use.sum()),
                "latent_exact_r": _corr(signal_values),
                "theory_plus_realization_r": predicted,
                "simulation_r": observed,
                "simulation_r_sd": float(repeats.std(ddof=1)),
            }
        )
    delta_repeats = values["node_deleted"][2] - values["baseline"][2]
    rows.append(
        {
            "network": "delta",
            "n_valid": int(use.sum()),
            "latent_exact_r": np.nan,
            "theory_plus_realization_r": values["node_deleted"][0] - values["baseline"][0],
            "simulation_r": values["node_deleted"][1] - values["baseline"][1],
            "simulation_r_sd": float(delta_repeats.std(ddof=1)),
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(result_path, index=False)
    np.savez_compressed(
        detail_path, frequencies=frequencies, phase_psds=phase_psds, calibration=calibration,
        heldout=heldout, use=use, z=z, u=u, baseline_theory=theory["baseline"],
        deleted_theory=theory["node_deleted"],
    )
    return result, selected


def _plot_matrices(model, selected: dict) -> None:
    deleted_j, _ = _node_deleted_j(model, [selected["node"]])
    n = len(model.areas)
    baseline = model.J[:n, :n].copy()
    deleted = deleted_j[:n, :n].copy()
    np.fill_diagonal(baseline, np.nan)
    np.fill_diagonal(deleted, np.nan)
    positive = baseline[np.isfinite(baseline) & (baseline > 0)]
    vmin = float(10.0 ** np.floor(np.log10(np.nanpercentile(positive, 3))))
    vmax = float(10.0 ** np.ceil(np.log10(np.nanpercentile(positive, 99.5))))
    deleted_mask = np.zeros_like(deleted, dtype=bool)
    idx = model.areas.index(selected["node"])
    deleted_mask[idx, :] = model.fln[idx, :] > 0
    deleted_mask[:, idx] |= model.fln[:, idx] > 0
    baseline_display = np.where(np.isfinite(baseline) & (baseline > 0), baseline, vmin / 10.0)
    deleted_display = np.where(np.isfinite(deleted) & (deleted > 0), deleted, vmin / 10.0)
    deleted_display[deleted_mask] = np.nan
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad("#B8B8B8")
    cmap.set_under("#050505")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
    im = axes[0].imshow(baseline_display, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), aspect="equal")
    axes[1].imshow(deleted_display, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax), aspect="equal")
    ticks = [0, 13, 27, 41, 54]
    for ax, title in zip(axes, ("Baseline long-range J", f"Delete node {selected['node']}")):
        ax.set_xticks(ticks, [model.areas[i] for i in ticks], rotation=45, ha="right")
        ax.set_yticks(ticks, [model.areas[i] for i in ticks])
        ax.set(xlabel="Source", ylabel="Target", title=title)
    cb = fig.colorbar(im, ax=axes, shrink=0.80, pad=0.025)
    cb.set_label("E-to-E coupling (ms$^{-1}$, log scale)")
    cb.set_ticks(10.0 ** np.arange(np.log10(vmin), np.log10(vmax) + 1.0))
    cb.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext(base=10.0))
    _save(fig, OUTPUT / "single_node_baseline_deleted_J_log")


def _plot_correlations(results: pd.DataFrame, selected: dict) -> None:
    ordered = results.set_index("network").loc[["baseline", "node_deleted", "delta"]]
    x = np.arange(3)
    width = 0.34
    fig, ax = plt.subplots(figsize=(5.5, 4.5), constrained_layout=True)
    ax.bar(x - width / 2, ordered.theory_plus_realization_r, width, color="#247BA0", label="theory")
    ax.bar(x + width / 2, ordered.simulation_r, width, color="#D1495B", label="simulation")
    ax.errorbar(x + width / 2, ordered.simulation_r, yerr=ordered.simulation_r_sd, fmt="none", ecolor="0.15", capsize=4, lw=2.0)
    ax.axhline(0, color="0.2", lw=2.0)
    ax.set_xticks(x, ["Baseline", "Node-deleted", "Delta"])
    ax.set(ylabel="P-A correlation r", title=f"{INPUT_AREA} input to {OUTPUT_AREA} readout")
    ax.legend(frameon=False, loc="best")
    _save(fig, OUTPUT / "single_node_theory_simulation_correlations")


def main() -> None:
    _style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config(ROOT / "experiments" / "pap_theory_validation" / "configs" / "full.yaml")
    model = load_model(ROOT / cfg["model_dir"])
    screen, _ = _screen(model)
    selected = _observation_refine(model, screen)
    results, selected = _simulate(model, selected)
    _plot_matrices(model, selected)
    _plot_correlations(results, selected)
    summary = {
        "selected": selected,
        "analysis_s": EXP3_ANALYSIS_S,
        "n_trials": FINAL_TRIALS,
        "calibration_phases": CALIBRATION_PHASES,
        "heldout_phases": HELDOUT_PHASES,
        "results": results.to_dict("records"),
    }
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
