from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from .core import AREAS9, central_moments, frequency_grid, load_model, network_features, prediction_metrics, save_figure, save_json, transfer_power
from .exp1 import Z0, ZHI, ZLO, ZSCALE, _draw_design, _feature_fun
from .io import load_config, prepare_output


def _gradient(fun, h: float = 2e-3) -> np.ndarray:
    eye = np.eye(len(Z0))
    return np.stack([(fun(h * e) - fun(-h * e)) / (2 * h) for e in eye], axis=1)


def _conditions(cfg: dict) -> list[tuple[str, float, float]]:
    designs = ["independent_gaussian", "rank_one", "uniform", "fixed_total", "skewed", "one_factor_mixture", "finite_sample"]
    rows = []
    for eps in cfg["exp1"]["epsilons"]:
        rows.extend((design, float(eps), 0.0) for design in designs)
        rows.extend(("correlated_gaussian", float(eps), float(rho)) for rho in cfg["exp1"]["rhos"])
    return rows


def _simulate_pair(args) -> tuple[dict, list[dict]]:
    input_area, output_area, cfg, n_trials = args
    model = load_model(Path(cfg["model_dir"]))
    freqs = frequency_grid()
    input_idx, output_idx = model.areas.index(input_area), model.areas.index(output_area)
    transfer = transfer_power(model.J, input_idx, output_idx, freqs)
    peak_band, slope_band = tuple(cfg["peak_band_hz"]), tuple(cfg["slope_band_hz"])
    fun = _feature_fun(transfer, freqs, peak_band, slope_band)
    grad = _gradient(fun)
    condition_rows = []

    for ci, (design, eps, rho) in enumerate(_conditions(cfg)):
        rng = np.random.default_rng(int(cfg["seed"]) + 1000 + ci)
        u = _draw_design(design, n_trials, eps, rng, rho)
        raw = Z0 + u * ZSCALE
        z = np.clip(raw, ZLO, ZHI)
        clipped = np.any(z != raw, axis=1)
        vals = np.full((n_trials, 2), np.nan)
        valid = np.zeros(n_trials, dtype=bool)
        for i in range(n_trials):
            feature, _ = network_features(
                model.J, z[i], input_idx, output_idx, freqs, transfer, peak_band, slope_band
            )
            vals[i] = feature["peak_omega"], feature["exponent"]
            valid[i] = feature["peak_valid"]
        use = valid & ~clipped & np.isfinite(vals).all(axis=1)
        if use.sum() < 3:
            observed = predicted = np.nan
        else:
            accepted_u = (z[use] - Z0) / ZSCALE
            sigma, _ = central_moments(accepted_u)
            observed = float(np.cov(vals[use, 0], vals[use, 1], ddof=1)[0, 1])
            predicted = float(grad[0] @ sigma @ grad[1])
        condition_rows.append({
            "input_area": input_area,
            "output_area": output_area,
            "condition": f"{design}_eps{eps:g}_rho{rho:g}",
            "design": design,
            "epsilon": eps,
            "rho": rho,
            "n_trials": n_trials,
            "n_valid": int(use.sum()),
            "valid_rate": float(use.mean()),
            "observed_covariance": observed,
            "predicted_covariance_C49": predicted,
        })

    frame = pd.DataFrame(condition_rows)
    metric = prediction_metrics(frame.observed_covariance.values, frame.predicted_covariance_C49.values)
    metric.update({
        "input_area": input_area,
        "output_area": output_area,
        "median_valid_rate": float(frame.valid_rate.median()),
        "min_valid_rate": float(frame.valid_rate.min()),
    })
    return metric, condition_rows


def run(cfg: dict) -> Path:
    started = time.time()
    out = prepare_output(cfg, "exp1_pairwise_io")
    pcfg = cfg.get("exp1_pairwise", {})
    n_trials = int(pcfg.get("n_trials", 1024))
    workers = int(pcfg.get("workers", min(8, len(AREAS9))))
    jobs = [(source, target, cfg, n_trials) for source in AREAS9 for target in AREAS9]
    metric_rows, condition_rows = [], []

    if workers == 1:
        results = map(_simulate_pair, jobs)
        for metric, rows in results:
            metric_rows.append(metric)
            condition_rows.extend(rows)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_simulate_pair, job): (job[0], job[1]) for job in jobs}
            for future in as_completed(futures):
                metric, rows = future.result()
                metric_rows.append(metric)
                condition_rows.extend(rows)

    metrics = pd.DataFrame(metric_rows).sort_values(["input_area", "output_area"])
    conditions = pd.DataFrame(condition_rows).sort_values(["input_area", "output_area", "epsilon", "design", "rho"])
    metrics.to_csv(out / "pair_similarity.csv", index=False)
    conditions.to_csv(out / "condition_predictions.csv.gz", index=False, compression="gzip")
    save_json(out / "metrics.json", {
        "n_pairs": len(metrics),
        "n_conditions_per_pair": int(conditions.groupby(["input_area", "output_area"]).size().iloc[0]),
        "n_trials_per_condition": n_trials,
        "R2_min": float(metrics.R2.min()),
        "R2_median": float(metrics.R2.median()),
        "R2_max": float(metrics.R2.max()),
        "pearson_r_min": float(metrics.pearson_r.min()),
        "pearson_r_median": float(metrics.pearson_r.median()),
        "median_valid_rate": float(metrics.median_valid_rate.median()),
        "runtime_s": time.time() - started,
    })
    save_json(out / "manifest.json", {
        "experiment": "input_sampling_pairwise_io",
        "network": "full 55-area model",
        "areas": AREAS9,
        "input_population": "E",
        "output_population": "E",
        "common_random_numbers": True,
        "similarity_metrics": ["Eq49 observed-vs-predicted R2", "Pearson r"],
    })
    _figure(out, metrics)
    return out


def _figure(out: Path, metrics: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, column, title, digits in [
        (axes[0], "R2", "Eq. 49 similarity: R^2 (observed color range)", 4),
        (axes[1], "pearson_r", "Eq. 49 similarity: Pearson r (observed color range)", 5),
    ]:
        matrix = metrics.pivot(index="output_area", columns="input_area", values=column).reindex(index=AREAS9, columns=AREAS9)
        vmin = float(matrix.min().min())
        vmax = float(matrix.max().max())
        image = ax.imshow(matrix.values, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xticks(range(len(AREAS9)), AREAS9, rotation=45, ha="right")
        ax.set_yticks(range(len(AREAS9)), AREAS9)
        ax.set_xlabel("Input area (E population)")
        ax.set_ylabel("Output area (E population)")
        ax.set_title(title)
        threshold = (vmin + vmax) / 2
        for row in range(len(AREAS9)):
            for col in range(len(AREAS9)):
                value = matrix.iloc[row, col]
                ax.text(col, row, f"{value:.{digits}f}", ha="center", va="center", fontsize=6.7,
                        color="white" if value < threshold else "black")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    save_figure(fig, out / "figures" / "pairwise_io_theory_similarity")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
