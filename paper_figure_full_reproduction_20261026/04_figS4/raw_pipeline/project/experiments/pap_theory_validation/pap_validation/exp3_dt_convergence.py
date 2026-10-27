from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .core import frequency_grid, input_psd, load_model, save_figure, save_json
from .exp3 import Z0, _input_trials
from .exp3_targeted_degree10 import (
    CHUNK_SIZE,
    GROUPS,
    N_MASKS,
    N_TRIALS,
    _joint_multi_edge_theory,
    _networks,
    _targeted_masks,
)
from .exp3_trial_seed_validation import _feature_arrays_vectorized
from .io import load_config, prepare_output
from .nonlinear_network_validation import _run_multioutput_batch
from .pairwise_nonlinear_window_validation import _fit_metrics, _rooted, _window_specs


OUTPUT_NAME = "exp3_targeted_degree10_dt_convergence"
DT_VALUES = (0.25, 0.05)
INPUT_CUTOFF_HZ = 2000.0


def _dt_tag(dt_ms: float) -> str:
    return f"dt{dt_ms:g}".replace(".", "p")


def _matched_bandlimited_inputs(
    zs: np.ndarray,
    dt_ms: float,
    analysis_s: float,
    burn_s: float,
    sigma: float,
    seed: int,
    trial_offset: int = 0,
    cutoff_hz: float = INPUT_CUTOFF_HZ,
) -> np.ndarray:
    """Fourier trials whose <=cutoff coefficients match across sampling steps."""
    fs = 1000.0 / dt_ms
    analysis_steps = int(round(analysis_s * 1000.0 / dt_ms))
    burn_steps = int(round(burn_s * 1000.0 / dt_ms))
    frequencies = np.fft.rfftfreq(analysis_steps, d=1.0 / fs)
    # Exclude the coarse-grid Nyquist bin so every retained coefficient has the
    # same conjugate-pair interpretation at both sampling steps.
    active = frequencies < min(cutoff_hz, fs / 2.0 + 1e-12)
    active_frequencies = frequencies[active]
    reference_spectrum = np.zeros(len(frequencies), dtype=np.complex128)
    reference_spectrum[active] = np.sqrt(input_psd(active_frequencies, Z0))
    reference_spectrum[0] = 0.0
    reference_signal = np.fft.irfft(reference_spectrum, n=analysis_steps)
    global_scale = sigma / (np.std(reference_signal) + 1e-15)
    result = np.empty((len(zs), analysis_steps + burn_steps), dtype=np.float64)
    for local_trial, z in enumerate(zs):
        trial = trial_offset + local_trial
        rng = np.random.default_rng(seed + 1000003 * trial)
        phase = rng.uniform(0.0, 2.0 * np.pi, int(active.sum()))
        spectrum = np.zeros(len(frequencies), dtype=np.complex128)
        spectrum[active] = np.sqrt(input_psd(active_frequencies, z)) * np.exp(1j * phase)
        spectrum[0] = 0.0
        nyquist = np.flatnonzero(active)[-1]
        if analysis_steps % 2 == 0 and nyquist == len(frequencies) - 1:
            spectrum[nyquist] = spectrum[nyquist].real
        signal = np.fft.irfft(spectrum, n=analysis_steps) * global_scale
        signal -= signal.mean()
        result[local_trial, :burn_steps] = signal[-burn_steps:]
        result[local_trial, burn_steps:] = signal
    return result


def _simulate_dt_group(cfg, out, model, group, group_pos, z, dt_ms, force=False):
    group_tag = f"{group['input_area']}_{group['lesion_area']}_{group['degree_type']}"
    tag = f"{group_tag}_{_dt_tag(dt_ms)}"
    feature_path = out / f"features_{tag}.npz"
    qc_path = out / f"qc_{tag}.csv.gz"
    edge_path = out / f"edges_{group_tag}.csv.gz"
    if feature_path.exists() and qc_path.exists() and edge_path.exists() and not force:
        print(f"Experiment 3 dt convergence {tag}: cached", flush=True)
        cached = np.load(feature_path)
        return cached["features"], cached["valid"], pd.read_csv(qc_path), pd.read_csv(edge_path)

    masks, candidate_count = _targeted_masks(
        model,
        group["lesion_area"],
        group["degree_type"],
        int(cfg["seed"]) + 1010000 + group_pos * 10000,
    )
    networks = _networks(model, masks)
    input_idx = model.areas.index(group["input_area"])
    output_areas = [group["significant_output"], group["null_output"]]
    output_indices = np.array([model.areas.index(area) for area in output_areas], dtype=np.int64)
    analysis_s = _window_specs(cfg)["short"]["analysis_s"]
    burn_s = float(cfg["exp3_nonlinear"]["burn_s"])
    burn_steps = int(round(burn_s * 1000.0 / dt_ms))
    specs = {"short": _window_specs(cfg)["short"]}
    features = np.empty((len(networks), len(output_areas), N_TRIALS, 2), dtype=float)
    valid = np.empty((len(networks), len(output_areas), N_TRIALS), dtype=bool)
    qc_rows = []
    for start in range(0, N_TRIALS, CHUNK_SIZE):
        stop = min(start + CHUNK_SIZE, N_TRIALS)
        inputs = _matched_bandlimited_inputs(
            z[start:stop],
            dt_ms,
            analysis_s,
            burn_s,
            float(cfg["exp3_nonlinear"]["input_sigma"]),
            int(cfg["seed"]) + 1410000,
            trial_offset=start,
        )
        for network_pos, (network_id, params, _, _) in enumerate(networks):
            outputs, rect, min_e, min_i, min_de, min_di = _run_multioutput_batch(
                params, inputs, dt_ms, burn_steps, output_indices, input_idx=input_idx
            )
            values, is_valid = _feature_arrays_vectorized(outputs, dt_ms, specs)["short"]
            features[network_pos, :, start:stop] = values
            valid[network_pos, :, start:stop] = is_valid
            qc_rows.append(
                {
                    "group": group_tag,
                    "dt_ms": dt_ms,
                    "trial_start": start,
                    "trial_stop": stop,
                    "network": network_id,
                    "rectification_fraction": float(
                        rect.sum() / (2 * len(model.areas) * inputs.shape[1] * (stop - start))
                    ),
                    "min_rate_E": float(min_e.min()),
                    "min_rate_I": float(min_i.min()),
                    "min_drive_E": float(min_de.min()),
                    "min_drive_I": float(min_di.min()),
                }
            )
        print(f"Experiment 3 dt convergence {tag}: trials {start + 1}-{stop}/{N_TRIALS}", flush=True)

    edge_rows = []
    for mask_id, _, _, edges in masks:
        for source, target, weight in edges:
            edge_rows.append(
                {
                    "group": group_tag,
                    "mask_id": mask_id,
                    "source_area": source,
                    "target_area": target,
                    "deleted_weight": weight,
                    "n_degree_candidates": candidate_count,
                }
            )
    edges = pd.DataFrame(edge_rows)
    qc = pd.DataFrame(qc_rows)
    np.savez_compressed(feature_path, features=features, valid=valid)
    qc.to_csv(qc_path, index=False, compression="gzip")
    edges.to_csv(edge_path, index=False, compression="gzip")
    return features, valid, qc, edges


def _analyze_group(cfg, model, group, group_pos, features_by_dt, valid_by_dt, u):
    masks, _ = _targeted_masks(
        model,
        group["lesion_area"],
        group["degree_type"],
        int(cfg["seed"]) + 1010000 + group_pos * 10000,
    )
    input_idx = model.areas.index(group["input_area"])
    output_areas = [group["significant_output"], group["null_output"]]
    output_indices = [model.areas.index(area) for area in output_areas]
    common_valid = []
    for output_pos in range(len(output_areas)):
        common_valid.append(
            np.logical_and.reduce(
                [np.all(valid_by_dt[dt][:, output_pos, :], axis=0) for dt in DT_VALUES]
            )
        )
    sigmas = [np.cov(u[use], rowvar=False, ddof=1) for use in common_valid]
    freqs = frequency_grid()
    theory = {}
    for mask_pos, (mask_id, _, delta_j, _) in enumerate(masks, 1):
        theory[mask_id] = _joint_multi_edge_theory(
            model, delta_j, sigmas, freqs, input_idx, output_indices
        )

    rows = []
    group_tag = f"{group['input_area']}_{group['lesion_area']}_{group['degree_type']}"
    for dt_ms in DT_VALUES:
        features = features_by_dt[dt_ms]
        for output_pos, output_area in enumerate(output_areas):
            use = common_valid[output_pos]
            baseline_r = float(np.corrcoef(features[0, output_pos, use].T)[0, 1])
            for mask_pos, (mask_id, _, _, _) in enumerate(masks, 1):
                first, second, exact = theory[mask_id]
                perturbed_r = float(np.corrcoef(features[mask_pos, output_pos, use].T)[0, 1])
                rows.append(
                    {
                        "group": group_tag,
                        "dt_ms": dt_ms,
                        "input_area": group["input_area"],
                        "lesion_area": group["lesion_area"],
                        "degree_type": group["degree_type"],
                        "output_area": output_area,
                        "prior_class": "previously significant" if output_pos == 0 else "previously null",
                        "mask_id": mask_id,
                        "n_input_trials": N_TRIALS,
                        "n_valid_common_across_dt": int(use.sum()),
                        "baseline_actual_r": baseline_r,
                        "perturbed_actual_r": perturbed_r,
                        "delta_rho_actual": perturbed_r - baseline_r,
                        "delta_rho_first": float(first[output_pos]),
                        "delta_rho_second": float(second[output_pos]),
                        "delta_rho_exact_resolvent": float(exact[output_pos]),
                    }
                )
    return pd.DataFrame(rows)


def _metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["dt_ms", "group", "input_area", "lesion_area", "degree_type", "output_area", "prior_class"]
    for key, group in results.groupby(keys, sort=False):
        base = dict(zip(keys, key))
        for order, column in (
            ("first", "delta_rho_first"),
            ("second", "delta_rho_second"),
            ("exact_resolvent", "delta_rho_exact_resolvent"),
        ):
            rows.append(base | {"order": order} | _fit_metrics(group.delta_rho_actual, group[column]))
    return pd.DataFrame(rows)


def _dt_agreement(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["group", "input_area", "lesion_area", "degree_type", "output_area", "prior_class"]
    for key, group in results.groupby(keys, sort=False):
        pivot = group.pivot(index="mask_id", columns="dt_ms", values="delta_rho_actual")
        rows.append(dict(zip(keys, key)) | _fit_metrics(pivot[0.05], pivot[0.25]))
    return pd.DataFrame(rows)


def run(cfg: dict, force: bool = False) -> Path:
    started = time.time()
    out = prepare_output(cfg, OUTPUT_NAME)
    model = load_model(_rooted(cfg["model_dir"]))
    z, u = _input_trials(N_TRIALS, int(cfg["seed"]) + 300)
    group_results = []
    all_qc = []
    for group_pos, group in enumerate(GROUPS):
        features_by_dt = {}
        valid_by_dt = {}
        for dt_ms in DT_VALUES:
            features, valid, qc, _ = _simulate_dt_group(
                cfg, out, model, group, group_pos, z, dt_ms, force=force
            )
            features_by_dt[dt_ms] = features
            valid_by_dt[dt_ms] = valid
            all_qc.append(qc)
        group_results.append(
            _analyze_group(cfg, model, group, group_pos, features_by_dt, valid_by_dt, u)
        )
    results = pd.concat(group_results, ignore_index=True)
    qc = pd.concat(all_qc, ignore_index=True)
    metrics = _metrics(results)
    agreement = _dt_agreement(results)
    results.to_csv(out / "dt_convergence_predictions.csv.gz", index=False, compression="gzip")
    metrics.to_csv(out / "dt_convergence_metrics.csv", index=False)
    agreement.to_csv(out / "actual_delta_dt_agreement.csv", index=False)
    qc.to_csv(out / "dt_convergence_qc.csv.gz", index=False, compression="gzip")
    _plot(out, results, metrics, agreement)
    save_json(
        out / "manifest.json",
        {
            "experiment": "full55_targeted_10_edge_dt_convergence",
            "dt_ms": DT_VALUES,
            "input_cutoff_hz": INPUT_CUTOFF_HZ,
            "matched_fourier_phase_per_trial_across_dt": True,
            "common_valid_trials_across_all_networks_and_both_dt": True,
            "n_masks_per_group": N_MASKS,
            "n_trials_per_correlation": N_TRIALS,
            "analysis_s": _window_specs(cfg)["short"]["analysis_s"],
            "burn_s": cfg["exp3_nonlinear"]["burn_s"],
            "runtime_s": time.time() - started,
        },
    )
    return out


def _ordered_combinations(results):
    combinations = results[
        ["group", "input_area", "lesion_area", "degree_type", "output_area", "prior_class"]
    ].drop_duplicates()
    return pd.concat(
        [
            combinations[combinations.prior_class == "previously significant"],
            combinations[combinations.prior_class == "previously null"],
        ],
        ignore_index=True,
    )


def _plot(out, results, metrics, agreement):
    plt.style.use("seaborn-v0_8-whitegrid")
    ordered = _ordered_combinations(results)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    colors = {0.25: "#4C78A8", 0.05: "#E45756"}
    for ax, row in zip(axes.flat, ordered.itertuples(index=False)):
        data = results[(results.group == row.group) & (results.output_area == row.output_area)]
        for dt_ms in DT_VALUES:
            subset = data[data.dt_ms == dt_ms]
            x = subset.delta_rho_exact_resolvent.to_numpy()
            y = subset.delta_rho_actual.to_numpy()
            ax.scatter(x, y, s=22, alpha=0.65, color=colors[dt_ms], label=f"dt={dt_ms:g} ms")
            if np.std(x) > 0:
                fit = np.polyfit(x, y, 1)
                xx = np.linspace(x.min(), x.max(), 100)
                ax.plot(xx, np.polyval(fit, xx), color=colors[dt_ms], lw=1.6)
        lo = min(data.delta_rho_exact_resolvent.min(), data.delta_rho_actual.min())
        hi = max(data.delta_rho_exact_resolvent.max(), data.delta_rho_actual.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set(
            xlabel="joint 10-edge exact-resolvent Delta r",
            ylabel="full55 time-domain actual Delta r",
            title=f"{row.input_area}; delete {row.lesion_area}-{row.degree_type}; read {row.output_area}",
        )
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Experiment 3: does reducing Euler dt improve exact-resolvent calibration?", fontsize=13)
    save_figure(fig, out / "figures" / "exp3_dt_convergence_scatter")
    plt.close(fig)

    labels = [f"{row.input_area}/{row.lesion_area}-{row.degree_type}/{row.output_area}" for row in ordered.itertuples(index=False)]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), constrained_layout=True)
    for ax, metric, title in (
        (axes[0], "R2", "Identity R2 vs exact-resolvent"),
        (axes[1], "ols_R2", "OLS R2 vs exact-resolvent"),
    ):
        for dt_ms, offset in ((0.25, -0.08), (0.05, 0.08)):
            values = []
            for row in ordered.itertuples(index=False):
                item = metrics[
                    (metrics.dt_ms == dt_ms)
                    & (metrics.group == row.group)
                    & (metrics.output_area == row.output_area)
                    & (metrics.order == "exact_resolvent")
                ].iloc[0]
                values.append(item[metric])
            ax.plot(x + offset, values, "o-", color=colors[dt_ms], label=f"dt={dt_ms:g} ms")
        ax.axhline(1, color="0.25", ls="--", lw=1)
        ax.axhline(0, color="0.5", ls=":", lw=1)
        ax.set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
        ax.set(ylabel=metric, title=title)
    axes[0].legend(frameon=False, fontsize=8)
    values = []
    for row in ordered.itertuples(index=False):
        values.append(agreement[(agreement.group == row.group) & (agreement.output_area == row.output_area)].pearson_r.iloc[0])
    axes[2].bar(x, values, color=["#59A14F" if value >= 0 else "#E45756" for value in values])
    axes[2].axhline(1, color="0.25", ls="--", lw=1)
    axes[2].axhline(0, color="0.5", ls=":", lw=1)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_xticks(x, labels, rotation=38, ha="right", fontsize=8)
    axes[2].set(ylabel="Pearson r", title="Actual Delta r reproducibility across dt")
    fig.suptitle("Step-size convergence summary across 40 matched masks", fontsize=13)
    save_figure(fig, out / "figures" / "exp3_dt_convergence_summary")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(run(load_config(args.config), force=args.force))


if __name__ == "__main__":
    main()
