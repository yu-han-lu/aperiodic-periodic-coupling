from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .core import load_model
from .io import load_config
from .paper_figure_dense_revision import plot_exp1_dense
from .paper_figure_node_deletion import (
    CALIBRATION_PHASES,
    TOTAL_PHASES,
    _corr,
    _cov_to_corr,
    _full_error_layer,
    _rooted,
    _style,
    make_combined,
    make_panel_a,
    make_panel_d,
    make_panel_e,
    make_panel_h_i,
)


def _correct_exp3_from_cache(out: Path) -> tuple[pd.DataFrame, dict]:
    cache = np.load(out / "exp3_node_deletion_features_robust_v3.npz")
    selected = json.loads((out / "exp3_selected_candidate_robust_v3.json").read_text())
    features = cache["features"].astype(float)
    valid = cache["valid"]
    use = cache["use"]
    theory = {
        "baseline": cache["baseline_theory"],
        "node_deleted": cache["deleted_theory"],
    }
    rows = []
    values = {}
    for network_pos, name in enumerate(("baseline", "node_deleted")):
        signal_cov, error_cov, cross_cov, total_cov = _full_error_layer(
            theory[name],
            features[:CALIBRATION_PHASES, network_pos],
            valid[:CALIBRATION_PHASES, network_pos],
            use,
        )
        predicted = _cov_to_corr(total_cov)
        phase_r = []
        phase_n = []
        for phase in range(CALIBRATION_PHASES, TOTAL_PHASES):
            phase_use = use & valid[phase, network_pos]
            phase_n.append(int(phase_use.sum()))
            phase_r.append(_corr(features[phase, network_pos, phase_use]))
        phase_r = np.asarray(phase_r)
        values[name] = {"theory": predicted, "simulation": phase_r.mean(), "phase_r": phase_r}
        rows.append(
            {
                "network": name,
                "n_valid": int(use.sum()),
                "latent_exact_r": _cov_to_corr(signal_cov),
                "residual_only_r": _cov_to_corr(error_cov),
                "signal_error_cross_fro": float(np.linalg.norm(cross_cov)),
                "theory_plus_realization_r": predicted,
                "simulation_r": float(phase_r.mean()),
                "simulation_r_sd": float(phase_r.std(ddof=1)),
                "heldout_phase_r_json": json.dumps(phase_r.tolist()),
                "heldout_phase_n_json": json.dumps(phase_n),
            }
        )
    delta_phase = values["node_deleted"]["phase_r"] - values["baseline"]["phase_r"]
    rows.append(
        {
            "network": "delta",
            "n_valid": int(use.sum()),
            "latent_exact_r": np.nan,
            "residual_only_r": np.nan,
            "signal_error_cross_fro": np.nan,
            "theory_plus_realization_r": values["node_deleted"]["theory"] - values["baseline"]["theory"],
            "simulation_r": values["node_deleted"]["simulation"] - values["baseline"]["simulation"],
            "simulation_r_sd": float(delta_phase.std(ddof=1)),
            "heldout_phase_r_json": json.dumps(delta_phase.tolist()),
            "heldout_phase_n_json": "",
        }
    )
    result = pd.DataFrame(rows)
    result.to_csv(out / "exp3_node_deletion_long_results_robust_v3_corrected.csv", index=False)
    return result, selected


def run(cfg: dict) -> Path:
    _style()
    output_root = _rooted(cfg["output_dir"])
    out = output_root / "paper_figure_node_deletion"
    model = load_model(_rooted(cfg["model_dir"]))
    updates = {
        "panel_A": make_panel_a(cfg, model, out),
        "panel_B": plot_exp1_dense(
            pd.read_csv(output_root / "paper_figure_dense_revision" / "exp1_dense_conditions.csv"), out
        ),
        "panel_D": make_panel_d(out),
        "panel_E": make_panel_e(out),
    }
    corrected, selected = _correct_exp3_from_cache(out)
    updates["panels_H_I"] = make_panel_h_i(model, out, corrected, selected)
    updates["exp3_corrected_results"] = {
        row.network: {
            "theory_r": float(row.theory_plus_realization_r),
            "simulation_r": float(row.simulation_r),
            "simulation_sd": float(row.simulation_r_sd),
        }
        for row in corrected.itertuples(index=False)
    }
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(updates)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    make_combined(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/full.yaml")
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
