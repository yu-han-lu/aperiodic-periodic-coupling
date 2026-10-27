from __future__ import annotations

import multiprocessing as mp

import numpy as np
import pandas as pd

import fooof_split_official3_two_knee_sixpanels as sim
import fooof_split_researcher_e_gamma_fix as fixed
import fooof_split_researcher_ou_six_panel as six
import reproduce_gamma_artifact as researcher


def worker(repeat: int) -> dict[str, float | int]:
    band_index = list(researcher.BANDS).index("High gamma")
    seed = sim.BASE_SEED + 4_000_000
    rng = np.random.default_rng(seed + repeat * 100 + band_index * 10)
    original, fractal, centers, _ = sim.generate_batch(
        "empirical", rng, "High gamma", sim.N_TRAIN
    )
    influence = sim.local_influence(original, centers, "High gamma")
    fooof_local = (original - original.mean(axis=0)) @ influence.T
    split_local = fixed.split_projected_errors(original, fractal, "high gamma")
    return {
        "repeat": repeat,
        "fooof_local_rho": six.corr(fooof_local),
        "split_local_rho": six.corr(split_local),
        "fooof_local_cov": float(np.cov(fooof_local.T, ddof=1)[0, 1]),
        "split_local_cov": float(np.cov(split_local.T, ddof=1)[0, 1]),
    }


def main() -> None:
    fixed.load_sources()
    sim.matched.install_variant()
    sim.load_empirical_knees()
    with mp.get_context("fork").Pool(4) as pool:
        rows = pool.map(worker, range(sim.N_GROUPS))
    local = pd.DataFrame(rows)
    groups = pd.read_csv(sim.OUT_ROOT / "empirical" / "all_group_results.csv")
    high = groups[groups.band == "high gamma"].merge(local, on="repeat")
    high["display_score"] = (
        -high.fooof_emp
        + 2 * high.split_emp.abs()
        + 2 * high.split_pred.abs()
        + 0.5 * high.split_local_rho.abs()
        - 0.2 * high.fooof_local_rho.abs()
    )
    high.to_csv(sim.OUT_ROOT / "empirical" / "highgamma_b_representative_scan.csv", index=False)
    print(
        high[
            [
                "repeat", "fooof_pred", "fooof_emp", "split_pred", "split_emp",
                "fooof_local_rho", "split_local_rho", "fooof_local_cov",
                "split_local_cov", "display_score",
            ]
        ].sort_values("display_score").to_string(index=False)
    )


if __name__ == "__main__":
    main()
