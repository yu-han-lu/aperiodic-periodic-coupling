from __future__ import annotations

"""Extend the frozen high-gamma peak-strength scan into stronger inputs."""

import json
import multiprocessing as mp
from pathlib import Path

import pandas as pd

import scan_high_gamma_peak_strength_gaussian_mixing as base


OUTPUT = Path(
    "/PORTABLE_HOME/Documents/New project/output/"
    "fooof_irasa_split_high_gamma_strength_scan_extension2_2026-09-13"
)
STRENGTH_RANGES = [
    (28.0, 34.0),
    (40.0, 50.0),
    (60.0, 75.0),
    (90.0, 115.0),
]


def worker(task: tuple[int, int]) -> dict[str, float | int]:
    range_index, repeat = task
    original_ranges = base.STRENGTH_RANGES
    try:
        base.STRENGTH_RANGES = STRENGTH_RANGES
        return base.worker((range_index, repeat))
    finally:
        base.STRENGTH_RANGES = original_ranges


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    base.shared.fixed.load_sources()
    base.shared.matched.install_variant()
    base.shared.frozen.official._fooof_group_class()
    base.shared.frozen.load_empirical_knees()
    tasks = [
        (range_index, repeat)
        for range_index in range(len(STRENGTH_RANGES))
        for repeat in range(base.N_REPEATS)
    ]
    rows: list[dict[str, float | int]] = []
    with mp.get_context("fork").Pool(4) as pool:
        for completed, row in enumerate(
            pool.imap_unordered(worker, tasks), start=1
        ):
            rows.append(row)
            pd.DataFrame(rows).sort_values(
                ["range_index", "repeat"]
            ).to_csv(OUTPUT / "scan_checkpoint.csv", index=False)
            print(f"completed extension tasks: {completed}/{len(tasks)}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["range_index", "repeat"])
    summary = base.summarize(frame)
    frame.to_csv(OUTPUT / "scan_all_repeats.csv", index=False)
    summary.to_csv(OUTPUT / "scan_summary.csv", index=False)
    (OUTPUT / "scan_manifest.json").write_text(
        json.dumps(
            {
                "band": base.DISPLAY_BAND,
                "ranges": STRENGTH_RANGES,
                "n_repeats": base.N_REPEATS,
                "n_train": base.N_TRAIN,
                "n_heldout": base.N_HELDOUT,
                "n_gaussian_covariance_mc": base.N_MC,
                "all_other_settings": "frozen",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
