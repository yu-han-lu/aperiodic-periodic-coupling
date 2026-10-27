from __future__ import annotations

import argparse

from .io import load_config
from .pairwise_nonlinear_window_validation import run_exp3_pairwise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    out = run_exp3_pairwise(
        load_config(args.config),
        n_trials=64,
        n_masks=120,
        output_name="finite_window_exp3_pairwise_single_phase_12c",
        phase_seed_offset=220000,
    )
    print(out)


if __name__ == "__main__":
    main()
