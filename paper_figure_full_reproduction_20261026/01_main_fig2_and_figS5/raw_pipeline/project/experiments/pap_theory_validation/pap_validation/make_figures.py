from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output-dir",default="experiments/pap_theory_validation/outputs"); args=parser.parse_args()
    root=Path(args.output_dir)
    figs=sorted(root.glob("*/figures/*"))
    print(f"Found {len(figs)} generated figure files under {root.resolve()}")
    for f in figs: print(f)


if __name__=="__main__": main()
