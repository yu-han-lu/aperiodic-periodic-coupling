from __future__ import annotations

import argparse
import time
from pathlib import Path

from . import exp1, exp1_pairwise, exp2, exp3, finite_window_validation, nonlinear_network_validation
from .io import load_config


def main():
    parser=argparse.ArgumentParser(description="Validate periodic-aperiodic coupling theory")
    parser.add_argument("--config",required=True)
    parser.add_argument("--experiment",choices=["all","input_sampling","input_pairwise","input_nonlinear","nonlinear_hopf","structural_kernel","structural_nonlinear","finite_window"],default="all")
    args=parser.parse_args(); cfg=load_config(args.config); started=time.time(); outputs=[]
    if args.experiment in ("all","input_sampling"): outputs.append(exp1.run(cfg))
    if args.experiment in ("all","input_pairwise"): outputs.append(exp1_pairwise.run(cfg))
    if args.experiment in ("all","input_nonlinear"): outputs.append(nonlinear_network_validation.run_exp1(cfg))
    if args.experiment in ("all","nonlinear_hopf"): outputs.append(exp2.run(cfg))
    if args.experiment in ("all","structural_kernel"): outputs.append(exp3.run(cfg))
    if args.experiment in ("all","structural_nonlinear"): outputs.append(nonlinear_network_validation.run_exp3(cfg))
    if args.experiment == "finite_window": outputs.append(finite_window_validation.run(cfg))
    if cfg.get("mode")=="smoke":
        report=Path(__file__).resolve().parents[1]/"smoke_test_report.md"
        report.write_text("# Smoke test report\n\nStatus: completed\n\n"+"\n".join(f"- {p}" for p in outputs)+f"\n\nRuntime: {time.time()-started:.1f} s\n")
    print("\n".join(map(str,outputs)))


if __name__=="__main__": main()
