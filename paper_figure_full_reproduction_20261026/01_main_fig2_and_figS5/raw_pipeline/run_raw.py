#!/usr/bin/env python3
"""Dispatch the registered whole-brain raw-data generators.

No numerical setting is altered here. The launcher only supplies portable
paths and invokes the archived entry points with their registered arguments.
"""

from __future__ import annotations

import argparse
import os
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT / "workspace"
RERUN = ROOT / "rerun"
PROJECT = ROOT / "project"

EXP1 = WORKSPACE / "outputs/wholebrain_exp1_resting_heun_20260909/run_exp1_resting_heun.py"
EXP2 = WORKSPACE / "outputs/full_network_heun_20260908_retry/run_wholebrain_heun_retry.py"
FIRST_ORDER = RERUN / "outputs/first_order_rerun_20260914/run_first_order_rerun.py"
V6 = RERUN / "outputs/i2_predefined_exp1_ensemble_20260915/run_i2_predefined_exp1_ensemble.py"
ADD2 = WORKSPACE / "outputs/i2_add_2nodes_v1_v6_20260930/run_add_2nodes.py"
COUNT = WORKSPACE / "outputs/deletion_count_direct_nondirect_20260910/run_full55_count_direct_nondirect.py"
FIG2I = WORKSPACE / "outputs/fig2i_fig2j_matched_single_realization_20261008/run_fig2i_matched_calibration.py"
FIG2J = WORKSPACE / "outputs/fig2i_fig2j_matched_single_realization_20261008/run_fig2j_single_realization_Ce.py"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PAP_RAW_ROOT"] = str(ROOT)
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(PROJECT / "experiments/pap_theory_validation"),
            str(WORKSPACE / "outputs"),
            str(ROOT / "vendor/nature_figure/scripts"),
            env.get("PYTHONPATH", ""),
        ]
    )
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    return env


def run(script: Path, *args: str) -> None:
    subprocess.run(
        [sys.executable, str(script), *args],
        check=True,
        cwd=script.parent,
        env=environment(),
    )


def check() -> None:
    scripts = (EXP1, EXP2, FIRST_ORDER, V6, ADD2, COUNT, FIG2I, FIG2J)
    for path in scripts:
        py_compile.compile(str(path), doraise=True)
        subprocess.run(
            [sys.executable, str(path), "--help"],
            check=True,
            cwd=path.parent,
            env=environment(),
            stdout=subprocess.DEVNULL,
        )
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "from pap_validation.core import load_model; "
                "from pap_validation.io import load_config; "
                "from pathlib import Path; import os; "
                "r=Path(os.environ['PAP_RAW_ROOT']); "
                "c=load_config(r/'project/experiments/pap_theory_validation/configs/full.yaml'); "
                "m=load_model(r/'project'/c['model_dir']); "
                "assert len(m.areas)==55; print('55-area model load: PASS')"
            ),
        ],
        check=True,
        cwd=ROOT,
        env=environment(),
    )
    print("Main Fig. 2/Fig. S5 raw pipeline check: PASS")


def run_fig2bc(workers: int, force: bool) -> None:
    args = ["--assert-input", "--pilot", "--exp1", "--resting", "--finalize", "--workers", str(workers)]
    if force:
        args.append("--force")
    run(EXP1, *args)


def run_fig2fg(workers: int, force: bool) -> None:
    args = ["--assert-input", "--pilot", "--full", "--finalize", "--workers", str(workers)]
    if force:
        args.append("--force")
    run(EXP2, *args)
    run(FIRST_ORDER, "--exp2", "--workers", str(workers))


def run_fig2i(workers: int, force: bool) -> None:
    # Generate the unchanged version-6 held-out nonlinear Heun features first.
    v6_args = ["--simulate", "--assemble", "--workers", str(workers)]
    if force:
        v6_args.append("--force")
    run(V6, *v6_args)
    # Generate the registered two-node extension under the same held-out design.
    run(ADD2, "--v6-simulate", "--workers", str(workers))
    run(ADD2, "--v6-assemble", "--workers", str(workers))
    # Finally generate the matched calibration realizations and accepted panel data.
    args = ["--all", "--workers", str(workers)]
    if force:
        args.append("--force")
    run(FIG2I, *args)


def run_fig2j(workers: int, force: bool) -> None:
    # Recreate the frozen deletion-count inputs, transfer/calibration states and
    # first-order maps before the accepted single-realization held-out rerun.
    run(COUNT, "--prepare")
    run(COUNT, "--worker-start", "0", "--worker-end", "61")
    run(COUNT, "--assemble")
    code = (
        "import importlib.util, pathlib; "
        f"p=pathlib.Path({str(FIRST_ORDER)!r}); "
        "s=importlib.util.spec_from_file_location('first_order_portable',p); "
        "m=importlib.util.module_from_spec(s); s.loader.exec_module(m); m.run_i1()"
    )
    subprocess.run([sys.executable, "-c", code], check=True, cwd=FIRST_ORDER.parent, env=environment())
    args = ["--all", "--workers", str(workers)]
    if force:
        args.append("--force")
    run(FIG2J, *args)


def run_figs5(max_items: int | None, walltime: float | None) -> None:
    args = ["pap_validation.nonlinear_resimulations_exact", "--panel", "wb_9x9"]
    if max_items is not None:
        args += ["--max-items", str(max_items)]
    if walltime is not None:
        args += ["--walltime-minutes", str(walltime)]
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT, env=environment())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fig2bc", action="store_true")
    parser.add_argument("--fig2fg", action="store_true")
    parser.add_argument("--fig2i", action="store_true")
    parser.add_argument("--fig2j", action="store_true")
    parser.add_argument("--figS5", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--walltime-minutes", type=float)
    ns = parser.parse_args()
    if not any((ns.fig2bc, ns.fig2fg, ns.fig2i, ns.fig2j, ns.figS5, ns.all)):
        check()
        return
    if ns.all or ns.fig2bc:
        run_fig2bc(ns.workers, ns.force)
    if ns.all or ns.fig2fg:
        run_fig2fg(ns.workers, ns.force)
    if ns.all or ns.fig2i:
        run_fig2i(ns.workers, ns.force)
    if ns.all or ns.fig2j:
        run_fig2j(ns.workers, ns.force)
    if ns.all or ns.figS5:
        run_figs5(ns.max_items, ns.walltime_minutes)


if __name__ == "__main__":
    main()
