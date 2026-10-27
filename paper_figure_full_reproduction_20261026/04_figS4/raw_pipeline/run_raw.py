#!/usr/bin/env python3
"""Generate the registered Hopf observation-model and exact nonlinear outputs."""

from __future__ import annotations

import argparse
import os
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "project" / "experiments" / "pap_theory_validation"
PACKAGE = PACKAGE_ROOT / "pap_validation"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    return env


def invoke(*args: str) -> None:
    subprocess.run([sys.executable, "-m", *args], check=True, cwd=ROOT, env=environment())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--observation-model", action="store_true")
    parser.add_argument("--exact-b", action="store_true")
    parser.add_argument("--exact-f", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--walltime-minutes", type=float)
    parser.add_argument("--force", action="store_true")
    ns = parser.parse_args()
    selected = ns.observation_model or ns.exact_b or ns.exact_f or ns.full
    if not selected:
        for name in ("two_dimensional_ode_validation.py", "nonlinear_hopf_exact.py"):
            py_compile.compile(str(PACKAGE / name), doraise=True)
        invoke("pap_validation.nonlinear_hopf_exact", "--help")
        print("Fig. S4 raw pipeline check: PASS")
        return
    if ns.full or ns.observation_model:
        args = [
            "pap_validation.two_dimensional_ode_validation",
            "--model",
            "hopf",
        ]
        if ns.force:
            args.append("--force")
        invoke(*args)
    for panel, enabled in (("B", ns.full or ns.exact_b), ("F", ns.full or ns.exact_f)):
        if not enabled:
            continue
        args = ["pap_validation.nonlinear_hopf_exact", "--panel", panel]
        if ns.max_items is not None:
            args += ["--max-items", str(ns.max_items)]
        if ns.walltime_minutes is not None:
            args += ["--walltime-minutes", str(ns.walltime_minutes)]
        invoke(*args)


if __name__ == "__main__":
    main()
