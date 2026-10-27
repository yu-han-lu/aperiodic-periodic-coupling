#!/usr/bin/env python3
"""Generate the registered linear two-dimensional model outputs."""

from __future__ import annotations

import argparse
import os
import py_compile
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = ROOT / "project" / "experiments" / "pap_theory_validation"
MODULE = PACKAGE_ROOT / "pap_validation" / "two_dimensional_ode_validation.py"


def environment() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PACKAGE_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    return env


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--force", action="store_true")
    ns = parser.parse_args()
    if not ns.full:
        py_compile.compile(str(MODULE), doraise=True)
        subprocess.run(
            [sys.executable, "-c", "import pap_validation.two_dimensional_ode_validation"],
            check=True,
            cwd=ROOT,
            env=environment(),
        )
        print("Fig. S3 raw pipeline check: PASS")
        return
    cmd = [
        sys.executable,
        "-m",
        "pap_validation.two_dimensional_ode_validation",
        "--model",
        "damped",
    ]
    if ns.force:
        cmd.append("--force")
    subprocess.run(cmd, check=True, cwd=ROOT, env=environment())


if __name__ == "__main__":
    main()
