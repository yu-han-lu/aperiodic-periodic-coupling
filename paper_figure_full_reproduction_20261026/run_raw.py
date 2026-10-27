#!/usr/bin/env python3
"""Dispatch the registered raw-data generators without changing their settings."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
GROUPS = {
    "main": ROOT / "01_main_fig2_and_figS5" / "raw_pipeline" / "run_raw.py",
    "s2": ROOT / "02_figS2" / "raw_pipeline" / "run_raw.py",
    "s3": ROOT / "03_figS3" / "raw_pipeline" / "run_raw.py",
    "s4": ROOT / "04_figS4" / "raw_pipeline" / "run_raw.py",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a raw-simulation/estimation pipeline. Full runs are intentionally "
            "not the default because several registered analyses are large."
        )
    )
    parser.add_argument("group", choices=GROUPS)
    parser.add_argument("args", nargs=argparse.REMAINDER)
    ns = parser.parse_args()
    extra = ns.args or ["--check"]
    raise SystemExit(subprocess.call([sys.executable, str(GROUPS[ns.group]), *extra]))


if __name__ == "__main__":
    main()
