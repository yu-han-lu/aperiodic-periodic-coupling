#!/usr/bin/env python3
"""Portable entry point for the registered Supplementary Figure 2 analysis."""

from __future__ import annotations

import argparse
import importlib
import os
import py_compile
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PIPELINE = ROOT / "pipeline"
PAP = ROOT / "project" / "experiments" / "pap_theory_validation" / "pap_validation"
FIGURE_SCRIPTS = ROOT / "vendor" / "nature_figure" / "scripts"
FOOOF_PACKAGE = ROOT / "vendor" / "fooof_1_1_1"
OUTPUTS = ROOT / "outputs"


def configure() -> None:
    os.environ["PAP_RAW_ROOT"] = str(ROOT)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
    for path in (PIPELINE, PAP, FIGURE_SCRIPTS, FOOOF_PACKAGE):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))


def bind_registered_inputs(broad) -> None:
    """Point the archived modules at the bundled, read-only reference inputs."""
    data = ROOT / "external_data"
    broad.local.shared.fixed.DATA = data
    broad.local.shared.researcher.DATA = data
    broad.local.shared.frozen.EMPIRICAL_KNEE_FILE = data / "single_knee_data.mat"


def check() -> None:
    configure()
    required = [
        "build_broad_total_covariance_3x3.py",
        "rerun_figs2_def_equal_setting_weights.py",
        "build_fooof_split_local_perturbation_3x3.py",
        "pilot_four_parameter_plus_realization_theory.py",
    ]
    for name in required:
        py_compile.compile(str(PIPELINE / name), doraise=True)
    broad = importlib.import_module("build_broad_total_covariance_3x3")
    bind_registered_inputs(broad)
    broad.local.initialize()
    if len(broad.local.shared.frozen.load_empirical_knees()) != 3126:
        raise RuntimeError("The registered empirical-knee input is incomplete")
    importlib.import_module("rerun_figs2_def_equal_setting_weights")
    print("Fig. S2 raw pipeline check: PASS")


def full() -> None:
    configure()
    broad = importlib.import_module("build_broad_total_covariance_3x3")
    broad.OUTPUT = OUTPUTS / "broad_total_covariance"
    bind_registered_inputs(broad)
    broad.main()

    equal = importlib.import_module("rerun_figs2_def_equal_setting_weights")
    equal.SOURCE = broad.OUTPUT
    equal.SOURCE_CSV = broad.OUTPUT / "all_group_results.csv"
    equal.OUTPUT = OUTPUTS / "equal_setting_weights"
    equal.FOOOF_PACKAGE = FOOOF_PACKAGE
    equal.main()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--full", action="store_true")
    ns = parser.parse_args()
    if ns.full:
        full()
    else:
        check()


if __name__ == "__main__":
    main()
