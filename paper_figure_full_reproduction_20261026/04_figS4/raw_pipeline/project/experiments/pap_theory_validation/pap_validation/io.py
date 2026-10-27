from __future__ import annotations

import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import yaml


def load_config(path: str | Path) -> dict:
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    cfg["config_path"] = str(path)
    return cfg


def prepare_output(cfg: dict, experiment: str) -> Path:
    root = Path(__file__).resolve().parents[3]
    out = root / cfg["output_dir"] / experiment
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg["config_path"], out / "config.yaml")
    (out / "environment.txt").write_text(
        f"python={sys.version}\nplatform={platform.platform()}\nnumpy={np.__version__}\nscipy={scipy.__version__}\n"
    )
    try:
        status = subprocess.run(["git", "status", "--short"], cwd=root, capture_output=True, text=True).stdout
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip() or "NO_HEAD"
    except OSError:
        status, commit = "git unavailable", "unknown"
    (out / "git_info.txt").write_text(f"commit={commit}\nstatus:\n{status}")
    return out
