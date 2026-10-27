from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def scripts() -> list[Path]:
    result = []
    for folder in ("01_main_fig2_and_figS5", "02_figS2", "03_figS3", "04_figS4"):
        result.extend(sorted((ROOT / folder).glob("*/plot.py")))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Main Fig. 2 and Figs. S2–S5")
    parser.add_argument("--only", help="substring used to select panel paths")
    args = parser.parse_args()
    chosen = [p for p in scripts() if not args.only or args.only in str(p.relative_to(ROOT))]
    failures = []
    for pos, script in enumerate(chosen, 1):
        label = str(script.parent.relative_to(ROOT))
        print(f"[{pos:02d}/{len(chosen):02d}] {label}", flush=True)
        run = subprocess.run([sys.executable, str(script)], cwd=script.parent)
        if run.returncode:
            failures.append({"panel": label, "returncode": run.returncode})
    report = {"panels_attempted": len(chosen), "failures": failures}
    (ROOT / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    if failures:
        raise SystemExit(f"{len(failures)} panel scripts failed; see run_report.json")
    print(f"PASS: generated {len(chosen)} panel groups", flush=True)


if __name__ == "__main__":
    main()
