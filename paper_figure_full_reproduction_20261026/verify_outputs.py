from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
scripts = sorted(ROOT.glob("0*/*/plot.py"))
missing = []
for script in scripts:
    out = script.parent / "output"
    if not list(out.glob("*.svg")) or not list(out.glob("*.pdf")):
        missing.append(str(script.parent.relative_to(ROOT)))
report = {"panel_groups": len(scripts), "missing_vector_outputs": missing}
(ROOT / "verification_report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
if missing:
    raise SystemExit(1)
