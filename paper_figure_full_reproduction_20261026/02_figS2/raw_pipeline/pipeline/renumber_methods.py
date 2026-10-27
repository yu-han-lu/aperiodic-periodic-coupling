from pathlib import Path
import re


path = Path("/PORTABLE_HOME/Documents/Codex/2026-09-05/referenced-chatgpt-conversation-this-is-an/outputs/manuscript_second_revision_20260910/methods_working.md")
text = path.read_text(encoding="utf-8")
head, tail = text.split("# 4 Two-dimensional dynamical models", 1)

mapping = {
    12: 17,
    13: 18,
    14: 19,
    15: 20,
    16: 21,
    17: 22,
    18: 23,
    19: 24,
    21: 25,
    22: 26,
    23: 27,
    24: 28,
    25: 29,
    26: 30,
    27: 31,
    28: 32,
    29: 33,
    30: 34,
    31: 35,
}


def replace(match: re.Match[str]) -> str:
    old = int(match.group(1))
    return f"M{mapping.get(old, old)}"


tail = re.sub(r"M(1[2-9]|2[0-9]|3[0-1])", replace, tail)
path.write_text(head + "# 4 Two-dimensional dynamical models" + tail, encoding="utf-8")
