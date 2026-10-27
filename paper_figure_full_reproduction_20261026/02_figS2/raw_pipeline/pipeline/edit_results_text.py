from pathlib import Path


path = Path("/PORTABLE_HOME/Documents/Codex/2026-09-05/referenced-chatgpt-conversation-this-is-an/outputs/manuscript_second_revision_20260910/results_working.md")
text = path.read_text(encoding="utf-8")
replacements = {
    "We developed first-order stable-filter theory": "We next developed first-order stable-filter theory",
    "The high agreement across paths indicates that the prediction is not restricted to a selected input–output pair.": "The high agreement across paths indicated that the prediction was not restricted to a selected input–output pair.",
    "showing that graph adjacency alone does not explain deletion-induced changes.": "indicating that graph adjacency alone did not explain deletion-induced changes.",
}
for old, new in replacements.items():
    if old not in text:
        raise RuntimeError(f"Missing expected text: {old}")
    text = text.replace(old, new, 1)
path.write_text(text, encoding="utf-8")
