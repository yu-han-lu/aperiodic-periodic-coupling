from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter


ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "two_dimensional_ode_validation"
SHARED = ROOT / "experiments" / "pap_theory_validation" / "outputs" / "fooof_split_surrogate"
OUTPUT = ROOT / "output" / "pdf"


def _merge(paths: list[Path], output: Path) -> None:
    writer = PdfWriter()
    for path in paths:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        writer.write(stream)


def main() -> None:
    common_tail = [SHARED / "fooof_vs_split_surrogate_validation.pdf"]
    specs = {
        "damped": OUTPUT / "linear_2d_complete_results.pdf",
        "hopf": OUTPUT / "nonlinear_2d_complete_results.pdf",
    }
    for model_key, output in specs.items():
        model_dir = SOURCE / model_key
        phase_dir = SOURCE / "phase_plane_and_pa" / model_key
        pages = [
            model_dir / "figures" / f"{model_key}_figure_A_to_G.pdf",
            model_dir / "panels" / "D_eigenvalue_transitions.pdf",
            phase_dir / f"{model_key}_phase_plane_three_regimes.pdf",
            phase_dir / f"{model_key}_complex_regime_PA_scatter.pdf",
            *common_tail,
        ]
        missing = [str(path) for path in pages if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing PDF pages: {missing}")
        _merge(pages, output)
        print(output)


if __name__ == "__main__":
    main()
