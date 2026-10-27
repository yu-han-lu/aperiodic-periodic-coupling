from pathlib import Path
from PIL import Image, ImageOps, ImageDraw


source = Path("/PORTABLE_HOME/Documents/Codex/2026-09-05/referenced-chatgpt-conversation-this-is-an/outputs/manuscript_second_revision_20260910/render_methods_cycle1b")
pages = sorted(source.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))
for sheet_index in range(0, len(pages), 4):
    batch = pages[sheet_index:sheet_index + 4]
    tiles = []
    for path in batch:
        image = Image.open(path).convert("RGB")
        image.thumbnail((700, 990))
        canvas = Image.new("RGB", (720, 1040), "white")
        canvas.paste(image, ((720 - image.width) // 2, 30))
        ImageDraw.Draw(canvas).text((15, 10), path.stem, fill="black")
        tiles.append(canvas)
    sheet = Image.new("RGB", (1440, 2080), "#d9d9d9")
    for idx, tile in enumerate(tiles):
        sheet.paste(tile, ((idx % 2) * 720, (idx // 2) * 1040))
    sheet.save(source / f"contact-{sheet_index // 4 + 1}.png")
