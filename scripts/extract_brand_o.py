"""Cut the round "O" out of a Nous wordmark JPG into a transparent PNG.

The O is the only saturated (coloured) part of the artwork, so its bounding
box is found from pixel saturation. Output goes to frontend/brand/nous-o-<name>.png.
Usage: python scripts/extract_brand_o.py frontend/brand/nous-wordmark-cosmos-galaxy.jpg cosmos-galaxy
"""

from __future__ import annotations

import colorsys
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "frontend" / "brand"


def extract(source: Path, name: str, size: int = 384, margin: float = 1.1) -> Path:
    image = Image.open(source).convert("RGB")
    pixels = image.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(0, image.height, 2):
        for x in range(0, image.width, 2):
            r, g, b = pixels[x, y]
            _, sat, val = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if sat > 0.45 and val > 0.35:
                xs.append(x)
                ys.append(y)
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    radius = max(max(xs) - min(xs), max(ys) - min(ys)) / 2 * margin
    box = (int(cx - radius), int(cy - radius), int(cx + radius), int(cy + radius))
    crop = image.crop(box).resize((size, size), Image.LANCZOS).convert("RGBA")
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).ellipse((2, 2, size - 2, size - 2), fill=255)
    crop.putalpha(mask.filter(ImageFilter.GaussianBlur(size * 0.01)))
    out = OUT / f"nous-o-{name}.png"
    crop.save(out, optimize=True)
    return out


if __name__ == "__main__":
    print(extract(Path(sys.argv[1]), sys.argv[2]))
