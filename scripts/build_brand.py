"""Build the Nous raster brand assets from the source artwork.

Sources (committed): frontend/brand/source/nous-wordmark-cosmos-s.jpg and
frontend/brand/source/nous-cosmos-s-mark.jpg. The wordmark on the site is
HTML text "NOU" plus the cosmos S as an image, so the letters stay crisp.

Outputs in frontend/brand/:
  nous-s.png / nous-s.webp       the S, black keyed to transparency
  icon-16/32/180/192/512.png     app icon sizes cropped from the mark
  icon-512.webp, apple-touch-icon.png
  ../favicon.ico                 16/32/48
nous-og.png (1200x630) is a browser screenshot of the HTML wordmark + tagline.
Run with any Python that has Pillow: python3 scripts/build_brand.py
"""

from __future__ import annotations

import colorsys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "frontend"
BRAND = ROOT / "brand"
SOURCE = BRAND / "source"


def _s_box(image: Image.Image) -> tuple[int, int, int, int]:
    """Bounding box of the coloured S (the only saturated pixels)."""
    pixels = image.load()
    xs: list[int] = []
    ys: list[int] = []
    for y in range(0, image.height, 2):
        for x in range(0, image.width, 2):
            r, g, b = pixels[x, y]
            _, sat, val = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
            if sat > 0.4 and val > 0.25:
                xs.append(x)
                ys.append(y)
    pad = 6
    return (min(xs) - pad, min(ys) - pad, max(xs) + pad, max(ys) + pad)


def key_black(image: Image.Image) -> Image.Image:
    """Black → transparent, keeping glow: alpha = brightest channel."""
    rgba = Image.new("RGBA", image.size)
    src = image.convert("RGB").load()
    out = rgba.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = src[x, y]
            alpha = max(r, g, b)
            if alpha < 6:
                out[x, y] = (0, 0, 0, 0)
                continue
            scale = 255 / alpha
            out[x, y] = (
                min(255, round(r * scale)),
                min(255, round(g * scale)),
                min(255, round(b * scale)),
                alpha,
            )
    return rgba


def build_s() -> None:
    wordmark = Image.open(SOURCE / "nous-wordmark-cosmos-s.jpg").convert("RGB")
    crop = wordmark.crop(_s_box(wordmark))
    keyed = key_black(crop)
    keyed.save(BRAND / "nous-s.png", optimize=True)
    keyed.save(BRAND / "nous-s.webp", quality=90, method=6)


def build_icons() -> None:
    mark = Image.open(SOURCE / "nous-cosmos-s-mark.jpg").convert("RGB")
    side = min(mark.size)
    mark = mark.crop((0, 0, side, side))
    for size in (16, 32, 180, 192, 512):
        mark.resize((size, size), Image.LANCZOS).save(
            BRAND / f"icon-{size}.png", optimize=True
        )
    mark.resize((180, 180), Image.LANCZOS).save(
        BRAND / "apple-touch-icon.png", optimize=True
    )
    mark.resize((512, 512), Image.LANCZOS).save(
        BRAND / "icon-512.webp", quality=90, method=6
    )
    big = mark.resize((256, 256), Image.LANCZOS)
    big.save(ROOT / "favicon.ico", sizes=[(16, 16), (32, 32), (48, 48)])


if __name__ == "__main__":
    BRAND.mkdir(parents=True, exist_ok=True)
    build_s()
    build_icons()
    print("Built Nous brand assets in", BRAND)
