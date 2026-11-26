"""Build the Nous raster brand assets from the source artwork.

Source (committed): frontend/brand/source/nous-final-blackhole-o-swirl-s.jpg
(white N and U, black hole O, purple-cyan swirl S). On the site N and U are HTML text,
so they stay crisp; the O and S are images cut from the artwork.

Outputs in frontend/brand/:
  nous-o.png / nous-o.webp       the black hole O, black keyed to transparency
  nous-s.png / nous-s.webp       the swirl S, black keyed to transparency
  icon-16/32/180/192/512.png     app icons: the black hole O on black
  icon-512.webp, apple-touch-icon.png
  ../favicon.ico                 16/32/48
nous-og.png (1200x630) is a browser screenshot of the HTML wordmark + tagline.
Run with any Python that has Pillow: python3 scripts/build_brand.py
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent / "frontend"
BRAND = ROOT / "brand"
SOURCE = BRAND / "source"


def key_black(image: Image.Image) -> Image.Image:
    """Black → transparent, keeping glow: alpha = brightest channel."""
    rgba = Image.new("RGBA", image.size)
    src = image.convert("RGB").load()
    out = rgba.load()
    for y in range(image.height):
        for x in range(image.width):
            r, g, b = src[x, y]
            # Ramp from a small floor so JPEG noise in the black goes fully clear.
            alpha = max(0, round((max(r, g, b) - 14) * 255 / 241))
            if alpha < 6:
                out[x, y] = (0, 0, 0, 0)
                continue
            scale = 255 / max(r, g, b)
            out[x, y] = (
                min(255, round(r * scale)),
                min(255, round(g * scale)),
                min(255, round(b * scale)),
                alpha,
            )
    return rgba


# Boxes in the 1280x720 source. Each crop is vertically centred on the cap
# centre of the white N/U (y 281-429), so CSS can line them up with the text.
SOURCE_FILE = "nous-final-blackhole-o-swirl-s.jpg"
CAP_TOP, CAP_BOTTOM = 282, 429
O_BOX = (424, 235, 711, 476)
S_BOX = (862, 228, 1110, 483)
ICON_CENTER, ICON_SIDE = (567, 353), 282


def build_letters() -> None:
    art = Image.open(SOURCE / SOURCE_FILE).convert("RGB")
    for name, box in (("nous-o", O_BOX), ("nous-s", S_BOX)):
        keyed = key_black(art.crop(box))
        keyed.save(BRAND / f"{name}.png", optimize=True)
        keyed.save(BRAND / f"{name}.webp", quality=90, method=6)


def build_icons() -> None:
    art = Image.open(SOURCE / SOURCE_FILE).convert("RGB")
    cx, cy = ICON_CENTER
    half = ICON_SIDE // 2
    mark = art.crop((cx - half, cy - half, cx + half, cy + half))
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
    build_letters()
    build_icons()
    print("Built Nous brand assets in", BRAND)
