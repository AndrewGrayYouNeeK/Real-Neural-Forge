"""Build the Nous brand SVGs (wordmark, Flower of Life mark, social card).

The Flower of Life is drawn exactly: 19 circles of radius r on a hexagonal
lattice (centre, 6 at distance r, 6 at r*sqrt(3), 6 at 2r) inside an outer
circle of radius 3r. Run: python scripts/build_brand.py

Swapping the "O": the letters N, U, S are SVG paths; the O is a slot.
Put one word in frontend/brand/o-choice.txt and rerun this script:
  flower         exact geometric Flower of Life (pure SVG)
  cosmos-flower  frontend/brand/nous-o-cosmos-flower.png (Flower over a galaxy)
  cosmos-galaxy  frontend/brand/nous-o-cosmos-galaxy.png (pure galaxy)
Any other name uses frontend/brand/nous-o-<name>.png (make one with
scripts/extract_brand_o.py). Pages only reference nous-wordmark.svg and
nous-mark.svg, so nothing else changes.
"""

from __future__ import annotations

import base64
import math
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "frontend" / "brand"
CYAN = "#3fe3ea"
WHITE = "#f5f7fa"


def flower_centres(cx: float, cy: float, r: float) -> list[tuple[float, float]]:
    points = [(cx, cy)]
    for ring, distance, offset in ((1, r, 0.0), (2, r * math.sqrt(3), 30.0), (3, 2 * r, 0.0)):
        del ring
        for k in range(6):
            angle = math.radians(offset + 60 * k)
            points.append((cx + distance * math.cos(angle), cy + distance * math.sin(angle)))
    return points


def flower(cx: float, cy: float, r: float, stroke: float, glow: str | None = "glow") -> str:
    circles = "".join(
        f'<circle cx="{x:.3f}" cy="{y:.3f}" r="{r:.3f}"/>' for x, y in flower_centres(cx, cy, r)
    )
    outer = f'<circle cx="{cx:.3f}" cy="{cy:.3f}" r="{3 * r:.3f}" stroke-width="{stroke * 1.6:.3f}"/>'
    clip_id = f"fol-clip-{int(cx)}-{int(cy)}"
    filt = f' filter="url(#{glow})"' if glow else ""
    return (
        f'<clipPath id="{clip_id}"><circle cx="{cx:.3f}" cy="{cy:.3f}" r="{3 * r:.3f}"/></clipPath>'
        f'<g fill="none" stroke="{CYAN}" stroke-width="{stroke:.3f}"{filt}>'
        f'<g clip-path="url(#{clip_id})">{circles}</g>{outer}</g>'
    )


def glow_filter(blur: float) -> str:
    return (
        '<filter id="glow" x="-30%" y="-30%" width="160%" height="160%">'
        f'<feGaussianBlur in="SourceGraphic" stdDeviation="{blur}" result="b"/>'
        '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>'
        "</filter>"
    )


def o_choice() -> str:
    if len(sys.argv) > 1:
        return sys.argv[1].strip()
    choice_file = OUT / "o-choice.txt"
    if choice_file.exists():
        return choice_file.read_text(encoding="utf-8").strip() or "flower"
    return "flower"


def o_slot(cx: float, cy: float, radius: float, stroke: float, choice: str) -> str:
    """The O: exact Flower of Life SVG, or an embedded round image."""
    if choice == "flower":
        return flower(cx, cy, radius / 3, stroke)
    png = OUT / f"nous-o-{choice}.png"
    data = base64.b64encode(png.read_bytes()).decode("ascii")
    # The extracted PNG has a 10% margin for the glow around the ring.
    size = radius * 2 * 1.02
    return (
        f'<image x="{cx - size / 2:.2f}" y="{cy - size / 2:.2f}" width="{size:.2f}" '
        f'height="{size:.2f}" href="data:image/png;base64,{data}"/>'
    )


def letters(offset_x: float = 0.0) -> str:
    """N, U, S as geometric shapes on a 40..160 cap height, stroke 18."""
    n = (
        "30,160 30,40 50,40 112,128 112,40 130,40 130,160 110,160 48,72 48,160"
    )
    n_pts = " ".join(
        f"{float(x) + offset_x:g},{y}" for x, y in (p.split(",") for p in n.split())
    )
    ux = 290 + offset_x
    sx = offset_x
    return (
        f'<polygon fill="{WHITE}" points="{n_pts}"/>'
        f'<path fill="none" stroke="{WHITE}" stroke-width="18" '
        f'd="M{ux + 9} 40 V110 A41 41 0 0 0 {ux + 91} 110 V40"/>'
        f'<path fill="none" stroke="{WHITE}" stroke-width="18" stroke-linejoin="round" '
        f'd="M{sx + 528} 49 H{sx + 454} A25.5 25.5 0 0 0 {sx + 454} 100 H{sx + 496} '
        f'A25.5 25.5 0 0 1 {sx + 496} 151 H{sx + 422}"/>'
    )


def wordmark(choice: str) -> str:
    body = letters() + o_slot(210, 100, 72, 2.4, choice)
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 560 200" role="img" '
        'aria-label="Nous"><title>Nous</title>'
        f"<defs>{glow_filter(2.2)}</defs>{body}</svg>\n"
    )


def mark(choice: str, size: int = 512, background: bool = True) -> str:
    centre = size / 2
    r = size * 0.115
    stroke = max(size * 0.012, 1.2)
    bg = (
        f'<rect width="{size}" height="{size}" rx="{size * 0.22:.1f}" fill="#000"/>'
        if background
        else ""
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {size} {size}" role="img" '
        f'aria-label="Nous"><title>Nous</title><defs>{glow_filter(size * 0.008)}</defs>'
        f"{bg}{o_slot(centre, centre, 3 * r, stroke, choice)}</svg>\n"
    )


def social(choice: str) -> str:
    # 1200x630 card: wordmark scaled 1.6x, centred, tagline below.
    scale = 1.6
    tx = (1200 - 560 * scale) / 2
    ty = 150
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" role="img" '
        'aria-label="Nous — forecast anything, just ask">'
        f"<defs>{glow_filter(2.2)}</defs>"
        '<rect width="1200" height="630" fill="#000"/>'
        f'<g transform="translate({tx:.1f} {ty}) scale({scale})">'
        f"{letters()}{o_slot(210, 100, 72, 2.4, choice)}</g>"
        '<text x="600" y="520" text-anchor="middle" fill="#9aa3b2" '
        'font-family="Helvetica, Arial, sans-serif" font-size="40" letter-spacing="2">'
        "Forecast anything. Just ask.</text></svg>\n"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    choice = o_choice()
    (OUT / "nous-wordmark.svg").write_text(wordmark(choice), encoding="utf-8")
    (OUT / "nous-mark.svg").write_text(mark(choice), encoding="utf-8")
    (OUT / "nous-mark-plain.svg").write_text(
        mark(choice, background=False), encoding="utf-8"
    )
    (OUT / "nous-og.svg").write_text(social(choice), encoding="utf-8")
    # Always keep the pure-SVG Flower of Life versions too.
    (OUT / "nous-wordmark-flower.svg").write_text(wordmark("flower"), encoding="utf-8")
    (OUT / "nous-mark-flower.svg").write_text(mark("flower"), encoding="utf-8")
    print(f"Built Nous brand with O = {choice}")


if __name__ == "__main__":
    main()
