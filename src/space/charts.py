"""Small SVG charts of a validation window.

The training environment does not depend on matplotlib. These files are the
charts the fetch/train commands write under ``data/space/charts/``.
"""

from __future__ import annotations

from pathlib import Path


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _stride(values: list[float], max_points: int) -> list[float]:
    if len(values) <= max_points:
        return values
    if max_points < 2:
        raise ValueError("max_points must be at least 2")
    last = len(values) - 1
    return [values[round(i * last / (max_points - 1))] for i in range(max_points)]


def write_forecast_svg(
    path: str | Path,
    *,
    title: str,
    series: list[tuple[str, str, list[float]]],
    hline: float | None = None,
    hline_label: str | None = None,
    max_points: int = 480,
) -> Path:
    """Write one line chart. ``series`` entries are ``(name, color, values)``."""
    if not series:
        raise ValueError("chart needs at least one series")
    strided = [
        (name, color, _stride(list(values), max_points))
        for name, color, values in series
    ]
    width_n = len(strided[0][2])
    if width_n < 2:
        raise ValueError("chart series need at least two points")
    if any(len(values) != width_n for _name, _color, values in strided):
        raise ValueError("chart series must have the same length")

    flat = [value for _name, _color, values in strided for value in values]
    if hline is not None:
        flat.append(hline)
    low = min(flat)
    high = max(flat)
    if high == low:
        high = low + 1.0

    width = 960
    height = 420
    left, right, top, bottom = 56, 20, 48, 36
    plot_w = width - left - right
    plot_h = height - top - bottom

    def x_of(index: int) -> float:
        return left + plot_w * index / (width_n - 1)

    def y_of(value: float) -> float:
        return top + plot_h * (1.0 - (value - low) / (high - low))

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<text x="{left}" y="28" font-family="sans-serif" font-size="16" '
        f'fill="#0f172a">{_escape(title)}</text>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="#94a3b8" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        'stroke="#94a3b8" stroke-width="1"/>',
        f'<text x="{left - 8}" y="{top + 4}" text-anchor="end" font-family="sans-serif" '
        f'font-size="11" fill="#475569">{high:.3g}</text>',
        f'<text x="{left - 8}" y="{top + plot_h}" text-anchor="end" '
        f'font-family="sans-serif" font-size="11" fill="#475569">{low:.3g}</text>',
    ]
    if hline is not None:
        y_line = y_of(hline)
        label = hline_label or f"{hline:g}"
        parts.append(
            f'<line x1="{left}" y1="{y_line:.1f}" x2="{left + plot_w}" '
            f'y2="{y_line:.1f}" stroke="#b91c1c" stroke-dasharray="4 4" '
            'stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{left + plot_w}" y="{y_line - 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="11" fill="#b91c1c">{_escape(label)}</text>'
        )
    legend_x = left
    for name, color, values in strided:
        points = " ".join(
            f"{x_of(index):.1f},{y_of(value):.1f}" for index, value in enumerate(values)
        )
        parts.append(
            f'<polyline fill="none" stroke="{_escape(color)}" stroke-width="1.5" '
            f'points="{points}"/>'
        )
        parts.append(
            f'<rect x="{legend_x}" y="{height - 22}" width="12" height="8" '
            f'fill="{_escape(color)}"/>'
        )
        parts.append(
            f'<text x="{legend_x + 16}" y="{height - 14}" font-family="sans-serif" '
            f'font-size="12" fill="#0f172a">{_escape(name)}</text>'
        )
        legend_x += 16 + 8 * len(name) + 24
    parts.append("</svg>")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return destination
