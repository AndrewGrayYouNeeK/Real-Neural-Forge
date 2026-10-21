"""Dependency-free SVG charts for the conflict-interval report."""

from __future__ import annotations

import math
from dataclasses import dataclass


def svg_histogram(
    values: list[float],
    title: str,
    xlabel: str,
    *,
    bins: int = 24,
    exponential_mean: float | None = None,
    width: int = 720,
    height: int = 420,
) -> str:
    """Histogram of ``values``. An exponential pdf is overlaid when a mean is given."""
    if not values:
        return _empty(title, width, height)
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        hi = lo + 1.0
    width_bin = (hi - lo) / bins
    counts = [0] * bins
    for value in values:
        index = min(bins - 1, int((value - lo) / width_bin))
        counts[index] += 1
    peak = max(counts) or 1
    plot = _canvas(title, xlabel, "count", width, height)
    bar_w = plot.inner_w / bins
    parts = [plot.open]
    for index, count in enumerate(counts):
        bar_h = plot.inner_h * count / peak
        x = plot.left + index * bar_w
        y = plot.bottom - bar_h
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 1:.1f}" '
            f'height="{bar_h:.1f}" fill="#2c6e8a"/>'
        )
    if exponential_mean and exponential_mean > 0:
        rate = 1.0 / exponential_mean
        scale = (len(values) * width_bin) * (plot.inner_h / peak)
        coords = []
        for step in range(bins * 4):
            value = lo + (hi - lo) * step / (bins * 4 - 1)
            density = rate * math.exp(-rate * value) * scale
            x = plot.left + plot.inner_w * (value - lo) / (hi - lo)
            y = plot.bottom - min(plot.inner_h, density)
            coords.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polyline fill="none" stroke="#b45309" stroke-width="2" points="{" ".join(coords)}"/>'
        )
    parts.append(plot.close)
    return "".join(parts)


def svg_scatter(
    xs: list[float],
    ys: list[float],
    title: str,
    xlabel: str,
    ylabel: str,
    *,
    width: int = 720,
    height: int = 420,
) -> str:
    """A point series. A line connects points that are already in order."""
    if not xs or len(xs) != len(ys):
        return _empty(title, width, height)
    plot = _canvas(title, xlabel, ylabel, width, height)
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_max == x_min:
        x_max = x_min + 1
    if y_max == y_min:
        y_max = y_min + 1
    parts = [plot.open]
    coords = []
    for x_value, y_value in zip(xs, ys, strict=True):
        x = plot.left + plot.inner_w * (x_value - x_min) / (x_max - x_min)
        y = plot.bottom - plot.inner_h * (y_value - y_min) / (y_max - y_min)
        coords.append(f"{x:.1f},{y:.1f}")
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#1f4e79"/>')
    parts.append(
        f'<polyline fill="none" stroke="#1f4e79" stroke-width="1" points="{" ".join(coords)}"/>'
    )
    parts.append(plot.close)
    return "".join(parts)


def svg_km(
    curves: list[tuple[str, list[float], list[float]]],
    title: str,
    *,
    width: int = 720,
    height: int = 420,
) -> str:
    """Step-function Kaplan-Meier curves. Each item is ``(name, times, survival)``."""
    plot = _canvas(title, "days", "survival", width, height)
    colors = ("#1f4e79", "#b45309", "#3f6b4d", "#7c3a6e")
    parts = [plot.open]
    max_time = max((max(times) for _name, times, _s in curves if times), default=1.0)
    if max_time <= 0:
        max_time = 1.0
    for index, (name, times, survival) in enumerate(curves):
        if not times:
            continue
        color = colors[index % len(colors)]
        coords = [f"{plot.left:.1f},{plot.top:.1f}"]
        for stamp, level in zip(times, survival, strict=True):
            x = plot.left + plot.inner_w * stamp / max_time
            y = plot.bottom - plot.inner_h * level
            coords.append(f"{x:.1f},{coords_y(coords, y)}")
            coords.append(f"{x:.1f},{y:.1f}")
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(coords)}"/>'
        )
        legend_y = plot.top + 16 + index * 16
        parts.append(
            f'<text x="{plot.left + 8}" y="{legend_y}" fill="{color}" font-size="12">{_escape(name)}</text>'
        )
    parts.append(plot.close)
    return "".join(parts)


def svg_bars(
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
    *,
    width: int = 720,
    height: int = 420,
) -> str:
    """Vertical bars. ``labels`` are drawn under every bar when there are few."""
    if not labels or len(labels) != len(values):
        return _empty(title, width, height)
    plot = _canvas(title, "", ylabel, width, height)
    peak = max(values) or 1.0
    bar_w = plot.inner_w / len(values)
    parts = [plot.open]
    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        bar_h = plot.inner_h * value / peak
        x = plot.left + index * bar_w
        y = plot.bottom - bar_h
        parts.append(
            f'<rect x="{x + 2:.1f}" y="{y:.1f}" width="{max(1.0, bar_w - 4):.1f}" '
            f'height="{bar_h:.1f}" fill="#2c6e8a"/>'
        )
        if len(labels) <= 16:
            parts.append(
                f'<text x="{x + bar_w / 2:.1f}" y="{plot.bottom + 16}" '
                f'text-anchor="middle" font-size="11" fill="#333">{_escape(label)}</text>'
            )
    parts.append(plot.close)
    return "".join(parts)


def coords_y(coords: list[str], y: float) -> str:
    """Y of the previous point, so the Kaplan-Meier step stays flat until ``y``."""
    if not coords:
        return f"{y:.1f}"
    return coords[-1].split(",")[1]


@dataclass(frozen=True)
class _Canvas:
    open: str
    close: str
    left: float
    right: float
    top: float
    bottom: float
    inner_w: float
    inner_h: float


def _canvas(title: str, xlabel: str, ylabel: str, width: int, height: int) -> _Canvas:
    left, top = 64.0, 36.0
    right, bottom = float(width - 20), float(height - 48)
    open_tag = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fff"/>'
        f'<text x="{width / 2}" y="22" text-anchor="middle" font-size="15" '
        f'font-family="sans-serif">{_escape(title)}</text>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="#222"/>'
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="#222"/>'
        f'<text x="{(left + right) / 2}" y="{height - 12}" text-anchor="middle" '
        f'font-size="12" font-family="sans-serif">{_escape(xlabel)}</text>'
        f'<text x="16" y="{(top + bottom) / 2}" font-size="12" font-family="sans-serif" '
        f'transform="rotate(-90 16 {(top + bottom) / 2})">{_escape(ylabel)}</text>'
    )
    return _Canvas(
        open=open_tag,
        close="</svg>",
        left=left,
        right=right,
        top=top,
        bottom=bottom,
        inner_w=right - left,
        inner_h=bottom - top,
    )


def _empty(title: str, width: int, height: int) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<text x="20" y="40">{_escape(title)} (no data)</text></svg>'
    )


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
