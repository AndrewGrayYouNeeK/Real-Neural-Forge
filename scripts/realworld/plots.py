"""PNG copies of the conflict charts. Matplotlib is optional and not a dependency."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional local extra
    plt = None

from src.realworld.spells import Spell, kaplan_meier, spell_arrays


def write_png_copies(
    directory: Path,
    gap_values: list[float],
    gap_years: list[float],
    peace_conflict: list[Spell],
    peace_country: list[Spell],
    episode_lengths: list[Spell],
    decades: list[dict[str, float]],
    exponential_mean: float | None,
) -> None:
    """Write PNG siblings of the SVG charts when matplotlib is installed."""
    if plt is None:
        return
    _hist(directory / "onset_gap_histogram.png", gap_values, exponential_mean)
    _scatter(directory / "onset_gaps_over_time.png", gap_years, gap_values)
    _km(directory / "peace_kaplan_meier.png", peace_conflict, peace_country)
    observed = [float(spell.length_days) for spell in episode_lengths if spell.observed]
    _hist(
        directory / "duration_histogram.png", observed, None, xlabel="duration (days)"
    )
    _bars(directory / "gap_by_decade.png", decades)


def _hist(
    path: Path,
    values: list[float],
    exponential_mean: float | None,
    xlabel: str = "gap (days)",
) -> None:
    assert plt is not None
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.hist(values, bins=24, color="#2c6e8a", edgecolor="white")
    if exponential_mean and exponential_mean > 0:
        xs = np.linspace(min(values), max(values), 200)
        rate = 1.0 / exponential_mean
        width = (max(values) - min(values)) / 24
        axis.plot(xs, len(values) * width * rate * np.exp(-rate * xs), color="#b45309")
    axis.set_xlabel(xlabel)
    axis.set_ylabel("count")
    axis.set_title(path.stem.replace("_", " "))
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _scatter(path: Path, xs: list[float], ys: list[float]) -> None:
    assert plt is not None
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(xs, ys, color="#1f4e79", linewidth=1)
    axis.set_xlabel("year")
    axis.set_ylabel("days since the previous onset")
    axis.set_title("Inter-onset gap by year")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _km(path: Path, conflict: list[Spell], country: list[Spell]) -> None:
    assert plt is not None
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for spells, label, color in (
        (conflict, "conflict / incompatibility", "#1f4e79"),
        (country, "country", "#b45309"),
    ):
        lengths, observed = spell_arrays(spells)
        curve = kaplan_meier(lengths, observed)
        axis.step(
            curve["times"], curve["survival"], where="post", label=label, color=color
        )
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("days")
    axis.set_ylabel("survival")
    axis.set_title("Peace spells after fighting stops")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def write_mae_bars(
    path: Path, labels: list[str], values: list[float], title: str, ylabel: str
) -> None:
    """Bar chart of holdout MAE. No-op when matplotlib is not installed."""
    if plt is None or not labels:
        return
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, values, color="#2c6e8a")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.tick_params(axis="x", labelrotation=20)
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)


def _bars(path: Path, decades: list[dict[str, float]]) -> None:
    assert plt is not None
    figure, axis = plt.subplots(figsize=(8, 4.5))
    labels = [str(int(row["decade"])) for row in decades]
    axis.bar(labels, [row["mean"] for row in decades], color="#2c6e8a")
    axis.set_ylabel("mean gap (days)")
    axis.set_title("Mean days between new onsets, by decade")
    figure.tight_layout()
    figure.savefig(path, dpi=120)
    plt.close(figure)
