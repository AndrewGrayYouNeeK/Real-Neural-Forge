"""Render the forecast charts cited from the pull request.

matplotlib is used only here. It is not a runtime dependency of the API.
"""

from __future__ import annotations

import argparse
import json
from itertools import accumulate
from pathlib import Path
from typing import cast

import matplotlib

from src.utils.paths import resolve_repo_path

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _load(path: str) -> dict[str, object]:
    return cast(dict[str, object], json.loads(resolve_repo_path(path).read_text()))


def _series_chart(
    payload: dict,
    column: str,
    title: str,
    ylabel: str,
    destination: Path,
    xlabel: str | None = None,
) -> None:
    dates = list(payload["test_dates"])
    actual = _column(payload, "test_actual", column)
    model = _column(payload, "test_model", column)
    persistence = _column(payload, "test_persistence", column)
    log_linear = _column(payload, "test_log_linear", column)
    x = range(len(dates))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(x, actual, label="actual", color="#1f4e79", linewidth=1.8)
    ax.plot(x, model, label="transformer", color="#c45911", linewidth=1.4)
    ax.plot(x, log_linear, label="log-linear", color="#548235", linewidth=1.2)
    ax.plot(
        x,
        persistence,
        label="persistence",
        color="#7f7f7f",
        linewidth=1.0,
        linestyle="--",
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    ax.legend(frameon=False)
    _sparse_ticks(ax, dates)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def _column(payload: dict, key: str, column: str) -> list[float]:
    index = list(payload["columns"]).index(column)
    return [float(row[index]) for row in payload[key]]


def _sparse_ticks(ax: plt.Axes, labels: list[str]) -> None:
    step = 1 if len(labels) <= 8 else max(1, len(labels) // 6)
    positions = list(range(0, len(labels), step))
    ax.set_xticks(positions)
    ax.set_xticklabels([labels[i] for i in positions], rotation=45, ha="right")


def _bankroll(payload: dict, destination: Path) -> None:
    races = payload["test_races"]
    model = list(accumulate(row["model_profit"] for row in races))
    favourite = list(accumulate(row["favourite_profit"] for row in races))
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    ax.plot(model, label="model, flat 1-unit stake", color="#c45911")
    ax.plot(favourite, label="favourite, flat 1-unit stake", color="#1f4e79")
    ax.axhline(0.0, color="#7f7f7f", linewidth=0.8)
    ax.set_title("Hong Kong test period, cumulative profit at published dividends")
    ax.set_xlabel("test race")
    ax.set_ylabel("profit (stake units)")
    ax.legend(frameon=False)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="/opt/cursor/artifacts")
    args = parser.parse_args()
    out = Path(args.out_dir)
    charts = [
        (
            "results/ai_frontier_compute.json",
            "log10_compute",
            "Frontier training compute",
            "log10 FLOP",
            "ai_frontier_compute.png",
        ),
        (
            "results/ai_arxiv_counts.json",
            "cs.AI",
            "Monthly cs.AI submissions",
            "submissions",
            "ai_arxiv_cs_ai.png",
        ),
        (
            "results/ai_arxiv_counts.json",
            "cs.LG",
            "Monthly cs.LG submissions",
            "submissions",
            "ai_arxiv_cs_lg.png",
        ),
        (
            "results/ai_arxiv_counts.json",
            "cs.CL",
            "Monthly cs.CL submissions",
            "submissions",
            "ai_arxiv_cs_cl.png",
        ),
        (
            "results/ai_github_repos.json",
            "count",
            "New machine-learning repositories",
            "repositories",
            "ai_github_repos.png",
        ),
        (
            "results/quantum_qubits.json",
            "log10_qubits",
            "Record physical qubits",
            "log10 qubits",
            "quantum_qubits.png",
        ),
        (
            "results/quantum_qubits_rearranged.json",
            "log10_qubits_excluding_unrearranged",
            "Record qubits excluding unrearranged arrays",
            "log10 qubits",
            "quantum_qubits_rearranged.png",
        ),
        (
            "results/quantum_gate_error.json",
            "log10_error",
            "Best reported two-qubit error",
            "log10 error",
            "quantum_gate_error.png",
        ),
        (
            "results/quantum_arxiv.json",
            "count",
            "Monthly quant-ph submissions",
            "submissions",
            "quantum_arxiv.png",
        ),
        (
            "results/quantum_random.json",
            "code",
            "Quantum-random symbol codes",
            "symbol code",
            "quantum_random.png",
            "symbol index (not a calendar clock)",
        ),
        (
            "results/quantum_oulu_neutrons.json",
            "count_rate",
            "Oulu neutron monitor",
            "count rate",
            "oulu_neutrons.png",
        ),
    ]
    for spec in charts:
        path, column, title, ylabel, name = spec[:5]
        xlabel = spec[5] if len(spec) > 5 else None
        source = resolve_repo_path(path)
        if not source.exists():
            print(f"skip {name}, missing {path}")
            continue
        _series_chart(_load(path), column, title, ylabel, out / name, xlabel=xlabel)
        print(f"wrote {out / name}")
    racing = resolve_repo_path("results/horse_racing.json")
    if racing.exists():
        _bankroll(_load(str(racing)), out / "horse_racing_bankroll.png")
        print(f"wrote {out / 'horse_racing_bankroll.png'}")


if __name__ == "__main__":
    main()
