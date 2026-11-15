"""Build a one-click shareable series summary (forecast, accuracy, regimes)."""

from __future__ import annotations

from typing import Any

from src.analysis.kmeans import cluster_csv_bytes
from src.evaluation.run import backtest_starter, stored_backtest
from src.workspace.errors import WorkspaceError
from src.workspace.starter_forecast import starter_forecast
from src.workspace.starters import starter_by_id


def _trim_forecast(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("history")
    history_tail: list[Any] = []
    if isinstance(history, list):
        history_tail = history[-12:]
    forecast = payload.get("forecast")
    forecast_steps = forecast[:8] if isinstance(forecast, list) else forecast
    return {
        "starter_id": payload.get("starter_id"),
        "name": payload.get("name"),
        "unit": payload.get("unit"),
        "summary": payload.get("summary"),
        "horizon": payload.get("horizon"),
        "history": history_tail,
        "forecast": forecast_steps,
        "uncertainty": payload.get("uncertainty"),
        "why": payload.get("why"),
        "checkpoint_loaded": payload.get("checkpoint_loaded"),
    }


def _trim_accuracy(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    return {
        "horizon": report.get("horizon"),
        "n": report.get("n"),
        "metrics": report.get("metrics"),
        "baselines": report.get("baselines"),
        "best_baseline": report.get("best_baseline"),
        "comparison": report.get("comparison"),
        "improvement_pct": report.get("improvement_pct"),
    }


def _trim_regimes(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(report, dict):
        return None
    anomalies = report.get("anomalies")
    anomaly_rows = anomalies[:8] if isinstance(anomalies, list) else []
    return {
        "k": report.get("k"),
        "summary": report.get("summary"),
        "regimes": report.get("regimes"),
        "anomaly_count": report.get("anomaly_count"),
        "anomalies": anomaly_rows,
        "silhouette": report.get("silhouette"),
    }


def build_starter_report(series: str, *, points: int = 120) -> dict[str, Any]:
    """Assemble a frozen shareable snapshot for a shipped starter series."""
    starter = starter_by_id(series)
    if starter is None:
        raise WorkspaceError(404, "Series not found.")
    forecast_payload = starter_forecast(starter.id, points=points, lite=True)
    accuracy = stored_backtest(starter.id)
    if accuracy is None:
        try:
            accuracy = backtest_starter(starter, horizon=1, refresh=False)
        except (OSError, ValueError, RuntimeError, TypeError, KeyError, WorkspaceError):
            accuracy = None
    regimes: dict[str, Any] | None = None
    try:
        if starter.dataset_path.is_file():
            regimes = cluster_csv_bytes(starter.dataset_path.read_bytes()).as_dict()
    except (OSError, ValueError):
        regimes = None
    title = f"{starter.name} report"
    return {
        "v": 1,
        "series": starter.id,
        "title": title,
        "forecast": _trim_forecast(forecast_payload),
        "accuracy": _trim_accuracy(accuracy),
        "regimes": _trim_regimes(regimes),
    }
