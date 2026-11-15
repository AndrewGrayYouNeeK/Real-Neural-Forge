"""Evaluate in-app alert rules against starter series (threshold / anomalies)."""

from __future__ import annotations

from typing import Any

from src.analysis.kmeans import cluster_csv_bytes
from src.workspace.errors import WorkspaceError
from src.workspace.starter_forecast import starter_forecast
from src.workspace.starters import starter_by_id


def _latest_value(series: str) -> float:
    payload = starter_forecast(series, points=64, lite=True)
    history = payload.get("history")
    if isinstance(history, list):
        for row in reversed(history):
            if not isinstance(row, dict):
                continue
            actual = row.get("actual", row.get("value"))
            if isinstance(actual, int | float) and not isinstance(actual, bool):
                return float(actual)
    forecast = payload.get("forecast")
    if isinstance(forecast, list) and forecast:
        first = forecast[0]
        if isinstance(first, dict):
            center = first.get("p50", first.get("value"))
            if isinstance(center, int | float) and not isinstance(center, bool):
                return float(center)
    raise WorkspaceError(422, "Could not read a latest value for this series.")


def evaluate_threshold(series: str, config: dict[str, Any]) -> dict[str, Any]:
    latest = _latest_value(series)
    value = config.get("value")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise WorkspaceError(422, "threshold config.value must be a number.")
    op = str(config.get("op") or ">")
    threshold = float(value)
    fired = False
    if op == ">":
        fired = latest > threshold
    elif op == "<":
        fired = latest < threshold
    elif op == ">=":
        fired = latest >= threshold
    elif op == "<=":
        fired = latest <= threshold
    elif op == "abs>":
        fired = abs(latest) > threshold
    else:
        raise WorkspaceError(422, "threshold op must be one of >, <, >=, <=, abs>.")
    message = (
        f"Latest value {latest:g} crossed threshold {op} {threshold:g}."
        if fired
        else f"Latest value {latest:g} is within threshold {op} {threshold:g}."
    )
    return {
        "fired": fired,
        "kind": "threshold",
        "series": series,
        "latest": latest,
        "op": op,
        "value": threshold,
        "message": message,
    }


def evaluate_anomaly(series: str, config: dict[str, Any]) -> dict[str, Any]:
    starter = starter_by_id(series)
    if starter is None:
        raise WorkspaceError(404, "Series not found.")
    if not starter.dataset_path.is_file():
        raise WorkspaceError(503, "Sample dataset is unavailable.")
    report = cluster_csv_bytes(starter.dataset_path.read_bytes()).as_dict()
    anomalies = report.get("anomalies")
    rows = anomalies if isinstance(anomalies, list) else []
    min_distance = config.get("min_distance")
    filtered: list[Any] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        distance = row.get("distance")
        if (
            isinstance(min_distance, int | float)
            and not isinstance(min_distance, bool)
            and isinstance(distance, int | float)
            and float(distance) < float(min_distance)
        ):
            continue
        filtered.append(row)
    fired = len(filtered) > 0
    message = (
        f"Found {len(filtered)} unusual point(s) via K-Means."
        if fired
        else "No unusual points matched this anomaly alert."
    )
    return {
        "fired": fired,
        "kind": "anomaly",
        "series": series,
        "anomaly_count": len(filtered),
        "anomalies": filtered[:8],
        "summary": report.get("summary"),
        "message": message,
    }


def evaluate_alert(
    series: str, kind: str, config: dict[str, Any] | None = None
) -> dict[str, Any]:
    starter = starter_by_id(series)
    if starter is None:
        raise WorkspaceError(404, "Series not found.")
    cfg = config if isinstance(config, dict) else {}
    if kind == "threshold":
        return evaluate_threshold(series, cfg)
    if kind == "anomaly":
        return evaluate_anomaly(series, cfg)
    raise WorkspaceError(422, "kind must be threshold or anomaly.")
