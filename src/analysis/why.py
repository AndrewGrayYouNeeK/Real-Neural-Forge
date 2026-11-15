"""Plain-English “why this forecast” narratives from recent history + optional drivers."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_series(values: list[float] | np.ndarray | None) -> np.ndarray:
    if values is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return np.asarray(arr[np.isfinite(arr)], dtype=np.float64)


def describe_trend(history: np.ndarray) -> str | None:
    """Short trend sentence from recent levels (rising / falling / flat)."""
    if history.size < 4:
        return None
    window = min(14, max(3, history.size // 3))
    recent = history[-window:]
    prior = (
        history[-(2 * window) : -window]
        if history.size >= 2 * window
        else history[:-window]
    )
    if prior.size == 0:
        return None
    recent_mean = float(np.mean(recent))
    prior_mean = float(np.mean(prior))
    scale = max(abs(prior_mean), abs(recent_mean), 1e-9)
    delta = (recent_mean - prior_mean) / scale
    if abs(delta) < 0.02:
        return "Recent values have been roughly flat."
    if delta > 0:
        return "Recent values have been trending higher."
    return "Recent values have been trending lower."


def describe_seasonality(history: np.ndarray) -> str | None:
    """Cheap lag autocorrelation hint (weekly-ish or short-cycle)."""
    if history.size < 16:
        return None
    centered = history - float(np.mean(history))
    denom = float(np.dot(centered, centered))
    if denom <= 1e-12:
        return None
    best_lag = 0
    best_corr = 0.0
    for lag in (7, 5, 12, 4):
        if history.size <= lag + 4:
            continue
        corr = float(np.dot(centered[lag:], centered[:-lag]) / denom)
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag
    if best_lag == 0 or abs(best_corr) < 0.25:
        return None
    if best_corr > 0:
        return f"A repeating pattern around every {best_lag} steps shows up in the recent history."
    return (
        f"The series tends to reverse on a roughly {best_lag}-step cycle "
        "in the recent history."
    )


def describe_forecast_move(history: np.ndarray, forecast: float | None) -> str | None:
    if forecast is None or history.size == 0:
        return None
    last = float(history[-1])
    scale = max(abs(last), abs(forecast), 1e-9)
    delta = (forecast - last) / scale
    if abs(delta) < 0.01:
        return "The next-step forecast stays close to the latest observed value."
    if delta > 0:
        return "The next-step forecast sits above the latest observed value."
    return "The next-step forecast sits below the latest observed value."


def _pca_driver_line(pca: dict[str, Any] | None) -> str | None:
    if not isinstance(pca, dict):
        return None
    summary = pca.get("summary")
    if isinstance(summary, str) and summary.strip():
        return summary.strip()
    groups = pca.get("groups")
    if not isinstance(groups, list) or not groups:
        return None
    first = groups[0]
    if not isinstance(first, dict):
        return None
    features = first.get("features")
    if isinstance(features, list) and features:
        names = ", ".join(str(item) for item in features[:4])
        return f"PCA drivers highlight co-moving columns: {names}."
    pos = first.get("features_positive") or []
    neg = first.get("features_negative") or []
    bits: list[str] = []
    if isinstance(pos, list) and pos:
        bits.append("with " + ", ".join(str(item) for item in pos[:3]))
    if isinstance(neg, list) and neg:
        bits.append("against " + ", ".join(str(item) for item in neg[:3]))
    if not bits:
        return None
    return "PCA drivers move " + " ".join(bits) + "."


def _regime_line(regimes: dict[str, Any] | None) -> str | None:
    if not isinstance(regimes, dict):
        return None
    summary = regimes.get("summary")
    if isinstance(summary, str) and summary.strip():
        # Prefer a short first sentence from the regimes summary.
        first = summary.strip().split(".")[0].strip()
        if first:
            return first + "."
    rows = regimes.get("regimes")
    if isinstance(rows, list) and rows:
        top = rows[0]
        if isinstance(top, dict):
            label = top.get("label") or top.get("short_label")
            if isinstance(label, str) and label.strip():
                return f"The recent window looks like a “{label.strip()}” regime."
    anomalies = regimes.get("anomaly_count")
    if isinstance(anomalies, int) and anomalies > 0:
        return f"K-Means flagged {anomalies} unusual point(s) far from cluster centers."
    return None


def build_why_explanation(
    *,
    history: list[float] | np.ndarray | None,
    forecast: float | None = None,
    unit: str | None = None,
    pca: dict[str, Any] | None = None,
    regimes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a short plain-English explanation and structured driver list.

    Degrades gracefully when PCA / regimes payloads are missing: trend and
    seasonality from history alone are enough for a useful sentence.
    """
    series = _finite_series(history)
    drivers: list[dict[str, str]] = []

    trend = describe_trend(series)
    if trend:
        drivers.append({"kind": "trend", "text": trend})
    seasonality = describe_seasonality(series)
    if seasonality:
        drivers.append({"kind": "seasonality", "text": seasonality})
    move = describe_forecast_move(series, forecast)
    if move:
        drivers.append({"kind": "forecast", "text": move})

    pca_line = _pca_driver_line(pca)
    if pca_line:
        drivers.append({"kind": "pca", "text": pca_line})
    regime_line = _regime_line(regimes)
    if regime_line:
        drivers.append({"kind": "regime", "text": regime_line})

    if not drivers:
        explanation = "There is not enough recent history yet to explain this forecast in plain English."
    else:
        # Keep the narrative short: lead with trend/move, then one enrichment.
        lead = [
            d["text"]
            for d in drivers
            if d["kind"] in {"trend", "forecast", "seasonality"}
        ]
        enrich = [d["text"] for d in drivers if d["kind"] in {"pca", "regime"}]
        parts = lead[:2]
        if enrich:
            parts.append(enrich[0])
        elif len(lead) > 2:
            parts.append(lead[2])
        explanation = " ".join(parts)
        if (
            unit
            and forecast is not None
            and "forecast" not in {d["kind"] for d in drivers[:1]}
        ):
            pass

    return {
        "explanation": explanation,
        "drivers": drivers,
        "available": bool(drivers),
    }


def history_values_from_payload(payload: dict[str, Any]) -> list[float]:
    """Extract numeric actuals from a starter / dashboard forecast payload."""
    history = payload.get("history")
    values: list[float] = []
    if isinstance(history, list):
        for row in history:
            if isinstance(row, dict):
                actual = row.get("actual", row.get("value"))
                if isinstance(actual, bool) or not isinstance(actual, int | float):
                    continue
                values.append(float(actual))
            elif isinstance(row, int | float) and not isinstance(row, bool):
                values.append(float(row))
    return values


def forecast_value_from_payload(payload: dict[str, Any]) -> float | None:
    forecast = payload.get("forecast")
    if isinstance(forecast, list) and forecast:
        first = forecast[0]
        if isinstance(first, dict):
            center = first.get("p50", first.get("value"))
            if isinstance(center, int | float) and not isinstance(center, bool):
                return float(center)
        elif isinstance(first, int | float) and not isinstance(first, bool):
            return float(first)
    prediction = payload.get("prediction")
    if isinstance(prediction, list) and prediction:
        point = prediction[0]
        if isinstance(point, int | float) and not isinstance(point, bool):
            return float(point)
    if isinstance(prediction, int | float) and not isinstance(prediction, bool):
        return float(prediction)
    return None


def attach_why(
    payload: dict[str, Any],
    *,
    pca: dict[str, Any] | None = None,
    regimes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mutate ``payload`` with a ``why`` block derived from its history/forecast."""
    unit = payload.get("unit") if isinstance(payload.get("unit"), str) else None
    why = build_why_explanation(
        history=history_values_from_payload(payload),
        forecast=forecast_value_from_payload(payload),
        unit=unit,
        pca=pca,
        regimes=regimes,
    )
    payload["why"] = why
    return payload
