"""Rolling-origin backtests against three simple baselines.

The model is compared with a naive last-value forecast, a seasonal naive
forecast, and a moving average. The headline is how the model's MAE sits
against the best of those three, as "X% better than the best simple baseline"
or "X% worse than the best simple baseline".
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

ForecastFn = Callable[
    [Sequence[float], int],
    tuple[Sequence[float], Sequence[float] | None, Sequence[float] | None],
]


def rolling_origin_backtest(
    values: Sequence[float],
    predict: ForecastFn,
    *,
    horizon: int = 1,
    season_length: int = 7,
    ma_window: int = 7,
    min_train: int = 16,
    max_origins: int = 40,
) -> dict[str, Any]:
    """Score ``predict`` on origins that all three baselines can also score."""
    series = [float(value) for value in values]
    if any(not math.isfinite(value) for value in series):
        raise ValueError("Backtest values must be finite.")
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if season_length < 1 or ma_window < 1:
        raise ValueError("season_length and ma_window must be at least 1.")
    if max_origins < 1:
        raise ValueError("max_origins must be at least 1.")
    first = max(min_train, season_length, ma_window) - 1
    last = len(series) - horizon - 1
    origins = list(range(first, last + 1))
    if len(origins) > max_origins:
        stride = math.ceil(len(origins) / max_origins)
        origins = origins[::stride][:max_origins]
    if len(origins) < 4:
        raise ValueError("Not enough history for a rolling-origin backtest.")

    Row = tuple[float, float, float, float | None, float | None]
    model_rows: list[Row] = []
    naive_rows: list[Row] = []
    seasonal_rows: list[Row] = []
    average_rows: list[Row] = []
    for origin in origins:
        history = series[: origin + 1]
        actual = series[origin + 1 : origin + 1 + horizon]
        predicted, low, high = predict(history, horizon)
        if len(predicted) < horizon or len(actual) < horizon:
            raise ValueError("The forecast was shorter than the requested horizon.")
        level = history[-1]
        naive = [level] * horizon
        seasonal = [
            series[origin + lead - season_length] for lead in range(1, horizon + 1)
        ]
        window = history[-ma_window:]
        average = [sum(window) / len(window)] * horizon
        for lead in range(horizon):
            point = float(predicted[lead])
            target = float(actual[lead])
            lower = None if low is None or len(low) <= lead else float(low[lead])
            upper = None if high is None or len(high) <= lead else float(high[lead])
            model_rows.append((point, target, level, lower, upper))
            naive_rows.append((naive[lead], target, level, None, None))
            seasonal_rows.append((seasonal[lead], target, level, None, None))
            average_rows.append((average[lead], target, level, None, None))

    model_metrics = _score(model_rows)
    baselines = {
        "naive": _score(naive_rows),
        "seasonal_naive": _score(seasonal_rows),
        "moving_average": _score(average_rows),
    }
    best_name = min(baselines, key=lambda name: _mae(baselines[name]))
    comparison, improvement = _comparison(
        _mae(model_metrics), _mae(baselines[best_name])
    )
    return {
        "horizon": horizon,
        "n": len(origins),
        "season_length": season_length,
        "ma_window": ma_window,
        "metrics": model_metrics,
        "baselines": baselines,
        "best_baseline": best_name,
        "improvement_pct": improvement,
        "comparison": comparison,
    }


def _mae(metrics: dict[str, float | None]) -> float:
    value = metrics["mae"]
    if not isinstance(value, float):
        raise ValueError("mae is missing.")
    return value


def _comparison(model_mae: float, best_mae: float) -> tuple[str, float | None]:
    if best_mae == 0.0:
        if model_mae == 0.0:
            return "0.0% better than the best simple baseline", 0.0
        return "worse than the best simple baseline", None
    improvement = (best_mae - model_mae) / best_mae * 100.0
    if improvement >= 0.0:
        return f"{improvement:.1f}% better than the best simple baseline", improvement
    return f"{abs(improvement):.1f}% worse than the best simple baseline", improvement


def _score(
    rows: Sequence[tuple[float, float, float, float | None, float | None]],
) -> dict[str, float | None]:
    absolute = [
        abs(predicted - actual) for predicted, actual, _level, _low, _high in rows
    ]
    squared = [
        (predicted - actual) ** 2 for predicted, actual, _level, _low, _high in rows
    ]
    percent = [
        abs((predicted - actual) / actual)
        for predicted, actual, _level, _low, _high in rows
        if actual != 0.0
    ]
    directions = [
        _direction_match(predicted - level, actual - level)
        for predicted, actual, level, _low, _high in rows
    ]
    covered = [
        low <= actual <= high
        for _predicted, actual, _level, low, high in rows
        if low is not None and high is not None
    ]
    return {
        "mae": sum(absolute) / len(absolute),
        "rmse": math.sqrt(sum(squared) / len(squared)),
        "mape": None if not percent else sum(percent) / len(percent),
        "directional_accuracy": sum(directions) / len(directions),
        "coverage": None
        if not covered
        else sum(1 for hit in covered if hit) / len(covered),
    }


def _direction_match(predicted_change: float, actual_change: float) -> bool:
    if predicted_change == 0.0 or actual_change == 0.0:
        return predicted_change == actual_change
    return (predicted_change > 0.0) == (actual_change > 0.0)
