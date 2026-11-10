"""Rolling-origin backtests and the evaluation endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.evaluation.backtest import rolling_origin_backtest


def _perfect(history, horizon):
    # values are 1, 2, 3, ... so the next point is len(history) + 1.
    start = len(history)
    predicted = [float(start + lead) for lead in range(1, horizon + 1)]
    return (
        predicted,
        [value - 0.5 for value in predicted],
        [value + 0.5 for value in predicted],
    )


def test_perfect_forecast_beats_the_simple_baselines():
    values = [float(index) for index in range(1, 40)]
    report = rolling_origin_backtest(
        values,
        _perfect,
        horizon=1,
        season_length=7,
        ma_window=3,
        min_train=12,
        max_origins=20,
    )
    assert set(report["baselines"]) == {"naive", "seasonal_naive", "moving_average"}
    assert report["metrics"]["mae"] < report["baselines"]["naive"]["mae"]
    assert report["metrics"]["coverage"] == 1.0
    assert report["metrics"]["directional_accuracy"] == 1.0
    assert report["comparison"].endswith("better than the best simple baseline")
    assert report["improvement_pct"] > 0


def test_matching_the_best_baseline_is_zero_percent_better():
    values = [float(index) for index in range(1, 30)]

    def naive(history, horizon):
        return [history[-1]] * horizon, None, None

    report = rolling_origin_backtest(
        values, naive, horizon=1, season_length=4, ma_window=3, min_train=8
    )
    assert report["best_baseline"] == "naive"
    assert report["comparison"] == "0.0% better than the best simple baseline"


def test_short_series_is_rejected():
    try:
        rolling_origin_backtest([1.0, 2.0, 3.0], _perfect, min_train=8, season_length=2)
    except ValueError as exc:
        assert "Not enough history" in str(exc)
    else:
        raise AssertionError("expected a short series to be rejected")


def test_evaluation_endpoints_cache_the_weather_backtest():
    with TestClient(app, raise_server_exceptions=True) as client:
        missing = client.get("/evaluation/not-a-model")
        assert missing.status_code == 404
        first = client.post("/evaluation/weather/backtest", json={})
        assert first.status_code == 200, first.text
        body = first.json()
        assert body["model_id"] == "weather"
        assert set(body["baselines"]) == {"naive", "seasonal_naive", "moving_average"}
        assert "than the best simple baseline" in body["comparison"]
        assert body["metrics"]["mae"] >= 0
        second = client.post("/evaluation/weather/backtest", json={})
        assert second.json()["comparison"] == body["comparison"]
        summary = client.get("/evaluation/weather")
        assert summary.status_code == 200
        assert summary.json()["backtest"]["comparison"] == body["comparison"]
        assert "lottery" not in first.text.lower()
