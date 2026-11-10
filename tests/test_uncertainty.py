"""Additive uncertainty on forecasts, without removing existing fields."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.forecasting.intervals import interval_confidence, interval_is_calibrated
from src.forecasting.output import uncertainty_report


def test_confidence_rises_as_coverage_matches_and_the_window_grows():
    assert interval_confidence(None, 0) == 0.0
    assert interval_confidence(0.8, 30) == 1.0
    assert interval_confidence(0.8, 15) == 0.5
    assert interval_is_calibrated(0.8, 29) is False
    assert interval_is_calibrated(0.8, 30) is True
    assert interval_is_calibrated(0.69, 30) is False


def test_missing_bounds_collapse_to_the_point_and_stay_uncalibrated():
    report = uncertainty_report([1.25])
    assert report["p10"] == report["p50"] == report["p90"] == [1.25]
    assert report["interval_80"] == {"lower": [1.25], "upper": [1.25]}
    assert report["confidence"] == 0.0
    assert report["calibrated"] is False
    assert report["coverage"] is None


def test_real_interval_quotes_stored_coverage():
    report = uncertainty_report(
        [2.0, 3.0],
        [1.0, 2.0],
        [4.0, 5.0],
        coverage_80=0.8,
        p10_hit_rate=0.9,
        p90_hit_rate=0.9,
        coverage_n=40,
        coverage_window=90,
        coverage_horizon=1,
    )
    assert report["calibrated"] is True
    assert report["confidence"] == 1.0
    assert report["coverage"]["n"] == 40
    assert report["coverage"]["coverage_80"] == 0.8
    assert report["p50"] == [2.0, 3.0]


def test_forecast_and_starter_predict_keep_their_old_fields():
    with TestClient(app, raise_server_exceptions=True) as client:
        forecast = client.post(
            "/forecast", json={"sequence": [[0.1], [0.2], [0.3], [0.4]], "horizon": 3}
        )
        starter = client.get("/starters/weather/forecast", params={"points": 16})
    assert forecast.status_code == 200
    body = forecast.json()
    assert body["horizon"] == 3
    assert body["forecast"]
    assert body["p50"]
    assert body["uncertainty"]["p50"] == body["p50"]
    assert "prediction" in body
    assert starter.status_code == 200
    sample = starter.json()
    assert sample["starter_id"] == "weather"
    assert sample["forecast"]
    assert sample["history"]
    assert "uncertainty" in sample
    assert sample["uncertainty"]["p50"]
