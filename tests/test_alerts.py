"""Tests for in-app threshold / anomaly alert checks."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.analysis.alerts import evaluate_alert
from src.api import app
from src.workspace.starter_forecast import clear_forecast_cache


def test_threshold_alert_against_weather():
    clear_forecast_cache()
    low = evaluate_alert("weather", "threshold", {"op": ">", "value": 1e9})
    assert low["fired"] is False
    high = evaluate_alert("weather", "threshold", {"op": "<", "value": 1e9})
    assert high["fired"] is True
    assert "message" in high


def test_anomaly_alert_returns_structure():
    clear_forecast_cache()
    result = evaluate_alert("weather", "anomaly", {})
    assert result["kind"] == "anomaly"
    assert "fired" in result
    assert "message" in result


def test_alert_check_endpoint():
    clear_forecast_cache()
    client = TestClient(app)
    response = client.post(
        "/analysis/alert-check",
        json={
            "series": "weather",
            "kind": "threshold",
            "config": {"op": "<", "value": 1e9},
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["fired"] is True
