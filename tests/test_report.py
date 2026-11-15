"""Tests for shareable series report snapshots."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.analysis.report import build_starter_report
from src.api import app
from src.workspace.starter_forecast import clear_forecast_cache


def test_build_starter_report_includes_sections():
    clear_forecast_cache()
    report = build_starter_report("weather", points=32)
    assert report["v"] == 1
    assert report["series"] == "weather"
    assert report["forecast"]["starter_id"] == "weather"
    assert "why" in report["forecast"]
    # Accuracy / regimes may be null if the sample lacks a stored backtest,
    # but the keys must always be present.
    assert "accuracy" in report
    assert "regimes" in report


def test_analysis_report_endpoint():
    clear_forecast_cache()
    client = TestClient(app)
    response = client.post("/analysis/report", json={"series": "weather", "points": 32})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["title"]
    assert body["forecast"]["forecast"]
