"""Tests for plain-English why-this-forecast narratives."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.analysis.why import attach_why, build_why_explanation
from src.api import app
from src.workspace.starter_forecast import clear_forecast_cache, starter_forecast


def test_trend_and_forecast_move():
    rising = list(range(30))
    why = build_why_explanation(history=rising, forecast=35.0)
    assert why["available"] is True
    text = why["explanation"].lower()
    assert "trending higher" in text or "above" in text
    kinds = {row["kind"] for row in why["drivers"]}
    assert "trend" in kinds
    assert "forecast" in kinds


def test_pca_and_regimes_enrich_gracefully():
    history = [10.0, 10.2, 10.1, 10.3, 10.0, 10.4, 10.2, 10.5]
    why = build_why_explanation(
        history=history,
        forecast=10.6,
        pca={"summary": "PC1 explains 80% of the variance across a and b."},
        regimes={"summary": "Two regimes: calm and volatile. 1 unusual point."},
    )
    assert "PC1" in why["explanation"] or "calm" in why["explanation"].lower()
    kinds = {row["kind"] for row in why["drivers"]}
    assert "pca" in kinds or "regime" in kinds


def test_missing_enrichment_still_works():
    why = build_why_explanation(history=[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], forecast=7.0)
    assert why["available"] is True
    assert "PCA" not in why["explanation"]


def test_analysis_why_endpoint():
    client = TestClient(app)
    response = client.post(
        "/analysis/why",
        json={
            "history": list(range(20)),
            "forecast": 22.0,
            "pca": {"summary": "Humidity and temperature move together."},
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["available"] is True
    assert "explanation" in body
    assert isinstance(body["drivers"], list)


def test_starter_forecast_includes_why():
    clear_forecast_cache()
    payload = starter_forecast("weather", points=32, lite=True)
    assert "why" in payload
    assert payload["why"]["available"] is True
    assert payload["why"]["explanation"]

    client = TestClient(app)
    response = client.get(
        "/starters/weather/forecast", params={"points": 32, "lite": 1}
    )
    assert response.status_code == 200
    assert response.json()["why"]["explanation"]


def test_attach_why_mutates_payload():
    payload = {
        "history": [{"actual": float(i)} for i in range(20)],
        "forecast": [{"p50": 21.0, "value": 21.0}],
        "unit": "°C",
    }
    attach_why(payload)
    assert payload["why"]["available"] is True
