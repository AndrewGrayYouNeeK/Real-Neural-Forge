"""Tests for what-if counterfactual forecasts."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.analysis.whatif import apply_shock
from src.api import app
from src.workspace.starter_forecast import clear_forecast_cache, starter_what_if


def test_apply_shock_pct_and_abs():
    window = [[1.0], [2.0], [10.0]]
    before, after, shocked = apply_shock(window, mode="pct", value=-10)
    assert before == 10.0
    assert after == 9.0
    assert shocked[-1][0] == 9.0
    assert window[-1][0] == 10.0
    _, after_abs, _ = apply_shock(window, mode="abs", value=1.5)
    assert after_abs == 11.5


def test_starter_what_if_changes_path():
    clear_forecast_cache()
    result = starter_what_if("weather", mode="pct", value=-25.0, horizon=1)
    assert result["starter_id"] == "weather"
    assert result["shock"]["before"] != result["shock"]["after"]
    assert "summary" in result
    assert len(result["baseline"]["p50"]) == 1
    assert len(result["shocked"]["p50"]) == 1


def test_analysis_what_if_endpoint():
    clear_forecast_cache()
    client = TestClient(app)
    response = client.post(
        "/analysis/what-if",
        json={"starter_id": "weather", "mode": "pct", "value": -10, "horizon": 1},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["series"] == "weather"
    assert "delta" in body
    assert isinstance(body["summary"], str) and body["summary"]
