"""Public starter JEPA vs transformer compare for the dashboard tool view."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.experiments.jepa_demo import clear_jepa_demo_cache


def test_starter_jepa_compare_returns_mae_table() -> None:
    clear_jepa_demo_cache()
    client = TestClient(app)
    response = client.get("/starters/weather/jepa-compare")
    assert response.status_code == 200
    payload = response.json()
    assert payload["starter_id"] == "weather"
    names = [row["name"] for row in payload["models"]]
    assert "jepa_forecaster" in names
    assert "time_series_transformer" in names
    for row in payload["models"]:
        assert row["mae"] >= 0
    assert payload["winner"] in {
        "jepa_forecaster",
        "time_series_transformer",
        "tie",
    }
    assert payload["n_train"] > 0
    assert payload["n_val"] > 0


def test_starter_jepa_compare_unknown_is_404() -> None:
    client = TestClient(app)
    response = client.get("/starters/not-a-starter/jepa-compare")
    assert response.status_code == 404


def test_jepa_tool_view_is_wired() -> None:
    client = TestClient(app)
    html = client.get("/").text
    assert 'data-tool-view="jepa"' in html
    assert 'id="jepa-run-btn"' in html
    tools = client.get("/assets/tools.js").text
    assert 'id: "jepa"' in tools
    assert 'view: "jepa"' in tools
    app_js = client.get("/assets/app.js").text
    assert "/starters/weather/jepa-compare" in app_js
