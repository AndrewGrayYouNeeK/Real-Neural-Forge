"""Public starter diffusion futures demo for the dashboard tool view."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.experiments.diffusion_demo import clear_diffusion_demo_cache


def test_starter_diffusion_returns_band_and_paths() -> None:
    clear_diffusion_demo_cache()
    client = TestClient(app)
    response = client.get(
        "/starters/weather/diffusion", params={"horizon": 7, "n_paths": 8}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["starter_id"] == "weather"
    assert payload["horizon"] == 7
    assert len(payload["forecast"]) == 7
    first = payload["forecast"][0]
    assert first["p10"] <= first["p50"] <= first["p90"]
    assert payload["sample_paths"]
    assert len(payload["sample_paths"][0]) == 7
    assert payload["history"]
    assert "summary" in payload


def test_starter_diffusion_unknown_is_404() -> None:
    client = TestClient(app)
    response = client.get("/starters/not-a-starter/diffusion")
    assert response.status_code == 404


def test_diffusion_tool_view_is_wired() -> None:
    client = TestClient(app)
    html = client.get("/").text
    assert 'data-tool-view="diffusion"' in html
    assert 'id="diffusion-run-btn"' in html
    tools = client.get("/assets/tools.js").text
    assert 'id: "diffusion"' in tools
    assert 'view: "diffusion"' in tools
    app_js = client.get("/assets/app.js").text
    assert "/starters/weather/diffusion" in app_js
