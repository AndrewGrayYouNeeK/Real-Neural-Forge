"""Public starter diffusion futures demo for the dashboard tool view."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.experiments.diffusion_demo import clear_diffusion_demo_cache


def test_starter_diffusion_returns_band_and_paths() -> None:
    clear_diffusion_demo_cache()
    client = TestClient(app)
    response = client.get(
        "/starters/weather/diffusion", params={"horizon": 7, "n_paths": 12}
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


def test_starter_diffusion_bands_stay_near_historical_range() -> None:
    """p10/p50/p90 must stay within a sane pad of the training data range."""
    clear_diffusion_demo_cache()
    client = TestClient(app)
    response = client.get(
        "/starters/weather/diffusion", params={"horizon": 7, "n_paths": 16}
    )
    assert response.status_code == 200
    payload = response.json()
    data_min = float(payload["data_min"])
    data_max = float(payload["data_max"])
    span = max(data_max - data_min, 1e-3)
    # Allow the same pad the demo uses (0.35× range), plus a tiny float slack.
    lo = data_min - 0.35 * span - 1e-3
    hi = data_max + 0.35 * span + 1e-3
    last = float(payload["history"][-1]["actual"])
    for step in payload["forecast"]:
        for key in ("p10", "p50", "p90"):
            value = float(step[key])
            assert lo <= value <= hi, (key, value, lo, hi)
        # Median should track near the recent level (champion-anchored demo).
        assert abs(float(step["p50"]) - last) <= 0.5 * span + 5.0
        # Band width should stay a modest fraction of the historical span.
        width = float(step["p90"]) - float(step["p10"])
        assert width <= 0.75 * span + 1.0
    for path in payload["sample_paths"]:
        for value in path:
            assert lo <= float(value) <= hi


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
