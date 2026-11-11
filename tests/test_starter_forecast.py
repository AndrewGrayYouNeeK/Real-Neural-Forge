"""Public starter forecasts used by the chat-first weather and markets replies."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api import app
from src.workspace.starter_forecast import clear_forecast_cache, starter_forecast
from src.workspace.starters import default_demos


def setup_function() -> None:
    clear_forecast_cache()


def test_weather_forecast_uses_central_park_sample() -> None:
    payload = starter_forecast("weather", points=16)
    assert payload["starter_id"] == "weather"
    assert payload["name"] == "Central Park temperature"
    assert payload["unit"] == "°C"
    assert payload["dataset"] == "Central Park temperature"
    assert payload["source"] == "csv"
    assert payload["checkpoint_loaded"] is True
    assert len(payload["forecast"]) == 7
    value = payload["forecast"][0]["value"]
    assert isinstance(value, (int, float))
    # The NOAA sample highs are ordinary Celsius temperatures, not a 0–1 sine wave.
    assert -20 < float(value) < 45
    history = payload["history"]
    assert history
    assert history[-1]["time"] == "2026-10-31"
    assert history[-1]["actual"] == pytest.approx(14.4)
    assert payload["demo"] is True
    assert payload["forecast"][0]["p10"] is not None
    assert payload["forecast"][0]["p10"] <= payload["forecast"][0]["p50"]
    assert payload["forecast"][0]["p50"] <= payload["forecast"][0]["p90"]
    uncertainty = payload["uncertainty"]
    assert uncertainty["p50"][0] == payload["forecast"][0]["p50"]
    assert uncertainty["p10"][0] <= uncertainty["p50"][0] <= uncertainty["p90"][0]
    assert uncertainty["interval_80"]["lower"] == uncertainty["p10"]
    assert uncertainty["interval_80"]["upper"] == uncertainty["p90"]
    assert isinstance(uncertainty["confidence"], float)
    assert isinstance(uncertainty["calibrated"], bool)


def test_stocks_and_crypto_forecasts() -> None:
    stocks = starter_forecast("stocks", points=16)
    assert stocks["starter_id"] == "stocks"
    assert "AAPL" in stocks["name"]
    assert stocks["forecast"]
    crypto = starter_forecast("crypto", points=16)
    assert crypto["starter_id"] == "crypto"
    assert "Bitcoin" in crypto["name"]
    assert crypto["forecast"]


def test_unknown_starter_is_404() -> None:
    client = TestClient(app, raise_server_exceptions=True)
    response = client.get("/starters/not-a-starter/forecast")
    assert response.status_code == 404
    assert response.json()["detail"] == "Starter not found."


def test_public_weather_forecast_endpoint() -> None:
    client = TestClient(app, raise_server_exceptions=True)
    response = client.get("/starters/weather/forecast", params={"points": 16})
    assert response.status_code == 200
    body = response.json()
    assert body["starter_id"] == "weather"
    assert body["unit"] == "°C"
    assert body["forecast"]
    assert body["name"] == "Central Park temperature"


def test_public_starter_catalog_lists_the_five_samples() -> None:
    client = TestClient(app, raise_server_exceptions=True)
    response = client.get("/starters")
    assert response.status_code == 200
    body = response.json()
    ids = [starter["id"] for starter in body["starters"]]
    assert ids == ["weather", "stocks", "crypto", "near-earth", "lunar"]
    assert "lottery" not in ids
    names = {starter["id"]: starter["name"] for starter in body["starters"]}
    assert names["weather"] == "Central Park temperature"
    assert "AAPL" in names["stocks"]
    assert "Bitcoin" in names["crypto"]
    weather = next(
        starter for starter in body["starters"] if starter["id"] == "weather"
    )
    assert weather["unit"] == "°C"
    assert weather["demo"] is True
    assert "NOAA" in weather["source"]
    demos = [starter["id"] for starter in body["starters"] if starter.get("demo")]
    assert demos == ["weather", "near-earth"]


def test_default_demos_use_real_weather_and_near_earth_series() -> None:
    demos = list(default_demos())
    assert [starter.id for starter in demos] == ["weather", "near-earth"]
    weather = next(starter for starter in demos if starter.id == "weather")
    weather_lines = weather.dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(weather_lines) == 731
    assert weather_lines[1].startswith("2024-11-01,")
    assert weather_lines[-1].startswith("2026-10-31,")
    near_starter = next(starter for starter in demos if starter.id == "near-earth")
    near_lines = near_starter.dataset_path.read_text(encoding="utf-8").splitlines()
    assert len(near_lines) == 366
    assert near_lines[1].startswith("2026-10-22,")
    near = starter_forecast("near-earth", points=16)
    assert near["demo"] is True
    assert len(near["forecast"]) == 7
    history = near["history"]
    assert history[0]["time"] >= "2026-10-22"
    assert history[-1]["time"] < "2028-01-01"
    highlights = near["highlights"]
    assert highlights
    assert highlights[0]["kind"] == "asteroid"
    assert "lottery" not in near["summary"].lower()
