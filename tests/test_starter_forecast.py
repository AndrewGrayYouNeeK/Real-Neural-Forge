"""Public starter forecasts used by the chat-first weather and markets replies."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.workspace.starter_forecast import clear_forecast_cache, starter_forecast


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
    assert payload["forecast"]
    value = payload["forecast"][0]["value"]
    assert isinstance(value, (int, float))
    # The NOAA sample highs are ordinary Celsius temperatures, not a 0–1 sine wave.
    assert 0 < float(value) < 50
    history = payload["history"]
    assert history
    assert history[-1]["time"] == "2024-06-30"
    assert history[-1]["actual"] == 29.4


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
    assert "NOAA" in weather["source"]
