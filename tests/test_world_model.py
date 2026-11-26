"""Tests for the Phase 5 latent world model / simulator."""

from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from src.analysis.world_model import (
    clear_simulation_cache,
    fit_world_model,
    simulate_series,
    starter_simulation,
)
from src.api import app


def _wave(n: int = 120) -> list[float]:
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(3)
    return list(50 + 10 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.5, n))


def test_fit_is_stable_and_has_regimes():
    fitted = fit_world_model(np.asarray(_wave()))
    assert 2 <= fitted.latent_dim <= 6
    assert 0.5 <= fitted.damping <= 1.0
    assert fitted.regime_labels
    assert abs(sum(fitted.regime_shares) - 1.0) < 1e-6


def test_baseline_bands_are_ordered_and_backtest_is_reported():
    result = simulate_series(_wave(), horizon=10, n_paths=64)
    base = result["baseline"]
    assert len(base["p50"]) == 10
    for lo, mid, hi in zip(base["p10"], base["p50"], base["p90"], strict=True):
        assert lo <= mid <= hi
    bt = result["backtest"]
    assert bt["holdout"] > 0
    # A clean periodic signal is easy: the simulator should beat persistence.
    assert bt["beats_persistence"] is True
    assert "Backtest" in result["summary"]


def test_scenarios_share_draws_and_move_in_the_right_direction():
    scenarios = [
        {"name": "Up", "events": [{"kind": "shock", "step": 1, "value": 8.0}]},
        {"name": "Down", "events": [{"kind": "shock", "step": 1, "value": -8.0}]},
        {"name": "Same", "events": []},
    ]
    result = simulate_series(_wave(), horizon=6, n_paths=64, scenarios=scenarios)
    up, down, same = result["scenarios"]
    assert up["delta_p50"][0] > 7.0
    assert down["delta_p50"][0] < -7.0
    # No events + common random numbers = identical to the baseline.
    assert same["p50"] == result["baseline"]["p50"]
    assert all(abs(d) < 1e-9 for d in same["delta_p50"])


def test_set_drift_and_volatility_events():
    scenarios = [
        {"events": [{"kind": "set", "step": 2, "value": 70.0}]},
        {"events": [{"kind": "drift", "step": 1, "value": 1.0}]},
        {"events": [{"kind": "volatility", "step": 1, "value": 3.0}]},
    ]
    result = simulate_series(_wave(), horizon=8, n_paths=64, scenarios=scenarios)
    set_s, drift_s, vol_s = result["scenarios"]
    assert set_s["p50"][1] == 70.0
    assert drift_s["end_delta"] > 0
    base_width = result["baseline"]["p90"][-1] - result["baseline"]["p10"][-1]
    assert vol_s["p90"][-1] - vol_s["p10"][-1] > base_width


def test_non_negative_series_stays_non_negative():
    rng = np.random.default_rng(0)
    counts = list(rng.poisson(1.0, 200).astype(float))
    result = simulate_series(
        counts,
        horizon=10,
        n_paths=64,
        scenarios=[{"events": [{"kind": "shock", "step": 1, "value": -5}]}],
    )
    assert min(result["baseline"]["p10"]) >= 0.0
    assert min(result["scenarios"][0]["p10"]) >= 0.0


def test_weather_starter_in_fahrenheit_uses_seasonal_baseline():
    clear_simulation_cache()
    result = starter_simulation(
        "weather",
        horizon=7,
        n_paths=32,
        display_unit="°F",
        scenarios=[
            {
                "name": "Cold snap",
                "events": [{"kind": "shock", "step": 1, "value": -15}],
            }
        ],
    )
    assert result["unit"] == "°F"
    assert result["storage_unit"] == "°C"
    assert result["model"]["seasonal"] is True
    # Central Park highs in °F, not °C.
    assert all(20 < v < 110 for v in result["baseline"]["p50"])
    assert result["scenarios"][0]["delta_p50"][0] < -10
    assert "°F" in result["summary"]
    again = starter_simulation(
        "weather",
        horizon=7,
        n_paths=32,
        display_unit="°F",
        scenarios=[
            {
                "name": "Cold snap",
                "events": [{"kind": "shock", "step": 1, "value": -15}],
            }
        ],
    )
    assert again is result


def test_too_short_series_is_rejected():
    try:
        simulate_series([1.0] * 10)
    except ValueError as exc:
        assert "24" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_simulate_endpoint_starter_and_custom():
    client = TestClient(app)
    response = client.post(
        "/analysis/simulate",
        json={
            "starter_id": "lunar",
            "horizon": 5,
            "n_paths": 32,
            "scenarios": [
                {"name": "Bump", "events": [{"kind": "shock", "step": 1, "value": 0.1}]}
            ],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["series"] == "lunar"
    assert len(body["baseline"]["p50"]) == 5
    assert body["scenarios"][0]["name"] == "Bump"

    custom = client.post(
        "/analysis/simulate",
        json={"values": _wave(60), "horizon": 4, "n_paths": 16, "unit": "kWh"},
    )
    assert custom.status_code == 200, custom.text
    assert custom.json()["series"] == "custom"
    assert "kWh" in custom.json()["summary"]

    bad = client.post("/analysis/simulate", json={"horizon": 4})
    assert bad.status_code == 422
    bad_step = client.post(
        "/analysis/simulate",
        json={
            "values": _wave(60),
            "horizon": 4,
            "scenarios": [{"events": [{"kind": "shock", "step": 9, "value": 1}]}],
        },
    )
    assert bad_step.status_code == 422
