"""Tests for the self-supervised JEPA encoder option in the world model."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.analysis import jepa_world
from src.analysis.jepa_world import JepaBusyError, train_jepa_encoder
from src.analysis.world_model import (
    clear_simulation_cache,
    fit_world_model,
    simulate_series,
)
from src.api import app


def _wave(n: int = 160) -> list[float]:
    t = np.arange(n, dtype=np.float64)
    rng = np.random.default_rng(5)
    return list(50 + 10 * np.sin(2 * np.pi * t / 20) + rng.normal(0, 0.5, n))


@pytest.fixture(autouse=True)
def _fresh_budget():
    jepa_world.reset_budget()
    clear_simulation_cache()
    yield
    jepa_world.reset_budget()


def test_encoder_learns_bounded_non_collapsed_features():
    values = np.asarray(_wave(), dtype=np.float64)
    window = 20
    xs = np.stack([values[i : i + window] for i in range(len(values) - window)])
    scaled = (xs - xs.mean(axis=0)) / xs.std(axis=0)
    enc = train_jepa_encoder(scaled, seed=0)
    feats = enc.encode(scaled)
    assert feats.shape == (len(scaled), enc.latent_dim)
    assert np.all(np.abs(feats) <= 1.0)
    # Variance regularizer keeps features alive (no collapse to a constant).
    assert enc.feature_std > 0.05
    # The SSL prediction loss goes down during training.
    assert enc.loss_end < enc.loss_start
    # Deterministic for a fixed seed.
    again = train_jepa_encoder(scaled, seed=0)
    np.testing.assert_allclose(again.encode(scaled), feats, atol=1e-5)


def test_fit_with_jepa_adds_latent_dims_and_stays_stable():
    fitted = fit_world_model(np.asarray(_wave()), encoder="jepa")
    assert fitted.encoder == "jepa"
    assert fitted.latent_dim == fitted.pca_dim + fitted.jepa.latent_dim  # type: ignore[union-attr]
    assert 0.5 <= fitted.damping <= 1.0


def test_auto_reports_both_encoders_and_picks_by_backtest():
    result = simulate_series(_wave(), horizon=8, n_paths=32, encoder="auto")
    cmp = result["encoder_comparison"]
    assert cmp["pca"]["mae"] > 0
    assert cmp["jepa"]["mae"] > 0
    assert cmp["chosen"] in {"pca", "jepa"}
    assert result["model"]["encoder"] == cmp["chosen"]
    if cmp["chosen"] == "jepa":
        assert cmp["improvement_pct"] >= 2.0
        assert result["backtest"]["mae_model"] == cmp["jepa"]["mae"]
    else:
        assert result["backtest"]["mae_model"] == cmp["pca"]["mae"]
    assert "Encoder check" in result["summary"]
    lo, mid, hi = (result["baseline"][k] for k in ("p10", "p50", "p90"))
    assert all(a <= b <= c for a, b, c in zip(lo, mid, hi, strict=True))


def test_default_stays_pca_without_comparison():
    result = simulate_series(_wave(), horizon=5, n_paths=16)
    assert result["model"]["encoder"] == "pca"
    assert "encoder_comparison" not in result


def test_daily_cap_falls_back_to_pca_on_auto(monkeypatch):
    monkeypatch.setattr(jepa_world, "DAILY_FIT_CAP", 0)
    result = simulate_series(_wave(), horizon=5, n_paths=16, encoder="auto")
    assert result["model"]["encoder"] == "pca"
    assert result["encoder_comparison"]["jepa"] is None
    assert "cap" in result["encoder_comparison"]["note"]
    with pytest.raises(JepaBusyError):
        simulate_series(_wave(), horizon=5, n_paths=16, encoder="jepa")


def test_endpoint_accepts_encoder_and_rejects_unknown(monkeypatch):
    client = TestClient(app)
    ok = client.post(
        "/analysis/simulate",
        json={"starter_id": "lunar", "horizon": 5, "n_paths": 32, "encoder": "auto"},
    )
    assert ok.status_code == 200, ok.text
    body = ok.json()
    assert body["encoder_comparison"]["chosen"] in {"pca", "jepa"}
    bad = client.post(
        "/analysis/simulate",
        json={"starter_id": "lunar", "horizon": 5, "encoder": "gpu"},
    )
    assert bad.status_code == 422
    monkeypatch.setattr(jepa_world, "DAILY_FIT_CAP", 0)
    busy = client.post(
        "/analysis/simulate",
        json={"values": _wave(80), "horizon": 4, "n_paths": 16, "encoder": "jepa"},
    )
    assert busy.status_code == 503


def test_panel_is_wired_into_the_shell():
    client = TestClient(app)
    html = client.get("/").text
    assert 'data-tool-view="world-encoder"' in html
    assert "/assets/world-encoder.js" in html
    assert client.get("/assets/world-encoder.js").status_code == 200
    tools = client.get("/assets/tools.js").text
    assert 'id: "world-encoder"' in tools
