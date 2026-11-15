"""Tests for synthetic series generation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.analysis.synthetic import generate_synthetic_series
from src.api import app


def test_generate_sine_series():
    payload = generate_synthetic_series(pattern="sine", length=64, noise=0.01, seed=1)
    assert payload["length"] == 64
    assert len(payload["series"]) == 64
    assert "date,value" in payload["csv"]
    assert payload["summary"]


def test_generate_diffusion_or_fallback():
    payload = generate_synthetic_series(
        pattern="diffusion", length=64, noise=0.05, seed=2
    )
    assert payload["pattern"] == "diffusion"
    assert len(payload["series"]) == 64
    assert payload["method"] in {"diffusion", "seasonal_fallback"}


def test_analysis_synthetic_endpoint():
    client = TestClient(app)
    response = client.post(
        "/analysis/synthetic",
        json={"pattern": "seasonal", "length": 48, "noise": 0.02, "seed": 0},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["pattern"] == "seasonal"
    assert len(body["series"]) == 48
