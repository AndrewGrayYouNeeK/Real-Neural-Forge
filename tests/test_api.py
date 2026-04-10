"""Tests for the FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api import app, load_model


@pytest.fixture(scope="module")
def client():
    """Create a test client with the model loaded from the default config."""
    load_model("config/config.yaml")
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


class TestHealthEndpoint:
    def test_health_ok(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


class TestPredictEndpoint:
    def test_valid_request(self, client):
        payload = {"sequence": [[0.1], [0.2], [0.3], [0.4], [0.5]]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert isinstance(data["prediction"], list)
        assert len(data["prediction"]) == 1

    def test_single_timestep(self, client):
        payload = {"sequence": [[0.0]]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_empty_sequence_rejected(self, client):
        payload = {"sequence": []}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_wrong_feature_dim_rejected(self, client):
        # model expects input_dim=1, send 3 features
        payload = {"sequence": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_missing_sequence_field_rejected(self, client):
        resp = client.post("/predict", json={})
        assert resp.status_code == 422

    def test_response_is_list_of_floats(self, client):
        payload = {"sequence": [[i * 0.1] for i in range(10)]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        for val in resp.json()["prediction"]:
            assert isinstance(val, float)
