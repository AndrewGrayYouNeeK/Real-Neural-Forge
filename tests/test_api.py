"""Tests for the FastAPI endpoints."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

from src.api import _state, app, load_model


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

    def test_model_not_loaded_error(self):
        """Test that predict returns 503 when model is not in state."""
        # Create a fresh TestClient without loading model via lifespan
        from src.api import FastAPI, predict, health

        # Create a minimal app without lifespan
        test_app = FastAPI()
        test_app.get("/health")(health)
        test_app.post("/predict")(predict)

        # Clear state to simulate model not loaded
        _state.clear()

        with TestClient(test_app) as client:
            payload = {"sequence": [[0.1], [0.2], [0.3]]}
            resp = client.post("/predict", json=payload)
            assert resp.status_code == 503
            assert "Model not loaded" in resp.json()["detail"]


class TestLoadModel:
    def test_load_model_with_checkpoint(self, tmp_path):
        """Test loading a model with an existing checkpoint."""
        # Load default config
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        # Create a temporary checkpoint
        from src.model import TimeSeriesTransformer

        model = TimeSeriesTransformer(
            input_dim=cfg["model"]["input_dim"],
            output_dim=cfg["model"]["output_dim"],
            d_model=cfg["model"]["d_model"],
            nhead=cfg["model"]["nhead"],
            num_encoder_layers=cfg["model"]["num_encoder_layers"],
            dim_feedforward=cfg["model"]["dim_feedforward"],
            dropout=cfg["model"]["dropout"],
            max_seq_len=cfg["model"]["max_seq_len"],
        )

        checkpoint_path = tmp_path / "test_checkpoint.pt"
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "loss": 0.1,
                "config": cfg,
            },
            checkpoint_path,
        )

        # Modify config to point to the temporary checkpoint
        cfg["inference"]["checkpoint_path"] = str(checkpoint_path)
        tmp_config = tmp_path / "config.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(cfg, f)

        # Load model with the checkpoint
        load_model(str(tmp_config))

        # Verify model was loaded
        assert _state["model"] is not None
        assert _state["device"] is not None
        assert _state["config"] is not None

    @patch("torch.cuda.is_available")
    def test_load_model_cuda_fallback(self, mock_cuda_available, tmp_path):
        """Test CUDA fallback when CUDA is requested but not available."""
        mock_cuda_available.return_value = False

        # Load default config and modify it
        with open("config/config.yaml") as f:
            cfg = yaml.safe_load(f)

        cfg["inference"]["device"] = "cuda"
        tmp_config = tmp_path / "config.yaml"
        with open(tmp_config, "w") as f:
            yaml.dump(cfg, f)

        # Load model - should fall back to CPU
        load_model(str(tmp_config))

        # Verify device is CPU
        assert _state["device"] == torch.device("cpu")


class TestPredictRequestValidation:
    def test_inconsistent_feature_dims(self, client):
        """Test that sequences with inconsistent feature dimensions are rejected."""
        # First step has 1 feature, second has 2
        payload = {"sequence": [[0.1], [0.2, 0.3]]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 422

    def test_long_sequence(self, client):
        """Test prediction with a long sequence."""
        # Create a sequence of 100 timesteps
        payload = {"sequence": [[i * 0.01] for i in range(100)]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "prediction" in data
        assert len(data["prediction"]) == 1

    def test_negative_values(self, client):
        """Test that negative values are accepted."""
        payload = {"sequence": [[-0.5], [-0.3], [-0.1]]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200

    def test_zero_values(self, client):
        """Test that zero values are accepted."""
        payload = {"sequence": [[0.0], [0.0], [0.0]]}
        resp = client.post("/predict", json=payload)
        assert resp.status_code == 200
