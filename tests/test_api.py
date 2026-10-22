"""Tests for the FastAPI endpoints."""

import os
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
import torch
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import src.api as api_module
import src.space.approaches as approaches
import src.space.solar_system as solar_system
from src import project_version
from src.api import _state, app, load_model
from src.models.registry import build_model
from src.utils.paths import REPO_ROOT


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
        from src.api import FastAPI, health, predict

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
        assert "step 1 has 2 feature(s)" in resp.json()["detail"]

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


class TestNewEndpoints:
    def test_model_info(self, client):
        resp = client.get("/model/info")
        assert resp.status_code == 200
        data = resp.json()
        assert data["architecture"] == "time_series_transformer"
        assert "parameters" in data
        assert "checkpoint_loaded" in data
        assert "checkpoint_stale" in data

    def test_training_status(self, client):
        resp = client.get("/training/status")
        assert resp.status_code == 200
        assert "status" in resp.json()

    def test_list_experiments(self, client):
        resp = client.get("/experiments")
        assert resp.status_code == 200
        assert "experiments" in resp.json()

    def test_list_experiments_rejects_invalid_limit(self, client):
        # SQLite treats a negative LIMIT as "no limit", so reject it up front.
        for bad in (-1, 0, 501):
            resp = client.get("/experiments", params={"limit": bad})
            assert resp.status_code == 422

    def test_dashboard(self, client):
        resp = client.get("/")
        assert resp.status_code == 200


def _idle_training() -> None:
    _state["training"] = {"status": "idle", "result": None}


class TestTrainConfigPath:
    @pytest.fixture(autouse=True)
    def _reset_training_state(self):
        _idle_training()
        yield
        _idle_training()

    def test_default_config_starts_training(self, client):
        with patch("src.api._run_training") as run:
            resp = client.post("/train", json={})
        assert resp.status_code == 200
        assert resp.json()["status"] == "running"
        started = Path(run.call_args.args[0])
        assert started == (Path("config/config.yaml")).resolve()
        assert started.is_relative_to(Path("config").resolve())

    def test_normalized_path_inside_config_is_accepted(self, client):
        with patch("src.api._run_training") as run:
            resp = client.post(
                "/train", json={"config_path": "config/../config/config.yaml"}
            )
        assert resp.status_code == 200
        assert Path(run.call_args.args[0]) == Path("config/config.yaml").resolve()

    def test_absolute_config_inside_directory_is_accepted(self, client):
        absolute = str(Path("config/config.yaml").resolve())
        with patch("src.api._run_training") as run:
            resp = client.post("/train", json={"config_path": absolute})
        assert resp.status_code == 200
        assert run.call_args.args[0] == absolute

    def test_rejects_paths_outside_config(self, client):
        outside = [
            "../README.md",
            "config/../../etc/passwd",
            "/etc/passwd",
            str(Path("pyproject.toml").resolve()),
            "config",
        ]
        for config_path in outside:
            resp = client.post("/train", json={"config_path": config_path})
            assert resp.status_code == 422, config_path
            assert "config/" in resp.json()["detail"]
            assert _state["training"]["status"] == "idle"

    def test_rejects_empty_config_path(self, client):
        resp = client.post("/train", json={"config_path": "   "})
        assert resp.status_code == 422
        assert _state["training"]["status"] == "idle"

    def test_rejects_non_yaml_suffix(self, client):
        path = Path("config/_train_reject.txt")
        path.write_text("not yaml\n", encoding="utf-8")
        try:
            resp = client.post("/train", json={"config_path": path.as_posix()})
        finally:
            path.unlink(missing_ok=True)
        assert resp.status_code == 422
        assert ".yaml" in resp.json()["detail"]
        assert _state["training"]["status"] == "idle"

    def test_missing_config_is_not_found(self, client):
        resp = client.post("/train", json={"config_path": "config/does-not-exist.yaml"})
        assert resp.status_code == 404
        assert _state["training"]["status"] == "idle"

    def test_rejects_invalid_yaml(self, client):
        path = Path("config/_train_invalid.yaml")
        path.write_text("model: [\n", encoding="utf-8")
        try:
            resp = client.post("/train", json={"config_path": path.as_posix()})
        finally:
            path.unlink(missing_ok=True)
        assert resp.status_code == 422
        assert "Invalid YAML" in resp.json()["detail"]
        assert _state["training"]["status"] == "idle"

    def test_rejects_config_missing_sections(self, client):
        path = Path("config/_train_incomplete.yaml")
        path.write_text("model:\n  name: demo\n", encoding="utf-8")
        try:
            resp = client.post("/train", json={"config_path": path.as_posix()})
        finally:
            path.unlink(missing_ok=True)
        assert resp.status_code == 422
        assert "model" in resp.json()["detail"]
        assert _state["training"]["status"] == "idle"

    def test_rejects_symlink_that_leaves_config(self, client, tmp_path):
        outside = tmp_path / "secret.yaml"
        outside.write_text("model: {}\ntraining: {}\n", encoding="utf-8")
        link = Path("config/_train_escape.yaml")
        link.symlink_to(outside)
        try:
            resp = client.post("/train", json={"config_path": link.as_posix()})
        finally:
            link.unlink(missing_ok=True)
        assert resp.status_code == 422
        assert "config/" in resp.json()["detail"]
        assert _state["training"]["status"] == "idle"

    def test_conflict_when_already_running(self, client):
        _state["training"] = {"status": "running", "result": None}
        with patch("src.api._run_training") as run:
            resp = client.post("/train", json={})
        assert resp.status_code == 409
        run.assert_not_called()

    def test_resolves_config_from_repo_root_when_cwd_differs(
        self, client, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        with patch("src.api._run_training") as run:
            resp = client.post("/train", json={"config_path": "config/config.yaml"})
        assert resp.status_code == 200
        assert (
            Path(run.call_args.args[0])
            == (REPO_ROOT / "config" / "config.yaml").resolve()
        )


def _write_cfg(path: Path, cfg: dict) -> None:
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")


def _save_checkpoint(path: Path, cfg: dict) -> None:
    model = build_model(cfg)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "loss": 0.1,
            "config": cfg,
        },
        path,
    )


def _project_version_from_file() -> str:
    in_project = False
    for line in (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        key, separator, raw = stripped.partition("=")
        if separator and key.strip() == "version":
            return raw.strip().strip('"').strip("'")
    raise AssertionError("version is not set in pyproject.toml")


class TestPredictRobustness:
    def test_sequence_at_positional_limit(self, client):
        resp = client.post("/predict", json={"sequence": [[0.0]] * 512})
        assert resp.status_code == 200
        assert all(isinstance(value, float) for value in resp.json()["prediction"])

    def test_sequence_over_positional_limit(self, client):
        resp = client.post("/predict", json={"sequence": [[0.0]] * 513})
        assert resp.status_code == 422
        assert (
            resp.json()["detail"] == "sequence length 513 exceeds the maximum of 512."
        )

    def test_limit_follows_loaded_model(self, client, tmp_path):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        cfg["model"]["max_seq_len"] = 4
        cfg["inference"]["checkpoint_path"] = str(tmp_path / "missing.pt")
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            accepted = client.post("/predict", json={"sequence": [[0.1]] * 4})
            rejected = client.post("/predict", json={"sequence": [[0.1]] * 5})
        finally:
            load_model("config/config.yaml")
        assert accepted.status_code == 200
        assert rejected.status_code == 422
        assert (
            rejected.json()["detail"] == "sequence length 5 exceeds the maximum of 4."
        )

    @pytest.mark.parametrize("literal", ["NaN", "Infinity", "-Infinity"])
    def test_non_finite_values_rejected(self, client, literal):
        # Send the token literally. Recent HTTP clients refuse to encode NaN,
        # but Python's JSON parser accepts it and the API must still reject it.
        body = f'{{"sequence": [[0.1], [{literal}]]}}'.encode()
        resp = client.post(
            "/predict",
            content=body,
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 422
        assert "step 1" in resp.text
        assert "finite" in resp.text


class TestCheckpointReporting:
    def test_missing_checkpoint_is_not_loaded(self, client, tmp_path):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        missing = tmp_path / "missing.pt"
        cfg["inference"]["checkpoint_path"] = str(missing)
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            info = client.get("/model/info").json()
            assert info["checkpoint_loaded"] is False
            assert info["checkpoint_stale"] is False
            missing.write_bytes(b"checkpoint appeared after startup")
            info = client.get("/model/info").json()
            assert info["checkpoint_loaded"] is False
            assert info["checkpoint_stale"] is True
        finally:
            load_model("config/config.yaml")

    def test_loaded_checkpoint_reports_newer_file(self, client, tmp_path):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        checkpoint = tmp_path / "model.pt"
        _save_checkpoint(checkpoint, cfg)
        cfg["inference"]["checkpoint_path"] = str(checkpoint)
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            info = client.get("/model/info").json()
            assert info["checkpoint_loaded"] is True
            assert info["checkpoint_stale"] is False
            loaded_ns = checkpoint.stat().st_mtime_ns
            os.utime(
                checkpoint, ns=(loaded_ns + 1_000_000_000, loaded_ns + 1_000_000_000)
            )
            info = client.get("/model/info").json()
            assert info["checkpoint_loaded"] is True
            assert info["checkpoint_stale"] is True
            assert info["checkpoint_path"] == str(checkpoint.resolve())
        finally:
            load_model("config/config.yaml")

    def test_relative_checkpoint_loads_from_repo_root(
        self, client, tmp_path, monkeypatch
    ):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        repo = tmp_path / "repo"
        checkpoint = repo / "checkpoints" / "best_model.pt"
        checkpoint.parent.mkdir(parents=True)
        _save_checkpoint(checkpoint, cfg)
        cfg["inference"]["checkpoint_path"] = "checkpoints/best_model.pt"
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setattr("src.utils.paths.REPO_ROOT", repo)
        try:
            load_model(config_path)
            info = client.get("/model/info").json()
            assert info["checkpoint_loaded"] is True
            assert info["checkpoint_stale"] is False
            assert info["checkpoint_path"] == str(checkpoint.resolve())
            assert not (elsewhere / "checkpoints").exists()
        finally:
            monkeypatch.setattr("src.utils.paths.REPO_ROOT", REPO_ROOT)
            load_model("config/config.yaml")


class TestExperimentStoreConfig:
    def test_api_uses_configured_db_path(self, client, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        db_path = tmp_path / "custom.db"
        cfg["storage"]["db_path"] = str(db_path)
        cfg["inference"]["checkpoint_path"] = str(tmp_path / "missing.pt")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            assert api_module._store is not None
            assert api_module._store.db_path == db_path.resolve()
            exp_id = api_module._store.create_experiment(
                name="from-config", config={"epochs": 1}, model_name="m"
            )
            listed = client.get("/experiments").json()["experiments"]
            assert listed[0]["id"] == exp_id
            assert listed[0]["name"] == "from-config"
            assert not (tmp_path / "data" / "experiments.db").exists()
        finally:
            load_model("config/config.yaml")

    def test_relative_db_path_ignores_cwd(self, client, tmp_path, monkeypatch):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        cfg["storage"]["db_path"] = "nested/runs.db"
        cfg["inference"]["checkpoint_path"] = str(tmp_path / "missing.pt")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        repo = tmp_path / "repo"
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        monkeypatch.setattr("src.utils.paths.REPO_ROOT", repo)
        try:
            load_model(config_path)
            assert api_module._store is not None
            assert api_module._store.db_path == (repo / "nested" / "runs.db").resolve()
            assert not (elsewhere / "nested").exists()
        finally:
            monkeypatch.setattr("src.utils.paths.REPO_ROOT", REPO_ROOT)
            load_model("config/config.yaml")

    def test_default_config_loads_from_other_cwd(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        try:
            load_model("config/config.yaml")
            assert _state["config"]["model"]["name"] == "time_series_transformer"
            checkpoint_path = _state["checkpoint_path"]
            assert isinstance(checkpoint_path, Path)
            assert checkpoint_path.is_relative_to(REPO_ROOT)
            assert not (tmp_path / "checkpoints").exists()
            assert not (tmp_path / "data").exists()
        finally:
            load_model("config/config.yaml")


class TestTrainProtection:
    @pytest.fixture(autouse=True)
    def _reset_training_state(self, monkeypatch):
        monkeypatch.delenv("NEURAL_FORGE_API_TOKEN", raising=False)
        previous = _state["config"].get("api", {}).get("training_enabled", True)
        _idle_training()
        yield
        _state["config"].setdefault("api", {})["training_enabled"] = previous
        _idle_training()

    def test_open_when_token_unset(self, client):
        with patch("src.api._run_training") as run:
            resp = client.post("/train", json={})
        assert resp.status_code == 200
        run.assert_called_once()

    def test_blank_token_keeps_training_open(self, client, monkeypatch):
        monkeypatch.setenv("NEURAL_FORGE_API_TOKEN", "   ")
        with patch("src.api._run_training") as run:
            resp = client.post("/train", json={})
        assert resp.status_code == 200
        run.assert_called_once()

    def test_token_required_when_set(self, client, monkeypatch):
        monkeypatch.setenv("NEURAL_FORGE_API_TOKEN", "secret-token")
        with patch("src.api._run_training") as run:
            missing = client.post("/train", json={})
            wrong = client.post(
                "/train", json={}, headers={"Authorization": "Bearer nope"}
            )
            ok = client.post(
                "/train",
                json={},
                headers={"Authorization": "Bearer secret-token"},
            )
        assert missing.status_code == 401
        assert "token" in missing.json()["detail"].lower()
        assert missing.headers["www-authenticate"].lower().startswith("bearer")
        assert wrong.status_code == 401
        assert ok.status_code == 200
        run.assert_called_once()
        assert _state["training"]["status"] == "running"

    def test_disabled_training_rejects_even_with_token(self, client, monkeypatch):
        monkeypatch.setenv("NEURAL_FORGE_API_TOKEN", "secret-token")
        _state["config"].setdefault("api", {})["training_enabled"] = False
        with patch("src.api._run_training") as run:
            resp = client.post(
                "/train",
                json={},
                headers={"Authorization": "Bearer secret-token"},
            )
        assert resp.status_code == 403
        assert "disabled" in resp.json()["detail"].lower()
        run.assert_not_called()
        assert _state["training"]["status"] == "idle"


class TestForecastDashboard:
    def test_series_recent_matches_predict(self, client):
        load_model("config/config.yaml")
        first = client.get("/series/recent", params={"points": 16})
        second = client.get("/series/recent", params={"points": 16})
        assert first.status_code == 200
        payload = first.json()
        assert payload["source"] == "synthetic"
        assert len(payload["history"]) == 16
        assert payload["forecast"][0]["p10"] is None
        assert payload["forecast"][0]["p90"] is None
        assert "persistence" in payload["metrics"]["baselines"]
        assert "mean" in payload["metrics"]["baselines"]
        assert "checkpoint_loaded" in payload
        assert first.json()["history"] == second.json()["history"]
        predicted = client.post(
            "/predict", json={"sequence": payload["predict_window"]}
        )
        assert predicted.status_code == 200
        assert predicted.json()["prediction"][0] == pytest.approx(
            payload["forecast"][0]["value"], abs=1e-5
        )

    def test_series_rejects_bad_points(self, client):
        assert client.get("/series/recent", params={"points": 7}).status_code == 422
        assert client.get("/series/recent", params={"points": 401}).status_code == 422

    def test_series_requires_model(self):
        test_app = FastAPI()
        test_app.get("/series/recent")(api_module.series_recent)
        saved = dict(_state)
        saved_store = api_module._store
        _state.clear()
        try:
            with TestClient(test_app) as local:
                response = local.get("/series/recent")
                assert response.status_code == 503
        finally:
            _state.clear()
            _state.update(saved)
            api_module._store = saved_store

    def test_csv_series_and_stored_predictions(self, client, tmp_path):
        csv_path = tmp_path / "series.csv"
        csv_path.write_text(
            "timestamp,value\n"
            + "".join(f"2024-01-{day:02d},{day}\n" for day in range(1, 13)),
            encoding="utf-8",
        )
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        cfg["model"].update(
            {
                "d_model": 8,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 16,
                "max_seq_len": 32,
            }
        )
        cfg["data"] = {
            "source": "csv",
            "path": str(csv_path),
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 4,
        }
        cfg["inference"]["checkpoint_path"] = str(tmp_path / "missing.pt")
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            recent = client.get("/series/recent", params={"points": 8})
            assert recent.status_code == 200
            body = recent.json()
            assert body["source"] == "csv"
            assert body["history"][0]["time"].startswith("2024-01-")
            assert body["checkpoint_loaded"] is False

            store = api_module._store
            assert store is not None
            experiment_id = store.create_experiment(
                name="<script>alert(1)</script>",
                config={"model": {}},
                model_name="transformer",
            )
            store.complete_experiment(
                experiment_id,
                status="completed",
                metrics={
                    "mae": 0.2,
                    "validation_predictions": {
                        "actual": [1.0, 2.0],
                        "predicted": [1.2, 1.8],
                        "p10": [0.4, 1.0],
                        "p90": [1.6, 2.4],
                    },
                },
            )
            found = client.get(f"/experiments/{experiment_id}/predictions")
            assert found.status_code == 200
            assert found.json()["source"] == "stored"
            assert found.json()["points"][0]["p90"] == 1.6
            assert client.get("/experiments/999999/predictions").status_code == 404
        finally:
            load_model("config/config.yaml")

    def test_missing_dataset_is_422(self, client, tmp_path):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        cfg["data"] = {
            "source": "csv",
            "path": str(tmp_path / "absent.csv"),
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 4,
        }
        cfg["inference"]["checkpoint_path"] = str(tmp_path / "missing.pt")
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            response = client.get("/series/recent")
        finally:
            load_model("config/config.yaml")
        assert response.status_code == 422
        assert "absent.csv" in response.json()["detail"]

    def test_dashboard_markup_and_escaping(self, client):
        page = client.get("/")
        assert page.status_code == 200
        html = page.text
        assert 'id="forecast-chart"' in html
        assert 'id="powerball-summary"' in html
        assert 'id="powerball-body"' in html
        assert 'id="near-earth-heading"' in html
        assert 'id="near-earth-summary"' in html
        assert 'id="near-earth-days"' in html
        assert 'id="solar-system-heading"' in html
        assert 'id="solar-system-days"' in html
        assert 'id="solar-system-summary"' in html
        assert "/assets/chart.js" in html
        assert 'id="api-token"' in html
        for path in ("/assets/app.js", "/assets/chart.js"):
            script = client.get(path)
            assert script.status_code == 200
            source = script.text
            assert ".innerHTML" not in source
            assert "insertAdjacentHTML" not in source
            assert "document.write" not in source
            assert "textContent" in source
        app_js = client.get("/assets/app.js").text
        assert "neuralForgeApiToken" in app_js
        assert "Authorization" in app_js
        assert "Bearer " in app_js
        assert "/space/near-earth" in app_js
        assert "/space/solar-system" in app_js


class TestPowerballScorecard:
    def test_missing_file_is_available_false(self, client, monkeypatch, tmp_path):
        monkeypatch.setattr(
            api_module, "POWERBALL_SCORECARD", str(tmp_path / "missing.json")
        )
        response = client.get("/powerball/scorecard")
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is False
        assert "summary" in body

    def test_published_file_is_available(self, client, monkeypatch, tmp_path):
        path = tmp_path / "scorecard.json"
        path.write_text('{"summary": "Scored 1 draws.", "cumulative": {}}')
        monkeypatch.setattr(api_module, "POWERBALL_SCORECARD", str(path))
        response = client.get("/powerball/scorecard")
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is True
        assert body["summary"] == "Scored 1 draws."


class TestAppVersion:
    def test_version_comes_from_pyproject(self, client):
        assert app.version == project_version() == _project_version_from_file()
        assert client.get("/openapi.json").json()["info"]["version"] == app.version


class TestForecastApi:
    def test_default_predict_response_is_unchanged(self, client):
        resp = client.post("/predict", json={"sequence": [[0.1], [0.2], [0.3], [0.4]]})
        assert resp.status_code == 200
        assert set(resp.json()) == {"prediction"}
        assert len(resp.json()["prediction"]) == 1

    def test_explicit_one_step_horizon_stays_a_point(self, client):
        resp = client.post(
            "/predict",
            json={"sequence": [[0.1], [0.2], [0.3]], "horizon": 1},
        )
        assert resp.status_code == 200
        assert set(resp.json()) == {"prediction"}

    def test_multi_step_predict_extends_the_response(self, client):
        resp = client.post(
            "/predict",
            json={"sequence": [[0.1], [0.2], [0.3], [0.4]], "horizon": 4},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert len(body["forecast"]) == 4
        assert body["prediction"] == body["forecast"][0]
        assert body["horizon"] == 4
        assert body["strategy"] == "recursive"
        assert "lower" not in body
        assert body["p50"] == [step[0] for step in body["forecast"]]
        assert "p10" not in body

    def test_forecast_endpoint_defaults_to_configured_horizon(self, client):
        resp = client.post("/forecast", json={"sequence": [[0.1], [0.2], [0.3]]})
        assert resp.status_code == 200
        body = resp.json()
        assert body["horizon"] == 1
        assert body["strategy"] == "direct"
        assert body["forecast"] == [body["prediction"]]
        assert body["p50"] == [body["prediction"][0]]

    def test_model_info_reports_forecast_settings(self, client):
        info = client.get("/model/info").json()
        assert info["horizon"] == 1
        assert info["forecast_strategy"] == "direct"
        assert info["quantiles"] == []
        assert "lstm" in info["available_models"]

    def test_horizon_zero_is_rejected(self, client):
        resp = client.post("/predict", json={"sequence": [[0.1]], "horizon": 0})
        assert resp.status_code == 422

    def test_recursive_without_a_target_slot_is_rejected(self, client, tmp_path):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        cfg["model"]["input_dim"] = 2
        cfg["model"]["output_dim"] = 1
        cfg["forecast"] = {
            "horizon": 1,
            "strategy": "direct",
            "target_feature_index": None,
        }
        cfg["inference"]["checkpoint_path"] = str(tmp_path / "missing.pt")
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            resp = client.post(
                "/predict",
                json={"sequence": [[0.1, 0.2], [0.3, 0.4]], "horizon": 3},
            )
        finally:
            load_model("config/config.yaml")
        assert resp.status_code == 422
        assert "target_feature_index" in resp.json()["detail"]

    def test_loaded_quantile_checkpoint_returns_intervals(self, client, tmp_path):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        cfg["model"].update(
            d_model=8, nhead=2, num_encoder_layers=1, dim_feedforward=16
        )
        cfg["forecast"] = {
            "horizon": 2,
            "strategy": "direct",
            "quantiles": [0.1, 0.5, 0.9],
        }
        checkpoint = tmp_path / "quantiles.pt"
        model = build_model(cfg)
        torch.save(
            {
                "epoch": 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": {},
                "loss": 0.1,
                "config": cfg,
                "scaler": None,
                "forecast": {
                    "horizon": 2,
                    "strategy": "direct",
                    "quantiles": [0.1, 0.5, 0.9],
                    "season_length": None,
                    "target_feature_index": 0,
                    "nominal_coverage": 0.8,
                    "n_lags": None,
                    "interval_method": "cqr",
                    "conformal_qhat": [0.2, 0.4],
                    "validation_coverage": 0.7,
                    "coverage_horizon": 2,
                },
            },
            checkpoint,
        )
        cfg["inference"]["checkpoint_path"] = str(checkpoint)
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            multi = client.post(
                "/forecast",
                json={"sequence": [[0.1], [0.2], [0.3], [0.4]]},
            )
            one = client.post(
                "/predict",
                json={
                    "sequence": [[0.1], [0.2], [0.3], [0.4]],
                    "horizon": 1,
                    "include_intervals": True,
                },
            )
        finally:
            load_model("config/config.yaml")
        body = multi.json()
        assert multi.status_code == 200
        assert body["horizon"] == 2
        assert body["interval_method"] == "cqr"
        assert len(body["lower"]) == 2
        assert len(body["upper"]) == 2
        assert body["validation_coverage"] == 0.7
        assert body["quantiles"] == [0.1, 0.5, 0.9]
        assert body["p50"] == [step[0] for step in body["forecast"]]
        assert len(body["p10"]) == len(body["p90"]) == 2
        assert body["p10"][0] <= body["p50"][0] <= body["p90"][0]
        one_body = one.json()
        assert one.status_code == 200
        assert "forecast" not in one_body
        assert len(one_body["lower"]) == 1
        assert one_body["p50"] == [one_body["prediction"][0]]
        assert len(one_body["p10"]) == len(one_body["p90"]) == 1
        assert "validation_coverage" not in one_body

    def test_comparison_endpoint_returns_the_latest_stored_table(
        self, client, tmp_path
    ):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "config.yaml").read_text())
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        cfg["inference"]["checkpoint_path"] = str(tmp_path / "missing.pt")
        config_path = tmp_path / "config.yaml"
        _write_cfg(config_path, cfg)
        try:
            load_model(config_path)
            empty = client.get("/comparison")
            assert empty.status_code == 200
            assert empty.json()["models"] == []
            store = api_module._store
            assert store is not None
            older = store.create_experiment(
                name="old:comparison", config={}, model_name="comparison"
            )
            store.complete_experiment(
                older,
                status="completed",
                metrics={"persistence.mae": 1.0, "horizon": 1.0},
            )
            latest = store.create_experiment(
                name="new:comparison", config={}, model_name="comparison"
            )
            store.complete_experiment(
                latest,
                status="completed",
                metrics={"linear_lags.mae": 0.25, "linear_lags.rmse": 0.4},
            )
            found = client.get("/comparison")
        finally:
            load_model("config/config.yaml")
        assert found.status_code == 200
        body = found.json()
        assert body["experiment_id"] == latest
        assert body["models"] == [{"name": "linear_lags", "mae": 0.25, "rmse": 0.4}]


def _hide_near_earth_files(monkeypatch, root: Path) -> None:
    for name in (
        "EVENTS_PATH",
        "FORECAST_PATH",
        "REENTRY_EVENTS_PATH",
        "FIREBALL_EVENTS_PATH",
        "SENTRY_PATH",
        "INTERSTELLAR_PATH",
        "MANIFEST_PATH",
    ):
        monkeypatch.setattr(approaches, name, str(root / name))
    monkeypatch.setattr(approaches, "utc_today", lambda: pd.Timestamp("2026-10-22"))


class TestNearEarth:
    def test_default_window_is_730_days(self, client, monkeypatch, tmp_path):
        _hide_near_earth_files(monkeypatch, tmp_path)
        response = client.get("/space/near-earth")
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is False
        assert body["days"] == 730
        assert body["outlook_months"] == 24
        start = pd.Timestamp(body["window_start"])
        end = pd.Timestamp(body["window_end"])
        assert (end - start).days == 730

    def test_days_above_730_is_rejected(self, client):
        response = client.get("/space/near-earth", params={"days": 731})
        assert response.status_code == 422

    def test_days_query_shortens_the_window(self, client, monkeypatch, tmp_path):
        _hide_near_earth_files(monkeypatch, tmp_path)
        response = client.get("/space/near-earth", params={"days": 90})
        assert response.status_code == 200
        body = response.json()
        assert body["days"] == 90
        start = pd.Timestamp(body["window_start"])
        end = pd.Timestamp(body["window_end"])
        assert (end - start).days == 90


def _hide_solar_files(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "solar"
    for name in (
        "EVENTS_PATH",
        "COMET_CATALOG_PATH",
        "HORIZONS_DAILY_PATH",
        "HORIZONS_MONTHLY_PATH",
        "MOONS_PATH",
        "FORECAST_PATH",
    ):
        monkeypatch.setattr(solar_system, name, str(root / name))
    monkeypatch.setattr(solar_system, "utc_today", lambda: pd.Timestamp("2026-10-22"))


class TestSolarSystem:
    def test_default_window_is_730_days(self, client, monkeypatch, tmp_path):
        _hide_solar_files(monkeypatch, tmp_path)
        response = client.get("/space/solar-system")
        assert response.status_code == 200
        body = response.json()
        assert body["available"] is False
        assert body["days"] == 730
        assert body["outlook_months"] == 24
        assert body["showers"]["available"] is False
        start = pd.Timestamp(body["window_start"])
        end = pd.Timestamp(body["window_end"])
        assert (end - start).days == 730

    def test_days_above_730_is_rejected(self, client):
        response = client.get("/space/solar-system", params={"days": 731})
        assert response.status_code == 422

    def test_days_query_shortens_the_window(self, client, monkeypatch, tmp_path):
        _hide_solar_files(monkeypatch, tmp_path)
        response = client.get("/space/solar-system", params={"days": 90})
        assert response.status_code == 200
        body = response.json()
        assert body["days"] == 90
        start = pd.Timestamp(body["window_start"])
        end = pd.Timestamp(body["window_end"])
        assert (end - start).days == 90
