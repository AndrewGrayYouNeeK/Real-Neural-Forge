"""Tests for the FastAPI endpoints."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import yaml
from fastapi.testclient import TestClient

import src.api as api_module
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


class TestAppVersion:
    def test_version_comes_from_pyproject(self, client):
        assert app.version == project_version() == _project_version_from_file()
        assert client.get("/openapi.json").json()["info"]["version"] == app.version
