"""Tests for forecast-dashboard series and validation payloads."""

import pytest
import torch
import yaml

from src.dashboard_data import recent_series, validation_predictions
from src.storage.experiments import ExperimentStore
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config


class PersistencePredictor:
    def predict(self, windows: torch.Tensor) -> torch.Tensor:
        if windows.dim() == 2:
            windows = windows.unsqueeze(0)
        return windows[:, -1, :1]


def _synthetic_config() -> dict:
    return {
        "model": {"input_dim": 1, "output_dim": 1},
        "data": {
            "source": "synthetic",
            "seq_len": 8,
            "noise": 0.0,
            "seed": 1,
            "n_samples": 12,
        },
    }


class TestRecentSeries:
    def test_persistence_model_matches_persistence_baseline(self):
        payload = recent_series(_synthetic_config(), PersistencePredictor(), points=12)
        assert payload["source"] == "synthetic"
        assert payload["dataset"] == "synthetic"
        assert len(payload["history"]) == 12
        assert payload["history"][0]["predicted"] is not None
        assert payload["forecast"][0]["p10"] is None
        assert payload["forecast"][0]["value"] == payload["history"][-1]["actual"]
        model = payload["metrics"]["model"]
        persistence = payload["metrics"]["baselines"]["persistence"]
        assert model["n"] == 12
        assert persistence["mae"] == model["mae"]
        assert "mean" in payload["metrics"]["baselines"]
        assert len(payload["predict_window"]) == 8

    def test_repeat_calls_stay_fixed(self):
        first = recent_series(_synthetic_config(), PersistencePredictor(), points=10)
        second = recent_series(_synthetic_config(), PersistencePredictor(), points=10)
        assert first["history"] == second["history"]
        assert first["forecast"] == second["forecast"]

    def test_table_source_with_timestamps(self, tmp_path):
        path = tmp_path / "weather.csv"
        path.write_text(
            "timestamp,value\n"
            "2024-01-01,0\n"
            "2024-01-02,1\n"
            "2024-01-03,nan\n"
            "2024-01-04,3\n"
            "2024-01-05,4\n"
            "2024-01-06,5\n"
            "2024-01-07,6\n",
            encoding="utf-8",
        )
        cfg = {
            "model": {"input_dim": 1, "output_dim": 1},
            "data": {
                "source": "weather",
                "name": "city-temps",
                "path": str(path),
                "feature_columns": ["value"],
                "target_column": "value",
                "seq_len": 3,
            },
        }
        payload = recent_series(cfg, PersistencePredictor(), points=7)
        assert payload["source"] == "weather"
        assert payload["dataset"] == "city-temps"
        assert payload["history"][0]["time"] == "2024-01-01"
        nan_row = next(point for point in payload["history"] if point["index"] == 2)
        assert nan_row["actual"] is None
        assert nan_row["predicted"] is None

    def test_missing_file_raises(self, tmp_path):
        cfg = {
            "model": {"input_dim": 1, "output_dim": 1},
            "data": {
                "source": "csv",
                "path": str(tmp_path / "missing.csv"),
                "feature_columns": ["value"],
                "target_column": "value",
                "seq_len": 4,
            },
        }
        try:
            recent_series(cfg, PersistencePredictor(), points=8)
        except FileNotFoundError as exc:
            assert "missing.csv" in str(exc)
        else:
            raise AssertionError("expected FileNotFoundError")

    def test_unknown_source_without_file_raises(self):
        cfg = {
            "model": {"input_dim": 1, "output_dim": 1},
            "data": {"source": "lottery", "seq_len": 4},
        }
        try:
            recent_series(cfg, PersistencePredictor(), points=8)
        except ValueError as exc:
            assert "lottery" in str(exc)
        else:
            raise AssertionError("expected ValueError")


class TestValidationPredictions:
    def test_stored_points_and_intervals_win(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        experiment_id = store.create_experiment(
            name="stored", config={"model": {}}, model_name="transformer"
        )
        store.complete_experiment(
            experiment_id,
            status="completed",
            best_loss=0.2,
            checkpoint_path=str(tmp_path / "missing.pt"),
            metrics={
                "mae": 0.2,
                "validation_predictions": {
                    "actual": [1, 2, 3],
                    "predicted": [1.1, 1.9, 3.2],
                    "p10": [0.5, 1.5, 2.5],
                    "p50": [1.1, 1.9, 3.2],
                    "p90": [1.4, 2.2, 3.6],
                },
                "comparison": {
                    "models": [
                        {"name": "transformer", "mae": 0.2},
                        {"name": "persistence", "mae": 0.5},
                    ]
                },
            },
        )
        payload = validation_predictions(store, experiment_id, limit=2)
        assert payload is not None
        assert payload["available"] is True
        assert payload["source"] == "stored"
        assert payload["truncated"] is True
        assert [point["actual"] for point in payload["points"]] == [2.0, 3.0]
        assert payload["points"][0]["p10"] == 1.5
        assert payload["comparison"]["models"][1]["name"] == "persistence"
        assert payload["metrics"]["baselines"]["persistence"]["n"] == 1

    def test_missing_checkpoint_is_unavailable(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        experiment_id = store.create_experiment(
            name="empty", config={"data": {}}, model_name="transformer"
        )
        store.complete_experiment(experiment_id, status="failed")
        payload = validation_predictions(store, experiment_id, limit=10)
        assert payload is not None
        assert payload["available"] is False
        assert payload["points"] == []

    def test_bad_checkpoint_does_not_raise(self, tmp_path):
        bad = tmp_path / "bad.pt"
        bad.write_bytes(b"not a checkpoint")
        store = ExperimentStore(tmp_path / "exp.db")
        experiment_id = store.create_experiment(
            name="bad",
            config={"model": {"name": "time_series_transformer", "input_dim": 1}},
            model_name="transformer",
        )
        store.complete_experiment(
            experiment_id,
            status="completed",
            checkpoint_path=str(bad),
            metrics={"mae": 1.0},
        )
        payload = validation_predictions(store, experiment_id, limit=5)
        assert payload is not None
        assert payload["available"] is False
        assert "could not be replayed" in payload["note"]

    def test_unknown_experiment_is_none(self, tmp_path):
        store = ExperimentStore(tmp_path / "exp.db")
        assert validation_predictions(store, 99, limit=5) is None

    def test_replay_matches_trainer_metrics(self, tmp_path):
        cfg = load_config("config/config.yaml")
        cfg["model"].update(
            {
                "d_model": 8,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 16,
                "dropout": 0.0,
                "max_seq_len": 16,
            }
        )
        cfg["data"].update(
            {
                "n_samples": 40,
                "seq_len": 8,
                "noise": 0.01,
                "val_split": 0.25,
                "seed": 0,
                "split_seed": 0,
                "normalize": False,
            }
        )
        cfg["training"].update(
            {
                "epochs": 1,
                "batch_size": 8,
                "device": "cpu",
                "checkpoint_dir": str(tmp_path / "ckpt"),
                "experiment_name": "replay",
                "seed": 0,
                "log_interval": 1,
            }
        )
        cfg["storage"]["db_path"] = str(tmp_path / "exp.db")
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

        trainer = TimeSeriesTrainer(str(config_path))
        result = trainer.train()
        payload = validation_predictions(
            trainer.store, result["experiment_id"], limit=100
        )
        assert payload is not None
        assert payload["available"] is True
        assert payload["source"] == "replay"
        assert payload["split"] == "validation"
        assert payload["metrics"]["model"]["mae"] == pytest.approx(
            result["metrics"]["mae"], abs=1e-4
        )
