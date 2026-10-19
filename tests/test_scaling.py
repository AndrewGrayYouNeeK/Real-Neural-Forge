"""Tests for train-split feature scaling."""

import numpy as np
import pandas as pd
import pytest
import torch

from src.data.scaling import StandardScaler
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.storage.experiments import ExperimentStore
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config


class TestStandardScaler:
    def test_fit_transform_roundtrip(self):
        x = torch.randn(10, 5, 2) * 50 + 1000
        y = torch.randn(10, 1) * 50 + 1000
        s = StandardScaler.fit(x, y)
        xs = s.transform_inputs(x).reshape(-1, 2)
        assert torch.allclose(xs.mean(0), torch.zeros(2), atol=1e-4)
        assert torch.allclose(xs.std(0, unbiased=False), torch.ones(2), atol=1e-4)
        assert torch.allclose(s.inverse_targets(s.transform_targets(y)), y, atol=1e-3)

    def test_constant_column_does_not_nan(self):
        s = StandardScaler.fit(torch.ones(4, 3, 1), torch.ones(4, 1))
        assert torch.isfinite(s.transform_inputs(torch.ones(1, 3, 1))).all()

    def test_state_dict_roundtrip(self):
        s = StandardScaler.fit(torch.randn(6, 4, 2), torch.randn(6, 1))
        r = StandardScaler.from_state_dict(s.state_dict())
        assert torch.allclose(r.x_mean, s.x_mean) and torch.allclose(r.y_std, s.y_std)

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            StandardScaler.fit(torch.empty(0, 3, 1), torch.empty(0, 1))


def _csv_cfg(tmp_path, val_split=0.25):
    values = 1000 + 100 * np.sin(np.linspace(0, 12 * np.pi, 120))
    path = tmp_path / "series.csv"
    pd.DataFrame({"value": values}).to_csv(path, index=False)
    cfg = load_config("config/config.yaml")
    cfg["data"] = {
        "source": "csv",
        "path": str(path),
        "feature_columns": ["value"],
        "target_column": "value",
        "seq_len": 8,
        "val_split": val_split,
        "split_seed": 0,
    }
    cfg["training"].update(
        epochs=2,
        batch_size=16,
        device="cpu",
        seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    cfg["model"].update(d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32)
    return cfg


def _train(tmp_path, cfg):
    t = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
    t.cfg = cfg
    t.store = ExperimentStore(str(tmp_path / "exp.db"))
    return t.train()


class TestCsvScalingPipeline:
    def test_scaler_saved_and_fit_on_train_split_only(self, tmp_path):
        from torch.utils.data import random_split

        from src.data.timeseries import load_csv_timeseries

        cfg = _csv_cfg(tmp_path)
        cfg["data"]["split_mode"] = "random"
        _train(tmp_path, cfg)
        ckpt = torch.load(tmp_path / "ckpt" / "best_model.pt", weights_only=True)
        assert ckpt["scaler"] is not None

        x, y = load_csv_timeseries(cfg["data"]["path"], ["value"], "value", 8)
        n_val = int(len(x) * 0.25)
        train_ds, _ = random_split(
            range(len(x)),
            [len(x) - n_val, n_val],
            generator=torch.Generator().manual_seed(0),
        )
        idx = list(train_ds.indices)
        expected = StandardScaler.fit(x[idx], y[idx])
        assert ckpt["scaler"]["y_mean"] == pytest.approx(
            expected.y_mean.tolist(), rel=1e-5
        )
        full = StandardScaler.fit(x, y)
        assert ckpt["scaler"]["y_mean"] != pytest.approx(full.y_mean.tolist(), rel=1e-7)

    def test_metrics_and_predictions_in_original_units(self, tmp_path):
        cfg = _csv_cfg(tmp_path)
        result = _train(tmp_path, cfg)
        # Targets are ~1000±100; an untrained model in scaled space predicts ~0,
        # so original-unit predictions must land near the data's range.
        assert result["metrics"]["mae"] < 300
        predictor = Predictor.from_checkpoint(
            build_model(cfg), tmp_path / "ckpt" / "best_model.pt"
        )
        assert predictor.scaler is not None
        out = predictor.predict(torch.full((8, 1), 1000.0))
        assert 700 < out.item() < 1300

    def test_synthetic_data_unscaled_by_default(self, tmp_path):
        cfg = load_config("config/config.yaml")
        cfg["data"].update(n_samples=16, seq_len=8)
        cfg["training"].update(
            epochs=1, batch_size=8, checkpoint_dir=str(tmp_path / "ckpt")
        )
        cfg["model"].update(
            d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32
        )
        _train(tmp_path, cfg)
        ckpt = torch.load(tmp_path / "ckpt" / "best_model.pt", weights_only=True)
        assert ckpt["scaler"] is None


class TestSplitMode:
    def test_csv_defaults_to_temporal_split(self, tmp_path):
        from src.data.timeseries import load_csv_timeseries

        cfg = _csv_cfg(tmp_path)
        _train(tmp_path, cfg)
        ckpt = torch.load(tmp_path / "ckpt" / "best_model.pt", weights_only=True)
        x, y = load_csv_timeseries(cfg["data"]["path"], ["value"], "value", 8)
        n_fit = len(x) - int(len(x) * 0.25)
        expected = StandardScaler.fit(x[:n_fit], y[:n_fit])
        assert ckpt["scaler"]["y_mean"] == pytest.approx(
            expected.y_mean.tolist(), rel=1e-5
        )

    def test_temporal_split_on_synthetic_data(self, tmp_path):
        cfg = load_config("config/config.yaml")
        cfg["data"].update(
            n_samples=20, seq_len=8, val_split=0.25, split_mode="temporal"
        )
        cfg["training"].update(
            epochs=1, batch_size=8, checkpoint_dir=str(tmp_path / "c")
        )
        cfg["model"].update(
            d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32
        )
        result = _train(tmp_path, cfg)
        assert "val_loss" in result["history"][0]

    def test_unknown_split_mode_rejected(self, tmp_path):
        cfg = _csv_cfg(tmp_path)
        cfg["data"]["split_mode"] = "shuffle"
        with pytest.raises(ValueError, match="split_mode"):
            _train(tmp_path, cfg)
