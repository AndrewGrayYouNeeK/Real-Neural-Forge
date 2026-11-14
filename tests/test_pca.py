"""Tests for covariance-matrix PCA analysis and optional training reduction."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from src.analysis.pca import analyze_dataframe, analyze_matrix, fit_pca
from src.api import app
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.storage.experiments import ExperimentStore
from src.training.trainer import TimeSeriesTrainer
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config


def _correlated_frame(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    a = rng.normal(size=n)
    b = a + 0.05 * rng.normal(size=n)
    c = rng.normal(size=n)
    return pd.DataFrame({"a": a, "b": b, "c": c})


class TestFitPca:
    def test_explained_variance_sums_to_one(self):
        frame = _correlated_frame()
        model = fit_pca(frame.to_numpy())
        assert model.n_components == 3
        assert (
            pytest.approx(float(model.explained_variance_ratio.sum()), abs=1e-9) == 1.0
        )

    def test_a_and_b_move_together(self):
        report = analyze_dataframe(_correlated_frame())
        assert "a" in report.summary and "b" in report.summary
        same = [
            group
            for group in report.groups
            if group["direction"] == "same" and set(group["features"]) >= {"a", "b"}
        ]
        assert same, report.groups

    def test_transform_roundtrip_rank(self):
        x = _correlated_frame().to_numpy()
        model = fit_pca(x, n_components=2)
        z = model.transform(x)
        assert z.shape == (len(x), 2)

    def test_rejects_single_feature(self):
        with pytest.raises(ValueError, match="at least 2"):
            fit_pca(np.ones((10, 1)))

    def test_state_dict_roundtrip(self):
        model = fit_pca(_correlated_frame().to_numpy(), feature_names=["a", "b", "c"])
        restored = type(model).from_state_dict(model.state_dict())
        assert restored.feature_names == ("a", "b", "c")
        assert np.allclose(restored.components, model.components)


class TestPcaApi:
    def test_matrix_endpoint(self):
        client = TestClient(app)
        frame = _correlated_frame(80)
        resp = client.post(
            "/analysis/pca/matrix",
            json={
                "columns": ["a", "b", "c"],
                "rows": frame.values.tolist(),
                "n_components": 2,
            },
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["n_components"] == 2
        assert len(payload["explained_variance_ratio"]) == 2
        assert "summary" in payload

    def test_csv_upload_endpoint(self):
        client = TestClient(app)
        csv_buf = io.StringIO()
        _correlated_frame(60).to_csv(csv_buf, index=False)
        resp = client.post(
            "/analysis/pca",
            content=csv_buf.getvalue().encode("utf-8"),
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 200
        assert resp.json()["n_features"] == 3


def _multi_cfg(tmp_path, *, pca_components: int | None = None):
    frame = _correlated_frame(160)
    path = tmp_path / "multi.csv"
    frame.to_csv(path, index=False)
    cfg = load_config("config/config.yaml")
    cfg["data"] = {
        "source": "csv",
        "path": str(path),
        "feature_columns": ["a", "b", "c"],
        "target_column": "a",
        "seq_len": 8,
        "val_split": 0.25,
        "split_mode": "temporal",
        "normalize": True,
        "pca_report": True,
    }
    if pca_components is not None:
        cfg["data"]["pca_reduce"] = True
        cfg["data"]["pca_components"] = pca_components
        cfg["model"]["input_dim"] = pca_components
    else:
        cfg["model"]["input_dim"] = 3
    cfg["model"].update(
        d_model=16,
        nhead=2,
        num_encoder_layers=1,
        dim_feedforward=32,
        dropout=0.0,
        max_seq_len=64,
    )
    cfg["training"].update(
        epochs=2,
        batch_size=16,
        device="cpu",
        seed=0,
        checkpoint_dir=str(tmp_path / "ckpt"),
        patience=0,
    )
    return cfg


class TestPcaTraining:
    def test_report_without_reduce(self, tmp_path):
        cfg = _multi_cfg(tmp_path)
        trainer = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
        trainer.cfg = cfg
        trainer.store = ExperimentStore(str(tmp_path / "exp.db"))
        result = trainer.train()
        assert "pca" in result
        assert result["pca"]["n_features"] == 3
        assert "summary" in result["pca"]

    def test_reduce_and_checkpoint(self, tmp_path):
        cfg = _multi_cfg(tmp_path, pca_components=2)
        trainer = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
        trainer.cfg = cfg
        trainer.store = ExperimentStore(str(tmp_path / "exp.db"))
        result = trainer.train()
        assert result["pca"]["n_components"] == 2
        ckpt = tmp_path / "ckpt" / "best_model.pt"
        assert ckpt.is_file()
        model = build_model(cfg, ensemble=False)
        predictor = Predictor.from_checkpoint(model, ckpt, device="cpu")
        assert predictor.pca is not None
        assert predictor.pca.n_components == 2
        out = predictor.predict(torch.randn(1, 8, 3))
        assert out.shape[-1] == 1

    def test_reduce_requires_matching_input_dim(self, tmp_path):
        cfg = _multi_cfg(tmp_path, pca_components=2)
        cfg["model"]["input_dim"] = 3
        trainer = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
        trainer.cfg = cfg
        trainer.store = ExperimentStore(str(tmp_path / "exp.db"))
        with pytest.raises(ValueError, match="input_dim"):
            trainer.train()


class TestPcaCheckpointHelper:
    def test_save_includes_pca(self, tmp_path):
        cfg = {
            "model": {
                "name": "time_series_transformer",
                "input_dim": 2,
                "output_dim": 1,
                "d_model": 16,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "max_seq_len": 64,
            }
        }
        model = build_model(cfg, ensemble=False)
        pca = fit_pca(np.random.default_rng(0).normal(size=(40, 3)), n_components=2)
        path = tmp_path / "m.pt"
        save_checkpoint(
            model,
            torch.optim.Adam(model.parameters()),
            1,
            0.1,
            cfg,
            path,
            pca=pca.state_dict(),
        )
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        assert "pca" in loaded
        assert len(loaded["pca"]["components"]) == 2


class TestAnalyzeMatrixWindows:
    def test_accepts_window_tensor(self):
        windows = torch.randn(20, 8, 4)
        report = analyze_matrix(windows, feature_names=["w", "x", "y", "z"])
        assert report.n_samples == 20 * 8
        assert report.model.n_features == 4
