"""Tests for the experimental conditional diffusion forecaster."""

from __future__ import annotations

import torch

from src.evaluation.compare import compare_models
from src.models.diffusion import DiffusionForecaster
from src.models.registry import ModelRegistry, build_model
from src.storage.experiments import ExperimentStore
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config


class TestDiffusionForecaster:
    def test_registered(self):
        assert "diffusion_forecaster" in ModelRegistry.list_models()

    def test_forward_point_shape(self):
        model = DiffusionForecaster(
            input_dim=1,
            output_dim=1,
            hidden_size=16,
            horizon=1,
            diffusion_steps=8,
            sample_steps=3,
            n_paths=8,
        )
        model.eval()
        out = model(torch.randn(4, 16, 1))
        assert out.shape == (4, 1)

    def test_forward_quantile_shape(self):
        model = DiffusionForecaster(
            input_dim=1,
            output_dim=1,
            hidden_size=16,
            horizon=3,
            diffusion_steps=8,
            sample_steps=3,
            n_paths=12,
            quantiles=(0.1, 0.5, 0.9),
        )
        model.eval()
        out = model(torch.randn(2, 12, 1))
        assert out.shape == (2, 3, 3, 1)
        # Soft ordering is not enforced on empirical quantiles, but p10 <= p90.
        assert torch.all(out[:, :, 0, :] <= out[:, :, 2, :] + 1e-5)

    def test_sample_paths_shape(self):
        model = DiffusionForecaster(
            hidden_size=16, horizon=2, diffusion_steps=6, sample_steps=2, n_paths=5
        )
        model.eval()
        paths = model.sample_paths(torch.randn(3, 10, 1), n_paths=5, steps=2)
        assert paths.shape == (3, 5, 2, 1)

    def test_diffusion_loss_finite(self):
        model = DiffusionForecaster(hidden_size=16, diffusion_steps=6, sample_steps=2)
        loss = model.diffusion_loss(torch.randn(8, 12, 1), torch.randn(8, 1))
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_build_from_config(self):
        cfg = {
            "model": {
                "name": "diffusion_forecaster",
                "input_dim": 1,
                "output_dim": 1,
                "hidden_size": 16,
                "diffusion_steps": 6,
                "sample_steps": 2,
                "n_paths": 4,
            },
            "forecast": {"horizon": 1, "quantiles": [0.1, 0.5, 0.9]},
        }
        model = build_model(cfg, ensemble=False)
        assert isinstance(model, DiffusionForecaster)
        assert model.quantiles == (0.1, 0.5, 0.9)


class TestDiffusionTraining:
    def test_short_train_smoke(self, tmp_path):
        cfg = load_config("config/diffusion_experiment.yaml")
        cfg["training"].update(
            epochs=1,
            batch_size=16,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
            seed=0,
        )
        cfg["data"].update(n_samples=64, seq_len=16, val_split=0.25)
        cfg["model"].update(
            hidden_size=16, diffusion_steps=6, sample_steps=2, n_paths=4
        )
        trainer = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
        trainer.cfg = cfg
        trainer.store = ExperimentStore(str(tmp_path / "exp.db"))
        result = trainer.train()
        assert result["best_loss"] < float("inf")
        assert (tmp_path / "ckpt" / "best_model.pt").is_file()

    def test_compare_includes_diffusion(self, tmp_path):
        cfg = load_config("config/config.yaml")
        cfg["model"] = {
            "name": "diffusion_forecaster",
            "input_dim": 1,
            "output_dim": 1,
            "hidden_size": 16,
            "diffusion_steps": 6,
            "sample_steps": 2,
            "n_paths": 4,
            "max_seq_len": 64,
        }
        cfg["data"].update(n_samples=80, seq_len=12, val_split=0.25, seed=0)
        cfg["training"].update(
            epochs=1,
            batch_size=16,
            device="cpu",
            seed=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cfg["storage"] = {"db_path": str(tmp_path / "cmp.db")}
        report = compare_models(cfg, epochs=1)
        names = [row.name for row in report.rows]
        assert "diffusion_forecaster" in names
        assert "lstm" in names
        assert "persistence" in names
        diffusion = next(
            row for row in report.rows if row.name == "diffusion_forecaster"
        )
        assert "mae" in diffusion.metrics
