"""Tests for JEPA-style SSL pretrain + forecast fine-tune."""

from __future__ import annotations

import torch

from src.evaluation.compare import compare_models
from src.models.jepa import TimeSeriesJEPA
from src.models.registry import ModelRegistry, build_model
from src.storage.experiments import ExperimentStore
from src.training.trainer import TimeSeriesTrainer
from src.utils.config import load_config


class TestTimeSeriesJEPA:
    def test_registered(self):
        assert "jepa_forecaster" in ModelRegistry.list_models()

    def test_forward_point_shape(self):
        model = TimeSeriesJEPA(
            input_dim=1,
            output_dim=1,
            d_model=16,
            nhead=4,
            num_encoder_layers=1,
            dim_feedforward=32,
            max_seq_len=64,
            horizon=1,
        )
        model.eval()
        out = model(torch.randn(4, 16, 1))
        assert out.shape == (4, 1)

    def test_forward_quantile_shape(self):
        model = TimeSeriesJEPA(
            d_model=16,
            nhead=4,
            dim_feedforward=32,
            horizon=3,
            quantiles=(0.1, 0.5, 0.9),
            max_seq_len=64,
        )
        model.eval()
        out = model(torch.randn(2, 12, 1))
        assert out.shape == (2, 3, 3, 1)

    def test_jepa_loss_finite(self):
        model = TimeSeriesJEPA(d_model=16, nhead=4, dim_feedforward=32, max_seq_len=64)
        loss = model.jepa_loss(torch.randn(8, 16, 1))
        assert torch.isfinite(loss)
        assert loss.ndim == 0

    def test_ema_updates_target(self):
        model = TimeSeriesJEPA(d_model=16, nhead=4, dim_feedforward=32, max_seq_len=64)
        before = {
            name: param.detach().clone()
            for name, param in model.target_encoder.named_parameters()
        }
        # Move student away from the target copy, then EMA should pull target.
        with torch.no_grad():
            for param in model.context_encoder.parameters():
                param.add_(0.1)
        model.update_ema()
        moved = False
        for name, param in model.target_encoder.named_parameters():
            if not torch.allclose(param, before[name]):
                moved = True
                break
        assert moved

    def test_build_from_config(self):
        cfg = {
            "model": {
                "name": "jepa_forecaster",
                "input_dim": 1,
                "output_dim": 1,
                "d_model": 16,
                "nhead": 4,
                "num_encoder_layers": 1,
                "dim_feedforward": 32,
                "mask_ratio": 0.3,
                "ema_momentum": 0.99,
            },
            "forecast": {"horizon": 1},
        }
        model = build_model(cfg, ensemble=False)
        assert isinstance(model, TimeSeriesJEPA)
        assert model.mask_ratio == 0.3
        assert model.ema_momentum == 0.99


class TestJEPATraining:
    def test_short_pretrain_and_finetune(self, tmp_path):
        cfg = load_config("config/jepa_experiment.yaml")
        cfg["training"].update(
            jepa_pretrain_epochs=1,
            epochs=1,
            batch_size=16,
            checkpoint_dir=str(tmp_path / "ckpt"),
            device="cpu",
            seed=0,
        )
        cfg["data"].update(n_samples=64, seq_len=16, val_split=0.25)
        cfg["model"].update(d_model=16, dim_feedforward=32, max_seq_len=64)
        trainer = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
        trainer.cfg = cfg
        trainer.store = ExperimentStore(str(tmp_path / "exp.db"))
        result = trainer.train()
        assert result["best_loss"] < float("inf")
        assert (tmp_path / "ckpt" / "best_model.pt").is_file()

    def test_compare_includes_jepa(self, tmp_path):
        cfg = load_config("config/config.yaml")
        cfg["model"] = {
            "name": "jepa_forecaster",
            "input_dim": 1,
            "output_dim": 1,
            "d_model": 16,
            "nhead": 4,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "max_seq_len": 64,
            "mask_ratio": 0.25,
        }
        cfg["data"].update(n_samples=80, seq_len=12, val_split=0.25, seed=0)
        cfg["training"].update(
            jepa_pretrain_epochs=1,
            epochs=1,
            batch_size=16,
            device="cpu",
            seed=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cfg["storage"] = {"db_path": str(tmp_path / "cmp.db")}
        report = compare_models(cfg, epochs=1)
        names = [row.name for row in report.rows]
        assert "jepa_forecaster" in names
        assert "lstm" in names
        jepa = next(row for row in report.rows if row.name == "jepa_forecaster")
        assert "mae" in jepa.metrics
