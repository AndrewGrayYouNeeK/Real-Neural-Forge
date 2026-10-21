"""Tests for the training utilities."""

import os
from unittest.mock import patch

import torch

from src.train import build_model, load_config, make_sine_dataset, train


class TestLoadConfig:
    def test_returns_dict(self):
        cfg = load_config("config/config.yaml")
        assert isinstance(cfg, dict)
        assert "model" in cfg
        assert "training" in cfg
        assert "inference" in cfg

    def test_model_keys_present(self):
        cfg = load_config("config/config.yaml")
        expected = {
            "input_dim",
            "output_dim",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "dim_feedforward",
            "dropout",
            "max_seq_len",
        }
        assert expected <= cfg["model"].keys()


class TestBuildModel:
    def test_returns_model_on_cpu(self):
        cfg = load_config("config/config.yaml")
        cfg["training"]["device"] = "cpu"
        model = build_model(cfg, torch.device("cpu"))
        assert isinstance(model, torch.nn.Module)
        next_param = next(model.parameters())
        assert next_param.device == torch.device("cpu")


class TestMakeSineDataset:
    def test_output_shapes(self):
        n, seq_len = 16, 32
        inputs, targets = make_sine_dataset(n_samples=n, seq_len=seq_len)
        assert inputs.shape == (n, seq_len, 1)
        assert targets.shape == (n, 1)

    def test_values_in_range(self):
        inputs, targets = make_sine_dataset(n_samples=100, seq_len=20, noise=0.0)
        # With no noise, values should be close to [-1, 1]
        assert inputs.abs().max().item() <= 1.5
        assert targets.abs().max().item() <= 1.5


class TestTrain:
    def test_training_creates_checkpoint(self, tmp_path):
        cfg = load_config("config/config.yaml")
        cfg["training"]["epochs"] = 2
        cfg["training"]["batch_size"] = 16
        cfg["training"]["device"] = "cpu"
        cfg["training"]["checkpoint_dir"] = str(tmp_path)
        cfg["training"]["log_interval"] = 1

        # Patch config file with tmp checkpoint dir
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(cfg, fh)
            tmp_config = fh.name

        try:
            train(tmp_config)
            assert (tmp_path / "best_model.pt").exists()
        finally:
            os.unlink(tmp_config)

    def test_checkpoint_loadable(self, tmp_path):
        cfg = load_config("config/config.yaml")
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 16
        cfg["training"]["device"] = "cpu"
        cfg["training"]["checkpoint_dir"] = str(tmp_path)

        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(cfg, fh)
            tmp_config = fh.name

        try:
            train(tmp_config)
            ckpt = torch.load(tmp_path / "best_model.pt", weights_only=True)
            assert "model_state_dict" in ckpt
            assert "epoch" in ckpt
        finally:
            os.unlink(tmp_config)

    @patch("torch.cuda.is_available")
    def test_training_cuda_fallback(self, mock_cuda_available, tmp_path):
        """Test CUDA fallback when CUDA is requested but not available."""
        mock_cuda_available.return_value = False

        cfg = load_config("config/config.yaml")
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 8
        cfg["training"]["device"] = "cuda"
        cfg["training"]["checkpoint_dir"] = str(tmp_path)

        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(cfg, fh)
            tmp_config = fh.name

        try:
            # Should complete without error, falling back to CPU
            train(tmp_config)
            assert (tmp_path / "best_model.pt").exists()
        finally:
            os.unlink(tmp_config)

    def test_training_with_different_batch_sizes(self, tmp_path):
        """Test training with different batch sizes."""
        cfg = load_config("config/config.yaml")
        cfg["training"]["epochs"] = 1
        cfg["training"]["batch_size"] = 4
        cfg["training"]["device"] = "cpu"
        cfg["training"]["checkpoint_dir"] = str(tmp_path)

        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as fh:
            yaml.dump(cfg, fh)
            tmp_config = fh.name

        try:
            train(tmp_config)
            assert (tmp_path / "best_model.pt").exists()
        finally:
            os.unlink(tmp_config)


class TestMakeSineDatasetEdgeCases:
    def test_large_dataset(self):
        """Test generating a large dataset."""
        n, seq_len = 1000, 50
        inputs, targets = make_sine_dataset(n_samples=n, seq_len=seq_len)
        assert inputs.shape == (n, seq_len, 1)
        assert targets.shape == (n, 1)

    def test_small_sequence(self):
        """Test generating a dataset with very small sequence length."""
        n, seq_len = 10, 2
        inputs, targets = make_sine_dataset(n_samples=n, seq_len=seq_len)
        assert inputs.shape == (n, seq_len, 1)
        assert targets.shape == (n, 1)

    def test_high_noise(self):
        """Test generating a dataset with high noise."""
        inputs, targets = make_sine_dataset(n_samples=50, seq_len=20, noise=0.5)
        assert inputs.shape == (50, 20, 1)
        assert targets.shape == (50, 1)
        # With high noise, values should be more spread out
        assert inputs.std().item() > 0.3


class TestBuildModelEdgeCases:
    def test_different_model_sizes(self):
        """Test building models with different configurations."""
        cfg = load_config("config/config.yaml")
        cfg["model"]["d_model"] = 32
        cfg["model"]["nhead"] = 2
        cfg["model"]["num_encoder_layers"] = 2

        model = build_model(cfg, torch.device("cpu"))
        assert isinstance(model, torch.nn.Module)

        # Verify the model has parameters
        param_count = sum(p.numel() for p in model.parameters())
        assert param_count > 0

    def test_multi_dimensional_io(self):
        """Test building a model with multi-dimensional input/output."""
        cfg = load_config("config/config.yaml")
        cfg["model"]["input_dim"] = 3
        cfg["model"]["output_dim"] = 2

        model = build_model(cfg, torch.device("cpu"))
        assert isinstance(model, torch.nn.Module)

        # Test forward pass
        batch, seq_len = 4, 10
        x = torch.randn(batch, seq_len, 3)
        out = model(x)
        assert out.shape == (batch, 2)


class TestTrainerFailureHandling:
    def test_failed_training_marks_experiment_failed(self, tmp_path):
        import pytest

        from src.storage.experiments import ExperimentStore
        from src.training.trainer import TimeSeriesTrainer

        trainer = TimeSeriesTrainer("config/config.yaml")
        trainer.cfg["training"]["epochs"] = 1
        trainer.cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        trainer.cfg["data"] = {"n_samples": 8, "seq_len": 8}
        trainer.store = ExperimentStore(tmp_path / "exp.db")

        with patch(
            "src.training.trainer.save_checkpoint",
            side_effect=RuntimeError("disk full"),
        ):
            with pytest.raises(RuntimeError, match="disk full"):
                trainer.train()

        (experiment,) = trainer.store.list_experiments()
        assert experiment["status"] == "failed"

    def test_metrics_cover_full_dataset(self, tmp_path):
        from src.storage.experiments import ExperimentStore
        from src.training.trainer import TimeSeriesTrainer

        trainer = TimeSeriesTrainer("config/config.yaml")
        trainer.cfg["training"]["epochs"] = 1
        trainer.cfg["training"]["batch_size"] = 4
        trainer.cfg["training"]["checkpoint_dir"] = str(tmp_path / "ckpt")
        trainer.cfg["data"] = {"n_samples": 10, "seq_len": 8}
        trainer.store = ExperimentStore(tmp_path / "exp.db")

        result = trainer.train()
        (experiment,) = trainer.store.list_experiments()
        assert experiment["status"] == "completed"
        assert abs(result["metrics"]["rmse"] ** 2 - result["metrics"]["mse"]) < 1e-5


class TestValidationSplit:
    def _cfg(self, tmp_path, val_split):
        cfg = load_config("config/config.yaml")
        cfg["training"].update(
            epochs=2,
            batch_size=16,
            device="cpu",
            checkpoint_dir=str(tmp_path),
            log_interval=1,
        )
        cfg["data"].update(n_samples=40, seq_len=8, val_split=val_split)
        cfg["model"].update(
            d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32
        )
        return cfg

    def _trainer(self, tmp_path, cfg):
        from src.storage.experiments import ExperimentStore
        from src.training.trainer import TimeSeriesTrainer

        t = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
        t.cfg = cfg
        t.store = ExperimentStore(str(tmp_path / "exp.db"))
        return t

    def test_val_loss_recorded_and_used(self, tmp_path):
        t = self._trainer(tmp_path, self._cfg(tmp_path, 0.25))
        result = t.train()
        assert all("val_loss" in h for h in result["history"])
        assert result["best_loss"] == min(h["val_loss"] for h in result["history"])

    def test_no_split_keeps_train_loss(self, tmp_path):
        t = self._trainer(tmp_path, self._cfg(tmp_path, 0.0))
        result = t.train()
        assert all("val_loss" not in h for h in result["history"])
        assert result["best_loss"] == min(h["train_loss"] for h in result["history"])

    def test_invalid_split_rejected(self, tmp_path):
        import pytest

        t = self._trainer(tmp_path, self._cfg(tmp_path, 1.0))
        with pytest.raises(ValueError, match="val_split"):
            t.train()


class TestReproducibility:
    def test_seeded_dataset_is_deterministic(self):
        a = make_sine_dataset(n_samples=8, seq_len=6, seed=3)
        b = make_sine_dataset(n_samples=8, seq_len=6, seed=3)
        c = make_sine_dataset(n_samples=8, seq_len=6, seed=4)
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
        assert not torch.equal(a[0], c[0])

    def test_seeded_training_is_reproducible(self, tmp_path):
        from src.storage.experiments import ExperimentStore
        from src.training.trainer import TimeSeriesTrainer

        losses = []
        for run in range(2):
            cfg = load_config("config/config.yaml")
            cfg["training"].update(
                epochs=2,
                batch_size=8,
                device="cpu",
                seed=7,
                checkpoint_dir=str(tmp_path / f"c{run}"),
            )
            cfg["data"].update(n_samples=24, seq_len=8, val_split=0.25, seed=7)
            cfg["model"].update(
                d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32
            )
            t = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
            t.cfg = cfg
            t.store = ExperimentStore(str(tmp_path / f"e{run}.db"))
            losses.append([h["val_loss"] for h in t.train()["history"]])
        assert losses[0] == losses[1]


class TestEarlyStoppingAndClipping:
    def _trainer(self, tmp_path, **training):
        from src.storage.experiments import ExperimentStore
        from src.training.trainer import TimeSeriesTrainer

        cfg = load_config("config/config.yaml")
        cfg["training"].update(
            epochs=30,
            batch_size=8,
            device="cpu",
            seed=0,
            checkpoint_dir=str(tmp_path / "ckpt"),
        )
        cfg["training"].update(training)
        cfg["data"].update(n_samples=24, seq_len=8, val_split=0.25, seed=0)
        cfg["model"].update(
            d_model=16, nhead=2, num_encoder_layers=1, dim_feedforward=32
        )
        t = TimeSeriesTrainer.__new__(TimeSeriesTrainer)
        t.cfg = cfg
        t.store = ExperimentStore(str(tmp_path / "exp.db"))
        return t

    def test_patience_stops_after_no_improvement(self, tmp_path):
        result = self._trainer(tmp_path, patience=2, learning_rate=0.5).train()
        history = result["history"]
        assert len(history) < 30
        losses = [h["val_loss"] for h in history]
        best_epoch = losses.index(min(losses))
        assert len(history) - 1 - best_epoch == 2

    def test_clipping_bounds_gradient_norm(self, tmp_path):
        from unittest.mock import patch

        import torch.nn.utils as nn_utils

        real = nn_utils.clip_grad_norm_
        calls: list[float] = []

        def spy(params, max_norm, *a, **k):
            calls.append(max_norm)
            return real(params, max_norm, *a, **k)

        with patch("torch.nn.utils.clip_grad_norm_", side_effect=spy):
            self._trainer(tmp_path, epochs=1, max_grad_norm=0.5).train()
        assert calls and all(c == 0.5 for c in calls)

    def test_negative_patience_rejected(self, tmp_path):
        import pytest

        with pytest.raises(ValueError, match="patience"):
            self._trainer(tmp_path, patience=-1).train()
