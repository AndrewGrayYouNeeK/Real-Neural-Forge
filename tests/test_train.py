"""Tests for the training utilities."""

import os
from pathlib import Path

import pytest
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
            "input_dim", "output_dim", "d_model", "nhead",
            "num_encoder_layers", "dim_feedforward", "dropout", "max_seq_len",
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as fh:
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

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as fh:
            yaml.dump(cfg, fh)
            tmp_config = fh.name

        try:
            train(tmp_config)
            ckpt = torch.load(tmp_path / "best_model.pt", weights_only=True)
            assert "model_state_dict" in ckpt
            assert "epoch" in ckpt
        finally:
            os.unlink(tmp_config)
