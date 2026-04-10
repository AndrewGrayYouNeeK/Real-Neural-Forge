"""Unit tests for utility modules."""

import tempfile
from pathlib import Path

import pytest
import torch

from neural_forge.models.mlp import MLP
from neural_forge.training.optimizer import build_optimizer
from neural_forge.utils.checkpoint import load_checkpoint, save_checkpoint
from neural_forge.utils.config import load_config
from neural_forge.utils.seed import set_seed


class TestSetSeed:
    def test_reproducibility(self):
        set_seed(42)
        t1 = torch.randn(5)
        set_seed(42)
        t2 = torch.randn(5)
        assert torch.allclose(t1, t2)


class TestCheckpoint:
    def test_save_and_load(self):
        model = MLP(4, [8], 2)
        opt = build_optimizer(model, lr=1e-3)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "ckpt.pt"
            save_checkpoint(model, opt, epoch=5, path=path)
            assert path.exists()

            model2 = MLP(4, [8], 2)
            ckpt = load_checkpoint(path, model=model2)
            assert ckpt["epoch"] == 5
            # Weights should be identical
            for p1, p2 in zip(model.parameters(), model2.parameters()):
                assert torch.allclose(p1, p2)


class TestLoadConfig:
    def test_load_yaml(self, tmp_path):
        yaml_content = "training:\n  lr: 0.001\n  epochs: 10\n"
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)

        cfg = load_config(cfg_file)
        assert cfg.training.lr == pytest.approx(0.001)
        assert cfg.training.epochs == 10

    def test_overrides(self, tmp_path):
        yaml_content = "training:\n  lr: 0.001\n  epochs: 10\n"
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)

        cfg = load_config(cfg_file, overrides=["training.epochs=50"])
        assert cfg.training.epochs == 50
