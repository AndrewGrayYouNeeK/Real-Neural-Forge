"""Tests for model registry."""

import torch

from src.models.registry import ModelRegistry, build_model


class TestModelRegistry:
    def test_list_models(self):
        models = ModelRegistry.list_models()
        assert "time_series_transformer" in models

    def test_build_from_config(self):
        cfg = {
            "model": {
                "name": "time_series_transformer",
                "input_dim": 1,
                "output_dim": 1,
                "d_model": 16,
                "nhead": 2,
                "num_encoder_layers": 1,
                "dim_feedforward": 32,
                "dropout": 0.0,
                "max_seq_len": 64,
            }
        }
        model = build_model(cfg)
        x = torch.randn(2, 10, 1)
        out = model(x)
        assert out.shape == (2, 1)
