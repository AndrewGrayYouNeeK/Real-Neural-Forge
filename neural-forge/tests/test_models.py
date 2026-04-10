"""Unit tests for model architectures."""

import pytest
import torch

from neural_forge.models.mlp import MLP
from neural_forge.models.cnn import SimpleCNN
from neural_forge.models.registry import ModelRegistry


class TestMLP:
    def test_output_shape(self):
        model = MLP(input_dim=16, hidden_dims=[32, 32], output_dim=5)
        x = torch.randn(8, 16)
        out = model(x)
        assert out.shape == (8, 5)

    def test_single_sample(self):
        model = MLP(input_dim=4, hidden_dims=[8], output_dim=3)
        x = torch.randn(1, 4)
        out = model(x)
        assert out.shape == (1, 3)

    def test_dropout_applied(self):
        model = MLP(input_dim=8, hidden_dims=[16], output_dim=2, dropout=0.5)
        # In eval mode outputs should be deterministic
        model.eval()
        x = torch.randn(4, 8)
        out1 = model(x)
        out2 = model(x)
        assert torch.allclose(out1, out2)

    def test_no_hidden_layers(self):
        model = MLP(input_dim=4, hidden_dims=[], output_dim=2)
        x = torch.randn(3, 4)
        out = model(x)
        assert out.shape == (3, 2)


class TestSimpleCNN:
    def test_output_shape(self):
        model = SimpleCNN(in_channels=3, num_classes=10)
        x = torch.randn(4, 3, 32, 32)
        out = model(x)
        assert out.shape == (4, 10)

    def test_grayscale(self):
        model = SimpleCNN(in_channels=1, num_classes=5)
        x = torch.randn(2, 1, 28, 28)
        out = model(x)
        assert out.shape == (2, 5)


class TestModelRegistry:
    def test_list_models(self):
        models = ModelRegistry.list_models()
        assert "mlp" in models
        assert "simple_cnn" in models

    def test_build_mlp(self):
        model = ModelRegistry.build(
            "mlp", input_dim=8, hidden_dims=[16], output_dim=4
        )
        x = torch.randn(2, 8)
        assert model(x).shape == (2, 4)

    def test_build_unknown_raises(self):
        with pytest.raises(KeyError):
            ModelRegistry.build("nonexistent_model")

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError):
            ModelRegistry.register("mlp", MLP)
