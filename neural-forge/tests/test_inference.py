"""Unit tests for the inference Predictor."""

import torch

from neural_forge.inference.predictor import Predictor
from neural_forge.models.mlp import MLP


class TestPredictor:
    def _predictor(self, input_dim=8, num_classes=5):
        model = MLP(input_dim, [16], num_classes)
        return Predictor(model, device="cpu")

    def test_predict_batch_shape(self):
        predictor = self._predictor()
        x = torch.randn(4, 8)
        out = predictor.predict(x)
        assert out.shape == (4, 5)

    def test_predict_single_sample_auto_unsqueeze(self):
        predictor = self._predictor()
        x = torch.randn(8)
        out = predictor.predict(x)
        assert out.shape == (1, 5)

    def test_predict_proba_sums_to_one(self):
        predictor = self._predictor()
        x = torch.randn(3, 8)
        proba = predictor.predict_proba(x)
        assert proba.shape == (3, 5)
        sums = proba.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(3), atol=1e-5)

    def test_predict_class_valid_indices(self):
        predictor = self._predictor(num_classes=5)
        x = torch.randn(6, 8)
        classes = predictor.predict_class(x)
        assert classes.shape == (6,)
        assert all(0 <= c.item() < 5 for c in classes)
