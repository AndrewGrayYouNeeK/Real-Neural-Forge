"""Tests for evaluation metrics."""

import torch

from src.evaluation.metrics import mae, mse, rmse


class TestMetrics:
    def test_mse_rmse_mae(self):
        preds = torch.tensor([[1.0], [3.0]])
        targets = torch.tensor([[0.0], [2.0]])
        assert mse(preds, targets) == 1.0
        assert rmse(preds, targets) == 1.0
        assert mae(preds, targets) == 1.0
