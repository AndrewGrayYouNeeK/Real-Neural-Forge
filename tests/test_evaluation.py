"""Tests for evaluation metrics."""

import pytest
import torch

from src.evaluation.metrics import accuracy, mae, mse, rmse


class TestMetrics:
    def test_mse_rmse_mae(self):
        preds = torch.tensor([[1.0], [3.0]])
        targets = torch.tensor([[0.0], [2.0]])
        assert mse(preds, targets) == 1.0
        assert rmse(preds, targets) == 1.0
        assert mae(preds, targets) == 1.0


class TestAccuracy:
    def test_class_scores_use_argmax(self):
        scores = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
        assert accuracy(scores, torch.tensor([0, 1, 1])) == pytest.approx(2 / 3)

    def test_column_labels_not_argmaxed(self):
        # Regression: (N, 1) labels were argmax'd to all zeros.
        preds = torch.tensor([[1], [0], [1], [1]])
        assert accuracy(preds, torch.tensor([1, 0, 1, 0])) == 0.75

    def test_mismatched_layouts_do_not_broadcast(self):
        # Regression: (N,) vs (N, 1) broadcast to an (N, N) comparison.
        preds = torch.tensor([0, 1, 2])
        targets = torch.tensor([[0], [1], [2]])
        assert accuracy(preds, targets) == 1.0
        assert accuracy(targets, preds) == 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            accuracy(torch.tensor([0, 1]), torch.tensor([0, 1, 1]))

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            accuracy(torch.tensor([]), torch.tensor([]))
