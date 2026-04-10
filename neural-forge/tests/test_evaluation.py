"""Unit tests for evaluation metrics and reporter."""

import pytest
import torch

from neural_forge.evaluation.metrics import accuracy, auroc, f1_score_macro
from neural_forge.evaluation.reporter import EvaluationReporter


class TestAccuracy:
    def test_perfect(self):
        preds = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        assert accuracy(preds, targets) == pytest.approx(1.0)

    def test_zero(self):
        preds = torch.tensor([0, 0, 0])
        targets = torch.tensor([1, 2, 1])
        assert accuracy(preds, targets) == pytest.approx(0.0)

    def test_logits_input(self):
        logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        targets = torch.tensor([1, 0])
        assert accuracy(logits, targets) == pytest.approx(1.0)


class TestF1ScoreMacro:
    def test_perfect(self):
        preds = torch.tensor([0, 1, 2])
        targets = torch.tensor([0, 1, 2])
        assert f1_score_macro(preds, targets, 3) == pytest.approx(1.0)

    def test_all_wrong(self):
        preds = torch.tensor([1, 2, 0])
        targets = torch.tensor([0, 1, 2])
        score = f1_score_macro(preds, targets, 3)
        assert score == pytest.approx(0.0)


class TestAUROC:
    def test_perfect(self):
        scores = torch.tensor([0.9, 0.8, 0.2, 0.1])
        targets = torch.tensor([1, 1, 0, 0])
        assert auroc(scores, targets) == pytest.approx(1.0)

    def test_random(self):
        torch.manual_seed(0)
        scores = torch.rand(100)
        targets = torch.randint(0, 2, (100,))
        score = auroc(scores, targets)
        # A random classifier should be near 0.5
        assert 0.3 < score < 0.7


class TestEvaluationReporter:
    def test_compute_accuracy(self):
        reporter = EvaluationReporter(num_classes=3)
        logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        targets = torch.tensor([0, 1])
        reporter.update(logits, targets)
        results = reporter.compute()
        assert results["accuracy"] == pytest.approx(1.0)
        assert results["num_samples"] == 2

    def test_reset_clears_state(self):
        reporter = EvaluationReporter(num_classes=2)
        reporter.update(torch.randn(4, 2), torch.randint(0, 2, (4,)))
        reporter.reset()
        with pytest.raises(RuntimeError):
            reporter.compute()
