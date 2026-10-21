"""Tests for persistence, seasonal naive, and lag regression."""

import torch

from src.evaluation.metrics import mse
from src.evaluation.references import (
    LagLinearBaseline,
    PersistenceBaseline,
    SeasonalNaiveBaseline,
)


class TestPersistence:
    def test_repeats_the_last_target(self):
        inputs = torch.tensor([[[1.0], [2.0], [3.0]]])
        pred = PersistenceBaseline(horizon=3, target_index=0, output_dim=1).predict(
            inputs
        )
        assert torch.equal(pred, torch.tensor([[[3.0], [3.0], [3.0]]]))


class TestSeasonalNaive:
    def test_reads_the_previous_season(self):
        history = torch.tensor([0.0, 1.0, 2.0, 0.0, 1.0, 2.0]).view(1, 6, 1)
        pred = SeasonalNaiveBaseline(
            season_length=3, horizon=4, target_index=0, output_dim=1
        ).predict(history)
        expected = torch.tensor([[[0.0], [1.0], [2.0], [0.0]]])
        assert torch.equal(pred, expected)


class TestLagLinear:
    def test_recovers_a_linear_function_of_lags(self):
        inputs = torch.randn(40, 4, 1)
        targets = (1.5 * inputs[:, 0, 0] - 0.25 * inputs[:, -1, 0]).unsqueeze(-1)
        model = LagLinearBaseline(horizon=1, target_index=0, output_dim=1)
        model.fit(inputs[:30], targets[:30])
        pred = model.predict(inputs[30:])
        assert mse(pred.squeeze(1), targets[30:]) < 1e-6

    def test_multi_step_shape(self):
        inputs = torch.randn(20, 5, 1)
        targets = torch.randn(20, 3, 1)
        model = LagLinearBaseline(horizon=3, target_index=0, output_dim=1, n_lags=2)
        pred = model.fit(inputs, targets).predict(inputs[:4])
        assert pred.shape == (4, 3, 1)
