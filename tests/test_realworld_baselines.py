"""Holdout indexes and the three simple forecasts."""

import math

import numpy as np

from src.realworld.baselines import (
    mean_forecast,
    persistence,
    regression_scores,
    seasonal_naive,
    skill_score,
    target_positions,
)
from src.realworld.charts import svg_histogram


def test_target_positions_match_the_trainer_formula():
    n_rows, seq_len, val_split = 100, 8, 0.2
    n_windows = n_rows - seq_len
    n_val = int(n_windows * val_split)
    n_train, targets = target_positions(n_rows, seq_len, val_split)
    assert n_train == n_windows - n_val
    assert len(targets) == n_val
    assert int(targets[0]) == n_train + seq_len
    assert int(targets[-1]) == n_rows - 1


def test_persistence_on_a_unit_ramp_has_error_one():
    values = np.arange(12, dtype=float)
    _n_train, targets = target_positions(len(values), seq_len=4, val_split=0.25)
    predicted = persistence(values, targets)
    assert np.allclose(predicted, values[targets] - 1)
    scores = regression_scores(values[targets], predicted)
    assert scores["mae"] == 1
    assert scores["rmse"] == 1


def test_seasonal_naive_is_exact_on_a_repeating_series():
    values = np.tile(np.array([1.0, 2.0, 3.0]), 6)
    _n_train, targets = target_positions(len(values), seq_len=3, val_split=0.5)
    predicted = seasonal_naive(values, targets, period=3)
    assert np.allclose(predicted, values[targets])
    assert regression_scores(values[targets], predicted)["mae"] == 0


def test_seasonal_naive_falls_back_when_the_period_is_too_long():
    values = np.arange(8, dtype=float)
    targets = np.array([2, 6])
    predicted = seasonal_naive(values, targets, period=5)
    assert predicted[0] == values[1]
    assert predicted[1] == values[1]


def test_mean_forecast_uses_only_pre_holdout_history():
    values = np.array([1.0, 3.0, 100.0, 100.0])
    predicted = mean_forecast(values, np.array([2, 3]))
    assert np.allclose(predicted, [2.0, 2.0])


def test_skill_score_is_positive_when_the_model_is_closer():
    assert skill_score(1.0, 2.0) == 0.5
    assert skill_score(2.0, 0.0) is None


def test_histogram_svg_is_well_formed():
    svg = svg_histogram([1, 2, 2, 8], "gaps", "days", exponential_mean=3)
    assert svg.startswith("<svg")
    assert svg.endswith("</svg>")
    assert "gaps" in svg
    assert math.isfinite(3)
