"""Real-world dataset parsers, baselines, and conflict-interval analysis.

Training still goes through :class:`src.training.trainer.TimeSeriesTrainer`.
This package only adds the dataset-specific pieces: file parsers, forecast
baselines in original units, and the UCDP interval analysis.
"""

from src.realworld.baselines import (
    regression_scores,
    seasonal_naive,
    target_positions,
)

__all__ = [
    "regression_scores",
    "seasonal_naive",
    "target_positions",
]
