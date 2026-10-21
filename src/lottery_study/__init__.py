"""Compare external datasets with Powerball draws.

The study is intentionally self-contained. Dataset fetchers hook in by
exporting ``LOTTERY_STUDY_HOOKS`` or by writing a CSV under ``data/``; see
:mod:`src.lottery_study.registry`.
"""

from src.lottery_study.closecall import close_call_scorecard
from src.lottery_study.run import run_study

__all__ = ["close_call_scorecard", "run_study"]
