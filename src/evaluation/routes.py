"""Public evaluation routes. Coverage is cheap; backtests are cached."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.evaluation.run import backtest_starter, starter_or_none, stored_backtest
from src.evaluation.tracking import default_tracker
from src.workspace.store import SqlError

router = APIRouter()


class BacktestBody(BaseModel):
    horizon: int = Field(default=1, ge=1, le=24)
    season_length: int = Field(default=7, ge=1, le=366)
    refresh: bool = False


def _coverage(model_id: str) -> list[dict[str, Any]]:
    try:
        return [row.as_dict() for row in default_tracker().coverage(model_id)]
    except (OSError, RuntimeError, TypeError, ValueError, SqlError):
        return []


@router.get("/evaluation/{model_id}", tags=["evaluation"])
def evaluation_summary(model_id: str) -> dict[str, Any]:
    """Stored coverage and the latest backtest, if one has been run."""
    try:
        report = stored_backtest(model_id)
    except (OSError, RuntimeError, TypeError, ValueError, SqlError) as exc:
        raise HTTPException(
            status_code=503, detail="Evaluation storage is unavailable."
        ) from exc
    if report is None and starter_or_none(model_id) is None and not _coverage(model_id):
        raise HTTPException(
            status_code=404, detail="No evaluation is stored for this model."
        )
    return {
        "model_id": model_id,
        "coverage": _coverage(model_id),
        "backtest": report,
    }


@router.post("/evaluation/{model_id}/backtest", tags=["evaluation"])
def evaluation_backtest(
    model_id: str, body: BacktestBody | None = None
) -> dict[str, Any]:
    """Run a rolling-origin backtest and cache it.

    A repeated call returns the cached report unless ``refresh`` is true.
    """
    payload = body or BacktestBody()
    starter = starter_or_none(model_id)
    if starter is None:
        raise HTTPException(
            status_code=404,
            detail="Backtests are available for shipped sample models.",
        )
    try:
        report = backtest_starter(
            starter,
            horizon=payload.horizon,
            season_length=payload.season_length,
            refresh=payload.refresh,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (OSError, RuntimeError, TypeError, SqlError) as exc:
        raise HTTPException(
            status_code=503, detail="The backtest could not be stored."
        ) from exc
    return report
