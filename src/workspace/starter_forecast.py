"""Public sample forecasts from the shipped starter datasets and models.

Chat questions about weather or markets should not hit the synthetic default
series. These helpers load the catalogued starter files that ship in the image
and run the same recent-history forecast used by the dashboard.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import date, timedelta
from typing import Any

import torch

from src.dashboard_data import recent_series
from src.evaluation.tracking import default_tracker, steps_from_payload, uncertainty_for
from src.forecasting.spec import forecast_spec_from_config
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.workspace.errors import WorkspaceError
from src.workspace.starters import Starter, load_starters, starter_by_id
from src.workspace.store import SqlError

# Display units for the starter target, used in plain-language replies.
STARTER_UNITS = {
    "weather": "°C",
    "stocks": "daily log return",
    "crypto": "daily log return",
    "near-earth": "daily approaches",
    "lunar": "illumination fraction",
}

_predictors: dict[tuple[str, int], Predictor] = {}
# Deterministic lite payloads keyed by (starter_id, points, model mtime).
_lite_payloads: dict[tuple[str, int, int], dict[str, Any]] = {}


def clear_forecast_cache() -> None:
    """Drop cached starter predictors. Tests call this after swapping files."""
    _predictors.clear()
    _lite_payloads.clear()


def predictor_for_starter(starter: Starter, cfg: dict[str, Any]) -> Predictor:
    """Return a memoized Predictor for a starter checkpoint (shared with backtests)."""
    if not starter.model_path.is_file():
        raise WorkspaceError(503, "The sample model is not available right now.")
    stamp = starter.model_path.stat().st_mtime_ns
    key = (starter.id, stamp)
    cached = _predictors.get(key)
    if cached is not None:
        return cached
    try:
        built = build_model(cfg, ensemble=False)
        predictor = Predictor.from_checkpoint(built, starter.model_path, device="cpu")
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        raise WorkspaceError(
            503, "The sample model could not be loaded right now."
        ) from exc
    _predictors[key] = predictor
    return predictor


def _predictor(starter: Starter, cfg: dict[str, Any]) -> Predictor:
    return predictor_for_starter(starter, cfg)


def _highlights(starter: Starter) -> list[dict[str, Any]]:
    path = starter.dataset_path.with_name(f"{starter.id}-highlights.json")
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _attach_horizon(
    payload: dict[str, Any], predictor: Predictor, cfg: dict[str, Any]
) -> None:
    """Replace the one-step chart forecast when the sample is a multi-day demo."""
    spec = forecast_spec_from_config(cfg)
    window = payload.get("predict_window")
    if spec.horizon <= 1 or not isinstance(window, list) or not window:
        return
    batch = predictor.forecast(
        torch.tensor(window, dtype=torch.float32), horizon=spec.horizon
    )
    values = [float(item) for item in batch.point[0, :, 0].detach().cpu().tolist()]
    lower = None if batch.lower is None else batch.lower[0, :, 0].detach().cpu()
    upper = None if batch.upper is None else batch.upper[0, :, 0].detach().cpu()
    history = payload.get("history")
    last_index = 0
    if isinstance(history, list) and history and isinstance(history[-1], dict):
        index = history[-1].get("index")
        if isinstance(index, int):
            last_index = index
    steps: list[dict[str, Any]] = []
    for offset, value in enumerate(values, start=1):
        step: dict[str, Any] = {
            "step": offset,
            "index": last_index + offset,
            "value": value,
            "p10": None,
            "p50": value,
            "p90": None,
        }
        if lower is not None and upper is not None:
            step["p10"] = float(lower[offset - 1].item())
            step["p90"] = float(upper[offset - 1].item())
        steps.append(step)
    payload["forecast"] = steps
    payload["horizon"] = spec.horizon


def starter_forecast(
    starter_id: str, *, points: int = 120, lite: bool = False
) -> dict[str, Any]:
    """Forecast the next step of a shipped starter series. No account required.

    ``lite=True`` skips history-window replay and outcome tracking so agent and
    cache-refresh calls stay cheap on a single CPU worker.
    """
    starter = starter_by_id(starter_id)
    if starter is None:
        raise WorkspaceError(404, "Starter not found.")
    if not starter.dataset_path.is_file():
        raise WorkspaceError(503, "The sample dataset is not available right now.")
    if not starter.model_path.is_file():
        raise WorkspaceError(503, "The sample model is not available right now.")
    stamp = starter.model_path.stat().st_mtime_ns
    if lite:
        cached = _lite_payloads.get((starter.id, points, stamp))
        if cached is not None:
            return cached
    try:
        cfg = starter.config()
    except (OSError, ValueError) as exc:
        raise WorkspaceError(503, "The sample template is unreadable.") from exc
    data_cfg = cfg.get("data")
    if not isinstance(data_cfg, dict):
        data_cfg = {}
        cfg["data"] = data_cfg
    data_cfg["path"] = str(starter.dataset_path)
    data_cfg["source"] = "csv"
    data_cfg["name"] = starter.name
    predictor = _predictor(starter, cfg)
    try:
        payload = recent_series(
            cfg,
            predictor,
            points=points,
            include_history_predictions=not lite,
        )
    except (FileNotFoundError, ValueError, OSError, TypeError, RuntimeError) as exc:
        raise WorkspaceError(
            503, "The sample forecast could not be built right now."
        ) from exc
    _attach_horizon(payload, predictor, cfg)
    payload["starter_id"] = starter.id
    payload["name"] = starter.name
    payload["summary"] = starter.summary
    payload["description"] = starter.source
    payload["unit"] = starter.unit or STARTER_UNITS.get(starter.id)
    payload["demo"] = starter.demo
    highlights = _highlights(starter)
    if highlights:
        payload["highlights"] = highlights
    payload["checkpoint_loaded"] = True
    payload["lite"] = lite
    _attach_uncertainty(starter.id, payload)
    if not lite:
        _track_starter(starter.id, payload)
    if lite:
        _lite_payloads[(starter.id, points, stamp)] = payload
    return payload


def _attach_uncertainty(starter_id: str, payload: dict[str, Any]) -> None:
    steps = payload.get("forecast")
    if not isinstance(steps, list) or not steps:
        return
    midpoint: list[float] = []
    lows: list[float] = []
    highs: list[float] = []
    bounded = True
    for row in steps:
        if not isinstance(row, dict):
            return
        center = row.get("p50", row.get("value"))
        if isinstance(center, bool) or not isinstance(center, int | float):
            return
        midpoint.append(float(center))
        low = row.get("p10")
        high = row.get("p90")
        if (
            isinstance(low, bool)
            or isinstance(high, bool)
            or not isinstance(low, int | float)
            or not isinstance(high, int | float)
        ):
            bounded = False
            continue
        lows.append(float(low))
        highs.append(float(high))
    payload["uncertainty"] = uncertainty_for(
        starter_id,
        midpoint,
        lows if bounded else None,
        highs if bounded else None,
    )


def _track_starter(starter_id: str, payload: dict[str, Any]) -> None:
    """Score earlier sample forecasts, then store this one. Never raises."""
    try:
        history = payload.get("history")
        rows = history if isinstance(history, list) else []
        actuals: dict[str, float] = {}
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("time"), str):
                continue
            actual = row.get("actual")
            if isinstance(actual, bool) or not isinstance(actual, int | float):
                continue
            actuals[row["time"][:10]] = float(actual)
        tracker = default_tracker()
        if actuals:
            tracker.score_actuals(starter_id, actuals)
        steps = steps_from_payload(payload)
        targets = _daily_targets(rows, len(steps))
        if targets is not None:
            for step, target in zip(steps, targets, strict=True):
                step["target_time"] = target
        if steps:
            tracker.record(starter_id, steps)
    except (
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
        KeyError,
        SqlError,
        sqlite3.Error,
    ):
        # A sample forecast is still returned when the outcome log is unavailable.
        return


def _daily_targets(history: list[Any], horizon: int) -> list[str] | None:
    if horizon < 1 or len(history) < 2:
        return None
    times: list[date] = []
    for row in history[-3:]:
        if not isinstance(row, dict) or not isinstance(row.get("time"), str):
            return None
        try:
            times.append(date.fromisoformat(row["time"][:10]))
        except ValueError:
            return None
    gaps = [
        (later - earlier).days for earlier, later in zip(times, times[1:], strict=False)
    ]
    if any(gap != 1 for gap in gaps):
        return None
    return [
        (times[-1] + timedelta(days=offset)).isoformat()
        for offset in range(1, horizon + 1)
    ]


def public_starter_catalog() -> dict[str, Any]:
    """Shipped sample models. No account required. Lottery is not included."""
    return {
        "starters": [
            {
                "id": starter.id,
                "name": starter.name,
                "summary": starter.summary,
                "source": starter.source,
                "unit": starter.unit or STARTER_UNITS.get(starter.id),
                "demo": starter.demo,
            }
            for starter in load_starters()
        ]
    }
