"""Public sample forecasts from the shipped starter datasets and models.

Chat questions about weather or markets should not hit the synthetic default
series. These helpers load the catalogued starter files that ship in the image
and run the same recent-history forecast used by the dashboard.
"""

from __future__ import annotations

from typing import Any

from src.dashboard_data import recent_series
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.workspace.errors import WorkspaceError
from src.workspace.starters import Starter, load_starters, starter_by_id

# Display units for the starter target, used in plain-language replies.
STARTER_UNITS = {
    "weather": "°C",
    "stocks": "daily log return",
    "crypto": "daily log return",
    "near-earth": "daily approaches",
    "lunar": "illumination fraction",
}

_predictors: dict[tuple[str, int], Predictor] = {}


def clear_forecast_cache() -> None:
    """Drop cached starter predictors. Tests call this after swapping files."""
    _predictors.clear()


def _predictor(starter: Starter, cfg: dict[str, Any]) -> Predictor:
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


def starter_forecast(starter_id: str, *, points: int = 120) -> dict[str, Any]:
    """Forecast the next step of a shipped starter series. No account required."""
    starter = starter_by_id(starter_id)
    if starter is None:
        raise WorkspaceError(404, "Starter not found.")
    if not starter.dataset_path.is_file():
        raise WorkspaceError(503, "The sample dataset is not available right now.")
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
        payload = recent_series(cfg, predictor, points=points)
    except (FileNotFoundError, ValueError, OSError, TypeError, RuntimeError) as exc:
        raise WorkspaceError(
            503, "The sample forecast could not be built right now."
        ) from exc
    payload["starter_id"] = starter.id
    payload["name"] = starter.name
    payload["summary"] = starter.summary
    payload["description"] = starter.source
    payload["unit"] = STARTER_UNITS.get(starter.id)
    payload["checkpoint_loaded"] = True
    return payload


def public_starter_catalog() -> dict[str, Any]:
    """Shipped sample models. No account required. Lottery is not included."""
    return {
        "starters": [
            {
                "id": starter.id,
                "name": starter.name,
                "summary": starter.summary,
                "source": starter.source,
                "unit": STARTER_UNITS.get(starter.id),
            }
            for starter in load_starters()
        ]
    }
