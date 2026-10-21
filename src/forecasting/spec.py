"""Forecast settings that stay optional for existing configs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

MAX_HORIZON = 512
_STRATEGIES = {"direct", "recursive"}


@dataclass(frozen=True)
class ForecastSpec:
    """How far ahead to predict and how intervals are scored.

    ``horizon`` 1 and ``strategy`` ``direct`` with no quantiles leaves the
    one-step network, its checkpoints, and ``POST /predict`` unchanged.
    ``model_horizon`` is the width of the network head: recursive models keep
    a one-step head and roll it forward ``horizon`` times.
    """

    horizon: int = 1
    strategy: str = "direct"
    quantiles: tuple[float, ...] = ()
    season_length: int | None = None
    target_feature_index: int | None = None
    nominal_coverage: float = 0.8
    n_lags: int | None = None

    @property
    def model_horizon(self) -> int:
        return 1 if self.strategy == "recursive" else self.horizon

    def as_dict(self) -> dict[str, Any]:
        return {
            "horizon": self.horizon,
            "strategy": self.strategy,
            "quantiles": list(self.quantiles),
            "season_length": self.season_length,
            "target_feature_index": self.target_feature_index,
            "nominal_coverage": self.nominal_coverage,
            "n_lags": self.n_lags,
        }


def forecast_spec_from_config(cfg: dict[str, Any] | None) -> ForecastSpec:
    """Read ``forecast`` (and a few ``data`` fallbacks) from a config mapping."""
    if not isinstance(cfg, dict):
        raise ValueError("Config must be a mapping.")
    forecast = cfg.get("forecast", {})
    data = cfg.get("data", {})
    if forecast is None:
        forecast = {}
    if data is None:
        data = {}
    if not isinstance(forecast, dict):
        raise ValueError("forecast must be a mapping.")
    if not isinstance(data, dict):
        raise ValueError("data must be a mapping.")

    horizon = _positive_int(forecast.get("horizon", 1), "forecast.horizon")
    if horizon > MAX_HORIZON:
        raise ValueError(f"forecast.horizon must be <= {MAX_HORIZON}, got {horizon}.")
    strategy = str(forecast.get("strategy", "direct")).lower()
    if strategy not in _STRATEGIES:
        raise ValueError("forecast.strategy must be 'direct' or 'recursive'.")

    quantiles = _parse_quantiles(forecast.get("quantiles"))
    if "nominal_coverage" in forecast and forecast["nominal_coverage"] is not None:
        nominal = float(forecast["nominal_coverage"])
    elif quantiles:
        nominal = float(quantiles[-1] - quantiles[0])
    else:
        nominal = 0.8
    if not 0.0 < nominal < 1.0:
        raise ValueError("forecast.nominal_coverage must lie in (0, 1).")

    return ForecastSpec(
        horizon=horizon,
        strategy=strategy,
        quantiles=quantiles,
        season_length=_optional_positive_int(
            _override(forecast, data, "season_length"), "forecast.season_length"
        ),
        target_feature_index=_target_index(cfg, forecast, data),
        nominal_coverage=nominal,
        n_lags=_optional_positive_int(
            _override(forecast, data, "n_lags"), "forecast.n_lags"
        ),
    )


def _override(forecast: dict[str, Any], data: dict[str, Any], key: str) -> Any:
    if key in forecast:
        return forecast[key]
    return data.get(key)


def _target_index(
    cfg: dict[str, Any], forecast: dict[str, Any], data: dict[str, Any]
) -> int | None:
    if "target_feature_index" in forecast:
        return _optional_index(forecast["target_feature_index"])
    if "target_feature_index" in data:
        return _optional_index(data["target_feature_index"])
    columns = data.get("feature_columns")
    target = data.get("target_column")
    if isinstance(columns, list) and isinstance(target, str) and target in columns:
        return int(columns.index(target))
    # The synthetic series is the target, so recursive rollouts can write it back.
    if data.get("source", "synthetic") != "csv":
        return 0
    model = cfg.get("model")
    if isinstance(model, dict) and model.get("input_dim") == model.get("output_dim"):
        return None
    return None


def _parse_quantiles(raw: Any) -> tuple[float, ...]:
    if raw is None or raw == []:
        return ()
    if not isinstance(raw, list | tuple):
        raise ValueError("forecast.quantiles must be a list of probabilities.")
    if len(raw) < 2:
        raise ValueError(
            "forecast.quantiles needs at least two levels, such as [0.1, 0.9]."
        )
    quantiles = tuple(float(value) for value in raw)
    if any(not 0.0 < value < 1.0 for value in quantiles):
        raise ValueError("forecast.quantiles must lie in (0, 1).")
    if any(
        later <= earlier
        for earlier, later in zip(quantiles, quantiles[1:], strict=False)
    ):
        raise ValueError("forecast.quantiles must be strictly increasing.")
    return quantiles


def _positive_int(value: Any, name: str) -> int:
    parsed = _optional_positive_int(value, name)
    if parsed is None:
        raise ValueError(f"{name} must be an integer >= 1.")
    return parsed


def _optional_positive_int(value: Any, name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be an integer >= 1.")
    if int(value) != value:
        raise ValueError(f"{name} must be an integer >= 1.")
    parsed = int(value)
    if parsed < 1:
        raise ValueError(f"{name} must be an integer >= 1.")
    return parsed


def _optional_index(value: Any) -> int | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, int | float)
        or int(value) != value
    ):
        raise ValueError("forecast.target_feature_index must be an integer >= 0.")
    parsed = int(value)
    if parsed < 0:
        raise ValueError("forecast.target_feature_index must be an integer >= 0.")
    return parsed
