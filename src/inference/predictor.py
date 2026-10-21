"""Inference predictor for time-series models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
import torch.nn as nn

from src.data.scaling import StandardScaler
from src.forecasting.intervals import apply_qhat
from src.forecasting.output import point_forecast, quantile_bounds
from src.forecasting.recursive import advance_window
from src.forecasting.spec import ForecastSpec, forecast_spec_from_config


@dataclass(frozen=True)
class ForecastBatch:
    """A batch of point forecasts and, when available, interval bounds.

    ``point``, ``lower``, and ``upper`` are ``(batch, horizon, output_dim)``
    in the series' original units. ``lower`` and ``upper`` are ``None`` when
    this checkpoint cannot cover every requested step.
    """

    point: torch.Tensor
    lower: torch.Tensor | None
    upper: torch.Tensor | None
    quantiles: tuple[float, ...]
    nominal_coverage: float | None
    validation_coverage: float | None
    interval_method: str | None
    strategy: str
    horizon: int


class Predictor:
    """Wraps a trained model for batch inference."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        scaler: StandardScaler | None = None,
        spec: ForecastSpec | None = None,
        forecast_meta: dict[str, object] | None = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.scaler = scaler
        self.spec = spec or ForecastSpec()
        meta = forecast_meta or {}
        qhat = meta.get("conformal_qhat")
        self.conformal_qhat = (
            tuple(float(value) for value in qhat)
            if isinstance(qhat, list | tuple) and qhat
            else None
        )
        method = meta.get("interval_method")
        self.interval_method = method if isinstance(method, str) else None
        self.validation_coverage = _optional_float(meta.get("validation_coverage"))
        coverage_horizon = meta.get("coverage_horizon")
        self.coverage_horizon = (
            int(coverage_horizon)
            if isinstance(coverage_horizon, int | float)
            and not isinstance(coverage_horizon, bool)
            else None
        )
        self.head_horizon = int(getattr(model, "horizon", 1) or 1)
        head_quantiles = getattr(model, "quantiles", None)
        self.head_quantiles = (
            tuple(float(level) for level in head_quantiles) if head_quantiles else ()
        )

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        spec: ForecastSpec | None = None,
    ) -> Predictor:
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            scaler_state = ckpt.get("scaler")
            forecast_meta = ckpt.get("forecast")
        else:
            state_dict = ckpt
            scaler_state = None
            forecast_meta = None
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise RuntimeError(
                "Checkpoint does not match this model. The architecture, "
                "forecast.horizon, and forecast.quantiles must match the "
                f"run that saved the checkpoint. {exc}"
            ) from exc
        scaler = (
            StandardScaler.from_state_dict(scaler_state)
            if isinstance(scaler_state, dict)
            else None
        )
        meta = forecast_meta if isinstance(forecast_meta, dict) else None
        resolved = _spec_from_checkpoint(meta, spec)
        return cls(
            model,
            device=device,
            scaler=scaler,
            spec=resolved,
            forecast_meta=meta,
        )

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Point or head output in original units.

        One-step point models return ``(batch, output_dim)``, the same tensor
        this method has always returned. Multi-step and quantile heads return
        their full output; use :meth:`forecast` for a point forecast and bounds.
        """
        raw = self._forward(self._batch(x))
        return self._to_original(raw)

    @torch.no_grad()
    def forecast(self, x: torch.Tensor, horizon: int | None = None) -> ForecastBatch:
        """Forecast ``horizon`` steps. ``None`` uses the configured horizon."""
        steps = self.spec.horizon if horizon is None else horizon
        if steps < 1:
            raise ValueError(f"horizon must be >= 1, got {steps}.")
        window = self._batch(x)
        if self.spec.strategy == "direct" and steps <= self.head_horizon:
            point, lower, upper = self._direct_chunk(window, steps)
        else:
            point, lower, upper = self._roll(window, steps)
        lower, upper, method = self._serve_intervals(point, lower, upper, steps)
        coverage = (
            self.validation_coverage
            if self.coverage_horizon == steps and self.validation_coverage is not None
            else None
        )
        rolled = self.spec.strategy == "direct" and steps > self.head_horizon
        return ForecastBatch(
            point=point,
            lower=lower,
            upper=upper,
            quantiles=self.head_quantiles,
            nominal_coverage=self.spec.nominal_coverage if lower is not None else None,
            validation_coverage=coverage,
            interval_method=method,
            strategy="recursive" if rolled else self.spec.strategy,
            horizon=steps,
        )

    def _batch(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if x.dim() != 3:
            raise ValueError(
                f"Expected inputs (batch, time, features) or (time, features), got {tuple(x.shape)}."
            )
        return x

    def _forward(self, window: torch.Tensor) -> torch.Tensor:
        scaled = (
            self.scaler.transform_inputs(window) if self.scaler is not None else window
        )
        return cast(torch.Tensor, self.model(scaled))

    def _to_original(self, output: torch.Tensor) -> torch.Tensor:
        if self.scaler is None:
            return output
        return self.scaler.inverse_targets(output)

    def _direct_chunk(
        self, window: torch.Tensor, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        raw = self._to_original(self._forward(window))
        point = point_forecast(
            raw, horizon=self.head_horizon, quantiles=self.head_quantiles
        )
        point = _as_steps(point)[:, :steps]
        lower, upper = self._raw_bounds(raw, steps)
        return point, lower, upper

    def _raw_bounds(
        self, raw: torch.Tensor, steps: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if len(self.head_quantiles) < 2:
            return None, None
        lower, upper = quantile_bounds(raw, horizon=self.head_horizon)
        return _as_steps(lower)[:, :steps], _as_steps(upper)[:, :steps]

    def _roll(
        self, window: torch.Tensor, steps: int
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        points: list[torch.Tensor] = []
        lowers: list[torch.Tensor] = []
        uppers: list[torch.Tensor] = []
        bounds_ok = True
        produced = 0
        # A direct head can emit several steps before the window has to move.
        chunk = self.head_horizon if self.spec.strategy == "direct" else 1
        while produced < steps:
            take = min(chunk, steps - produced)
            point, lower, upper = self._direct_chunk(window, take)
            points.append(point)
            if lower is None or upper is None:
                bounds_ok = False
            else:
                lowers.append(lower)
                uppers.append(upper)
            produced += take
            if produced >= steps:
                break
            window = advance_window(window, point, self.spec.target_feature_index)
        point_out = torch.cat(points, dim=1)
        if bounds_ok and lowers:
            return point_out, torch.cat(lowers, dim=1), torch.cat(uppers, dim=1)
        return point_out, None, None

    def _serve_intervals(
        self,
        point: torch.Tensor,
        lower: torch.Tensor | None,
        upper: torch.Tensor | None,
        steps: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, str | None]:
        qhat = self.conformal_qhat
        if (
            self.interval_method == "residual"
            and qhat is not None
            and len(qhat) >= steps
        ):
            served_lower, served_upper = apply_qhat(point, point, qhat[:steps])
            return served_lower, served_upper, "residual"
        if lower is None or upper is None:
            return None, None, None
        if self.interval_method == "cqr" and qhat is not None and len(qhat) >= steps:
            served_lower, served_upper = apply_qhat(lower, upper, qhat[:steps])
            return served_lower, served_upper, "cqr"
        if qhat is None:
            return lower, upper, "quantile"
        return None, None, None


def _optional_float(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _as_steps(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(1)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"Cannot broadcast forecast shape {tuple(tensor.shape)}.")


def _spec_from_checkpoint(
    meta: dict[str, object] | None, fallback: ForecastSpec | None
) -> ForecastSpec:
    base = fallback or ForecastSpec()
    if not meta:
        return base
    # Reuse the config parser so checkpoint metadata obeys the same limits.
    return forecast_spec_from_config({"forecast": {**base.as_dict(), **meta}})
