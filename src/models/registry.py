"""Model registry for configurable architecture selection."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from src.forecasting.spec import forecast_spec_from_config
from src.models.ensemble import SeedEnsemble
from src.models.lstm import LSTMForecaster
from src.models.transformer import TimeSeriesTransformer

_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "time_series_transformer": TimeSeriesTransformer,
    "lstm": LSTMForecaster,
}

_MODEL_KEYS = (
    "input_dim",
    "output_dim",
    "d_model",
    "nhead",
    "num_encoder_layers",
    "dim_feedforward",
    "dropout",
    "max_seq_len",
    "hidden_size",
    "num_layers",
    "patch_len",
    "patch_stride",
    "use_revin",
    "residual",
    "target_index",
)


class ModelRegistry:
    """Central registry for model architectures."""

    @staticmethod
    def register(name: str, factory: Callable[..., nn.Module]) -> None:
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = factory

    @staticmethod
    def build(name: str, **kwargs: Any) -> nn.Module:
        if name not in _REGISTRY:
            raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
        factory = _REGISTRY[name]
        return factory(**_accepted_kwargs(factory, kwargs))

    @staticmethod
    def list_models() -> list[str]:
        return list(_REGISTRY)


def build_model(cfg: dict[str, Any], *, ensemble: bool = True) -> nn.Module:
    """Build a model from a configuration dictionary.

    Forecast horizon and quantiles come from the ``forecast`` section. They
    are passed through only when they change the head, so a config without
    that section still builds the historical one-step point model.

    ``model.ensemble_size`` above 1 returns a :class:`SeedEnsemble` of identical
    architectures. Pass ``ensemble=False`` to build one member, which is what
    training does before averaging the copies.
    """
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "time_series_transformer")
    spec = forecast_spec_from_config(cfg)
    kwargs: dict[str, Any] = {
        key: model_cfg[key] for key in _MODEL_KEYS if key in model_cfg
    }
    if spec.model_horizon != 1:
        kwargs["horizon"] = spec.model_horizon
    if spec.quantiles:
        kwargs["quantiles"] = spec.quantiles
    size = int(model_cfg.get("ensemble_size", 1) or 1)
    if size < 1:
        raise ValueError(f"model.ensemble_size must be positive, got {size}")
    if not ensemble or size == 1:
        return ModelRegistry.build(name, **kwargs)
    members = [ModelRegistry.build(name, **kwargs) for _ in range(size)]
    return SeedEnsemble(members)


def _accepted_kwargs(
    factory: Callable[..., nn.Module], kwargs: dict[str, Any]
) -> dict[str, Any]:
    parameters = inspect.signature(factory).parameters
    if any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    ):
        return kwargs
    return {key: value for key, value in kwargs.items() if key in parameters}
