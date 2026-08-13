"""Model registry for configurable architecture selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.nn as nn

from src.models.transformer import TimeSeriesTransformer

_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "time_series_transformer": TimeSeriesTransformer,
}


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
            raise KeyError(
                f"Unknown model '{name}'. Available: {list(_REGISTRY)}"
            )
        return _REGISTRY[name](**kwargs)

    @staticmethod
    def list_models() -> list[str]:
        return list(_REGISTRY)


def build_model(cfg: dict[str, Any]) -> nn.Module:
    """Build a model from a configuration dictionary."""
    model_cfg = cfg["model"]
    name = model_cfg.get("name", "time_series_transformer")
    kwargs = {
        key: model_cfg[key]
        for key in (
            "input_dim",
            "output_dim",
            "d_model",
            "nhead",
            "num_encoder_layers",
            "dim_feedforward",
            "dropout",
            "max_seq_len",
        )
        if key in model_cfg
    }
    return ModelRegistry.build(name, **kwargs)
