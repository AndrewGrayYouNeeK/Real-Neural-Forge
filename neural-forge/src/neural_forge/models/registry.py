"""Model registry — maps string names to model constructor factories."""

from __future__ import annotations

from typing import Callable

import torch.nn as nn

from neural_forge.models.cnn import SimpleCNN
from neural_forge.models.mlp import MLP

_REGISTRY: dict[str, Callable[..., nn.Module]] = {
    "mlp": MLP,
    "simple_cnn": SimpleCNN,
}


class ModelRegistry:
    """Central registry for model architectures."""

    @staticmethod
    def register(name: str, factory: Callable[..., nn.Module]) -> None:
        """Register a new model *factory* under *name*."""
        if name in _REGISTRY:
            raise ValueError(f"Model '{name}' is already registered.")
        _REGISTRY[name] = factory

    @staticmethod
    def build(name: str, **kwargs: object) -> nn.Module:
        """Instantiate a model by *name* with keyword arguments."""
        if name not in _REGISTRY:
            raise KeyError(f"Unknown model '{name}'. Available: {list(_REGISTRY)}")
        return _REGISTRY[name](**kwargs)

    @staticmethod
    def list_models() -> list[str]:
        """Return names of all registered models."""
        return list(_REGISTRY)
