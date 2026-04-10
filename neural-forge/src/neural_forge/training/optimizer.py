"""Optimiser factory."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    name: str = "adamw",
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    **kwargs: object,
) -> torch.optim.Optimizer:
    """Create an optimiser for *model* parameters.

    Args:
        model:        Model whose parameters will be optimised.
        name:         Optimiser name — ``"adam"``, ``"adamw"``, or ``"sgd"``.
        lr:           Learning rate.
        weight_decay: L2 regularisation coefficient.
        **kwargs:     Additional keyword arguments forwarded to the optimiser.

    Returns:
        Configured :class:`torch.optim.Optimizer` instance.
    """
    _OPTIMIZERS: dict[str, type[torch.optim.Optimizer]] = {
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
        "sgd": torch.optim.SGD,
    }

    key = name.lower()
    if key not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimiser '{name}'. Choose from: {list(_OPTIMIZERS)}")

    return _OPTIMIZERS[key](model.parameters(), lr=lr, weight_decay=weight_decay, **kwargs)
