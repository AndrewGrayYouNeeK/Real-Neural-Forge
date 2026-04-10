"""Multi-layer perceptron (MLP) implementation."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

from neural_forge.models.base import BaseModel


class MLP(BaseModel):
    """Fully-connected feed-forward network.

    Args:
        input_dim:   Number of input features.
        hidden_dims: Sequence of hidden layer widths.
        output_dim:  Number of output units.
        activation:  Activation class applied between hidden layers.
        dropout:     Dropout probability (0 = disabled).
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        output_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = h
        layers.append(nn.Linear(prev, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)
