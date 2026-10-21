"""Reversible instance normalization (RevIN).

Each window is normalized with its own mean and standard deviation, then the
network's output is mapped back with those statistics. Affine scale and shift
are inverted before the statistics are restored.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RevIN(nn.Module):
    """Per-window normalization over the time axis of ``(batch, time, channels)``."""

    def __init__(self, n_features: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if n_features < 1:
            raise ValueError(f"n_features must be positive, got {n_features}")
        self.n_features = n_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(n_features))
            self.bias = nn.Parameter(torch.zeros(n_features))

    def normalize(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Return the normalized window and the ``(mean, stdev)`` used."""
        mean = x.mean(dim=1, keepdim=True)
        variance = x.var(dim=1, keepdim=True, unbiased=False)
        stdev = torch.sqrt(variance + self.eps)
        normalized = (x - mean) / stdev
        if self.affine:
            normalized = normalized * self.weight + self.bias
        return normalized, (mean, stdev)

    def denormalize(
        self,
        y: torch.Tensor,
        stats: tuple[torch.Tensor, torch.Tensor],
        channel: int,
    ) -> torch.Tensor:
        """Map predictions back using one input channel's statistics.

        ``y`` has shape ``(batch, outputs)``. Every output uses ``channel``.
        """
        mean, stdev = stats
        if channel < 0 or channel >= self.n_features:
            raise ValueError(
                f"RevIN channel {channel} is outside 0..{self.n_features - 1}."
            )
        restored = y
        if self.affine:
            restored = (restored - self.bias[channel]) / self.weight[channel]
        center = mean[:, 0, channel].unsqueeze(-1)
        spread = stdev[:, 0, channel].unsqueeze(-1)
        return restored * spread + center
