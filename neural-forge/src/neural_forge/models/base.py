"""Abstract base class that all Neural Forge models must implement."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn


class BaseModel(ABC, nn.Module):
    """Base class for all Neural Forge models.

    Sub-classes must implement :meth:`forward`.  Optional hooks
    ``on_train_start`` / ``on_train_end`` may be overridden.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the forward pass."""

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save model weights to *path*."""
        torch.save(self.state_dict(), path)

    def load(self, path: str | Path, *, map_location: str | torch.device = "cpu") -> None:
        """Load model weights from *path*."""
        state = torch.load(path, map_location=map_location)
        self.load_state_dict(state)

    # ------------------------------------------------------------------
    # Optional training lifecycle hooks
    # ------------------------------------------------------------------

    def on_train_start(self) -> None:  # noqa: B027
        """Called once before the first training epoch."""

    def on_train_end(self) -> None:  # noqa: B027
        """Called once after the last training epoch."""
