"""Inference predictor — load a checkpoint and run predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


class Predictor:
    """Wraps a trained model for single-sample or batch inference.

    Args:
        model:        Trained :class:`nn.Module`.
        device:       Device to run inference on.
        postprocess:  Optional callable applied to raw model outputs.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device = "cpu",
        postprocess: Any = None,
    ) -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.postprocess = postprocess

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        model: nn.Module,
        checkpoint_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        **kwargs: Any,
    ) -> "Predictor":
        """Restore weights from *checkpoint_path* and wrap in a :class:`Predictor`."""
        ckpt = torch.load(checkpoint_path, map_location=device)
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict)
        return cls(model, device=device, **kwargs)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run inference on a single sample or batch.

        Args:
            x: Input tensor of shape ``(…, *feature_dims)``.
               A leading batch dimension is added automatically if the tensor
               has the same number of dimensions as the model's expected input.

        Returns:
            Output tensor (post-processed if a postprocess function was supplied).
        """
        x = x.to(self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        logits = self.model(x)
        if self.postprocess is not None:
            return self.postprocess(logits)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        logits = self.predict(x)
        return torch.softmax(logits, dim=-1)

    @torch.no_grad()
    def predict_class(self, x: torch.Tensor) -> torch.Tensor:
        """Return the argmax class index."""
        return self.predict_proba(x).argmax(dim=-1)
