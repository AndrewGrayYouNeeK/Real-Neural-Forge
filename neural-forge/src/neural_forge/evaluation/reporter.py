"""Evaluation reporter — collects metrics and writes a summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from neural_forge.evaluation.metrics import accuracy, f1_score_macro
from neural_forge.utils.logger import get_logger

logger = get_logger(__name__)


class EvaluationReporter:
    """Collect predictions and targets, then compute and report metrics.

    Example::

        reporter = EvaluationReporter(num_classes=10)
        for x, y in test_loader:
            logits = model(x)
            reporter.update(logits, y)
        results = reporter.compute()
        reporter.save("reports/eval.json")
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes
        self._all_preds: list[torch.Tensor] = []
        self._all_targets: list[torch.Tensor] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """Accumulate a batch of logits and targets."""
        self._all_preds.append(logits.detach().cpu())
        self._all_targets.append(targets.detach().cpu())

    def compute(self) -> dict[str, Any]:
        """Compute all metrics over accumulated data.

        Returns:
            Dictionary of metric name → value.
        """
        if not self._all_preds:
            raise RuntimeError("No data accumulated. Call update() first.")

        preds = torch.cat(self._all_preds)
        targets = torch.cat(self._all_targets)

        results: dict[str, Any] = {
            "accuracy": accuracy(preds, targets),
            "f1_macro": f1_score_macro(preds, targets, self.num_classes),
            "num_samples": len(targets),
        }

        logger.info("Evaluation results: %s", results)
        return results

    def save(self, path: str | Path) -> None:
        """Save computed metrics as JSON to *path*."""
        results = self.compute()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(results, f, indent=2)
        logger.info("Saved evaluation report to %s", path)

    def reset(self) -> None:
        """Clear accumulated predictions and targets."""
        self._all_preds.clear()
        self._all_targets.clear()
