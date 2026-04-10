"""Evaluation metrics."""

from __future__ import annotations

import torch


def accuracy(preds: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy.

    Args:
        preds:   Logits or class indices of shape ``(N,)`` or ``(N, C)``.
        targets: Ground-truth class indices of shape ``(N,)``.

    Returns:
        Accuracy in ``[0, 1]``.
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def f1_score_macro(
    preds: torch.Tensor,
    targets: torch.Tensor,
    num_classes: int,
) -> float:
    """Macro-averaged F1 score.

    Args:
        preds:       Logits or class indices of shape ``(N,)`` or ``(N, C)``.
        targets:     Ground-truth class indices of shape ``(N,)``.
        num_classes: Total number of classes.

    Returns:
        Macro-F1 in ``[0, 1]``.
    """
    if preds.dim() > 1:
        preds = preds.argmax(dim=-1)

    f1s: list[float] = []
    for c in range(num_classes):
        tp = ((preds == c) & (targets == c)).sum().float()
        fp = ((preds == c) & (targets != c)).sum().float()
        fn = ((preds != c) & (targets == c)).sum().float()
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom).item() if denom > 0 else 0.0)

    return sum(f1s) / num_classes


def auroc(scores: torch.Tensor, targets: torch.Tensor) -> float:
    """Binary AUROC (Area Under the ROC Curve).

    Args:
        scores:  Positive-class probability scores of shape ``(N,)``.
        targets: Binary ground-truth labels of shape ``(N,)``.

    Returns:
        AUROC in ``[0, 1]``.
    """
    scores = scores.float()
    targets = targets.float()

    # Sort by descending score
    order = torch.argsort(scores, descending=True)
    targets_sorted = targets[order]

    n_pos = targets.sum().item()
    n_neg = len(targets) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    tp_cumsum = targets_sorted.cumsum(0)
    fp_cumsum = (1 - targets_sorted).cumsum(0)

    tpr = tp_cumsum / n_pos
    fpr = fp_cumsum / n_neg

    # Prepend origin
    tpr = torch.cat([torch.tensor([0.0]), tpr])
    fpr = torch.cat([torch.tensor([0.0]), fpr])

    # Trapezoidal integration
    auc = torch.trapezoid(tpr, fpr).item()
    return float(auc)
