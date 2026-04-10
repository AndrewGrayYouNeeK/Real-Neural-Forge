"""Learning-rate scheduler factory."""

from __future__ import annotations

import torch


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str = "cosine",
    **kwargs: object,
) -> torch.optim.lr_scheduler.LRScheduler:
    """Create a learning-rate scheduler.

    Args:
        optimizer: The optimiser to attach the scheduler to.
        name:      Scheduler name.  Supported values:

                   * ``"cosine"`` — :class:`CosineAnnealingLR` (requires ``T_max``)
                   * ``"step"`` — :class:`StepLR` (requires ``step_size``)
                   * ``"plateau"`` — :class:`ReduceLROnPlateau`
                   * ``"onecycle"`` — :class:`OneCycleLR` (requires ``max_lr``, ``total_steps``)

        **kwargs:  Extra keyword arguments forwarded to the scheduler constructor.

    Returns:
        Configured :class:`LRScheduler` instance.
    """
    _SCHEDULERS: dict[str, type] = {
        "cosine": torch.optim.lr_scheduler.CosineAnnealingLR,
        "step": torch.optim.lr_scheduler.StepLR,
        "plateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
        "onecycle": torch.optim.lr_scheduler.OneCycleLR,
    }

    key = name.lower()
    if key not in _SCHEDULERS:
        raise ValueError(f"Unknown scheduler '{name}'. Choose from: {list(_SCHEDULERS)}")

    return _SCHEDULERS[key](optimizer, **kwargs)
