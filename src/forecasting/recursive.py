"""Roll a one-step forecast forward by appending it to the window."""

from __future__ import annotations

import torch


def advance_window(
    window: torch.Tensor, prediction: torch.Tensor, target_index: int | None
) -> torch.Tensor:
    """Drop the oldest steps and append ``prediction``.

    ``window`` is ``(batch, time, features)`` and ``prediction`` is
    ``(batch, steps, output_dim)`` or ``(batch, output_dim)``. When the
    prediction has one value per feature it replaces the whole next step.
    Otherwise ``target_index`` selects the feature that receives the
    prediction, and every other feature is held at its last observed value.
    """
    if prediction.dim() == 2:
        prediction = prediction.unsqueeze(1)
    if window.dim() != 3 or prediction.dim() != 3:
        raise ValueError(
            "advance_window expected window (batch, time, features) and "
            f"prediction (batch, steps, output_dim), got {tuple(window.shape)} and {tuple(prediction.shape)}."
        )
    batch, time, features = window.shape
    steps = int(prediction.size(1))
    output_dim = int(prediction.size(-1))
    if prediction.size(0) != batch:
        raise ValueError("advance_window batch sizes differ.")
    if steps < 1 or steps > time:
        raise ValueError(
            f"Cannot roll forward {steps} steps inside a window of length {time}."
        )
    if features == output_dim:
        nxt = prediction
    else:
        if target_index is None:
            raise ValueError(
                "Recursive forecasting needs input_dim == output_dim, or "
                "forecast.target_feature_index so the prediction can be written "
                "back into the window."
            )
        if target_index < 0 or target_index + output_dim > features:
            raise ValueError(
                f"forecast.target_feature_index {target_index} does not fit "
                f"output_dim {output_dim} inside input_dim {features}."
            )
        last = window[:, -1, :].unsqueeze(1).expand(batch, steps, features).clone()
        last[:, :, target_index : target_index + output_dim] = prediction
        nxt = last
    return torch.cat([window[:, steps:, :], nxt], dim=1)
