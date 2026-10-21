"""Windowed forecast references scored on the same samples as the neural models.

Calendar persistence and seasonal naive for a dated CSV live in
``src.evaluation.baselines``. These classes score multi-step windows directly.
"""

from __future__ import annotations

import torch


def target_history(
    inputs: torch.Tensor, target_index: int | None, output_dim: int
) -> torch.Tensor:
    """Slice the target feature out of ``(samples, time, features)`` inputs.

    When the input vector is already the target, the whole window is returned.
    """
    if inputs.dim() != 3:
        raise ValueError(
            f"Expected inputs (samples, time, features), got {tuple(inputs.shape)}."
        )
    features = int(inputs.size(-1))
    if features == output_dim:
        return inputs
    if target_index is None:
        raise ValueError(
            "Baselines need forecast.target_feature_index when the target is not the whole input."
        )
    if target_index < 0 or target_index + output_dim > features:
        raise ValueError(
            f"forecast.target_feature_index {target_index} does not fit "
            f"output_dim {output_dim} inside input_dim {features}."
        )
    return inputs[:, :, target_index : target_index + output_dim]


class PersistenceBaseline:
    """Repeat the last observed target value for every lead time."""

    def __init__(self, horizon: int, target_index: int | None, output_dim: int) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        self.horizon = horizon
        self.target_index = target_index
        self.output_dim = output_dim

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> PersistenceBaseline:
        del inputs, targets
        return self

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        history = target_history(inputs, self.target_index, self.output_dim)
        last = history[:, -1, :]
        return last.unsqueeze(1).expand(-1, self.horizon, -1).contiguous()


class SeasonalNaiveBaseline:
    """Repeat the last season: ``y(t + h) = y(t + h - season)``."""

    def __init__(
        self,
        season_length: int,
        horizon: int,
        target_index: int | None,
        output_dim: int,
    ) -> None:
        if season_length < 1:
            raise ValueError(f"season_length must be >= 1, got {season_length}.")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        self.season_length = season_length
        self.horizon = horizon
        self.target_index = target_index
        self.output_dim = output_dim

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> SeasonalNaiveBaseline:
        del inputs, targets
        return self

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        history = target_history(inputs, self.target_index, self.output_dim)
        time = int(history.size(1))
        if self.season_length > time:
            raise ValueError(
                f"season_length {self.season_length} is longer than the input window ({time})."
            )
        steps = []
        for lead in range(1, self.horizon + 1):
            position = (time - 1) + lead - self.season_length
            while position >= time:
                position -= self.season_length
            if position < 0:
                raise ValueError(
                    f"season_length {self.season_length} does not fall inside the input window."
                )
            steps.append(history[:, position, :])
        return torch.stack(steps, dim=1)


class LagLinearBaseline:
    """Ordinary least squares of the forecast on target lags, with an intercept."""

    def __init__(
        self,
        horizon: int,
        target_index: int | None,
        output_dim: int,
        n_lags: int | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        if n_lags is not None and n_lags < 1:
            raise ValueError(f"n_lags must be >= 1, got {n_lags}.")
        self.horizon = horizon
        self.target_index = target_index
        self.output_dim = output_dim
        self.n_lags = n_lags
        self._coef: torch.Tensor | None = None

    def fit(self, inputs: torch.Tensor, targets: torch.Tensor) -> LagLinearBaseline:
        design, _ = self._design(inputs)
        observed = targets.reshape(targets.size(0), -1).to(dtype=torch.float64)
        if design.size(0) != observed.size(0):
            raise ValueError(
                "Lag regression inputs and targets have different batch sizes."
            )
        solution = torch.linalg.lstsq(design, observed).solution
        if not torch.isfinite(solution).all():
            raise ValueError("Lag regression did not produce a finite solution.")
        self._coef = solution
        return self

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        if self._coef is None:
            raise RuntimeError("LagLinearBaseline.predict called before fit.")
        design, _ = self._design(inputs)
        flat = design @ self._coef.to(device=design.device)
        return flat.reshape(inputs.size(0), self.horizon, self.output_dim).to(
            dtype=inputs.dtype
        )

    def _design(self, inputs: torch.Tensor) -> tuple[torch.Tensor, int]:
        history = target_history(inputs, self.target_index, self.output_dim)
        if self.n_lags is not None:
            if self.n_lags > history.size(1):
                raise ValueError(
                    f"n_lags {self.n_lags} is longer than the input window ({int(history.size(1))})."
                )
            history = history[:, -self.n_lags :, :]
        flat = history.reshape(history.size(0), -1).to(dtype=torch.float64)
        bias = torch.ones(flat.size(0), 1, dtype=torch.float64, device=flat.device)
        return torch.cat([flat, bias], dim=1), int(history.size(1))
