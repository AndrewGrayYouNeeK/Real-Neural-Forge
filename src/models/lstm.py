"""Small LSTM forecaster used as a comparison model and an optional architecture."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from src.forecasting.head import ForecastHead


class LSTMForecaster(nn.Module):
    """LSTM that pools the last observed step and projects a forecast.

    ``hidden_size`` and ``num_layers`` default to a small network. A one-step
    point head is a linear layer; longer horizons and quantile levels use
    :class:`ForecastHead`.
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
        horizon: int = 1,
        quantiles: tuple[float, ...] | list[float] | None = None,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        if hidden_size < 1 or num_layers < 1:
            raise ValueError("LSTM hidden_size and num_layers must be >= 1.")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.quantiles = (
            tuple(float(level) for level in quantiles) if quantiles else None
        )
        # nn.LSTM applies dropout only between stacked layers.
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        if self.quantiles is None and horizon == 1:
            self.output_projection: nn.Module = nn.Linear(hidden_size, output_dim)
        else:
            self.output_projection = ForecastHead(
                hidden_size, output_dim, horizon, self.quantiles or ()
            )

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        encoded, _ = self.lstm(x)
        if src_key_padding_mask is None:
            hidden = encoded[:, -1, :]
        else:
            valid = (~src_key_padding_mask).to(dtype=torch.long)
            index = valid.sum(dim=1).clamp(min=1) - 1
            gather_index = index.view(-1, 1, 1).expand(-1, 1, encoded.size(-1))
            hidden = encoded.gather(1, gather_index).squeeze(1)
        return cast(torch.Tensor, self.output_projection(hidden))
