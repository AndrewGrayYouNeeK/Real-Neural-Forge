"""Transformer encoder model for time-series prediction."""

import math
from typing import cast

import torch
import torch.nn as nn

from src.forecasting.head import ForecastHead


class PositionalEncoding(nn.Module):
    """Inject positional information into token embeddings."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, self.pe)
        if x.size(1) > pe.size(1):
            raise ValueError(
                f"Sequence length {int(x.size(1))} exceeds the maximum positional "
                f"length {int(pe.size(1))}."
            )
        x = x + pe[:, : x.size(1), :]
        return cast(torch.Tensor, self.dropout(x))


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder network for time-series prediction."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        horizon: int = 1,
        quantiles: tuple[float, ...] | list[float] | None = None,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.horizon = horizon
        # None (not an empty tuple) marks the legacy one-step point head.
        self.quantiles = (
            tuple(float(level) for level in quantiles) if quantiles else None
        )
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, dropout, max_len=max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )
        # Keep nn.Linear for the one-step point model so checkpoints written
        # before forecast horizons still load into output_projection.*.
        if self.quantiles is None and horizon == 1:
            self.output_projection: nn.Module = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = ForecastHead(
                d_model, output_dim, horizon, self.quantiles or ()
            )

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        x = x[:, -1, :]
        return cast(torch.Tensor, self.output_projection(x))
