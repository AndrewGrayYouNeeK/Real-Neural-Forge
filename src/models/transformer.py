"""Transformer encoder model for time-series prediction."""

import math
from typing import cast

import torch
import torch.nn as nn

from src.forecasting.head import ForecastHead
from src.models.revin import RevIN


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
        patch_len: int = 1,
        patch_stride: int | None = None,
        use_revin: bool = False,
        residual: bool = False,
        target_index: int = 0,
    ):
        super().__init__()
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        if patch_len < 1:
            raise ValueError(f"patch_len must be positive, got {patch_len}")
        stride = patch_len if patch_stride is None else patch_stride
        if stride < 1:
            raise ValueError(f"patch_stride must be positive, got {stride}")
        if target_index < 0 or target_index >= input_dim:
            raise ValueError(
                f"target_index {target_index} is outside the {input_dim} input channels."
            )
        levels = tuple(float(level) for level in quantiles) if quantiles else None
        if (use_revin or residual) and (horizon != 1 or levels):
            raise ValueError(
                "RevIN and residual forecasts apply to one-step point models."
            )
        if residual and output_dim != 1:
            raise ValueError("residual forecasts require output_dim 1.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.horizon = horizon
        # None (not an empty tuple) marks the legacy one-step point head.
        self.quantiles = levels
        self.patch_len = patch_len
        self.patch_stride = stride
        self.use_revin = use_revin
        self.residual = residual
        self.target_index = target_index
        if use_revin:
            self.revin = RevIN(input_dim)
        self.input_projection = nn.Linear(input_dim * patch_len, d_model)
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
        if self.use_revin:
            x, stats = self.revin.normalize(x)
        else:
            stats = None
        last_target = x[:, -1, self.target_index]
        tokens = self._patch(x)
        if src_key_padding_mask is not None and (
            self.patch_len != 1 or self.patch_stride != 1
        ):
            raise ValueError("Padding masks are only supported when patch_len is 1.")
        hidden = self.input_projection(tokens)
        hidden = self.pos_encoding(hidden)
        hidden = self.transformer_encoder(
            hidden, src_key_padding_mask=src_key_padding_mask
        )
        output = self.output_projection(hidden[:, -1, :])
        if self.residual:
            output = output + last_target.unsqueeze(-1)
        if self.use_revin:
            if stats is None:
                raise RuntimeError("RevIN statistics were not recorded.")
            output = self.revin.denormalize(output, stats, self.target_index)
        return cast(torch.Tensor, output)

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        """Fold time into patches, keeping the most recent observations."""
        if self.patch_len == 1 and self.patch_stride == 1:
            return x
        length = int(x.size(1))
        if length < self.patch_len:
            raise ValueError(
                f"Sequence length {length} is shorter than patch_len {self.patch_len}."
            )
        n_patches = (length - self.patch_len) // self.patch_stride + 1
        start = length - (self.patch_len + (n_patches - 1) * self.patch_stride)
        window = x[:, start:, :]
        channels_first = window.transpose(1, 2)
        patches = channels_first.unfold(
            dimension=2, size=self.patch_len, step=self.patch_stride
        )
        batch, _channels, n_tokens, _patch = patches.shape
        return patches.permute(0, 2, 1, 3).reshape(batch, n_tokens, -1)
