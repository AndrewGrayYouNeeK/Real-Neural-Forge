"""JEPA-style joint-embedding predictive pretraining for time series (experimental).

Mask contiguous spans of a window, encode visible context with a student
encoder, encode the full window with an EMA target encoder, and train a
predictor to match **target embeddings** (not raw values). After pretraining,
the same student encoder fine-tunes a forecast head on the supervised path.
Not the default architecture unless backtests show a transfer win.
"""

from __future__ import annotations

import copy
import math
from typing import cast

import torch
import torch.nn as nn

from src.forecasting.head import ForecastHead


class _PatchEncoder(nn.Module):
    """Tiny transformer encoder over patch tokens."""

    def __init__(
        self,
        *,
        input_dim: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        max_seq_len: int,
    ) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        pe = torch.zeros(1, max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, input_dim)
        if x.size(1) > cast(torch.Tensor, self.pe).size(1):
            raise ValueError(
                f"Sequence length {int(x.size(1))} exceeds max_seq_len "
                f"{int(cast(torch.Tensor, self.pe).size(1))}."
            )
        hidden = self.input_projection(x)
        hidden = hidden + cast(torch.Tensor, self.pe)[:, : x.size(1), :]
        hidden = self.dropout(hidden)
        return cast(torch.Tensor, self.encoder(hidden))


class TimeSeriesJEPA(nn.Module):
    """Context/target encoders + predictor for SSL, then a forecast head."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        d_model: int = 32,
        nhead: int = 4,
        num_encoder_layers: int = 1,
        dim_feedforward: int = 64,
        dropout: float = 0.0,
        max_seq_len: int = 128,
        horizon: int = 1,
        quantiles: tuple[float, ...] | list[float] | None = None,
        ema_momentum: float = 0.996,
        mask_ratio: float = 0.25,
        mask_spans: int = 1,
        predictor_hidden: int = 64,
    ) -> None:
        super().__init__()
        if d_model < 8 or d_model % nhead:
            raise ValueError(
                f"d_model must be >= 8 and divisible by nhead, got {d_model}/{nhead}."
            )
        if not 0.0 < mask_ratio < 1.0:
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}.")
        if mask_spans < 1:
            raise ValueError("mask_spans must be >= 1.")
        if not 0.0 < ema_momentum < 1.0:
            raise ValueError(f"ema_momentum must be in (0, 1), got {ema_momentum}.")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.horizon = horizon
        self.max_seq_len = max_seq_len
        self.ema_momentum = float(ema_momentum)
        self.mask_ratio = float(mask_ratio)
        self.mask_spans = int(mask_spans)
        self.quantiles = (
            tuple(float(level) for level in quantiles) if quantiles else None
        )

        self.context_encoder = _PatchEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.target_encoder = copy.deepcopy(self.context_encoder)
        for parameter in self.target_encoder.parameters():
            parameter.requires_grad_(False)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.mask_token, std=0.02)
        self.predictor = nn.Sequential(
            nn.Linear(d_model, predictor_hidden),
            nn.SiLU(),
            nn.Linear(predictor_hidden, d_model),
        )

        if self.quantiles is None and horizon == 1:
            self.output_projection: nn.Module = nn.Linear(d_model, output_dim)
        else:
            self.output_projection = ForecastHead(
                d_model, output_dim, horizon, self.quantiles or ()
            )

    @torch.no_grad()
    def update_ema(self) -> None:
        """Polyak-average target encoder toward the context encoder."""
        momentum = self.ema_momentum
        context_params = dict(self.context_encoder.named_parameters())
        for name, target_param in self.target_encoder.named_parameters():
            target_param.data.mul_(momentum).add_(
                context_params[name].data, alpha=1.0 - momentum
            )
        context_buffers = dict(self.context_encoder.named_buffers())
        for name, target_buffer in self.target_encoder.named_buffers():
            if name in context_buffers:
                target_buffer.copy_(context_buffers[name])

    def jepa_loss(self, history: torch.Tensor) -> torch.Tensor:
        """Embedding MSE on masked spans (student predicts EMA targets)."""
        if history.dim() != 3:
            raise ValueError(
                f"history must be (batch, time, features), got {tuple(history.shape)}."
            )
        batch, length, _ = history.shape
        if length < 2:
            raise ValueError("JEPA pretraining needs sequence length >= 2.")
        device = history.device
        mask = self._sample_mask(batch, length, device)
        # Zero masked positions for the context encoder (visible-only context).
        visible = history.masked_fill(mask.unsqueeze(-1), 0.0)
        context = self.context_encoder(visible)
        # Replace masked token embeddings with a learned mask token before predicting.
        mask_tokens = self.mask_token.expand(batch, length, -1)
        predictor_in = torch.where(mask.unsqueeze(-1), mask_tokens, context)
        predicted = self.predictor(predictor_in)

        with torch.no_grad():
            targets = self.target_encoder(history)

        # Average over masked positions only.
        diff = (predicted - targets) ** 2
        masked = diff[mask]
        if masked.numel() == 0:
            return cast(torch.Tensor, predicted.sum() * 0.0)
        return cast(torch.Tensor, masked.mean())

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Supervised forecast from the context encoder + head."""
        del src_key_padding_mask
        encoded = self.context_encoder(x)
        return cast(torch.Tensor, self.output_projection(encoded[:, -1, :]))

    def _sample_mask(
        self, batch: int, length: int, device: torch.device
    ) -> torch.Tensor:
        """Boolean mask True = masked (target) positions."""
        total = max(1, int(round(self.mask_ratio * length)))
        total = min(total, length - 1)
        spans = min(self.mask_spans, total)
        base = total // spans
        remainder = total - base * spans
        mask = torch.zeros(batch, length, dtype=torch.bool, device=device)
        for row in range(batch):
            lengths = [base + (1 if i < remainder else 0) for i in range(spans)]
            lengths = [span for span in lengths if span > 0]
            used = 0
            for span_len in lengths:
                max_start = length - span_len
                start = int(torch.randint(0, max_start + 1, (1,), device=device).item())
                # Prefer non-overlapping; fall back if the window is tiny.
                if mask[row, start : start + span_len].any() and used < 8:
                    used += 1
                    start = int(
                        torch.randint(0, max_start + 1, (1,), device=device).item()
                    )
                mask[row, start : start + span_len] = True
        # Guarantee at least one masked and one visible step per row.
        if not mask.any(dim=1).all():
            mask[:, 0] = True
        if mask.all(dim=1).any():
            mask[:, -1] = False
        return mask
