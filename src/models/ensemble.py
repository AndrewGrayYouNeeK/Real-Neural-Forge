"""Average of independently trained copies of one architecture."""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn


class SeedEnsemble(nn.Module):
    """Forward pass is the mean of ``members``."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        if len(members) < 2:
            raise ValueError("A seed ensemble needs at least two members.")
        self.members = nn.ModuleList(members)
        # nn.Module attributes are typed as Tensor | Module; these are ints.
        first = cast(Any, members[0])
        self.input_dim = int(first.input_dim)
        self.output_dim = int(first.output_dim)
        self.max_seq_len = int(first.max_seq_len)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        predictions = [
            member(x, src_key_padding_mask=src_key_padding_mask)
            for member in self.members
        ]
        return torch.stack(predictions, dim=0).mean(dim=0)
