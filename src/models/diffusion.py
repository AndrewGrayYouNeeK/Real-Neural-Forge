"""Small conditional 1D denoising diffusion forecaster (experimental).

``f_theta(x_t, t, c)`` denoises a noisy future path ``x_t`` at diffusion step
``t``, conditioned on recent history ``c``. Training is offline / CPU-small;
inference uses few DDIM-style steps and can draw many futures for p10/p50/p90.
Not the default architecture unless backtests show it beats the transformer.
"""

from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn as nn


def _cosine_betas(steps: int, *, s: float = 0.008) -> torch.Tensor:
    """Nichol & Dhariwal cosine schedule, length ``steps``."""
    if steps < 1:
        raise ValueError(f"diffusion steps must be >= 1, got {steps}.")
    t = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alphas_cum = torch.cos(((t / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cum = alphas_cum / alphas_cum[0]
    betas = 1.0 - (alphas_cum[1:] / alphas_cum[:-1])
    return betas.clamp(min=1e-5, max=0.999).float()


class SinusoidalTimeEmbedding(nn.Module):
    """Map an integer diffusion step to a dense vector."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim < 2 or dim % 2:
            raise ValueError(f"time embedding dim must be even and >= 2, got {dim}.")
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=torch.float32)
            / half
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionForecaster(nn.Module):
    """Conditional denoising diffusion for a short future path.

    History ``c`` is encoded with a tiny GRU. The denoiser is an MLP that
    sees the flat noisy future, a time embedding, and the context vector, then
    predicts the noise (ε-prediction).
    """

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        hidden_size: int = 32,
        horizon: int = 1,
        diffusion_steps: int = 20,
        sample_steps: int = 5,
        n_paths: int = 16,
        quantiles: tuple[float, ...] | list[float] | None = None,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_size < 8:
            raise ValueError("diffusion hidden_size must be >= 8.")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}.")
        if diffusion_steps < 2:
            raise ValueError("diffusion_steps must be >= 2.")
        if sample_steps < 1 or sample_steps > diffusion_steps:
            raise ValueError(
                f"sample_steps must be in [1, {diffusion_steps}], got {sample_steps}."
            )
        if n_paths < 1:
            raise ValueError("n_paths must be >= 1.")
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_size = hidden_size
        self.horizon = horizon
        self.diffusion_steps = diffusion_steps
        self.sample_steps = sample_steps
        self.n_paths = n_paths
        self.max_seq_len = max_seq_len
        self.quantiles = (
            tuple(float(level) for level in quantiles) if quantiles else None
        )
        self.future_dim = horizon * output_dim

        self.context_encoder = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        time_dim = hidden_size
        self.time_embed = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.denoise = nn.Sequential(
            nn.Linear(self.future_dim + hidden_size + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, self.future_dim),
        )

        betas = _cosine_betas(diffusion_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )

    def encode_context(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() != 3:
            raise ValueError(
                f"history must be (batch, time, features), got {tuple(history.shape)}."
            )
        if history.size(1) > self.max_seq_len:
            raise ValueError(
                f"Sequence length {int(history.size(1))} exceeds max_seq_len "
                f"{self.max_seq_len}."
            )
        _, hidden = self.context_encoder(history)
        return cast(torch.Tensor, hidden[-1])

    def predict_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """ε̂ = f_θ(x_t, t, c). ``x_t`` is flat ``(B, horizon * output_dim)``."""
        time_feat = self.time_mlp(self.time_embed(t))
        return cast(
            torch.Tensor,
            self.denoise(torch.cat([x_t, context, time_feat], dim=-1)),
        )

    def diffusion_loss(
        self, history: torch.Tensor, future: torch.Tensor
    ) -> torch.Tensor:
        """Noise-prediction MSE on a random diffusion step (training only)."""
        clean = self._flat_future(future)
        batch = clean.size(0)
        device = clean.device
        t = torch.randint(0, self.diffusion_steps, (batch,), device=device)
        noise = torch.randn_like(clean)
        x_t = self._q_sample(clean, t, noise)
        context = self.encode_context(history)
        predicted = self.predict_noise(x_t, t, context)
        return cast(torch.Tensor, nn.functional.mse_loss(predicted, noise))

    @torch.no_grad()
    def sample_paths(
        self,
        history: torch.Tensor,
        *,
        n_paths: int | None = None,
        steps: int | None = None,
    ) -> torch.Tensor:
        """Draw ``n_paths`` futures → ``(batch, n_paths, horizon, output_dim)``."""
        paths = int(self.n_paths if n_paths is None else n_paths)
        denoise_steps = int(self.sample_steps if steps is None else steps)
        if paths < 1:
            raise ValueError("n_paths must be >= 1.")
        if denoise_steps < 1 or denoise_steps > self.diffusion_steps:
            raise ValueError(
                f"steps must be in [1, {self.diffusion_steps}], got {denoise_steps}."
            )
        context = self.encode_context(history)
        batch = history.size(0)
        device = history.device
        # Expand context for path dimension.
        context = (
            context.unsqueeze(1)
            .expand(batch, paths, context.size(-1))
            .reshape(batch * paths, context.size(-1))
        )
        x = torch.randn(batch * paths, self.future_dim, device=device)
        timeline = torch.linspace(
            self.diffusion_steps - 1, 0, denoise_steps, device=device
        ).long()
        alphas_cumprod = cast(torch.Tensor, self.alphas_cumprod)
        for index, step in enumerate(timeline):
            t = torch.full((batch * paths,), int(step.item()), device=device)
            eps = self.predict_noise(x, t, context)
            alpha_bar = alphas_cumprod[step]
            x0 = (x - torch.sqrt(1.0 - alpha_bar) * eps) / torch.sqrt(alpha_bar).clamp(
                min=1e-8
            )
            if index + 1 < len(timeline):
                next_step = timeline[index + 1]
                alpha_bar_next = alphas_cumprod[next_step]
                x = (
                    torch.sqrt(alpha_bar_next) * x0
                    + torch.sqrt(1.0 - alpha_bar_next) * eps
                )
            else:
                x = x0
        shaped = x.view(batch, paths, self.horizon, self.output_dim)
        return shaped

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Point or quantile forecast from history windows.

        Uses a small number of cached diffusion steps. When ``quantiles`` is
        set, returns ``(B, H, Q, D)`` (or ``(B, Q, D)`` for horizon 1) from
        empirical path percentiles; otherwise returns the path median.
        """
        del src_key_padding_mask
        paths = self.sample_paths(x, n_paths=self.n_paths, steps=self.sample_steps)
        if self.quantiles:
            levels = torch.tensor(
                self.quantiles, device=paths.device, dtype=paths.dtype
            )
            # paths: (B, P, H, D) → quantiles over P → (B, H, Q, D)
            quant = torch.quantile(paths, levels, dim=1).permute(1, 2, 0, 3)
            if self.horizon == 1:
                return cast(torch.Tensor, quant[:, 0, :, :])
            return cast(torch.Tensor, quant)
        median = paths.median(dim=1).values
        if self.horizon == 1:
            return cast(torch.Tensor, median[:, 0, :])
        return cast(torch.Tensor, median)

    def _flat_future(self, future: torch.Tensor) -> torch.Tensor:
        if future.dim() == 2:
            # (B, output_dim) one-step or (B, horizon) single-feature
            if future.size(-1) == self.future_dim:
                return future
            if self.horizon == 1 and future.size(-1) == self.output_dim:
                return future
            if self.output_dim == 1 and future.size(-1) == self.horizon:
                return future
        if future.dim() == 3:
            # (B, horizon, output_dim)
            if future.size(1) == self.horizon and future.size(2) == self.output_dim:
                return future.reshape(future.size(0), self.future_dim)
        raise ValueError(
            f"future shape {tuple(future.shape)} is incompatible with "
            f"horizon={self.horizon}, output_dim={self.output_dim}."
        )

    def _q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        sqrt_alpha = cast(torch.Tensor, self.sqrt_alphas_cumprod)[t].unsqueeze(-1)
        sqrt_one_minus = cast(torch.Tensor, self.sqrt_one_minus_alphas_cumprod)[
            t
        ].unsqueeze(-1)
        return sqrt_alpha * x0 + sqrt_one_minus * noise
