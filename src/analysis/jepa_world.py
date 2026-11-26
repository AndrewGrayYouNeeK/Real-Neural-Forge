"""Self-supervised JEPA encoder for the Phase 5 world model.

The PCA encoder in :mod:`src.analysis.world_model` is linear. This module
learns a small **non-linear** latent state with a JEPA-style objective
(joint-embedding predictive architecture):

* a context encoder ``f`` maps a standardised window of the recent past to a
  latent vector;
* a target encoder ``g`` is an exponential moving average (EMA) of ``f`` and
  receives no gradients;
* a predictor ``p`` must match ``g(window shifted ahead by k steps)`` from
  ``f(window)``. It learns to predict the *embedding* of the future, never the
  raw future values.

A variance + covariance penalty (VICReg-style) keeps the embedding from
collapsing to a constant. Training is tiny and deterministic (fixed seed, a
few hundred Adam steps) so it fits on the shared CPU container. After
training, weights are copied into NumPy so rollouts do not touch torch.

The learned features are bounded (``tanh``) so the simulator stays stable even
though the encoder is non-linear. Whether they actually help is decided per
series by the rolling-origin backtest, never assumed.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, cast

import numpy as np

# Cost caps: the encoder trains on the container CPU only (no Workers AI).
MAX_TRAIN_WINDOWS = 1200
TRAIN_STEPS = 240
BATCH = 128
HIDDEN = 48
LATENT = 6
DAILY_FIT_CAP = 600  # process-wide JEPA fits per UTC day
_LOCK_WAIT_S = 25.0

_FIT_LOCK = threading.Semaphore(1)
_DAY_STATE: dict[str, Any] = {"day": "", "count": 0}
_DAY_LOCK = threading.Lock()


class JepaBusyError(RuntimeError):
    """Raised when the CPU budget for JEPA fits is used up or busy."""


@dataclass
class JepaEncoder:
    """NumPy copy of the trained context encoder ``f``."""

    w1: np.ndarray  # (window, hidden)
    b1: np.ndarray
    w2: np.ndarray  # (hidden, latent)
    b2: np.ndarray
    shift: int
    steps: int
    loss_start: float
    loss_end: float
    feature_std: float
    seconds: float

    @property
    def latent_dim(self) -> int:
        return int(self.w2.shape[1])

    def encode(self, scaled: np.ndarray) -> np.ndarray:
        hidden = np.maximum(scaled @ self.w1 + self.b1, 0.0)
        return np.asarray(np.tanh(hidden @ self.w2 + self.b2))

    def describe(self) -> dict[str, Any]:
        return {
            "objective": "JEPA (predict EMA-target embedding of the window "
            f"{self.shift} steps ahead) + variance/covariance regularizer",
            "latent_dim": self.latent_dim,
            "train_steps": self.steps,
            "ssl_loss_start": round(self.loss_start, 4),
            "ssl_loss_end": round(self.loss_end, 4),
            "feature_std": round(self.feature_std, 4),
            "train_seconds": round(self.seconds, 3),
        }


def _today() -> str:
    return time.strftime("%Y-%m-%d", time.gmtime())


def fits_today() -> int:
    with _DAY_LOCK:
        if _DAY_STATE["day"] != _today():
            return 0
        return int(_DAY_STATE["count"])


def _take_budget() -> None:
    with _DAY_LOCK:
        day = _today()
        if _DAY_STATE["day"] != day:
            _DAY_STATE["day"] = day
            _DAY_STATE["count"] = 0
        if int(_DAY_STATE["count"]) >= DAILY_FIT_CAP:
            raise JepaBusyError(
                "The JEPA encoder hit today's CPU cap. The PCA encoder still works."
            )
        _DAY_STATE["count"] = int(_DAY_STATE["count"]) + 1


def reset_budget() -> None:
    """Tests: clear the daily counter."""
    with _DAY_LOCK:
        _DAY_STATE["day"] = ""
        _DAY_STATE["count"] = 0


def train_jepa_encoder(scaled: np.ndarray, *, seed: int = 0) -> JepaEncoder:
    """Train the JEPA context encoder on standardised windows ``(N, window)``.

    Windows must be consecutive (row ``i + 1`` is row ``i`` shifted by one
    step), which is how the world model builds them.
    """
    import torch
    from torch import nn

    scaled = np.asarray(scaled, dtype=np.float32)
    count, window = scaled.shape
    shift = int(max(1, min(window // 4, 7)))
    if count - shift < 16:
        raise ValueError("Series too short for the JEPA encoder.")
    if not _FIT_LOCK.acquire(timeout=_LOCK_WAIT_S):
        raise JepaBusyError("The JEPA encoder is busy. Try again in a moment.")
    try:
        _take_budget()
        started = time.perf_counter()
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            gen = torch.Generator().manual_seed(int(seed))
            torch.manual_seed(int(seed))
            data = torch.from_numpy(scaled[-MAX_TRAIN_WINDOWS - shift :])
            context_x = data[:-shift]
            target_x = data[shift:]
            n = len(context_x)

            encoder = nn.Sequential(
                nn.Linear(window, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT), nn.Tanh()
            )
            target = nn.Sequential(
                nn.Linear(window, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT), nn.Tanh()
            )
            target.load_state_dict(encoder.state_dict())
            for param in target.parameters():
                param.requires_grad_(False)
            predictor = nn.Sequential(
                nn.Linear(LATENT, HIDDEN), nn.ReLU(), nn.Linear(HIDDEN, LATENT)
            )
            opt = torch.optim.Adam(
                list(encoder.parameters()) + list(predictor.parameters()), lr=3e-3
            )
            momentum = 0.98
            losses: list[float] = []
            for _ in range(TRAIN_STEPS):
                idx = torch.randint(0, n, (min(BATCH, n),), generator=gen)
                ctx = encoder(context_x[idx])
                with torch.no_grad():
                    tgt = target(target_x[idx])
                pred_loss = ((predictor(ctx) - tgt) ** 2).mean()
                std = torch.sqrt(ctx.var(dim=0) + 1e-4)
                var_loss = torch.relu(0.5 - std).mean()
                centred = ctx - ctx.mean(dim=0)
                cov = centred.T @ centred / max(len(ctx) - 1, 1)
                off = cov - torch.diag(torch.diag(cov))
                cov_loss = (off**2).sum() / LATENT
                loss = pred_loss + 1.0 * var_loss + 0.05 * cov_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    for t_param, s_param in zip(
                        target.parameters(), encoder.parameters(), strict=True
                    ):
                        t_param.mul_(momentum).add_(s_param, alpha=1 - momentum)
                losses.append(float(pred_loss.detach()))
            with torch.no_grad():
                feats = encoder(data).numpy()
            lin1 = cast(nn.Linear, encoder[0])
            lin2 = cast(nn.Linear, encoder[2])
            head = max(1, len(losses) // 10)
            return JepaEncoder(
                w1=lin1.weight.detach().numpy().T.astype(np.float64).copy(),
                b1=lin1.bias.detach().numpy().astype(np.float64).copy(),
                w2=lin2.weight.detach().numpy().T.astype(np.float64).copy(),
                b2=lin2.bias.detach().numpy().astype(np.float64).copy(),
                shift=shift,
                steps=TRAIN_STEPS,
                loss_start=float(np.mean(losses[:head])),
                loss_end=float(np.mean(losses[-head:])),
                feature_std=float(np.mean(np.std(feats, axis=0))),
                seconds=time.perf_counter() - started,
            )
        finally:
            torch.set_num_threads(prev_threads)
    finally:
        _FIT_LOCK.release()
