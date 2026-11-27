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

import math
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

# Cost caps: the encoder trains on the container CPU only (no Workers AI).
MAX_TRAIN_WINDOWS = 1200
TRAIN_STEPS = 240
BATCH = 128
HIDDEN = 48
LATENT = 6
DAILY_FIT_CAP = 3000  # process-wide JEPA trainings per UTC day (~30 per auto run)
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


def jepa_fits_left() -> int:
    """JEPA trainings still allowed today."""
    return max(0, DAILY_FIT_CAP - fits_today())


def _take_budget(count: int = 1) -> None:
    with _DAY_LOCK:
        day = _today()
        if _DAY_STATE["day"] != day:
            _DAY_STATE["day"] = day
            _DAY_STATE["count"] = 0
        if int(_DAY_STATE["count"]) + count > DAILY_FIT_CAP:
            raise JepaBusyError(
                "The JEPA encoder hit today's CPU cap. The PCA encoder still works."
            )
        _DAY_STATE["count"] = int(_DAY_STATE["count"]) + count


def reset_budget() -> None:
    """Tests: clear the daily counter."""
    with _DAY_LOCK:
        _DAY_STATE["day"] = ""
        _DAY_STATE["count"] = 0


def train_jepa_encoder(scaled: np.ndarray, *, seed: int = 0) -> JepaEncoder:
    """Train one JEPA context encoder on standardised windows ``(N, window)``."""
    return train_jepa_ensemble(scaled, seeds=[seed])[0]


def train_jepa_ensemble(scaled: np.ndarray, *, seeds: list[int]) -> list[JepaEncoder]:
    """Train ``len(seeds)`` independent JEPA encoders in one batched loop.

    Each member has its own seeded initialisation and batch order; the weights
    live in stacked tensors and train with ``bmm``, so three members cost
    little more than one (the loop is overhead-bound on a small CPU).
    Windows must be consecutive (row ``i + 1`` is row ``i`` shifted one step),
    which is how the world model builds them.
    """
    import torch

    scaled = np.asarray(scaled, dtype=np.float32)
    count, window = scaled.shape
    members = len(seeds)
    shift = int(max(1, min(window // 4, 7)))
    if count - shift < 16:
        raise ValueError("Series too short for the JEPA encoder.")
    if not _FIT_LOCK.acquire(timeout=_LOCK_WAIT_S):
        raise JepaBusyError("The JEPA encoder is busy. Try again in a moment.")
    try:
        _take_budget(members)
        started = time.perf_counter()
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            data = torch.from_numpy(scaled[-MAX_TRAIN_WINDOWS - shift :])
            context_x = data[:-shift]
            target_x = data[shift:]
            n = len(context_x)
            batch = min(BATCH, n)
            gens = [torch.Generator().manual_seed(int(sd)) for sd in seeds]

            def init(rows: int, cols: int) -> tuple[Any, Any]:
                bound = 1.0 / math.sqrt(rows)
                ws = [
                    (torch.rand(rows, cols, generator=g) * 2 - 1) * bound for g in gens
                ]
                bs = [(torch.rand(1, cols, generator=g) * 2 - 1) * bound for g in gens]
                return (
                    torch.stack(ws).requires_grad_(True),
                    torch.stack(bs).requires_grad_(True),
                )

            w1, b1 = init(window, HIDDEN)
            w2, b2 = init(HIDDEN, LATENT)
            p1, pb1 = init(LATENT, HIDDEN)
            p2, pb2 = init(HIDDEN, LATENT)
            tw1, tb1 = w1.detach().clone(), b1.detach().clone()
            tw2, tb2 = w2.detach().clone(), b2.detach().clone()
            params = [w1, b1, w2, b2, p1, pb1, p2, pb2]
            opt = torch.optim.Adam(params, lr=3e-3)

            def encode(x: Any, a1: Any, c1: Any, a2: Any, c2: Any) -> Any:
                hidden = torch.relu(torch.baddbmm(c1, x, a1))
                return torch.tanh(torch.baddbmm(c2, hidden, a2))

            momentum = 0.98
            losses: list[Any] = []
            for _ in range(TRAIN_STEPS):
                idx = torch.stack(
                    [torch.randint(0, n, (batch,), generator=g) for g in gens]
                )
                ctx = encode(context_x[idx], w1, b1, w2, b2)
                with torch.no_grad():
                    tgt = encode(target_x[idx], tw1, tb1, tw2, tb2)
                pred = torch.baddbmm(pb2, torch.relu(torch.baddbmm(pb1, ctx, p1)), p2)
                pred_loss = ((pred - tgt) ** 2).mean(dim=(1, 2))
                std = torch.sqrt(ctx.var(dim=1) + 1e-4)
                var_loss = torch.relu(0.5 - std).mean(dim=1)
                centred = ctx - ctx.mean(dim=1, keepdim=True)
                cov = centred.transpose(1, 2) @ centred / max(batch - 1, 1)
                eye = torch.eye(LATENT)
                off = cov * (1 - eye)
                cov_loss = (off**2).sum(dim=(1, 2)) / LATENT
                loss = (pred_loss + 1.0 * var_loss + 0.05 * cov_loss).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    tw1.mul_(momentum).add_(w1, alpha=1 - momentum)
                    tb1.mul_(momentum).add_(b1, alpha=1 - momentum)
                    tw2.mul_(momentum).add_(w2, alpha=1 - momentum)
                    tb2.mul_(momentum).add_(b2, alpha=1 - momentum)
                losses.append(pred_loss.detach().numpy().copy())
            with torch.no_grad():
                full = data.unsqueeze(0).expand(members, -1, -1).contiguous()
                feats = encode(full, w1, b1, w2, b2).numpy()
            loss_arr = np.stack(losses)  # (steps, members)
            head = max(1, len(loss_arr) // 10)
            seconds = time.perf_counter() - started
            out: list[JepaEncoder] = []
            for m in range(members):
                out.append(
                    JepaEncoder(
                        w1=w1[m].detach().numpy().astype(np.float64).copy(),
                        b1=b1[m, 0].detach().numpy().astype(np.float64).copy(),
                        w2=w2[m].detach().numpy().astype(np.float64).copy(),
                        b2=b2[m, 0].detach().numpy().astype(np.float64).copy(),
                        shift=shift,
                        steps=TRAIN_STEPS,
                        loss_start=float(np.mean(loss_arr[:head, m])),
                        loss_end=float(np.mean(loss_arr[-head:, m])),
                        feature_std=float(np.mean(np.std(feats[m], axis=0))),
                        seconds=seconds / members,
                    )
                )
            return out
        finally:
            torch.set_num_threads(prev_threads)
    finally:
        _FIT_LOCK.release()
