"""FastAPI REST endpoint for time-series inference."""

import logging
from contextlib import asynccontextmanager
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.common import (
    apply_checkpoint_config,
    build_model,
    checkpoint_path_from_cfg,
    load_config,
    resolve_device,
)
from src.data import apply_scaler
from src.model import TimeSeriesTransformer

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


def load_model(config_path: str = "config/config.yaml") -> None:
    """Load (or lazily initialise) the model into the global state dict."""
    cfg = load_config(config_path)
    i_cfg = cfg["inference"]
    device = resolve_device(i_cfg.get("device", "cpu"))
    ckpt_path = checkpoint_path_from_cfg(cfg)
    checkpoint_loaded = False
    scaler = None
    metrics: dict[str, Any] = {}
    checkpoint = None

    if ckpt_path.exists():
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
        cfg = apply_checkpoint_config(cfg, checkpoint)
        scaler = checkpoint.get("scaler")
        checkpoint_loaded = True
        metrics = {
            "epoch": checkpoint.get("epoch"),
            "val_mse": checkpoint.get("val_mse", checkpoint.get("loss")),
            "val_mae": checkpoint.get("val_mae"),
            "source": checkpoint.get("source"),
        }
        logger.info("Loaded checkpoint from %s", ckpt_path)
    else:
        logger.warning(
            "Checkpoint not found at '%s'. Using untrained model.", ckpt_path
        )

    model = build_model(cfg, device)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _state["model"] = model
    _state["device"] = device
    _state["config"] = cfg
    _state["checkpoint_loaded"] = checkpoint_loaded
    _state["require_checkpoint"] = bool(i_cfg.get("require_checkpoint", False))
    _state["scaler"] = scaler
    _state["metrics"] = metrics


def _forward_once(sequence: list[list[float]]) -> list[float]:
    model: TimeSeriesTransformer = _state["model"]
    device: torch.device = _state["device"]
    scaler = _state.get("scaler")
    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    if scaler:
        x = apply_scaler(x, scaler)
    with torch.no_grad():
        output = model(x)
        if scaler:
            output = apply_scaler(output, scaler, inverse=True)
    return output.squeeze(0).tolist()


def run_predict(sequence: list[list[float]], horizon: int = 1) -> list[float]:
    """Run one or more autoregressive inference steps."""
    model: TimeSeriesTransformer | None = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if _state.get("require_checkpoint") and not _state.get("checkpoint_loaded"):
        raise HTTPException(
            status_code=503,
            detail="Trained checkpoint is required but was not loaded.",
        )

    cfg: dict = _state["config"]
    expected_input_dim: int = cfg["model"]["input_dim"]
    output_dim: int = cfg["model"]["output_dim"]
    max_seq_len: int = cfg["model"]["max_seq_len"]
    max_horizon = int(cfg.get("inference", {}).get("max_horizon", 64))

    if not sequence:
        raise HTTPException(status_code=422, detail="sequence must not be empty.")
    for index, step in enumerate(sequence):
        if len(step) != expected_input_dim:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Each time step must have {expected_input_dim} feature(s); "
                    f"step {index} has {len(step)}."
                ),
            )
    if len(sequence) > max_seq_len:
        raise HTTPException(
            status_code=422,
            detail=(
                f"sequence length {len(sequence)} exceeds max_seq_len {max_seq_len}."
            ),
        )
    if horizon < 1 or horizon > max_horizon:
        raise HTTPException(
            status_code=422,
            detail=f"horizon must be between 1 and {max_horizon}.",
        )
    if horizon > 1 and expected_input_dim != output_dim:
        raise HTTPException(
            status_code=422,
            detail="Multi-step forecast requires input_dim == output_dim.",
        )

    current = [list(step) for step in sequence]
    preds: list[float] = []
    for _ in range(horizon):
        step_pred = _forward_once(current)
        preds.extend(step_pred)
        current = current[1:] + [step_pred]
    return preds


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):  # noqa: ARG001
    load_model()
    yield
    _state.clear()


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Neural Forge",
    description="Production-ready transformer pipeline for time-series prediction.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input payload for the /predict endpoint."""

    sequence: list[list[float]] = Field(
        ...,
        description=(
            "Time-series input as a 2-D list of shape [seq_len, input_dim]. "
            "Each inner list is one time step."
        ),
        examples=[[[0.0], [0.1], [0.2]]],
    )
    horizon: int = Field(
        1,
        ge=1,
        description="Number of future steps to roll out (1 keeps the original API).",
    )


class PredictResponse(BaseModel):
    """Output payload for the /predict endpoint."""

    prediction: list[float] = Field(
        ...,
        description="Model output of shape [horizon * output_dim].",
    )


class HealthResponse(BaseModel):
    """Liveness plus whether a trained checkpoint is actually loaded."""

    status: str
    ready: bool
    checkpoint_loaded: bool
    device: str
    input_dim: int | None = None
    output_dim: int | None = None
    seq_len: int | None = None
    max_seq_len: int | None = None
    epoch: int | None = None
    val_mse: float | None = None
    val_mae: float | None = None
    source: str | None = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Return service liveness and model/checkpoint status."""
    cfg: dict = _state.get("config") or {}
    model_cfg = cfg.get("model") or {}
    data_cfg = cfg.get("data") or {}
    metrics = _state.get("metrics") or {}
    device = _state.get("device")
    model_ready = _state.get("model") is not None
    if _state.get("require_checkpoint"):
        model_ready = model_ready and bool(_state.get("checkpoint_loaded"))
    return HealthResponse(
        status="ok",
        ready=model_ready,
        checkpoint_loaded=bool(_state.get("checkpoint_loaded")),
        device=str(device) if device is not None else "",
        input_dim=model_cfg.get("input_dim"),
        output_dim=model_cfg.get("output_dim"),
        seq_len=data_cfg.get("seq_len"),
        max_seq_len=model_cfg.get("max_seq_len"),
        epoch=metrics.get("epoch"),
        val_mse=metrics.get("val_mse"),
        val_mae=metrics.get("val_mae"),
        source=metrics.get("source"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(body: PredictRequest) -> PredictResponse:
    """
    Run inference on a time-series sequence.

    The *sequence* field must be a 2-D list of shape ``[seq_len, input_dim]``
    where ``input_dim`` matches the value configured in ``config/config.yaml``.
    Set *horizon* > 1 to roll the forecast forward autoregressively.
    """
    return PredictResponse(prediction=run_predict(body.sequence, body.horizon))
