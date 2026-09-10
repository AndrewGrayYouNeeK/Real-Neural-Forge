"""FastAPI REST endpoint for time-series inference."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data import apply_scaler
from src.model import TimeSeriesTransformer

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Global model state
# ---------------------------------------------------------------------------
_state: dict[str, Any] = {}


def _load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def _build_model_from_config(cfg: dict) -> TimeSeriesTransformer:
    m_cfg = cfg["model"]
    return TimeSeriesTransformer(
        input_dim=m_cfg["input_dim"],
        output_dim=m_cfg["output_dim"],
        d_model=m_cfg["d_model"],
        nhead=m_cfg["nhead"],
        num_encoder_layers=m_cfg["num_encoder_layers"],
        dim_feedforward=m_cfg["dim_feedforward"],
        dropout=m_cfg["dropout"],
        max_seq_len=m_cfg["max_seq_len"],
    )


def load_model(config_path: str = "config/config.yaml") -> None:
    """Load (or lazily initialise) the model into the global state dict."""
    cfg = _load_config(config_path)
    i_cfg = cfg["inference"]

    device_str = i_cfg.get("device", "cpu")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available – falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)

    model = _build_model_from_config(cfg)
    checkpoint_path = Path(i_cfg.get("checkpoint_path", ""))
    checkpoint_loaded = False
    scaler = None

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        scaler = checkpoint.get("scaler")
        checkpoint_loaded = True
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.warning(
            "Checkpoint not found at '%s'. Using untrained model.", checkpoint_path
        )

    model.to(device)
    model.eval()

    _state["model"] = model
    _state["device"] = device
    _state["config"] = cfg
    _state["checkpoint_loaded"] = checkpoint_loaded
    _state["scaler"] = scaler


def run_predict(sequence: list[list[float]]) -> list[float]:
    """Run one inference pass against the loaded model."""
    model: TimeSeriesTransformer | None = _state.get("model")
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    device: torch.device = _state["device"]
    cfg: dict = _state["config"]
    expected_input_dim: int = cfg["model"]["input_dim"]
    max_seq_len: int = cfg["model"]["max_seq_len"]

    if not sequence:
        raise HTTPException(status_code=422, detail="sequence must not be empty.")
    if any(len(step) != expected_input_dim for step in sequence):
        raise HTTPException(
            status_code=422,
            detail=(
                f"Each time step must have {expected_input_dim} feature(s); "
                f"got a step with {len(sequence[0])} feature(s)."
            ),
        )
    if len(sequence) > max_seq_len:
        raise HTTPException(
            status_code=422,
            detail=(
                f"sequence length {len(sequence)} exceeds max_seq_len {max_seq_len}."
            ),
        )

    x = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)
    scaler = _state.get("scaler")
    if scaler:
        x = apply_scaler(x, scaler)

    with torch.no_grad():
        output = model(x)
        if scaler:
            output = apply_scaler(output, scaler, inverse=True)

    return output.squeeze(0).tolist()


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


class PredictResponse(BaseModel):
    """Output payload for the /predict endpoint."""

    prediction: list[float] = Field(
        ...,
        description="Model output of shape [output_dim].",
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


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["monitoring"])
def health() -> HealthResponse:
    """Return service liveness and model/checkpoint status."""
    cfg: dict = _state.get("config") or {}
    model_cfg = cfg.get("model") or {}
    data_cfg = cfg.get("data") or {}
    device = _state.get("device")
    return HealthResponse(
        status="ok",
        ready=_state.get("model") is not None,
        checkpoint_loaded=bool(_state.get("checkpoint_loaded")),
        device=str(device) if device is not None else "",
        input_dim=model_cfg.get("input_dim"),
        output_dim=model_cfg.get("output_dim"),
        seq_len=data_cfg.get("seq_len"),
        max_seq_len=model_cfg.get("max_seq_len"),
    )


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(body: PredictRequest) -> PredictResponse:
    """
    Run inference on a time-series sequence.

    The *sequence* field must be a 2-D list of shape ``[seq_len, input_dim]``
    where ``input_dim`` matches the value configured in ``config/config.yaml``.
    """
    return PredictResponse(prediction=run_predict(body.sequence))
