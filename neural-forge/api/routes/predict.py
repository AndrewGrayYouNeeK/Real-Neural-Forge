"""Prediction endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()


class PredictRequest(BaseModel):
    """Request body for the predict endpoint."""

    inputs: list[list[float]]
    """List of feature vectors, each of shape ``[feature_dim]``."""


class PredictResponse(BaseModel):
    """Response body for the predict endpoint."""

    predictions: list[int]
    probabilities: list[list[float]]


@router.post("", response_model=PredictResponse, summary="Run model inference")
async def predict(request: PredictRequest) -> Any:
    """Run inference on a batch of feature vectors.

    .. note::
        This is a placeholder implementation.  In a production deployment
        you would inject a :class:`~neural_forge.inference.predictor.Predictor`
        via FastAPI dependency injection and call it here.
    """
    try:
        import torch

        x = torch.tensor(request.inputs, dtype=torch.float32)
        # Placeholder: return zeros until a real model is wired in.
        batch_size = x.shape[0]
        predictions = [0] * batch_size
        probabilities = [[1.0]] * batch_size
        return PredictResponse(predictions=predictions, probabilities=probabilities)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
