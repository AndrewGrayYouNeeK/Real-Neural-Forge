"""Shared config, device, and model construction."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
import yaml

from src.model import TimeSeriesTransformer

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path) as handle:
        return yaml.safe_load(handle)


def resolve_device(device_str: str | None = "cpu") -> torch.device:
    """Return a torch device, falling back to CPU when CUDA is unavailable."""
    requested = device_str or "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available – falling back to CPU.")
        requested = "cpu"
    return torch.device(requested)


def build_model(cfg: dict, device: torch.device) -> TimeSeriesTransformer:
    """Instantiate and return the transformer model."""
    m_cfg = cfg["model"]
    model = TimeSeriesTransformer(
        input_dim=m_cfg["input_dim"],
        output_dim=m_cfg["output_dim"],
        d_model=m_cfg["d_model"],
        nhead=m_cfg["nhead"],
        num_encoder_layers=m_cfg["num_encoder_layers"],
        dim_feedforward=m_cfg["dim_feedforward"],
        dropout=m_cfg["dropout"],
        max_seq_len=m_cfg["max_seq_len"],
    )
    return model.to(device)


def checkpoint_path_from_cfg(cfg: dict) -> Path:
    """Resolve the inference checkpoint path from config."""
    return Path(cfg.get("inference", {}).get("checkpoint_path", "checkpoints/best_model.pt"))


def apply_checkpoint_config(cfg: dict, checkpoint: dict) -> dict:
    """Prefer the model/data blocks saved with the weights."""
    saved = checkpoint.get("config")
    if isinstance(saved, dict):
        if "model" in saved:
            cfg["model"] = saved["model"]
        if "data" in saved:
            cfg["data"] = saved["data"]
    return cfg
