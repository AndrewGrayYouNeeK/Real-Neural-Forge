"""Configuration file loading with OmegaConf."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(path: str | Path, overrides: list[str] | None = None) -> DictConfig:
    """Load a YAML config and apply optional dot-notation overrides.

    Args:
        path:      Path to the YAML configuration file.
        overrides: List of OmegaConf dot-notation overrides,
                   e.g. ``["training.lr=1e-4", "training.epochs=50"]``.

    Returns:
        Merged :class:`DictConfig`.
    """
    cfg: DictConfig = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg
