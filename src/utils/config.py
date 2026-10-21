"""YAML configuration helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, cast

import yaml

from src.utils.paths import resolve_repo_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config. Relative paths resolve from the repository root."""
    with open(resolve_repo_path(config_path)) as f:
        return cast(dict[str, Any], yaml.safe_load(f))


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overrides`` onto a copy of ``base``.

    Mapping values merge recursively. Every other value replaces the base.
    Neither input is mutated.
    """
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(value, dict) and isinstance(current, dict):
            merged[key] = merge_config(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged
