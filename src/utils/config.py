"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml

from src.utils.paths import resolve_repo_path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML config. Relative paths resolve from the repository root."""
    with open(resolve_repo_path(config_path)) as f:
        return cast(dict[str, Any], yaml.safe_load(f))
