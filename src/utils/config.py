"""YAML configuration helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def load_config(config_path: str | Path) -> dict[str, Any]:
    with open(config_path) as f:
        return cast(dict[str, Any], yaml.safe_load(f))
