"""JSON results written by the domain training scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.paths import resolve_repo_path


def to_builtin(value: Any) -> Any:
    """Convert numpy containers so ``json`` can dump a results payload."""
    if isinstance(value, dict):
        return {str(key): to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_builtin(value.tolist())
    if isinstance(value, np.floating):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_results(path: str | Path, payload: dict[str, Any]) -> Path:
    destination = resolve_repo_path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(to_builtin(payload), indent=2, sort_keys=True) + "\n"
    )
    return destination
