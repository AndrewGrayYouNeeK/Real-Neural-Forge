"""Configurable caps that keep a shared container inside a small plan."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

# Environment variables override config/config.yaml ``limits`` which override
# these defaults. Names are listed in the Live hosting section of the README.
_ENV = {
    "max_dataset_bytes": "NEURAL_FORGE_MAX_DATASET_BYTES",
    "max_datasets_per_user": "NEURAL_FORGE_MAX_DATASETS",
    "max_config_bytes": "NEURAL_FORGE_MAX_CONFIG_BYTES",
    "max_epochs": "NEURAL_FORGE_MAX_EPOCHS",
    "max_run_seconds": "NEURAL_FORGE_MAX_RUN_SECONDS",
    "max_samples": "NEURAL_FORGE_MAX_SAMPLES",
    "max_seq_len": "NEURAL_FORGE_MAX_SEQ_LEN",
    "global_queue_limit": "NEURAL_FORGE_GLOBAL_QUEUE_LIMIT",
    "max_predictions_stored": "NEURAL_FORGE_MAX_PREDICTIONS",
    "max_prediction_points": "NEURAL_FORGE_MAX_PREDICTION_POINTS",
    "max_checkpoint_bytes": "NEURAL_FORGE_MAX_CHECKPOINT_BYTES",
    "max_active_jobs_per_user": "NEURAL_FORGE_MAX_ACTIVE_JOBS",
    "max_batch_size": "NEURAL_FORGE_MAX_BATCH_SIZE",
}


@dataclass(frozen=True)
class Limits:
    """Per-user and global caps for visitor work."""

    max_dataset_bytes: int = 2_000_000
    max_datasets_per_user: int = 12
    max_config_bytes: int = 64_000
    max_epochs: int = 5
    max_run_seconds: int = 90
    max_samples: int = 512
    max_seq_len: int = 32
    global_queue_limit: int = 8
    max_predictions_stored: int = 30
    max_prediction_points: int = 512
    max_checkpoint_bytes: int = 32_000_000
    max_active_jobs_per_user: int = 1
    max_batch_size: int = 32

    def as_dict(self) -> dict[str, int]:
        return {
            "max_dataset_bytes": self.max_dataset_bytes,
            "max_datasets_per_user": self.max_datasets_per_user,
            "max_config_bytes": self.max_config_bytes,
            "max_epochs": self.max_epochs,
            "max_run_seconds": self.max_run_seconds,
            "max_samples": self.max_samples,
            "max_seq_len": self.max_seq_len,
            "global_queue_limit": self.global_queue_limit,
            "max_predictions_stored": self.max_predictions_stored,
            "max_prediction_points": self.max_prediction_points,
            "max_checkpoint_bytes": self.max_checkpoint_bytes,
            "max_active_jobs_per_user": self.max_active_jobs_per_user,
            "max_batch_size": self.max_batch_size,
        }


def _positive(name: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{name} must be at least 1, got {value}.")
    return value


def load_limits(config: dict[str, Any] | None = None) -> Limits:
    """Resolve limits from defaults, optional config, then environment."""
    section: dict[str, Any] = {}
    if isinstance(config, dict):
        raw = config.get("limits")
        if isinstance(raw, dict):
            section = raw
    values: dict[str, int] = {}
    for field_name, env_name in _ENV.items():
        default = int(getattr(Limits, field_name))
        configured = section.get(field_name, default)
        if isinstance(configured, bool) or not isinstance(
            configured, int | float | str
        ):
            raise ValueError(f"limits.{field_name} must be an integer.")
        try:
            number = int(configured)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"limits.{field_name} must be an integer.") from exc
        override = os.environ.get(env_name, "").strip()
        if override:
            try:
                number = int(override)
            except ValueError as exc:
                raise ValueError(f"{env_name} must be an integer.") from exc
        values[field_name] = _positive(field_name, number)
    return Limits(**values)
