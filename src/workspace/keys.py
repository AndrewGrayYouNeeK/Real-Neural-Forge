"""Object keys for shared starter bases and private user bytes."""

from __future__ import annotations

import re

_KEY = re.compile(
    r"^(bases/starters/[a-z0-9-]+/(dataset\.csv|model\.pt)"
    r"|users/[A-Za-z0-9_]+/(datasets|models)/[A-Za-z0-9_.-]+)$"
)


def valid_object_key(key: str) -> bool:
    """Return True when ``key`` is a workspace object this app is allowed to touch."""
    return bool(_KEY.fullmatch(key))


def base_dataset_key(starter_id: str) -> str:
    return f"bases/starters/{starter_id}/dataset.csv"


def base_model_key(starter_id: str) -> str:
    return f"bases/starters/{starter_id}/model.pt"


def user_dataset_key(user_id: str, dataset_id: str) -> str:
    return f"users/{user_id}/datasets/{dataset_id}"


def user_model_key(user_id: str, model_id: str) -> str:
    return f"users/{user_id}/models/{model_id}.pt"
