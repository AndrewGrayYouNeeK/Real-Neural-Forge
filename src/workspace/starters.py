"""Shared starter kits shipped with the image.

The files under ``starters/`` are the read-only bases. A new account stores
rows that point at those objects. Editing or training writes a private copy
and leaves the base files alone.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import REPO_ROOT
from src.workspace.keys import base_dataset_key, base_model_key

CATALOG_PATH = REPO_ROOT / "starters" / "catalog.json"
_ID = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class Starter:
    """One sample a new account can predict with and train on."""

    id: str
    name: str
    summary: str
    source: str
    dataset_path: Path
    template_path: Path
    model_path: Path

    def template_text(self) -> str:
        return self.template_path.read_text(encoding="utf-8")

    def config(self) -> dict[str, Any]:
        loaded = yaml.safe_load(self.template_text())
        if not isinstance(loaded, dict):
            raise ValueError(f"Starter template {self.id} is not a mapping.")
        return loaded

    def dataset_key(self) -> str:
        return base_dataset_key(self.id)

    def model_key(self) -> str:
        return base_model_key(self.id)


def _starter(entry: dict[str, Any]) -> Starter:
    starter_id = str(entry.get("id", ""))
    if not _ID.fullmatch(starter_id):
        raise ValueError(f"Invalid starter id {starter_id!r}.")
    dataset = REPO_ROOT / str(entry["dataset"])
    template = REPO_ROOT / str(entry["template"])
    model = REPO_ROOT / str(entry["model"])
    return Starter(
        id=starter_id,
        name=str(entry.get("name") or starter_id),
        summary=str(entry.get("summary") or ""),
        source=str(entry.get("source") or ""),
        dataset_path=dataset,
        template_path=template,
        model_path=model,
    )


@lru_cache(maxsize=1)
def load_starters() -> tuple[Starter, ...]:
    """Return the starter catalog shipped in the repository."""
    payload = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    entries = payload.get("starters") if isinstance(payload, dict) else None
    if not isinstance(entries, list) or not entries:
        raise ValueError("starters/catalog.json must list at least one starter.")
    starters = tuple(_starter(entry) for entry in entries if isinstance(entry, dict))
    if len(starters) != len(entries):
        raise ValueError("starters/catalog.json has an entry that is not an object.")
    ids = [starter.id for starter in starters]
    if len(ids) != len(set(ids)):
        raise ValueError("starter ids must be unique.")
    return starters


def starter_by_id(starter_id: str) -> Starter | None:
    for starter in load_starters():
        if starter.id == starter_id:
            return starter
    return None


def clear_starter_cache() -> None:
    """Drop the cached catalog. Tests reload it after writing a temp tree."""
    load_starters.cache_clear()
