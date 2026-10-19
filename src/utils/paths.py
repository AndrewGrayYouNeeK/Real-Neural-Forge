"""Repository-relative path helpers."""

from __future__ import annotations

from pathlib import Path

# src/utils/paths.py -> repository root
REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve ``path`` against the repository root when it is not absolute."""
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    return candidate.resolve()
