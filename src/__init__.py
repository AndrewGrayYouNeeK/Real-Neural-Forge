"""Neural Forge: transformer pipeline for time-series prediction."""

from __future__ import annotations

from pathlib import Path

_PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def project_version() -> str:
    """Return the ``[project]`` version declared in ``pyproject.toml``."""
    in_project = False
    for line in _PYPROJECT.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        key, separator, raw = stripped.partition("=")
        if separator and key.strip() == "version":
            return raw.strip().strip('"').strip("'")
    raise RuntimeError(f"version is not set in {_PYPROJECT}")


__version__ = project_version()

__all__ = ["__version__", "project_version"]
