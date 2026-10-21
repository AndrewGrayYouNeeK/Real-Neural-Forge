"""Find dataset configs and the optional command that refreshes their data.

A dataset is any ``.yaml`` or ``.yml`` file under the config directory whose
document has ``model`` and ``training`` mappings. ``dataset.id`` names it;
otherwise the file stem is the id (``config/config.yaml`` in this repo sets
``dataset.id: default``). ``dataset.enabled: false`` or ``retrain.enabled: false``
skips the file.

Fresh data, when configured, comes from the first match below. Commands are
argument lists, never a shell.

1. ``data.fetch`` as a list of strings, a mapping with ``command`` / ``script`` /
   ``argv``, or a string split with ``shlex`` (no shell).
2. ``data.fetch_command`` or ``data.fetch_script``.
3. A top-level ``fetch`` key in the same shapes.
4. A conventional script, if the file exists: ``scripts/fetch_<id>.py``,
   ``scripts/fetch_<stem>.py``, ``scripts/<id>_fetch.py``, ``scripts/fetch_<id>.sh``,
   or the same names under ``data/``. The CSV stem of ``data.path`` is tried too,
   so ``data/powerball.csv`` finds ``scripts/fetch_powerball.py``.
"""

from __future__ import annotations

import re
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_SUFFIXES = {".yaml", ".yml"}
_DEFAULT_FETCH_TIMEOUT = 180.0


@dataclass(frozen=True)
class FetchPlan:
    """An argv that refreshes one dataset, plus its timeout in seconds."""

    argv: list[str]
    timeout: float


@dataclass(frozen=True)
class DatasetSpec:
    """One trainable config discovered on disk."""

    id: str
    path: Path
    root: Path
    fetch: FetchPlan | None

    def config_label(self) -> str:
        """Return a repo-relative path when the file lives under ``root``."""
        try:
            return self.path.resolve().relative_to(self.root.resolve()).as_posix()
        except ValueError:
            return str(self.path)


def discover(config_dir: Path) -> tuple[list[DatasetSpec], list[str]]:
    """Return enabled datasets under ``config_dir`` and human-readable errors."""
    root = config_dir.resolve().parent
    specs: list[DatasetSpec] = []
    errors: list[str] = []
    if not config_dir.is_dir():
        return [], [f"Config directory not found: {config_dir}"]
    paths = [
        path
        for path in sorted(config_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in _SUFFIXES
    ]
    for path in paths:
        spec, error = _load_path(path, root)
        if error:
            errors.append(error)
        elif spec is not None:
            specs.append(spec)
    seen: dict[str, Path] = {}
    unique: list[DatasetSpec] = []
    for spec in specs:
        previous = seen.get(spec.id)
        if previous is not None:
            errors.append(
                f"Dataset id {spec.id!r} is used by {previous} and {spec.path}."
            )
            continue
        seen[spec.id] = spec.path
        unique.append(spec)
    unique.sort(key=lambda spec: spec.id)
    return unique, errors


def load_dataset_file(path: Path) -> tuple[DatasetSpec | None, str | None]:
    """Load one config. The root is the parent of a ``config`` directory."""
    resolved = path.resolve()
    root = (
        resolved.parent.parent if resolved.parent.name == "config" else resolved.parent
    )
    return _load_path(resolved, root)


def _load_path(path: Path, root: Path) -> tuple[DatasetSpec | None, str | None]:
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        return None, f"{path}: {exc}"
    if not isinstance(loaded, dict):
        return None, None
    has_model = isinstance(loaded.get("model"), dict)
    has_training = isinstance(loaded.get("training"), dict)
    if not has_model or not has_training:
        if has_model or has_training or "dataset" in loaded or "retrain" in loaded:
            return (
                None,
                f"{path}: dataset configs need mapping sections 'model' and 'training'.",
            )
        return None, None
    try:
        if not _is_enabled(loaded):
            return None, None
        dataset_id = _dataset_id(loaded, path)
        fetch = resolve_fetch(loaded, dataset_id, path.stem, root)
    except ValueError as exc:
        return None, f"{path}: {exc}"
    return DatasetSpec(dataset_id, path, root, fetch), None


def _is_enabled(cfg: dict[str, Any]) -> bool:
    for section in ("dataset", "retrain"):
        block = cfg.get(section)
        if not isinstance(block, dict) or "enabled" not in block:
            continue
        enabled = block["enabled"]
        if not isinstance(enabled, bool):
            raise ValueError(f"{section}.enabled must be a boolean.")
        if not enabled:
            return False
    return True


def _dataset_id(cfg: dict[str, Any], path: Path) -> str:
    block = cfg.get("dataset")
    raw = block.get("id") if isinstance(block, dict) else None
    dataset_id = str(raw).strip() if isinstance(raw, str) and raw.strip() else path.stem
    if _ID.fullmatch(dataset_id) is None:
        raise ValueError(
            f"dataset id {dataset_id!r} must match {_ID.pattern} so release asset names stay safe."
        )
    return dataset_id


def resolve_fetch(
    cfg: dict[str, Any],
    dataset_id: str,
    stem: str,
    root: Path,
) -> FetchPlan | None:
    """Return the configured fetch command, or a conventional script if one exists."""
    data = cfg.get("data")
    data = data if isinstance(data, dict) else {}
    explicit = _explicit_fetch(data, cfg)
    if explicit is not None:
        return _parse_fetch(explicit, root, _fetch_timeout(data, cfg))
    for candidate in _conventional_scripts(root, dataset_id, stem, _csv_stems(data)):
        if candidate.is_file():
            return _script_plan(candidate, _fetch_timeout(data, cfg))
    return None


def _explicit_fetch(data: dict[str, Any], cfg: dict[str, Any]) -> Any:
    if "fetch" in data and data["fetch"] is not None:
        return data["fetch"]
    if data.get("fetch_command") is not None:
        return data["fetch_command"]
    if data.get("fetch_script") is not None:
        return data["fetch_script"]
    if "fetch" in cfg and cfg["fetch"] is not None:
        return cfg["fetch"]
    return None


def _fetch_timeout(data: dict[str, Any], cfg: dict[str, Any]) -> float:
    retrain = cfg.get("retrain")
    raw = data.get("fetch_timeout_seconds")
    if raw is None and isinstance(retrain, dict):
        raw = retrain.get("fetch_timeout_seconds")
    if raw is None:
        return _DEFAULT_FETCH_TIMEOUT
    try:
        timeout = float(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("fetch timeout must be a number of seconds.") from exc
    if timeout <= 0:
        raise ValueError("fetch timeout must be positive.")
    return timeout


def _csv_stems(data: dict[str, Any]) -> list[str]:
    """Return the file stem of ``data.path`` when it is a safe script name."""
    path = data.get("path")
    if not isinstance(path, str) or not path.strip():
        return []
    stem = Path(path).stem
    if _ID.fullmatch(stem) is None:
        return []
    return [stem]


def _conventional_scripts(
    root: Path,
    dataset_id: str,
    stem: str,
    extra_names: list[str] | None = None,
) -> list[Path]:
    names = [dataset_id]
    if stem != dataset_id:
        names.append(stem)
    for name in extra_names or []:
        if name not in names:
            names.append(name)
    paths: list[Path] = []
    for name in names:
        paths.extend(
            [
                root / "scripts" / f"fetch_{name}.py",
                root / "scripts" / f"{name}_fetch.py",
                root / "scripts" / f"fetch_{name}.sh",
                root / "data" / f"fetch_{name}.py",
                root / "data" / f"fetch_{name}.sh",
            ]
        )
    return paths


def _parse_fetch(value: Any, root: Path, timeout: float) -> FetchPlan:
    if isinstance(value, dict):
        if "timeout_seconds" in value:
            try:
                timeout = float(value["timeout_seconds"])
            except (TypeError, ValueError) as exc:
                raise ValueError("fetch.timeout_seconds must be a number.") from exc
            if timeout <= 0:
                raise ValueError("fetch.timeout_seconds must be positive.")
        if "command" in value:
            value = value["command"]
        elif "argv" in value:
            value = value["argv"]
        elif "script" in value:
            value = value["script"]
        else:
            raise ValueError("fetch mapping needs command, argv, or script.")
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            raise ValueError("fetch command must not be empty.")
        script = (
            (root / stripped).resolve()
            if not Path(stripped).is_absolute()
            else Path(stripped)
        )
        if " " not in stripped and script.suffix in {".py", ".sh"} and script.is_file():
            return _script_plan(script, timeout)
        argv = shlex.split(stripped)
        if not argv:
            raise ValueError("fetch command must not be empty.")
        return FetchPlan(argv, timeout)
    if isinstance(value, list):
        if not value or not all(isinstance(part, str) and part for part in value):
            raise ValueError("fetch command must be a list of non-empty strings.")
        return FetchPlan(list(value), timeout)
    raise ValueError("fetch must be a string, a list of strings, or a mapping.")


def _script_plan(script: Path, timeout: float) -> FetchPlan:
    if script.suffix == ".py":
        return FetchPlan([sys.executable, str(script)], timeout)
    if script.suffix == ".sh":
        return FetchPlan(["bash", str(script)], timeout)
    raise ValueError(f"Unsupported fetch script {script.name}.")
