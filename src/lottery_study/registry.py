"""Find every dataset the lottery study can see.

Catalog rows in :mod:`src.lottery_study.catalog` are the TODO list. A row
becomes available when its module exports ``LOTTERY_STUDY_HOOKS`` or one of
its CSV paths exists. Anything else under ``src/data/`` or ``data/`` is
picked up as well, so a fetcher that lands on main is included on the next
run without editing the study.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.lottery_study.align import PRIOR_CALENDAR_DAY, SAME_DAY_BEFORE_DRAW
from src.lottery_study.catalog import DatasetCandidate, dataset_catalog
from src.lottery_study.sources import LoadedFrame, load_wired
from src.utils.paths import resolve_repo_path

_SKIP_MODULES = frozenset({"lottery", "scaling", "timeseries"})
_SKIP_CSV_NAMES = frozenset({"powerball.csv"})
_MARKET_TOKENS = ("stock", "equity", "fx", "forex", "currenc", "commodit", "future")


@dataclass
class DatasetLoad:
    """A catalog row after the registry has looked for its files."""

    dataset_id: str
    group: str
    description: str
    available: bool
    detail: str
    availability: str
    columns: tuple[str, ...]
    observed_at: pd.Series | None
    values: np.ndarray | None
    source: str
    display_name: str = ""
    consumed_paths: tuple[str, ...] = ()


def resolve_datasets(cfg: dict[str, Any]) -> list[DatasetLoad]:
    """Load catalog datasets and, unless disabled, discover any others."""
    overrides = cfg.get("dataset_paths") or {}
    if not isinstance(overrides, dict):
        raise ValueError("dataset_paths must be a mapping of dataset id to CSV path")
    fetch = bool(cfg.get("fetch_missing", False))
    discover = bool(cfg.get("discover_unlisted", True))
    only = cfg.get("only_datasets")
    only_ids = {str(item) for item in only} if only else None
    modules: dict[str, Any] = {}
    loads: list[DatasetLoad] = []
    claimed: set[str] = set()
    used_paths: set[Path] = set()
    for candidate in dataset_catalog():
        if only_ids is not None and candidate.dataset_id not in only_ids:
            continue
        load = _load_candidate(
            candidate,
            overrides.get(candidate.dataset_id),
            fetch=fetch,
            cfg=cfg,
            modules=modules,
        )
        loads.append(load)
        claimed.add(candidate.dataset_id)
        if load.source and load.source != "in-memory hook":
            used_paths.add(Path(load.source).resolve())
        for consumed in load.consumed_paths:
            used_paths.add(Path(consumed).resolve())
    if discover:
        loads.extend(
            _discover_unlisted(claimed, used_paths, modules, fetch=fetch, cfg=cfg)
        )
    return loads


def coerce_table(
    frame: pd.DataFrame,
    date_columns: tuple[str, ...],
    value_columns: tuple[str, ...] | None,
) -> tuple[pd.Series, np.ndarray, tuple[str, ...]]:
    """Pull a date column and numeric columns out of a frame."""
    if frame.empty:
        raise ValueError("dataset frame is empty")
    date_column = _date_column(frame, date_columns)
    chosen = _value_columns(frame, date_column, value_columns)
    values = np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
            for column in chosen
        ]
    )
    return frame[date_column], values, tuple(str(column) for column in chosen)


def _load_candidate(
    candidate: DatasetCandidate,
    override: object,
    *,
    fetch: bool,
    cfg: dict[str, Any],
    modules: dict[str, Any],
) -> DatasetLoad:
    hook = _matching_hook(candidate, modules)
    if hook is not None:
        try:
            return _load_from_hook(candidate, hook)
        except (OSError, ValueError, KeyError) as exc:
            return _unavailable(candidate, f"hook failed: {exc}")

    if candidate.loader:
        try:
            loaded = load_wired(candidate.dataset_id, candidate.loader, fetch=fetch)
        except (OSError, RuntimeError, ValueError, KeyError) as exc:
            return _unavailable(candidate, f"loader failed: {exc}")
        return _from_wired(candidate, loaded)

    paths = _candidate_paths(candidate, override)
    existing = next((path for path in paths if path.is_file()), None)
    if existing is None and fetch and candidate.fetcher:
        try:
            fetched = _call_fetcher(candidate, cfg)
        except (OSError, RuntimeError, ValueError) as exc:
            return _unavailable(candidate, f"fetch failed: {exc}")
        if fetched is not None:
            existing = fetched if fetched.is_file() else None
        if existing is None:
            existing = next((path for path in paths if path.is_file()), None)
    if existing is None:
        return _unavailable(
            candidate,
            "not on main yet; export LOTTERY_STUDY_HOOKS or write one of: "
            + ", ".join(candidate.csv_candidates),
        )
    try:
        observed_at, values, columns = _read_csv(
            existing, candidate.date_columns, candidate.value_columns
        )
    except (OSError, ValueError) as exc:
        return _unavailable(candidate, f"could not read {existing}: {exc}")
    return DatasetLoad(
        dataset_id=candidate.dataset_id,
        group=candidate.group,
        description=candidate.description,
        available=True,
        detail=f"loaded {len(observed_at)} rows from {existing}",
        availability=candidate.availability,
        columns=columns,
        observed_at=observed_at.reset_index(drop=True),
        values=values,
        source=str(existing),
    )


def _discover_unlisted(
    claimed: set[str],
    used_paths: set[Path],
    modules: dict[str, Any],
    *,
    fetch: bool,
    cfg: dict[str, Any],
) -> list[DatasetLoad]:
    found: list[DatasetLoad] = []
    data_dir = resolve_repo_path("src/data")
    if data_dir.is_dir():
        for path in sorted(data_dir.glob("*.py")):
            if path.stem in _SKIP_MODULES or path.stem.startswith("_"):
                continue
            module = _import_module(f"src.data.{path.stem}", modules)
            if module is None:
                continue
            for hook in _hooks(module):
                dataset_id = str(hook.get("dataset_id") or path.stem)
                if dataset_id in claimed:
                    continue
                candidate = _candidate_from_hook(
                    dataset_id, hook, module_name=f"src.data.{path.stem}"
                )
                load = _load_candidate(
                    candidate, None, fetch=fetch, cfg=cfg, modules=modules
                )
                if load.available:
                    found.append(load)
                    claimed.add(dataset_id)
            default_output = getattr(module, "DEFAULT_OUTPUT", None)
            if default_output is None:
                continue
            csv_path = resolve_repo_path(str(default_output))
            dataset_id = csv_path.stem
            if (
                dataset_id in claimed
                or csv_path.name in _SKIP_CSV_NAMES
                or csv_path.resolve() in used_paths
            ):
                continue
            if not csv_path.is_file():
                continue
            candidate = DatasetCandidate(
                dataset_id=dataset_id,
                group="discovered",
                description=f"Discovered from {path.stem}.DEFAULT_OUTPUT.",
                modules=(f"src.data.{path.stem}",),
                csv_candidates=(str(csv_path),),
                availability=_guess_availability(dataset_id),
            )
            load = _load_candidate(
                candidate, csv_path, fetch=False, cfg=cfg, modules=modules
            )
            if load.available:
                found.append(load)
                claimed.add(dataset_id)
                used_paths.add(csv_path.resolve())

    data_root = resolve_repo_path(str(cfg.get("data_dir", "data")))
    if data_root.is_dir():
        for csv_path in sorted(data_root.rglob("*.csv")):
            if (
                csv_path.name in _SKIP_CSV_NAMES
                or csv_path.resolve() in used_paths
                or "raw" in csv_path.parts
                or "cache" in csv_path.parts
            ):
                continue
            dataset_id = csv_path.stem
            if dataset_id in claimed:
                continue
            candidate = DatasetCandidate(
                dataset_id=dataset_id,
                group="discovered",
                description=f"Discovered CSV at {csv_path}.",
                modules=(),
                csv_candidates=(str(csv_path),),
                availability=_guess_availability(dataset_id),
            )
            load = _load_candidate(
                candidate, csv_path, fetch=False, cfg=cfg, modules=modules
            )
            if load.available:
                found.append(load)
                claimed.add(dataset_id)
                used_paths.add(csv_path.resolve())
    return found


def _matching_hook(
    candidate: DatasetCandidate, modules: dict[str, Any]
) -> dict[str, Any] | None:
    for module_name in candidate.modules:
        module = _import_module(module_name, modules)
        if module is None:
            continue
        for hook in _hooks(module):
            hook_id = hook.get("dataset_id")
            if hook_id is None or str(hook_id) == candidate.dataset_id:
                return hook
    return None


def _load_from_hook(candidate: DatasetCandidate, hook: dict[str, Any]) -> DatasetLoad:
    availability = str(hook.get("availability", candidate.availability))
    if availability not in (PRIOR_CALENDAR_DAY, SAME_DAY_BEFORE_DRAW):
        raise ValueError(f"unsupported availability {availability!r}")
    date_columns = tuple(hook.get("date_columns", candidate.date_columns))
    value_columns = hook.get("value_columns", candidate.value_columns)
    if value_columns is not None:
        value_columns = tuple(value_columns)
    frame = hook.get("frame")
    source = "in-memory hook"
    if frame is None:
        csv_path = hook.get("csv") or hook.get("path")
        if not csv_path:
            raise ValueError("hook has neither frame nor csv")
        path = resolve_repo_path(str(csv_path))
        if not path.is_file():
            raise ValueError(f"hook CSV is missing: {path}")
        frame = pd.read_csv(path)
        source = str(path)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("hook frame must be a DataFrame")
    observed_at, values, columns = coerce_table(frame, date_columns, value_columns)
    description = str(hook.get("description") or candidate.description)
    return DatasetLoad(
        dataset_id=candidate.dataset_id,
        group=str(hook.get("group") or candidate.group),
        description=description,
        available=True,
        detail=f"loaded {len(observed_at)} rows from {source}",
        availability=availability,
        columns=columns,
        observed_at=observed_at.reset_index(drop=True),
        values=values,
        source=source,
    )


def _candidate_from_hook(
    dataset_id: str, hook: dict[str, Any], *, module_name: str
) -> DatasetCandidate:
    csv_value = hook.get("csv") or hook.get("path")
    csv_candidates = (str(csv_value),) if csv_value else ()
    value_columns = hook.get("value_columns")
    return DatasetCandidate(
        dataset_id=dataset_id,
        group=str(hook.get("group") or "discovered"),
        description=str(
            hook.get("description") or f"Discovered hook in {module_name}."
        ),
        modules=(module_name,),
        csv_candidates=csv_candidates,
        availability=str(hook.get("availability") or _guess_availability(dataset_id)),
        value_columns=tuple(value_columns) if value_columns is not None else None,
    )


def _candidate_paths(candidate: DatasetCandidate, override: object) -> list[Path]:
    paths: list[Path] = []
    if override:
        paths.append(resolve_repo_path(str(override)))
    paths.extend(resolve_repo_path(path) for path in candidate.csv_candidates)
    return paths


def _call_fetcher(candidate: DatasetCandidate, cfg: dict[str, Any]) -> Path | None:
    if not candidate.fetcher or ":" not in candidate.fetcher:
        return None
    module_name, func_name = candidate.fetcher.split(":", 1)
    module = importlib.import_module(module_name)
    func = getattr(module, func_name)
    destination = resolve_repo_path(candidate.csv_candidates[0])
    kwargs: dict[str, Any] = {}
    parameters = inspect.signature(func).parameters
    if "end_date" in parameters:
        end = cfg.get("fetch_end_date") or date.today().isoformat()
        kwargs["end_date"] = str(end)
    if "output" in parameters:
        result = func(destination, **kwargs)
    else:
        result = func(**kwargs)
    if isinstance(result, Path):
        return result
    return destination


def _read_csv(
    path: Path,
    date_columns: tuple[str, ...],
    value_columns: tuple[str, ...] | None,
) -> tuple[pd.Series, np.ndarray, tuple[str, ...]]:
    return coerce_table(pd.read_csv(path), date_columns, value_columns)


def _date_column(frame: pd.DataFrame, date_columns: tuple[str, ...]) -> str:
    lowered = {str(column).lower(): str(column) for column in frame.columns}
    for candidate in date_columns:
        if candidate in frame.columns:
            return str(candidate)
        found = lowered.get(candidate.lower())
        if found is not None:
            return found
    for column in frame.columns:
        name = str(column).lower()
        if "date" in name or name in {"time", "timestamp", "datetime"}:
            return str(column)
    raise ValueError("no date column")


def _value_columns(
    frame: pd.DataFrame,
    date_column: str,
    value_columns: tuple[str, ...] | None,
) -> list[str]:
    if value_columns:
        missing = [column for column in value_columns if column not in frame.columns]
        if missing:
            raise ValueError(f"missing columns: {missing}")
        return list(value_columns)
    chosen: list[str] = []
    for column in frame.columns:
        if str(column) == date_column:
            continue
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if int(numeric.notna().sum()) >= min(len(frame), 3):
            chosen.append(str(column))
    if not chosen:
        raise ValueError("no numeric columns")
    return chosen


def _hooks(module: Any) -> list[dict[str, Any]]:
    raw = getattr(module, "LOTTERY_STUDY_HOOKS", None)
    if raw is None:
        reader = getattr(module, "lottery_study_hooks", None)
        if callable(reader):
            raw = reader()
    if raw is None:
        return []
    if isinstance(raw, dict):
        return [raw]
    return [item for item in raw if isinstance(item, dict)]


def _import_module(name: str, cache: dict[str, Any]) -> Any | None:
    if name in cache:
        return cache[name]
    try:
        module = importlib.import_module(name)
    except ModuleNotFoundError:
        module = None
    cache[name] = module
    return module


def _guess_availability(dataset_id: str) -> str:
    lowered = dataset_id.lower()
    if any(token in lowered for token in _MARKET_TOKENS):
        return SAME_DAY_BEFORE_DRAW
    return PRIOR_CALENDAR_DAY


def _from_wired(candidate: DatasetCandidate, loaded: LoadedFrame) -> DatasetLoad:
    return DatasetLoad(
        dataset_id=candidate.dataset_id,
        group=candidate.group,
        description=candidate.description,
        available=loaded.available,
        detail=loaded.detail,
        availability=candidate.availability,
        columns=loaded.columns,
        observed_at=(
            None
            if loaded.observed_at is None
            else loaded.observed_at.reset_index(drop=True)
        ),
        values=loaded.values,
        source=loaded.source,
        display_name=candidate.display_name,
        consumed_paths=loaded.consumed,
    )


def _unavailable(candidate: DatasetCandidate, detail: str) -> DatasetLoad:
    return DatasetLoad(
        dataset_id=candidate.dataset_id,
        group=candidate.group,
        description=candidate.description,
        available=False,
        detail=detail,
        availability=candidate.availability,
        columns=(),
        observed_at=None,
        values=None,
        source="",
        display_name=candidate.display_name,
    )
