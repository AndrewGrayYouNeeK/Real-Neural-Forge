"""Champion files, forecast JSON, metrics history, and the rolling GitHub Release.

Release tag ``champions`` (a prerelease, not the project's latest software
release) stores flat asset names:

- ``champion-<dataset_id>.pt``
- ``forecast-<dataset_id>.json``
- ``forecasts-index.json``
- ``metrics-history.json``

Local directories use ``<dataset_id>.pt``, ``<dataset_id>.json``, and
``index.json`` so the ids stay easy to read. Nothing here is committed.
"""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RELEASE_TAG = "champions"
SCHEMA_VERSION = 1
HISTORY_NAME = "metrics-history.json"
INDEX_NAME = "index.json"
RELEASE_INDEX_NAME = "forecasts-index.json"
MAX_RECORDS_PER_DATASET = 400
_ID_PREFIX = "champion-"
_FORECAST_PREFIX = "forecast-"


def utc_now() -> str:
    """Return a UTC timestamp without microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def champion_path(champions_dir: Path, dataset_id: str) -> Path:
    return champions_dir / f"{dataset_id}.pt"


def forecast_path(forecasts_dir: Path, dataset_id: str) -> Path:
    return forecasts_dir / f"{dataset_id}.json"


def write_json(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def promote_checkpoint(source: Path, dest: Path) -> None:
    """Atomically replace ``dest`` with a copy of ``source``."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    temporary = dest.with_name(dest.name + ".promoting")
    shutil.copy2(source, temporary)
    os.replace(temporary, dest)


def append_history(path: Path, record: dict[str, Any]) -> None:
    """Append ``record``, keeping the newest records for its dataset."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    with lock_path.open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            if path.is_file():
                loaded = json.loads(path.read_text(encoding="utf-8"))
                document = loaded if isinstance(loaded, dict) else {}
            else:
                document = {}
            records = document.get("records")
            if not isinstance(records, list):
                records = []
            records.append(record)
            document["schema_version"] = SCHEMA_VERSION
            document["records"] = _trim_records(records)
            write_json(path, document)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _trim_records(records: list[Any]) -> list[Any]:
    kept: list[Any] = []
    seen: dict[str, int] = {}
    for record in reversed(records):
        dataset_id = record.get("dataset_id") if isinstance(record, dict) else None
        if isinstance(dataset_id, str):
            seen[dataset_id] = seen.get(dataset_id, 0) + 1
            if seen[dataset_id] > MAX_RECORDS_PER_DATASET:
                continue
        kept.append(record)
    kept.reverse()
    return kept


def import_release(release_dir: Path, champions_dir: Path) -> None:
    """Copy release assets into the local champions directory."""
    if not release_dir.is_dir():
        return
    champions_dir.mkdir(parents=True, exist_ok=True)
    history = release_dir / HISTORY_NAME
    if history.is_file():
        shutil.copy2(history, champions_dir / HISTORY_NAME)
    for asset in sorted(release_dir.glob(f"{_ID_PREFIX}*.pt")):
        dataset_id = asset.name[len(_ID_PREFIX) : -len(".pt")]
        if not dataset_id or "/" in dataset_id or dataset_id in {".", ".."}:
            continue
        shutil.copy2(asset, champion_path(champions_dir, dataset_id))


def export_release(champions_dir: Path, forecasts_dir: Path, release_dir: Path) -> None:
    """Write flat release asset names for every local champion and forecast."""
    release_dir.mkdir(parents=True, exist_ok=True)
    history = champions_dir / HISTORY_NAME
    if history.is_file():
        shutil.copy2(history, release_dir / HISTORY_NAME)
    if champions_dir.is_dir():
        for ckpt in sorted(champions_dir.glob("*.pt")):
            if ckpt.name.endswith(".promoting"):
                continue
            shutil.copy2(ckpt, release_dir / f"{_ID_PREFIX}{ckpt.stem}.pt")
    if forecasts_dir.is_dir():
        index = forecasts_dir / INDEX_NAME
        if index.is_file():
            shutil.copy2(index, release_dir / RELEASE_INDEX_NAME)
        for document in sorted(forecasts_dir.glob("*.json")):
            if document.name == INDEX_NAME:
                continue
            shutil.copy2(
                document, release_dir / f"{_FORECAST_PREFIX}{document.stem}.json"
            )


def write_index(
    forecasts_dir: Path,
    entries: list[dict[str, Any]],
    *,
    release_dir: Path | None = None,
) -> dict[str, Any]:
    """Merge ``entries`` into the previous index and write ``forecasts/index.json``."""
    merged: dict[str, dict[str, Any]] = {}
    if release_dir is not None:
        previous = release_dir / RELEASE_INDEX_NAME
        if previous.is_file():
            loaded = json.loads(previous.read_text(encoding="utf-8"))
            datasets = loaded.get("datasets") if isinstance(loaded, dict) else None
            if isinstance(datasets, list):
                for item in datasets:
                    if isinstance(item, dict) and isinstance(item.get("id"), str):
                        merged[item["id"]] = item
    for entry in entries:
        dataset_id = entry.get("id")
        if isinstance(dataset_id, str):
            merged[dataset_id] = entry
    document = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": utc_now(),
        "datasets": [merged[key] for key in sorted(merged)],
    }
    write_json(forecasts_dir / INDEX_NAME, document)
    return document


def release_files(release_dir: Path) -> list[Path]:
    """Return the files that belong in the champions release."""
    if not release_dir.is_dir():
        return []
    selected: list[Path] = []
    for path in sorted(release_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if name in {HISTORY_NAME, RELEASE_INDEX_NAME}:
            selected.append(path)
        elif name.startswith(_ID_PREFIX) and name.endswith(".pt"):
            selected.append(path)
        elif name.startswith(_FORECAST_PREFIX) and name.endswith(".json"):
            selected.append(path)
    return selected


def _repo_args() -> list[str]:
    repo = os.environ.get("GITHUB_REPOSITORY", "").strip()
    if repo:
        return ["--repo", repo]
    return []


def _run_gh(args: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(args, check=check, capture_output=True, text=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "gh is not installed. Install the GitHub CLI and authenticate to publish champions."
        ) from exc


def pull_release(release_dir: Path) -> bool:
    """Download the champions release into ``release_dir``. Missing release is fine."""
    release_dir.mkdir(parents=True, exist_ok=True)
    repo = _repo_args()
    view = _run_gh(["gh", "release", "view", RELEASE_TAG, *repo], check=False)
    if view.returncode != 0:
        return False
    download = _run_gh(
        [
            "gh",
            "release",
            "download",
            RELEASE_TAG,
            "--dir",
            str(release_dir),
            "--clobber",
            *repo,
        ],
        check=False,
    )
    if download.returncode != 0:
        detail = (download.stderr or download.stdout or "").strip()
        raise RuntimeError(f"Could not download the champions release: {detail}")
    return True


def publish_release(release_dir: Path) -> None:
    """Create the rolling champions release if needed and upload its assets."""
    files = release_files(release_dir)
    if not files:
        raise RuntimeError(f"No champion assets to publish in {release_dir}.")
    repo = _repo_args()
    view = _run_gh(["gh", "release", "view", RELEASE_TAG, *repo], check=False)
    if view.returncode != 0:
        notes = (
            "Rolling champion checkpoints, per-dataset forecast JSON, and metrics "
            "history. This is not a software release."
        )
        created = _run_gh(
            [
                "gh",
                "release",
                "create",
                RELEASE_TAG,
                "--prerelease",
                "--latest=false",
                "--title",
                "Champion checkpoints",
                "--notes",
                notes,
                *repo,
            ],
            check=False,
        )
        if created.returncode != 0:
            detail = (created.stderr or created.stdout or "").strip()
            raise RuntimeError(f"Could not create the champions release: {detail}")
    uploaded = _run_gh(
        [
            "gh",
            "release",
            "upload",
            RELEASE_TAG,
            *[str(path) for path in files],
            "--clobber",
            *repo,
        ],
        check=False,
    )
    if uploaded.returncode != 0:
        detail = (uploaded.stderr or uploaded.stdout or "").strip()
        raise RuntimeError(f"Could not upload champion assets: {detail}")
