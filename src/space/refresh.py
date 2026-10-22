"""Install and refresh the space event files the API serves.

``data/`` is excluded from the image. ``space_snapshot/`` is the committed
copy of the event files, baked into the image and used when a fetch fails.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import tempfile
import threading
from pathlib import Path

from src.space.near_earth import fetch_near_earth
from src.space.solar_system import fetch_solar_system
from src.utils.paths import REPO_ROOT

logger = logging.getLogger("uvicorn.error")

SNAPSHOT_DIR = REPO_ROOT / "space_snapshot"
LIVE_DIR = REPO_ROOT / "data" / "space"
REFRESH_INTERVAL_SECONDS = 24 * 60 * 60

NEAR_EARTH_FILES = (
    "near_earth_events.csv",
    "reentry_events.csv",
    "fireball_events.csv",
    "sentry_summary.json",
    "interstellar_objects.csv",
    "near_earth_manifest.json",
)
SOLAR_FILES = (
    "solar_approach_events.csv",
    "solar_comet_catalog.csv",
    "horizons_daily.csv",
    "horizons_monthly.csv",
    "solar_moons.json",
    "solar_system_manifest.json",
)
SNAPSHOT_FILES = NEAR_EARTH_FILES + SOLAR_FILES

_refresh_lock = threading.Lock()


def space_refresh_enabled() -> bool:
    """Background refresh is on unless ``NEURAL_FORGE_SPACE_REFRESH`` disables it."""
    raw = os.environ.get("NEURAL_FORGE_SPACE_REFRESH", "1").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def ensure_space_snapshot() -> list[str]:
    """Copy snapshot files that are not already in ``data/space``.

    Existing live files are left alone so a later refresh is not replaced by
    the baked copy on the next process start. A copy error is logged and does
    not stop the API.
    """
    copied: list[str] = []
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create %s", LIVE_DIR)
        return copied
    for name in SNAPSHOT_FILES:
        source = SNAPSHOT_DIR / name
        target = LIVE_DIR / name
        if target.is_file():
            continue
        if not source.is_file():
            logger.warning("Space snapshot is missing %s", source)
            continue
        try:
            shutil.copy2(source, target)
        except OSError:
            logger.exception("Could not install space snapshot %s", name)
            continue
        copied.append(name)
    if copied:
        logger.info("Installed space snapshot files: %s", ", ".join(copied))
    return copied


def refresh_space_events() -> dict[str, str]:
    """Download fresh event files. Keep the previous files when a fetch fails."""
    if not _refresh_lock.acquire(blocking=False):
        logger.info("Space refresh already running")
        return {"near_earth": "busy", "solar_system": "busy"}
    try:
        logger.info("Refreshing space event files")
        return {
            "near_earth": _refresh_group(
                "near-Earth", fetch_near_earth, NEAR_EARTH_FILES
            ),
            "solar_system": _refresh_group(
                "solar-system", fetch_solar_system, SOLAR_FILES
            ),
        }
    finally:
        _refresh_lock.release()


def _refresh_group(
    label: str,
    fetch,
    names: tuple[str, ...],
) -> str:
    backup = _backup(names)
    try:
        fetch(refresh=True)
    except Exception:
        logger.exception("%s refresh failed; keeping the previous event files", label)
        _restore(backup, names)
        return "fallback"
    shutil.rmtree(backup, ignore_errors=True)
    logger.info("%s event files refreshed", label)
    return "refreshed"


def _backup(names: tuple[str, ...]) -> Path:
    directory = Path(tempfile.mkdtemp(prefix="space-refresh-"))
    for name in names:
        source = LIVE_DIR / name
        if not source.is_file():
            continue
        target = directory / name
        try:
            shutil.copy2(source, target)
        except OSError:
            logger.exception("Could not back up %s", source)
    return directory


def _restore(backup: Path, names: tuple[str, ...]) -> None:
    """Put the pre-refresh files back, then fill any gap from the snapshot."""
    try:
        LIVE_DIR.mkdir(parents=True, exist_ok=True)
    except OSError:
        logger.exception("Could not create %s for restore", LIVE_DIR)
        return
    for name in names:
        previous = backup / name
        snapshot = SNAPSHOT_DIR / name
        source = previous if previous.is_file() else snapshot
        if not source.is_file():
            continue
        target = LIVE_DIR / name
        try:
            shutil.copy2(source, target)
        except OSError:
            logger.exception("Could not restore %s", target)
    shutil.rmtree(backup, ignore_errors=True)


async def space_refresh_loop(stop: asyncio.Event) -> None:
    """Refresh once at startup, then once a day, until ``stop`` is set."""
    while not stop.is_set():
        try:
            await asyncio.to_thread(refresh_space_events)
        except Exception:
            logger.exception("Space refresh failed")
        if stop.is_set():
            return
        try:
            await asyncio.wait_for(stop.wait(), timeout=REFRESH_INTERVAL_SECONDS)
        except TimeoutError:
            continue
        return
