"""Snapshot install, failed-fetch fallback, and the baked event files."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

import src.space.approaches as approaches
import src.space.refresh as refresh
import src.space.solar_system as solar_system
from scripts.check_image_events import require_served_events
from src.api import read_powerball_scorecard
from src.utils.paths import REPO_ROOT


def _write_snapshot(root: Path, payload: str = "kept\n") -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in refresh.SNAPSHOT_FILES:
        (root / name).write_text(payload, encoding="utf-8")


def _use_dirs(monkeypatch: pytest.MonkeyPatch, snap: Path, live: Path) -> None:
    monkeypatch.setattr(refresh, "SNAPSHOT_DIR", snap)
    monkeypatch.setattr(refresh, "LIVE_DIR", live)


def test_ensure_copies_missing_files_and_keeps_existing(tmp_path, monkeypatch):
    snap = tmp_path / "space_snapshot"
    live = tmp_path / "data" / "space"
    _write_snapshot(snap, "from-snapshot\n")
    live.mkdir(parents=True)
    (live / "near_earth_events.csv").write_text("already-live\n", encoding="utf-8")
    _use_dirs(monkeypatch, snap, live)

    copied = refresh.ensure_space_snapshot()

    assert "near_earth_events.csv" not in copied
    assert (live / "near_earth_events.csv").read_text(encoding="utf-8") == (
        "already-live\n"
    )
    assert set(copied) == set(refresh.SNAPSHOT_FILES) - {"near_earth_events.csv"}
    assert (live / "solar_moons.json").read_text(encoding="utf-8") == "from-snapshot\n"


def test_failed_refresh_restores_previous_files(tmp_path, monkeypatch):
    snap = tmp_path / "space_snapshot"
    live = tmp_path / "data" / "space"
    payload = "body,date\nEarth,2026-11-01\n"
    _write_snapshot(snap, payload)
    _use_dirs(monkeypatch, snap, live)
    refresh.ensure_space_snapshot()

    def fail_near(*, refresh: bool = False, output_dir: str = "data/space") -> None:
        assert refresh is True
        (live / "near_earth_events.csv").write_text("broken", encoding="utf-8")
        raise RuntimeError("cad down")

    def ok_solar(*, refresh: bool = False, output_dir: str = "data/space") -> None:
        assert refresh is True
        (live / "solar_approach_events.csv").write_text("updated\n", encoding="utf-8")

    monkeypatch.setattr(refresh, "fetch_near_earth", fail_near)
    monkeypatch.setattr(refresh, "fetch_solar_system", ok_solar)

    result = refresh.refresh_space_events()

    assert result == {"near_earth": "fallback", "solar_system": "refreshed"}
    assert (live / "near_earth_events.csv").read_text(encoding="utf-8") == payload
    assert (live / "solar_approach_events.csv").read_text(encoding="utf-8") == (
        "updated\n"
    )
    assert (live / "solar_moons.json").read_text(encoding="utf-8") == payload


def test_refresh_fills_from_snapshot_when_live_file_was_missing(tmp_path, monkeypatch):
    snap = tmp_path / "space_snapshot"
    live = tmp_path / "data" / "space"
    _write_snapshot(snap, "snapshot-row\n")
    live.mkdir(parents=True)
    _use_dirs(monkeypatch, snap, live)

    def fail_near(*, refresh: bool = False, output_dir: str = "data/space") -> None:
        (live / "near_earth_events.csv").write_text("partial", encoding="utf-8")
        raise RuntimeError("cad down")

    def fail_solar(*, refresh: bool = False, output_dir: str = "data/space") -> None:
        raise RuntimeError("horizons down")

    monkeypatch.setattr(refresh, "fetch_near_earth", fail_near)
    monkeypatch.setattr(refresh, "fetch_solar_system", fail_solar)

    result = refresh.refresh_space_events()

    assert result == {"near_earth": "fallback", "solar_system": "fallback"}
    assert (live / "near_earth_events.csv").read_text(encoding="utf-8") == (
        "snapshot-row\n"
    )
    assert (live / "horizons_daily.csv").read_text(encoding="utf-8") == "snapshot-row\n"


def test_refresh_loop_runs_once_at_startup_then_stops(monkeypatch):
    calls: list[str] = []
    stop = asyncio.Event()

    def fake_refresh() -> dict[str, str]:
        calls.append("refresh")
        stop.set()
        return {"near_earth": "refreshed", "solar_system": "refreshed"}

    monkeypatch.setattr(refresh, "refresh_space_events", fake_refresh)
    monkeypatch.setattr(refresh, "REFRESH_INTERVAL_SECONDS", 3600)

    asyncio.run(refresh.space_refresh_loop(stop))

    assert calls == ["refresh"]


def test_refresh_is_disabled_for_the_test_suite():
    assert refresh.space_refresh_enabled() is False


def test_committed_snapshot_serves_events(monkeypatch):
    root = REPO_ROOT / "space_snapshot"
    for name in refresh.SNAPSHOT_FILES:
        path = root / name
        assert path.is_file()
        assert path.stat().st_size > 0
    monkeypatch.setattr(approaches, "EVENTS_PATH", str(root / "near_earth_events.csv"))
    monkeypatch.setattr(
        approaches, "REENTRY_EVENTS_PATH", str(root / "reentry_events.csv")
    )
    monkeypatch.setattr(
        approaches, "FIREBALL_EVENTS_PATH", str(root / "fireball_events.csv")
    )
    monkeypatch.setattr(approaches, "SENTRY_PATH", str(root / "sentry_summary.json"))
    monkeypatch.setattr(
        approaches, "INTERSTELLAR_PATH", str(root / "interstellar_objects.csv")
    )
    monkeypatch.setattr(
        approaches, "MANIFEST_PATH", str(root / "near_earth_manifest.json")
    )
    monkeypatch.setattr(
        solar_system, "EVENTS_PATH", str(root / "solar_approach_events.csv")
    )
    monkeypatch.setattr(
        solar_system, "COMET_CATALOG_PATH", str(root / "solar_comet_catalog.csv")
    )
    monkeypatch.setattr(
        solar_system, "HORIZONS_DAILY_PATH", str(root / "horizons_daily.csv")
    )
    monkeypatch.setattr(
        solar_system, "HORIZONS_MONTHLY_PATH", str(root / "horizons_monthly.csv")
    )
    monkeypatch.setattr(solar_system, "MOONS_PATH", str(root / "solar_moons.json"))
    today = pd.Timestamp("2026-10-22")
    near = approaches.near_earth_report(days=730, today=today)
    solar = solar_system.solar_system_report(days=730, today=today)
    scorecard = read_powerball_scorecard()
    require_served_events(
        near,
        solar,
        scorecard,
        {"history": [{"actual": 1.0}], "forecast": [{"value": 1.0}]},
    )
    assert near["reentries"] or near["fireballs_recent"]
    assert scorecard["available"] is True


def test_require_served_events_rejects_an_empty_approach_list():
    payload = {
        "approaches": [{"name": "x"}],
        "perihelia": [{"name": "x"}],
        "alignments": [{"body": "Mars"}],
        "forecasts": [{"id": "x"}],
        "moons": {"by_planet": [{"planet": "Mars"}]},
    }
    scorecard = {"available": True}
    series = {"history": [{}], "forecast": [{}]}
    with pytest.raises(SystemExit, match="near-earth"):
        require_served_events(
            {**payload, "approaches": []},
            payload,
            scorecard,
            series,
        )
