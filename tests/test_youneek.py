"""Tests for YouNeeK Time, calendar, and lunar clocks."""

from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from src.api import app, load_model
from src.youneek import (
    DAYS_PER_YEAR,
    EPOCH,
    MEAN_NEW_MOON_EPOCH,
    SYNODIC_MONTH_DAYS,
    snapshot,
    ticks_from_fraction,
)


@pytest.fixture(scope="module")
def client():
    load_model("config/config.yaml")
    with TestClient(app, raise_server_exceptions=True) as c:
        yield c


def test_noon_is_fifty():
    noon = datetime(2026, 6, 1, 12, 0, 0, tzinfo=timezone.utc)
    snap = snapshot(noon)
    assert snap.time_of_day["display"] == "50:00:00"
    assert snap.time_of_day["units"] == 50
    assert snap.time_of_day["minutes"] == 0
    assert snap.time_of_day["seconds"] == 0


def test_midnight_is_zero():
    midnight = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    snap = snapshot(midnight)
    assert snap.time_of_day["display"] == "00:00:00"


def test_mid_year_clock_is_fifty():
    mid = EPOCH + timedelta(days=DAYS_PER_YEAR / 2)
    snap = snapshot(mid)
    assert snap.calendar["year_clock"]["display"] == "50:00:00"
    assert snap.calendar["days_per_year"] == 354
    assert snap.calendar["days_per_week"] == 10


def test_ten_day_weeks_and_epoch_year_zero():
    day_nine = EPOCH + timedelta(days=9)
    day_ten = EPOCH + timedelta(days=10)
    first = snapshot(day_nine).calendar
    second = snapshot(day_ten).calendar
    assert first["year_index"] == 0
    assert first["week"] == 0
    assert first["weekday"] == 9
    assert second["week"] == 1
    assert second["weekday"] == 0


def test_new_moon_and_full_moon_clocks():
    new = snapshot(MEAN_NEW_MOON_EPOCH)
    assert new.lunar["clock"]["display"] == "00:00:00"
    full = snapshot(MEAN_NEW_MOON_EPOCH + timedelta(days=SYNODIC_MONTH_DAYS / 2))
    assert full.lunar["clock"]["display"] == "50:00:00"
    assert full.lunar["new_moon"] == "00:00:00"
    assert full.lunar["full_moon"] == "50:00:00"


def test_ticks_wrap():
    assert ticks_from_fraction(0.0) == 0
    assert ticks_from_fraction(1.0) == 0


def test_youneek_api_endpoints(client: TestClient):
    now = client.get("/youneek/now")
    assert now.status_code == 200
    body = now.json()
    assert body["scale"]["units_per_day"] == 100
    assert body["scale"]["noon"] == "50:00:00"
    assert "10/10/10" in body["scale"]["note"]
    assert "time" in body
    assert "calendar" in body
    assert "lunar" in body

    converted = client.get("/youneek/convert", params={"at": "2026-01-01T12:00:00Z"})
    assert converted.status_code == 200
    assert converted.json()["time"]["display"] == "50:00:00"

    calendar = client.get("/youneek/calendar", params={"at": "2026-01-01T00:00:00Z"})
    assert calendar.status_code == 200
    assert calendar.json()["year_index"] == 0
    assert calendar.json()["day_in_year"] == 0

    lunar = client.get("/youneek/lunar", params={"at": MEAN_NEW_MOON_EPOCH.isoformat()})
    assert lunar.status_code == 200
    assert lunar.json()["clock"]["display"] == "00:00:00"

    forecast = client.get(
        "/youneek/forecast/next-minute", params={"at": "2026-01-01T00:00:00Z"}
    )
    assert forecast.status_code == 200
    data = forecast.json()
    assert data["youneek_display"] == "00:01:00"
    assert abs(data["youneek_minute_earth_seconds"] - 8.64) < 1e-9


def test_convert_rejects_bad_timestamp(client: TestClient):
    resp = client.get("/youneek/convert", params={"at": "not-a-date"})
    assert resp.status_code == 422
