"""Solar-system parsers and the summary report. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.space import solar_system
from src.space.planets import (
    alignment_events,
    comet_passages,
    jd_to_timestamp,
    parse_horizons_table,
    parse_moon_discoveries,
    perihelion_times,
    valid_discovery_dates,
)
from src.utils.config import load_config

MARS_EPHEMERIS = """
$$SOE
 2025-Jan-15 00:00     0.64293470581846   0.9363767  175.4464 /L
 2025-Jan-16 00:00     0.64358688181095   1.3221450  175.7109 /L
 2025-Jan-17 00:00     0.64446196177428   1.7082143  175.5216 /T
 2025-Jan-18 00:00     0.64556001230599   2.0942062  174.9304 /T
$$EOE
"""

SUN_EPHEMERIS = """
$$SOE
 1900-Jan-01 00:00     0.42072585671066   8.2089144
 1900-Feb-01 00:00     0.44519627039132  -5.8505591
$$EOE
"""

MOON_HTML = """
<table>
<tr class="sat-discovery-planet"><td colspan="6">Satellites of  Mars:
    2</td></tr>
<tr><td class="text-right">I</td><td>Phobos</td><td></td><td>1877</td><td>Hall</td></tr>
<tr><td class="text-right">II</td><td>Deimos</td><td></td><td>1877</td><td>Hall</td></tr>
<tr class="sat-discovery-planet"><td colspan="6">Satellites of  Jupiter: 1</td></tr>
<tr>
  <td class="text-right">XVIII</td><td>Themisto</td>
  <td>S/1975 J1 = S/2000 J1</td><td>1975, 2000</td><td>Kowal</td>
</tr>
</table>
"""

HALLEY_JD = 2_446_469.97


def test_horizons_parser_keeps_elongation_side_and_sun_range():
    mars = parse_horizons_table(MARS_EPHEMERIS)
    assert list(mars["date"].dt.strftime("%Y-%m-%d"))[1] == "2025-01-16"
    assert mars.iloc[1]["elongation_deg"] == 175.7109
    assert mars.iloc[1]["elongation_side"] == "L"
    assert mars.iloc[2]["elongation_side"] == "T"
    sun = parse_horizons_table(SUN_EPHEMERIS)
    assert len(sun) == 2
    assert pd.isna(sun.iloc[0]["elongation_deg"])
    assert sun.iloc[0]["elongation_side"] == ""
    assert abs(float(sun.iloc[0]["delta_au"]) - 0.42072585671066) < 1e-12


def test_mars_opposition_is_the_interior_elongation_peak():
    events = alignment_events(parse_horizons_table(MARS_EPHEMERIS), inner=False)
    assert len(events) == 1
    assert events[0]["kind"] == "opposition"
    assert events[0]["date"] == "2025-01-16"
    assert events[0]["side"] == "L"
    assert events[0]["elongation_deg"] == 175.7109


def test_inner_planet_extrema_are_elongation_and_conjunction():
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
            ),
            "elongation_deg": [12.0, 22.0, 11.0, 2.0, 14.0],
            "elongation_side": ["", "T", "", "", "L"],
        }
    )
    events = alignment_events(frame, inner=True)
    assert [event["kind"] for event in events] == [
        "greatest elongation",
        "conjunction",
    ]
    assert events[0]["date"] == "2026-01-02"
    assert events[0]["side"] == "T"
    assert events[1]["date"] == "2026-01-04"
    # The first and last samples are window edges and are not events.
    assert all(event["date"] not in {"2026-01-01", "2026-01-05"} for event in events)


def test_halley_osculating_perihelion_is_february_1986():
    stamp = jd_to_timestamp(HALLEY_JD)
    assert stamp.year == 1986
    assert stamp.month == 2
    times = perihelion_times(
        HALLEY_JD, 27_700.0, pd.Timestamp("1986-01-01"), pd.Timestamp("1986-03-01")
    )
    assert len(times) == 1
    assert times[0].year == 1986 and times[0].month == 2


def test_short_period_steps_and_long_period_does_not():
    origin = jd_to_timestamp(HALLEY_JD)
    # A 1-year period from the same instant produces the anniversary inside a later year.
    stepped = perihelion_times(
        HALLEY_JD,
        365.25,
        origin + pd.Timedelta(days=360),
        origin + pd.Timedelta(days=400),
    )
    assert len(stepped) == 1
    assert abs((stepped[0] - (origin + pd.Timedelta(days=365.25))).total_seconds()) < 2
    single = perihelion_times(
        HALLEY_JD,
        300 * 365.25,
        pd.Timestamp("1980-01-01"),
        pd.Timestamp("2100-01-01"),
    )
    assert len(single) == 1
    assert single[0].year == 1986


def test_far_osculating_perihelion_does_not_overflow():
    times = perihelion_times(
        10_000_000.0,
        365.25,
        pd.Timestamp("2020-01-01"),
        pd.Timestamp("2020-06-01"),
    )
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-06-01")
    assert all(start <= stamp < end for stamp in times)


def test_known_at_time_drops_passages_before_discovery():
    catalog = pd.DataFrame(
        [
            {
                "pdes": "1P",
                "full_name": "Halley",
                "first_obs": "1990-01-01",
                "tp": HALLEY_JD,
                "per": 365.25,
            },
            {
                "pdes": "2P",
                "full_name": "Encke",
                "first_obs": "1980-01-01",
                "tp": HALLEY_JD,
                "per": 365.25,
            },
        ]
    )
    passages = comet_passages(
        catalog,
        start=pd.Timestamp("1986-01-01"),
        end=pd.Timestamp("1986-03-01"),
        known_only=True,
    )
    assert list(passages["designation"]) == ["2P"]
    upcoming = comet_passages(
        catalog,
        start=pd.Timestamp("1986-01-01"),
        end=pd.Timestamp("1986-03-01"),
        known_only=False,
    )
    assert set(upcoming["designation"]) == {"1P", "2P"}


def test_incomplete_discovery_dates_are_dropped():
    dated, dropped = valid_discovery_dates(
        pd.Series(["2020-01-15", "2020-??-??", "2019-05-??", "1699-01-01"]),
        through=pd.Timestamp("2026-10-01"),
    )
    assert dropped == 3
    assert len(dated) == 1
    assert dated.iloc[0] == pd.Timestamp("2020-01-15")


def test_moon_table_uses_the_first_discovery_year():
    parsed = parse_moon_discoveries(MOON_HTML)
    counts = {row["planet"]: row["count"] for row in parsed["by_planet"]}
    assert counts == {"Mars": 2, "Jupiter": 1}
    themisto = next(row for row in parsed["discoveries"] if row["name"] == "Themisto")
    assert themisto["year"] == 1975
    assert themisto["planet"] == "Jupiter"
    assert sum(1 for row in parsed["discoveries"] if row["name"] == "Themisto") == 1


def test_missing_cache_is_available_false(monkeypatch):
    root = Path("/tmp/solar-system-missing")
    for name in (
        "EVENTS_PATH",
        "COMET_CATALOG_PATH",
        "HORIZONS_DAILY_PATH",
        "HORIZONS_MONTHLY_PATH",
        "MOONS_PATH",
        "FORECAST_PATH",
    ):
        monkeypatch.setattr(solar_system, name, str(root / name))
    monkeypatch.setattr(solar_system, "utc_today", lambda: pd.Timestamp("2026-10-22"))
    report = solar_system.solar_system_report()
    assert report["available"] is False
    assert report["days"] == 730
    assert report["outlook_months"] == 24
    assert report["showers"]["available"] is False
    start = pd.Timestamp(report["window_start"])
    end = pd.Timestamp(report["window_end"])
    assert (end - start).days == 730
    short = solar_system.solar_system_report(days=90)
    assert short["days"] == 90
    assert (
        pd.Timestamp(short["window_end"]) - pd.Timestamp(short["window_start"])
    ).days == 90


def test_report_filters_the_window_and_joins_horizons(tmp_path, monkeypatch):
    events = pd.DataFrame(
        [
            {
                "body": "Mercury",
                "body_code": "Merc",
                "designation": "2026 AA",
                "name": "(2026 AA)",
                "cd": "2026-11-02 03:00:00",
                "dist_au": 0.01,
                "h": 20.0,
                "dist_max_au": 0.05,
            },
            {
                "body": "Neptune",
                "body_code": "Neptn",
                "designation": "far",
                "name": "far out",
                "cd": "2028-01-01 00:00:00",
                "dist_au": 9.0,
                "h": "",
                "dist_max_au": 10.0,
            },
        ]
    )
    catalog = pd.DataFrame(
        [
            {
                "pdes": "2P",
                "full_name": "Encke",
                "first_obs": "1800-01-01",
                "tp": HALLEY_JD,
                "per": 365.25,
            }
        ]
    )
    # Anniversary of the 1986-02-09 osculating instant falls in November only if
    # the period is stepped; pin a passage by using the JD of 2026-11-15.
    passage = pd.Timestamp("2026-11-15")
    catalog.loc[0, "tp"] = (passage - pd.Timestamp("1970-01-01")) / pd.Timedelta(
        days=1
    ) + 2_440_587.5
    catalog.loc[0, "per"] = 10_000.0
    daily = pd.DataFrame(
        {
            "date": ["2026-10-21", "2026-10-22", "2026-10-23", "2026-10-24"],
            "body": ["Mars", "Mars", "Mars", "Mars"],
            "earth_au": [1.0, 1.0, 1.0, 1.0],
            "elongation_deg": [170.0, 176.0, 174.0, 160.0],
            "elongation_side": ["L", "L", "T", "T"],
        }
    )
    horizons = pd.DataFrame(
        {"date": ["2026-11-01"], "mars_earth_au": [1.5], "mercury_earth_au": [0.8]}
    )
    forecast = {
        "series": [
            {
                "id": "mars_distance",
                "title": "Mars distance from Earth",
                "units": "AU",
                "survey_growth": False,
                "control": "ephemeris",
                "physics_column": "mars_earth_au",
                "model": {"mae": 0.2, "rmse": 0.3},
                "trainer_baselines": {
                    "persistence": {"mae": 0.1},
                    "seasonal_naive": {"mae": 0.05},
                },
                "mean": {"mae": 0.4},
                "outlook": {
                    "months": [
                        {"date": "2026-11-01", "p10": 1.0, "p50": 2.0, "p90": 3.0}
                    ]
                },
            },
            {
                "id": "mercury_approaches",
                "title": "Mercury approaches within 0.05 AU",
                "units": "count",
                "survey_growth": True,
                "control": "raw",
                "model": {"mae": 3.0, "rmse": 4.0},
                "trainer_baselines": {
                    "persistence": {"mae": 1.0},
                    "seasonal_naive": {"mae": 1.0},
                },
                "mean": {"mae": 2.0},
                "outlook": {
                    "months": [
                        {"date": "2026-11-01", "p10": 0.0, "p50": 1.0, "p90": 2.0}
                    ]
                },
            },
        ]
    }
    events_path = tmp_path / "events.csv"
    catalog_path = tmp_path / "comets.csv"
    daily_path = tmp_path / "daily.csv"
    horizons_path = tmp_path / "horizons.csv"
    forecast_path = tmp_path / "forecast.json"
    events.to_csv(events_path, index=False)
    catalog.to_csv(catalog_path, index=False)
    daily.to_csv(daily_path, index=False)
    horizons.to_csv(horizons_path, index=False)
    forecast_path.write_text(json.dumps(forecast), encoding="utf-8")
    monkeypatch.setattr(solar_system, "EVENTS_PATH", str(events_path))
    monkeypatch.setattr(solar_system, "COMET_CATALOG_PATH", str(catalog_path))
    monkeypatch.setattr(solar_system, "HORIZONS_DAILY_PATH", str(daily_path))
    monkeypatch.setattr(solar_system, "HORIZONS_MONTHLY_PATH", str(horizons_path))
    monkeypatch.setattr(
        solar_system, "MOONS_PATH", str(tmp_path / "missing-moons.json")
    )
    monkeypatch.setattr(solar_system, "FORECAST_PATH", str(forecast_path))
    monkeypatch.setattr(solar_system, "utc_today", lambda: pd.Timestamp("2026-10-22"))

    report = solar_system.solar_system_report(days=90)
    assert [row["name"] for row in report["approaches"]] == ["(2026 AA)"]
    assert report["approaches"][0]["body"] == "Mercury"
    assert report["perihelia"][0]["name"] == "Encke"
    assert report["perihelia"][0]["date"] == "2026-11-15"
    assert report["alignments"][0]["kind"] == "opposition"
    assert report["alignments"][0]["body"] == "Mars"
    assert report["alignments"][0]["date"] == "2026-10-22"
    neptune = next(row for row in report["bodies"] if row["name"] == "Neptune")
    assert neptune["trained"] is False
    assert neptune["upcoming_count"] == 0
    mercury = next(row for row in report["bodies"] if row["name"] == "Mercury")
    assert mercury["upcoming_count"] == 1
    assert mercury["dist_max_au"] == 0.05
    distance = next(row for row in report["forecasts"] if row["id"] == "mars_distance")
    assert distance["outlook"]["months"][0]["physics"] == 1.5
    assert distance["physics_outlook_mae"] == 0.5
    counts = next(
        row for row in report["forecasts"] if row["id"] == "mercury_approaches"
    )
    assert counts["jpl_monthly_key"] == "mercury"
    november = next(row for row in report["monthly"] if row["month"] == "2026-11-01")
    assert november["mercury"] == 1
    assert november["perihelia"] == 1
    wide = solar_system.solar_system_report(days=730)
    assert any(row["body"] == "Neptune" for row in wide["approaches"])


def test_recorded_solar_forecast_covers_every_config():
    recorded = json.loads(
        Path("results/solar_system_forecast.json").read_text(encoding="utf-8")
    )
    configs = {
        load_config(path)["space"]["id"]
        for path in Path("config/space").glob("*.yaml")
        if load_config(path)["space"].get("group") == "solar_system"
    }
    series = recorded["series"]
    assert {row["id"] for row in series} == configs
    assert recorded["seed"] == 0
    assert recorded["lookahead_days"] == 730
    assert recorded["outlook_months"] == 24
    assert recorded["total_train_seconds"] > 0
    for row in series:
        months = row["outlook"]["months"]
        assert len(months) == 24
        for month in months:
            assert month["p10"] <= month["p50"] <= month["p90"]
