"""Parsers for the public files. No network."""

import json
from datetime import date, datetime, timezone

from src.realworld.parsers import (
    aggregate_ged_monthly,
    daily_air_quality,
    daily_earthquake_rows,
    parse_eia_series_line,
    parse_fred_csv,
    parse_silso_monthly,
    parse_team_goal_difference,
    tail_rows,
    year_over_year_percent,
)


def test_fred_csv_drops_missing_and_sorts():
    text = "observation_date,UNRATE\n2020-02-01,3.5\n2020-01-01,.\n2019-12-01,3.6\n"
    rows = parse_fred_csv(text)
    assert rows == [(date(2019, 12, 1), 3.6), (date(2020, 2, 1), 3.5)]


def test_cpi_year_over_year_percent():
    observations = [
        (date(2020, 1, 1), 100.0),
        (date(2020, 2, 1), 101.0),
        (date(2021, 1, 1), 110.0),
        (date(2021, 2, 1), 111.1),
    ]
    changes = year_over_year_percent(observations)
    assert changes[0][0] == date(2021, 1, 1)
    assert abs(changes[0][1] - 10.0) < 1e-9
    assert abs(changes[1][1] - 10.0) < 1e-9


def test_silso_skips_missing_months():
    text = "1749;01;1749.042;  96.7; -1.0; -1;1\n1749;02;1749.123; -1.0; -1.0; -1;1\n"
    rows = parse_silso_monthly(text)
    assert rows == [(date(1749, 1, 1), 96.7)]


def test_team_goal_difference_is_from_the_team_side():
    text = (
        "Div,Date,HomeTeam,AwayTeam,FTHG,FTAG,FTR\n"
        "E0,14/08/93,Arsenal,Coventry,0,3,A\n"
        "E0,18/08/93,Spurs,Arsenal,1,2,A\n"
        "E0,21/08/93,Everton,Liverpool,0,0,D\n"
    )
    rows = parse_team_goal_difference(text, "Arsenal")
    assert [row["value"] for row in rows] == [-3, 1]
    assert rows[0]["date"] == "1993-08-14"
    assert rows[1]["opponent"] == "Spurs"
    assert rows[1]["home"] == 0


def test_eia_line_sorts_and_ignores_other_series():
    payload = {
        "series_id": "EBA.US48-ALL.D.H",
        "data": [["20190101T01", "2"], ["20190101T00", None], ["20190101T02", 3]],
    }
    other = b'{"series_id":"EBA.CISO-ALL.D.H","data":[]}\n'
    parsed = parse_eia_series_line(json.dumps(payload).encode())
    assert parsed == [
        ("2019-01-01T01:00:00Z", 2.0),
        ("2019-01-01T02:00:00Z", 3.0),
    ]
    assert parse_eia_series_line(other) is None
    assert tail_rows(parsed or [], 1) == [("2019-01-01T02:00:00Z", 3.0)]


def test_daily_earthquakes_fill_silent_days_and_sum_energy():
    events = [
        (datetime(2020, 1, 1, 3, tzinfo=timezone.utc), 5.0),
        (datetime(2020, 1, 3, 1, tzinfo=timezone.utc), 4.5),
    ]
    rows = daily_earthquake_rows(events, date(2020, 1, 1), date(2020, 1, 3))
    assert [row["value"] for row in rows] == [1, 0, 1]
    assert rows[1]["energy_rel"] == 0
    assert rows[0]["log10_energy"] == "" or float(rows[0]["log10_energy"]) > 0
    assert abs(float(rows[0]["energy_rel"]) - 10 ** (1.5 * 5.0)) < 1e-6


def test_daily_air_quality_averages_hours_and_sums_pollen():
    payload = {
        "hourly": {
            "time": ["2024-04-01T00:00", "2024-04-01T01:00", "2024-04-02T00:00"],
            "pm2_5": [10.0, 20.0, 5.0],
            "ozone": [40.0, 60.0, 30.0],
            "alder_pollen": [1.0, 1.0, 0.0],
            "birch_pollen": [2.0, 0.0, None],
            "grass_pollen": [0.0, 0.0, 0.0],
            "mugwort_pollen": [None, None, None],
            "olive_pollen": [0.0, 0.0, 0.0],
            "ragweed_pollen": [0.0, 0.0, 0.0],
        }
    }
    rows = daily_air_quality(payload)
    assert rows[0]["date"] == "2024-04-01"
    assert rows[0]["pm2_5"] == 15
    assert rows[0]["ozone"] == 50
    assert rows[0]["pollen"] == 2.0
    assert rows[1]["pollen"] == 0.0


def test_ged_monthly_fills_empty_months_and_regions():
    records = [
        {"date_start": "1989-01-15 00:00:00.000", "best": "2", "region": "Africa"},
        {"date_start": "1989-03-01 00:00:00.000", "best": "5", "region": "Asia"},
    ]
    rows = aggregate_ged_monthly(records)
    assert [row["date"] for row in rows] == ["1989-01-01", "1989-02-01", "1989-03-01"]
    assert rows[1]["events"] == 0
    assert rows[0]["fatalities"] == 2
    assert rows[0]["events_africa"] == 1
    assert rows[2]["events_asia"] == 1
