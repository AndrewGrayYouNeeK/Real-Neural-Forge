"""Tests for the NOAA Central Park temperature download."""

import math

import pandas as pd
import pytest

from src.data.weather import (
    END_DATE,
    LICENCE,
    SOURCE_URL,
    START_DATE,
    STATION_ID,
    STATION_NAME,
    add_calendar_features,
    build_download_url,
    build_station_url,
    fetch_weather,
    main,
    parse_noaa_daily_summaries,
    parse_station_observations,
)

_TWO_DAYS = (
    '"STATION","DATE","TMAX"\n'
    '"USW00094728","2020-01-01","1.5"\n'
    '"USW00094728","2020-01-02","2.5"\n'
)


def test_download_url_selects_metric_tmax_for_central_park():
    url = build_download_url()
    assert url.startswith(SOURCE_URL + "?")
    assert f"stations={STATION_ID}" in url
    assert "dataTypes=TMAX" in url
    assert "units=metric" in url
    assert f"startDate={START_DATE}" in url
    assert f"endDate={END_DATE}" in url
    assert "public domain" in LICENCE.lower()
    assert STATION_ID in LICENCE
    assert "CENTRAL PARK" in STATION_NAME


def test_parse_renames_tmax_and_sorts():
    text = (
        '"STATION","DATE","TMAX"\n'
        '"USW00094728","2020-01-02","2.5"\n'
        '"USW00094728","2020-01-01","1.5"\n'
    )
    frame = parse_noaa_daily_summaries(text)
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2020-01-01",
        "2020-01-02",
    ]
    assert frame["temperature_c"].tolist() == [1.5, 2.5]


def test_calendar_features_use_day_of_year_over_mean_year():
    frame = parse_noaa_daily_summaries(_TWO_DAYS)
    enriched = add_calendar_features(frame)
    angle = 2.0 * math.pi * 1 / 365.25
    assert enriched.loc[0, "doy_sin"] == pytest.approx(math.sin(angle))
    assert enriched.loc[0, "doy_cos"] == pytest.approx(math.cos(angle))


def test_parse_rejects_gaps_duplicates_and_bad_payloads():
    gapped = (
        '"STATION","DATE","TMAX"\n'
        '"USW00094728","2020-01-01","1.0"\n'
        '"USW00094728","2020-01-03","2.0"\n'
    )
    with pytest.raises(ValueError, match="not continuous"):
        parse_noaa_daily_summaries(gapped)

    duplicated = (
        '"STATION","DATE","TMAX"\n'
        '"USW00094728","2020-01-01","1.0"\n'
        '"USW00094728","2020-01-01","2.0"\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_noaa_daily_summaries(duplicated)

    with pytest.raises(ValueError, match="missing columns"):
        parse_noaa_daily_summaries('"STATION","DATE"\n"USW00094728","2020-01-01"\n')

    with pytest.raises(ValueError, match="no observations"):
        parse_noaa_daily_summaries('"STATION","DATE","TMAX"\n')

    missing_temp = '"STATION","DATE","TMAX"\n"USW00094728","2020-01-01",\n'
    with pytest.raises(ValueError, match="missing TMAX"):
        parse_noaa_daily_summaries(missing_temp)

    other_station = '"STATION","DATE","TMAX"\n"OTHER","2020-01-01","1.0"\n'
    with pytest.raises(ValueError, match="Expected only station"):
        parse_noaa_daily_summaries(other_station)


def test_fetch_writes_csv_from_the_noaa_payload(tmp_path, monkeypatch):
    seen: dict[str, str] = {}

    def fake_download(url: str, timeout: float = 120.0) -> str:
        seen["url"] = url
        return _TWO_DAYS

    monkeypatch.setattr("src.data.weather.download_noaa_csv", fake_download)
    path = fetch_weather(tmp_path / "weather.csv", start_date="2020-01-01")
    saved = pd.read_csv(path)
    assert list(saved.columns) == ["date", "temperature_c", "doy_sin", "doy_cos"]
    assert len(saved) == 2
    assert "startDate=2020-01-01" in seen["url"]
    assert STATION_ID in seen["url"]


def test_main_records_source_and_licence(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        "src.data.weather.download_noaa_csv", lambda url, timeout=120.0: _TWO_DAYS
    )
    destination = tmp_path / "weather.csv"
    main(
        [
            "--output",
            str(destination),
            "--start-date",
            "2020-01-01",
            "--end-date",
            "2020-01-02",
        ]
    )
    printed = capsys.readouterr().out
    assert "Public domain" in printed
    assert SOURCE_URL in printed
    assert destination.exists()


def test_station_url_keeps_commas_between_data_types():
    url = build_station_url(
        station="USW00093805",
        start_date="2015-10-04",
        end_date="2026-10-21",
        data_types=("TMAX", "TMIN", "PRCP"),
    )
    assert "stations=USW00093805" in url
    assert "dataTypes=TMAX,TMIN,PRCP" in url
    assert "TMAX%2C" not in url
    assert "units=metric" in url


def test_station_parser_allows_gaps_and_ignores_the_station_name():
    text = (
        '"STATION","NAME","DATE","TMAX","TMIN","PRCP"\n'
        '"USW00093805","TALLAHASSEE","2026-10-01","28.3","16.1","0.0"\n'
        '"USW00093805","TALLAHASSEE","2026-10-03","25.0","14.0","12.7"\n'
    )
    frame = parse_station_observations(
        text, station="USW00093805", data_types=("TMAX", "TMIN", "PRCP")
    )
    assert list(frame.columns) == ["date", "tmax_c", "tmin_c", "prcp_mm"]
    assert frame["date"].dt.strftime("%Y-%m-%d").tolist() == [
        "2026-10-01",
        "2026-10-03",
    ]
    assert frame["prcp_mm"].tolist() == [0.0, 12.7]

    duplicated = (
        '"STATION","DATE","TMAX"\n'
        '"USW00093805","2026-10-01","1.0"\n'
        '"USW00093805","2026-10-01","2.0"\n'
    )
    with pytest.raises(ValueError, match="duplicate"):
        parse_station_observations(
            duplicated, station="USW00093805", data_types=("TMAX",)
        )
