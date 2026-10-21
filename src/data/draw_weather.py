"""Join Powerball draw dates to public-domain NOAA daily weather.

Powerball draws are held in Tallahassee, Florida. The weather-augmented
lottery model sees weather on the dates already in its input window. The
target draw's own weather is not an input, because that day has not happened
when the ticket is logged.

Observations are a U.S. government work (17 U.S.C. § 105). NOAA NCEI Daily
Summaries distribute GHCNd with no use fee.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.data.weather import STATION_ID as CENTRAL_PARK_STATION
from src.data.weather import fetch_station_observations

TALLAHASSEE_STATION = "USW00093805"
TALLAHASSEE_NAME = "TALLAHASSEE REGIONAL AIRPORT, FL US"
TALLAHASSEE_DATA_TYPES = ("TMAX", "TMIN", "PRCP")
TALLAHASSEE_COLUMNS = ("tmax_c", "tmin_c", "prcp_mm")
CENTRAL_PARK_COLUMN = "central_park_tmax_c"
# A draw on a day the station did not report can use the previous one or two
# days. Longer gaps stay missing rather than inventing a measurement.
MAX_WEATHER_GAP_DAYS = 2
WEATHER_LICENCE = (
    "Public domain (U.S. government work, 17 U.S.C. § 105). "
    "NOAA NCEI Daily Summaries, GHCNd, metric units."
)


@dataclass(frozen=True)
class WeatherJoin:
    """Draw table plus the weather columns that were actually aligned."""

    frame: pd.DataFrame
    tallahassee_columns: tuple[str, ...]
    central_park_column: str | None
    tallahassee_gaps: int
    central_park_note: str


def align_station_to_draws(
    draw_dates: pd.Series,
    observations: pd.DataFrame,
    columns: tuple[str, ...] | list[str],
    *,
    max_gap_days: int = MAX_WEATHER_GAP_DAYS,
) -> pd.DataFrame:
    """Map each draw date to the latest observation on or before that date.

    Gaps longer than ``max_gap_days`` leave the value columns as NaN. The
    returned frame has one row per draw, in the original order.
    """
    if max_gap_days < 0:
        raise ValueError("max_gap_days must be non-negative.")
    missing = [column for column in columns if column not in observations.columns]
    if missing:
        raise ValueError(f"Weather observations are missing columns: {missing}")
    left = pd.DataFrame(
        {
            "_row": range(len(draw_dates)),
            "draw_date": pd.to_datetime(draw_dates),
        }
    )
    right = observations.loc[:, ["date", *columns]].rename(
        columns={"date": "observed_on"}
    )
    right = right.sort_values("observed_on")
    ordered = left.sort_values("draw_date")
    merged = pd.merge_asof(
        ordered,
        right,
        left_on="draw_date",
        right_on="observed_on",
        direction="backward",
    )
    gap_days = (merged["draw_date"] - merged["observed_on"]).dt.days
    usable = gap_days.notna() & (gap_days <= max_gap_days)
    merged["weather_gap_days"] = gap_days
    merged.loc[~usable, "weather_gap_days"] = pd.NA
    for column in columns:
        merged.loc[~usable, column] = pd.NA
    return merged.sort_values("_row").drop(columns=["_row"]).reset_index(drop=True)


def join_draw_weather(
    draws: pd.DataFrame,
    tallahassee: pd.DataFrame,
    central_park: pd.DataFrame | None = None,
    *,
    max_gap_days: int = MAX_WEATHER_GAP_DAYS,
) -> WeatherJoin:
    """Attach Tallahassee weather, and Central Park only when every draw aligns.

    Central Park is optional. A partial Central Park series is left off so the
    model input width stays fixed. Rows with incomplete Tallahassee weather are
    kept in ``frame`` and marked by NaN so the caller can drop them.
    """
    aligned = align_station_to_draws(
        draws["draw_date"],
        tallahassee,
        TALLAHASSEE_COLUMNS,
        max_gap_days=max_gap_days,
    )
    frame = draws.reset_index(drop=True).copy()
    for column in TALLAHASSEE_COLUMNS:
        frame[column] = pd.to_numeric(aligned[column], errors="coerce")
    gaps = int(frame[list(TALLAHASSEE_COLUMNS)].isna().any(axis=1).sum())

    central_column: str | None = None
    central_note = "not requested"
    if central_park is not None:
        park = align_station_to_draws(
            draws["draw_date"],
            central_park,
            ("tmax_c",),
            max_gap_days=max_gap_days,
        )
        park_values = pd.to_numeric(park["tmax_c"], errors="coerce")
        if bool(park_values.isna().any()):
            missing = int(park_values.isna().sum())
            central_note = (
                f"omitted: {missing} draw dates have no Central Park "
                f"{CENTRAL_PARK_STATION} observation within {max_gap_days} days"
            )
        else:
            frame[CENTRAL_PARK_COLUMN] = park_values
            central_column = CENTRAL_PARK_COLUMN
            central_note = f"included from {CENTRAL_PARK_STATION}"
    return WeatherJoin(
        frame=frame,
        tallahassee_columns=TALLAHASSEE_COLUMNS,
        central_park_column=central_column,
        tallahassee_gaps=gaps,
        central_park_note=central_note,
    )


def complete_weather_rows(
    frame: pd.DataFrame, columns: tuple[str, ...] | list[str]
) -> pd.DataFrame:
    """Drop draws whose weather columns are missing. Order is preserved."""
    if not columns:
        return frame.reset_index(drop=True)
    mask = ~frame.loc[:, list(columns)].isna().any(axis=1)
    return frame.loc[mask].reset_index(drop=True)


def fetch_tallahassee(
    start_date: str,
    end_date: str,
    *,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Download Tallahassee airport max temp, min temp, and precipitation."""
    return fetch_station_observations(
        station=TALLAHASSEE_STATION,
        data_types=TALLAHASSEE_DATA_TYPES,
        start_date=start_date,
        end_date=end_date,
        timeout=timeout,
    )


def fetch_central_park_tmax(
    start_date: str,
    end_date: str,
    *,
    timeout: float = 120.0,
) -> pd.DataFrame:
    """Download Central Park daily maximum temperature for an optional feature."""
    return fetch_station_observations(
        station=CENTRAL_PARK_STATION,
        data_types=("TMAX",),
        start_date=start_date,
        end_date=end_date,
        timeout=timeout,
    )
