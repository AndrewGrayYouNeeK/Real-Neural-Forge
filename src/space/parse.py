"""Pure parsers for the public space-weather and space-catalog files.

Network access stays in ``src.space.fetch``. These functions turn the
upstream text into regular time series and never keep the SILSO sunspot
column that rides along in the geomagnetic file.
"""

from __future__ import annotations

import io
import re
from collections.abc import Iterable

import numpy as np
import pandas as pd

# NOAA/CelesTrak store Kp in tenths (43 means Kp = 4.3, including the
# minus/zero/plus thirds). A numeric Kp of 5.0 is the 5o bin, not 5-.
KP_TENTHS = 10.0

# OMNI2 daily-average words (1-based in omni2.text). Fill values from that note.
_OMNI_DENSITY_INDEX = 23  # word 24, fill 999.9
_OMNI_SPEED_INDEX = 24  # word 25, fill 9999.
_OMNI_DST_INDEX = 40  # word 41, fill 99999
_OMNI_PLASMA_HOURS_INDEX = 7  # word 8
_DENSITY_FILL = 900.0
_SPEED_FILL = 9000.0
_DST_FILL = 9000.0

_XRS_CLASS = re.compile(r"\b([BCMX])\s+(\d{1,3})\s+G")
_ORBITAL_LAUNCH_PREFIXES = frozenset({"O", "D"})


def _as_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.normalize()


def parse_space_weather_csv(text: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a CelesTrak ``SW-All.csv`` file into Kp/Ap and F10.7 frames.

    Kp/Ap rows are the days with all eight 3-hour Kp values. Those values are
    the GFZ index. F10.7 rows are observed or the few officially interpolated
    days (``OBS`` and ``INT``). Predicted flux (``PRD``, ``PRM``) is dropped.
    The international sunspot number column is not returned.
    """
    frame = pd.read_csv(io.StringIO(text))
    required = [
        "DATE",
        "KP1",
        "KP2",
        "KP3",
        "KP4",
        "KP5",
        "KP6",
        "KP7",
        "KP8",
        "AP_AVG",
        "F10.7_OBS",
        "F10.7_DATA_TYPE",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Space-weather CSV is missing columns: {missing}")

    kp_columns = [f"KP{i}" for i in range(1, 9)]
    working = frame.copy()
    working["date"] = _as_dates(working["DATE"])
    for column in [*kp_columns, "AP_AVG", "F10.7_OBS"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")

    kp_ready = working.dropna(subset=[*kp_columns, "AP_AVG"]).copy()
    if kp_ready.empty:
        raise ValueError("Space-weather CSV has no complete Kp days.")
    raw_max = float(kp_ready[kp_columns].max().max())
    if raw_max > 9.0:
        kp_ready[kp_columns] = kp_ready[kp_columns] / KP_TENTHS
    kp = pd.DataFrame(
        {
            "date": kp_ready["date"],
            "kp_max": kp_ready[kp_columns].max(axis=1),
            "ap": kp_ready["AP_AVG"],
        }
    )
    kp = kp.sort_values("date").drop_duplicates("date", keep="last")

    flux_type = working["F10.7_DATA_TYPE"].astype(str).str.strip().str.upper()
    flux = working.loc[flux_type.isin(["OBS", "INT"])].copy()
    flux = flux.loc[flux["F10.7_OBS"] > 0].copy()
    if flux.empty:
        raise ValueError("Space-weather CSV has no observed F10.7 values.")
    f107 = pd.DataFrame({"date": flux["date"], "f107": flux["F10.7_OBS"]})
    f107 = f107.sort_values("date").drop_duplicates("date", keep="last")
    return (
        kp.reset_index(drop=True),
        f107.reset_index(drop=True),
    )


def parse_omni_daily(
    text: str, *, min_plasma_hours: int = 4
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse NASA OMNI2 daily averages into solar-wind and Dst frames.

    ``omni_01_av.dat`` uses the hourly OMNI2 layout. Solar-wind speed and
    density are kept only when the daily mean used at least
    ``min_plasma_hours`` hourly samples and the values are not fill codes.
    Dst is kept on its own, including days with no plasma coverage. Dst in
    this file is the daily mean, not the daily minimum.
    """
    if min_plasma_hours < 1:
        raise ValueError("min_plasma_hours must be positive")
    wind_rows: list[tuple[pd.Timestamp, float, float]] = []
    dst_rows: list[tuple[pd.Timestamp, float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) <= _OMNI_DST_INDEX:
            continue
        try:
            year = int(parts[0])
            doy = int(parts[1])
            stamp = pd.to_datetime(f"{year:04d}{doy:03d}", format="%Y%j")
        except ValueError:
            continue
        dst = float(parts[_OMNI_DST_INDEX])
        if dst < _DST_FILL:
            dst_rows.append((stamp, dst))
        if len(parts) <= _OMNI_SPEED_INDEX:
            continue
        try:
            plasma_hours = int(float(parts[_OMNI_PLASMA_HOURS_INDEX]))
        except ValueError:
            continue
        density = float(parts[_OMNI_DENSITY_INDEX])
        speed = float(parts[_OMNI_SPEED_INDEX])
        if (
            plasma_hours >= min_plasma_hours
            and density < _DENSITY_FILL
            and speed < _SPEED_FILL
        ):
            wind_rows.append((stamp, speed, density))
    if not wind_rows:
        raise ValueError("OMNI daily file has no usable solar-wind days.")
    if not dst_rows:
        raise ValueError("OMNI daily file has no usable Dst days.")
    wind = pd.DataFrame(wind_rows, columns=["date", "speed", "density"])
    dst = pd.DataFrame(dst_rows, columns=["date", "dst"])
    wind = wind.sort_values("date").drop_duplicates("date", keep="last")
    dst = dst.sort_values("date").drop_duplicates("date", keep="last")
    return wind.reset_index(drop=True), dst.reset_index(drop=True)


def parse_goes_xrs_reports(texts: Iterable[str]) -> pd.DataFrame:
    """Count B, C, M, and X class flares per month from NCEI GOES XRS reports.

    Each report line starts with a code whose last six digits are ``YYMMDD``.
    Years 70–99 are 1970–1999 and 00–69 are 2000–2069, which covers the
    GOES record beginning in 1975. Months inside the catalog with no flares
    are explicit zeros. A trailing month that does not reach its last day is
    dropped because the file may have stopped mid-month.
    """
    events: list[tuple[pd.Timestamp, str]] = []
    for text in texts:
        for line in text.splitlines():
            parts = line.split()
            if not parts or not parts[0].isdigit() or len(parts[0]) < 6:
                continue
            ymd = parts[0][-6:]
            yy = int(ymd[:2])
            month = int(ymd[2:4])
            day = int(ymd[4:6])
            if not (1 <= month <= 12 and 1 <= day <= 31):
                continue
            year = 1900 + yy if yy >= 70 else 2000 + yy
            match = _XRS_CLASS.search(line)
            if match is None:
                continue
            try:
                stamp = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            events.append((stamp, match.group(1)))
    if not events:
        raise ValueError("GOES XRS reports contained no flare classes.")
    events_frame = pd.DataFrame(events, columns=["date", "flare_class"])
    events_frame["month"] = events_frame["date"].dt.to_period("M").dt.to_timestamp()
    counts = events_frame.groupby(["month", "flare_class"]).size().unstack(fill_value=0)
    for column in ("B", "C", "M", "X"):
        if column not in counts.columns:
            counts[column] = 0
    monthly = counts[["B", "C", "M", "X"]].rename(
        columns={
            "B": "b_class",
            "C": "c_class",
            "M": "m_class",
            "X": "x_class",
        }
    )
    monthly.index = pd.DatetimeIndex(monthly.index)
    full_index = pd.date_range(monthly.index.min(), monthly.index.max(), freq="MS")
    monthly = monthly.reindex(full_index, fill_value=0)
    last_event = pd.Timestamp(events_frame["date"].max())
    month_end = last_event + pd.offsets.MonthEnd(0)
    if last_event.normalize() < pd.Timestamp(month_end).normalize():
        monthly = monthly.iloc[:-1]
    if monthly.empty:
        raise ValueError(
            "GOES XRS monthly series is empty after dropping a partial month."
        )
    monthly.index.name = "date"
    return monthly.reset_index()


def parse_gcat_launches(text: str) -> pd.DataFrame:
    """Monthly orbital launch attempts from a GCAT ``launch.tsv`` file.

    McDowell's ``LaunchCode`` uses ``O`` for orbital and ``D`` for deep space.
    Both are launch attempts that target orbit (deep-space flights leave Earth
    orbit). The second character is the success fraction (``S``, ``F``, ``U``,
    and the rarer pad codes); failures stay in the count. Suborbital (``S``),
    missile (``M``), atmospheric (``A``), test (``T``), human suborbital
    (``Y``), and near-orbital (``N``) rows are excluded.
    """
    header: list[str] | None = None
    dates: list[pd.Timestamp] = []
    for line in text.splitlines():
        if line.startswith("#Launch_Tag"):
            header = line[1:].split("\t")
            continue
        if not line or line.startswith("#"):
            continue
        if header is None:
            raise ValueError("GCAT launch file has no #Launch_Tag header.")
        parts = line.split("\t")
        if len(parts) < len(header):
            parts = [*parts, *[""] * (len(header) - len(parts))]
        record = dict(zip(header, parts[: len(header)], strict=True))
        code = record.get("LaunchCode", "").strip()
        if not code or code[0] not in _ORBITAL_LAUNCH_PREFIXES:
            continue
        dates.append(_parse_gcat_date(record.get("Launch_Date", "")))
    if not dates:
        raise ValueError("GCAT launch file has no orbital attempts.")
    return _monthly_event_counts(pd.Series(dates), "attempts")


def parse_close_approaches(payload: dict[str, object]) -> pd.DataFrame:
    """Monthly counts of SBDB close approaches from a CAD API JSON body."""
    fields = payload.get("fields")
    data = payload.get("data")
    if not isinstance(fields, list) or not isinstance(data, list):
        raise ValueError("CAD payload is missing fields or data.")
    if "cd" not in fields:
        raise ValueError("CAD payload has no cd (calendar date) field.")
    cd_index = fields.index("cd")
    stamps: list[pd.Timestamp] = []
    for row in data:
        if not isinstance(row, list) or cd_index >= len(row):
            continue
        stamps.append(pd.to_datetime(str(row[cd_index]), format="%Y-%b-%d %H:%M"))
    if not stamps:
        raise ValueError("CAD payload has no close-approach times.")
    return _monthly_event_counts(pd.Series(stamps), "approaches")


def parse_neo_discoveries(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Monthly counts of NEOs by SBDB ``first_obs``.

    ``first_obs`` is the first observation in the orbit fit. For NEOs that is
    the usual discovery date, and the counts mostly track survey effort.
    Duplicate designations across orbit classes are counted once.
    """
    parts = [frame for frame in frames if not frame.empty]
    if not parts:
        raise ValueError("SBDB discovery frames are empty.")
    combined = pd.concat(parts, ignore_index=True)
    if "pdes" not in combined.columns or "first_obs" not in combined.columns:
        raise ValueError("SBDB frames need pdes and first_obs columns.")
    combined = combined.drop_duplicates("pdes", keep="first")
    dates = pd.to_datetime(combined["first_obs"], errors="coerce")
    dates = dates.dropna()
    if dates.empty:
        raise ValueError("SBDB frames have no parseable first_obs dates.")
    return _monthly_event_counts(dates, "discoveries")


def parse_exoplanet_years(text: str) -> pd.DataFrame:
    """Yearly planet counts from an Exoplanet Archive TAP CSV.

    Accepts either an aggregated ``disc_year,n`` table or one row per planet.
    Missing years between the first and last discovery year are zeros.
    """
    frame = pd.read_csv(io.StringIO(text))
    if "disc_year" not in frame.columns:
        raise ValueError("Exoplanet table has no disc_year column.")
    years = pd.to_numeric(frame["disc_year"], errors="coerce")
    if "n" in frame.columns:
        counts = pd.to_numeric(frame["n"], errors="coerce")
        yearly = pd.DataFrame({"year": years, "discoveries": counts}).dropna()
    else:
        yearly = (
            years.dropna()
            .astype(int)
            .value_counts()
            .rename_axis("year")
            .reset_index(name="discoveries")
        )
    if yearly.empty:
        raise ValueError("Exoplanet table has no discovery years.")
    yearly["year"] = yearly["year"].astype(int)
    yearly = yearly.groupby("year", as_index=False)["discoveries"].sum()
    full = pd.DataFrame(
        {"year": np.arange(int(yearly["year"].min()), int(yearly["year"].max()) + 1)}
    )
    yearly = full.merge(yearly, on="year", how="left").fillna({"discoveries": 0})
    yearly["date"] = pd.to_datetime(yearly["year"].astype(str) + "-01-01")
    return yearly.loc[:, ["date", "discoveries"]].reset_index(drop=True)


def regularize_daily(
    frame: pd.DataFrame,
    columns: list[str],
    *,
    max_gap_days: int = 3,
) -> tuple[pd.DataFrame, int]:
    """Return the latest daily stretch whose gaps are at most ``max_gap_days``.

    Longer gaps start a new segment, and the returned frame is the segment
    after the last such gap. Shorter holes are linear interpolations. The
    second value is how many days were filled.
    """
    if max_gap_days < 0:
        raise ValueError("max_gap_days must be >= 0")
    if "date" not in frame.columns:
        raise ValueError("daily frame needs a date column")
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"daily frame is missing columns: {missing}")
    df = frame.loc[:, ["date", *columns]].copy()
    df["date"] = _as_dates(df["date"])
    df = df.dropna(subset=columns)
    df = df.sort_values("date").drop_duplicates("date", keep="last")
    if df.empty:
        raise ValueError("daily frame has no finite rows")
    day_values = df["date"].to_numpy(dtype="datetime64[D]")
    gaps = np.diff(day_values).astype(np.int64)
    breaks = np.flatnonzero(gaps > max_gap_days + 1)
    start_pos = int(breaks[-1] + 1) if len(breaks) else 0
    segment = df.iloc[start_pos:].reset_index(drop=True)
    grid = pd.DataFrame(
        {
            "date": pd.date_range(
                segment["date"].iloc[0], segment["date"].iloc[-1], freq="D"
            )
        }
    )
    merged = grid.merge(segment, on="date", how="left")
    filled = int(merged[columns].isna().any(axis=1).sum())
    merged[columns] = merged[columns].interpolate(method="linear", limit=max_gap_days)
    if bool(merged[columns].isna().any().any()):
        raise ValueError("a gap remained after daily interpolation")
    deltas = np.diff(merged["date"].to_numpy(dtype="datetime64[D]")).astype(np.int64)
    if len(deltas) and not bool(np.all(deltas == 1)):
        raise ValueError("regularized daily series is not contiguous")
    return merged.reset_index(drop=True), filled


def trim_open_period(
    frame: pd.DataFrame,
    freq: str,
    *,
    today: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Drop the current, still-open day, month, or year."""
    if "date" not in frame.columns:
        raise ValueError("frame needs a date column")
    stamp = (
        pd.Timestamp.today().normalize()
        if today is None
        else pd.Timestamp(today).normalize()
    )
    dated = frame.copy()
    dated["date"] = _as_dates(dated["date"])
    if freq == "D":
        keep = dated["date"] < stamp
    elif freq == "MS":
        keep = dated["date"] < stamp.replace(day=1)
    elif freq == "YS":
        keep = dated["date"] < pd.Timestamp(year=int(stamp.year), month=1, day=1)
    else:
        raise ValueError(f"unsupported frequency {freq!r}")
    trimmed = dated.loc[keep].reset_index(drop=True)
    if trimmed.empty:
        raise ValueError("trimming the open period removed every row")
    return trimmed


def align_on_dates(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """Inner-join two daily frames on ``date`` and keep a contiguous suffix."""
    merged = left.merge(right, on="date", how="inner")
    if merged.empty:
        raise ValueError("daily frames have no dates in common")
    value_columns = [column for column in merged.columns if column != "date"]
    aligned, _filled = regularize_daily(merged, value_columns, max_gap_days=3)
    return aligned


def _parse_gcat_date(value: str) -> pd.Timestamp:
    """Parse a GCAT ``Launch_Date``.

    A trailing ``?`` marks a vague clock time or day (``2012 May 23?``,
    ``1964 Jun  6 0600?``). The day is kept when it is numeric; otherwise the
    attempt is counted on the first of the month.
    """
    parts = value.replace("?", "").split()
    if len(parts) < 2:
        raise ValueError(f"unparseable GCAT launch date {value!r}")
    day = parts[2] if len(parts) >= 3 and parts[2].isdigit() else "1"
    return pd.Timestamp(f"{parts[0]} {parts[1]} {day}")


def _monthly_event_counts(dates: pd.Series, column: str) -> pd.DataFrame:
    stamps = pd.to_datetime(dates)
    months = stamps.dt.to_period("M").dt.to_timestamp()
    counts = months.value_counts().sort_index()
    full_index = pd.date_range(counts.index.min(), counts.index.max(), freq="MS")
    series = counts.reindex(full_index, fill_value=0).astype(int)
    last = pd.Timestamp(stamps.max())
    month_end = last + pd.offsets.MonthEnd(0)
    if last.normalize() < pd.Timestamp(month_end).normalize():
        series = series.iloc[:-1]
    if series.empty:
        raise ValueError("monthly event series is empty")
    return pd.DataFrame({"date": series.index, column: series.to_numpy()})
