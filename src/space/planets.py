"""Parsers for planetary approaches, discoveries, perihelia, and ephemerides.

JPL CAD body codes and distance cutoffs are fixed here so the fetch and the
report agree. Mercury, Venus, and Mars use 0.05 AU. Jupiter uses 1 AU because
0.05 AU barely occurs in the public catalog. Saturn, Uranus, Neptune, and
Pluto are queried at 10 AU and are not trained: wider cutoffs do not add rows,
and Neptune and Pluto come back empty.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

import pandas as pd

JD_UNIX_EPOCH = 2_440_587.5
MIN_PERIOD_DAYS = 100.0
MAX_PERIOD_DAYS = 200 * 365.25
EARLIEST_DISCOVERY = pd.Timestamp("1700-01-01")
MOON_SOURCE = "https://ssd.jpl.nasa.gov/sats/discovery.html"

SOLAR_SYSTEM_NOTE = (
    "Close-approach counts are from the JPL CAD catalog and include approaches "
    "that happened before an object was discovered, so the history rises as "
    "surveys improve. Mercury, Venus, and Mars are counted within 0.05 AU. "
    "Jupiter is counted within 1 AU: 0.05 AU has only a handful of Jupiter "
    "encounters since 1900. Saturn, Uranus, Neptune, and Pluto are listed from "
    "a 10 AU query and are not trained. Saturn and Uranus have a few public "
    "rows at every cutoff of 2 AU and wider, and Neptune and Pluto have none. "
    "Discovery counts by orbit class measure survey effort. The discovery "
    "regression uses the number of objects of that class known before the "
    "month. Comet perihelia in the training series are passages stepped from "
    "each comet's osculating perihelion and period, kept only when first_obs "
    "is on or before the passage. Periods outside 100 days to 200 years "
    "contribute only the single osculating passage. Earth-planet distances are "
    "JPL Horizons ephemerides; the network is a statistical extrapolation and "
    "is scored against that ephemeris on the outlook months. CAD times are "
    "TDB. Horizons observer times are UT. Moon discovery years come from the "
    "JPL satellite table and are not trained, because only a year is known. "
    "The IMO shower calendar was not a usable ZHR table when this feed was "
    "built, so no shower forecast is invented."
)

SHOWER_NOTE = (
    "The IMO meteor shower calendar (imo.net/resources/calendar) was a "
    "site-restoration page, and the linked shower-calendar PDF URLs returned "
    "HTML rather than a ZHR table. No shower calendar is invented here."
)

TIME_SCALE_NOTE = (
    "CAD close-approach times are TDB calendar dates. Horizons observer times "
    "are UT (UTC after 1962)."
)


@dataclass(frozen=True)
class Planet:
    """One CAD body and the Horizons command that tracks it."""

    code: str
    name: str
    column: str
    horizons_command: str
    dist_max_au: float
    train: bool
    inner: bool


PLANETS: tuple[Planet, ...] = (
    Planet("Merc", "Mercury", "mercury", "199", 0.05, True, True),
    Planet("Venus", "Venus", "venus", "299", 0.05, True, True),
    Planet("Mars", "Mars", "mars", "499", 0.05, True, False),
    Planet("Juptr", "Jupiter", "jupiter", "599", 1.0, True, False),
    Planet("Satrn", "Saturn", "saturn", "699", 10.0, False, False),
    Planet("Urnus", "Uranus", "uranus", "799", 10.0, False, False),
    Planet("Neptn", "Neptune", "neptune", "899", 10.0, False, False),
    Planet("Pluto", "Pluto", "pluto", "999", 10.0, False, False),
)

# Orbit classes from the SBDB filter list. Main belt is inner, central, and
# outer belt. MCA is Mars-crossing and is not included. Comets are sb-kind=c,
# not the small unclassified COM class.
DISCOVERY_SPECS: tuple[dict[str, Any], ...] = (
    {"column": "neo", "classes": ("APO", "AMO", "ATE", "IEO"), "kind": None},
    {"column": "main_belt", "classes": ("IMB", "MBA", "OMB"), "kind": None},
    {"column": "trojan", "classes": ("TJN",), "kind": None},
    {"column": "centaur", "classes": ("CEN",), "kind": None},
    {"column": "tno", "classes": ("TNO",), "kind": None},
    {"column": "comet", "classes": (), "kind": "c"},
)

SOLAR_JPL_KEYS = {
    "mercury_approaches": "mercury",
    "venus_approaches": "venus",
    "mars_approaches": "mars",
    "jupiter_approaches": "jupiter",
    "comet_perihelia": "perihelia",
}

_PLANET_HEADER = re.compile(r"Satellites of\s+(.+?):\s*(\d+)")
_YEAR = re.compile(r"\d{4}")
_HORIZONS_DATE = "%Y-%b-%d %H:%M"


def planet_by_name(name: str) -> Planet:
    """Return the planet record for a display name."""
    for planet in PLANETS:
        if planet.name == name:
            return planet
    raise KeyError(name)


def jd_to_timestamp(jd: float) -> pd.Timestamp:
    """Convert a Julian Date (TDB instant, treated as a civil timestamp) to UTC-naive."""
    seconds = (float(jd) - JD_UNIX_EPOCH) * 86400.0
    return pd.Timestamp("1970-01-01") + pd.to_timedelta(seconds, unit="s")


def _timestamp_to_jd(stamp: pd.Timestamp) -> float:
    """Julian Date for a civil timestamp near 1700–2262."""
    days = (pd.Timestamp(stamp) - pd.Timestamp("1970-01-01")) / pd.Timedelta(days=1)
    return JD_UNIX_EPOCH + float(days)


def perihelion_times(
    tp_jd: float,
    period_days: float | None,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> list[pd.Timestamp]:
    """Perihelion instants in ``[start, end)``.

    A period from 100 days through 200 years is stepped from the osculating
    perihelion. Anything shorter, longer, or missing contributes that single
    instant when it falls in the window. SBDB stores one osculating perihelion,
    not a historical list. The step is done in Julian Date so a perihelion
    centuries outside the window does not overflow ``Timedelta``.
    """
    if not math.isfinite(tp_jd):
        return []
    start_jd = _timestamp_to_jd(pd.Timestamp(start))
    end_jd = _timestamp_to_jd(pd.Timestamp(end))
    period = (
        None
        if period_days is None or not math.isfinite(period_days)
        else float(period_days)
    )
    if period is None or period < MIN_PERIOD_DAYS or period > MAX_PERIOD_DAYS:
        if start_jd <= tp_jd < end_jd:
            return [jd_to_timestamp(tp_jd)]
        return []
    first = math.ceil((start_jd - tp_jd) / period - 1e-9)
    last = math.floor((end_jd - tp_jd) / period - 1e-9)
    if last < first:
        return []
    if last - first > 20_000:
        raise ValueError("perihelion step count is unreasonable")
    times: list[pd.Timestamp] = []
    for step_index in range(first, last + 1):
        instant = tp_jd + step_index * period
        if start_jd <= instant < end_jd:
            times.append(jd_to_timestamp(instant))
    return times


def passage_was_known(passage: pd.Timestamp, first_obs: object) -> bool:
    """True when ``first_obs`` is a real date on or before the passage."""
    stamp = pd.to_datetime(first_obs, errors="coerce")
    if pd.isna(stamp):
        return False
    return bool(pd.Timestamp(stamp) <= pd.Timestamp(passage))


def valid_discovery_dates(
    first_obs: pd.Series, *, through: pd.Timestamp
) -> tuple[pd.Series, int]:
    """Dated discoveries strictly before ``through``.

    Incomplete SBDB dates such as ``YYYY-??-??`` become missing and are
    counted in the dropped total instead of being forced onto January.
    Dates before 1700 are treated as parse failures.
    """
    parsed = pd.to_datetime(first_obs, errors="coerce")
    dropped = int(parsed.isna().sum())
    dated = parsed.dropna()
    too_early = dated < EARLIEST_DISCOVERY
    dropped += int(too_early.sum())
    usable = dated.loc[~too_early & (dated < pd.Timestamp(through))]
    return usable.reset_index(drop=True), dropped


def monthly_count_frame(
    columns: dict[str, pd.Series], *, through: pd.Timestamp
) -> pd.DataFrame:
    """Month-start counts and a ``{column}_prev`` catalog size before the month.

    ``columns`` maps a series name to discovery or passage timestamps. The
    index runs from the earliest month through the last month before
    ``through``. An empty input raises.
    """
    usable = {name: pd.to_datetime(stamps).dropna() for name, stamps in columns.items()}
    present = [stamps for stamps in usable.values() if not stamps.empty]
    if not present:
        raise ValueError("no dated rows for a monthly count frame")
    start = min(stamps.min() for stamps in present).to_period("M").to_timestamp()
    end = (pd.Timestamp(through) - pd.Timedelta(days=1)).to_period("M").to_timestamp()
    if end < start:
        raise ValueError("monthly count frame ends before it starts")
    index = pd.date_range(start, end, freq="MS")
    frame = pd.DataFrame({"date": index})
    for name, stamps in usable.items():
        months = stamps.dt.to_period("M").dt.to_timestamp()
        counts = months.value_counts().reindex(index, fill_value=0).astype(int)
        frame[name] = counts.to_numpy()
        frame[f"{name}_prev"] = (
            counts.cumsum().shift(1).fillna(0).astype(int).to_numpy()
        )
    return frame


def approach_month_frame(
    events: pd.DataFrame,
    *,
    through: pd.Timestamp,
    known_neos_prev: pd.Series,
) -> pd.DataFrame:
    """Monthly counts for the trained planets, plus NEOs known before the month."""
    trained = [planet for planet in PLANETS if planet.train]
    cutoff = pd.Timestamp(through)
    historical = events.loc[pd.to_datetime(events["cd"]) < cutoff].copy()
    if historical.empty:
        raise ValueError("no historical planetary approaches before the open month")
    historical["cd"] = pd.to_datetime(historical["cd"])
    historical["month"] = historical["cd"].dt.to_period("M").dt.to_timestamp()
    end = (cutoff - pd.Timedelta(days=1)).to_period("M").to_timestamp()
    start = pd.Timestamp(historical["month"].min())
    index = pd.date_range(start, end, freq="MS")
    frame = pd.DataFrame({"date": index})
    for planet in trained:
        chosen = historical.loc[historical["body"] == planet.name, "month"]
        counts = chosen.value_counts().reindex(index, fill_value=0)
        frame[planet.column] = counts.astype(int).to_numpy()
    aligned = known_neos_prev.copy()
    aligned.index = pd.to_datetime(aligned.index)
    carried = aligned.reindex(aligned.index.union(index)).sort_index().ffill()
    frame["known_neos_prev"] = carried.reindex(index).fillna(0).astype(int).to_numpy()
    return frame


def parse_horizons_table(text: str) -> pd.DataFrame:
    """Parse a Horizons ``$$SOE`` block.

    Quantity 20 is observer range (AU) and range rate. Quantity 23, when
    present, is the Sun-Observer-Target elongation in degrees plus ``/T``
    (trailing) or ``/L`` (leading). A sun-centered range block has no
    elongation columns.
    """
    if "$$SOE" not in text or "$$EOE" not in text:
        raise ValueError("Horizons result has no ephemeris table.")
    block = text.split("$$SOE", 1)[1].split("$$EOE", 1)[0]
    rows: list[dict[str, object]] = []
    for line in block.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            stamp = pd.to_datetime(f"{parts[0]} {parts[1]}", format=_HORIZONS_DATE)
            delta = float(parts[2])
            deldot = float(parts[3])
        except (TypeError, ValueError):
            continue
        elongation = math.nan
        side = ""
        if len(parts) >= 5:
            elongation = float(parts[4])
        if len(parts) >= 6 and str(parts[5]).startswith("/"):
            side = str(parts[5])[1:]
        rows.append(
            {
                "date": stamp,
                "delta_au": delta,
                "deltadot": deldot,
                "elongation_deg": elongation,
                "elongation_side": side,
            }
        )
    if not rows:
        raise ValueError("Horizons table had no rows.")
    return pd.DataFrame(rows)


def alignment_events(frame: pd.DataFrame, *, inner: bool) -> list[dict[str, Any]]:
    """Interior elongation extrema.

    Outer planets: a local maximum at or above 150 degrees is an opposition,
    and a local minimum at or below 20 degrees is a conjunction. Mercury and
    Venus do not have oppositions; a local maximum at or above 15 degrees is
    a greatest elongation, and a local minimum at or below 10 degrees is a
    conjunction. Window-edge samples are ignored so a cutoff cannot look like
    a peak.
    """
    if frame.empty or "elongation_deg" not in frame.columns:
        return []
    ordered = frame.sort_values("date").reset_index(drop=True)
    elongation = pd.to_numeric(ordered["elongation_deg"], errors="coerce")
    events: list[dict[str, Any]] = []
    for index in range(1, len(ordered) - 1):
        mid = float(elongation.iloc[index])
        left = float(elongation.iloc[index - 1])
        right = float(elongation.iloc[index + 1])
        if not (math.isfinite(mid) and math.isfinite(left) and math.isfinite(right)):
            continue
        is_max = mid > left and mid >= right
        is_min = mid < left and mid <= right
        kind: str | None = None
        if inner:
            if is_max and mid >= 15.0:
                kind = "greatest elongation"
            elif is_min and mid <= 10.0:
                kind = "conjunction"
        elif is_max and mid >= 150.0:
            kind = "opposition"
        elif is_min and mid <= 20.0:
            kind = "conjunction"
        if kind is None:
            continue
        side = str(ordered.iloc[index].get("elongation_side") or "")
        events.append(
            {
                "date": pd.Timestamp(ordered.iloc[index]["date"]).strftime("%Y-%m-%d"),
                "kind": kind,
                "elongation_deg": round(mid, 4),
                "side": side or None,
            }
        )
    return events


class _DiscoveryParser(HTMLParser):
    """Collect satellite rows under each ``sat-discovery-planet`` header."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.in_planet = False
        self.in_cell = False
        self.cells: list[str] = []
        self._buffer: list[str] = []
        self.planet = ""
        self.header_counts: dict[str, int] = {}
        self.rows: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = {key: value or "" for key, value in attrs}
        if tag == "tr":
            self.in_planet = (
                "sat-discovery-planet" in attributes.get("class", "").split()
            )
            self.cells = []
        elif tag == "td":
            self.in_cell = True
            self._buffer = []

    def handle_data(self, data: str) -> None:
        if self.in_cell:
            self._buffer.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.in_cell:
            text = re.sub(r"\s+", " ", "".join(self._buffer)).strip()
            self.cells.append(text)
            self.in_cell = False
        elif tag == "tr":
            self._finish_row()

    def _finish_row(self) -> None:
        if self.in_planet:
            match = _PLANET_HEADER.search(" ".join(self.cells))
            if match:
                name = match.group(1).strip()
                if name.startswith("Dwarf Planet "):
                    name = name.removeprefix("Dwarf Planet ")
                self.planet = name
                self.header_counts[name] = int(match.group(2))
            self.in_planet = False
        elif self.planet and len(self.cells) >= 4:
            year_match = _YEAR.search(self.cells[3])
            if year_match:
                self.rows.append(
                    {
                        "planet": self.planet,
                        "iau_number": self.cells[0],
                        "name": self.cells[1],
                        "designation": self.cells[2],
                        "year": int(year_match.group(0)),
                    }
                )
        self.cells = []


def parse_moon_discoveries(html: str) -> dict[str, Any]:
    """Planet counts and one row per satellite from the JPL discovery table.

    A year cell such as ``1975, 2000`` keeps the first year, the original
    discovery, so a later recovery is not a second moon.
    """
    parser = _DiscoveryParser()
    parser.feed(html)
    parser.close()
    by_planet = [
        {
            "planet": planet,
            "count": sum(1 for row in parser.rows if row["planet"] == planet),
            "listed": listed,
        }
        for planet, listed in parser.header_counts.items()
    ]
    return {"by_planet": by_planet, "discoveries": parser.rows}


def comet_passages(
    catalog: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    known_only: bool,
) -> pd.DataFrame:
    """Stepped perihelia for each catalog comet inside ``[start, end)``."""
    columns = ["date", "designation", "name", "period_days", "known"]
    if catalog.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for record in catalog.to_dict(orient="records"):
        tp = pd.to_numeric(record.get("tp"), errors="coerce")
        period = pd.to_numeric(record.get("per"), errors="coerce")
        period_value = None if pd.isna(period) else float(period)
        if pd.isna(tp):
            continue
        times = perihelion_times(float(tp), period_value, start, end)
        name = str(record.get("full_name") or "").strip()
        designation = str(record.get("pdes") or "").strip()
        for instant in times:
            known = passage_was_known(instant, record.get("first_obs"))
            if known_only and not known:
                continue
            rows.append(
                {
                    "date": instant,
                    "designation": designation,
                    "name": name or designation,
                    "period_days": period_value,
                    "known": known,
                }
            )
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    return frame.sort_values("date").reset_index(drop=True)
