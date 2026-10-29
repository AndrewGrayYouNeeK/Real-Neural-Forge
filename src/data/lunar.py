"""Offline lunar calendar features for any dated table.

The Moon and Sun positions are Jean Meeus's truncated series (Astronomical
Algorithms), evaluated locally. Nothing here calls an astronomy service.
Chinese dates follow the winter-solstice rule in China Standard Time
(UTC+8, no daylight saving): a month begins on the civil date of the new
moon, month 11 contains the winter solstice, and a thirteenth month is the
first month in that span with no major solar term.

Eclipse flags are a geometric test, not a catalog. A new moon within
``SOLAR_ECLIPSE_LATITUDE_DEG`` of the ecliptic, or a full moon within
``LUNAR_ECLIPSE_LATITUDE_DEG``, is marked. Grazing and penumbral edges can
disagree with a published bulletin. A supermoon is a full moon whose
center distance is under ``PERIGEE_DISTANCE_KM`` (360,000 km). ``near_perigee``
uses that same distance at the requested clock time, full moon or not.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

SYNODIC_MONTH_DAYS = 29.530588853
J2000 = 2451545.0
# Refined later; this is only the search seed (Meeus mean new moon).
_NEW_MOON_SEED = 2451550.25972
_SOLSTICE_SEED = 2451899.068  # 2000-12-21 ~13:37 UTC, winter solstice
CHINA = timezone(timedelta(hours=8))
PERIGEE_DISTANCE_KM = 360_000.0
# Geometric limits. Tuned so the 2024 eclipse season is flagged and the
# neighboring non-eclipse syzygies are not. See tests/test_lunar.py.
SOLAR_ECLIPSE_LATITUDE_DEG = 1.45
LUNAR_ECLIPSE_LATITUDE_DEG = 1.55
MEAN_DISTANCE_KM = 385_000.56

PHASE_NAMES: tuple[str, ...] = (
    "new",
    "waxing crescent",
    "first quarter",
    "waxing gibbous",
    "full",
    "waning gibbous",
    "last quarter",
    "waning crescent",
)
ZODIAC: tuple[str, ...] = (
    "Rat",
    "Ox",
    "Tiger",
    "Rabbit",
    "Dragon",
    "Snake",
    "Horse",
    "Goat",
    "Monkey",
    "Rooster",
    "Dog",
    "Pig",
)

# Numeric columns a model can consume. Names stay stable for checkpoints.
LUNAR_MODEL_COLUMNS: tuple[str, ...] = (
    "moon_phase",
    "illumination",
    "days_since_new",
    "phase_sin",
    "phase_cos",
    "distance_km",
    "lunar_month",
    "lunar_day",
    "is_leap_month",
    "zodiac_index",
    "is_new_moon",
    "is_full_moon",
    "near_perigee",
    "is_supermoon",
    "eclipse_solar",
    "eclipse_lunar",
)

# Meeus Table 47.A, units of 1e-6 degree (longitude and latitude) and 0.001 km
# (distance). Arguments are (D, M, M', F). The leading terms dominate phase,
# latitude, and the perigee distance; the rest pull syzygies inside a few minutes.
_LON_TERMS: tuple[tuple[int, int, int, int, float], ...] = (
    (0, 0, 1, 0, 6_288_774),
    (2, 0, -1, 0, 1_274_027),
    (2, 0, 0, 0, 658_314),
    (0, 0, 2, 0, 213_618),
    (0, 1, 0, 0, -185_116),
    (0, 0, 0, 2, -114_332),
    (2, 0, -2, 0, 58_793),
    (2, -1, -1, 0, 57_066),
    (2, 0, 1, 0, 53_322),
    (2, -1, 0, 0, 45_758),
    (0, 1, -1, 0, -40_923),
    (1, 0, 0, 0, -34_720),
    (0, 1, 1, 0, -30_383),
    (2, 0, 0, -2, 15_327),
    (0, 0, 1, 2, -12_528),
    (0, 0, 1, -2, 10_980),
    (4, 0, -1, 0, 10_675),
    (0, 0, 3, 0, 10_034),
    (4, 0, -2, 0, 8_548),
    (2, 1, -1, 0, -7_888),
    (2, 1, 0, 0, -6_766),
    (1, 0, -1, 0, -5_163),
    (1, 1, 0, 0, 4_987),
    (2, -1, 1, 0, 4_036),
    (2, 0, 2, 0, 3_994),
    (4, 0, 0, 0, 3_861),
    (2, 0, -3, 0, 3_665),
    (0, 1, -2, 0, -2_689),
    (2, 0, -1, 2, -2_602),
    (2, -1, -2, 0, 2_390),
    (1, 0, 1, 0, -2_348),
    (2, -2, 0, 0, 2_236),
    (0, 1, 2, 0, -2_120),
    (0, 2, 0, 0, -2_062),
    (2, -2, -1, 0, 2_048),
    (2, 0, 1, -2, -1_773),
    (2, 0, 0, 2, -1_595),
    (4, -1, -1, 0, 1_217),
    (0, 0, 2, 2, -1_113),
    (3, 0, -1, 0, -892),
    (2, 1, 1, 0, -810),
    (4, -1, -2, 0, 759),
    (0, 2, -1, 0, -713),
    (2, 2, -1, 0, -700),
    (2, 1, -2, 0, 691),
    (2, -1, 0, -2, 596),
    (4, 0, 1, 0, 549),
    (0, 0, 4, 0, 537),
    (4, -1, 0, 0, 520),
    (1, 0, -2, 0, -487),
    (2, -1, 2, 0, -399),
    (0, 0, 2, -2, -381),
    (1, 1, 1, 0, 351),
    (3, 0, -2, 0, -340),
    (4, 0, -3, 0, 330),
    (2, -1, -1, 2, 327),
    (2, 0, 3, 0, 299),
    (2, 0, -1, -2, 294),
)
_LAT_TERMS: tuple[tuple[int, int, int, int, float], ...] = (
    (0, 0, 0, 1, 5_128_122),
    (0, 0, 1, 1, 280_602),
    (0, 0, 1, -1, 277_693),
    (2, 0, 0, -1, 173_237),
    (2, 0, -1, 1, 55_413),
    (2, 0, -1, -1, 46_271),
    (2, 0, 0, 1, 32_573),
    (0, 0, 2, 1, 17_198),
    (2, 0, 1, -1, 9_266),
    (0, 0, 2, -1, 8_822),
    (2, -1, 0, -1, 8_216),
    (2, 0, -2, -1, 4_324),
    (2, 0, 1, 1, 4_200),
    (2, 1, 0, -1, -3_359),
    (2, -1, -1, 1, 2_463),
    (2, -1, 0, 1, 2_211),
    (2, -1, -1, -1, 2_065),
    (0, 1, -1, -1, -1_870),
    (4, 0, -1, -1, 1_828),
    (0, 1, 0, 1, -1_794),
    (0, 0, 0, 3, -1_749),
    (0, 1, -1, 1, -1_565),
    (1, 0, 0, 1, -1_495),
    (0, 1, 1, 1, -1_411),
    (0, 1, 1, -1, -1_344),
    (1, 0, 0, -1, -1_335),
    (0, 0, 3, 1, 1_107),
    (4, 0, 0, -1, 1_021),
    (4, 0, -2, 1, 833),
    (2, 0, 0, -3, 777),
    (2, 0, 2, -1, 671),
    (2, -1, 1, -1, 607),
    (0, 0, 1, 3, 596),
    (2, 0, -2, 1, 491),
    (2, 1, -1, 1, -451),
    (2, 1, 0, 1, 422),
    (2, -1, -2, -1, 421),
    (0, 0, 3, -1, -366),
    (2, -2, 0, -1, -351),
    (2, 1, 1, -1, 331),
    (4, 0, -2, -1, 315),
    (2, -2, -1, 1, 302),
    (0, 0, 1, -3, -283),
    (2, 2, 0, -1, -229),
    (0, 2, 1, 1, 223),
    (2, -1, 1, 1, 223),
)
_DIST_TERMS: tuple[tuple[int, int, int, int, float], ...] = (
    (0, 0, 1, 0, -20_905_355),
    (2, 0, -1, 0, -3_699_111),
    (2, 0, 0, 0, -2_955_968),
    (0, 0, 2, 0, -569_925),
    (2, 0, -2, 0, 246_158),
    (2, -1, 0, 0, -204_586),
    (2, 0, 1, 0, -170_733),
    (2, -1, -1, 0, -152_138),
    (0, 1, -1, 0, -129_620),
    (1, 0, 0, 0, 108_743),
    (0, 1, 1, 0, 104_755),
    (0, 1, 0, 0, 48_888),
    (4, 0, -1, 0, -34_780),
    (0, 0, 3, 0, -34_752),
    (4, 0, -2, 0, -23_210),
    (3, 0, -1, 0, 14_414),
    (2, 0, 0, -2, 10_321),
    (2, 0, -3, 0, 10_056),
    (4, 0, 0, 0, -9_978),
    (2, 0, 2, 0, 8_752),
    (0, 0, 4, 0, -7_718),
    (0, 0, 0, 2, -3_149),
    (1, 0, -1, 0, -3_471),
    (0, 1, -2, 0, 2_491),
    (1, 1, 0, 0, -1_048),
)


def _clean_terms(
    terms: tuple[tuple[int, int, int, int, float], ...], scale: float
) -> np.ndarray:
    rows = [row for row in terms if len(row) == 5]
    table = np.asarray(rows, dtype=np.float64)
    table[:, 4] *= scale
    return table


_LON = _clean_terms(_LON_TERMS, 1e-6)
_LAT = _clean_terms(_LAT_TERMS, 1e-6)
_DIST = _clean_terms(_DIST_TERMS, 1e-3)


def _as_1d(jd: float | np.ndarray) -> tuple[np.ndarray, bool]:
    values = np.asarray(jd, dtype=np.float64)
    scalar = values.ndim == 0
    if scalar:
        values = values.reshape(1)
    if values.ndim != 1:
        raise ValueError("Julian dates must be a scalar or a 1-d array")
    return values, scalar


def _mod360(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.mod(values, 360.0), dtype=np.float64)


def wrap180(degrees: np.ndarray | float) -> np.ndarray:
    """Wrap degrees into (-180, 180]."""
    wrapped = (np.asarray(degrees, dtype=np.float64) + 180.0) % 360.0 - 180.0
    return np.asarray(wrapped, dtype=np.float64)


def julian_day(moment: datetime) -> float:
    """Julian day of a timezone-aware or naive (UTC) instant."""
    if moment.tzinfo is None:
        utc = moment.replace(tzinfo=timezone.utc)
    else:
        utc = moment.astimezone(timezone.utc)
    year = utc.year
    month = utc.month
    day = utc.day
    hour = (
        utc.hour
        + utc.minute / 60.0
        + utc.second / 3600.0
        + utc.microsecond / 3_600_000_000.0
    )
    shift = (14 - month) // 12
    y = year + 4800 - shift
    m = month + 12 * shift - 3
    jdn = day + (153 * m + 2) // 5 + 365 * y + y // 4 - y // 100 + y // 400 - 32045
    return float(jdn) + (hour - 12.0) / 24.0


def datetime_from_jd(jd: float) -> datetime:
    """UTC datetime for a Julian day. Round-trips with :func:`julian_day`."""
    z = math.floor(jd + 0.5)
    f = jd + 0.5 - z
    if z < 2299161:
        a = z
    else:
        alpha = math.floor((z - 1867216.25) / 36524.25)
        a = z + 1 + alpha - math.floor(alpha / 4)
    b = a + 1524
    c = math.floor((b - 122.1) / 365.25)
    d = math.floor(365.25 * c)
    e = math.floor((b - d) / 30.6001)
    day_float = b - d - math.floor(30.6001 * e) + f
    month = e - 1 if e < 14 else e - 13
    year = c - 4716 if month > 2 else c - 4715
    day = math.floor(day_float)
    seconds = (day_float - day) * 86400.0
    # Keep the fractional day from pushing the clock onto the next date twice.
    whole = int(math.floor(seconds + 1e-6))
    if whole >= 86400:
        whole = 86399
    micro = int(round((seconds - whole) * 1_000_000))
    if micro >= 1_000_000:
        micro = 999_999
    return datetime(int(year), int(month), int(day), tzinfo=timezone.utc) + timedelta(
        seconds=whole, microseconds=micro
    )


def _arguments(jd: np.ndarray) -> tuple[np.ndarray, ...]:
    t = (jd - J2000) / 36525.0
    lp = _mod360(
        218.3164477
        + 481267.88123421 * t
        - 0.0015786 * t**2
        + t**3 / 538841.0
        - t**4 / 65_194_000.0
    )
    d = _mod360(
        297.8501921
        + 445267.1114034 * t
        - 0.0018819 * t**2
        + t**3 / 545868.0
        - t**4 / 113_065_000.0
    )
    m = _mod360(
        357.5291092 + 35999.0502909 * t - 0.0001536 * t**2 + t**3 / 24_490_000.0
    )
    mp = _mod360(
        134.9633964
        + 477198.8675055 * t
        + 0.0087414 * t**2
        + t**3 / 69699.0
        - t**4 / 14_712_000.0
    )
    f = _mod360(
        93.2720950
        + 483202.0175233 * t
        - 0.0036539 * t**2
        - t**3 / 3_526_000.0
        + t**4 / 863_310_000.0
    )
    return lp, d, m, mp, f, t


def _series(
    table: np.ndarray,
    d: np.ndarray,
    m: np.ndarray,
    mp: np.ndarray,
    f: np.ndarray,
    *,
    sine: bool,
) -> np.ndarray:
    angle = (
        table[:, 0] * d[:, None]
        + table[:, 1] * m[:, None]
        + table[:, 2] * mp[:, None]
        + table[:, 3] * f[:, None]
    )
    wave = np.sin(np.deg2rad(angle)) if sine else np.cos(np.deg2rad(angle))
    return np.asarray(wave @ table[:, 4], dtype=np.float64)


def sun_longitude(jd: float | np.ndarray) -> np.ndarray:
    """Apparent geocentric ecliptic longitude of the Sun, degrees."""
    values, scalar = _as_1d(jd)
    t = (values - J2000) / 36525.0
    l0 = 280.46646 + 36000.76983 * t + 0.0003032 * t**2
    m = np.deg2rad(357.52911 + 35999.05029 * t - 0.0001537 * t**2)
    c = (
        (1.914602 - 0.004817 * t - 0.000014 * t**2) * np.sin(m)
        + (0.019993 - 0.000101 * t) * np.sin(2.0 * m)
        + 0.000289 * np.sin(3.0 * m)
    )
    omega = np.deg2rad(125.04 - 1934.136 * t)
    lon = _mod360(l0 + c - 0.00569 - 0.00478 * np.sin(omega))
    return lon[0] if scalar else lon


def moon_position(jd: float | np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ecliptic longitude, latitude (degrees), and distance (km)."""
    values, scalar = _as_1d(jd)
    lp, d, m, mp, f, t = _arguments(values)
    lon = lp + _series(_LON, d, m, mp, f, sine=True)
    a1 = np.deg2rad(119.75 + 131.849 * t)
    a2 = np.deg2rad(53.09 + 479264.290 * t)
    a3 = np.deg2rad(313.45 + 481266.484 * t)
    lon = lon + (
        0.003958 * np.sin(a1)
        + 0.001962 * np.sin(np.deg2rad(lp - f))
        + 0.000318 * np.sin(a2)
    )
    lat = _series(_LAT, d, m, mp, f, sine=True)
    lat = lat + (
        -0.002235 * np.sin(np.deg2rad(lp))
        + 0.000382 * np.sin(a3)
        + 0.000175 * np.sin(a1 - np.deg2rad(f))
        + 0.000175 * np.sin(a1 + np.deg2rad(f))
        + 0.000127 * np.sin(np.deg2rad(lp - mp))
        - 0.000115 * np.sin(np.deg2rad(lp + mp))
    )
    dist = MEAN_DISTANCE_KM + _series(_DIST, d, m, mp, f, sine=False)
    lon = _mod360(lon)
    if scalar:
        return lon[0], lat[0], dist[0]
    return lon, lat, dist


def phase_fraction(jd: float | np.ndarray) -> np.ndarray:
    """Synodic phase in ``[0, 1)``. 0 is new moon and 0.5 is full moon."""
    values, scalar = _as_1d(jd)
    lon, _lat, _dist = moon_position(values)
    elong = _mod360(lon - sun_longitude(values))
    phase = elong / 360.0
    return phase[0] if scalar else phase


def illumination_fraction(phase: np.ndarray | float) -> np.ndarray:
    """Illuminated fraction of the lunar disk, from 0 at new to 1 at full."""
    elong = np.asarray(phase, dtype=np.float64) * 2.0 * math.pi
    return (1.0 - np.cos(elong)) / 2.0


def named_phase(phase: float) -> str:
    """One of the eight named phases, centered on the principal phase."""
    shifted = (float(phase) + 1.0 / 16.0) % 1.0
    index = int(math.floor(shifted * 8.0)) % 8
    return PHASE_NAMES[index]


def _refine(jd: np.ndarray, target: float, *, solar: bool) -> np.ndarray:
    guess = np.asarray(jd, dtype=np.float64).copy()
    rate = 0.98564736 if solar else 12.19074912
    for _ in range(12):
        if solar:
            lon = np.asarray(sun_longitude(guess), dtype=np.float64)
            delta = wrap180(lon - target)
        else:
            lon, _lat, _dist = moon_position(guess)
            elong = _mod360(np.asarray(lon) - np.asarray(sun_longitude(guess)))
            delta = wrap180(elong - target)
        guess = guess - delta / rate
    return guess


def _arange_periods(
    start: float, stop: float, origin: float, step: float
) -> np.ndarray:
    k0 = math.floor((start - origin) / step) - 1
    k1 = math.ceil((stop - origin) / step) + 1
    return origin + np.arange(k0, k1 + 1, dtype=np.float64) * step


@dataclass(frozen=True)
class _Syzygy:
    jd: float
    kind: str  # "new" or "full"
    latitude_deg: float
    distance_km: float

    @property
    def utc(self) -> datetime:
        return datetime_from_jd(self.jd)


@dataclass(frozen=True)
class ChineseDate:
    """A civil date on the Chinese lunisolar calendar."""

    year: int
    month: int
    day: int
    leap: bool
    zodiac_index: int

    @property
    def zodiac(self) -> str:
        return ZODIAC[self.zodiac_index]

    def text(self) -> str:
        leap = "leap " if self.leap else ""
        return (
            f"{self.zodiac} year {self.year}, {leap}month {self.month} day {self.day}"
        )


@dataclass(frozen=True)
class _Month:
    start: date
    end: date
    year: int
    month: int
    leap: bool
    zodiac_index: int


@dataclass(frozen=True)
class _Ephemeris:
    new_moons: tuple[_Syzygy, ...]
    full_moons: tuple[_Syzygy, ...]
    months: tuple[_Month, ...]
    month_starts: tuple[date, ...]


def _civil(moment: datetime, tz: timezone | ZoneInfo) -> date:
    return moment.astimezone(tz).date()


@lru_cache(maxsize=4)
def _ephemeris(year_start: int, year_end: int) -> _Ephemeris:
    if year_end < year_start:
        raise ValueError("year_end is before year_start")
    start = julian_day(datetime(year_start, 1, 1, tzinfo=timezone.utc)) - 40.0
    stop = julian_day(datetime(year_end, 12, 31, tzinfo=timezone.utc)) + 40.0
    new_jd = _refine(
        _arange_periods(start, stop, _NEW_MOON_SEED, SYNODIC_MONTH_DAYS),
        0.0,
        solar=False,
    )
    full_jd = _refine(
        _arange_periods(
            start, stop, _NEW_MOON_SEED + SYNODIC_MONTH_DAYS / 2.0, SYNODIC_MONTH_DAYS
        ),
        180.0,
        solar=False,
    )
    solstice_jd = _refine(
        _arange_periods(start, stop, _SOLSTICE_SEED, 365.24219),
        270.0,
        solar=True,
    )
    zhongqi_jd = []
    for angle in range(0, 360, 30):
        zhongqi_jd.append(
            _refine(
                _arange_periods(start, stop, _SOLSTICE_SEED, 365.24219),
                float(angle),
                solar=True,
            )
        )
    zhongqi = np.sort(np.concatenate(zhongqi_jd))

    def pack(jds: np.ndarray, kind: str) -> tuple[_Syzygy, ...]:
        order = np.argsort(jds)
        packed: list[_Syzygy] = []
        previous = -1e9
        for jd in jds[order]:
            value = float(jd)
            if value - previous < 10.0:
                continue
            if value < start or value > stop:
                continue
            _lon, lat, dist = moon_position(value)
            packed.append(
                _Syzygy(
                    jd=value,
                    kind=kind,
                    latitude_deg=float(lat),
                    distance_km=float(dist),
                )
            )
            previous = value
        return tuple(packed)

    new_moons = pack(new_jd, "new")
    full_moons = pack(full_jd, "full")
    months = _label_months(new_moons, solstice_jd, zhongqi)
    return _Ephemeris(
        new_moons=new_moons,
        full_moons=full_moons,
        months=months,
        month_starts=tuple(month.start for month in months),
    )


def _label_months(
    new_moons: tuple[_Syzygy, ...],
    solstice_jd: np.ndarray,
    zhongqi_jd: np.ndarray,
) -> tuple[_Month, ...]:
    starts = [_civil(moon.utc, CHINA) for moon in new_moons]
    # Unique China dates, keeping the first new moon of a civil date.
    kept: list[int] = []
    seen: set[date] = set()
    for index, start in enumerate(starts):
        if start in seen:
            continue
        seen.add(start)
        kept.append(index)
    new_moons = tuple(new_moons[index] for index in kept)
    starts = [starts[index] for index in kept]
    if len(starts) < 3:
        return ()
    zhongqi_dates = sorted(
        {_civil(datetime_from_jd(float(jd)), CHINA) for jd in zhongqi_jd}
    )
    solstices = sorted(
        {
            _civil(datetime_from_jd(float(jd)), CHINA)
            for jd in solstice_jd
            if starts[0] <= _civil(datetime_from_jd(float(jd)), CHINA) < starts[-1]
        }
    )
    m11: list[int] = []
    for solstice in solstices:
        index = bisect_right(starts, solstice) - 1
        if index < 0 or index >= len(starts) - 1:
            continue
        if not m11 or m11[-1] != index:
            m11.append(index)
    labeled: list[_Month] = []
    for left, right in zip(m11, m11[1:], strict=False):
        count = right - left
        if count not in (12, 13):
            continue
        has_term = []
        for index in range(left, right):
            begin = starts[index]
            end = starts[index + 1]
            has_term.append(any(begin <= term < end for term in zhongqi_dates))
        leap_at = -1
        if count == 13:
            # Month 11 contains the winter solstice, so the leap month is a
            # later month that has no major solar term.
            missing = [
                offset
                for offset, present in enumerate(has_term)
                if not present and offset > 0
            ]
            leap_at = missing[0] if missing else -1
        number = 11
        labels: list[tuple[int, bool]] = []
        for offset in range(count):
            if offset == leap_at:
                labels.append((labels[-1][0], True))
                continue
            labels.append((number, False))
            number = 1 if number == 12 else number + 1
        new_year = next(
            (
                starts[left + offset]
                for offset, (month, leap) in enumerate(labels)
                if month == 1 and not leap
            ),
            None,
        )
        if new_year is None:
            continue
        year_number = new_year.year
        for offset, (month, leap) in enumerate(labels):
            start = starts[left + offset]
            year = year_number if start >= new_year else year_number - 1
            zodiac = (year - 4) % 12
            labeled.append(
                _Month(
                    start=start,
                    end=starts[left + offset + 1],
                    year=year,
                    month=month,
                    leap=leap,
                    zodiac_index=zodiac,
                )
            )
    labeled.sort(key=lambda month: month.start)
    deduped: list[_Month] = []
    occupied: set[date] = set()
    for item in labeled:
        if item.start in occupied:
            continue
        occupied.add(item.start)
        deduped.append(item)
    return tuple(deduped)


def _calendar_span(first: date, last: date) -> _Ephemeris:
    return _ephemeris(first.year - 1, last.year + 1)


def chinese_date(day: date) -> ChineseDate:
    """Chinese lunisolar date of a Gregorian civil date."""
    found = _lookup_month(day)
    return ChineseDate(
        year=found.year,
        month=found.month,
        day=(day - found.start).days + 1,
        leap=found.leap,
        zodiac_index=found.zodiac_index,
    )


def _lookup_month(day: date) -> _Month:
    sky = _calendar_span(day, day)
    index = bisect_right(sky.month_starts, day) - 1
    if index < 0 or index >= len(sky.months):
        raise ValueError(f"no Chinese month covers {day.isoformat()}")
    month = sky.months[index]
    if not (month.start <= day < month.end):
        raise ValueError(f"no Chinese month covers {day.isoformat()}")
    return month


def _parse_clock(at: str) -> tuple[int, int, int]:
    parts = at.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"clock time must be HH:MM or HH:MM:SS, got {at!r}")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError as exc:
        raise ValueError(f"clock time must be numeric, got {at!r}") from exc
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"clock time is out of range: {at!r}")
    return hour, minute, second


def _zone(name: str) -> timezone | ZoneInfo:
    if name in {"UTC", "Etc/UTC", "Z"}:
        return timezone.utc
    try:
        return ZoneInfo(name)
    except Exception as exc:
        raise ValueError(f"unknown timezone {name!r}") from exc


def instant_on(day: date, *, tz: str = "UTC", at: str = "12:00") -> datetime:
    """Clock time ``at`` on ``day`` in ``tz``."""
    hour, minute, second = _parse_clock(at)
    return datetime(
        day.year, day.month, day.day, hour, minute, second, tzinfo=_zone(tz)
    )


def _events_on(
    events: tuple[_Syzygy, ...], day: date, tz: timezone | ZoneInfo
) -> list[_Syzygy]:
    return [event for event in events if _civil(event.utc, tz) == day]


@dataclass(frozen=True)
class LunarDay:
    """Moon and Chinese-calendar fields for one civil date."""

    date: date
    instant: datetime
    timezone: str
    moon_phase: float
    illumination: float
    days_since_new: float
    named_phase: str
    synodic_month_index: int
    distance_km: float
    moon_latitude_deg: float
    is_new_moon: bool
    is_full_moon: bool
    near_perigee: bool
    is_supermoon: bool
    eclipse_kind: str | None
    eclipse_latitude_deg: float | None
    chinese: ChineseDate

    def as_dict(self) -> dict[str, Any]:
        eclipse = None
        if self.eclipse_kind is not None:
            eclipse = {
                "kind": self.eclipse_kind,
                "moon_latitude_deg": round(float(self.eclipse_latitude_deg or 0.0), 4),
                "note": (
                    "Geometric syzygy near the node. Not a published eclipse catalog."
                ),
            }
        return {
            "date": self.date.isoformat(),
            "instant": self.instant.astimezone(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "timezone": self.timezone,
            "moon_phase": round(self.moon_phase, 6),
            "illumination": round(self.illumination, 6),
            "illumination_percent": round(100.0 * self.illumination, 2),
            "days_since_new": round(self.days_since_new, 4),
            "named_phase": self.named_phase,
            "synodic_month_index": self.synodic_month_index,
            "lunar_month": self.chinese.month,
            "lunar_day": self.chinese.day,
            "is_leap_month": self.chinese.leap,
            "chinese_year": self.chinese.year,
            "zodiac": self.chinese.zodiac,
            "zodiac_index": self.chinese.zodiac_index,
            "chinese_date": self.chinese.text(),
            "distance_km": round(self.distance_km, 1),
            "moon_latitude_deg": round(self.moon_latitude_deg, 4),
            "is_new_moon": self.is_new_moon,
            "is_full_moon": self.is_full_moon,
            "near_perigee": self.near_perigee,
            "is_supermoon": self.is_supermoon,
            "eclipse": eclipse,
            "method": (
                "Offline Meeus series. Chinese dates use the winter-solstice "
                "rule in China Standard Time. Supermoon means a full moon "
                "closer than 360,000 km. Eclipse flags are geometric."
            ),
        }

    def model_row(self) -> dict[str, float]:
        return {name: float(self.as_features()[name]) for name in LUNAR_MODEL_COLUMNS}

    def as_features(self) -> dict[str, float]:
        angle = 2.0 * math.pi * self.moon_phase
        eclipse_solar = 1.0 if self.eclipse_kind == "solar" else 0.0
        eclipse_lunar = 1.0 if self.eclipse_kind == "lunar" else 0.0
        return {
            "moon_phase": self.moon_phase,
            "illumination": self.illumination,
            "days_since_new": self.days_since_new,
            "phase_sin": math.sin(angle),
            "phase_cos": math.cos(angle),
            "distance_km": self.distance_km,
            "lunar_month": float(self.chinese.month),
            "lunar_day": float(self.chinese.day),
            "is_leap_month": 1.0 if self.chinese.leap else 0.0,
            "zodiac_index": float(self.chinese.zodiac_index),
            "is_new_moon": 1.0 if self.is_new_moon else 0.0,
            "is_full_moon": 1.0 if self.is_full_moon else 0.0,
            "near_perigee": 1.0 if self.near_perigee else 0.0,
            "is_supermoon": 1.0 if self.is_supermoon else 0.0,
            "eclipse_solar": eclipse_solar,
            "eclipse_lunar": eclipse_lunar,
        }


def lunar_on(day: date, *, tz: str = "UTC", at: str = "12:00") -> LunarDay:
    """Lunar features at ``at`` on ``day`` in ``tz``.

    Full, new, supermoon, and eclipse flags use the civil date of the event
    in ``tz``, not the phase bin. A named phase of "full" covers about four
    days; ``is_full_moon`` is only the date of the full moon itself.
    """
    zone = _zone(tz)
    moment = instant_on(day, tz=tz, at=at)
    jd = julian_day(moment)
    phase = float(phase_fraction(jd))
    _lon, latitude, distance = moon_position(jd)
    sky = _calendar_span(day, day)
    new_today = _events_on(sky.new_moons, day, zone)
    full_today = _events_on(sky.full_moons, day, zone)
    previous = [moon for moon in sky.new_moons if moon.jd <= jd]
    if not previous:
        raise ValueError(f"no new moon on or before {day.isoformat()}")
    last_new = previous[-1]
    synodic_index = len([moon for moon in sky.new_moons if moon.jd <= last_new.jd]) - 1
    # Index from the seed new moon near 2000-01-06, stable across cache windows.
    epoch_index = int(round((last_new.jd - _NEW_MOON_SEED) / SYNODIC_MONTH_DAYS))
    eclipse_kind = None
    eclipse_lat = None
    if new_today and abs(new_today[0].latitude_deg) <= SOLAR_ECLIPSE_LATITUDE_DEG:
        eclipse_kind = "solar"
        eclipse_lat = new_today[0].latitude_deg
    elif full_today and abs(full_today[0].latitude_deg) <= LUNAR_ECLIPSE_LATITUDE_DEG:
        eclipse_kind = "lunar"
        eclipse_lat = full_today[0].latitude_deg
    supermoon = bool(full_today and full_today[0].distance_km < PERIGEE_DISTANCE_KM)
    return LunarDay(
        date=day,
        instant=moment,
        timezone=tz,
        moon_phase=phase,
        illumination=float(illumination_fraction(phase)),
        days_since_new=jd - last_new.jd,
        named_phase=named_phase(phase),
        synodic_month_index=epoch_index if epoch_index >= 0 else synodic_index,
        distance_km=float(distance),
        moon_latitude_deg=float(latitude),
        is_new_moon=bool(new_today),
        is_full_moon=bool(full_today),
        near_perigee=float(distance) < PERIGEE_DISTANCE_KM,
        is_supermoon=supermoon,
        eclipse_kind=eclipse_kind,
        eclipse_latitude_deg=eclipse_lat,
        chinese=chinese_date(day),
    )


def lunar_now(moment: datetime | None = None) -> LunarDay:
    """Features at ``moment`` (UTC now when omitted). Flags use that UTC date."""
    utc = moment.astimezone(timezone.utc) if moment else datetime.now(timezone.utc)
    if utc.tzinfo is None:
        utc = utc.replace(tzinfo=timezone.utc)
    day = utc.date()
    clock = utc.strftime("%H:%M:%S")
    return lunar_on(day, tz="UTC", at=clock)


def _coerce_dates(values: pd.Series | np.ndarray | list[Any]) -> list[date]:
    if isinstance(values, pd.Series):
        parsed = pd.to_datetime(values, errors="raise")
        if getattr(parsed.dt, "tz", None) is not None:
            parsed = parsed.dt.tz_convert("UTC")
        stamps = parsed.dt.date
        return list(stamps.tolist())
    dates: list[date] = []
    for value in list(values):
        if isinstance(value, datetime):
            dates.append(value.date())
        elif isinstance(value, date):
            dates.append(value)
        else:
            text = str(value)[:10]
            dates.append(date.fromisoformat(text))
    return dates


def lunar_dataframe(
    dates: pd.Series | np.ndarray | list[Any],
    *,
    tz: str = "UTC",
    at: str = "12:00",
) -> pd.DataFrame:
    """One feature row per date, including the model columns.

    Dates are civil dates. ``tz`` and ``at`` choose the instant used for the
    phase, distance, and the full/new/eclipse flags.
    """
    days = _coerce_dates(dates)
    if not days:
        return pd.DataFrame(columns=["date", "named_phase", *LUNAR_MODEL_COLUMNS])
    return _lunar_dataframe_once(days, tz=tz, at=at)


def _chinese_from_sky(day: date, sky: _Ephemeris) -> ChineseDate:
    index = bisect_right(sky.month_starts, day) - 1
    if index < 0 or index >= len(sky.months):
        raise ValueError(f"no Chinese month covers {day.isoformat()}")
    month = sky.months[index]
    if not (month.start <= day < month.end):
        raise ValueError(f"no Chinese month covers {day.isoformat()}")
    return ChineseDate(
        year=month.year,
        month=month.month,
        day=(day - month.start).days + 1,
        leap=month.leap,
        zodiac_index=month.zodiac_index,
    )


def _lunar_dataframe_once(days: list[date], *, tz: str, at: str) -> pd.DataFrame:
    """Vectorized features. Single-day :func:`lunar_on` is the same calculation."""
    zone = _zone(tz)
    moments = [instant_on(day, tz=tz, at=at) for day in days]
    jds = np.array([julian_day(moment) for moment in moments], dtype=np.float64)
    phase = np.asarray(phase_fraction(jds), dtype=np.float64).reshape(-1)
    _lon, latitude, distance = moon_position(jds)
    latitude = np.asarray(latitude, dtype=np.float64).reshape(-1)
    distance = np.asarray(distance, dtype=np.float64).reshape(-1)
    illuminated = np.asarray(illumination_fraction(phase), dtype=np.float64).reshape(-1)
    sky = _calendar_span(min(days), max(days))
    new_jds = np.array([moon.jd for moon in sky.new_moons], dtype=np.float64)
    if new_jds.size == 0:
        raise ValueError("the lunar ephemeris has no new moons for these dates")
    previous = np.searchsorted(new_jds, jds, side="right") - 1
    if int(previous.min()) < 0:
        raise ValueError("a date is before the first computed new moon")
    days_since = jds - new_jds[previous]
    new_on = {_civil(moon.utc, zone): moon for moon in sky.new_moons}
    full_on = {_civil(moon.utc, zone): moon for moon in sky.full_moons}
    chinese = [_chinese_from_sky(day, sky) for day in days]
    angle = 2.0 * math.pi * phase
    is_new = np.array([day in new_on for day in days], dtype=np.float64)
    is_full = np.array([day in full_on for day in days], dtype=np.float64)
    eclipse_solar = np.zeros(len(days), dtype=np.float64)
    eclipse_lunar = np.zeros(len(days), dtype=np.float64)
    supermoon = np.zeros(len(days), dtype=np.float64)
    for index, day in enumerate(days):
        new_moon = new_on.get(day)
        full_moon = full_on.get(day)
        if (
            new_moon is not None
            and abs(new_moon.latitude_deg) <= SOLAR_ECLIPSE_LATITUDE_DEG
        ):
            eclipse_solar[index] = 1.0
        if (
            full_moon is not None
            and abs(full_moon.latitude_deg) <= LUNAR_ECLIPSE_LATITUDE_DEG
        ):
            eclipse_lunar[index] = 1.0
        if full_moon is not None and full_moon.distance_km < PERIGEE_DISTANCE_KM:
            supermoon[index] = 1.0
    names = [named_phase(float(value)) for value in phase]
    frame = pd.DataFrame(
        {
            "date": [day.isoformat() for day in days],
            "named_phase": names,
            "zodiac": [item.zodiac for item in chinese],
            "chinese_year": [item.year for item in chinese],
            "moon_phase": phase,
            "illumination": illuminated,
            "days_since_new": days_since,
            "phase_sin": np.sin(angle),
            "phase_cos": np.cos(angle),
            "distance_km": distance,
            "lunar_month": [item.month for item in chinese],
            "lunar_day": [item.day for item in chinese],
            "is_leap_month": [1.0 if item.leap else 0.0 for item in chinese],
            "zodiac_index": [item.zodiac_index for item in chinese],
            "is_new_moon": is_new,
            "is_full_moon": is_full,
            "near_perigee": (distance < PERIGEE_DISTANCE_KM).astype(np.float64),
            "is_supermoon": supermoon,
            "eclipse_solar": eclipse_solar,
            "eclipse_lunar": eclipse_lunar,
        }
    )
    # Latitude is not a model column. Keep it so a caller can audit eclipses.
    frame["moon_latitude_deg"] = latitude
    return frame


def lunar_feature_names(*, future: bool = False) -> list[str]:
    """Model column names, optionally prefixed for the day being forecast."""
    if future:
        return [f"future_{name}" for name in LUNAR_MODEL_COLUMNS]
    return list(LUNAR_MODEL_COLUMNS)


def lunar_feature_frame(
    dates: pd.Series | np.ndarray | list[Any],
    *,
    tz: str = "UTC",
    at: str = "12:00",
) -> pd.DataFrame:
    """Model columns only, aligned with ``dates`` and indexed like a Series."""
    frame = _lunar_dataframe_once(_coerce_dates(dates), tz=tz, at=at)
    columns = frame.loc[:, list(LUNAR_MODEL_COLUMNS)]
    if isinstance(dates, pd.Series):
        columns.index = dates.index
    return columns


def join_lunar(
    frame: pd.DataFrame,
    date_column: str = "date",
    *,
    tz: str = "UTC",
    at: str = "12:00",
) -> pd.DataFrame:
    """Copy ``frame`` and append lunar columns for ``date_column``.

    The date column may be a civil date, a timestamp, or an ISO string. The
    calendar date (not the clock on the timestamp) is the join key. ``tz`` and
    ``at`` select the instant on that date.
    """
    if date_column not in frame.columns:
        raise ValueError(f"missing date column {date_column!r}")
    overlap = [name for name in LUNAR_MODEL_COLUMNS if name in frame.columns]
    if overlap:
        raise ValueError(f"lunar columns already present: {overlap}")
    features = lunar_feature_frame(frame[date_column], tz=tz, at=at)
    features.index = frame.index
    return pd.concat([frame.copy(), features], axis=1)


def daily_lunar_frame(
    start: date, end: date, *, tz: str = "UTC", at: str = "12:00"
) -> pd.DataFrame:
    """Every civil date from ``start`` through ``end``, inclusive."""
    if end < start:
        raise ValueError("end is before start")
    days = []
    cursor = start
    while cursor <= end:
        days.append(cursor)
        cursor += timedelta(days=1)
    return _lunar_dataframe_once(days, tz=tz, at=at)


def lottery_study_hooks() -> tuple[dict[str, Any], ...]:
    """Daily moon features the lottery study can align to draw dates.

    The value is the moon at 12:00 UTC, which is before a 22:59 ET draw, so
    the study treats the row as known on that calendar date.
    """
    frame = daily_lunar_frame(date(2000, 1, 1), date(2040, 12, 31))
    return (
        {
            "dataset_id": "lunar",
            "group": "environment",
            "description": (
                "Moon phase, illumination, Chinese lunar month, perigee, and "
                "geometric eclipse flags at 12:00 UTC. Computed offline from "
                "Meeus series; known before a 22:59 ET draw."
            ),
            "availability": "same_day_before_draw",
            "date_columns": ("date",),
            "value_columns": LUNAR_MODEL_COLUMNS,
            "frame": frame,
        },
    )
