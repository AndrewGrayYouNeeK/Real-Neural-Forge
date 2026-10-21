"""Load the dataset files that main actually trains on.

Market groups are wide panels of daily log returns. Monthly and annual
aggregates are stamped at the end of their period so a draw inside an open
period cannot see that period's total. The QRNG file has no calendar and is
refused rather than assigned invented dates.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.paths import resolve_repo_path

MARKET_PREFIXES = {
    "stocks": "stock-",
    "fx": "fx-",
    "crypto": "crypto-",
    "commodities": "cmdty-",
}

_REALWORLD_FETCH = {
    "electricity": "scripts.realworld.fetch_electricity",
    "air_quality": "scripts.realworld.fetch_air_quality",
    "earthquakes": "scripts.realworld.fetch_earthquakes",
    "sunspots": "scripts.realworld.fetch_sunspots",
    "economy": "scripts.realworld.fetch_fred",
    "sports": "scripts.realworld.fetch_sports",
    "ucdp": "scripts.realworld.fetch_ucdp",
}


@dataclass(frozen=True)
class InputFile:
    """One CSV contribution to a dataset panel."""

    paths: tuple[str, ...]
    columns: tuple[str, ...]
    names: tuple[str, ...]
    period: str | None = None
    date_column: str = "date"
    lag_hours: int = 0
    drop_incomplete: bool = False
    optional: bool = False
    also_consume: tuple[str, ...] = ()


@dataclass
class LoadedFrame:
    """A wired dataset after its files have been read."""

    available: bool
    detail: str
    observed_at: pd.Series | None
    values: np.ndarray | None
    columns: tuple[str, ...]
    source: str
    consumed: tuple[str, ...]


def load_wired(
    dataset_id: str,
    loader: str,
    *,
    fetch: bool,
    features_dir: Path | None = None,
) -> LoadedFrame:
    """Read one catalog dataset. ``features_dir`` overrides the market folder."""
    if loader == "qrng":
        return _load_qrng()
    if loader == "markets":
        return _load_market(dataset_id, fetch=fetch, features_dir=features_dir)
    if loader == "horse":
        return _load_horse(fetch=fetch)
    if loader == "files":
        return _load_files(dataset_id, fetch=fetch)
    raise ValueError(f"unknown dataset loader {loader!r}")


def stamp_known_dates(
    observed_at: pd.Series,
    *,
    period: str | None = None,
    lag_hours: int = 0,
) -> pd.Series:
    """Move a publication time to when the value could actually be known.

    ``month`` and ``year`` land on the last calendar day of that period.
    ``lag_hours`` moves a clock time forward, which is how an hour-beginning
    electricity stamp becomes the end of that hour.
    """
    parsed = pd.to_datetime(observed_at, errors="coerce", utc=False)
    if lag_hours:
        if getattr(parsed.dt, "tz", None) is None:
            parsed = parsed.dt.tz_localize("UTC")
        parsed = parsed + pd.to_timedelta(lag_hours, unit="h")
        return parsed.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if period == "month":
        stamped = parsed + pd.offsets.MonthEnd(0)
        return stamped.dt.strftime("%Y-%m-%d")
    if period == "year":
        return parsed.dt.year.astype("Int64").astype(str) + "-12-31"
    if period is not None:
        raise ValueError(f"period must be 'month', 'year', or None, got {period!r}")
    return observed_at.reset_index(drop=True)


def _load_files(dataset_id: str, *, fetch: bool) -> LoadedFrame:
    specs = _FILE_SPECS[dataset_id]
    if fetch and not _any_required_present(specs):
        _ensure(dataset_id)
    frames: list[pd.DataFrame] = []
    consumed: list[str] = []
    sources: list[str] = []
    for spec in specs:
        path = _first_existing(spec.paths)
        for candidate in (*spec.paths, *spec.also_consume):
            resolved = resolve_repo_path(candidate)
            if resolved.is_file():
                consumed.append(str(resolved.resolve()))
        if path is None:
            if spec.optional:
                continue
            missing = ", ".join(spec.paths)
            return _missing(f"missing {missing}")
        frame = _read_spec(path, spec)
        frames.append(frame)
        sources.append(str(path))
    if not frames:
        return _missing("no rows")
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on="date", how="outer")
    merged = merged.sort_values("date")
    columns = [column for column in merged.columns if column != "date"]
    values = np.column_stack(
        [
            pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=np.float64)
            for column in columns
        ]
    )
    return LoadedFrame(
        available=True,
        detail=f"loaded {len(merged)} rows from {', '.join(sources)}",
        observed_at=merged["date"].reset_index(drop=True),
        values=values,
        columns=tuple(columns),
        source=sources[0],
        consumed=tuple(dict.fromkeys(consumed)),
    )


def _load_market(
    dataset_id: str,
    *,
    fetch: bool,
    features_dir: Path | None,
) -> LoadedFrame:
    prefix = MARKET_PREFIXES[dataset_id]
    directory = features_dir or resolve_repo_path("data/markets/features")
    paths = _market_paths(directory, prefix)
    if not paths and fetch and features_dir is None:
        _ensure(dataset_id)
        paths = _market_paths(directory, prefix)
    if not paths:
        return _missing(
            f"no {prefix}* feature files in {directory}. Run scripts/fetch_markets.py."
        )
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, usecols=["date", "log_return"])
        name = path.stem[len(prefix) :]
        frames.append(frame.rename(columns={"log_return": name}).loc[:, ["date", name]])
    merged = frames[0]
    for extra in frames[1:]:
        merged = merged.merge(extra, on="date", how="outer")
    merged = merged.sort_values("date")
    columns = [column for column in merged.columns if column != "date"]
    values = np.column_stack(
        [
            pd.to_numeric(merged[column], errors="coerce").to_numpy(dtype=np.float64)
            for column in columns
        ]
    )
    consumed = tuple(str(path.resolve()) for path in paths)
    return LoadedFrame(
        available=True,
        detail=(
            f"loaded {len(merged)} days and {len(columns)} log-return series "
            f"from {directory}"
        ),
        observed_at=merged["date"].reset_index(drop=True),
        values=values,
        columns=tuple(columns),
        source=str(directory),
        consumed=consumed,
    )


def _load_horse(*, fetch: bool) -> LoadedFrame:
    path = resolve_repo_path("data/raw/horse_racing/performances.csv.gz")
    if not path.is_file() and fetch:
        _download_horse(path)
    if not path.is_file():
        return _missing(
            "not on main yet; data/raw/horse_racing/performances.csv.gz is missing. "
            "Run scripts/fetch_horse_racing.py."
        )
    from src.domain_forecasts.racing import tidy_hk_performances

    raw = pd.read_csv(path, low_memory=False)
    tidy = tidy_hk_performances(raw)
    if tidy.empty:
        return _missing(f"{path} has no Hong Kong runners")
    dates = pd.to_datetime(tidy["race_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    daily = (
        tidy.assign(_date=dates)
        .groupby("_date", as_index=False)
        .agg(
            n_races=("race_key", "nunique"),
            n_runners=("horse_id", "size"),
            mean_placing=("finish", "mean"),
            median_odds=("odds", "median"),
        )
        .rename(columns={"_date": "date"})
        .sort_values("date")
    )
    columns = ("n_races", "n_runners", "mean_placing", "median_odds")
    values = np.column_stack(
        [pd.to_numeric(daily[column], errors="coerce").to_numpy() for column in columns]
    )
    return LoadedFrame(
        available=True,
        detail=(
            f"loaded {len(daily)} race days {daily['date'].iloc[0]} to "
            f"{daily['date'].iloc[-1]} from {path}. Later draws keep that last race day."
        ),
        observed_at=daily["date"].reset_index(drop=True),
        values=values.astype(np.float64),
        columns=columns,
        source=str(path),
        consumed=(str(path.resolve()),),
    )


def _load_qrng() -> LoadedFrame:
    path = resolve_repo_path("datasets/quantum/qrng_symbols.csv")
    location = str(path) if path.is_file() else "datasets/quantum/qrng_symbols.csv"
    return _missing(
        f"{location} is an ordered symbol stream (column 'step'), not a "
        "calendar series, so it cannot be placed before a draw without "
        "inventing a clock."
    )


def _read_spec(path: Path, spec: InputFile) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if spec.date_column not in frame.columns:
        raise ValueError(f"{path} has no {spec.date_column!r} column")
    if spec.drop_incomplete and "incomplete" in frame.columns:
        flag = frame["incomplete"].astype(str).str.lower().isin({"true", "1"})
        frame = frame.loc[~flag]
    missing = [column for column in spec.columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    if len(spec.names) != len(spec.columns):
        raise ValueError(f"{path} column rename does not match the selection")
    kept = frame.loc[:, [spec.date_column, *spec.columns]].copy()
    kept = kept.rename(
        columns={
            spec.date_column: "date",
            **dict(zip(spec.columns, spec.names, strict=True)),
        }
    )
    kept["date"] = stamp_known_dates(
        kept["date"], period=spec.period, lag_hours=spec.lag_hours
    )
    return kept.reset_index(drop=True)


def _market_paths(directory: Path, prefix: str) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.glob(f"{prefix}*.csv")
        if path.is_file() and path.stem.startswith(prefix)
    )


def _first_existing(paths: tuple[str, ...]) -> Path | None:
    for raw in paths:
        path = resolve_repo_path(raw)
        if path.is_file():
            return path
    return None


def _any_required_present(specs: tuple[InputFile, ...]) -> bool:
    required = [spec for spec in specs if not spec.optional]
    if not required:
        return True
    return any(_first_existing(spec.paths) is not None for spec in required)


def _ensure(dataset_id: str) -> None:
    if dataset_id in MARKET_PREFIXES:
        from src.markets.fetch import fetch_all

        fetch_all()
        return
    if dataset_id.startswith("space_"):
        from src.space.fetch import fetch_space

        fetch_space()
        return
    module_name = _REALWORLD_FETCH.get(dataset_id)
    if module_name is not None:
        module = importlib.import_module(module_name)
        module.main()


def _download_horse(path: Path) -> None:
    import urllib.request

    from scripts.fetch_horse_racing import SOURCE_URL

    path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(SOURCE_URL, path)


def _missing(detail: str) -> LoadedFrame:
    return LoadedFrame(
        available=False,
        detail=detail,
        observed_at=None,
        values=None,
        columns=(),
        source="",
        consumed=(),
    )


_FILE_SPECS: dict[str, tuple[InputFile, ...]] = {
    "electricity": (
        InputFile(
            (
                "data/realworld/electricity_us48_hourly_full.csv",
                "data/realworld/electricity_us48_hourly.csv",
            ),
            ("value",),
            ("demand",),
            lag_hours=1,
            also_consume=(
                "data/realworld/electricity_us48_hourly_full.csv",
                "data/realworld/electricity_us48_hourly.csv",
            ),
        ),
    ),
    "air_quality": (
        InputFile(
            ("data/realworld/air_berlin_daily.csv",),
            ("pm2_5", "ozone", "pollen"),
            ("pm2_5", "ozone", "pollen"),
            also_consume=(
                "data/realworld/air_pm25_berlin.csv",
                "data/realworld/air_ozone_berlin.csv",
                "data/realworld/air_pollen_berlin.csv",
            ),
        ),
    ),
    "earthquakes": (
        InputFile(
            ("data/realworld/earthquakes_m45_daily.csv",),
            ("value", "log10_energy"),
            ("count", "log10_energy"),
        ),
    ),
    "sunspots": (
        InputFile(
            ("data/realworld/sunspots_monthly.csv",),
            ("value",),
            ("sunspot_number",),
            period="month",
        ),
    ),
    "economy": (
        InputFile(
            ("data/realworld/fred_unemployment.csv",),
            ("value",),
            ("unemployment",),
            period="month",
        ),
        InputFile(
            ("data/realworld/fred_cpi_inflation.csv",),
            ("value",),
            ("cpi_yoy",),
            period="month",
        ),
        InputFile(
            ("data/realworld/fred_gas.csv",),
            ("value",),
            ("gasoline",),
        ),
    ),
    "sports": (
        InputFile(
            ("data/realworld/sports_arsenal_goal_diff.csv",),
            ("value",),
            ("goal_diff",),
        ),
    ),
    "ucdp": (
        InputFile(
            ("data/realworld/ucdp_monthly.csv",),
            ("events", "fatalities"),
            ("events", "fatalities"),
            period="month",
            also_consume=(
                "data/realworld/ucdp_monthly_events.csv",
                "data/realworld/ucdp_monthly_fatalities.csv",
            ),
        ),
        InputFile(
            ("data/realworld/ucdp_onset_gaps.csv",),
            ("value",),
            ("onset_gap",),
            optional=True,
        ),
    ),
    "space_f107": (InputFile(("data/space/f107_daily.csv",), ("f107",), ("f107",)),),
    "space_flares": (
        InputFile(
            ("data/space/flares_monthly.csv",),
            ("b_class", "c_class", "m_class", "x_class"),
            ("b_class", "c_class", "m_class", "x_class"),
            period="month",
        ),
    ),
    "space_kp_ap": (
        InputFile(
            ("data/space/kp_daily.csv",),
            ("kp_max", "ap"),
            ("kp_max", "ap"),
            also_consume=("data/space/kp_solar_wind.csv",),
        ),
    ),
    "space_solar_wind": (
        InputFile(
            ("data/space/solar_wind_daily.csv",),
            ("speed", "density"),
            ("speed", "density"),
        ),
    ),
    "space_dst": (InputFile(("data/space/dst_daily.csv",), ("dst",), ("dst",)),),
    "space_neos": (
        InputFile(
            ("data/space/neo_approaches_monthly.csv",),
            ("approaches",),
            ("approaches",),
            period="month",
        ),
        InputFile(
            ("data/space/neo_discoveries_monthly.csv",),
            ("discoveries",),
            ("discoveries",),
            period="month",
        ),
    ),
    "space_launches": (
        InputFile(
            ("data/space/launches_monthly.csv",),
            ("attempts",),
            ("attempts",),
            period="month",
        ),
    ),
    "space_exoplanets": (
        InputFile(
            ("data/space/exoplanets_yearly.csv",),
            ("discoveries",),
            ("discoveries",),
            period="year",
        ),
    ),
    "ai_progress": (
        InputFile(
            ("datasets/ai_progress/arxiv_counts_monthly.csv",),
            ("cs.AI", "cs.LG", "cs.CL"),
            ("cs_ai", "cs_lg", "cs_cl"),
            period="month",
            date_column="month",
        ),
        InputFile(
            ("datasets/ai_progress/frontier_compute_monthly.csv",),
            ("log10_compute",),
            ("frontier_log10_compute",),
            period="month",
        ),
        InputFile(
            ("datasets/ai_progress/github_ml_repos_monthly.csv",),
            ("count",),
            ("github_ml_repos",),
            period="month",
            date_column="month",
            drop_incomplete=True,
        ),
    ),
    "quantum_qubits": (
        InputFile(
            ("datasets/quantum/qubit_record_annual.csv",),
            ("log10_qubits", "log10_qubits_excluding_unrearranged"),
            ("log10_qubits", "log10_qubits_excluding_unrearranged"),
            period="year",
        ),
    ),
    "quantum_gate_error": (
        InputFile(
            ("datasets/quantum/two_qubit_error_annual.csv",),
            ("log10_error",),
            ("log10_error",),
            period="year",
        ),
    ),
    "quantum_neutrons": (
        InputFile(
            ("datasets/quantum/oulu_neutrons_monthly.csv",),
            ("count_rate",),
            ("count_rate",),
            period="month",
        ),
    ),
    "arxiv_quant_ph": (
        InputFile(
            ("datasets/quantum/arxiv_quant_ph_monthly.csv",),
            ("count",),
            ("count",),
            period="month",
            date_column="month",
        ),
    ),
}
