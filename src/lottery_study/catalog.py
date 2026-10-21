"""Datasets the lottery study knows how to attach.

Entries that are not on main yet stay in this catalog as explicit hooks. A
fetcher is picked up on the next run, without an edit here, when it does any
of the following:

* exports ``LOTTERY_STUDY_HOOKS`` (or ``lottery_study_hooks()``) from a module
  listed on the entry, or from any new module under ``src/data/``;
* writes one of the CSV paths listed on the entry;
* writes any other CSV under ``data/`` that has a date column and at least one
  numeric column (discovery is on by default).

Date-only rows use ``availability``. Clock times are never read past the
Powerball draw at 22:59 America/New_York; a timestamped file uses that cutoff
even when the catalog row says ``prior_calendar_day``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetCandidate:
    """One dataset the comparison will load when its files or module appear."""

    dataset_id: str
    group: str
    description: str
    modules: tuple[str, ...]
    csv_candidates: tuple[str, ...]
    availability: str = "prior_calendar_day"
    date_columns: tuple[str, ...] = (
        "date",
        "timestamp",
        "datetime",
        "observation_date",
        "time",
        "ds",
    )
    value_columns: tuple[str, ...] | None = None
    fetcher: str | None = None


def dataset_catalog() -> tuple[DatasetCandidate, ...]:
    """Return the built-in hook list, available datasets first."""
    return (
        DatasetCandidate(
            dataset_id="weather",
            group="environment",
            description=(
                "NOAA NCEI daily maximum temperature at Central Park "
                "(station USW00094728), with the day-of-year harmonics stored "
                "in that file. Public domain (17 U.S.C. § 105)."
            ),
            modules=("src.data.weather",),
            csv_candidates=("data/weather_central_park.csv",),
            availability="prior_calendar_day",
            date_columns=("date",),
            value_columns=("temperature_c", "doy_sin", "doy_cos"),
            fetcher="src.data.weather:fetch_weather",
        ),
        # Markets. Same-calendar-day rows are usable when the series is a
        # daily close: the US cash session ends at 16:00 ET, before the draw.
        # Crypto trades through the draw, so a date-only crypto file stays on
        # the previous day unless the file carries clock times.
        DatasetCandidate(
            dataset_id="stocks",
            group="markets",
            description="Equity prices or index levels. TODO: fetcher is not on main yet.",
            modules=("src.data.stocks", "src.data.equities", "src.data.markets"),
            csv_candidates=(
                "data/stocks.csv",
                "data/equities.csv",
                "data/stock_prices.csv",
                "data/markets/stocks.csv",
            ),
            availability="same_day_before_draw",
        ),
        DatasetCandidate(
            dataset_id="fx",
            group="markets",
            description=(
                "Foreign-exchange rates, about 30 currencies. "
                "TODO: fetcher is not on main yet."
            ),
            modules=("src.data.fx", "src.data.forex", "src.data.currencies"),
            csv_candidates=(
                "data/fx.csv",
                "data/forex.csv",
                "data/currencies.csv",
                "data/markets/fx.csv",
            ),
            availability="same_day_before_draw",
        ),
        DatasetCandidate(
            dataset_id="crypto",
            group="markets",
            description=(
                "Large-cap crypto prices, about the top 50 coins. "
                "TODO: fetcher is not on main yet."
            ),
            modules=("src.data.crypto", "src.data.cryptocurrencies"),
            csv_candidates=(
                "data/crypto.csv",
                "data/cryptocurrencies.csv",
                "data/markets/crypto.csv",
            ),
            availability="prior_calendar_day",
        ),
        DatasetCandidate(
            dataset_id="commodities",
            group="markets",
            description="Commodity or futures settles. TODO: fetcher is not on main yet.",
            modules=("src.data.commodities", "src.data.futures"),
            csv_candidates=(
                "data/commodities.csv",
                "data/futures.csv",
                "data/markets/commodities.csv",
            ),
            availability="same_day_before_draw",
        ),
        DatasetCandidate(
            dataset_id="electricity",
            group="infrastructure",
            description="Electricity load or price. TODO: fetcher is not on main yet.",
            modules=("src.data.electricity", "src.data.power"),
            csv_candidates=("data/electricity.csv", "data/power.csv", "data/grid.csv"),
        ),
        DatasetCandidate(
            dataset_id="air_quality",
            group="environment",
            description="Air-quality measurements. TODO: fetcher is not on main yet.",
            modules=("src.data.air_quality", "src.data.aqi"),
            csv_candidates=("data/air_quality.csv", "data/aqi.csv"),
        ),
        DatasetCandidate(
            dataset_id="earthquakes",
            group="environment",
            description="Earthquake counts or magnitudes. TODO: fetcher is not on main yet.",
            modules=("src.data.earthquakes", "src.data.seismic"),
            csv_candidates=("data/earthquakes.csv", "data/seismic.csv"),
        ),
        DatasetCandidate(
            dataset_id="sunspots",
            group="environment",
            description="Sunspot counts. TODO: fetcher is not on main yet.",
            modules=("src.data.sunspots",),
            csv_candidates=("data/sunspots.csv",),
        ),
        DatasetCandidate(
            dataset_id="economy",
            group="economy",
            description="Macroeconomic series. TODO: fetcher is not on main yet.",
            modules=("src.data.economy", "src.data.fred", "src.data.macro"),
            csv_candidates=("data/economy.csv", "data/fred.csv", "data/macro.csv"),
        ),
        DatasetCandidate(
            dataset_id="sports",
            group="society",
            description="Sports results or betting lines. TODO: fetcher is not on main yet.",
            modules=("src.data.sports",),
            csv_candidates=("data/sports.csv",),
        ),
        DatasetCandidate(
            dataset_id="traffic",
            group="infrastructure",
            description="Traffic volume. TODO: fetcher is not on main yet.",
            modules=("src.data.traffic",),
            csv_candidates=("data/traffic.csv",),
        ),
        DatasetCandidate(
            dataset_id="ucdp",
            group="society",
            description=("UCDP conflict events. TODO: fetcher is not on main yet."),
            modules=("src.data.ucdp", "src.data.conflict"),
            csv_candidates=("data/ucdp.csv", "data/conflict.csv"),
        ),
        DatasetCandidate(
            dataset_id="space_f107",
            group="space",
            description="Solar radio flux F10.7. TODO: fetcher is not on main yet.",
            modules=("src.data.space", "src.data.space_weather", "src.data.f107"),
            csv_candidates=(
                "data/f107.csv",
                "data/space_f107.csv",
                "data/solar_f107.csv",
            ),
        ),
        DatasetCandidate(
            dataset_id="space_flares",
            group="space",
            description="Solar flares. TODO: fetcher is not on main yet.",
            modules=("src.data.space", "src.data.space_weather", "src.data.flares"),
            csv_candidates=("data/flares.csv", "data/solar_flares.csv"),
        ),
        DatasetCandidate(
            dataset_id="space_kp_ap",
            group="space",
            description="Geomagnetic Kp and Ap. TODO: fetcher is not on main yet.",
            modules=("src.data.space", "src.data.space_weather", "src.data.kp"),
            csv_candidates=("data/kp_ap.csv", "data/geomagnetic.csv", "data/kp.csv"),
        ),
        DatasetCandidate(
            dataset_id="space_solar_wind",
            group="space",
            description="Solar-wind measurements. TODO: fetcher is not on main yet.",
            modules=("src.data.space", "src.data.space_weather", "src.data.solar_wind"),
            csv_candidates=("data/solar_wind.csv",),
        ),
        DatasetCandidate(
            dataset_id="space_neos",
            group="space",
            description="Near-Earth objects. TODO: fetcher is not on main yet.",
            modules=("src.data.space", "src.data.neos"),
            csv_candidates=("data/neos.csv", "data/near_earth_objects.csv"),
        ),
        DatasetCandidate(
            dataset_id="space_launches",
            group="space",
            description="Rocket launches. TODO: fetcher is not on main yet.",
            modules=("src.data.space", "src.data.launches"),
            csv_candidates=("data/launches.csv",),
        ),
        DatasetCandidate(
            dataset_id="space_exoplanets",
            group="space",
            description="Exoplanet discoveries. TODO: fetcher is not on main yet.",
            modules=("src.data.space", "src.data.exoplanets"),
            csv_candidates=("data/exoplanets.csv",),
        ),
        DatasetCandidate(
            dataset_id="horse_racing",
            group="society",
            description="Horse-racing results. TODO: fetcher is not on main yet.",
            modules=("src.data.horse_racing", "src.data.horses"),
            csv_candidates=("data/horse_racing.csv", "data/horses.csv"),
        ),
        DatasetCandidate(
            dataset_id="ai_progress",
            group="technology",
            description="AI progress benchmarks. TODO: fetcher is not on main yet.",
            modules=("src.data.ai_progress", "src.data.ai"),
            csv_candidates=("data/ai_progress.csv",),
        ),
        DatasetCandidate(
            dataset_id="quantum_qubits",
            group="technology",
            description="Published qubit counts. TODO: fetcher is not on main yet.",
            modules=("src.data.quantum", "src.data.qubits"),
            csv_candidates=("data/qubits.csv", "data/quantum_qubits.csv"),
        ),
        DatasetCandidate(
            dataset_id="arxiv_quant_ph",
            group="technology",
            description=(
                "arXiv quant-ph publication counts. TODO: fetcher is not on main yet."
            ),
            modules=("src.data.quantum", "src.data.arxiv_quant_ph"),
            csv_candidates=("data/arxiv_quant_ph.csv", "data/quant_ph.csv"),
        ),
        DatasetCandidate(
            dataset_id="qrng",
            group="technology",
            description=(
                "Quantum random-number generator output. "
                "TODO: fetcher is not on main yet."
            ),
            modules=("src.data.quantum", "src.data.qrng"),
            csv_candidates=("data/qrng.csv",),
        ),
    )
