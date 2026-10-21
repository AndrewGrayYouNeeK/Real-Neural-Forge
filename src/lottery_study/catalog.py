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
    # ``files`` reads the paths in :mod:`src.lottery_study.sources`. ``markets``
    # joins that group's daily log-return files. ``horse`` and ``qrng`` are
    # special cases in the same module.
    loader: str | None = None
    display_name: str = ""
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
            display_name="Weather",
        ),
        # Markets. Same-calendar-day rows are usable when the series is a
        # daily close: the US cash session ends at 16:00 ET, before the draw.
        # Crypto trades through the draw, so a date-only crypto file stays on
        # the previous day unless the file carries clock times.
        DatasetCandidate(
            dataset_id="stocks",
            group="markets",
            description=(
                "Daily log returns for the equities and S&P 500 series in "
                "src/markets. The US cash close is 16:00 ET, before the draw. "
                "vol_10 is a function of those same returns and is not a second test."
            ),
            modules=("src.markets",),
            csv_candidates=("data/markets/features/stock-*.csv",),
            availability="same_day_before_draw",
            loader="markets",
            display_name="Stocks",
        ),
        DatasetCandidate(
            dataset_id="fx",
            group="markets",
            description=(
                "ECB and H.10 daily FX log returns, one column per currency. "
                "Reference rates are published before the 22:59 ET draw."
            ),
            modules=("src.markets",),
            csv_candidates=("data/markets/features/fx-*.csv",),
            availability="same_day_before_draw",
            loader="markets",
            display_name="FX",
        ),
        DatasetCandidate(
            dataset_id="crypto",
            group="markets",
            description=(
                "Daily log returns for the top crypto sample. A date-only "
                "candle has no clock, and crypto trades through the draw, so "
                "the value used is the previous calendar day."
            ),
            modules=("src.markets",),
            csv_candidates=("data/markets/features/crypto-*.csv",),
            availability="prior_calendar_day",
            loader="markets",
            display_name="Crypto",
        ),
        DatasetCandidate(
            dataset_id="commodities",
            group="markets",
            description=(
                "Commodity futures daily log returns. The daily settle is "
                "treated as known by the 16:00 ET cash close."
            ),
            modules=("src.markets",),
            csv_candidates=("data/markets/features/cmdty-*.csv",),
            availability="same_day_before_draw",
            loader="markets",
            display_name="Commodities",
        ),
        DatasetCandidate(
            dataset_id="electricity",
            group="infrastructure",
            description=(
                "EIA US48 hourly electricity demand. The hour stamp is the "
                "start of the hour, so the value is treated as known one hour later."
            ),
            modules=("src.realworld",),
            csv_candidates=("data/realworld/electricity_us48_hourly_full.csv",),
            loader="files",
            display_name="Electricity",
        ),
        DatasetCandidate(
            dataset_id="air_quality",
            group="environment",
            description=(
                "Berlin daily PM2.5, ozone, and pollen from Open-Meteo. "
                "A daily mean is not known until the day is over."
            ),
            modules=("src.realworld",),
            csv_candidates=("data/realworld/air_berlin_daily.csv",),
            loader="files",
            display_name="Air quality",
        ),
        DatasetCandidate(
            dataset_id="earthquakes",
            group="environment",
            description=(
                "USGS daily count of magnitude 4.5+ earthquakes and the log "
                "energy of those events. The day's total is used from the next day."
            ),
            modules=("src.realworld",),
            csv_candidates=("data/realworld/earthquakes_m45_daily.csv",),
            loader="files",
            display_name="Earthquakes",
        ),
        DatasetCandidate(
            dataset_id="sunspots",
            group="environment",
            description=(
                "SILSO monthly sunspot number. The month is stamped at month "
                "end so a draw during the month cannot see that month's mean."
            ),
            modules=("src.realworld",),
            csv_candidates=("data/realworld/sunspots_monthly.csv",),
            loader="files",
            display_name="Sunspots",
        ),
        DatasetCandidate(
            dataset_id="economy",
            group="economy",
            description=(
                "FRED unemployment, CPI inflation, and weekly regular gasoline. "
                "Monthly series are stamped at month end."
            ),
            modules=("src.realworld",),
            csv_candidates=("data/realworld/fred_unemployment.csv",),
            loader="files",
            display_name="Economy",
        ),
        DatasetCandidate(
            dataset_id="sports",
            group="society",
            description=(
                "Arsenal match goal difference from football-data.co.uk. "
                "The result is used from the next calendar day."
            ),
            modules=("src.realworld",),
            csv_candidates=("data/realworld/sports_arsenal_goal_diff.csv",),
            loader="files",
            display_name="Sports",
        ),
        DatasetCandidate(
            dataset_id="traffic",
            group="infrastructure",
            description="Traffic volume. TODO: fetcher is not on main yet.",
            modules=("src.data.traffic",),
            csv_candidates=("data/traffic.csv",),
            display_name="Traffic",
        ),
        DatasetCandidate(
            dataset_id="ucdp",
            group="society",
            description=(
                "UCDP monthly conflict events and fatalities, plus onset gaps. "
                "Monthly totals are stamped at month end."
            ),
            modules=("src.realworld",),
            csv_candidates=("data/realworld/ucdp_monthly.csv",),
            loader="files",
            display_name="UCDP",
        ),
        DatasetCandidate(
            dataset_id="space_f107",
            group="space",
            description="Daily solar radio flux F10.7. The day's value is used from the next day.",
            modules=("src.space",),
            csv_candidates=("data/space/f107_daily.csv",),
            loader="files",
            display_name="Space F10.7",
        ),
        DatasetCandidate(
            dataset_id="space_flares",
            group="space",
            description=(
                "Monthly GOES flare counts by class. A month is stamped at "
                "month end, after the partial month has been dropped."
            ),
            modules=("src.space",),
            csv_candidates=("data/space/flares_monthly.csv",),
            loader="files",
            display_name="Solar flares",
        ),
        DatasetCandidate(
            dataset_id="space_kp_ap",
            group="space",
            description="Daily geomagnetic Kp and Ap. Used from the next calendar day.",
            modules=("src.space",),
            csv_candidates=("data/space/kp_daily.csv",),
            loader="files",
            display_name="Space Kp/Ap",
        ),
        DatasetCandidate(
            dataset_id="space_solar_wind",
            group="space",
            description="Daily solar-wind speed and density. Used from the next calendar day.",
            modules=("src.space",),
            csv_candidates=("data/space/solar_wind_daily.csv",),
            loader="files",
            display_name="Solar wind",
        ),
        DatasetCandidate(
            dataset_id="space_dst",
            group="space",
            description="Daily disturbance storm time (Dst). Used from the next calendar day.",
            modules=("src.space",),
            csv_candidates=("data/space/dst_daily.csv",),
            loader="files",
            display_name="Space Dst",
        ),
        DatasetCandidate(
            dataset_id="space_neos",
            group="space",
            description=(
                "Monthly near-Earth close approaches and first observations. "
                "Stamped at month end."
            ),
            modules=("src.space",),
            csv_candidates=("data/space/neo_approaches_monthly.csv",),
            loader="files",
            display_name="Near-Earth objects",
        ),
        DatasetCandidate(
            dataset_id="space_launches",
            group="space",
            description="Monthly orbital launch attempts. Stamped at month end.",
            modules=("src.space",),
            csv_candidates=("data/space/launches_monthly.csv",),
            loader="files",
            display_name="Launches",
        ),
        DatasetCandidate(
            dataset_id="space_exoplanets",
            group="space",
            description="Yearly exoplanet discovery counts. Stamped at year end.",
            modules=("src.space",),
            csv_candidates=("data/space/exoplanets_yearly.csv",),
            loader="files",
            display_name="Exoplanets",
        ),
        DatasetCandidate(
            dataset_id="horse_racing",
            group="society",
            description=(
                "Hong Kong race-day counts, field size, average placing, and "
                "median dividend. A race day is used from the next calendar day."
            ),
            modules=("src.domain_forecasts.racing",),
            csv_candidates=("data/raw/horse_racing/performances.csv.gz",),
            loader="horse",
            display_name="Horse racing",
        ),
        DatasetCandidate(
            dataset_id="ai_progress",
            group="technology",
            description=(
                "Monthly arXiv cs.AI, cs.LG, and cs.CL counts, Epoch AI frontier "
                "compute, and new machine-learning GitHub repositories. "
                "Months are stamped at month end. An incomplete GitHub month is dropped."
            ),
            modules=("src.domain_forecasts",),
            csv_candidates=("datasets/ai_progress/arxiv_counts_monthly.csv",),
            loader="files",
            display_name="AI progress",
        ),
        DatasetCandidate(
            dataset_id="quantum_qubits",
            group="technology",
            description=(
                "Annual published qubit-count records, including the series "
                "that excludes unrearranged arrays. Stamped at year end."
            ),
            modules=("src.domain_forecasts",),
            csv_candidates=("datasets/quantum/qubit_record_annual.csv",),
            loader="files",
            display_name="Quantum qubits",
        ),
        DatasetCandidate(
            dataset_id="quantum_gate_error",
            group="technology",
            description=(
                "Annual best reported two-qubit error, as log10. Stamped at year end."
            ),
            modules=("src.domain_forecasts",),
            csv_candidates=("datasets/quantum/two_qubit_error_annual.csv",),
            loader="files",
            display_name="Quantum gate error",
        ),
        DatasetCandidate(
            dataset_id="quantum_neutrons",
            group="technology",
            description=(
                "Oulu monthly cosmic-ray neutron count rate. Stamped at month end."
            ),
            modules=("src.domain_forecasts",),
            csv_candidates=("datasets/quantum/oulu_neutrons_monthly.csv",),
            loader="files",
            display_name="Oulu neutrons",
        ),
        DatasetCandidate(
            dataset_id="arxiv_quant_ph",
            group="technology",
            description="Monthly arXiv quant-ph counts. Stamped at month end.",
            modules=("src.domain_forecasts",),
            csv_candidates=("datasets/quantum/arxiv_quant_ph_monthly.csv",),
            loader="files",
            display_name="arXiv quant-ph",
        ),
        DatasetCandidate(
            dataset_id="qrng",
            group="technology",
            description=(
                "Quantum random symbols from the shipped QRNG file. The file "
                "is an ordered stream with no calendar date."
            ),
            modules=("src.domain_forecasts",),
            csv_candidates=("datasets/quantum/qrng_symbols.csv",),
            loader="qrng",
            display_name="QRNG",
        ),
    )
