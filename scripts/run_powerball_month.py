#!/usr/bin/env python3
"""Log next month's Powerball tickets and score the months already drawn.

Run from the repository root:

    python -m scripts.run_powerball_month --month 2026-11

The draw file and NOAA weather are refreshed from their public sources.
Existing month logs are left unchanged. The scorecard is rebuilt from those
logs and the latest winning numbers. No ticket is purchased.

``--holdout-output`` also writes the plain-versus-weather temporal comparison.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd  # noqa: E402

from src.data.draw_weather import (  # noqa: E402
    MAX_WEATHER_GAP_DAYS,
    TALLAHASSEE_STATION,
    WEATHER_LICENCE,
    WeatherJoin,
    complete_weather_rows,
    fetch_central_park_tmax,
    fetch_tallahassee,
    join_draw_weather,
)
from src.data.lottery import fetch_and_write, read_draw_table, source_metadata  # noqa: E402
from src.evaluation.powerball_models import (  # noqa: E402
    PLAIN_FEATURES,
    compare_plain_and_weather,
    forecast_next_ticket,
    hyperparameters_from_config,
    weather_feature_columns,
)
from src.evaluation.powerball_tickets import (  # noqa: E402
    actuals_by_date,
    as_of_for_month,
    assemble_month_log,
    default_forecast_month,
    history_from_frame,
    load_month_logs,
    score_logged_months,
    write_json,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.paths import resolve_repo_path  # noqa: E402

DRAWS_PATH = "data/powerball.csv"
LOG_DIR = "predictions/powerball"
SCORECARD_NAME = "scorecard.json"


def _training_frame(draws: pd.DataFrame, through: date, join: WeatherJoin) -> pd.DataFrame:
    frame = join.frame.copy()
    frame["draw_date"] = pd.to_datetime(frame["draw_date"])
    cutoff = pd.Timestamp(through)
    frame = frame.loc[frame["draw_date"] <= cutoff].reset_index(drop=True)
    columns = weather_feature_columns(list(frame.columns))
    return complete_weather_rows(frame, [column for column in columns if column not in PLAIN_FEATURES])


def _load_weather(draws: pd.DataFrame) -> WeatherJoin:
    dates = pd.to_datetime(draws["draw_date"])
    start = (dates.min() - pd.Timedelta(days=3)).strftime("%Y-%m-%d")
    end = max(dates.max(), pd.Timestamp(date.today())).strftime("%Y-%m-%d")
    tallahassee = fetch_tallahassee(start, end)
    central_park = None
    try:
        central_park = fetch_central_park_tmax(start, end)
    except RuntimeError as exc:
        print(f"Central Park weather unavailable: {exc}")
    join = join_draw_weather(draws, tallahassee, central_park)
    print(
        f"Tallahassee gaps: {join.tallahassee_gaps}. Central Park: {join.central_park_note}"
    )
    return join


def publish_month(
    month: str,
    *,
    hyper: dict,
    refresh: bool = True,
    log_dir: str = LOG_DIR,
    draws_path: str = DRAWS_PATH,
) -> dict:
    """Write ``month`` if it is absent, then rebuild the scorecard."""
    destination = resolve_repo_path(draws_path)
    if refresh or not destination.is_file():
        fetch_and_write(destination)
    draws = read_draw_table(destination)
    join = _load_weather(draws)
    through = as_of_for_month(month)
    folder = resolve_repo_path(log_dir)
    log_path = folder / f"{month}.json"
    if log_path.is_file():
        print(f"Keeping existing tickets in {log_path}")
    else:
        trainable = _training_frame(draws, through, join)
        if len(trainable) <= int(hyper["seq_len"]):
            raise ValueError(f"not enough weather-aligned draws on or before {through}")
        features = weather_feature_columns(list(trainable.columns))
        model_ticket, model_info = forecast_next_ticket(trainable, PLAIN_FEATURES, hyper)
        weather_ticket, weather_info = forecast_next_ticket(trainable, features, hyper)
        history = history_from_frame(trainable, through=through)
        weather_note = (
            f"Tallahassee airport {TALLAHASSEE_STATION} daily max temp, min temp, "
            f"and precipitation on dates already inside the input window, carried "
            f"back at most {MAX_WEATHER_GAP_DAYS} days. The draw being forecast "
            f"does not contribute weather. Central Park: {join.central_park_note}."
        )
        log = assemble_month_log(
            month=month,
            history=history,
            model_ticket=model_ticket,
            weather_ticket=weather_ticket,
            training_draws_through=history[-1].draw_date,
            generated_on=date.today(),
            weather_note=weather_note,
            feature_columns=features,
            hyperparameters={
                **hyper,
                "plain_epochs_ran": model_info["epochs_ran"],
                "weather_epochs_ran": weather_info["epochs_ran"],
                "training_rows": int(len(trainable)),
                "tallahassee_station": TALLAHASSEE_STATION,
                "weather_licence": WEATHER_LICENCE,
                "draws_source": source_metadata(),
            },
        )
        write_json(log_path, log)
        print(f"Wrote {log_path}")
    scorecard = score_logged_months(load_month_logs(folder), actuals_by_date(draws))
    scorecard_path = write_json(folder / SCORECARD_NAME, scorecard)
    print(scorecard["summary"])
    print(f"Wrote {scorecard_path}")
    return scorecard


def write_holdout_report(path: str, hyper: dict, *, refresh: bool = True) -> dict:
    """Compare the plain and weather models on one shared temporal holdout."""
    destination = resolve_repo_path(DRAWS_PATH)
    if refresh or not destination.is_file():
        fetch_and_write(destination)
    draws = read_draw_table(destination)
    join = _load_weather(draws)
    frame = complete_weather_rows(
        join.frame,
        [column for column in weather_feature_columns(list(join.frame.columns)) if column not in PLAIN_FEATURES],
    )
    report = compare_plain_and_weather(frame, hyper)
    report["central_park"] = join.central_park_note
    report["tallahassee_station"] = "USW00093805"
    report["weather_licence"] = WEATHER_LICENCE
    report["draws_source"] = source_metadata()
    report["note"] = (
        "Both models see the same draws. The weather model also sees Tallahassee "
        "observations on the dates inside each input window. Early stopping uses "
        "this holdout, so squared error is slightly optimistic for both. A gap "
        "near zero means weather did not change the Powerball forecast."
    )
    write_json(resolve_repo_path(path), report)
    plain = report["plain"]["metrics"]
    weather = report["weather"]["metrics"]
    print(
        "Holdout Powerball hits "
        f"plain {plain['powerball_hits']}/{plain['n']} "
        f"weather {weather['powerball_hits']}/{weather['n']} "
        f"MAE plain {plain['powerball_mae']:.3f} weather {weather['powerball_mae']:.3f}"
    )
    print(
        "Mean white matches "
        f"plain {plain['mean_white_matches']:.3f} "
        f"weather {weather['mean_white_matches']:.3f} "
        f"chance {plain['chance_white_matches']:.3f}"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", default="", help="YYYY-MM to log. Default: the upcoming month.")
    parser.add_argument("--config", default="config/lottery_tickets.yaml")
    parser.add_argument("--holdout-output", default="", help="Also write the weather comparison JSON.")
    parser.add_argument("--no-fetch", action="store_true", help="Use the draw CSV already on disk.")
    args = parser.parse_args(argv)
    hyper = hyperparameters_from_config(load_config(args.config))
    refresh = not args.no_fetch
    month = args.month or default_forecast_month(date.today())
    publish_month(month, hyper=hyper, refresh=refresh)
    if args.holdout_output:
        write_holdout_report(args.holdout_output, hyper, refresh=False)


if __name__ == "__main__":
    main()
