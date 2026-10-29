#!/usr/bin/env python3
"""Build the small shared starter kits shipped with the image.

Each kit is a short real series, a template config, and a tiny pretrained
checkpoint. New accounts reference these files. They are not copied per user
until someone edits or trains.

Run from the repository root after dependencies are installed:

    python scripts/build_starters.py
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
import urllib.parse
import urllib.request
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.lunar import lunar_on  # noqa: E402
from src.training.trainer import TimeSeriesTrainer  # noqa: E402

DATA = ROOT / "starters" / "data"
TEMPLATES = ROOT / "starters" / "templates"
MODELS = ROOT / "starters" / "models"
ROWS = 180


def _write_series(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "value"])
        for day, value in rows[-ROWS:]:
            writer.writerow([day, f"{value:.6f}"])


def _get(url: str, timeout: float = 60.0) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "real-neural-forge-starters"}
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def weather_rows() -> tuple[list[tuple[str, float]], str]:
    from src.data.weather import (
        STATION_ID,
        build_station_url,
        parse_noaa_daily_summaries,
    )

    url = build_station_url(
        station=STATION_ID,
        start_date="2024-01-01",
        end_date="2024-06-30",
        data_types=("TMAX",),
    )
    frame = parse_noaa_daily_summaries(
        _get(url, timeout=90).decode("utf-8"), station=STATION_ID
    )
    rows = [
        (day.strftime("%Y-%m-%d"), float(value))
        for day, value in zip(frame["date"], frame["temperature_c"], strict=True)
    ]
    source = (
        "Daily maximum temperature in °C at Central Park, NOAA GHCNd USW00094728, "
        "2024-01-01 through 2024-06-30. Public-domain U.S. government work."
    )
    return rows, source


def _log_returns(closes: list[tuple[str, float]]) -> list[tuple[str, float]]:
    rows: list[tuple[str, float]] = []
    previous: float | None = None
    for day, close in closes:
        if previous is not None and previous > 0 and close > 0:
            rows.append((day, math.log(close / previous)))
        previous = close
    return rows


def stock_rows() -> tuple[list[tuple[str, float]], str]:
    payload = json.loads(
        _get(
            "https://query1.finance.yahoo.com/v8/finance/chart/AAPL?interval=1d&range=2y"
        ).decode("utf-8")
    )
    result = payload["chart"]["result"][0]
    closes: list[tuple[str, float]] = []
    quotes = result["indicators"]["quote"][0]["close"]
    for stamp, close in zip(result["timestamp"], quotes, strict=True):
        if close is None:
            continue
        day = datetime.fromtimestamp(int(stamp), timezone.utc).date().isoformat()
        closes.append((day, float(close)))
    source = (
        "Daily log return of AAPL from the public Yahoo Finance chart API. "
        "The sample is the latest 180 sessions in that download."
    )
    return _log_returns(closes), source


def crypto_rows() -> tuple[list[tuple[str, float]], str]:
    raw = json.loads(
        _get(
            "https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=220"
        ).decode("utf-8")
    )
    closes: list[tuple[str, float]] = []
    for candle in raw:
        moment = datetime.fromtimestamp(int(candle[0]) / 1000, timezone.utc)
        day = moment.date().isoformat()
        closes.append((day, float(candle[4])))
    source = (
        "Daily log return of BTCUSDT from the public Binance vision kline API. "
        "The sample is the latest 180 completed daily closes."
    )
    return _log_returns(closes), source


def lottery_rows() -> tuple[list[tuple[str, float]], str]:
    from src.data.lottery import SOURCE_API, parse_winning_numbers

    query = urllib.parse.urlencode({"$limit": 220, "$order": "draw_date DESC"})
    payload = json.loads(_get(f"{SOURCE_API}?{query}").decode("utf-8"))
    rows: list[tuple[str, float]] = []
    for item in payload:
        draw_date = str(item.get("draw_date", ""))[:10]
        _whites, powerball = parse_winning_numbers(str(item.get("winning_numbers", "")))
        rows.append((draw_date, float(powerball)))
    rows.sort()
    source = (
        "Powerball number (1-26) from the NY Open Data resource d6yy-54nr. "
        "The sample is the latest 180 draws. It is not a way to pick numbers."
    )
    return rows, source


def near_earth_rows() -> tuple[list[tuple[str, float]], str]:
    path = ROOT / "space_snapshot" / "near_earth_events.csv"
    counts: Counter[str] = Counter()
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record.get("body") == "Earth":
                counts[record["date"][:10]] += 1
    if not counts:
        raise RuntimeError("near-earth snapshot has no Earth approaches")
    start = date.fromisoformat(min(counts))
    end = date.fromisoformat(max(counts))
    rows: list[tuple[str, float]] = []
    day = start
    while day <= end:
        key = day.isoformat()
        rows.append((key, float(counts[key])))
        day += timedelta(days=1)
    source = (
        "Daily count of catalogued Earth approaches within 0.05 AU, "
        "taken from the committed JPL snapshot in space_snapshot/near_earth_events.csv."
    )
    return rows, source


def lunar_rows() -> tuple[list[tuple[str, float]], str]:
    end = date(2026, 10, 1)
    start = end - timedelta(days=ROWS + 20)
    rows: list[tuple[str, float]] = []
    day = start
    while day <= end:
        rows.append((day.isoformat(), float(lunar_on(day).illumination)))
        day += timedelta(days=1)
    source = (
        "Moon illumination fraction at 12:00 UTC, computed by src.data.lunar "
        "for the 180 days ending 2026-10-01."
    )
    return rows, source


def write_template(starter_id: str) -> None:
    document = {
        "dataset": {"enabled": False, "id": f"starter-{starter_id}"},
        "retrain": {"enabled": False},
        "model": {
            "name": "time_series_transformer",
            "input_dim": 1,
            "output_dim": 1,
            "d_model": 16,
            "nhead": 2,
            "num_encoder_layers": 1,
            "dim_feedforward": 32,
            "dropout": 0.0,
            "max_seq_len": 64,
        },
        "data": {
            "source": "csv",
            "path": f"starters/data/{starter_id}.csv",
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 16,
            "val_split": 0.2,
            "split_mode": "temporal",
            "normalize": True,
            "seed": 0,
        },
        "training": {
            "experiment_name": f"starter-{starter_id}",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 4,
            "patience": 2,
            "device": "cpu",
            "checkpoint_dir": f"starters/models/{starter_id}",
            "log_interval": 1,
            "seed": 0,
        },
        "inference": {
            "device": "cpu",
            "checkpoint_path": f"starters/models/{starter_id}.pt",
        },
        "forecast": {"horizon": 1, "strategy": "direct"},
        "storage": {"db_path": f"starters/models/{starter_id}-experiments.db"},
    }
    TEMPLATES.mkdir(parents=True, exist_ok=True)
    (TEMPLATES / f"{starter_id}.yaml").write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )


def train_one(starter_id: str) -> None:
    import torch

    config_path = TEMPLATES / f"{starter_id}.yaml"
    result = TimeSeriesTrainer(str(config_path)).train()
    source = MODELS / starter_id / "best_model.pt"
    destination = MODELS / f"{starter_id}.pt"
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint.pop("optimizer_state_dict", None)
    torch.save(checkpoint, destination)
    shutil.rmtree(MODELS / starter_id, ignore_errors=True)
    db_path = MODELS / f"{starter_id}-experiments.db"
    db_path.unlink(missing_ok=True)
    print(
        starter_id,
        "best_loss",
        result.get("best_loss"),
        "bytes",
        destination.stat().st_size,
    )


def main() -> None:
    builders = {
        "weather": weather_rows,
        "stocks": stock_rows,
        "crypto": crypto_rows,
        "lottery": lottery_rows,
        "near-earth": near_earth_rows,
        "lunar": lunar_rows,
    }
    names = {
        "weather": "Central Park temperature",
        "stocks": "AAPL daily return",
        "crypto": "Bitcoin daily return",
        "lottery": "Powerball sample",
        "near-earth": "Near-Earth approaches",
        "lunar": "Moon illumination",
    }
    summaries = {
        "weather": "A short NOAA sample of daily maximum temperature. Predict the next day, or train your own copy.",
        "stocks": "Recent AAPL daily log returns. The starter model forecasts the next return on this sample.",
        "crypto": "Recent BTCUSDT daily log returns. The starter model forecasts the next return on this sample.",
        "lottery": "Recent Powerball numbers. This is a demo that the draw is close to chance, not a picker.",
        "near-earth": "How many catalogued objects pass within 0.05 AU of Earth each day in the saved snapshot.",
        "lunar": "Computed moon illumination for 180 days. Predict the next fraction from the sample.",
    }
    catalog = []
    for starter_id, builder in builders.items():
        rows, source = builder()
        if len(rows) < ROWS:
            raise RuntimeError(f"{starter_id} produced {len(rows)} rows, need {ROWS}")
        _write_series(DATA / f"{starter_id}.csv", rows)
        write_template(starter_id)
        train_one(starter_id)
        catalog.append(
            {
                "id": starter_id,
                "name": names[starter_id],
                "summary": summaries[starter_id],
                "source": source,
                "dataset": f"starters/data/{starter_id}.csv",
                "template": f"starters/templates/{starter_id}.yaml",
                "model": f"starters/models/{starter_id}.pt",
            }
        )
    (ROOT / "starters" / "catalog.json").write_text(
        json.dumps({"starters": catalog}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("wrote", len(catalog), "starters")


if __name__ == "__main__":
    main()
