#!/usr/bin/env python3
"""Build the small shared starter kits shipped with the image.

Each kit is a short real series, a template config, and a tiny pretrained
checkpoint. New accounts reference these files. They are not copied per user
until someone edits or trains.

Run from the repository root after dependencies are installed:

    python scripts/build_starters.py
"""

from __future__ import annotations

import argparse
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
# Weather and near-Earth are the landing-page demos. They keep a longer
# real series and a slightly wider network than the other samples.
DEMO_IDS = ("weather", "near-earth")
DEMO_ROWS = {"weather": 730, "near-earth": 365}


def _write_series(
    path: Path,
    rows: list[tuple[str, float]],
    *,
    limit: int = ROWS,
    tail: bool = True,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen = rows[-limit:] if tail else rows[:limit]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["date", "value"])
        for day, value in chosen:
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

    start_date = "2024-01-01"
    end_date = "2026-10-31"
    url = build_station_url(
        station=STATION_ID,
        start_date=start_date,
        end_date=end_date,
        data_types=("TMAX",),
    )
    frame = parse_noaa_daily_summaries(
        _get(url, timeout=90).decode("utf-8"), station=STATION_ID
    )
    rows = [
        (day.strftime("%Y-%m-%d"), float(value))
        for day, value in zip(frame["date"], frame["temperature_c"], strict=True)
    ]
    kept = rows[-DEMO_ROWS["weather"] :]
    source = (
        "Daily maximum temperature in °C at Central Park, NOAA GHCNd USW00094728, "
        f"{kept[0][0]} through {kept[-1][0]} "
        f"(the latest {len(kept)} continuous days in the {start_date} to {end_date} "
        "download). Public-domain U.S. government work."
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


def near_earth_rows() -> tuple[list[tuple[str, float]], str]:
    path = ROOT / "space_snapshot" / "near_earth_events.csv"
    counts: Counter[str] = Counter()
    events: list[dict[str, str]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            if record.get("body") != "Earth":
                continue
            day = record["date"][:10]
            counts[day] += 1
            events.append(record)
    if not counts:
        raise RuntimeError("near-earth snapshot has no Earth approaches")
    DATA.mkdir(parents=True, exist_ok=True)
    start = date.fromisoformat(min(counts))
    end = date.fromisoformat(max(counts))
    rows: list[tuple[str, float]] = []
    day = start
    while day <= end:
        key = day.isoformat()
        rows.append((key, float(counts[key])))
        day += timedelta(days=1)
    window = DEMO_ROWS["near-earth"]
    kept = rows[:window]
    window_end = kept[-1][0]
    closest: list[tuple[float, dict[str, str]]] = []
    for record in events:
        if record["date"][:10] > window_end:
            continue
        try:
            distance = float(record["dist_ld"])
        except (TypeError, ValueError):
            continue
        closest.append((distance, record))
    closest.sort(key=lambda item: item[0])
    highlights = []
    for distance, record in closest[:8]:
        highlights.append(
            {
                "name": record.get("name") or record.get("designation") or "Object",
                "designation": record.get("designation") or "",
                "date": record.get("date") or "",
                "dist_ld": round(distance, 4),
                "dist_au": record.get("dist_au") or "",
                "kind": record.get("kind") or "asteroid",
            }
        )
    (DATA / "near-earth-highlights.json").write_text(
        json.dumps(highlights, indent=2) + "\n",
        encoding="utf-8",
    )
    source = (
        "Daily count of catalogued Earth approaches within 0.05 AU, "
        f"from the committed JPL snapshot, {kept[0][0]} through {window_end}. "
        "These are predicted approaches of objects known when the snapshot was saved."
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
    demo = starter_id in DEMO_IDS
    document = {
        "dataset": {"enabled": False, "id": f"starter-{starter_id}"},
        "retrain": {"enabled": False},
        "model": {
            "name": "time_series_transformer",
            "input_dim": 1,
            "output_dim": 1,
            "d_model": 32 if demo else 16,
            "nhead": 4 if demo else 2,
            "num_encoder_layers": 2 if demo else 1,
            "dim_feedforward": 64 if demo else 32,
            "dropout": 0.0,
            "max_seq_len": 128 if demo else 64,
        },
        "data": {
            "source": "csv",
            "path": f"starters/data/{starter_id}.csv",
            "feature_columns": ["value"],
            "target_column": "value",
            "seq_len": 28 if demo else 16,
            "val_split": 0.2,
            "split_mode": "temporal",
            "normalize": True,
            "seed": 0,
        },
        "training": {
            "experiment_name": f"starter-{starter_id}",
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 8 if demo else 4,
            "patience": 3 if demo else 2,
            "device": "cpu",
            "checkpoint_dir": f"starters/models/{starter_id}",
            "log_interval": 1,
            "seed": 0,
        },
        "inference": {
            "device": "cpu",
            "checkpoint_path": f"starters/models/{starter_id}.pt",
        },
        "forecast": {
            "horizon": 7 if demo else 1,
            "strategy": "recursive" if demo else "direct",
            "nominal_coverage": 0.8,
        },
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated starter ids to rebuild. Other catalog entries stay.",
    )
    args = parser.parse_args()
    selected = {part.strip() for part in args.only.split(",") if part.strip()}
    builders = {
        "weather": weather_rows,
        "stocks": stock_rows,
        "crypto": crypto_rows,
        "near-earth": near_earth_rows,
        "lunar": lunar_rows,
    }
    names = {
        "weather": "Central Park temperature",
        "stocks": "AAPL daily return",
        "crypto": "Bitcoin daily return",
        "near-earth": "Near-Earth approaches",
        "lunar": "Moon illumination",
    }
    summaries = {
        "weather": "Two years of real NOAA daily maximum temperature at Central Park. The sample model forecasts the next week.",
        "stocks": "Recent AAPL daily log returns. The starter model forecasts the next return on this sample.",
        "crypto": "Recent BTCUSDT daily log returns. The starter model forecasts the next return on this sample.",
        "near-earth": "Real JPL close-approach counts for the year beginning 2026-10-22, with the closest catalogued objects in that window.",
        "lunar": "Computed moon illumination for 180 days. Predict the next fraction from the sample.",
    }
    units = {
        "weather": "°C",
        "stocks": "daily log return",
        "crypto": "daily log return",
        "near-earth": "daily approaches",
        "lunar": "illumination fraction",
    }
    catalog = []
    for starter_id, builder in builders.items():
        if selected and starter_id not in selected:
            continue
        rows, source = builder()
        limit = DEMO_ROWS.get(starter_id, ROWS)
        if len(rows) < limit:
            raise RuntimeError(f"{starter_id} produced {len(rows)} rows, need {limit}")
        _write_series(
            DATA / f"{starter_id}.csv",
            rows,
            limit=limit,
            tail=starter_id != "near-earth",
        )
        write_template(starter_id)
        train_one(starter_id)
        catalog.append(
            {
                "id": starter_id,
                "name": names[starter_id],
                "summary": summaries[starter_id],
                "source": source,
                "unit": units[starter_id],
                "demo": starter_id in DEMO_IDS,
                "dataset": f"starters/data/{starter_id}.csv",
                "template": f"starters/templates/{starter_id}.yaml",
                "model": f"starters/models/{starter_id}.pt",
            }
        )
    catalog_path = ROOT / "starters" / "catalog.json"
    if selected and catalog_path.is_file():
        existing = json.loads(catalog_path.read_text(encoding="utf-8"))
        by_id = {
            str(row["id"]): row
            for row in existing.get("starters", [])
            if isinstance(row, dict) and "id" in row
        }
        for entry in catalog:
            by_id[entry["id"]] = entry
        ordered = []
        seen: set[str] = set()
        for row in existing.get("starters", []):
            if not isinstance(row, dict) or "id" not in row:
                continue
            starter_id = str(row["id"])
            ordered.append(by_id[starter_id])
            seen.add(starter_id)
        for entry in catalog:
            if entry["id"] not in seen:
                ordered.append(entry)
        catalog = ordered
    catalog_path.write_text(
        json.dumps({"starters": catalog}, indent=2) + "\n",
        encoding="utf-8",
    )
    print("wrote", len(catalog), "starters")


if __name__ == "__main__":
    main()
