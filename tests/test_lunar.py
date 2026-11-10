"""Offline lunar features."""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import pandas as pd
import pytest

from src.data.lunar import (
    LUNAR_MODEL_COLUMNS,
    chinese_date,
    join_lunar,
    julian_day,
    lunar_feature_frame,
    lunar_on,
)
from src.data.windows import prepare_csv_series
from src.retrain.discover import discover
from src.utils.paths import REPO_ROOT


def test_julian_day_round_trips_j2000():
    moment = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
    assert julian_day(moment) == 2451545.0


def test_named_events_and_chinese_new_year():
    eclipse = lunar_on(date(2024, 4, 8))
    assert eclipse.is_new_moon
    assert eclipse.eclipse_kind == "solar"
    assert eclipse.chinese.year == 2024
    assert eclipse.chinese.zodiac == "Dragon"

    full = lunar_on(date(2024, 10, 17))
    assert full.is_full_moon
    assert full.is_supermoon
    assert full.eclipse_kind is None
    assert full.named_phase == "full"
    assert full.illumination > 0.98
    assert abs(full.moon_phase - 0.5) < 0.03

    quiet = lunar_on(date(2024, 5, 8))
    assert quiet.is_new_moon
    assert quiet.eclipse_kind is None

    lunar_eclipse = lunar_on(date(2024, 9, 18))
    assert lunar_eclipse.is_full_moon
    assert lunar_eclipse.eclipse_kind == "lunar"
    assert lunar_eclipse.is_supermoon

    assert (
        chinese_date(date(2024, 2, 10))
        .text()
        .startswith("Dragon year 2024, month 1 day 1")
    )
    assert chinese_date(date(2024, 2, 9)).year == 2023
    assert chinese_date(date(2024, 9, 17)).month == 8
    assert chinese_date(date(2024, 9, 17)).day == 15
    assert chinese_date(date(2023, 3, 22)).leap
    assert chinese_date(date(2023, 3, 22)).month == 2
    assert chinese_date(date(2020, 5, 23)).leap
    assert chinese_date(date(2020, 5, 23)).month == 4
    assert (
        chinese_date(date(2025, 1, 29))
        .text()
        .startswith("Snake year 2025, month 1 day 1")
    )
    assert chinese_date(date(2025, 7, 25)).leap
    assert (
        chinese_date(date(2000, 2, 5))
        .text()
        .startswith("Dragon year 2000, month 1 day 1")
    )


def test_vectorized_frame_matches_single_days():
    days = [date(2020, 1, 1) + timedelta(days=17 * index) for index in range(12)]
    frame = lunar_feature_frame(days)
    assert list(frame.columns) == list(LUNAR_MODEL_COLUMNS)
    for index, day in enumerate(days):
        single = lunar_on(day).as_features()
        for name in LUNAR_MODEL_COLUMNS:
            assert frame.iloc[index][name] == pytest.approx(single[name], abs=1e-6)


def test_join_lunar_appends_columns_and_refuses_overlaps():
    frame = pd.DataFrame({"observed_on": ["2024-04-08", "2024-10-17"], "value": [1, 2]})
    joined = join_lunar(frame, "observed_on")
    assert set(LUNAR_MODEL_COLUMNS) <= set(joined.columns)
    assert list(joined["value"]) == [1, 2]
    try:
        join_lunar(joined, "observed_on")
    except ValueError as exc:
        assert "already present" in str(exc)
    else:
        raise AssertionError("overlapping lunar columns should be rejected")


def test_future_lunar_widens_the_training_windows(tmp_path):
    rows = []
    day = date(2024, 1, 1)
    for index in range(12):
        rows.append(
            {
                "observed_on": (day + timedelta(days=index)).isoformat(),
                "value": index + 1,
            }
        )
    path = tmp_path / "series.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    prepared = prepare_csv_series(
        {
            "path": str(path),
            "feature_columns": ["value"],
            "target_column": "value",
            "date_column": "observed_on",
            "seq_len": 4,
            "val_split": 0.25,
            "future_lunar": True,
        }
    )
    assert prepared.inputs.shape[-1] == 1 + len(LUNAR_MODEL_COLUMNS)
    assert prepared.feature_names[0] == "value"
    assert prepared.feature_names[-1] == f"future_{LUNAR_MODEL_COLUMNS[-1]}"


def test_lottery_configs_are_not_shipped():
    specs, errors = discover(REPO_ROOT / "config")
    assert errors == []
    ids = {spec.id for spec in specs}
    assert "lottery" not in ids
    assert "lottery_lunar" not in ids
    assert "lottery_tickets" not in ids
    assert "lottery_weather" not in ids
    assert not (REPO_ROOT / "config" / "lottery.yaml").exists()
