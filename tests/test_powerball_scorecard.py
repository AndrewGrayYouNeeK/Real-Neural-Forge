"""Tickets, weather join, and the monthly Powerball scorecard."""

from __future__ import annotations

import json
import zlib
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts import run_powerball_month as job
from src.data.draw_weather import (
    CENTRAL_PARK_COLUMN,
    MAX_WEATHER_GAP_DAYS,
    TALLAHASSEE_COLUMNS,
    WeatherJoin,
    align_station_to_draws,
    complete_weather_rows,
    join_draw_weather,
)
from src.data.lottery import PowerballDraw, write_draws_csv
from src.evaluation.powerball_models import (
    PLAIN_FEATURES,
    forecast_next_ticket,
    make_windows,
    score_holdout_predictions,
    weather_feature_columns,
)
from src.evaluation.powerball_tickets import (
    CHANCE_WHITE_MATCHES,
    TICKET_COMBINATIONS,
    WHITE_COMBINATIONS,
    HistoryDraw,
    actuals_by_date,
    as_of_for_month,
    assemble_month_log,
    default_forecast_month,
    draw_dates_for_month,
    expected_white_matches,
    history_from_frame,
    hot_pairs_ticket,
    most_hit_ticket,
    neighbor_ticket,
    overdue_ticket,
    pattern_tickets,
    prize_tier,
    random_ticket,
    repair_number_vector,
    score_logged_months,
    score_ticket,
    tier_probability,
    validate_ticket,
    white_match_probability,
)

ROOT = Path(__file__).resolve().parents[1]
TINY = {
    "seq_len": 4,
    "val_split": 0.2,
    "ticket_val_split": 0.0,
    "epochs": 1,
    "patience": 0,
    "batch_size": 8,
    "learning_rate": 0.001,
    "d_model": 8,
    "nhead": 2,
    "num_layers": 1,
    "dim_feedforward": 16,
    "dropout": 0.0,
    "seed": 0,
}


def _history() -> list[HistoryDraw]:
    return [
        HistoryDraw(date(2026, 9, 26), (1, 2, 3, 4, 5), 1),
        HistoryDraw(date(2026, 9, 28), (1, 2, 3, 4, 6), 1),
    ]


def _numbered_frame(rows: int, *, weather: bool = False) -> pd.DataFrame:
    records = []
    for index in range(rows):
        whites = [index + offset + 1 for offset in range(5)]
        record = {
            "draw_date": pd.Timestamp("2020-01-01") + pd.Timedelta(days=index),
            "white_1": whites[0],
            "white_2": whites[1],
            "white_3": whites[2],
            "white_4": whites[3],
            "white_5": whites[4],
            "powerball": (index % 26) + 1,
        }
        if weather:
            record["tmax_c"] = 20.0 + index
            record["tmin_c"] = 10.0 + index
            record["prcp_mm"] = float(index % 3)
        records.append(record)
    return pd.DataFrame(records)


def test_prize_probabilities_match_the_published_matrix():
    total = sum(
        tier_probability(matches, hit) for matches in range(6) for hit in (False, True)
    )
    assert total == pytest.approx(1.0)
    assert WHITE_COMBINATIONS == 11_238_513
    assert TICKET_COMBINATIONS == 292_201_338
    assert white_match_probability(5) == pytest.approx(1 / WHITE_COMBINATIONS)
    assert tier_probability(5, True) == pytest.approx(1 / 292_201_338)
    assert expected_white_matches() == pytest.approx(25 / 69)
    assert CHANCE_WHITE_MATCHES == pytest.approx(25 / 69)
    assert prize_tier(5, True) == {
        "tier": "jackpot",
        "base_prize_usd": None,
        "white_matches": 5,
        "powerball_hit": True,
    }
    assert prize_tier(4, True)["base_prize_usd"] == 50_000
    assert prize_tier(0, False)["tier"] == "none"
    assert prize_tier(0, False)["base_prize_usd"] == 0


def test_ticket_validation_and_repair():
    ticket = validate_ticket({"whites": [1, 2, 3, 4, 5], "powerball": 26})
    assert ticket["whites"] == [1, 2, 3, 4, 5]
    with pytest.raises(ValueError, match="sorted"):
        validate_ticket({"whites": [5, 4, 3, 2, 1], "powerball": 1})
    with pytest.raises(ValueError, match="unique"):
        validate_ticket({"whites": [1, 1, 2, 3, 4], "powerball": 1})
    with pytest.raises(ValueError, match="1..69"):
        validate_ticket({"whites": [1, 2, 3, 4, 70], "powerball": 1})
    repaired = repair_number_vector(np.array([1.2, 1.2, 1.2, 69.6, 69.4, 26.2]))
    assert repaired["whites"] == [1, 2, 3, 68, 69]
    assert len(set(repaired["whites"])) == 5
    assert repaired["whites"] == sorted(repaired["whites"])
    assert repaired["powerball"] == 26
    scored = score_ticket(
        {"whites": [1, 2, 3, 4, 7], "powerball": 6},
        {"whites": [1, 2, 3, 4, 5], "powerball": 6},
    )
    assert scored == {
        "white_matches": 4,
        "powerball_hit": True,
        "tier": "match_4_pb",
        "base_prize_usd": 50_000,
    }


def test_schedule_and_as_of_cutoff():
    november = draw_dates_for_month("2026-11")
    assert len(november) == 13
    assert november[0] == date(2026, 11, 2)
    assert november[-1] == date(2026, 11, 30)
    assert all(day.weekday() in (0, 2, 5) for day in november)
    october = draw_dates_for_month("2026-10")
    assert len(october) == 13
    assert october[0] == date(2026, 10, 3)
    assert draw_dates_for_month("2021-08")[0] == date(2021, 8, 23)
    with pytest.raises(ValueError, match="Monday/Wednesday/Saturday"):
        draw_dates_for_month("2021-07")
    assert default_forecast_month(date(2026, 10, 1)) == "2026-10"
    assert default_forecast_month(date(2026, 10, 21)) == "2026-11"
    assert default_forecast_month(date(2026, 12, 2)) == "2027-01"
    assert as_of_for_month("2026-10") == date(2026, 9, 30)
    assert as_of_for_month("2026-11") == date(2026, 10, 31)


def test_pattern_tickets_are_deterministic_and_the_random_seed_is_public():
    history = _history()
    assert most_hit_ticket(history) == {"whites": [1, 2, 3, 4, 5], "powerball": 1}
    assert hot_pairs_ticket(history) == {"whites": [1, 2, 3, 4, 5], "powerball": 1}
    assert neighbor_ticket([HistoryDraw(date(2026, 9, 30), (1, 2, 3, 4, 69), 26)]) == {
        "whites": [2, 3, 4, 5, 68],
        "powerball": 26,
    }
    assert overdue_ticket(history) == {"whites": [7, 8, 9, 10, 11], "powerball": 2}
    # Gaps differ, so the five most overdue whites are not chosen in numeric order.
    early = [
        (10, 20, 30, 40, 69),
        (1, 11, 21, 31, 41),
        (2, 12, 22, 32, 42),
        (3, 13, 23, 33, 43),
        (4, 14, 24, 34, 44),
    ]
    stale = [
        HistoryDraw(date(2020, 1, 1 + index), whites, 1)
        for index, whites in enumerate(early)
    ]
    refresh = [
        number for draw in early for number in draw if number not in {1, 2, 3, 4, 69}
    ]
    seen_early = {ball for draw in early for ball in draw}
    later = refresh + [number for number in range(1, 70) if number not in seen_early]
    for offset in range(0, len(later), 5):
        chunk = later[offset : offset + 5]
        while len(chunk) < 5:
            chunk.append(later[0])
        stale.append(
            HistoryDraw(
                date(2020, 1, 1) + timedelta(days=len(stale)),
                (chunk[0], chunk[1], chunk[2], chunk[3], chunk[4]),
                2,
            )
        )
    assert overdue_ticket(stale)["whites"] == [1, 2, 3, 4, 69]
    first = pattern_tickets(history, date(2026, 11, 2), "2026-11")
    second = pattern_tickets(history, date(2026, 11, 2), "2026-11")
    assert first == second
    seed_text = "powerball-control-2026-11-2026-11-02"
    seed = zlib.crc32(seed_text.encode("utf-8")) & 0xFFFFFFFF
    generator = np.random.default_rng(seed)
    whites = sorted(
        int(number)
        for number in generator.choice(np.arange(1, 70), size=5, replace=False)
    )
    powerball = int(generator.integers(1, 27))
    assert random_ticket(date(2026, 11, 2), "2026-11") == {
        "whites": whites,
        "powerball": powerball,
    }
    other_seed = zlib.crc32(b"powerball-control-2026-11-2026-11-04") & 0xFFFFFFFF
    assert seed != other_seed


def test_month_log_repeats_forecasts_and_scorecard_uses_only_known_draws():
    history = _history()
    model = {"whites": [1, 2, 3, 4, 5], "powerball": 6}
    weather = {"whites": [1, 2, 3, 10, 11], "powerball": 7}
    log = assemble_month_log(
        month="2026-11",
        history=history,
        model_ticket=model,
        weather_ticket=weather,
        training_draws_through=date(2026, 9, 28),
        generated_on=date(2026, 10, 21),
        weather_note="test",
        feature_columns=["white_1", "tmax_c"],
        hyperparameters={"seed": 0},
    )
    assert log["purchases"] is False
    first = log["draws"][0]["tickets"]
    later = log["draws"][1]["tickets"]
    for name in (
        "model",
        "model_weather",
        "most_hit",
        "hot_pairs",
        "neighbors",
        "overdue",
    ):
        assert first[name] == later[name]
    assert first["random"] == random_ticket(date(2026, 11, 2), "2026-11")
    assert later["random"] == random_ticket(date(2026, 11, 4), "2026-11")
    before = json.dumps(log)
    actual = {"whites": [1, 2, 3, 4, 20], "powerball": 6}
    card = score_logged_months([log], {date(2026, 11, 2): actual})
    assert json.dumps(log) == before
    month = card["months"][0]
    assert month["unscored_dates"][0] == "2026-11-04"
    assert len(month["unscored_dates"]) == 12
    model_totals = card["cumulative"]["model"]
    assert model_totals["scored"] == 1
    assert model_totals["white_matches"] == 4
    assert model_totals["powerball_hits"] == 1
    assert model_totals["prize_tickets"] == 1
    assert model_totals["expected_white_matches"] == pytest.approx(25 / 69)
    assert model_totals["expected_powerball_hits"] == pytest.approx(1 / 26)
    weather_totals = card["cumulative"]["model_weather"]
    assert weather_totals["white_matches"] == 3
    assert weather_totals["powerball_hits"] == 0
    assert weather_totals["prize_tickets"] == 1
    assert "model:" in card["summary"]
    assert "model_weather:" in card["summary"]


def test_windows_exclude_the_target_row_including_its_weather():
    frame = _numbered_frame(6, weather=True)
    columns = weather_feature_columns(frame.columns)
    windows, labels = make_windows(frame, columns, seq_len=3)
    assert windows.shape == (3, 3, len(columns))
    tmax_index = columns.index("tmax_c")
    np.testing.assert_array_equal(windows[0, :, tmax_index], [20.0, 21.0, 22.0])
    assert labels[0, 5] == frame.loc[3, "powerball"]
    assert 23.0 not in windows[0, :, tmax_index]

    seen: dict[str, np.ndarray] = {}

    def fake_fit(inputs, targets, hyper, *, val_split):  # noqa: ARG001
        seen["last_train"] = np.array(inputs[-1])
        return object(), object(), {"epochs_ran": 1, "n_val": 0}

    def fake_predict(model, scaler, inputs):  # noqa: ARG001
        seen["forecast"] = np.array(inputs[0])
        return np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr("src.evaluation.powerball_models.fit_transformer", fake_fit)
    monkeypatch.setattr(
        "src.evaluation.powerball_models.predict_original", fake_predict
    )
    try:
        ticket, _info = forecast_next_ticket(frame, columns, {**TINY, "seq_len": 3})
    finally:
        monkeypatch.undo()
    features = frame.loc[:, columns].to_numpy(dtype=np.float64)
    np.testing.assert_array_equal(seen["forecast"], features[-3:])
    np.testing.assert_array_equal(seen["last_train"], features[2:5])
    assert not np.array_equal(seen["forecast"], seen["last_train"])
    assert ticket == {"whites": [1, 2, 3, 4, 5], "powerball": 6}


def test_weather_gap_longer_than_two_days_stays_missing():
    observations = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "tmax_c": [1.0, 2.0],
            "tmin_c": [0.0, 0.5],
            "prcp_mm": [0.0, 3.0],
        }
    )
    aligned = align_station_to_draws(
        pd.Series(pd.to_datetime(["2026-01-01", "2026-01-04", "2026-01-05"])),
        observations,
        TALLAHASSEE_COLUMNS,
        max_gap_days=MAX_WEATHER_GAP_DAYS,
    )
    assert aligned.loc[0, "tmax_c"] == 1.0
    assert aligned.loc[1, "tmax_c"] == 2.0
    assert aligned.loc[1, "weather_gap_days"] == 2
    assert pd.isna(aligned.loc[2, "tmax_c"])
    assert pd.isna(aligned.loc[2, "prcp_mm"])
    assert pd.isna(aligned.loc[2, "weather_gap_days"])

    draws = pd.DataFrame(
        {
            "draw_date": pd.to_datetime(["2026-01-01", "2026-01-05"]),
            "white_1": [1, 2],
            "white_2": [3, 4],
            "white_3": [5, 6],
            "white_4": [7, 8],
            "white_5": [9, 10],
            "powerball": [1, 2],
        }
    )
    park = pd.DataFrame({"date": pd.to_datetime(["2026-01-01"]), "tmax_c": [15.0]})
    joined = join_draw_weather(draws, observations, park)
    assert joined.central_park_column is None
    assert CENTRAL_PARK_COLUMN not in joined.frame.columns
    assert "omitted" in joined.central_park_note
    kept = complete_weather_rows(joined.frame, TALLAHASSEE_COLUMNS)
    assert kept["draw_date"].dt.strftime("%Y-%m-%d").tolist() == ["2026-01-01"]


def test_training_frame_stops_at_the_day_before_the_month(tmp_path):
    draws = [
        PowerballDraw(date(2026, 9, 28), (1, 2, 3, 4, 5), 6),
        PowerballDraw(date(2026, 9, 30), (6, 7, 8, 9, 10), 7),
        PowerballDraw(date(2026, 10, 3), (11, 12, 13, 14, 15), 8),
    ]
    path = write_draws_csv(tmp_path / "powerball.csv", draws)
    frame = pd.read_csv(path)
    frame["tmax_c"] = [20.0, np.nan, 21.0]
    frame["tmin_c"] = [10.0, 11.0, 12.0]
    frame["prcp_mm"] = [0.0, 0.0, 1.0]
    joined = WeatherJoin(
        frame=frame,
        tallahassee_columns=TALLAHASSEE_COLUMNS,
        central_park_column=None,
        tallahassee_gaps=1,
        central_park_note="not requested",
    )
    trainable = job._training_frame(frame, date(2026, 9, 30), joined)
    assert pd.to_datetime(trainable["draw_date"]).dt.date.tolist() == [
        date(2026, 9, 28)
    ]
    history = history_from_frame(trainable, through=date(2026, 9, 30))
    assert [item.draw_date for item in history] == [date(2026, 9, 28)]


def test_existing_month_log_is_not_rewritten(tmp_path, monkeypatch):
    draws = [PowerballDraw(date(2026, 9, 28), (1, 2, 3, 4, 5), 6)]
    csv_path = write_draws_csv(tmp_path / "powerball.csv", draws)
    frame = pd.read_csv(csv_path)
    for column, value in (("tmax_c", 20.0), ("tmin_c", 10.0), ("prcp_mm", 0.0)):
        frame[column] = value
    joined = WeatherJoin(frame, TALLAHASSEE_COLUMNS, None, 0, "not requested")
    monkeypatch.setattr(job, "_load_weather", lambda _draws: joined)

    def fail_forecast(*_args, **_kwargs):
        raise AssertionError("an existing month log was retrained")

    monkeypatch.setattr(job, "forecast_next_ticket", fail_forecast)
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    month_path = log_dir / "2026-10.json"
    month_path.write_text('{"month": "2026-10", "sentinel": true, "draws": []}\n')
    job.publish_month(
        "2026-10",
        hyper=TINY,
        refresh=False,
        log_dir=str(log_dir),
        draws_path=str(csv_path),
    )
    assert json.loads(month_path.read_text())["sentinel"] is True
    scorecard = json.loads((log_dir / "scorecard.json").read_text())
    assert "No logged draw" in scorecard["summary"]


def test_one_epoch_forecast_is_a_legal_ticket():
    frame = _numbered_frame(8, weather=True)
    columns = weather_feature_columns(list(frame.columns))
    plain, _plain_info = forecast_next_ticket(frame, PLAIN_FEATURES, TINY)
    weather, _weather_info = forecast_next_ticket(frame, columns, TINY)
    assert plain == validate_ticket(plain)
    assert weather == validate_ticket(weather)


def test_holdout_scores_count_matches_against_chance():
    predicted = np.array(
        [[1, 2, 3, 4, 5, 10], [1, 2, 3, 4, 6, 11]],
        dtype=float,
    )
    actual = np.array(
        [[1, 2, 3, 4, 5, 10], [1, 2, 3, 4, 7, 12]],
        dtype=float,
    )
    metrics = score_holdout_predictions(predicted, actual)
    assert metrics["n"] == 2
    assert metrics["powerball_hits"] == 1
    assert metrics["mean_white_matches"] == pytest.approx(4.5)
    assert metrics["full_ticket_hits"] == 1
    assert metrics["chance_white_matches"] == pytest.approx(25 / 69)
    assert metrics["chance_powerball_hit_rate"] == pytest.approx(1 / 26)


def test_monthly_workflow_uses_the_forecast_month_and_github_token_only():
    text = (ROOT / ".github/workflows/powerball-monthly.yml").read_text()
    assert 'cron: "0 16 1 * *"' in text
    assert "workflow_dispatch:" in text
    assert "GH_TOKEN: ${{ github.token }}" in text
    assert "secrets." not in text
    assert "contents: write" in text
    assert "pull-requests: write" in text
    assert 'if [ "$day" = "01" ]; then' in text
    assert "+1 month" in text
    assert 'python -m scripts.run_powerball_month --month "$month"' in text
    assert "predictions/powerball-${month}" in text
    assert "git push origin main" not in text
    assert "No tickets are purchased." in text


def test_actuals_round_trip_through_the_draw_table(tmp_path):
    path = write_draws_csv(
        tmp_path / "powerball.csv",
        [PowerballDraw(date(2026, 10, 3), (5, 12, 19, 33, 61), 14)],
    )
    actuals = actuals_by_date(pd.read_csv(path))
    assert actuals[date(2026, 10, 3)] == {
        "whites": [5, 12, 19, 33, 61],
        "powerball": 14,
    }
