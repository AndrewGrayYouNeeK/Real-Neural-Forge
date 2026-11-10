"""Outcome tracking: record forecasts, score actuals, keep rolling coverage."""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.api import app
from src.evaluation.tracking import (
    WINDOW_SIZES,
    OutcomeTracker,
    clear_tracker_cache,
    steps_from_payload,
)
from src.workspace.store import SqliteSql


def _tracker(tmp_path):
    clear_tracker_cache()
    return OutcomeTracker(SqliteSql(tmp_path / "evaluation.db")).ensure()


def test_scores_coverage_and_tail_hit_rates(tmp_path):
    tracker = _tracker(tmp_path)
    for index in range(4):
        tracker.record(
            "weather",
            [
                {
                    "horizon": 1,
                    "target_time": f"2026-01-{index + 1:02d}",
                    "p10": 0.0,
                    "p50": 1.0,
                    "p90": 2.0,
                }
            ],
        )
    # Two inside the 80% interval, one below p10, one above p90.
    scored = tracker.score_actuals(
        "weather",
        {
            "2026-01-01": 1.0,
            "2026-01-02": 2.0,
            "2026-01-03": -1.0,
            "2026-01-04": 3.0,
        },
    )
    assert scored == 4
    stats = tracker.preferred_coverage("weather", 1)
    assert stats is not None
    assert stats.n == 4
    assert stats.window_size == WINDOW_SIZES[0]
    assert stats.coverage_80 == 0.5
    assert stats.p10_hit_rate == 0.75
    assert stats.p90_hit_rate == 0.75


def test_rolling_window_keeps_only_the_latest_rows(tmp_path):
    tracker = _tracker(tmp_path)
    for index in range(WINDOW_SIZES[0] + 2):
        day = index + 1
        tracker.record(
            "neo",
            [
                {
                    "horizon": 2,
                    "target_time": f"2026-06-{day:02d}"
                    if day <= 28
                    else f"2026-07-{day - 28:02d}",
                    "p10": 0.0,
                    "p50": 1.0,
                    "p90": 2.0,
                }
            ],
        )
    actuals = {}
    rows = tracker.sql.query(
        "SELECT target_time FROM forecast_outcomes WHERE model_id = 'neo' ORDER BY target_time"
    )
    for offset, row in enumerate(rows):
        # The early rows miss the interval. The latest 30 all land inside it.
        actuals[row["target_time"]] = 1.0 if offset >= 2 else 10.0
    assert tracker.score_actuals("neo", actuals) == len(rows)
    short = next(
        item for item in tracker.coverage("neo", horizon=2) if item.window_size == 30
    )
    wide = [
        item for item in tracker.coverage("neo", horizon=2) if item.window_size != 30
    ]
    assert short.n == 30
    assert short.coverage_80 == 1.0
    assert wide == []


def test_point_forecasts_do_not_invent_coverage(tmp_path):
    tracker = _tracker(tmp_path)
    saved = tracker.record(
        "champion",
        [{"horizon": 1, "target_time": "2026-02-01", "p50": 4.0}],
    )
    assert saved
    assert tracker.score_actuals("champion", {"2026-02-01": 4.0}) == 1
    assert tracker.coverage("champion") == []


def test_repeat_open_forecast_is_not_stored_twice(tmp_path):
    tracker = _tracker(tmp_path)
    step = {
        "horizon": 1,
        "target_time": "2026-03-01",
        "p10": 1.0,
        "p50": 2.0,
        "p90": 3.0,
    }
    assert tracker.record("weather", [step])
    assert tracker.record("weather", [step]) == []
    rows = tracker.sql.query("SELECT id FROM forecast_outcomes")
    assert len(rows) == 1


def test_steps_from_payload_keeps_existing_shapes():
    assert steps_from_payload({"prediction": [1.5]}) == [
        {"horizon": 1, "p50": 1.5, "p10": None, "p90": None}
    ]
    multi = steps_from_payload(
        {"p50": [1.0, 2.0], "p10": [0.0, 1.0], "p90": [2.0, 3.0]}
    )
    assert [step["horizon"] for step in multi] == [1, 2]
    assert multi[1]["p90"] == 3.0


def test_predict_response_stays_a_point_and_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("NEURAL_FORGE_EVAL_DB", str(tmp_path / "evaluation.db"))
    clear_tracker_cache()
    with TestClient(app, raise_server_exceptions=True) as client:
        response = client.post(
            "/predict", json={"sequence": [[0.1], [0.2], [0.3], [0.4]]}
        )
    assert response.status_code == 200
    assert set(response.json()) == {"prediction"}
    clear_tracker_cache()
    tracker = OutcomeTracker(SqliteSql(tmp_path / "evaluation.db")).ensure()
    rows = tracker.sql.query("SELECT model_id, p50, p10 FROM forecast_outcomes")
    assert len(rows) == 1
    assert rows[0]["model_id"].startswith("loaded:")
    assert rows[0]["p10"] is None
    assert isinstance(rows[0]["p50"], float)
