"""Run a rolling-origin backtest for a shipped starter or the loaded model."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

from src.evaluation.backtest import rolling_origin_backtest
from src.evaluation.tracking import default_tracker
from src.inference.predictor import Predictor
from src.models.registry import build_model
from src.utils.paths import REPO_ROOT
from src.workspace.starters import Starter, starter_by_id
from src.workspace.store import sql_statements, utc_now

_BACKTEST_SQL = REPO_ROOT / "deploy" / "cloudflare" / "migrations" / "0005_backtest.sql"


def read_series_csv(path: Path) -> list[float]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "value" not in rows[0]:
        raise ValueError("The series CSV needs a value column.")
    return [float(row["value"]) for row in rows]


def stored_backtest(model_id: str) -> dict[str, Any] | None:
    _ensure_table()
    rows = default_tracker().sql.query(
        "SELECT report_json FROM backtest_reports WHERE model_id = ?",
        (model_id,),
    )
    if not rows:
        return None
    payload = json.loads(str(rows[0]["report_json"]))
    return payload if isinstance(payload, dict) else None


def save_backtest(model_id: str, report: dict[str, Any]) -> None:
    _ensure_table()
    default_tracker().sql.query(
        """
        INSERT INTO backtest_reports (model_id, report_json, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(model_id) DO UPDATE SET
            report_json = excluded.report_json,
            updated_at = excluded.updated_at
        """,
        (model_id, json.dumps(report), utc_now()),
    )


def backtest_starter(
    starter: Starter,
    *,
    horizon: int = 1,
    season_length: int = 7,
    refresh: bool = False,
) -> dict[str, Any]:
    if not refresh:
        cached = stored_backtest(starter.id)
        if cached is not None:
            return cached
    cfg = starter.config()
    raw_data = cfg.get("data")
    data_cfg = raw_data if isinstance(raw_data, dict) else {}
    seq_len = int(data_cfg.get("seq_len", 16) or 16)
    values = read_series_csv(starter.dataset_path)
    built = build_model(cfg, ensemble=False)
    predictor = Predictor.from_checkpoint(built, starter.model_path, device="cpu")
    report = rolling_origin_backtest(
        values,
        _predictor_fn(predictor, seq_len),
        horizon=horizon,
        season_length=season_length,
        min_train=max(16, seq_len),
    )
    report["model_id"] = starter.id
    save_backtest(starter.id, report)
    return report


def backtest_values(
    model_id: str,
    values: Sequence[float],
    predict: Any,
    *,
    horizon: int = 1,
    season_length: int = 7,
    min_train: int = 16,
    max_origins: int = 40,
) -> dict[str, Any]:
    report = rolling_origin_backtest(
        values,
        predict,
        horizon=horizon,
        season_length=season_length,
        min_train=min_train,
        max_origins=max_origins,
    )
    report["model_id"] = model_id
    save_backtest(model_id, report)
    return report


def starter_or_none(model_id: str) -> Starter | None:
    return starter_by_id(model_id)


def _predictor_fn(predictor: Predictor, seq_len: int):
    def predict(
        history: Sequence[float], horizon: int
    ) -> tuple[list[float], list[float] | None, list[float] | None]:
        window = list(history[-seq_len:])
        if len(window) < seq_len:
            raise ValueError("History is shorter than the model window.")
        tensor = torch.tensor([[value] for value in window], dtype=torch.float32)
        batch = predictor.forecast(tensor, horizon=horizon)
        midpoint = [float(value) for value in batch.point[0, :, 0].tolist()]
        if batch.lower is None or batch.upper is None:
            return midpoint, None, None
        low = [float(value) for value in batch.lower[0, :, 0].tolist()]
        high = [float(value) for value in batch.upper[0, :, 0].tolist()]
        return midpoint, low, high

    return predict


def _ensure_table() -> None:
    default_tracker().sql.script(
        sql_statements(_BACKTEST_SQL.read_text(encoding="utf-8"))
    )
