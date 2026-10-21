"""Per-horse form sequences and a race-level comparison with the market.

Each runner is a short sequence of that horse's own past runs, plus one
current-race token that holds pre-race information (days since the last run,
weight, draw, jockey and trainer strike rates). The current race's odds and
the current result are not in that token. Odds stay available for the
favourite, the market probabilities, and the flat-stake settlement.

Sequences never include a run that had not happened yet. Jockey and trainer
rates are the strike rate strictly before the race being described.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.scaling import StandardScaler
from src.domain_forecasts.baselines import (
    flat_stake_profit,
    market_implied_probabilities,
    multiclass_brier,
    multiclass_log_loss,
    rank_best_first,
    scores_to_probabilities,
    uniform_race_expectations,
    window_split_bounds,
)
from src.domain_forecasts.reporting import write_results
from src.models.registry import build_model
from src.utils.checkpoint import save_checkpoint
from src.utils.config import load_config
from src.utils.paths import resolve_repo_path

FEATURE_NAMES: tuple[str, ...] = (
    "finish_pct",
    "won",
    "log_days",
    "weight",
    "draw_pct",
    "log_odds",
    "jockey_rate",
    "trainer_rate",
    "is_today",
)
FINISH_PCT = 0
WON = 1
LOG_DAYS = 2
WEIGHT = 3
DRAW_PCT = 4
LOG_ODDS = 5
JOCKEY_RATE = 6
TRAINER_RATE = 7
IS_TODAY = 8
N_FEATURES = len(FEATURE_NAMES)

NOT_BETTING_ADVICE = (
    "Not betting advice. This is a historical comparison of a forecasting "
    "model with the closing market. A simulated return is not a reason to "
    "bet, and public form usually does not beat the odds once the track "
    "take is included."
)

TEMPERATURES = (0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0)
logger = logging.getLogger(__name__)


@dataclass
class FormDataset:
    """One row per runner. Rows from the same race sit together."""

    inputs: np.ndarray
    padding_mask: np.ndarray
    targets: np.ndarray
    race_keys: np.ndarray
    race_dates: np.ndarray
    odds: np.ndarray
    finish: np.ndarray
    draw: np.ndarray
    horse_codes: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES


def tidy_hk_performances(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep Hong Kong runners with a decimal dividend above 1 and a placing.

    ``winning_odds`` in the source table is a stake-inclusive decimal dividend
    (favourites land near 1.1–3, the median runner near 14). Dividends already
    have the pool deduction taken out.
    """
    required = {
        "race_country",
        "race_date",
        "race_location",
        "race_no",
        "race_id",
        "horse_id",
        "final_placing",
        "winning_odds",
        "actual_weight",
        "draw",
        "jockey_id",
        "trainer_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"performances table is missing columns: {missing}")
    work = frame.loc[frame["race_country"] == "HK"].copy()
    work["race_date"] = pd.to_datetime(work["race_date"], errors="coerce")
    for column in ("final_placing", "winning_odds", "actual_weight", "draw"):
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work.dropna(
        subset=[
            "race_date",
            "final_placing",
            "winning_odds",
            "actual_weight",
            "draw",
            "horse_id",
        ]
    )
    work = work[
        (work["winning_odds"] > 1.0)
        & (work["final_placing"] >= 1)
        & (work["actual_weight"] > 0)
        & (work["draw"] > 0)
    ]
    work["jockey_id"] = work["jockey_id"].fillna("").astype(str)
    work["trainer_id"] = work["trainer_id"].fillna("").astype(str)
    work["race_location"] = work["race_location"].fillna("").astype(str)
    work["race_no"] = work["race_no"].fillna("").astype(str)
    work["race_id"] = work["race_id"].fillna("").astype(str)
    work["race_key"] = (
        work["race_date"].dt.strftime("%Y-%m-%d")
        + "|"
        + work["race_location"]
        + "|"
        + work["race_no"]
        + "|"
        + work["race_id"]
    )
    tidy = pd.DataFrame(
        {
            "race_date": work["race_date"].to_numpy(),
            "race_key": work["race_key"].to_numpy(),
            "horse_id": work["horse_id"].astype(str).to_numpy(),
            "finish": work["final_placing"].astype(int).to_numpy(),
            "odds": work["winning_odds"].astype(float).to_numpy(),
            "weight": work["actual_weight"].astype(float).to_numpy(),
            "draw": work["draw"].astype(float).to_numpy(),
            "jockey_id": work["jockey_id"].to_numpy(),
            "trainer_id": work["trainer_id"].to_numpy(),
        }
    )
    return tidy.reset_index(drop=True)


def _rate(wins: int, starts: int) -> float:
    if starts <= 0:
        return 0.0
    return wins / starts


def _bump(counts: dict[str, tuple[int, int]], key: str, won: bool) -> None:
    if not key:
        return
    wins, starts = counts.get(key, (0, 0))
    counts[key] = (wins + (1 if won else 0), starts + 1)


def build_form_sequences(tidy: pd.DataFrame, history_len: int) -> FormDataset:
    """Build leak-free sequences. ``history_len`` is the number of past runs."""
    if history_len < 1:
        raise ValueError("history_len must be at least 1")
    required = {
        "race_date",
        "race_key",
        "horse_id",
        "finish",
        "odds",
        "weight",
        "draw",
        "jockey_id",
        "trainer_id",
    }
    missing = sorted(required - set(tidy.columns))
    if missing:
        raise ValueError(f"tidy runner table is missing columns: {missing}")
    frame = tidy.sort_values(["race_date", "race_key", "horse_id"]).reset_index(
        drop=True
    )
    history: dict[str, list[np.ndarray]] = {}
    last_ran: dict[str, pd.Timestamp] = {}
    jockey_counts: dict[str, tuple[int, int]] = {}
    trainer_counts: dict[str, tuple[int, int]] = {}

    sequences: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    targets: list[float] = []
    race_keys: list[str] = []
    race_dates: list[np.datetime64] = []
    odds: list[float] = []
    finish: list[int] = []
    draw: list[float] = []
    horse_ids: list[str] = []

    for race_key, group in frame.groupby("race_key", sort=False):
        runners = list(group.itertuples(index=False))
        if sum(int(row.finish) == 1 for row in runners) != 1 or len(runners) < 2:
            continue
        field = len(runners)
        prepared: list[tuple[Any, float, float, int, np.ndarray, np.ndarray]] = []
        for row in runners:
            horse = str(row.horse_id)
            past = history.get(horse, [])
            if horse in last_ran:
                days = max(0, int((row.race_date - last_ran[horse]).days))
            else:
                days = 0
            jockey_rate = _rate(*jockey_counts.get(str(row.jockey_id), (0, 0)))
            trainer_rate = _rate(*trainer_counts.get(str(row.trainer_id), (0, 0)))
            kept = past[-history_len:]
            sequence = np.zeros((history_len + 1, N_FEATURES), dtype=np.float32)
            mask = np.zeros(history_len + 1, dtype=bool)
            pad = history_len - len(kept)
            if pad:
                mask[:pad] = True
            if kept:
                sequence[pad:history_len] = np.stack(kept)
            today = sequence[history_len]
            today[LOG_DAYS] = np.log1p(days)
            today[WEIGHT] = float(row.weight)
            today[DRAW_PCT] = float(row.draw) / field
            today[JOCKEY_RATE] = jockey_rate
            today[TRAINER_RATE] = trainer_rate
            today[IS_TODAY] = 1.0
            prepared.append((row, jockey_rate, trainer_rate, days, sequence, mask))

        for row, jockey_rate, trainer_rate, days, sequence, mask in prepared:
            horse = str(row.horse_id)
            won = int(row.finish) == 1
            token = np.zeros(N_FEATURES, dtype=np.float32)
            token[FINISH_PCT] = float(row.finish) / field
            token[WON] = 1.0 if won else 0.0
            token[LOG_DAYS] = np.log1p(days)
            token[WEIGHT] = float(row.weight)
            token[DRAW_PCT] = float(row.draw) / field
            token[LOG_ODDS] = float(np.log(float(row.odds)))
            token[JOCKEY_RATE] = jockey_rate
            token[TRAINER_RATE] = trainer_rate
            history.setdefault(horse, []).append(token)
            last_ran[horse] = row.race_date
            _bump(jockey_counts, str(row.jockey_id), won)
            _bump(trainer_counts, str(row.trainer_id), won)

            sequences.append(sequence)
            masks.append(mask)
            targets.append(float(row.finish) / field)
            race_keys.append(str(race_key))
            race_dates.append(np.datetime64(pd.Timestamp(row.race_date).date(), "D"))
            odds.append(float(row.odds))
            finish.append(int(row.finish))
            draw.append(float(row.draw))
            horse_ids.append(horse)

    if not sequences:
        raise ValueError("no races left after filtering")
    codes, _ = pd.factorize(pd.Index(horse_ids), sort=True)
    return FormDataset(
        inputs=np.stack(sequences),
        padding_mask=np.stack(masks),
        targets=np.asarray(targets, dtype=np.float32).reshape(-1, 1),
        race_keys=np.asarray(race_keys),
        race_dates=np.asarray(race_dates, dtype="datetime64[D]"),
        odds=np.asarray(odds, dtype=np.float64),
        finish=np.asarray(finish, dtype=np.int64),
        draw=np.asarray(draw, dtype=np.float64),
        horse_codes=np.asarray(codes, dtype=np.int64),
    )


def split_masks(
    dataset: FormDataset,
    test_fraction: float,
    val_fraction: float,
    min_date: str | None = None,
) -> dict[str, np.ndarray]:
    """Chronological race split. Every runner in a race shares a split."""
    eligible = np.ones(len(dataset.race_keys), dtype=bool)
    if min_date is not None:
        eligible &= dataset.race_dates >= np.datetime64(min_date, "D")
    first_date: dict[str, np.datetime64] = {}
    for key, day, keep in zip(
        dataset.race_keys, dataset.race_dates, eligible, strict=True
    ):
        if not keep or key in first_date:
            continue
        first_date[str(key)] = day
    races = sorted(first_date.items(), key=lambda item: (item[1], item[0]))
    train_end, val_end = window_split_bounds(len(races), test_fraction, val_fraction)
    label = {
        key: "train" if index < train_end else "val" if index < val_end else "test"
        for index, (key, _) in enumerate(races)
    }
    keys = dataset.race_keys
    return {
        name: np.array([label.get(str(key)) == name for key in keys], dtype=bool)
        for name in ("train", "val", "test")
    }


def score_race(race: dict[str, Any], temperature: float) -> dict[str, Any]:
    """Score one race. Lower model scores are treated as more likely winners."""
    scores = np.asarray(race["scores"], dtype=float)
    odds = np.asarray(race["odds"], dtype=float)
    finish = np.asarray(race["finish"], dtype=int)
    draw = np.asarray(race["draw"], dtype=float)
    horse_codes = np.asarray(race["horse_codes"], dtype=int)
    winners = np.flatnonzero(finish == 1)
    if winners.size != 1:
        raise ValueError("a scored race must have exactly one winner")
    winner = int(winners[0])
    model_pick = int(rank_best_first(scores, [draw, horse_codes])[0])
    favourite = int(rank_best_first(odds, [draw, horse_codes])[0])
    model_probs = scores_to_probabilities(scores, temperature)
    market_probs, overround = market_implied_probabilities(odds)
    field = int(finish.size)
    return {
        "date": race.get("date"),
        "model_won": bool(finish[model_pick] == 1),
        "favourite_won": bool(finish[favourite] == 1),
        "model_top3": bool(int(finish[model_pick]) <= 3),
        "favourite_top3": bool(int(finish[favourite]) <= 3),
        "model_log_loss": multiclass_log_loss(model_probs, winner),
        "market_log_loss": multiclass_log_loss(market_probs, winner),
        "model_brier": multiclass_brier(model_probs, winner),
        "market_brier": multiclass_brier(market_probs, winner),
        "model_profit": flat_stake_profit(
            float(odds[model_pick]), bool(finish[model_pick] == 1)
        ),
        "favourite_profit": flat_stake_profit(
            float(odds[favourite]), bool(finish[favourite] == 1)
        ),
        "overround": overround,
        "field": field,
        "uniform": uniform_race_expectations(field),
    }


def select_temperature(
    races: list[dict[str, Any]], grid: tuple[float, ...] = TEMPERATURES
) -> float:
    """Pick the softmax temperature that minimises validation log loss."""
    best_temperature = grid[0]
    best_loss = float("inf")
    for temperature in grid:
        loss = float(
            np.mean([score_race(race, temperature)["model_log_loss"] for race in races])
        )
        if loss < best_loss:
            best_loss = loss
            best_temperature = temperature
    return float(best_temperature)


def aggregate_races(scored: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(scored)
    if n == 0:
        raise ValueError("no races to aggregate")

    def mean(key: str) -> float:
        return float(np.mean([row[key] for row in scored]))

    uniform_keys = ("win_accuracy", "top3_hit_rate", "log_loss", "brier")
    model_profit = float(np.sum([row["model_profit"] for row in scored]))
    favourite_profit = float(np.sum([row["favourite_profit"] for row in scored]))
    model_log_loss = mean("model_log_loss")
    market_log_loss = mean("market_log_loss")
    model_roi = model_profit / n
    favourite_roi = favourite_profit / n
    summary = {
        "n_races": n,
        "mean_field": mean("field"),
        "mean_overround": mean("overround"),
        "model_win_accuracy": mean("model_won"),
        "favourite_win_accuracy": mean("favourite_won"),
        "uniform_win_accuracy": float(
            np.mean([row["uniform"]["win_accuracy"] for row in scored])
        ),
        "model_top3_hit_rate": mean("model_top3"),
        "favourite_top3_hit_rate": mean("favourite_top3"),
        "uniform_top3_hit_rate": float(
            np.mean([row["uniform"]["top3_hit_rate"] for row in scored])
        ),
        "model_log_loss": model_log_loss,
        "market_log_loss": market_log_loss,
        "uniform_log_loss": float(
            np.mean([row["uniform"]["log_loss"] for row in scored])
        ),
        "model_brier": mean("model_brier"),
        "market_brier": mean("market_brier"),
        "uniform_brier": float(np.mean([row["uniform"]["brier"] for row in scored])),
        "model_profit": model_profit,
        "favourite_profit": favourite_profit,
        "model_roi": model_roi,
        "favourite_roi": favourite_roi,
        "date_start": str(scored[0]["date"]),
        "date_end": str(scored[-1]["date"]),
        "beats_market": bool(
            model_log_loss < market_log_loss and model_roi > favourite_roi
        ),
    }
    summary["uniform_metric_names"] = list(uniform_keys)
    return summary


def collect_races(
    dataset: FormDataset, mask: np.ndarray, scores: np.ndarray
) -> list[dict[str, Any]]:
    buckets: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for index, score in zip(np.flatnonzero(mask), scores, strict=True):
        key = str(dataset.race_keys[index])
        if key not in buckets:
            buckets[key] = {
                "date": str(dataset.race_dates[index]),
                "scores": [],
                "odds": [],
                "finish": [],
                "draw": [],
                "horse_codes": [],
            }
            order.append(key)
        bucket = buckets[key]
        bucket["scores"].append(float(score))
        bucket["odds"].append(float(dataset.odds[index]))
        bucket["finish"].append(int(dataset.finish[index]))
        bucket["draw"].append(float(dataset.draw[index]))
        bucket["horse_codes"].append(int(dataset.horse_codes[index]))
    return [buckets[key] for key in order]


def _predict(
    model: nn.Module,
    inputs: torch.Tensor,
    mask: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    pieces: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch_x = inputs[start : start + batch_size].to(device)
            batch_m = mask[start : start + batch_size].to(device)
            pieces.append(model(batch_x, src_key_padding_mask=batch_m).cpu())
    return torch.cat(pieces, dim=0)


def train_racing(config_path: str) -> dict[str, Any]:
    """Train on past-form sequences and compare with the market on a later period."""
    started = time.perf_counter()
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    raw = pd.read_csv(resolve_repo_path(data_cfg["path"]))
    tidy = tidy_hk_performances(raw)
    history_len = int(data_cfg.get("history_len", 6))
    dataset = build_form_sequences(tidy, history_len)
    masks = split_masks(
        dataset,
        float(data_cfg.get("test_fraction", 0.2)),
        float(data_cfg.get("val_fraction", 0.1)),
        data_cfg.get("min_date"),
    )
    seed = train_cfg.get("seed")
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    device_name = str(train_cfg.get("device", "cpu"))
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)
    model = build_model(cfg).to(device)
    expected = history_len + 1
    if (
        int(cfg["model"]["input_dim"]) != N_FEATURES
        or int(cfg["model"]["max_seq_len"]) < expected
    ):
        raise ValueError(
            f"model input_dim must be {N_FEATURES} and max_seq_len must be >= {expected}"
        )

    inputs = torch.tensor(dataset.inputs)
    targets = torch.tensor(dataset.targets)
    padding = torch.tensor(dataset.padding_mask)
    train_idx = np.flatnonzero(masks["train"])
    observed = dataset.inputs[train_idx][~dataset.padding_mask[train_idx]]
    scaler = None
    if bool(data_cfg.get("normalize", True)):
        scaler = StandardScaler.fit(
            torch.tensor(observed).unsqueeze(1),
            targets[train_idx],
        )
        inputs = scaler.transform_inputs(inputs)
        targets = scaler.transform_targets(targets)

    generator = torch.Generator().manual_seed(int(seed)) if seed is not None else None
    loader = DataLoader(
        TensorDataset(inputs[train_idx], padding[train_idx], targets[train_idx]),
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=float(train_cfg["learning_rate"])
    )
    criterion = nn.MSELoss()
    patience = int(train_cfg.get("patience", 0) or 0)
    max_grad_norm = float(train_cfg.get("max_grad_norm", 0) or 0)
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }

    def selection_loss(index: np.ndarray) -> float:
        predicted = _predict(
            model,
            inputs[index],
            padding[index],
            int(train_cfg["batch_size"]),
            device,
        )
        actual = targets[index]
        return float(torch.mean((predicted - actual) ** 2).item())

    val_idx = np.flatnonzero(masks["val"])
    test_idx = np.flatnonzero(masks["test"])
    logger.info(
        "runners train/val/test %s/%s/%s",
        int(masks["train"].sum()),
        int(masks["val"].sum()),
        int(masks["test"].sum()),
    )
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        for batch_x, batch_m, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_m = batch_m.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(batch_x, src_key_padding_mask=batch_m), batch_y)
            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        current = selection_loss(val_idx)
        logger.info("epoch %s/%s val_loss=%.6f", epoch, train_cfg["epochs"], current)
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
            if patience and stale >= patience:
                break

    model.load_state_dict(best_state)
    checkpoint = resolve_repo_path(
        train_cfg.get("checkpoint_dir", "checkpoints/horse_racing")
    )
    save_checkpoint(
        model,
        optimizer,
        best_epoch,
        best_loss,
        cfg,
        checkpoint / "best_model.pt",
        scaler=scaler.state_dict() if scaler is not None else None,
    )

    def original_scores(index: np.ndarray) -> np.ndarray:
        predicted = _predict(
            model, inputs[index], padding[index], int(train_cfg["batch_size"]), device
        )
        if scaler is not None:
            predicted = scaler.inverse_targets(predicted)
        return predicted.numpy().reshape(-1)

    val_races = collect_races(dataset, masks["val"], original_scores(val_idx))
    temperature = select_temperature(val_races)
    test_races = collect_races(dataset, masks["test"], original_scores(test_idx))
    scored = [score_race(race, temperature) for race in test_races]
    summary = aggregate_races(scored)
    summary.update(
        {
            "config": str(config_path),
            "seconds": time.perf_counter() - started,
            "temperature": temperature,
            "best_epoch": best_epoch,
            "best_val_loss": best_loss,
            "n_train_runners": int(masks["train"].sum()),
            "n_val_runners": int(masks["val"].sum()),
            "n_test_runners": int(masks["test"].sum()),
            "history_len": history_len,
            "not_betting_advice": NOT_BETTING_ADVICE,
            "odds": (
                "winning_odds are stake-inclusive decimal dividends, so the "
                "flat-stake profit already nets out the pool deduction. "
                "HKJC's published win-pool deduction is 17.5%, which implies "
                "an overround near 0.175/0.825 ≈ 0.212. The test-set mean "
                "overround is reported beside the return and is not applied twice."
            ),
            "test_races": [
                {
                    "date": row["date"],
                    "model_profit": row["model_profit"],
                    "favourite_profit": row["favourite_profit"],
                    "model_won": row["model_won"],
                    "favourite_won": row["favourite_won"],
                }
                for row in scored
            ],
            "param_count": int(
                sum(p.numel() for p in model.parameters() if p.requires_grad)
            ),
        }
    )
    output = cfg.get("output", {}).get("results")
    if output:
        summary["results_path"] = str(write_results(output, summary))
    return summary
