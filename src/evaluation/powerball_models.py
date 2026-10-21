"""Train plain and weather-augmented Powerball ticket models.

Each input window is the previous draws only. Weather columns, when present,
are the observations aligned to those past draw dates. The target row's
weather is not in the window.

The published ticket is one step ahead of the last training draw. A temporal
holdout scores that same architecture on draws it did not train on. Early
stopping uses that holdout's scaled MSE, so the squared-error comparison is
slightly optimistic and uses the same protocol for both models.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.draw_weather import CENTRAL_PARK_COLUMN, TALLAHASSEE_COLUMNS
from src.data.scaling import StandardScaler
from src.evaluation.metrics import mae
from src.evaluation.powerball_tickets import (
    CHANCE_POWERBALL_HIT,
    CHANCE_WHITE_MATCHES,
    repair_number_vector,
    score_ticket,
    validate_ticket,
)
from src.models.registry import build_model

TARGET_COLUMNS = [
    "white_1",
    "white_2",
    "white_3",
    "white_4",
    "white_5",
    "powerball",
]
PLAIN_FEATURES = list(TARGET_COLUMNS)
DEFAULT_HYPERPARAMETERS: dict[str, Any] = {
    "seq_len": 16,
    "val_split": 0.2,
    "ticket_val_split": 0.1,
    "epochs": 12,
    "patience": 4,
    "batch_size": 64,
    "learning_rate": 0.001,
    "d_model": 32,
    "nhead": 4,
    "num_layers": 2,
    "dim_feedforward": 64,
    "dropout": 0.1,
    "seed": 0,
}


def weather_feature_columns(frame_columns: Sequence[str]) -> list[str]:
    """Plain numbers, Tallahassee weather, and Central Park when that column exists."""
    missing = [column for column in TALLAHASSEE_COLUMNS if column not in frame_columns]
    if missing:
        raise ValueError(f"weather features missing columns: {missing}")
    columns = [*PLAIN_FEATURES, *TALLAHASSEE_COLUMNS]
    if CENTRAL_PARK_COLUMN in frame_columns:
        columns.append(CENTRAL_PARK_COLUMN)
    return columns


def make_windows(
    frame: Any,
    feature_columns: Sequence[str],
    seq_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding windows whose target row is excluded from the input.

    ``frame`` is a DataFrame with the feature and target columns in draw order.
    """
    if seq_len < 1:
        raise ValueError("seq_len must be positive")
    features = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
    targets = frame.loc[:, TARGET_COLUMNS].to_numpy(dtype=np.float64)
    if len(frame) <= seq_len:
        raise ValueError("need more draws than seq_len")
    windows = [features[end - seq_len : end] for end in range(seq_len, len(frame))]
    labels = [targets[end] for end in range(seq_len, len(frame))]
    return np.stack(windows), np.stack(labels)


def _config(input_dim: int, output_dim: int, hyper: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": {
            "name": "time_series_transformer",
            "input_dim": input_dim,
            "output_dim": output_dim,
            "d_model": int(hyper["d_model"]),
            "nhead": int(hyper["nhead"]),
            "num_encoder_layers": int(hyper["num_layers"]),
            "dim_feedforward": int(hyper["dim_feedforward"]),
            "dropout": float(hyper["dropout"]),
            "max_seq_len": max(int(hyper["seq_len"]), 64),
        }
    }


def fit_transformer(
    inputs: np.ndarray,
    targets: np.ndarray,
    hyper: dict[str, Any],
    *,
    val_split: float,
) -> tuple[torch.nn.Module, StandardScaler, dict[str, Any]]:
    """Fit on the earliest windows and optionally early-stop on the latest ones."""
    if inputs.ndim != 3 or targets.ndim != 2:
        raise ValueError("inputs are (N, T, F) and targets are (N, O)")
    if len(inputs) != len(targets):
        raise ValueError("inputs and targets must have the same number of windows")
    seed = int(hyper["seed"])
    torch.manual_seed(seed)
    values_x = torch.tensor(inputs, dtype=torch.float32)
    values_y = torch.tensor(targets, dtype=torch.float32)
    count = int(values_x.shape[0])
    if not 0.0 <= val_split < 1.0:
        raise ValueError("val_split must be in [0, 1)")
    n_val = int(count * val_split)
    if val_split > 0 and (n_val == 0 or n_val == count):
        raise ValueError("val_split leaves an empty train or validation set")
    n_fit = count - n_val
    scaler = StandardScaler.fit(values_x[:n_fit], values_y[:n_fit])
    scaled_x = scaler.transform_inputs(values_x)
    scaled_y = scaler.transform_targets(values_y)
    model = build_model(
        _config(int(values_x.shape[-1]), int(values_y.shape[-1]), hyper)
    )
    model = model.to("cpu")
    optimizer = torch.optim.Adam(model.parameters(), lr=float(hyper["learning_rate"]))
    loss_fn = nn.MSELoss()
    batch_size = max(1, int(hyper["batch_size"]))
    loader = DataLoader(
        TensorDataset(scaled_x[:n_fit], scaled_y[:n_fit]),
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    best_loss = float("inf")
    stale = 0
    epochs_ran = 0
    patience = int(hyper["patience"])
    for epoch in range(1, int(hyper["epochs"]) + 1):
        model.train()
        running = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * int(batch_x.shape[0])
        epochs_ran = epoch
        if n_val:
            model.eval()
            with torch.no_grad():
                select = float(
                    loss_fn(model(scaled_x[n_fit:]), scaled_y[n_fit:]).item()
                )
        else:
            select = running / n_fit
        if select < best_loss:
            best_loss = select
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if patience and stale >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return (
        model,
        scaler,
        {
            "epochs_ran": epochs_ran,
            "best_scaled_loss": best_loss,
            "n_windows": count,
            "n_fit": n_fit,
            "n_val": n_val,
            "seed": seed,
        },
    )


def predict_original(
    model: torch.nn.Module, scaler: StandardScaler, inputs: np.ndarray
) -> np.ndarray:
    """Predict in original units. ``inputs`` is (N, T, F) before scaling."""
    model.eval()
    values = torch.tensor(np.asarray(inputs, dtype=np.float64), dtype=torch.float32)
    with torch.no_grad():
        predicted = model(scaler.transform_inputs(values))
        restored = scaler.inverse_targets(predicted)
    return restored.cpu().numpy()


def _powerball_rate_z(hits: int, count: int) -> float:
    if count < 1:
        raise ValueError("need at least one holdout draw")
    rate = hits / count
    standard_error = float(
        np.sqrt(CHANCE_POWERBALL_HIT * (1.0 - CHANCE_POWERBALL_HIT) / count)
    )
    return (rate - CHANCE_POWERBALL_HIT) / standard_error


def score_holdout_predictions(
    predicted: np.ndarray, actual: np.ndarray
) -> dict[str, Any]:
    """Ticket-level and Powerball metrics for a holdout, in original units."""
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual holdout rows differ")
    tickets = [repair_number_vector(row) for row in predicted]
    observed = [
        validate_ticket(
            {
                "whites": [int(round(value)) for value in row[:5]],
                "powerball": int(round(row[5])),
            }
        )
        for row in actual
    ]
    scores = [
        score_ticket(ticket, truth)
        for ticket, truth in zip(tickets, observed, strict=True)
    ]
    powerball_predictions = torch.tensor(
        predicted[:, 5], dtype=torch.float32
    ).unsqueeze(-1)
    powerball_actual = torch.tensor(actual[:, 5], dtype=torch.float32).unsqueeze(-1)
    hits = sum(int(score["powerball_hit"]) for score in scores)
    white_matches = sum(int(score["white_matches"]) for score in scores)
    count = len(scores)
    full_hits = sum(
        1 for score in scores if score["white_matches"] == 5 and score["powerball_hit"]
    )
    return {
        "n": count,
        "powerball_mae": mae(powerball_predictions, powerball_actual),
        "powerball_hits": hits,
        "powerball_hit_rate": hits / count,
        "powerball_hit_z": _powerball_rate_z(hits, count),
        "chance_powerball_hit_rate": CHANCE_POWERBALL_HIT,
        "mean_white_matches": white_matches / count,
        "chance_white_matches": CHANCE_WHITE_MATCHES,
        "full_ticket_hits": full_hits,
        "prize_tickets": sum(1 for score in scores if score["tier"] != "none"),
    }


def evaluate_feature_set(
    frame: Any,
    feature_columns: Sequence[str],
    hyper: dict[str, Any],
) -> dict[str, Any]:
    """Train on the earliest windows and score the temporal holdout."""
    windows, targets = make_windows(frame, feature_columns, int(hyper["seq_len"]))
    model, scaler, info = fit_transformer(
        windows, targets, hyper, val_split=float(hyper["val_split"])
    )
    n_val = int(info["n_val"])
    predicted = predict_original(model, scaler, windows[-n_val:])
    metrics = score_holdout_predictions(predicted, targets[-n_val:])
    return {
        "training": info,
        "metrics": metrics,
        "feature_columns": list(feature_columns),
    }


def forecast_next_ticket(
    frame: Any,
    feature_columns: Sequence[str],
    hyper: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One legal ticket for the draw after the last row of ``frame``."""
    windows, targets = make_windows(frame, feature_columns, int(hyper["seq_len"]))
    model, scaler, info = fit_transformer(
        windows, targets, hyper, val_split=float(hyper["ticket_val_split"])
    )
    features = frame.loc[:, list(feature_columns)].to_numpy(dtype=np.float64)
    window = features[-int(hyper["seq_len"]) :]
    predicted = predict_original(model, scaler, window[None, :, :])[0]
    return repair_number_vector(predicted), info


def compare_plain_and_weather(frame: Any, hyper: dict[str, Any]) -> dict[str, Any]:
    """Score both models on the same weather-complete draws."""
    features = weather_feature_columns(list(frame.columns))
    plain = evaluate_feature_set(frame, PLAIN_FEATURES, hyper)
    weather = evaluate_feature_set(frame, features, hyper)
    plain_mae = float(plain["metrics"]["powerball_mae"])
    weather_mae = float(weather["metrics"]["powerball_mae"])
    return {
        "rows": int(len(frame)),
        "feature_columns_weather": features,
        "plain": plain,
        "weather": weather,
        "powerball_mae_weather_minus_plain": weather_mae - plain_mae,
    }


def hyperparameters_from_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Read the ticket-model hyperparameters from a config mapping."""
    model = cfg["model"]
    data = cfg["data"]
    training = cfg["training"]
    return {
        "seq_len": int(data["seq_len"]),
        "val_split": float(data["val_split"]),
        "ticket_val_split": float(data.get("ticket_val_split", 0.1)),
        "epochs": int(training["epochs"]),
        "patience": int(training.get("patience", 0)),
        "batch_size": int(training["batch_size"]),
        "learning_rate": float(training["learning_rate"]),
        "d_model": int(model["d_model"]),
        "nhead": int(model["nhead"]),
        "num_layers": int(model["num_encoder_layers"]),
        "dim_feedforward": int(model["dim_feedforward"]),
        "dropout": float(model["dropout"]),
        "seed": int(training.get("seed", 0)),
    }
