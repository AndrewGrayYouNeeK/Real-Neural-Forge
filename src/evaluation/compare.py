"""Compare forecasters on one shared temporal split."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.scaling import StandardScaler
from src.data.timeseries import load_windows_from_config
from src.evaluation.baselines import (
    load_baseline_series,
    seasonal_naive_calendar,
    temporal_validation_target_rows,
)
from src.evaluation.metrics import mae, mse, rmse
from src.evaluation.references import (
    LagLinearBaseline,
    PersistenceBaseline,
    SeasonalNaiveBaseline,
)
from src.forecasting.intervals import calibrate_intervals, pinball_loss
from src.forecasting.output import horizon_targets, training_targets
from src.forecasting.spec import ForecastSpec, forecast_spec_from_config
from src.inference.predictor import Predictor
from src.models.lstm import LSTMForecaster
from src.models.registry import build_model
from src.storage.experiments import ExperimentStore, db_path_from_config
from src.utils.paths import resolve_repo_path

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ComparisonRow:
    """One model on the shared validation windows."""

    name: str
    metrics: dict[str, float]
    interval_method: str | None = None


@dataclass
class ComparisonReport:
    """Printed table plus the experiment-store id that holds the same numbers."""

    rows: list[ComparisonRow]
    table: str
    experiment_id: int
    db_path: str
    horizon: int
    strategy: str
    n_train: int
    n_val: int
    nominal_coverage: float
    notes: list[str] = field(default_factory=list)


def compare_models(
    cfg: dict[str, Any],
    *,
    epochs: int | None = None,
    season_length: int | None = None,
    lstm_hidden_size: int | None = None,
    store: ExperimentStore | None = None,
) -> ComparisonReport:
    """Fit every baseline and the configured model on the same temporal holdout.

    The split is always temporal, including when training itself uses a random
    split. That keeps the benchmark a forecast of later windows. Neural fits
    do not write ``checkpoints/best_model.pt``.
    """
    spec = forecast_spec_from_config(cfg)
    if season_length is not None:
        spec = ForecastSpec(
            horizon=spec.horizon,
            strategy=spec.strategy,
            quantiles=spec.quantiles,
            season_length=season_length,
            target_feature_index=spec.target_feature_index,
            nominal_coverage=spec.nominal_coverage,
            n_lags=spec.n_lags,
        )
    data_cfg = cfg.get("data", {})
    if not isinstance(data_cfg, dict):
        raise ValueError("data must be a mapping.")
    inputs, targets = load_windows_from_config(data_cfg, horizon=spec.horizon)
    configured_split = data_cfg.get("val_split", 0.0)
    val_split = float(configured_split) if configured_split else 0.2
    if not 0.0 < val_split < 1.0:
        raise ValueError(f"comparison val_split must be in (0, 1), got {val_split}.")
    n_val = int(len(inputs) * val_split)
    n_train = len(inputs) - n_val
    if n_train < 1 or n_val < 1:
        raise ValueError("comparison split leaves an empty train or validation set.")
    train_idx = list(range(n_train))
    val_idx = list(range(n_train, len(inputs)))

    notes = [
        "mse/rmse/mae use the full temporal validation set, in original units.",
        "coverage/width use the later validation half; the earlier half fits the conformal offset.",
        f"nominal coverage is {spec.nominal_coverage:.2f}.",
    ]
    if not configured_split:
        notes.append(
            "data.val_split is 0, so the comparison held out the last 20% of windows."
        )
    output_dim = int(horizon_targets(targets, spec.horizon).size(-1))
    x_train, y_train = inputs[train_idx], targets[train_idx]
    x_val, y_val = inputs[val_idx], targets[val_idx]
    observed = horizon_targets(y_val, spec.horizon)

    rows: list[ComparisonRow] = [
        _score(
            "persistence",
            PersistenceBaseline(spec.horizon, spec.target_feature_index, output_dim)
            .fit(x_train, y_train)
            .predict(x_val),
            observed,
        )
    ]
    window = int(inputs.size(1))
    if spec.season_length is not None and spec.season_length <= window:
        seasonal = SeasonalNaiveBaseline(
            spec.season_length, spec.horizon, spec.target_feature_index, output_dim
        )
        rows.append(_score("seasonal_naive", seasonal.predict(x_val), observed))
    else:
        calendar = _calendar_seasonal(data_cfg, spec, val_split, n_val, output_dim)
        if calendar is not None:
            rows.append(_score("seasonal_naive", calendar, observed))
            notes.append(
                "seasonal_naive is the same calendar day one year earlier "
                "(29 February uses 28 February)."
            )
        elif spec.season_length is not None:
            notes.append(
                f"seasonal_naive was skipped because season_length {spec.season_length} "
                f"is longer than the input window ({window}) and no daily date column "
                "is available for a calendar baseline."
            )
        else:
            notes.append(
                "seasonal_naive was skipped. Set forecast.season_length or pass "
                "--season-length, or set data.date_column on a daily CSV with horizon 1."
            )
    linear = LagLinearBaseline(
        spec.horizon, spec.target_feature_index, output_dim, n_lags=spec.n_lags
    )
    rows.append(
        _score("linear_lags", linear.fit(x_train, y_train).predict(x_val), observed)
    )

    training_cfg = cfg.get("training", {})
    if not isinstance(training_cfg, dict):
        raise ValueError("training must be a mapping.")
    fit_epochs = int(
        epochs if epochs is not None else _evaluation_epochs(cfg, training_cfg)
    )
    if fit_epochs < 1:
        raise ValueError(f"comparison epochs must be >= 1, got {fit_epochs}.")
    device = _device(str(training_cfg.get("device", "cpu")))
    normalize = bool(data_cfg.get("normalize", data_cfg.get("source") == "csv"))
    seed = training_cfg.get("seed")
    batch_size = int(training_cfg.get("batch_size", 32))
    learning_rate = float(training_cfg.get("learning_rate", 1e-3))
    model_name = str(cfg.get("model", {}).get("name", "time_series_transformer"))

    if model_name != "lstm":
        hidden = lstm_hidden_size
        if hidden is None:
            hidden = int(_evaluation_cfg(cfg).get("lstm_hidden_size", 32))
        layers = int(_evaluation_cfg(cfg).get("lstm_layers", 1))
        lstm = LSTMForecaster(
            input_dim=int(inputs.size(-1)),
            output_dim=output_dim,
            hidden_size=hidden,
            num_layers=layers,
            horizon=spec.model_horizon,
        )
        rows.append(
            _score_neural(
                "lstm",
                lstm,
                x_train,
                y_train,
                x_val,
                observed,
                spec=ForecastSpec(
                    horizon=spec.horizon,
                    strategy=spec.strategy,
                    quantiles=(),
                    target_feature_index=spec.target_feature_index,
                    nominal_coverage=spec.nominal_coverage,
                    n_lags=spec.n_lags,
                    season_length=spec.season_length,
                ),
                epochs=fit_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                seed=int(seed) if seed is not None else None,
                normalize=normalize,
                device=device,
            )
        )

    configured = build_model(cfg)
    rows.append(
        _score_neural(
            model_name,
            configured,
            x_train,
            y_train,
            x_val,
            observed,
            spec=spec,
            epochs=fit_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            seed=int(seed) if seed is not None else None,
            normalize=normalize,
            device=device,
        )
    )

    table = _render(rows, spec, n_train, n_val, notes)
    experiment_store = store or ExperimentStore(db_path_from_config(cfg))
    stored_cfg = copy.deepcopy(cfg)
    evaluation_cfg = stored_cfg.get("evaluation")
    if not isinstance(evaluation_cfg, dict):
        evaluation_cfg = {}
        stored_cfg["evaluation"] = evaluation_cfg
    evaluation_cfg["comparison"] = {
        "split_mode": "temporal",
        "horizon": spec.horizon,
        "strategy": spec.strategy,
        "n_train": n_train,
        "n_val": n_val,
        "nominal_coverage": spec.nominal_coverage,
        "models": [row.name for row in rows],
        "notes": notes,
    }
    base_name = str(training_cfg.get("experiment_name", "default"))
    experiment_id = experiment_store.create_experiment(
        name=f"{base_name}:comparison",
        config=stored_cfg,
        model_name="comparison",
    )
    try:
        flat: dict[str, float] = {
            "horizon": float(spec.horizon),
            "n_train": float(n_train),
            "n_val": float(n_val),
            "nominal_coverage": float(spec.nominal_coverage),
        }
        for row in rows:
            for key, value in row.metrics.items():
                metric_name = f"{row.name}.{key}"
                flat[metric_name] = value
                experiment_store.log_metric(experiment_id, metric_name, value)
        experiment_store.complete_experiment(
            experiment_id,
            status="completed",
            metrics=flat,
        )
    except Exception:
        experiment_store.complete_experiment(experiment_id, status="failed")
        raise

    return ComparisonReport(
        rows=rows,
        table=table,
        experiment_id=experiment_id,
        db_path=str(experiment_store.db_path),
        horizon=spec.horizon,
        strategy=spec.strategy,
        n_train=n_train,
        n_val=n_val,
        nominal_coverage=spec.nominal_coverage,
        notes=notes,
    )


def _calendar_seasonal(
    data_cfg: dict[str, Any],
    spec: ForecastSpec,
    val_split: float,
    n_val: int,
    output_dim: int,
) -> torch.Tensor | None:
    """Same-day-last-year forecast for a one-step dated CSV.

    In-window seasonal naive cannot see a 365-day period inside a short window.
    This uses the series dates instead, and only when the validation labels are
    exactly the one-step temporal holdout.
    """
    if spec.horizon != 1 or output_dim != 1:
        return None
    if data_cfg.get("source") != "csv" or not data_cfg.get("path"):
        return None
    date_column = data_cfg.get("date_column")
    if not date_column:
        return None
    seq_len = int(data_cfg.get("seq_len", 64))
    values, dates = load_baseline_series(
        resolve_repo_path(data_cfg["path"]),
        target_column=str(data_cfg.get("target_column", "value")),
        date_column=str(date_column),
    )
    if dates is None:
        return None
    target_rows = temporal_validation_target_rows(len(values), seq_len, val_split)
    if len(target_rows) != n_val:
        return None
    predicted = seasonal_naive_calendar(values, dates, target_rows)
    return torch.tensor(predicted, dtype=torch.float32).reshape(n_val, 1, 1)


def _score(
    name: str, prediction: torch.Tensor, observed: torch.Tensor
) -> ComparisonRow:
    return ComparisonRow(name=name, metrics=_point_metrics(prediction, observed))


def _score_neural(
    name: str,
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    observed: torch.Tensor,
    *,
    spec: ForecastSpec,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int | None,
    normalize: bool,
    device: torch.device,
) -> ComparisonRow:
    fit_targets = training_targets(y_train, spec)
    trained, scaler = _fit_model(
        model,
        x_train,
        fit_targets,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        seed=seed,
        quantiles=spec.quantiles,
        normalize=normalize,
        device=device,
    )
    predictor = Predictor(trained, device=device, scaler=scaler, spec=spec)
    batch = predictor.forecast(x_val, horizon=spec.horizon)
    point = batch.point.detach().cpu()
    metrics = _point_metrics(point, observed)
    method = "cqr" if spec.quantiles and batch.lower is not None else "residual"
    calibration = calibrate_intervals(
        point=point,
        targets=observed,
        nominal=spec.nominal_coverage,
        method=method,
        lower=None if batch.lower is None else batch.lower.detach().cpu(),
        upper=None if batch.upper is None else batch.upper.detach().cpu(),
    )
    if calibration.validation_coverage is not None:
        metrics["coverage"] = calibration.validation_coverage
    if calibration.raw_coverage is not None:
        metrics["raw_coverage"] = calibration.raw_coverage
    if calibration.interval_width is not None:
        metrics["width"] = calibration.interval_width
        metrics["interval_width"] = calibration.interval_width
    return ComparisonRow(
        name=name,
        metrics=metrics,
        interval_method=calibration.method
        if calibration.validation_coverage is not None
        else None,
    )


def _fit_model(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    *,
    epochs: int,
    learning_rate: float,
    batch_size: int,
    seed: int | None,
    quantiles: tuple[float, ...],
    normalize: bool,
    device: torch.device,
) -> tuple[nn.Module, StandardScaler | None]:
    if seed is not None:
        torch.manual_seed(seed)
    scaler = StandardScaler.fit(inputs, targets) if normalize else None
    x_fit = scaler.transform_inputs(inputs) if scaler is not None else inputs
    y_fit = scaler.transform_targets(targets) if scaler is not None else targets
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(x_fit, y_fit),
        batch_size=max(1, min(batch_size, len(inputs))),
        shuffle=True,
    )
    for _ in range(epochs):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            predicted = model(batch_x)
            loss = (
                pinball_loss(predicted, batch_y, quantiles)
                if quantiles
                else mse_loss(predicted, batch_y)
            )
            loss.backward()
            optimizer.step()
    model.eval()
    return model, scaler


def _point_metrics(
    prediction: torch.Tensor, observed: torch.Tensor
) -> dict[str, float]:
    prediction = prediction.detach().cpu().float()
    observed = observed.detach().cpu().float()
    return {
        "mse": mse(prediction, observed),
        "rmse": rmse(prediction, observed),
        "mae": mae(prediction, observed),
    }


def _render(
    rows: list[ComparisonRow],
    spec: ForecastSpec,
    n_train: int,
    n_val: int,
    notes: list[str],
) -> str:
    header = ("model", "mse", "rmse", "mae", "coverage", "width", "intervals")
    body = [
        [
            row.name,
            _fmt(row.metrics.get("mse")),
            _fmt(row.metrics.get("rmse")),
            _fmt(row.metrics.get("mae")),
            _fmt(row.metrics.get("coverage")),
            _fmt(row.metrics.get("width")),
            row.interval_method or "n/a",
        ]
        for row in rows
    ]
    widths = [
        max(len(header[index]), *(len(cells[index]) for cells in body))
        for index in range(len(header))
    ]

    def line(cells: list[str] | tuple[str, ...]) -> str:
        padded = []
        for index, cell in enumerate(cells):
            padded.append(
                cell.ljust(widths[index]) if index == 0 else cell.rjust(widths[index])
            )
        return "  ".join(padded)

    lines = [
        (
            f"horizon={spec.horizon}  strategy={spec.strategy}  "
            f"n_train={n_train}  n_val={n_val}  nominal_coverage={spec.nominal_coverage:.2f}"
        ),
        line(header),
        line(tuple("-" * width for width in widths)),
        *(line(cells) for cells in body),
        "",
        *notes,
    ]
    return "\n".join(lines)


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f}"


def _evaluation_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    evaluation = cfg.get("evaluation", {})
    return evaluation if isinstance(evaluation, dict) else {}


def _evaluation_epochs(cfg: dict[str, Any], training_cfg: dict[str, Any]) -> int:
    configured = _evaluation_cfg(cfg).get("epochs")
    if configured is not None:
        return int(configured)
    return int(training_cfg.get("epochs", 1))


def _device(device_str: str) -> torch.device:
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available – falling back to CPU.")
        device_str = "cpu"
    return torch.device(device_str)
