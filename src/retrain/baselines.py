"""Validation baselines a candidate must beat before it becomes champion.

Lower MAE, RMSE, and MSE are better. Persistence repeats the last observation.
The mean baseline repeats the training-split mean. A seasonal naive baseline is
added when ``data.season_length`` is a positive integer, or when ``retrain.baselines``
lists ``seasonal``.
"""

from __future__ import annotations

from typing import Any

import torch

from src.evaluation.metrics import mae, mse, rmse

METRICS = ("mae", "rmse", "mse")


def metric_names() -> tuple[str, ...]:
    return METRICS


def regression_metrics(preds: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    """Return MAE, RMSE, and MSE in the tensors' units."""
    if preds.shape != targets.shape:
        raise ValueError(
            f"predictions shape {tuple(preds.shape)} != targets shape {tuple(targets.shape)}"
        )
    if preds.numel() == 0:
        raise ValueError("Cannot score an empty validation window.")
    return {
        "mse": float(mse(preds, targets)),
        "rmse": float(rmse(preds, targets)),
        "mae": float(mae(preds, targets)),
    }


def baseline_names(cfg: dict[str, Any]) -> list[str]:
    """Return the baselines this config is compared against."""
    retrain = cfg.get("retrain")
    if isinstance(retrain, dict) and retrain.get("baselines") is not None:
        names = retrain["baselines"]
        if (
            not isinstance(names, list)
            or not names
            or not all(isinstance(n, str) for n in names)
        ):
            raise ValueError("retrain.baselines must be a non-empty list of names.")
        unknown = [
            name for name in names if name not in {"persistence", "mean", "seasonal"}
        ]
        if unknown:
            raise ValueError(f"Unknown baselines: {unknown}.")
        return list(names)
    names = ["persistence", "mean"]
    data = cfg.get("data")
    season = data.get("season_length") if isinstance(data, dict) else None
    if isinstance(season, int) and not isinstance(season, bool) and season > 0:
        names.append("seasonal")
    return names


def promotion_metric(cfg: dict[str, Any], override: str | None = None) -> str:
    """Return the metric used to decide promotion. Lower is better."""
    if override:
        name = override
    else:
        retrain = cfg.get("retrain")
        configured = retrain.get("metric") if isinstance(retrain, dict) else None
        name = str(configured) if configured else "mae"
    if name not in METRICS:
        raise ValueError(f"retrain.metric must be one of {METRICS}, got {name!r}.")
    return name


def promotion_val_fraction(cfg: dict[str, Any]) -> float:
    """Return the latest-window fraction used for the promotion decision.

    A configured ``data.val_split`` wins. When that split is zero, retraining
    holds out ``retrain.val_fraction`` (default 0.2) so there is a window to score.
    """
    data = cfg.get("data")
    data = data if isinstance(data, dict) else {}
    fraction = float(data.get("val_split", 0.0) or 0.0)
    if fraction <= 0:
        retrain = cfg.get("retrain")
        configured = (
            retrain.get("val_fraction", 0.2) if isinstance(retrain, dict) else 0.2
        )
        fraction = float(configured or 0.2)
    if not 0.0 < fraction < 1.0:
        raise ValueError(
            f"Promotion validation fraction must be in (0, 1), got {fraction}."
        )
    return fraction


def target_feature_index(cfg: dict[str, Any], n_features: int) -> int:
    """Return the input feature that carries the target, when it is also a feature."""
    data = cfg.get("data")
    if not isinstance(data, dict) or n_features < 1:
        return 0
    features = data.get("feature_columns")
    target = data.get("target_column")
    if isinstance(features, list) and isinstance(target, str) and target in features:
        return int(features.index(target))
    return 0


def score_baselines(
    names: list[str],
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    cfg: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Score each named baseline on the validation rows, in original units."""
    if not val_idx:
        raise ValueError("Baselines need a non-empty validation window.")
    actual = targets[val_idx]
    scores: dict[str, dict[str, float]] = {}
    for name in names:
        preds = _predict_baseline(name, inputs, targets, train_idx, val_idx, cfg)
        scores[name] = regression_metrics(preds, actual)
    return scores


def _predict_baseline(
    name: str,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    train_idx: list[int],
    val_idx: list[int],
    cfg: dict[str, Any],
) -> torch.Tensor:
    if name == "persistence":
        return _persistence(inputs, targets, val_idx, cfg)
    if name == "mean":
        mean = targets[train_idx].mean(dim=0, keepdim=True)
        return mean.expand(len(val_idx), targets.shape[-1]).clone()
    if name == "seasonal":
        return _seasonal(targets, val_idx, train_idx, cfg)
    raise ValueError(f"Unknown baseline {name!r}.")


def _persistence(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    val_idx: list[int],
    cfg: dict[str, Any],
) -> torch.Tensor:
    n_features = int(inputs.shape[-1])
    output_dim = int(targets.shape[-1])
    index = target_feature_index(cfg, n_features)
    rows = inputs[val_idx]
    if output_dim == 1:
        last = rows[:, -1, min(index, n_features - 1)]
        return last.unsqueeze(-1).to(dtype=targets.dtype)
    if n_features >= output_dim:
        return rows[:, -1, :output_dim].to(dtype=targets.dtype).clone()
    last = rows[:, -1, -1].unsqueeze(-1)
    return last.expand(len(val_idx), output_dim).to(dtype=targets.dtype).clone()


def _seasonal(
    targets: torch.Tensor,
    val_idx: list[int],
    train_idx: list[int],
    cfg: dict[str, Any],
) -> torch.Tensor:
    data = cfg.get("data")
    season = data.get("season_length") if isinstance(data, dict) else None
    if not isinstance(season, int) or isinstance(season, bool) or season < 1:
        raise ValueError(
            "The seasonal baseline requires data.season_length to be a positive integer."
        )
    fallback = targets[train_idx].mean(dim=0)
    rows = []
    for index in val_idx:
        source = index - season
        rows.append(targets[source] if source >= 0 else fallback)
    return torch.stack(rows).to(dtype=targets.dtype)


def beats(challenger: float, incumbent: float) -> bool:
    """Return whether ``challenger`` is strictly better (lower)."""
    return challenger < incumbent


def decide_promotion(
    *,
    candidate: dict[str, float],
    champion: dict[str, float] | None,
    baselines: dict[str, dict[str, float]],
    metric: str,
    champion_error: str | None,
) -> tuple[str, list[str]]:
    """Return ``(decision, baselines_not_beaten)``.

    The candidate is promoted only when it strictly beats every baseline and,
    when a champion scored successfully, that champion too. A champion file that
    cannot be scored blocks promotion.
    """
    if metric not in METRICS:
        raise ValueError(f"Unknown promotion metric {metric!r}.")
    if metric not in candidate:
        raise ValueError(f"Candidate metrics are missing {metric!r}.")
    lost_to = [
        name
        for name, scores in baselines.items()
        if metric not in scores or not beats(candidate[metric], scores[metric])
    ]
    if champion_error:
        return "champion_unevaluable", lost_to
    if lost_to:
        if champion is None:
            return "rejected_baseline", lost_to
        return "kept_champion", lost_to
    if champion is None:
        return "promoted", []
    if metric not in champion or not beats(candidate[metric], champion[metric]):
        return "kept_champion", []
    return "promoted", []
