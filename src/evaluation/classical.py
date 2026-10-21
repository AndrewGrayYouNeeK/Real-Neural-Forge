"""Classical forecasts fit only on rows before the scored labels.

Climatology is a circular day-of-year mean. The anomaly AR, ridge, and
gradient boosting models predict the departure from that climatology, which is
the standard regression form of a seasonal model when a 365-day SARIMA state
is too large to fit. Hyperparameters that are not supplied are chosen on the
latest 20% of the fitting labels and then refit on every fitting label. That
slice is earlier than the validation labels the caller passes in.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

_LAGS = (1, 2, 3, 7, 14, 28)
_AR_ORDERS = (1, 3, 7)
_RIDGE_ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
_DEFAULT_BOOSTING: dict[str, Any] = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "max_iter": 200,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
}
_BOOSTING_GRID: tuple[dict[str, Any], ...] = (
    _DEFAULT_BOOSTING,
    {
        "max_depth": 2,
        "learning_rate": 0.05,
        "max_iter": 300,
        "min_samples_leaf": 30,
        "l2_regularization": 1.0,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.08,
        "max_iter": 250,
        "min_samples_leaf": 15,
        "l2_regularization": 0.1,
    },
)


def score_classical_baselines(
    values: np.ndarray,
    target_rows: np.ndarray,
    *,
    dates: np.ndarray | None = None,
    covariates: dict[str, np.ndarray] | None = None,
    seed: int = 0,
    ridge_alpha: float | None = None,
    ar_order: int | None = None,
    boosting_params: dict[str, Any] | None = None,
    half_window: int = 7,
) -> dict[str, np.ndarray]:
    """Predict ``target_rows`` with climatology, anomaly AR, ridge, and boosting.

    ``values`` and each covariate are the full series in row order. Fitting
    uses rows strictly before ``target_rows.min()``. The returned arrays are
    aligned with ``target_rows`` and named ``climatology``, ``anomaly_ar``,
    ``ridge``, and ``gradient_boosting``.
    """
    series = np.asarray(values, dtype=np.float64)
    rows = np.asarray(target_rows, dtype=np.int64)
    if rows.size == 0:
        raise ValueError("Classical baselines need at least one target row.")
    first = int(rows.min())
    if int(rows.max()) >= len(series) or first < 2:
        raise ValueError("Classical baseline target rows fall outside the series.")
    if dates is None:
        raise ValueError("Classical baselines require a date for each row.")
    calendar = np.asarray(dates, dtype="datetime64[D]")
    if len(calendar) != len(series):
        raise ValueError("values and dates must have the same length.")
    doy = _day_of_year(calendar)
    lags = tuple(lag for lag in _LAGS if lag < first)
    if not lags:
        raise ValueError("Classical baselines need history before the first label.")
    origin = max(lags)
    supervised = np.arange(origin, first)
    if supervised.size < 10:
        raise ValueError("Not enough fitting labels for classical baselines.")
    cut = int(supervised.size * 0.8)
    if cut <= 0 or cut >= supervised.size:
        cut = supervised.size - 1
    selection_start = int(supervised[cut])
    extras = covariates or {}

    chosen_order = ar_order if ar_order is not None else 0
    chosen_alpha = ridge_alpha if ridge_alpha is not None else 0.0
    chosen_boosting = boosting_params
    if ar_order is None or ridge_alpha is None or boosting_params is None:
        selection_rows = supervised[cut:]
        probe = _climatology(series, doy, selection_start, half_window=half_window)
        probe_anomaly = series - probe
        if ar_order is None:
            chosen_order = _select_ar_order(
                probe_anomaly, np.arange(origin, selection_start), selection_rows
            )
        if ridge_alpha is None or boosting_params is None:
            fit_x, sel_x = _design(
                probe_anomaly,
                probe,
                calendar,
                extras,
                fit_end=selection_start,
                label_rows=selection_rows,
                lags=lags,
            )
            y_fit = probe_anomaly[np.arange(origin, selection_start)]
            y_sel = probe_anomaly[selection_rows]
            if ridge_alpha is None:
                chosen_alpha = _select_alpha(fit_x, y_fit, sel_x, y_sel)
            if boosting_params is None:
                chosen_boosting = _select_boosting(fit_x, y_fit, sel_x, y_sel, seed)
    if chosen_boosting is None:
        chosen_boosting = dict(_DEFAULT_BOOSTING)

    clim = _climatology(series, doy, first, half_window=half_window)
    anomaly = series - clim
    features_fit, features_out = _design(
        anomaly,
        clim,
        calendar,
        extras,
        fit_end=first,
        label_rows=rows,
        lags=lags,
    )
    fit_rows = np.arange(origin, first)
    y_fit = anomaly[fit_rows]
    ar_coef = _fit_ar(anomaly, int(chosen_order), fit_rows)
    ridge_packed = _fit_ridge(features_fit, y_fit, float(chosen_alpha))
    booster = _fit_boosting(features_fit, y_fit, chosen_boosting, seed)
    return {
        "climatology": clim[rows],
        "anomaly_ar": clim[rows] + _predict_ar(anomaly, ar_coef, rows),
        "ridge": clim[rows] + _apply_ridge(features_out, ridge_packed),
        "gradient_boosting": clim[rows] + booster.predict(features_out),
    }


def covariate_frame(
    frame: pd.DataFrame, columns: list[str], fit_row_end: int
) -> dict[str, np.ndarray]:
    """Numeric covariate columns, with NaNs left in place for the model matrix.

    A missing indicator is added when the fitting rows contain a NaN. Medians
    used by the linear models are computed inside the design matrix.
    """
    covariates: dict[str, np.ndarray] = {}
    for column in columns:
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        covariates[column] = values
        if np.isnan(values[:fit_row_end]).any():
            covariates[f"{column}_missing"] = np.isnan(values).astype(np.float64)
    return covariates


def _day_of_year(dates: np.ndarray) -> np.ndarray:
    stamps = pd.to_datetime(dates)
    return np.asarray(stamps.dayofyear, dtype=np.int64)


def _climatology(
    values: np.ndarray, doy: np.ndarray, fit_end: int, *, half_window: int
) -> np.ndarray:
    sums = np.zeros(367, dtype=np.float64)
    counts = np.zeros(367, dtype=np.float64)
    fit_doy = doy[:fit_end]
    fit_values = values[:fit_end]
    np.add.at(sums, fit_doy, fit_values)
    np.add.at(counts, fit_doy, 1.0)
    smoothed = np.zeros(367, dtype=np.float64)
    for day in range(1, 367):
        numerator = 0.0
        denominator = 0.0
        for offset in range(-half_window, half_window + 1):
            source = (day - 1 + offset) % 366 + 1
            numerator += sums[source]
            denominator += counts[source]
        smoothed[day] = numerator / denominator if denominator else np.nan
    fallback = float(np.mean(fit_values)) if fit_values.size else 0.0
    filled = np.where(np.isfinite(smoothed), smoothed, fallback)
    return np.asarray(filled[doy], dtype=np.float64)


def _design(
    anomaly: np.ndarray,
    clim: np.ndarray,
    dates: np.ndarray,
    covariates: dict[str, np.ndarray],
    *,
    fit_end: int,
    label_rows: np.ndarray,
    lags: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    origin = max(lags)
    fit_rows = np.arange(origin, fit_end)
    fills = {
        name: _median_fill(column[:fit_end])
        for name, column in covariates.items()
        if not name.endswith("_missing")
    }

    def rows(indexes: np.ndarray) -> np.ndarray:
        pieces = [clim[indexes][:, None], _sin_cos(dates[indexes])]
        for lag in lags:
            pieces.append(anomaly[indexes - lag][:, None])
        if origin >= 7:
            window = np.stack([anomaly[indexes - lag] for lag in range(1, 8)], axis=1)
            pieces.append(window.mean(axis=1, keepdims=True))
        for name, column in covariates.items():
            lagged = column[indexes - 1]
            if name.endswith("_missing"):
                pieces.append(lagged[:, None])
            else:
                pieces.append(
                    np.where(np.isfinite(lagged), lagged, fills[name])[:, None]
                )
        return np.column_stack(pieces)

    return rows(fit_rows), rows(np.asarray(label_rows, dtype=np.int64))


def _median_fill(history: np.ndarray) -> float:
    observed = history[np.isfinite(history)]
    if observed.size == 0:
        return 0.0
    return float(np.median(observed))


def _sin_cos(dates: np.ndarray) -> np.ndarray:
    day = _day_of_year(dates).astype(np.float64)
    angle = 2.0 * np.pi * day / 365.25
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _fit_ar(anomaly: np.ndarray, order: int, fit_rows: np.ndarray) -> np.ndarray:
    design = np.column_stack([anomaly[fit_rows - lag] for lag in range(1, order + 1)])
    coef, *_ = np.linalg.lstsq(design, anomaly[fit_rows], rcond=None)
    return np.asarray(coef, dtype=np.float64)


def _predict_ar(anomaly: np.ndarray, coef: np.ndarray, rows: np.ndarray) -> np.ndarray:
    design = np.column_stack([anomaly[rows - lag] for lag in range(1, len(coef) + 1)])
    return np.asarray(design @ coef, dtype=np.float64)


def _select_ar_order(
    anomaly: np.ndarray, fit_rows: np.ndarray, selection_rows: np.ndarray
) -> int:
    best_order = 1
    best_error = np.inf
    actual = anomaly[selection_rows]
    earliest = int(fit_rows.min()) if fit_rows.size else 1
    for order in _AR_ORDERS:
        if order > earliest:
            continue
        coef = _fit_ar(anomaly, order, fit_rows)
        predicted = _predict_ar(anomaly, coef, selection_rows)
        error = float(np.mean((predicted - actual) ** 2))
        if error < best_error:
            best_error = error
            best_order = order
    return best_order


def _fit_ridge(features: np.ndarray, target: np.ndarray, alpha: float) -> np.ndarray:
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    scaled = (features - mean) / std
    design = np.column_stack([np.ones(len(scaled)), scaled])
    gram = design.T @ design
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    solved, *_ = np.linalg.lstsq(gram + penalty, design.T @ target, rcond=None)
    coef = np.asarray(solved, dtype=np.float64)
    # Store standardization in the intercept/coefficient layout used at predict
    # time: predictions are ``[1, (x-mean)/std] @ coef``. Callers that need the
    # raw mapping use ``_apply_ridge``.
    return np.concatenate([coef, mean, std])


def _apply_ridge(features: np.ndarray, packed: np.ndarray) -> np.ndarray:
    n_features = features.shape[1]
    coef = packed[: n_features + 1]
    mean = packed[n_features + 1 : 2 * n_features + 1]
    std = packed[2 * n_features + 1 :]
    scaled = (features - mean) / std
    design = np.column_stack([np.ones(len(scaled)), scaled])
    return np.asarray(design @ coef, dtype=np.float64)


def _select_alpha(
    fit_x: np.ndarray,
    y_fit: np.ndarray,
    sel_x: np.ndarray,
    y_sel: np.ndarray,
) -> float:
    best_alpha = _RIDGE_ALPHAS[0]
    best_error = np.inf
    for alpha in _RIDGE_ALPHAS:
        packed = _fit_ridge(fit_x, y_fit, alpha)
        predicted = _apply_ridge(sel_x, packed)
        error = float(np.mean((predicted - y_sel) ** 2))
        if error < best_error:
            best_error = error
            best_alpha = alpha
    return best_alpha


def _fit_boosting(
    features: np.ndarray, target: np.ndarray, params: dict[str, Any], seed: int
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=False,
        random_state=seed,
        **params,
    )
    model.fit(features, target)
    return model


def _select_boosting(
    fit_x: np.ndarray,
    y_fit: np.ndarray,
    sel_x: np.ndarray,
    y_sel: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    best = _DEFAULT_BOOSTING
    best_error = np.inf
    for params in _BOOSTING_GRID:
        model = _fit_boosting(fit_x, y_fit, params, seed)
        predicted = model.predict(sel_x)
        error = float(np.mean((predicted - y_sel) ** 2))
        if error < best_error:
            best_error = error
            best = params
    return dict(best)
