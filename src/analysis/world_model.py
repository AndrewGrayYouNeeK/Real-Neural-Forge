"""Phase 5 world model: a small learned latent simulator for one series.

The model is deliberately small so it runs on the shared CPU container:

1. **Encoder** — sliding windows of the recent past are standardised and
   projected onto their top principal components (PCA via SVD). That latent
   vector is the model's "state of the world".
2. **Dynamics** — a ridge regression maps the latent state to the next value.
   Shifting the window by that value gives the next latent state, so the model
   can be rolled forward any number of steps.
3. **Regimes** — K-Means on the latent states groups history into calm vs
   choppy regimes. Rollout noise is bootstrapped from the residuals of the
   regime the simulated state is in, so a path that drifts into a volatile
   regime gets volatile noise.

Scenarios add events (a one-off shock, a sustained drift, a volatility change,
or forcing a value) at chosen steps. Baseline and every scenario share the
same random draws, so the differences come from the events, not luck.

Every response includes an honest holdout backtest against persistence (the
"tomorrow looks like today" baseline) so users can see whether the simulator
is worth trusting on their series.
"""

from __future__ import annotations

import json
import math
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from src.analysis.jepa_world import JepaBusyError, JepaEncoder, train_jepa_encoder
from src.workspace.errors import WorkspaceError
from src.workspace.starter_forecast import STARTER_UNITS
from src.workspace.starters import starter_by_id

EventKind = Literal["shock", "drift", "volatility", "set"]
EncoderKind = Literal["pca", "jepa", "auto"]
ENCODERS = ("pca", "jepa", "auto")

MIN_POINTS = 24
MAX_POINTS = 2000
MAX_HORIZON = 60
MAX_PATHS = 256
MAX_SCENARIOS = 3
MAX_EVENTS = 4
HISTORY_POINTS = 60
_MAX_WINDOW = 28
_RIDGE = 1e-2
_VAR_TARGET = 0.95
_STABLE_RADIUS = 0.995
_CACHE_SIZE = 64
_STARTER_CACHE: OrderedDict[str, dict[str, Any]] = OrderedDict()
# Simulated values stay within this share of the historical range past min/max.
_RANGE_PAD = 0.5


@dataclass
class _Fitted:
    window: int
    mean: np.ndarray
    std: np.ndarray
    components: np.ndarray  # (latent_dim, window)
    explained: float
    weights: np.ndarray  # (latent_dim,)
    bias: float
    damping: float
    kmeans: KMeans | None
    regime_residuals: list[np.ndarray]
    regime_labels: list[str]
    regime_shares: list[float]
    low: float
    high: float
    floor: float | None = None
    jepa: JepaEncoder | None = None

    @property
    def pca_dim(self) -> int:
        return int(self.components.shape[0])

    @property
    def latent_dim(self) -> int:
        extra = self.jepa.latent_dim if self.jepa is not None else 0
        return self.pca_dim + extra

    @property
    def encoder(self) -> str:
        return "jepa" if self.jepa is not None else "pca"

    def encode(self, windows: np.ndarray) -> np.ndarray:
        scaled = (windows - self.mean) / self.std
        latent = np.asarray(scaled @ self.components.T)
        if self.jepa is None:
            return latent
        return np.hstack([latent, self.jepa.encode(scaled)])

    def step(self, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (next mean value, regime id) for a batch of windows."""
        latent = self.encode(windows)
        nxt = latent @ self.weights * self.damping + self.bias
        if self.kmeans is not None:
            regimes = self.kmeans.predict(latent)
        else:
            regimes = np.zeros(len(windows), dtype=np.int64)
        return nxt, regimes


def _window_size(n: int) -> int:
    return int(min(_MAX_WINDOW, max(4, n // 6)))


def _make_windows(values: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    count = len(values) - window
    xs = np.stack([values[i : i + window] for i in range(count)])
    ys = values[window:]
    return xs, ys


def _ar_coefficients(fitted: _Fitted, damping: float) -> np.ndarray:
    """Effective AR(window) coefficients on raw values (oldest first)."""
    linear = fitted.weights[: fitted.pca_dim]
    return np.asarray((fitted.components.T @ linear) / fitted.std * damping)


def _spectral_radius(coeffs: np.ndarray) -> float:
    window = len(coeffs)
    companion = np.zeros((window, window))
    companion[0, :] = coeffs[::-1]
    if window > 1:
        companion[1:, :-1] = np.eye(window - 1)
    return float(np.max(np.abs(np.linalg.eigvals(companion))))


def _regime_names(noise: list[float], levels: list[float]) -> list[str]:
    if len(noise) == 1:
        return ["normal"]
    order = np.argsort(noise)
    names = [""] * len(noise)
    calm_words = ["calm", "normal", "choppy", "volatile"]
    if len(noise) == 2:
        calm_words = ["calm", "volatile"]
    elif len(noise) == 3:
        calm_words = ["calm", "normal", "volatile"]
    level_order = np.argsort(levels)
    level_rank = {int(idx): rank for rank, idx in enumerate(level_order)}
    for rank, idx in enumerate(order):
        lvl = level_rank[int(idx)]
        tag = "low" if lvl == 0 else ("high" if lvl == len(levels) - 1 else "mid")
        names[int(idx)] = f"{calm_words[rank]} ({tag} level)"
    return names


def fit_world_model(
    values: np.ndarray,
    *,
    seed: int = 0,
    floor: float | None = None,
    encoder: str = "pca",
) -> _Fitted:
    """Fit encoder, dynamics, and regime noise on a 1-D series.

    ``encoder="jepa"`` adds self-supervised JEPA features (bounded, non-linear)
    to the PCA state; the ridge dynamics then read both.
    """
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} points to fit the world model.")
    window = _window_size(n)
    xs, ys = _make_windows(values, window)
    mean = xs.mean(axis=0)
    std = xs.std(axis=0)
    std = np.where(std < 1e-9, 1.0, std)
    scaled = (xs - mean) / std
    _, sing, vt = np.linalg.svd(scaled, full_matrices=False)
    var = sing**2
    total = float(var.sum()) or 1.0
    cumulative = np.cumsum(var) / total
    dim = int(np.searchsorted(cumulative, _VAR_TARGET) + 1)
    dim = int(min(max(dim, 2), 6, window))
    components = vt[:dim]
    latent = scaled @ components.T
    jepa: JepaEncoder | None = None
    if encoder == "jepa":
        jepa = train_jepa_encoder(scaled, seed=seed)
        latent = np.hstack([latent, jepa.encode(scaled)])
    design = np.hstack([latent, np.ones((len(latent), 1))])
    reg = _RIDGE * (1.0 if jepa is None else 10.0) * np.eye(design.shape[1])
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(design.T @ design + reg, design.T @ ys)
    weights = coef[:-1]
    bias = float(coef[-1])
    span = float(values.max() - values.min()) or 1.0
    fitted = _Fitted(
        window=window,
        mean=mean,
        std=std,
        components=components,
        explained=float(cumulative[dim - 1]),
        weights=weights,
        bias=bias,
        damping=1.0,
        kmeans=None,
        regime_residuals=[],
        regime_labels=[],
        regime_shares=[],
        low=float(values.min()) - _RANGE_PAD * span,
        high=float(values.max()) + _RANGE_PAD * span,
        floor=floor,
        jepa=jepa,
    )
    # Keep the rollout stable: shrink the dynamics until the AR companion
    # matrix sits inside the unit circle.
    damping = 1.0
    while (
        damping > 0.5
        and _spectral_radius(_ar_coefficients(fitted, damping)) > _STABLE_RADIUS
    ):
        damping -= 0.02
    if damping != 1.0:
        # Re-centre so the damped model's one-step residuals average zero.
        fitted.bias = float(np.mean(ys - latent @ weights * damping))
    fitted.damping = damping
    preds = latent @ weights * damping + fitted.bias
    residuals = ys - preds

    k = 3 if len(latent) >= 90 else (2 if len(latent) >= 30 else 1)
    if k > 1:
        km = KMeans(n_clusters=k, n_init=4, random_state=seed)
        labels = km.fit_predict(latent)
        pools = [residuals[labels == c] for c in range(k)]
        if any(len(pool) < 5 for pool in pools):
            km = None
    else:
        km = None
    if km is None:
        labels = np.zeros(len(latent), dtype=np.int64)
        pools = [residuals]
    fitted.kmeans = km
    # Regimes shape the noise (calm vs choppy); centre each pool so the
    # regime choice does not add a hidden bias to the median path.
    fitted.regime_residuals = [
        np.asarray(pool, dtype=np.float64) - float(np.mean(pool)) for pool in pools
    ]
    noise = [float(np.std(pool)) for pool in pools]
    levels = [float(np.mean(ys[labels == c])) for c in range(len(pools))]
    fitted.regime_labels = _regime_names(noise, levels)
    fitted.regime_shares = [float(np.mean(labels == c)) for c in range(len(pools))]
    return fitted


def _validate_events(
    events: list[dict[str, Any]], horizon: int
) -> list[dict[str, Any]]:
    clean: list[dict[str, Any]] = []
    for raw in events[:MAX_EVENTS]:
        kind = str(raw.get("kind", "shock"))
        if kind not in {"shock", "drift", "volatility", "set"}:
            raise ValueError("event kind must be shock, drift, volatility, or set.")
        step = int(raw.get("step", 1))
        if step < 1 or step > horizon:
            raise ValueError(f"event step must be between 1 and {horizon}.")
        value = float(raw.get("value", 0.0))
        if not math.isfinite(value):
            raise ValueError("event value must be a finite number.")
        mode = str(raw.get("mode", "abs"))
        if mode not in {"abs", "pct"}:
            raise ValueError("event mode must be abs or pct.")
        if kind == "volatility" and value < 0:
            raise ValueError("volatility multiplier must be >= 0.")
        clean.append({"kind": kind, "step": step, "value": value, "mode": mode})
    return clean


def _rollout(
    fitted: _Fitted,
    history: np.ndarray,
    *,
    horizon: int,
    uniforms: np.ndarray,
    events: list[dict[str, Any]],
    offset: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (level paths[n_paths, horizon], regimes[n_paths, horizon]).

    ``history`` is in model space (anomalies when a seasonal climatology was
    removed). ``offset`` is the climatology for each future step; it is added
    back so paths and event values are in the series' own units.
    """
    n_paths = uniforms.shape[0]
    clim = np.zeros(horizon) if offset is None else np.asarray(offset, dtype=np.float64)
    windows = np.tile(history[-fitted.window :], (n_paths, 1))
    paths = np.zeros((n_paths, horizon))
    regimes = np.zeros((n_paths, horizon), dtype=np.int64)
    drift = 0.0
    vol = 1.0
    for t in range(horizon):
        step = t + 1
        todays = [event for event in events if event["step"] == step]
        for event in todays:
            if event["kind"] == "drift":
                drift += event["value"]
            elif event["kind"] == "volatility":
                vol = event["value"]
        mean_next, regime = fitted.step(windows)
        noise = np.zeros(n_paths)
        for r, pool in enumerate(fitted.regime_residuals):
            mask = regime == r
            if not mask.any() or len(pool) == 0:
                continue
            idx = np.minimum(
                (uniforms[mask, t] * len(pool)).astype(np.int64), len(pool) - 1
            )
            noise[mask] = pool[idx]
        level = mean_next + vol * noise + drift + clim[t]
        for event in todays:
            if event["kind"] == "shock":
                if event["mode"] == "pct":
                    level = level * (1.0 + event["value"] / 100.0)
                else:
                    level = level + event["value"]
            elif event["kind"] == "set":
                level = np.full(n_paths, event["value"], dtype=np.float64)
        level = np.clip(level, fitted.low + clim[t], fitted.high + clim[t])
        if fitted.floor is not None:
            level = np.maximum(level, fitted.floor)
        paths[:, t] = level
        regimes[:, t] = regime
        windows = np.hstack([windows[:, 1:], (level - clim[t])[:, None]])
    return paths, regimes


_SEASON_DAYS = 365.25
_SEASON_MIN_SPAN_DAYS = 540


@dataclass
class _Season:
    coef: np.ndarray
    start: pd.Timestamp
    step_days: float
    history_offsets: np.ndarray

    def at(self, day_offsets: np.ndarray) -> np.ndarray:
        return np.asarray(_season_design(day_offsets, self.start) @ self.coef)


def _season_design(day_offsets: np.ndarray, start: pd.Timestamp) -> np.ndarray:
    doy = (
        start.dayofyear - 1 + np.asarray(day_offsets, dtype=np.float64)
    ) / _SEASON_DAYS
    cols = [np.ones_like(doy)]
    for k in (1, 2):
        cols.append(np.sin(2 * np.pi * k * doy))
        cols.append(np.cos(2 * np.pi * k * doy))
    return np.stack(cols, axis=1)


def _fit_season(values: np.ndarray, dates: list[str] | None) -> _Season | None:
    """Annual climatology (2 harmonics) when there are 1.5+ years of daily dates."""
    if not dates or len(dates) != len(values):
        return None
    try:
        stamps = pd.to_datetime(pd.Series(dates), errors="coerce")
    except (ValueError, TypeError):
        return None
    if stamps.isna().any():
        return None
    start = stamps.iloc[0]
    offsets = (stamps - start).dt.days.to_numpy(dtype=np.float64)
    if offsets[-1] < _SEASON_MIN_SPAN_DAYS:
        return None
    steps = np.diff(offsets)
    step_days = float(np.median(steps)) if len(steps) else 1.0
    if step_days <= 0 or step_days > 7:
        return None
    design = _season_design(offsets, start)
    coef, *_ = np.linalg.lstsq(design, values, rcond=None)
    return _Season(coef=coef, start=start, step_days=step_days, history_offsets=offsets)


def _decompose(
    values: np.ndarray, dates: list[str] | None, horizon: int
) -> tuple[np.ndarray, np.ndarray | None, bool]:
    """Return (model-space history, future climatology or None, seasonal?)."""
    season = _fit_season(values, dates)
    if season is None:
        return values, None, False
    hist_offsets = season.history_offsets
    anomalies = values - season.at(hist_offsets)
    future = hist_offsets[-1] + season.step_days * np.arange(1, horizon + 1)
    return anomalies, season.at(future), True


def _quantiles(paths: np.ndarray) -> dict[str, list[float]]:
    q10, q50, q90 = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
    return {
        "p10": [round(float(v), 4) for v in q10],
        "p50": [round(float(v), 4) for v in q50],
        "p90": [round(float(v), 4) for v in q90],
    }


def _floor_for(values: np.ndarray) -> float | None:
    """Series that never go negative (counts, fractions) stay non-negative."""
    return 0.0 if float(np.min(values)) >= 0.0 else None


def _backtest(
    values: np.ndarray,
    dates: list[str] | None,
    horizon: int,
    n_paths: int,
    seed: int,
    encoder: str = "pca",
) -> dict[str, Any]:
    """Rolling-origin holdout: up to 6 origins, model p50 vs persistence."""
    hold = int(min(horizon, max(3, len(values) // 10), 14))
    model_err: list[float] = []
    persist_err: list[float] = []
    inside: list[float] = []
    origins = 0
    for k in range(6):
        cut = len(values) - hold * (k + 1)
        train = values[:cut]
        actual = values[cut : cut + hold]
        if len(train) < MIN_POINTS * 2:
            break
        train_dates = dates[:cut] if dates else None
        hist, clim, _ = _decompose(train, train_dates, hold)
        fitted = fit_world_model(
            hist, seed=seed, floor=_floor_for(values), encoder=encoder
        )
        rng = np.random.default_rng(seed + 7 + k)
        uniforms = rng.random((n_paths, hold))
        paths, _ = _rollout(
            fitted, hist, horizon=hold, uniforms=uniforms, events=[], offset=clim
        )
        q10, q50, q90 = np.quantile(paths, [0.1, 0.5, 0.9], axis=0)
        model_err.extend(np.abs(q50 - actual).tolist())
        persist_err.extend(np.abs(train[-1] - actual).tolist())
        inside.extend(((actual >= q10) & (actual <= q90)).astype(float).tolist())
        origins += 1
    if not origins:
        return {"holdout": 0, "text": "Series too short for a holdout backtest."}
    mae_model = float(np.mean(model_err))
    mae_persist = float(np.mean(persist_err))
    return {
        "encoder": encoder,
        "holdout": hold,
        "origins": origins,
        "mae_model": round(mae_model, 4),
        "mae_persistence": round(mae_persist, 4),
        "beats_persistence": bool(mae_model < mae_persist),
        "coverage_80": round(float(np.mean(inside)), 3),
    }


_JEPA_MIN_GAIN_PCT = 2.0


def _compare(pca_bt: dict[str, Any], jepa_bt: dict[str, Any] | None) -> dict[str, Any]:
    """Head-to-head holdout numbers for the PCA and JEPA encoders."""
    out: dict[str, Any] = {
        "holdout": pca_bt.get("holdout", 0),
        "origins": pca_bt.get("origins", 0),
        "mae_persistence": pca_bt.get("mae_persistence"),
        "pca": {
            "mae": pca_bt.get("mae_model"),
            "coverage_80": pca_bt.get("coverage_80"),
            "beats_persistence": pca_bt.get("beats_persistence"),
        },
        "jepa": None,
        "winner": "pca",
        "improvement_pct": None,
        "selection_note": (
            "JEPA is only used when it beats PCA by at least "
            f"{_JEPA_MIN_GAIN_PCT:g}% on these holdouts. The pick is made on the "
            "same holdouts it is scored on, so small gaps are close to a tie."
        ),
    }
    if jepa_bt is None or not jepa_bt.get("holdout") or not pca_bt.get("holdout"):
        return out
    out["jepa"] = {
        "mae": jepa_bt.get("mae_model"),
        "coverage_80": jepa_bt.get("coverage_80"),
        "beats_persistence": jepa_bt.get("beats_persistence"),
    }
    pca_mae = float(pca_bt["mae_model"])
    jepa_mae = float(jepa_bt["mae_model"])
    if pca_mae > 0:
        gain = (pca_mae - jepa_mae) / pca_mae * 100.0
        out["improvement_pct"] = round(gain, 1)
        if gain >= _JEPA_MIN_GAIN_PCT:
            out["winner"] = "jepa"
    return out


def _comparison_text(comparison: dict[str, Any], unit: str | None) -> str:
    jepa = comparison.get("jepa")
    pca = comparison.get("pca") or {}
    if not jepa:
        note = comparison.get("note") or "JEPA could not be scored on this series."
        return f"Encoder check: {note} Used PCA."
    gain = comparison.get("improvement_pct")
    chosen = comparison.get("chosen") or comparison.get("winner")
    winner = "JEPA" if chosen == "jepa" else "PCA"
    gap = ""
    if gain is not None:
        word = "better" if float(gain) > 0 else "worse"
        gap = f" ({abs(float(gain)):.1f}% {word} for JEPA)"
    return (
        f"Encoder check: JEPA MAE {_fmt(float(jepa['mae']), unit)} vs PCA "
        f"{_fmt(float(pca['mae']), unit)}{gap}; "
        f"Nous uses {winner} here (JEPA needs a 2%+ win)."
    )


def _fmt(value: float, unit: str | None) -> str:
    text = f"{value:.1f}" if abs(value) >= 1 else f"{value:.3g}"
    if unit in {"°F", "°C"}:
        return f"{text}{unit}"
    return f"{text} {unit}".strip() if unit else text


def _range(lo: float, hi: float, unit: str | None) -> str:
    if unit in {"°F", "°C"}:
        return f"{_fmt(lo, unit)} to {_fmt(hi, unit)}"
    return f"{_fmt(lo, None)} to {_fmt(hi, unit)}"


def _scenario_text(
    name: str,
    delta: float,
    scen_end: float,
    lo: float,
    hi: float,
    horizon: int,
    unit: str | None,
) -> str:
    if abs(delta) < 1e-9:
        move = "ends about where the baseline does"
    elif delta > 0:
        move = f"ends {_fmt(abs(delta), unit)} above the baseline"
    else:
        move = f"ends {_fmt(abs(delta), unit)} below the baseline"
    return (
        f"{name}: by step {horizon} the median is {_fmt(scen_end, unit)} "
        f"(80% range {_range(lo, hi, unit)}), and it {move}."
    )


def simulate_series(
    values: list[float] | np.ndarray,
    *,
    horizon: int = 14,
    n_paths: int = 128,
    scenarios: list[dict[str, Any]] | None = None,
    seed: int = 0,
    unit: str | None = None,
    dates: list[str] | None = None,
    encoder: str = "pca",
) -> dict[str, Any]:
    """Fit the world model to ``values`` and roll baseline + scenarios forward.

    ``encoder`` picks the latent state: ``"pca"`` (linear, the default),
    ``"jepa"`` (PCA + self-supervised JEPA features), or ``"auto"`` (backtest
    both on this series and simulate with whichever has the lower holdout MAE).
    """
    raw = np.asarray(values, dtype=np.float64)
    keep = np.isfinite(raw)
    series = raw[keep]
    if dates is not None and len(dates) == len(raw):
        dates = [d for d, ok in zip(dates, keep.tolist(), strict=True) if ok]
    else:
        dates = None
    if len(series) < MIN_POINTS:
        raise ValueError(f"Need at least {MIN_POINTS} finite points to simulate.")
    if len(series) > MAX_POINTS:
        series = series[-MAX_POINTS:]
        dates = dates[-MAX_POINTS:] if dates else None
    if horizon < 1 or horizon > MAX_HORIZON:
        raise ValueError(f"horizon must be between 1 and {MAX_HORIZON}.")
    if n_paths < 16 or n_paths > MAX_PATHS:
        raise ValueError(f"n_paths must be between 16 and {MAX_PATHS}.")
    scenarios = list(scenarios or [])[:MAX_SCENARIOS]
    if encoder not in ENCODERS:
        raise ValueError("encoder must be pca, jepa, or auto.")

    bt_paths = min(n_paths, 128)
    comparison: dict[str, Any] | None = None
    chosen = "pca"
    backtest: dict[str, Any] | None = None
    busy_note = ""
    if encoder in {"jepa", "auto"}:
        pca_bt = _backtest(series, dates, horizon, bt_paths, seed, encoder="pca")
        try:
            jepa_bt: dict[str, Any] | None = _backtest(
                series, dates, horizon, bt_paths, seed, encoder="jepa"
            )
        except JepaBusyError as exc:
            if encoder == "jepa":
                raise
            jepa_bt = None
            busy_note = str(exc)
        comparison = _compare(pca_bt, jepa_bt)
        if jepa_bt is None:
            comparison["note"] = busy_note
        if encoder == "jepa":
            chosen = "jepa"
        else:
            chosen = str(comparison.get("winner") or "pca")
            if chosen not in {"pca", "jepa"}:
                chosen = "pca"
        backtest = jepa_bt if chosen == "jepa" else pca_bt
        comparison["chosen"] = chosen

    hist, clim, seasonal = _decompose(series, dates, horizon)
    try:
        fitted = fit_world_model(
            hist, seed=seed, floor=_floor_for(series), encoder=chosen
        )
    except JepaBusyError as exc:
        if encoder != "auto":
            raise
        fitted = fit_world_model(hist, seed=seed, floor=_floor_for(series))
        backtest = None
        if comparison is not None:
            comparison["chosen"] = "pca"
            comparison["note"] = str(exc)
    rng = np.random.default_rng(seed)
    uniforms = rng.random((n_paths, horizon))
    base_paths, base_regimes = _rollout(
        fitted, hist, horizon=horizon, uniforms=uniforms, events=[], offset=clim
    )
    baseline: dict[str, Any] = dict(_quantiles(base_paths))
    baseline["regime"] = [
        fitted.regime_labels[Counter(base_regimes[:, t].tolist()).most_common(1)[0][0]]
        for t in range(horizon)
    ]
    sample_paths = [[round(float(v), 4) for v in row] for row in base_paths[:3]]

    out_scenarios: list[dict[str, Any]] = []
    base_end = baseline["p50"][-1]
    for index, spec in enumerate(scenarios):
        name = str(spec.get("name") or f"Scenario {chr(65 + index)}")[:40]
        events = _validate_events(list(spec.get("events") or []), horizon)
        paths, regimes = _rollout(
            fitted, hist, horizon=horizon, uniforms=uniforms, events=events, offset=clim
        )
        q = _quantiles(paths)
        delta = [round(float(v), 4) for v in np.median(paths - base_paths, axis=0)]
        end_regimes = Counter(regimes[:, -1].tolist())
        out_scenarios.append(
            {
                "name": name,
                "events": events,
                **q,
                "delta_p50": delta,
                "end_delta": delta[-1],
                "end_regime": fitted.regime_labels[end_regimes.most_common(1)[0][0]],
                "summary": _scenario_text(
                    name,
                    delta[-1],
                    q["p50"][-1],
                    q["p10"][-1],
                    q["p90"][-1],
                    horizon,
                    unit,
                ),
            }
        )

    if backtest is None:
        backtest = _backtest(series, dates, horizon, bt_paths, seed)
    tail = series[-HISTORY_POINTS:]
    offset = len(series) - len(tail)
    tail_dates = (dates or [])[-len(tail) :] if dates else []
    history = [
        {
            "index": offset + i,
            "value": round(float(v), 4),
            **({"date": tail_dates[i]} if i < len(tail_dates) else {}),
        }
        for i, v in enumerate(tail)
    ]

    base_text = (
        f"Baseline: by step {horizon} the median is {_fmt(base_end, unit)} "
        f"(80% range {_range(baseline['p10'][-1], baseline['p90'][-1], unit)})."
    )
    if backtest.get("holdout"):
        verdict = "beat" if backtest["beats_persistence"] else "did not beat"
        bt_text = (
            f"Backtest ({backtest['origins']} holdouts of {backtest['holdout']} steps): "
            f"it {verdict} "
            f"persistence (MAE {_fmt(backtest['mae_model'], unit)} vs "
            f"{_fmt(backtest['mae_persistence'], unit)}); "
            f"{round(backtest['coverage_80'] * 100)}% landed inside the 80% range."
        )
    else:
        bt_text = str(backtest.get("text", ""))
    enc_text = _comparison_text(comparison, unit) if comparison else ""
    summary = " ".join(
        [base_text, *[s["summary"] for s in out_scenarios], bt_text, enc_text]
    ).strip()
    return {
        "unit": unit,
        "horizon": horizon,
        "n_paths": n_paths,
        "seed": seed,
        "points_used": int(len(series)),
        "model": {
            "kind": "latent_world_model",
            "description": (
                ("Annual seasonal baseline + " if seasonal else "")
                + (
                    "PCA + self-supervised JEPA encoder"
                    if fitted.jepa is not None
                    else "PCA encoder"
                )
                + f" over the last {fitted.window} steps -> "
                f"{fitted.latent_dim}-D latent state, "
                "ridge dynamics, K-Means regime noise."
            ),
            "encoder": fitted.encoder,
            "pca_dim": fitted.pca_dim,
            **({"jepa": fitted.jepa.describe()} if fitted.jepa is not None else {}),
            "seasonal": seasonal,
            "window": fitted.window,
            "latent_dim": fitted.latent_dim,
            "explained_variance": round(fitted.explained, 4),
            "damping": round(fitted.damping, 3),
            "regimes": [
                {
                    "id": i,
                    "label": fitted.regime_labels[i],
                    "share": round(fitted.regime_shares[i], 3),
                    "noise_std": round(float(np.std(pool)), 4),
                }
                for i, pool in enumerate(fitted.regime_residuals)
            ],
        },
        "history": history,
        "baseline": baseline,
        "sample_paths": sample_paths,
        "scenarios": out_scenarios,
        "backtest": backtest,
        **({"encoder_comparison": comparison} if comparison else {}),
        "summary": summary,
    }


def _c_to_f(values: np.ndarray) -> np.ndarray:
    return values * 9.0 / 5.0 + 32.0


def starter_simulation(
    starter_id: str,
    *,
    horizon: int = 14,
    n_paths: int = 128,
    scenarios: list[dict[str, Any]] | None = None,
    seed: int = 0,
    display_unit: str | None = None,
    encoder: str = "pca",
) -> dict[str, Any]:
    """Run the world model on a shipped starter series.

    Weather starters store °C. With ``display_unit="°F"`` the series is
    converted first, so every number in and out (including event values) is °F.
    """
    starter = starter_by_id(starter_id)
    if starter is None:
        raise WorkspaceError(404, "Starter not found.")
    if not starter.dataset_path.is_file():
        raise WorkspaceError(503, "The sample dataset is not available right now.")
    key = json.dumps(
        [
            starter.id,
            starter.dataset_path.stat().st_mtime_ns,
            horizon,
            n_paths,
            scenarios or [],
            seed,
            display_unit,
            encoder,
        ],
        sort_keys=True,
        default=str,
    )
    cached = _STARTER_CACHE.get(key)
    if cached is not None:
        _STARTER_CACHE.move_to_end(key)
        return cached
    try:
        frame = pd.read_csv(starter.dataset_path)
    except (OSError, ValueError) as exc:
        raise WorkspaceError(503, "Could not read the sample series.") from exc
    column = "value" if "value" in frame.columns else frame.columns[-1]
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)
    dates = (
        [str(d) for d in frame["date"].tolist()] if "date" in frame.columns else None
    )
    unit = starter.unit or STARTER_UNITS.get(starter.id)
    storage_unit = unit
    if unit == "°C" and display_unit == "°F":
        values = _c_to_f(values)
        unit = "°F"
    try:
        result = simulate_series(
            values,
            horizon=horizon,
            n_paths=n_paths,
            scenarios=scenarios,
            seed=seed,
            unit=unit,
            dates=dates,
            encoder=encoder,
        )
    except JepaBusyError as exc:
        raise WorkspaceError(503, str(exc)) from exc
    except ValueError as exc:
        raise WorkspaceError(422, str(exc)) from exc
    result["starter_id"] = starter.id
    result["series"] = starter.id
    result["name"] = starter.name
    result["storage_unit"] = storage_unit
    _STARTER_CACHE[key] = result
    while len(_STARTER_CACHE) > _CACHE_SIZE:
        _STARTER_CACHE.popitem(last=False)
    return result


def clear_simulation_cache() -> None:
    """Drop cached starter simulations (tests)."""
    _STARTER_CACHE.clear()
