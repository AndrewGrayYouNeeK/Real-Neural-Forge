"""HTTP routes for lightweight analysis helpers (PCA, K-Means, what-if)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.analysis.alerts import evaluate_alert
from src.analysis.kmeans import cluster_csv_bytes, cluster_matrix
from src.analysis.pca import analyze_csv_bytes, analyze_matrix
from src.analysis.report import build_starter_report
from src.analysis.synthetic import generate_synthetic_series
from src.analysis.why import build_why_explanation
from src.analysis.world_model import simulate_series, starter_simulation
from src.workspace.errors import WorkspaceError
from src.workspace.starter_forecast import starter_what_if

router = APIRouter(tags=["analysis"])


class PcaMatrixRequest(BaseModel):
    """JSON body for PCA when the client already has a numeric matrix."""

    columns: list[str] = Field(min_length=2)
    rows: list[list[float]] = Field(min_length=2)
    n_components: int | None = Field(default=None, ge=1)


class KMeansMatrixRequest(BaseModel):
    """JSON body for K-Means when the client already has a numeric matrix."""

    columns: list[str] = Field(min_length=1)
    rows: list[list[float]] = Field(min_length=3)
    k: int | None = Field(default=None, ge=2, le=8)
    max_k: int = Field(default=8, ge=2, le=8)


class WhyRequest(BaseModel):
    """History + forecast (+ optional PCA / regimes) for a plain-English why."""

    history: list[float] = Field(default_factory=list)
    forecast: float | None = None
    unit: str | None = None
    pca: dict[str, Any] | None = None
    regimes: dict[str, Any] | None = None


class WhatIfRequest(BaseModel):
    """Shock the last input on a starter series and compare forecasts."""

    starter_id: str = Field(min_length=1)
    mode: Literal["pct", "abs"] = "pct"
    value: float = -10.0
    horizon: int | None = Field(default=None, ge=1, le=64)
    feature_index: int = Field(default=0, ge=0)


class ReportRequest(BaseModel):
    """Build a shareable series summary snapshot."""

    series: str = Field(min_length=1)
    points: int = Field(default=120, ge=8, le=400)


class AlertCheckRequest(BaseModel):
    """Evaluate one alert rule against a starter series."""

    series: str = Field(min_length=1)
    kind: Literal["threshold", "anomaly"]
    config: dict[str, Any] = Field(default_factory=dict)


class SyntheticRequest(BaseModel):
    """Generate a synthetic series for download / Train-on-my-CSV."""

    pattern: Literal["sine", "trend", "seasonal", "noise", "walk", "diffusion"] = "sine"
    length: int = Field(default=256, ge=32, le=2048)
    noise: float = Field(default=0.05, ge=0.0, le=1.0)
    seed: int = Field(default=0)


class SimEvent(BaseModel):
    """One scenario event applied at a simulated step."""

    kind: Literal["shock", "drift", "volatility", "set"] = "shock"
    step: int = Field(default=1, ge=1, le=60)
    value: float = 0.0
    mode: Literal["abs", "pct"] = "abs"


class SimScenario(BaseModel):
    """A named set of events rolled forward against the shared baseline."""

    name: str | None = Field(default=None, max_length=40)
    events: list[SimEvent] = Field(default_factory=list, max_length=4)


class SimulateRequest(BaseModel):
    """Phase 5 world model: a starter id or your own numbers, plus scenarios."""

    starter_id: str | None = None
    values: list[float] | None = Field(default=None, min_length=24, max_length=2000)
    dates: list[str] | None = Field(default=None, max_length=2000)
    unit: str | None = Field(default=None, max_length=32)
    display_unit: Literal["°F"] | None = None
    horizon: int = Field(default=14, ge=1, le=60)
    n_paths: int = Field(default=128, ge=16, le=256)
    seed: int = Field(default=0, ge=0, le=1_000_000)
    scenarios: list[SimScenario] = Field(default_factory=list, max_length=3)


@router.post("/analysis/pca")
async def analyze_pca_upload(
    request: Request,
    n_components: int | None = Query(default=None, ge=1),
    columns: str | None = Query(
        default=None,
        description="Comma-separated column names; default = all numeric.",
    ),
) -> dict[str, Any]:
    """Run PCA on a raw CSV body (eigenvectors of the covariance matrix)."""
    data = await request.body()
    if not data:
        raise HTTPException(
            status_code=422,
            detail="POST a CSV body, or JSON to /analysis/pca/matrix.",
        )
    chosen = (
        [part.strip() for part in columns.split(",") if part.strip()]
        if columns
        else None
    )
    try:
        report = analyze_csv_bytes(data, columns=chosen, n_components=n_components)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return report.as_dict()


@router.post("/analysis/pca/matrix")
def analyze_pca_matrix(body: PcaMatrixRequest) -> dict[str, Any]:
    """Run PCA on an explicit numeric matrix (for tests and agent tooling)."""
    try:
        matrix = np.asarray(body.rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(body.columns):
            raise ValueError("rows must be a 2-D matrix whose width matches columns.")
        report = analyze_matrix(
            matrix,
            n_components=body.n_components,
            feature_names=body.columns,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return report.as_dict()


@router.post("/analysis/kmeans")
async def analyze_kmeans_upload(
    request: Request,
    k: int | None = Query(default=None, ge=2, le=8),
    max_k: int = Query(default=8, ge=2, le=8),
    window: int | None = Query(default=None, ge=2, le=256),
    columns: str | None = Query(
        default=None,
        description="Comma-separated column names; default = all numeric.",
    ),
) -> dict[str, Any]:
    """Cluster CSV rows or univariate windows into regimes; flag anomalies."""
    data = await request.body()
    if not data:
        raise HTTPException(
            status_code=422,
            detail="POST a CSV body, or JSON to /analysis/kmeans/matrix.",
        )
    chosen = (
        [part.strip() for part in columns.split(",") if part.strip()]
        if columns
        else None
    )
    try:
        report = cluster_csv_bytes(
            data, columns=chosen, k=k, max_k=max_k, window=window
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return report.as_dict()


@router.post("/analysis/kmeans/matrix")
def analyze_kmeans_matrix(body: KMeansMatrixRequest) -> dict[str, Any]:
    """Run K-Means on an explicit numeric matrix."""
    try:
        matrix = np.asarray(body.rows, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(body.columns):
            raise ValueError("rows must be a 2-D matrix whose width matches columns.")
        report = cluster_matrix(
            matrix,
            feature_names=body.columns,
            k=body.k,
            max_k=body.max_k,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return report.as_dict()


@router.post("/analysis/why")
def explain_why_forecast(body: WhyRequest) -> dict[str, Any]:
    """Short plain-English explanation for a prediction (trend / seasonality / drivers)."""
    return build_why_explanation(
        history=body.history,
        forecast=body.forecast,
        unit=body.unit,
        pca=body.pca,
        regimes=body.regimes,
    )


@router.post("/analysis/what-if")
def run_what_if(body: WhatIfRequest) -> dict[str, Any]:
    """Compare baseline vs shocked last-input forecast for a sample starter."""
    try:
        return starter_what_if(
            body.starter_id,
            mode=body.mode,
            value=body.value,
            horizon=body.horizon,
            feature_index=body.feature_index,
        )
    except WorkspaceError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.post("/analysis/report")
def build_series_report(body: ReportRequest) -> dict[str, Any]:
    """Assemble forecast + accuracy + regimes into a shareable snapshot."""
    try:
        return build_starter_report(body.series, points=body.points)
    except WorkspaceError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.post("/analysis/alert-check")
def run_alert_check(body: AlertCheckRequest) -> dict[str, Any]:
    """Check a threshold or K-Means anomaly alert against a sample series."""
    try:
        return evaluate_alert(body.series, body.kind, body.config)
    except WorkspaceError as exc:
        raise HTTPException(status_code=exc.status, detail=exc.detail) from exc


@router.post("/analysis/synthetic")
def run_synthetic(body: SyntheticRequest) -> dict[str, Any]:
    """Generate a realistic sample series (classical patterns or tiny diffusion)."""
    try:
        return generate_synthetic_series(
            pattern=body.pattern,
            length=body.length,
            noise=body.noise,
            seed=body.seed,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/analysis/simulate")
def run_simulation(body: SimulateRequest) -> dict[str, Any]:
    """Fit the latent world model and roll baseline + scenarios forward.

    Pass ``starter_id`` for a shipped sample, or ``values`` (24-2000 numbers)
    for your own series. Weather samples are stored in °C; send
    ``display_unit: "°F"`` to get every number (and give event values) in °F.
    """
    scenarios = [scenario.model_dump() for scenario in body.scenarios]
    if body.starter_id:
        try:
            return starter_simulation(
                body.starter_id,
                horizon=body.horizon,
                n_paths=body.n_paths,
                scenarios=scenarios,
                seed=body.seed,
                display_unit=body.display_unit,
            )
        except WorkspaceError as exc:
            raise HTTPException(status_code=exc.status, detail=exc.detail) from exc
    if not body.values:
        raise HTTPException(
            status_code=422, detail="Send starter_id or at least 24 values."
        )
    try:
        result = simulate_series(
            body.values,
            horizon=body.horizon,
            n_paths=body.n_paths,
            scenarios=scenarios,
            seed=body.seed,
            unit=body.unit,
            dates=body.dates,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    result["series"] = "custom"
    result["name"] = "Your series"
    return result
