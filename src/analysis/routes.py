"""HTTP routes for lightweight analysis helpers (PCA, K-Means)."""

from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src.analysis.kmeans import cluster_csv_bytes, cluster_matrix
from src.analysis.pca import analyze_csv_bytes, analyze_matrix

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
