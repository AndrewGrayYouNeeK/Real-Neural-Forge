"""Lightweight analytical helpers (PCA, K-Means, why-forecast, …)."""

from src.analysis.kmeans import KMeansReport, cluster_dataframe, cluster_matrix
from src.analysis.pca import (
    PCAModel,
    PCAReport,
    analyze_dataframe,
    analyze_matrix,
    fit_pca,
)
from src.analysis.why import attach_why, build_why_explanation

__all__ = [
    "KMeansReport",
    "PCAModel",
    "PCAReport",
    "analyze_dataframe",
    "analyze_matrix",
    "attach_why",
    "build_why_explanation",
    "cluster_dataframe",
    "cluster_matrix",
    "fit_pca",
]
