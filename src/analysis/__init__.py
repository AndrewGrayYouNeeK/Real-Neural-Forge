"""Lightweight analytical helpers (PCA, K-Means, …) off the request-path train loop."""

from src.analysis.kmeans import KMeansReport, cluster_dataframe, cluster_matrix
from src.analysis.pca import (
    PCAModel,
    PCAReport,
    analyze_dataframe,
    analyze_matrix,
    fit_pca,
)

__all__ = [
    "KMeansReport",
    "PCAModel",
    "PCAReport",
    "analyze_dataframe",
    "analyze_matrix",
    "cluster_dataframe",
    "cluster_matrix",
    "fit_pca",
]
