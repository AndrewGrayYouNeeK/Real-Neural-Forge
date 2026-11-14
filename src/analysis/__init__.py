"""Lightweight analytical helpers (PCA, etc.) that stay off the request-path training loop."""

from src.analysis.pca import (
    PCAModel,
    PCAReport,
    analyze_dataframe,
    analyze_matrix,
    fit_pca,
)

__all__ = [
    "PCAModel",
    "PCAReport",
    "analyze_dataframe",
    "analyze_matrix",
    "fit_pca",
]
