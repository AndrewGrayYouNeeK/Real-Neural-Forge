"""Principal component analysis via eigenvectors of the covariance matrix.

Uses ``numpy.linalg.eigh`` (no sklearn). Fitted on rows × features; for
windowed series we flatten all timesteps as sample rows so multi-column CSVs
yield which features move together and how much variance each component keeps.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
import torch

_EPS = 1e-12


@dataclass(frozen=True)
class PCAModel:
    """Centered PCA projection fitted with ``numpy.linalg.eigh``."""

    mean: np.ndarray  # (F,)
    components: np.ndarray  # (k, F) — rows are principal axes (largest variance first)
    explained_variance: np.ndarray  # (k,)
    explained_variance_ratio: np.ndarray  # (k,)
    feature_names: tuple[str, ...]

    @property
    def n_features(self) -> int:
        return int(self.mean.shape[0])

    @property
    def n_components(self) -> int:
        return int(self.components.shape[0])

    def transform(self, x: np.ndarray | torch.Tensor) -> np.ndarray:
        """Project ``(..., F)`` onto the top ``k`` components → ``(..., k)``."""
        arr = _as_numpy(x)
        if arr.shape[-1] != self.n_features:
            raise ValueError(
                f"Expected last dim {self.n_features}, got {arr.shape[-1]}."
            )
        centered = arr - self.mean
        projected = centered @ self.components.T
        return np.asarray(projected, dtype=np.float64)

    def transform_torch(self, x: torch.Tensor) -> torch.Tensor:
        """Project a float tensor the same way as :meth:`transform`."""
        projected = self.transform(x)
        return torch.as_tensor(projected, dtype=x.dtype, device=x.device)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": self.mean.astype(np.float64).tolist(),
            "components": self.components.astype(np.float64).tolist(),
            "explained_variance": self.explained_variance.astype(np.float64).tolist(),
            "explained_variance_ratio": self.explained_variance_ratio.astype(
                np.float64
            ).tolist(),
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> PCAModel:
        names = state.get("feature_names") or []
        return cls(
            mean=np.asarray(state["mean"], dtype=np.float64),
            components=np.asarray(state["components"], dtype=np.float64),
            explained_variance=np.asarray(
                state["explained_variance"], dtype=np.float64
            ),
            explained_variance_ratio=np.asarray(
                state["explained_variance_ratio"], dtype=np.float64
            ),
            feature_names=tuple(str(name) for name in names),
        )


@dataclass(frozen=True)
class PCAReport:
    """Plain-language PCA summary for API / agent / UI."""

    model: PCAModel
    n_samples: int
    loadings: list[dict[str, Any]]
    groups: list[dict[str, Any]]
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.model.n_features,
            "n_components": self.model.n_components,
            "feature_names": list(self.model.feature_names),
            "explained_variance": self.model.explained_variance.tolist(),
            "explained_variance_ratio": self.model.explained_variance_ratio.tolist(),
            "cumulative_variance_ratio": np.cumsum(
                self.model.explained_variance_ratio
            ).tolist(),
            "components": self.model.components.tolist(),
            "loadings": self.loadings,
            "groups": self.groups,
            "summary": self.summary,
        }


def fit_pca(
    x: np.ndarray | torch.Tensor,
    *,
    n_components: int | None = None,
    feature_names: list[str] | tuple[str, ...] | None = None,
) -> PCAModel:
    """Fit PCA on a 2-D matrix ``(N, F)`` using eigenvectors of the covariance."""
    samples = _as_2d(_as_numpy(x))
    n_samples, n_features = samples.shape
    if n_features < 2:
        raise ValueError("PCA needs at least 2 feature columns.")
    if n_samples < 2:
        raise ValueError("PCA needs at least 2 sample rows.")

    names = _resolve_names(feature_names, n_features)
    k = n_features if n_components is None else int(n_components)
    if k < 1 or k > n_features:
        raise ValueError(
            f"n_components must be in [1, {n_features}], got {n_components!r}."
        )

    mean = samples.mean(axis=0)
    centered = samples - mean
    # Population covariance keeps constant columns at zero variance (no NaNs).
    cov = (centered.T @ centered) / max(n_samples, 1)
    # eigh returns ascending eigenvalues; reverse for largest-first PCs.
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]

    total = float(eigenvalues.sum())
    if total < _EPS:
        raise ValueError("All features have zero variance; PCA is undefined.")

    components = eigenvectors[:, :k].T  # (k, F)
    # Flip signs so the largest absolute loading on each PC is positive.
    for index in range(k):
        axis = components[index]
        pivot = int(np.argmax(np.abs(axis)))
        if axis[pivot] < 0:
            components[index] = -axis

    explained = eigenvalues[:k]
    ratio = explained / total
    return PCAModel(
        mean=mean.astype(np.float64),
        components=components.astype(np.float64),
        explained_variance=explained.astype(np.float64),
        explained_variance_ratio=ratio.astype(np.float64),
        feature_names=names,
    )


def analyze_matrix(
    x: np.ndarray | torch.Tensor,
    *,
    n_components: int | None = None,
    feature_names: list[str] | tuple[str, ...] | None = None,
    loading_threshold: float = 0.35,
) -> PCAReport:
    """Fit PCA and build a plain-language report of co-moving columns."""
    model = fit_pca(x, n_components=n_components, feature_names=feature_names)
    samples = _as_2d(_as_numpy(x))
    loadings = _loadings(model)
    groups = _comoving_groups(model, threshold=loading_threshold)
    summary = _plain_summary(model, groups)
    return PCAReport(
        model=model,
        n_samples=int(samples.shape[0]),
        loadings=loadings,
        groups=groups,
        summary=summary,
    )


def analyze_dataframe(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    n_components: int | None = None,
    loading_threshold: float = 0.35,
) -> PCAReport:
    """Run PCA on numeric columns of a DataFrame (default: all numeric)."""
    if columns is None:
        numeric = frame.select_dtypes(include=(np.number,)).columns.tolist()
    else:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        numeric = list(columns)
    if len(numeric) < 2:
        raise ValueError("Need at least two numeric columns for PCA.")
    subset = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if subset.isna().any().any():
        raise ValueError("PCA columns must be fully numeric with no missing values.")
    return analyze_matrix(
        subset.to_numpy(dtype=np.float64),
        n_components=n_components,
        feature_names=numeric,
        loading_threshold=loading_threshold,
    )


def analyze_csv_bytes(
    data: bytes,
    *,
    columns: list[str] | None = None,
    n_components: int | None = None,
) -> PCAReport:
    """Parse a CSV body and run :func:`analyze_dataframe`."""
    frame = pd.read_csv(BytesIO(data))
    if frame.empty:
        raise ValueError("CSV is empty.")
    return analyze_dataframe(frame, columns=columns, n_components=n_components)


def _as_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(x, dtype=np.float64)


def _as_2d(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 3:
        # (N, T, F) → pool every timestep as a sample.
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 2:
        return arr
    if arr.ndim == 1:
        raise ValueError("PCA expects a 2-D feature matrix, got 1-D.")
    raise ValueError(f"PCA expects 2-D or 3-D input, got shape {arr.shape}.")


def _resolve_names(
    feature_names: list[str] | tuple[str, ...] | None, n_features: int
) -> tuple[str, ...]:
    if feature_names is None:
        return tuple(f"f{i}" for i in range(n_features))
    names = tuple(str(name) for name in feature_names)
    if len(names) != n_features:
        raise ValueError(
            f"feature_names length {len(names)} != n_features {n_features}."
        )
    return names


def _loadings(model: PCAModel) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pc_index in range(model.n_components):
        axis = model.components[pc_index]
        entries: list[dict[str, Any]] = [
            {
                "feature": model.feature_names[feat_index],
                "loading": float(axis[feat_index]),
                "abs_loading": float(abs(axis[feat_index])),
            }
            for feat_index in range(model.n_features)
        ]
        ranked = sorted(
            entries,
            key=lambda item: float(item["abs_loading"]),
            reverse=True,
        )
        rows.append(
            {
                "component": pc_index + 1,
                "explained_variance_ratio": float(
                    model.explained_variance_ratio[pc_index]
                ),
                "features": ranked,
            }
        )
    return rows


def _comoving_groups(model: PCAModel, *, threshold: float) -> list[dict[str, Any]]:
    """Features that share a strong same-sign loading on the same PC move together."""
    groups: list[dict[str, Any]] = []
    for pc_index in range(model.n_components):
        axis = model.components[pc_index]
        positive = [
            model.feature_names[i]
            for i in range(model.n_features)
            if axis[i] >= threshold
        ]
        negative = [
            model.feature_names[i]
            for i in range(model.n_features)
            if axis[i] <= -threshold
        ]
        share = float(model.explained_variance_ratio[pc_index])
        if len(positive) >= 2:
            groups.append(
                {
                    "component": pc_index + 1,
                    "direction": "same",
                    "features": positive,
                    "explained_variance_ratio": share,
                }
            )
        if len(negative) >= 2:
            groups.append(
                {
                    "component": pc_index + 1,
                    "direction": "same",
                    "features": negative,
                    "explained_variance_ratio": share,
                }
            )
        if positive and negative:
            groups.append(
                {
                    "component": pc_index + 1,
                    "direction": "opposite",
                    "features_positive": positive,
                    "features_negative": negative,
                    "explained_variance_ratio": share,
                }
            )
    return groups


def _plain_summary(model: PCAModel, groups: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    top = model.explained_variance_ratio[0] * 100.0
    parts.append(
        f"PC1 explains {top:.1f}% of the variance across "
        f"{model.n_features} columns ({', '.join(model.feature_names)})."
    )
    cum = float(
        np.cumsum(model.explained_variance_ratio)[min(1, model.n_components - 1)]
    )
    if model.n_components >= 2:
        parts.append(
            f"The first two components together keep {cum * 100.0:.1f}% of the signal."
        )
    # Strongest absolute loadings on PC1.
    axis = model.components[0]
    ranked = sorted(
        range(model.n_features),
        key=lambda i: abs(float(axis[i])),
        reverse=True,
    )
    lead = [model.feature_names[i] for i in ranked[: min(3, model.n_features)]]
    parts.append(f"Columns that matter most on PC1: {', '.join(lead)}.")
    for group in groups[:3]:
        share = group["explained_variance_ratio"] * 100.0
        if group["direction"] == "same":
            feats = ", ".join(group["features"])
            parts.append(
                f"On PC{group['component']} ({share:.1f}% variance), "
                f"{feats} move together."
            )
        else:
            pos = ", ".join(group["features_positive"])
            neg = ", ".join(group["features_negative"])
            parts.append(
                f"On PC{group['component']} ({share:.1f}% variance), "
                f"{pos} move opposite to {neg}."
            )
    if not groups:
        parts.append(
            "No strong co-moving groups (loadings below the grouping threshold)."
        )
    return " ".join(parts)
