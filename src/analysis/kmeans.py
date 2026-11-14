"""K-Means regime clustering for series windows and multi-column CSV rows.

Uses scikit-learn's Lloyd K-Means (already a project dependency). Auto-picks
``k`` with silhouette score (``2..min(8, n-1)``), labels regimes in plain
English, and flags points far from every centroid as anomalies for Phase 3
alerts.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

_MAX_K = 8
_MIN_K = 2
_ANOMALY_Z = 2.0
_EPS = 1e-12

_LEVEL_LABELS = {
    2: ("low", "high"),
    3: ("low", "medium", "high"),
    4: ("very low", "low", "high", "very high"),
    5: ("very low", "low", "medium", "high", "very high"),
    6: ("very low", "low", "mid-low", "mid-high", "high", "very high"),
    7: (
        "very low",
        "low",
        "mid-low",
        "medium",
        "mid-high",
        "high",
        "very high",
    ),
    8: (
        "very low",
        "low",
        "mid-low",
        "medium-low",
        "medium-high",
        "mid-high",
        "high",
        "very high",
    ),
}

_TEMP_LABELS = {
    2: ("cold", "hot"),
    3: ("cold", "mild", "hot"),
    4: ("cold", "cool", "warm", "hot"),
    5: ("cold", "cool", "mild", "warm", "hot"),
}

_VOL_LABELS = {
    2: ("calm", "volatile"),
    3: ("calm", "steady", "volatile"),
    4: ("calm", "quiet", "active", "volatile"),
    5: ("calm", "quiet", "steady", "active", "volatile"),
}


@dataclass(frozen=True)
class KMeansReport:
    """Plain-language regime / anomaly report."""

    n_samples: int
    n_features: int
    k: int
    k_selection: str
    feature_names: tuple[str, ...]
    labels: np.ndarray
    centroids: np.ndarray
    inertia: float
    silhouette: float | None
    silhouette_by_k: dict[int, float]
    inertia_by_k: dict[int, float]
    regimes: list[dict[str, Any]]
    anomalies: list[dict[str, Any]]
    summary: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "k": self.k,
            "k_selection": self.k_selection,
            "feature_names": list(self.feature_names),
            "labels": self.labels.astype(int).tolist(),
            "centroids": self.centroids.tolist(),
            "inertia": float(self.inertia),
            "silhouette": self.silhouette,
            "silhouette_by_k": {
                str(key): float(value) for key, value in self.silhouette_by_k.items()
            },
            "inertia_by_k": {
                str(key): float(value) for key, value in self.inertia_by_k.items()
            },
            "regimes": self.regimes,
            "anomalies": self.anomalies,
            "anomaly_count": len(self.anomalies),
            "summary": self.summary,
        }


def cluster_matrix(
    x: np.ndarray,
    *,
    feature_names: list[str] | tuple[str, ...] | None = None,
    k: int | None = None,
    max_k: int = _MAX_K,
    random_state: int = 0,
    anomaly_z: float = _ANOMALY_Z,
) -> KMeansReport:
    """Cluster rows of ``x`` (``N × F``) into regimes and flag anomalies."""
    samples = np.asarray(x, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got shape {samples.shape}.")
    n_samples, n_features = samples.shape
    if n_samples < _MIN_K + 1:
        raise ValueError(
            f"Need at least {_MIN_K + 1} samples for clustering, got {n_samples}."
        )
    if n_features < 1:
        raise ValueError("Need at least one feature column.")
    if not np.isfinite(samples).all():
        raise ValueError("Clustering requires finite numeric values.")

    names = _resolve_names(feature_names, n_features)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(samples)

    silhouette_by_k: dict[int, float] = {}
    inertia_by_k: dict[int, float] = {}
    k_selection = "fixed"
    if k is None:
        k_selection = "silhouette"
        chosen = _auto_k(
            scaled,
            max_k=max_k,
            random_state=random_state,
            silhouette_by_k=silhouette_by_k,
            inertia_by_k=inertia_by_k,
        )
    else:
        chosen = int(k)
        if chosen < _MIN_K or chosen > min(max_k, n_samples - 1):
            raise ValueError(
                f"k must be in [{_MIN_K}, {min(max_k, n_samples - 1)}], got {k}."
            )
        # Still record silhouette/inertia for the chosen k.
        model = _fit(scaled, chosen, random_state)
        inertia_by_k[chosen] = float(model.inertia_)
        if chosen >= 2 and n_samples > chosen:
            silhouette_by_k[chosen] = float(
                silhouette_score(scaled, model.labels_, metric="euclidean")
            )

    model = _fit(scaled, chosen, random_state)
    labels = np.asarray(model.labels_, dtype=np.int64)
    centroids_scaled = np.asarray(model.cluster_centers_, dtype=np.float64)
    centroids = scaler.inverse_transform(centroids_scaled)
    sil = silhouette_by_k.get(chosen)
    if sil is None and chosen >= 2 and n_samples > chosen:
        sil = float(silhouette_score(scaled, labels, metric="euclidean"))
        silhouette_by_k[chosen] = sil

    distances = np.linalg.norm(scaled - centroids_scaled[labels], axis=1)
    anomalies = _flag_anomalies(
        distances, labels=labels, samples=samples, feature_names=names, z=anomaly_z
    )
    regimes = _describe_regimes(
        labels=labels,
        centroids=centroids,
        feature_names=names,
        samples=samples,
    )
    summary = _plain_summary(
        k=chosen,
        k_selection=k_selection,
        regimes=regimes,
        anomalies=anomalies,
        silhouette=sil,
        n_samples=n_samples,
    )
    return KMeansReport(
        n_samples=n_samples,
        n_features=n_features,
        k=chosen,
        k_selection=k_selection,
        feature_names=names,
        labels=labels,
        centroids=centroids,
        inertia=float(model.inertia_),
        silhouette=sil,
        silhouette_by_k=silhouette_by_k,
        inertia_by_k=inertia_by_k,
        regimes=regimes,
        anomalies=anomalies,
        summary=summary,
    )


def cluster_dataframe(
    frame: pd.DataFrame,
    *,
    columns: list[str] | None = None,
    k: int | None = None,
    max_k: int = _MAX_K,
    window: int | None = None,
    random_state: int = 0,
) -> KMeansReport:
    """Cluster numeric CSV columns, or sliding windows of a single series."""
    if columns is None:
        numeric = frame.select_dtypes(include=(np.number,)).columns.tolist()
    else:
        missing = [name for name in columns if name not in frame.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        numeric = list(columns)
    if not numeric:
        raise ValueError("Need at least one numeric column.")
    subset = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if subset.isna().any().any():
        raise ValueError("Columns must be fully numeric with no missing values.")

    if len(numeric) == 1:
        series = subset.iloc[:, 0].to_numpy(dtype=np.float64)
        win = int(window) if window is not None else min(16, max(4, len(series) // 10))
        matrix, names = _window_features(series, win, column=numeric[0])
        return cluster_matrix(
            matrix, feature_names=names, k=k, max_k=max_k, random_state=random_state
        )
    return cluster_matrix(
        subset.to_numpy(dtype=np.float64),
        feature_names=numeric,
        k=k,
        max_k=max_k,
        random_state=random_state,
    )


def cluster_csv_bytes(
    data: bytes,
    *,
    columns: list[str] | None = None,
    k: int | None = None,
    max_k: int = _MAX_K,
    window: int | None = None,
    random_state: int = 0,
) -> KMeansReport:
    """Parse a CSV body and run :func:`cluster_dataframe`."""
    frame = pd.read_csv(BytesIO(data))
    if frame.empty:
        raise ValueError("CSV is empty.")
    return cluster_dataframe(
        frame,
        columns=columns,
        k=k,
        max_k=max_k,
        window=window,
        random_state=random_state,
    )


def _fit(scaled: np.ndarray, k: int, random_state: int) -> KMeans:
    return KMeans(
        n_clusters=k,
        n_init=10,
        random_state=random_state,
        algorithm="lloyd",
    ).fit(scaled)


def _auto_k(
    scaled: np.ndarray,
    *,
    max_k: int,
    random_state: int,
    silhouette_by_k: dict[int, float],
    inertia_by_k: dict[int, float],
) -> int:
    n_samples = scaled.shape[0]
    upper = min(max_k, _MAX_K, n_samples - 1)
    if upper < _MIN_K:
        raise ValueError("Not enough samples to choose k automatically.")
    best_k = _MIN_K
    best_sil = -1.0
    for candidate in range(_MIN_K, upper + 1):
        model = _fit(scaled, candidate, random_state)
        inertia_by_k[candidate] = float(model.inertia_)
        # Silhouette needs at least 2 clusters and a leftover point.
        if n_samples <= candidate:
            continue
        score = float(silhouette_score(scaled, model.labels_, metric="euclidean"))
        silhouette_by_k[candidate] = score
        if score > best_sil + 1e-9:
            best_sil = score
            best_k = candidate
        elif abs(score - best_sil) <= 1e-9:
            # Tie-break with elbow: prefer smaller inertia drop diminishing —
            # keep the smaller k when silhouettes match.
            best_k = min(best_k, candidate)
    return best_k


def _window_features(
    series: np.ndarray, window: int, *, column: str
) -> tuple[np.ndarray, list[str]]:
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}.")
    if len(series) < window + _MIN_K:
        raise ValueError(
            f"Series needs more than {window + _MIN_K} points for window clustering."
        )
    rows: list[np.ndarray] = []
    for start in range(0, len(series) - window + 1):
        chunk = series[start : start + window]
        slope = float(chunk[-1] - chunk[0]) / max(window - 1, 1)
        rows.append(
            np.array(
                [
                    float(chunk.mean()),
                    float(chunk.std(ddof=0)),
                    float(chunk.min()),
                    float(chunk.max()),
                    float(chunk[-1]),
                    slope,
                ],
                dtype=np.float64,
            )
        )
    names = [
        f"{column}_mean",
        f"{column}_std",
        f"{column}_min",
        f"{column}_max",
        f"{column}_last",
        f"{column}_slope",
    ]
    return np.stack(rows), names


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


def _flag_anomalies(
    distances: np.ndarray,
    *,
    labels: np.ndarray,
    samples: np.ndarray,
    feature_names: tuple[str, ...],
    z: float,
) -> list[dict[str, Any]]:
    # Tiny / singleton clusters are themselves unusual (an outlier can become
    # its own centroid, which would make distance-to-assigned-centroid ≈ 0).
    counts = np.bincount(labels, minlength=int(labels.max()) + 1)
    tiny = {int(cluster) for cluster, count in enumerate(counts) if count <= 1}
    bulk = distances[[labels[i] not in tiny for i in range(len(labels))]]
    if bulk.size == 0:
        bulk = distances
    median = float(np.median(bulk))
    mad = float(np.median(np.abs(bulk - median)))
    scale = 1.4826 * mad
    if scale < _EPS:
        scale = float(bulk.std(ddof=0)) if bulk.size else 0.0
    threshold = median + z * scale if scale > _EPS else median + _EPS
    anomalies: list[dict[str, Any]] = []
    for index, distance in enumerate(distances):
        cluster = int(labels[index])
        far = float(distance) >= threshold
        lone = cluster in tiny
        if not far and not lone:
            continue
        row = samples[index]
        anomalies.append(
            {
                "index": int(index),
                "cluster": cluster,
                "distance": float(distance),
                "threshold": float(threshold),
                "reason": "singleton_cluster"
                if lone and not far
                else "far_from_centroid",
                "values": {
                    feature_names[feat]: float(row[feat])
                    for feat in range(len(feature_names))
                },
            }
        )
    return anomalies


def _describe_regimes(
    *,
    labels: np.ndarray,
    centroids: np.ndarray,
    feature_names: tuple[str, ...],
    samples: np.ndarray,
) -> list[dict[str, Any]]:
    k = int(centroids.shape[0])
    style, order = _label_style(centroids, feature_names)
    name_bank = _name_bank(style, k)
    regimes: list[dict[str, Any]] = []
    for rank, cluster_id in enumerate(order):
        mask = labels == cluster_id
        count = int(mask.sum())
        share = count / max(len(labels), 1)
        center = centroids[cluster_id]
        primary = int(np.argmax(np.abs(center - samples.mean(axis=0))))
        label = name_bank[rank] if rank < len(name_bank) else f"regime {cluster_id + 1}"
        # Attach units-ish hint from the driving feature name.
        if style == "temperature":
            plain = f"{label} periods"
        elif style == "volatility":
            plain = f"{label} days"
        else:
            feat = feature_names[primary]
            plain = f"{label} {feat}"
        regimes.append(
            {
                "cluster": int(cluster_id),
                "label": plain,
                "short_label": label,
                "count": count,
                "share": float(share),
                "centroid": {
                    feature_names[feat]: float(center[feat])
                    for feat in range(len(feature_names))
                },
            }
        )
    return regimes


def _label_style(
    centroids: np.ndarray, feature_names: tuple[str, ...]
) -> tuple[str, np.ndarray]:
    """Pick naming style and cluster order (low → high along the driving axis)."""
    names_l = [name.lower() for name in feature_names]
    mean_idx = _find_feature(names_l, ("mean", "temp", "temperature", "value", "close"))
    std_idx = _find_feature(names_l, ("std", "vol", "volatility", "range"))
    # Spread of centroids along each feature.
    spreads = centroids.max(axis=0) - centroids.min(axis=0)
    if std_idx is not None and (
        mean_idx is None or spreads[std_idx] >= spreads[mean_idx] * 0.85
    ):
        order = np.argsort(centroids[:, std_idx])
        return "volatility", order
    if mean_idx is not None:
        order = np.argsort(centroids[:, mean_idx])
        if any(
            token in names_l[mean_idx]
            for token in ("temp", "temperature", "heat", "deg")
        ):
            return "temperature", order
        return "level", order
    # Fallback: largest-spread feature.
    primary = int(np.argmax(spreads)) if spreads.size else 0
    order = np.argsort(centroids[:, primary])
    return "level", order


def _find_feature(names: list[str], tokens: tuple[str, ...]) -> int | None:
    for index, name in enumerate(names):
        if any(token in name for token in tokens):
            return index
    return None


def _name_bank(style: str, k: int) -> tuple[str, ...]:
    if style == "temperature":
        bank = _TEMP_LABELS.get(k) or _LEVEL_LABELS.get(k)
    elif style == "volatility":
        bank = _VOL_LABELS.get(k) or _LEVEL_LABELS.get(k)
    else:
        bank = _LEVEL_LABELS.get(k)
    if bank is None:
        return tuple(f"regime-{i + 1}" for i in range(k))
    return bank


def _plain_summary(
    *,
    k: int,
    k_selection: str,
    regimes: list[dict[str, Any]],
    anomalies: list[dict[str, Any]],
    silhouette: float | None,
    n_samples: int,
) -> str:
    pick = (
        f"Auto-selected k={k} by silhouette score"
        if k_selection == "silhouette"
        else f"Using k={k}"
    )
    if silhouette is not None:
        pick += f" (silhouette {silhouette:.2f})"
    pick += f" across {n_samples} samples."
    regime_bits = [
        f"“{item['label']}” ({item['count']} samples, {item['share'] * 100:.0f}%)"
        for item in regimes
    ]
    regimes_line = " Regimes: " + "; ".join(regime_bits) + "."
    if anomalies:
        anomaly_line = (
            f" Flagged {len(anomalies)} unusual point"
            f"{'s' if len(anomalies) != 1 else ''} far from every centroid."
        )
    else:
        anomaly_line = " No strong anomalies relative to cluster centroids."
    return pick + regimes_line + anomaly_line
