"""Tests for K-Means regime clustering and anomaly flags."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.analysis.kmeans import cluster_dataframe, cluster_matrix
from src.api import app


def _blob_regimes(n_per: int = 40) -> np.ndarray:
    rng = np.random.default_rng(0)
    low = rng.normal(loc=0.0, scale=0.3, size=(n_per, 2))
    mid = rng.normal(loc=5.0, scale=0.3, size=(n_per, 2))
    high = rng.normal(loc=10.0, scale=0.3, size=(n_per, 2))
    return np.vstack([low, mid, high])


class TestClusterMatrix:
    def test_auto_k_finds_three_blobs(self):
        report = cluster_matrix(_blob_regimes(), feature_names=["x", "y"])
        assert report.k_selection == "silhouette"
        assert 2 <= report.k <= 8
        # Well-separated blobs should prefer 3.
        assert report.k == 3
        assert len(report.regimes) == 3
        assert report.silhouette is not None and report.silhouette > 0.5

    def test_fixed_k(self):
        report = cluster_matrix(_blob_regimes(), k=2, feature_names=["x", "y"])
        assert report.k == 2
        assert report.k_selection == "fixed"

    def test_anomaly_far_from_centroids(self):
        data = _blob_regimes()
        # Inject a clear outlier.
        data = np.vstack([data, np.array([[100.0, -100.0]])])
        report = cluster_matrix(data, k=3, feature_names=["x", "y"])
        assert len(report.anomalies) >= 1
        indices = {item["index"] for item in report.anomalies}
        assert (len(data) - 1) in indices

    def test_plain_labels_present(self):
        report = cluster_matrix(_blob_regimes(), k=3, feature_names=["temp", "other"])
        labels = " ".join(item["label"] for item in report.regimes).lower()
        assert any(
            token in labels
            for token in ("low", "high", "cold", "hot", "mild", "medium")
        )

    def test_rejects_too_few_rows(self):
        with pytest.raises(ValueError, match="at least"):
            cluster_matrix(np.ones((2, 2)))


class TestClusterDataframe:
    def test_univariate_windows(self):
        rng = np.random.default_rng(1)
        # Two level regimes in a long series.
        series = np.concatenate(
            [rng.normal(0, 0.2, 80), rng.normal(4, 0.2, 80), rng.normal(0, 0.2, 80)]
        )
        frame = pd.DataFrame({"value": series})
        report = cluster_dataframe(frame, window=8, k=2)
        assert report.k == 2
        assert report.n_samples == len(series) - 8 + 1
        assert "value_mean" in report.feature_names

    def test_csv_bytes_roundtrip_via_api(self):
        client = TestClient(app)
        frame = pd.DataFrame(_blob_regimes(), columns=["x", "y"])
        buf = io.StringIO()
        frame.to_csv(buf, index=False)
        resp = client.post(
            "/analysis/kmeans?k=3",
            content=buf.getvalue().encode("utf-8"),
            headers={"Content-Type": "text/csv"},
        )
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["k"] == 3
        assert len(payload["regimes"]) == 3
        assert "summary" in payload

    def test_matrix_endpoint(self):
        client = TestClient(app)
        rows = _blob_regimes(30).tolist()
        resp = client.post(
            "/analysis/kmeans/matrix",
            json={"columns": ["a", "b"], "rows": rows},
        )
        assert resp.status_code == 200
        assert resp.json()["k_selection"] == "silhouette"
        assert 2 <= resp.json()["k"] <= 8
