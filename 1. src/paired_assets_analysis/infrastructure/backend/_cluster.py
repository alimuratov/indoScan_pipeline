"""Point clustering utilities."""
from __future__ import annotations

import numpy as np


def dbscan_labels(
    points: np.ndarray,
    eps: float = 0.1,
    min_samples: int = 10,
):
    """Cluster 3D points with DBSCAN.

    Returns (labels, n_clusters). Noise is labeled -1 and excluded from n_clusters.
    """
    from sklearn.cluster import DBSCAN

    if len(points) == 0:
        return np.array([]), 0
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return labels, n_clusters

