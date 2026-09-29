from __future__ import annotations

import numpy as np
import pytest

from reflow_analysis.splitter_inference import _kmeans_1d


def test_one_dimensional_kmeans_finds_global_two_cluster_split():
    values = np.array([-1.64, -0.65, -0.17, -0.01, 0.66, 1.66])

    centers, labels, _ = _kmeans_1d(values)

    assert labels.tolist() == [0, 0, 0, 0, 1, 1]
    assert centers == pytest.approx([-0.6175, 1.16])


def test_silhouette_excludes_each_sample_from_its_own_cluster_distance():
    values = np.array([-1.0, -0.9, 1.0, 1.1])

    _, _, silhouette = _kmeans_1d(values)

    expected = np.mean(
        [
            (2.05 - 0.10) / 2.05,
            (1.95 - 0.10) / 1.95,
            (1.95 - 0.10) / 1.95,
            (2.05 - 0.10) / 2.05,
        ]
    )
    assert silhouette == pytest.approx(expected)
