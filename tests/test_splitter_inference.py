from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reflow_analysis.splitter_inference import (
    SplitterInferenceConfig,
    _kmeans_1d,
    _upper_reference_edge,
    infer_effective_splitter,
)


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


def test_upper_reference_edge_is_horizontal_and_ignores_corner_order():
    corners = ((0.8, -0.8), (0.8, 0.4), (-0.8, -0.8), (-0.8, 0.4))
    assert _upper_reference_edge(corners) == pytest.approx((-0.8, 0.8, 0.4))


def test_baseline_reference_does_not_choose_response_gates():
    # Eight independent toy paths each cross the same horizontal sections.
    # Their changed-condition paths separate only after x reaches -0.2.
    # This tests the data flow, not the actual RoboFlow response geometry.
    rows = []
    x_values = np.linspace(-0.8, 0.8, 17)
    for condition in ("baseline", "barrier_v01"):
        for robot_id in range(8):
            natural_y = 0.44 + 0.004 * (robot_id - 3.5)
            for index, x in enumerate(x_values):
                y = natural_y
                if condition != "baseline" and x >= -0.2:
                    y += -0.20 if robot_id < 4 else 0.20
                rows.append({
                    "condition": condition,
                    "run_name": condition,
                    "robot_id": robot_id,
                    "time": pd.Timestamp("2026-01-01", tz="UTC")
                    + pd.Timedelta(seconds=index),
                    "x": x,
                    "y": y,
                    "vx": 0.1,
                    "vy": 0.0,
                    "valid_pose": True,
                })
    trajectories = pd.DataFrame(rows)
    corners = ((0.8, -0.8), (-0.8, -0.8), (-0.8, 0.4), (0.8, 0.4))
    first = infer_effective_splitter(
        trajectories, corners,
        SplitterInferenceConfig(baseline_split_repeats=5, random_seed=1),
    )
    second = infer_effective_splitter(
        trajectories, corners,
        SplitterInferenceConfig(baseline_split_repeats=50, random_seed=2),
    )
    g1 = first.geometry.g1_response_entry_s
    assert -0.28 <= g1 <= -0.16
    assert first.geometry.g0_approach_entry_s == pytest.approx(-0.64)
    assert first.geometry.tangent_x == pytest.approx(1.0)
    assert first.geometry.tangent_y == pytest.approx(0.0)
    assert first.geometry.g1_response_entry_s == second.geometry.g1_response_entry_s
    assert first.geometry.core_start_s == second.geometry.core_start_s
    assert first.section_diagnostics["baseline_reference_m"].iloc[0] != pytest.approx(
        second.section_diagnostics["baseline_reference_m"].iloc[0]
    )


def test_separator_starts_after_natural_route_is_vacated():
    """An early one-sided turn must not masquerade as a stable two-route core."""

    rows = []
    x_values = np.linspace(-0.8, 0.8, 17)
    for condition in ("baseline", "barrier_v01"):
        for robot_id in range(8):
            natural_y = 0.4 + 0.005 * (robot_id - 3.5)
            for index, x in enumerate(x_values):
                y = natural_y
                if condition == "barrier_v01" and x >= -0.3:
                    if robot_id < 4:
                        y -= 0.26
                    elif x >= -0.1:
                        y += 0.26
                rows.append(
                    {
                        "condition": condition,
                        "run_name": condition,
                        "robot_id": robot_id,
                        "time": pd.Timestamp("2026-01-01", tz="UTC")
                        + pd.Timedelta(seconds=index),
                        "x": x,
                        "y": y,
                        "vx": 0.1,
                        "vy": 0.0,
                        "valid_pose": True,
                    }
                )
    corners = ((0.8, -0.8), (-0.8, -0.8), (-0.8, 0.4), (0.8, 0.4))
    inferred = infer_effective_splitter(pd.DataFrame(rows), corners)

    assert inferred.geometry.g1_response_entry_s < -0.2
    # Interpolation starts moving the upper group slightly before its first
    # shifted recorded pose, so assert separation *after* the one-sided stage.
    assert inferred.geometry.core_start_s > -0.2
    early = inferred.section_diagnostics[
        inferred.section_diagnostics["s"].between(-0.25, -0.19)
    ]
    assert early["natural_path_fraction"].min() > 0.10
    assert not early["core_valid"].any()
