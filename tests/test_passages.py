"""Minimal complete and incomplete trajectories for passage pairing/labels."""

from __future__ import annotations

import pandas as pd
import pytest

from reflow_analysis.geometry import AnalysisGeometry
from reflow_analysis.passages import PassageConfig, detect_passages

GEOMETRY = AnalysisGeometry(
    g0_approach_entry_s=-0.40,
    g1_response_entry_s=-0.20,
    g2_route_s=0.10,
    g3_recovery_s=0.50,
    core_start_s=-0.10,
    core_end_s=0.30,
)


def _trajectory(d_route: float, stop_before_exit: bool = False) -> pd.DataFrame:
    # World x equals local s under the default frame and constant world y equals
    # route offset d, making the expected classification visually transparent.
    s_values = [-0.55, -0.40, -0.10, 0.30, 0.50]
    if stop_before_exit:
        s_values = s_values[:-1]
    times = pd.date_range("2026-01-01", periods=len(s_values), freq="500ms", tz="UTC")
    return pd.DataFrame(
        {
            "condition": "barrier_v01",
            "run_name": "synthetic",
            "robot_id": 20,
            "time": times,
            "x": s_values,
            "y": [d_route] * len(s_values),
            "vx": [0.5] * len(s_values),
            "vy": [0.0] * len(s_values),
            "speed": [0.5] * len(s_values),
            "valid_pose": [True] * len(s_values),
            "valid_kinematics": [True] * len(s_values),
        }
    )


def test_three_gate_detector_labels_inner_route():
    passages = detect_passages(
        _trajectory(-0.15), GEOMETRY, PassageConfig(minimum_core_samples=1)
    )

    assert len(passages) == 1
    assert bool(passages.loc[0, "valid"])
    assert passages.loc[0, "route_label"] == "inner"
    assert passages.loc[0, "passage_time_s"] == 1.5


def test_three_gate_detector_retains_incomplete_passage():
    passages = detect_passages(
        _trajectory(0.15, stop_before_exit=True),
        GEOMETRY,
        PassageConfig(minimum_core_samples=1),
    )

    assert len(passages) == 1
    assert not bool(passages.loc[0, "valid"])
    assert passages.loc[0, "invalid_reason"] == "missing_G3_before_run_end"


def test_new_entry_replaces_unmatched_entry_without_shifting_later_laps():
    s_values = [-0.55, -0.40, -0.10, 0.30, 0.40, -0.55, -0.40, -0.10, 0.30, 0.50]
    times = pd.date_range("2026-01-01", periods=len(s_values), freq="500ms", tz="UTC")
    trajectory = pd.DataFrame(
        {
            "condition": "barrier_v01",
            "run_name": "synthetic",
            "robot_id": 20,
            "time": times,
            "x": s_values,
            "y": [-0.15] * len(s_values),
            "vx": [0.5] * len(s_values),
            "vy": [0.0] * len(s_values),
            "speed": [0.5] * len(s_values),
            "valid_pose": [True] * len(s_values),
            "valid_kinematics": [True] * len(s_values),
        }
    )

    passages = detect_passages(
        trajectory, GEOMETRY, PassageConfig(minimum_core_samples=1)
    )

    assert len(passages) == 2
    assert passages.loc[0, "invalid_reason"] == "superseded_by_next_G0"
    assert bool(passages.loc[1, "valid"])
    assert passages.loc[1, "passage_time_s"] == 1.5


def test_path_length_uses_interpolated_gate_endpoints():
    s_values = [-0.60, -0.30, 0.20, 0.70]
    times = pd.date_range("2026-01-01", periods=len(s_values), freq="1s", tz="UTC")
    trajectory = pd.DataFrame(
        {
            "condition": "barrier_v01",
            "run_name": "synthetic",
            "robot_id": 20,
            "time": times,
            "x": s_values,
            "y": [-0.15] * len(s_values),
            "vx": [0.5] * len(s_values),
            "vy": [0.0] * len(s_values),
            "speed": [0.5] * len(s_values),
            "valid_pose": [True] * len(s_values),
            "valid_kinematics": [True] * len(s_values),
        }
    )

    passages = detect_passages(
        trajectory, GEOMETRY, PassageConfig(minimum_core_samples=1)
    )

    assert passages.loc[0, "path_length_m"] == pytest.approx(0.9)
    assert passages.loc[0, "excess_path_length_m"] == pytest.approx(0.0)
