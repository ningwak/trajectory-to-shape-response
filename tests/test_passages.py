from __future__ import annotations

import pandas as pd

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
    passages = detect_passages(_trajectory(-0.15), GEOMETRY, PassageConfig(minimum_core_samples=1))

    assert len(passages) == 1
    assert bool(passages.loc[0, "valid"])
    assert passages.loc[0, "route_label"] == "inner"
    assert passages.loc[0, "passage_time_s"] == 1.5


def test_three_gate_detector_retains_incomplete_passage():
    passages = detect_passages(_trajectory(0.15, stop_before_exit=True), GEOMETRY, PassageConfig(minimum_core_samples=1))

    assert len(passages) == 1
    assert not bool(passages.loc[0, "valid"])
    assert passages.loc[0, "invalid_reason"] == "missing_G3_before_run_end"
