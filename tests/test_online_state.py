from __future__ import annotations

import pandas as pd

from reflow_analysis.geometry import AnalysisGeometry
from reflow_analysis.online_state import OnlineStateConfig, extract_online_state


def test_recent_crossings_form_compact_quantile_state():
    rows = []
    times = pd.date_range("2026-01-01", periods=2, freq="500ms", tz="UTC")
    for elapsed, time in enumerate(times):
        for robot_id in range(6):
            rows.append(
                {
                    "condition": "baseline",
                    "run_name": "synthetic",
                    "time": time,
                    "elapsed_s": elapsed * 0.5,
                    "robot_id": robot_id,
                    "x": -0.70 + robot_id * 0.04 + elapsed * 0.01,
                    "y": -0.10 + robot_id * 0.02,
                    "vx": 0.10 + robot_id * 0.01,
                    "vy": 0.0,
                    "valid_pose": True,
                    "valid_kinematics": True,
                }
            )
    trajectories = pd.DataFrame(rows)
    passages = pd.DataFrame(
        {
            "condition": ["baseline"] * 3,
            "run_name": ["synthetic"] * 3,
            "robot_id": [0, 1, 2],
            "passage_id": [1, 1, 1],
            "t_g0": [times[0]] * 3,
            "d_g0": [-0.1, 0.0, 0.1],
        }
    )
    summary = extract_online_state(
        trajectories,
        passages,
        AnalysisGeometry(origin_y=0.0),
        OnlineStateConfig(state_rate_hz=2.0, crossing_window_s=15.0),
    )

    assert summary["quantiles_valid"].all()
    assert set(
        [
            "crossing_count_window",
            "time_since_last_crossing_s",
            "n_approach",
            "v_s_q33_mps",
            "v_s_q50_mps",
            "v_s_q67_mps",
            "d_q33_m",
            "d_q50_m",
            "d_q67_m",
        ]
    ).issubset(summary.columns)
