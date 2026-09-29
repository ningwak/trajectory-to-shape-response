"""Small JSON fixture demonstrating accepted mixed timestamp precision."""

from __future__ import annotations

import json

from reflow_analysis.io import load_robot_file


def test_load_robot_file_accepts_mixed_timestamp_precision(tmp_path):
    # The first time omits fractional seconds while the second includes them;
    # both must become timezone-aware UTC timestamps in one ordered table.
    payload = {
        "robot_id": 20,
        "function": "test_run",
        "run_id": "robot-run-id",
        "run_timestamp": "2026-09-03T15:00:00+02:00",
        "rectangle_corners": [
            {"x": 1.0, "y": -1.0},
            {"x": -1.0, "y": -1.0},
            {"x": -1.0, "y": 1.0},
            {"x": 1.0, "y": 1.0},
        ],
        "corridor_width_m": 0.2,
        "trajectory": [
            {"time": "2026-09-03 13:00:00+00:00", "x": 0.0, "y": 0.0, "theta": 0.0},
            {
                "time": "2026-09-03 13:00:00.050000+00:00",
                "x": 0.1,
                "y": 0.0,
                "theta": 0.0,
            },
        ],
    }
    path = tmp_path / "R20_test.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    robot = load_robot_file(path, condition="baseline")

    assert robot.robot_id == 20
    assert robot.condition == "baseline"
    assert str(robot.samples["time"].dt.tz) == "UTC"
    assert len(robot.samples) == 2
