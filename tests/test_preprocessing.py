from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from reflow_analysis.io import RobotTrajectory, RunData
from reflow_analysis.preprocessing import PreprocessingConfig, preprocess_run


def make_robot(robot_id: int, start_offset_s: float = 0.0) -> RobotTrajectory:
    base = pd.Timestamp("2026-09-03 13:00:00", tz="UTC") + pd.Timedelta(seconds=start_offset_s)
    samples = pd.DataFrame(
        {
            "time": [base, base + pd.Timedelta(seconds=0.10), base + pd.Timedelta(seconds=0.20)],
            "x": [0.0, 0.10, 0.20],
            "y": [0.0, 0.0, 0.0],
            "theta": [0.0, 0.0, 0.0],
        }
    )
    return RobotTrajectory(
        source_file=Path(f"R{robot_id}.json"),
        run_name="synthetic_baseline",
        condition="baseline",
        robot_id=robot_id,
        source_run_id=f"run-{robot_id}",
        run_timestamp="2026-09-03T15:00:00+02:00",
        function="test",
        rectangle_corners=((1.0, -1.0), (-1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)),
        corridor_width_m=0.2,
        samples=samples,
    )


def test_intersection_alignment_and_causal_velocity():
    robot_1 = make_robot(20)
    robot_2 = make_robot(24, start_offset_s=0.05)
    robot_2 = replace(
        robot_2,
        samples=pd.concat(
            [
                robot_2.samples,
                pd.DataFrame(
                    {
                        "time": [pd.Timestamp("2026-09-03 13:00:00.25", tz="UTC")],
                        "x": [0.20],
                        "y": [0.0],
                        "theta": [0.0],
                    }
                ),
            ],
            ignore_index=True,
        ),
    )
    run = RunData(
        directory=Path("synthetic_baseline"),
        run_name="synthetic_baseline",
        condition="baseline",
        robots=(robot_1, robot_2),
    )
    config = PreprocessingConfig(
        sample_rate_hz=20.0,
        max_hold_gap_s=0.15,
        velocity_window_s=0.10,
        alignment="intersection",
    )

    trajectories, quality = preprocess_run(run, config)

    assert trajectories["time"].min() == pd.Timestamp("2026-09-03 13:00:00.05", tz="UTC")
    assert trajectories["time"].max() == pd.Timestamp("2026-09-03 13:00:00.20", tz="UTC")
    assert set(trajectories["robot_id"]) == {20, 24}
    assert quality["valid_pose_fraction"].min() == 1.0
    assert trajectories["valid_kinematics"].any()
