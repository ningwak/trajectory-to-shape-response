"""Causal resampling and kinematic preprocessing for RoboFlow trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .io import RobotTrajectory, RunData, sort_by_condition

AlignmentMode = Literal["intersection", "union"]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Parameters for converting irregular logs into aligned trajectory samples."""

    sample_rate_hz: float = 20.0
    max_hold_gap_s: float = 0.25
    velocity_window_s: float = 0.25
    alignment: AlignmentMode = "intersection"

    def __post_init__(self) -> None:
        """Validate causal resampling parameters."""

        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.max_hold_gap_s <= 0:
            raise ValueError("max_hold_gap_s must be positive")
        if self.velocity_window_s <= 0:
            raise ValueError("velocity_window_s must be positive")
        if self.alignment not in {"intersection", "union"}:
            raise ValueError("alignment must be either 'intersection' or 'union'")


def _build_time_grid(
    robots: Iterable[RobotTrajectory], config: PreprocessingConfig
) -> pd.DatetimeIndex:
    """Build a regular shared or union time grid for one experiment run."""

    robots = tuple(robots)
    starts = [robot.samples["time"].iloc[0] for robot in robots]
    ends = [robot.samples["time"].iloc[-1] for robot in robots]

    if config.alignment == "intersection":
        start, end = max(starts), min(ends)
    else:
        start, end = min(starts), max(ends)

    period = pd.to_timedelta(1.0 / config.sample_rate_hz, unit="s")
    start = start.ceil(period)
    end = end.floor(period)
    if start >= end:
        raise ValueError(
            "The selected robot trajectories do not share a usable time interval"
        )
    return pd.date_range(start=start, end=end, freq=period, tz="UTC")


def _wrap_angle(angle: pd.Series) -> pd.Series:
    """Map an angle series to the interval [-pi, pi]."""

    return pd.Series(np.arctan2(np.sin(angle), np.cos(angle)), index=angle.index)


def _resample_robot(
    robot: RobotTrajectory,
    grid: pd.DatetimeIndex,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Resample one robot with a causal last-observation-held rule."""

    raw = robot.samples.copy()
    raw["theta_unwrapped"] = np.unwrap(raw["theta"].to_numpy())
    raw = raw.rename(columns={"time": "source_time"})

    aligned = pd.merge_asof(
        pd.DataFrame({"time": grid}),
        raw[["source_time", "x", "y", "theta_unwrapped"]],
        left_on="time",
        right_on="source_time",
        direction="backward",
        tolerance=pd.to_timedelta(config.max_hold_gap_s, unit="s"),
    )
    aligned["observation_age_s"] = (
        aligned["time"] - aligned["source_time"]
    ).dt.total_seconds()
    aligned["valid_pose"] = aligned["source_time"].notna()
    aligned["theta"] = _wrap_angle(aligned["theta_unwrapped"])

    lag_steps = max(1, int(round(config.velocity_window_s * config.sample_rate_hz)))
    delta_t = (aligned["time"] - aligned["time"].shift(lag_steps)).dt.total_seconds()
    aligned["vx"] = (aligned["x"] - aligned["x"].shift(lag_steps)) / delta_t
    aligned["vy"] = (aligned["y"] - aligned["y"].shift(lag_steps)) / delta_t
    aligned["omega"] = (
        aligned["theta_unwrapped"] - aligned["theta_unwrapped"].shift(lag_steps)
    ) / delta_t

    complete_window = (
        aligned["valid_pose"]
        .astype(int)
        .rolling(lag_steps + 1, min_periods=lag_steps + 1)
        .min()
        .fillna(0)
        .astype(bool)
    )
    aligned["valid_kinematics"] = complete_window
    aligned.loc[~aligned["valid_kinematics"], ["vx", "vy", "omega"]] = np.nan
    aligned["speed"] = np.hypot(aligned["vx"], aligned["vy"])

    aligned["condition"] = robot.condition
    aligned["run_name"] = robot.run_name
    aligned["robot_id"] = robot.robot_id
    aligned["source_run_id"] = robot.source_run_id
    aligned["elapsed_s"] = (aligned["time"] - grid[0]).dt.total_seconds()

    return aligned[
        [
            "condition",
            "run_name",
            "robot_id",
            "source_run_id",
            "time",
            "elapsed_s",
            "source_time",
            "observation_age_s",
            "x",
            "y",
            "theta",
            "vx",
            "vy",
            "speed",
            "omega",
            "valid_pose",
            "valid_kinematics",
        ]
    ]


def _quality_row(robot: RobotTrajectory, aligned: pd.DataFrame) -> dict[str, object]:
    """Summarize sampling gaps and pose/kinematic validity for one robot."""

    raw = robot.samples
    dt = raw["time"].diff().dt.total_seconds()
    return {
        "condition": robot.condition,
        "robot_id": robot.robot_id,
        "raw_sample_count": len(raw),
        "raw_duration_s": (raw["time"].iloc[-1] - raw["time"].iloc[0]).total_seconds(),
        "raw_median_dt_s": float(dt[dt > 0].median()),
        "raw_max_gap_s": float(dt.max()),
        "valid_pose_fraction": float(aligned["valid_pose"].mean()),
        "valid_kinematics_fraction": float(aligned["valid_kinematics"].mean()),
    }


def preprocess_run(
    run: RunData,
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess and time-align all robots in one experiment run."""

    config = config or PreprocessingConfig()
    grid = _build_time_grid(run.robots, config)
    aligned_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []

    for robot in run.robots:
        aligned = _resample_robot(robot, grid, config)
        aligned_frames.append(aligned)
        quality_rows.append(_quality_row(robot, aligned))

    trajectories = pd.concat(aligned_frames, ignore_index=True)
    trajectories = trajectories.sort_values(
        ["time", "robot_id"], kind="stable"
    ).reset_index(drop=True)
    quality = pd.DataFrame(quality_rows).sort_values("robot_id").reset_index(drop=True)
    return trajectories, quality


def preprocess_dataset(
    runs: Iterable[RunData],
    config: PreprocessingConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess multiple runs and return trajectory and quality tables."""

    config = config or PreprocessingConfig()
    trajectory_frames: list[pd.DataFrame] = []
    quality_frames: list[pd.DataFrame] = []
    for run in runs:
        trajectories, quality = preprocess_run(run, config)
        trajectory_frames.append(trajectories)
        quality_frames.append(quality)

    all_trajectories = sort_by_condition(
        pd.concat(trajectory_frames, ignore_index=True), "time", "robot_id"
    )
    all_quality = sort_by_condition(
        pd.concat(quality_frames, ignore_index=True), "robot_id"
    )
    return all_trajectories, all_quality
