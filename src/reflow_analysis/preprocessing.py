"""Convert irregular robot logs into a causal, shared-time trajectory table.

Two ideas matter throughout this module:

1. *Causality*: the resampled pose at time ``t`` is the newest observation at
   or before ``t``.  Future measurements are never used.
2. *Explicit validity*: a held pose or finite-difference velocity is marked
   invalid when its supporting raw observations are too old or incomplete.

Missing estimates therefore remain missing rather than being confused with a
physical zero velocity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np
import pandas as pd

from .io import RobotTrajectory, RunData, sort_by_condition

AlignmentMode = Literal["intersection", "union"]


@dataclass(frozen=True)
class PreprocessingConfig:
    """Sampling interval, hold tolerance, and past velocity-window settings."""

    sample_rate_hz: float = 20.0
    max_hold_gap_s: float = 0.25
    velocity_window_s: float = 0.25
    alignment: AlignmentMode = "intersection"

    def __post_init__(self) -> None:
        """Reject values that make a time grid or finite difference undefined."""

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
    """Build a regular UTC grid over the shared or union robot time range.

    The main analysis uses ``intersection`` so all six robots contribute over
    the complete selected interval.  ``union`` is retained for standalone data
    inspection, where missing poses at run edges can be useful evidence.
    """

    robots = tuple(robots)
    starts = [robot.samples["time"].iloc[0] for robot in robots]
    ends = [robot.samples["time"].iloc[-1] for robot in robots]

    # Intersection chooses the latest start and earliest end.  Union chooses
    # the opposite endpoints and relies on valid_pose to mark unsupported edges.
    if config.alignment == "intersection":
        start, end = max(starts), min(ends)
    else:
        start, end = min(starts), max(ends)

    # Ceil/floor align the boundaries to the regular sampling period without
    # creating samples outside the selected raw interval.
    period = pd.to_timedelta(1.0 / config.sample_rate_hz, unit="s")
    start = start.ceil(period)
    end = end.floor(period)
    if start >= end:
        raise ValueError(
            "The selected robot trajectories do not share a usable time interval"
        )
    return pd.date_range(start=start, end=end, freq=period, tz="UTC")


def _wrap_angle(angle: pd.Series) -> pd.Series:
    """Map angles to [-pi, pi] with the atan2(sin, cos) identity."""

    return pd.Series(np.arctan2(np.sin(angle), np.cos(angle)), index=angle.index)


def _resample_robot(
    robot: RobotTrajectory,
    grid: pd.DatetimeIndex,
    config: PreprocessingConfig,
) -> pd.DataFrame:
    """Causally align one robot and estimate past-window finite differences.

    ``merge_asof(..., direction='backward')`` implements last-observation hold.
    The tolerance is the maximum permitted age of that held observation.
    Velocity uses a fixed number of past grid steps, so it is also causal.
    """

    # Unwrap before differencing so a transition from +pi to -pi is treated as
    # a small continuous rotation instead of an artificial ~2*pi jump.
    raw = robot.samples.copy()
    raw["theta_unwrapped"] = np.unwrap(raw["theta"].to_numpy())
    raw = raw.rename(columns={"time": "source_time"})

    # For every regular grid timestamp, select the newest raw observation that
    # is not in the future and not older than max_hold_gap_s.
    aligned = pd.merge_asof(
        pd.DataFrame({"time": grid}),
        raw[["source_time", "x", "y", "theta_unwrapped"]],
        left_on="time",
        right_on="source_time",
        direction="backward",
        tolerance=pd.to_timedelta(config.max_hold_gap_s, unit="s"),
    )
    # A missing source_time means no admissible past observation was found.
    # Keeping the age and validity mask makes this decision auditable.
    aligned["observation_age_s"] = (
        aligned["time"] - aligned["source_time"]
    ).dt.total_seconds()
    aligned["valid_pose"] = aligned["source_time"].notna()
    aligned["theta"] = _wrap_angle(aligned["theta_unwrapped"])

    # Convert the requested physical window to an integer number of samples.
    # max(1, ...) prevents a zero-time denominator at coarse configurations.
    lag_steps = max(1, int(round(config.velocity_window_s * config.sample_rate_hz)))
    delta_t = (aligned["time"] - aligned["time"].shift(lag_steps)).dt.total_seconds()
    aligned["vx"] = (aligned["x"] - aligned["x"].shift(lag_steps)) / delta_t
    aligned["vy"] = (aligned["y"] - aligned["y"].shift(lag_steps)) / delta_t
    aligned["omega"] = (
        aligned["theta_unwrapped"] - aligned["theta_unwrapped"].shift(lag_steps)
    ) / delta_t

    # Both endpoints alone are not enough: a long invalid gap inside the lag
    # interval would make the finite difference poorly supported.  The rolling
    # minimum requires every pose in the complete window to be valid.
    complete_window = (
        aligned["valid_pose"]
        .astype(int)
        .rolling(lag_steps + 1, min_periods=lag_steps + 1)
        .min()
        .fillna(0)
        .astype(bool)
    )
    aligned["valid_kinematics"] = complete_window
    # Replace unsupported numerical differences with NaN so later quantiles and
    # filters cannot accidentally interpret them as real measurements.
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
    """Summarize raw timing and retained validity without adding analysis features."""

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
    """Apply one shared grid to all robots in a run and concatenate their rows."""

    config = config or PreprocessingConfig()
    grid = _build_time_grid(run.robots, config)
    aligned_frames: list[pd.DataFrame] = []
    quality_rows: list[dict[str, object]] = []

    # Each robot is resampled independently on the same timestamps.  This keeps
    # robot identity explicit while enabling later same-time pairwise metrics.
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
    """Preprocess all runs and enforce baseline-first row ordering."""

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
