"""Compact causal state for geometry-decision support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry, world_to_corridor
from .io import sort_by_condition


@dataclass(frozen=True)
class OnlineStateConfig:
    """Sampling and support settings for the causal 9-component state."""

    state_rate_hz: float = 2.0
    crossing_window_s: float = 15.0
    minimum_crossings_for_quantiles: int = 3


def _downsample_run(run: pd.DataFrame, rate_hz: float) -> pd.DataFrame:
    """Select every nth timestamp from the regular preprocessed time grid."""

    times = pd.DatetimeIndex(sorted(run["time"].unique()))
    if len(times) < 2:
        return run.copy()
    sample_period_s = pd.Series(times).diff().dropna().dt.total_seconds().median()
    stride = max(1, round(1.0 / (rate_hz * sample_period_s)))
    return run[run["time"].isin(times[::stride])].copy()


def _nearest_value(group: pd.DataFrame, time: pd.Timestamp, column: str) -> float:
    """Read a valid kinematic value nearest to an interpolated gate crossing."""

    index = (group["time"] - time).abs().idxmin()
    row = group.loc[index]
    return float(row[column]) if row["valid_kinematics"] else np.nan


def _build_crossing_events(
    trajectories: pd.DataFrame, passages: pd.DataFrame
) -> pd.DataFrame:
    """Build one causal observation per G0 crossing."""

    rows: list[dict[str, object]] = []
    for passage in passages.itertuples(index=False):
        group = trajectories[
            (trajectories["condition"] == passage.condition)
            & (trajectories["run_name"] == passage.run_name)
            & (trajectories["robot_id"] == passage.robot_id)
        ]
        rows.append(
            {
                "condition": passage.condition,
                "run_name": passage.run_name,
                "robot_id": passage.robot_id,
                "passage_id": passage.passage_id,
                "time": passage.t_g0,
                "d_cross_m": passage.d_g0,
                "v_s_cross_mps": _nearest_value(group, passage.t_g0, "v_s"),
            }
        )
    return pd.DataFrame(rows)


def _build_robot_states(
    aligned: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: OnlineStateConfig | None = None,
) -> pd.DataFrame:
    """Downsample positions used to count robots in the approach region."""

    config = config or OnlineStateConfig()
    frames: list[pd.DataFrame] = []
    for _, run in aligned.groupby(["condition", "run_name"], sort=True):
        selected = _downsample_run(run, config.state_rate_hz)
        selected["in_approach"] = (
            selected["valid_pose"]
            & selected["s"].between(
                geometry.g0_approach_entry_s,
                geometry.g1_response_entry_s,
                inclusive="left",
            )
            & (selected["d"].abs() <= geometry.gate_half_width_m)
        )
        frames.append(selected)
    columns = [
        "condition",
        "run_name",
        "time",
        "elapsed_s",
        "robot_id",
        "s",
        "d",
        "v_s",
        "v_d",
        "speed",
        "valid_pose",
        "valid_kinematics",
        "in_approach",
    ]
    result = pd.concat(frames, ignore_index=True)
    return sort_by_condition(
        result[[column for column in columns if column in result]],
        "run_name",
        "time",
        "robot_id",
    )


def _summarize_online_state(
    robot_states: pd.DataFrame,
    crossing_events: pd.DataFrame,
    config: OnlineStateConfig | None = None,
) -> pd.DataFrame:
    """Build the 9-component causal state [N, dt, n, qv(3), qd(3)]."""

    config = config or OnlineStateConfig()
    rows: list[dict[str, object]] = []
    quantiles = [1 / 3, 1 / 2, 2 / 3]
    for keys, group in robot_states.groupby(
        ["condition", "run_name", "time"], sort=True
    ):
        condition, run_name, time = keys
        events = crossing_events[
            (crossing_events["condition"] == condition)
            & (crossing_events["run_name"] == run_name)
            & (crossing_events["time"] <= time)
            & (
                crossing_events["time"]
                > time - pd.Timedelta(seconds=config.crossing_window_s)
            )
        ]
        all_prior = crossing_events[
            (crossing_events["condition"] == condition)
            & (crossing_events["run_name"] == run_name)
            & (crossing_events["time"] <= time)
        ]
        quantile_events = events.dropna(subset=["v_s_cross_mps", "d_cross_m"])
        enough = len(quantile_events) >= config.minimum_crossings_for_quantiles
        speed_q = (
            quantile_events["v_s_cross_mps"].quantile(quantiles).to_numpy()
            if enough
            else [np.nan] * 3
        )
        lateral_q = (
            quantile_events["d_cross_m"].quantile(quantiles).to_numpy()
            if enough
            else [np.nan] * 3
        )
        rows.append(
            {
                "condition": condition,
                "run_name": run_name,
                "time": time,
                "elapsed_s": (
                    float(group["elapsed_s"].iloc[0])
                    if "elapsed_s" in group
                    else np.nan
                ),
                "crossing_count_window": int(len(events)),
                "time_since_last_crossing_s": (
                    float((time - all_prior["time"].max()).total_seconds())
                    if len(all_prior)
                    else np.nan
                ),
                "n_approach": int(group["in_approach"].sum()),
                "v_s_q33_mps": speed_q[0],
                "v_s_q50_mps": speed_q[1],
                "v_s_q67_mps": speed_q[2],
                "d_q33_m": lateral_q[0],
                "d_q50_m": lateral_q[1],
                "d_q67_m": lateral_q[2],
                "quantiles_valid": bool(enough),
            }
        )
    return sort_by_condition(pd.DataFrame(rows), "run_name", "time")


def extract_online_state(
    trajectories: pd.DataFrame,
    passages: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: OnlineStateConfig | None = None,
) -> pd.DataFrame:
    """Return the compact causal state history used for decision support."""

    config = config or OnlineStateConfig()
    aligned = (
        trajectories
        if {"s", "d", "v_s", "v_d"}.issubset(trajectories)
        else world_to_corridor(trajectories, geometry)
    )
    robot_states = _build_robot_states(aligned, geometry, config)
    events = _build_crossing_events(aligned, passages)
    return _summarize_online_state(robot_states, events, config)
