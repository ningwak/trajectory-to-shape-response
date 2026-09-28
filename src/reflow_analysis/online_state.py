"""Compact causal state for geometry-decision support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry, world_to_corridor


@dataclass(frozen=True)
class OnlineStateConfig:
    state_rate_hz: float = 2.0
    crossing_window_s: float = 15.0
    minimum_crossings_for_quantiles: int = 3
    forward_speed_min_mps: float = 0.03
    forward_heading_limit_deg: float = 70.0


def _state_times(group: pd.DataFrame, rate_hz: float) -> pd.DatetimeIndex:
    start, end = group["time"].min(), group["time"].max()
    return pd.date_range(start, end, freq=pd.to_timedelta(1.0 / rate_hz, unit="s"), tz="UTC")


def _nearest_value(group: pd.DataFrame, time: pd.Timestamp, column: str) -> float:
    if group.empty or column not in group:
        return np.nan
    index = (group["time"] - time).abs().idxmin()
    return float(group.loc[index, column])


def build_crossing_events(
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
                "condition": passage.condition, "run_name": passage.run_name,
                "robot_id": passage.robot_id, "passage_id": passage.passage_id,
                "time": passage.t_g0, "d_cross_m": passage.d_g0,
                "v_s_cross_mps": _nearest_value(group, passage.t_g0, "v_s"),
            }
        )
    return pd.DataFrame(rows)


def build_online_robot_states(
    trajectories: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: OnlineStateConfig | None = None,
) -> pd.DataFrame:
    """Return downsampled internal robot states and quality masks."""

    config = config or OnlineStateConfig()
    aligned = trajectories if {"s", "d", "v_s", "v_d"}.issubset(trajectories) else world_to_corridor(trajectories, geometry)
    frames: list[pd.DataFrame] = []
    heading_limit = np.tan(np.deg2rad(config.forward_heading_limit_deg))
    for _, run in aligned.groupby(["condition", "run_name"], sort=True):
        times = _state_times(run, config.state_rate_hz)
        selected = run[run["time"].isin(times)].copy()
        if selected.empty:
            base_times = pd.DatetimeIndex(sorted(run["time"].unique()))
            stride = max(1, round(len(base_times) / max(len(times), 1)))
            selected = run[run["time"].isin(base_times[::stride])].copy()
        valid_pose = selected.get("valid_pose", True)
        valid_kinematics = selected.get("valid_kinematics", True)
        selected["forward_motion_valid"] = (
            valid_pose & valid_kinematics & (selected["v_s"] > config.forward_speed_min_mps)
            & (selected["v_d"].abs() <= heading_limit * selected["v_s"])
        )
        selected["in_approach"] = (
            valid_pose & selected["s"].between(
                geometry.g0_approach_entry_s, geometry.g1_response_entry_s, inclusive="left"
            ) & (selected["d"].abs() <= geometry.gate_half_width_m)
        )
        frames.append(selected)
    columns = [
        "condition", "run_name", "time", "elapsed_s", "robot_id", "s", "d",
        "v_s", "v_d", "speed", "valid_pose", "valid_kinematics",
        "forward_motion_valid", "in_approach",
    ]
    result = pd.concat(frames, ignore_index=True)
    return result[[column for column in columns if column in result]].sort_values(
        ["condition", "run_name", "time", "robot_id"]
    ).reset_index(drop=True)


def build_online_state_summary(
    robot_states: pd.DataFrame,
    crossing_events: pd.DataFrame,
    config: OnlineStateConfig | None = None,
) -> pd.DataFrame:
    """Build the 9-component causal state [N, dt, n, qv(3), qd(3)]."""

    config = config or OnlineStateConfig()
    rows: list[dict[str, object]] = []
    quantiles = [1 / 3, 1 / 2, 2 / 3]
    for keys, group in robot_states.groupby(["condition", "run_name", "time"], sort=True):
        condition, run_name, time = keys
        events = crossing_events[
            (crossing_events["condition"] == condition)
            & (crossing_events["run_name"] == run_name)
            & (crossing_events["time"] <= time)
            & (crossing_events["time"] > time - pd.Timedelta(seconds=config.crossing_window_s))
        ]
        all_prior = crossing_events[
            (crossing_events["condition"] == condition)
            & (crossing_events["run_name"] == run_name)
            & (crossing_events["time"] <= time)
        ]
        enough = len(events) >= config.minimum_crossings_for_quantiles
        speed_q = events["v_s_cross_mps"].quantile(quantiles).to_numpy() if enough else [np.nan] * 3
        lateral_q = events["d_cross_m"].quantile(quantiles).to_numpy() if enough else [np.nan] * 3
        rows.append(
            {
                "condition": condition, "run_name": run_name, "time": time,
                "elapsed_s": float(group["elapsed_s"].iloc[0]) if "elapsed_s" in group else np.nan,
                "crossing_count_window": int(len(events)),
                "time_since_last_crossing_s": float((time - all_prior["time"].max()).total_seconds()) if len(all_prior) else np.nan,
                "n_approach": int(group["in_approach"].sum()),
                "v_s_q33_mps": speed_q[0], "v_s_q50_mps": speed_q[1], "v_s_q67_mps": speed_q[2],
                "d_q33_m": lateral_q[0], "d_q50_m": lateral_q[1], "d_q67_m": lateral_q[2],
                "quantiles_valid": bool(enough),
            }
        )
    return pd.DataFrame(rows)


def extract_online_state(
    trajectories: pd.DataFrame,
    passages: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: OnlineStateConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    config = config or OnlineStateConfig()
    robot_states = build_online_robot_states(trajectories, geometry, config)
    events = build_crossing_events(trajectories, passages)
    summary = build_online_state_summary(robot_states, events, config)
    return robot_states, events, summary
