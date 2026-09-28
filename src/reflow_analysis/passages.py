"""Detect local passages and label routes against the inferred separator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry, world_to_corridor


@dataclass(frozen=True)
class PassageConfig:
    minimum_duration_s: float = 0.5
    maximum_duration_s: float = 15.0
    minimum_core_samples: int = 3
    minimum_sign_consistency: float = 0.70
    forward_speed_min_mps: float = 0.03
    forward_heading_limit_deg: float = 70.0


def _crossings(group: pd.DataFrame, gate_s: float, geometry: AnalysisGeometry) -> list[dict]:
    rows: list[dict] = []
    s, d = group["s"].to_numpy(float), group["d"].to_numpy(float)
    valid = group.get("valid_pose", pd.Series(True, index=group.index)).to_numpy(bool)
    for index in np.where(valid[:-1] & valid[1:] & (s[:-1] < gate_s) & (s[1:] >= gate_s))[0]:
        alpha = (gate_s - s[index]) / (s[index + 1] - s[index])
        d_cross = d[index] + alpha * (d[index + 1] - d[index])
        if abs(d_cross) > geometry.gate_half_width_m:
            continue
        t0, t1 = pd.Timestamp(group.iloc[index]["time"]), pd.Timestamp(group.iloc[index + 1]["time"])
        rows.append(
            {
                "time": t0 + (t1 - t0) * alpha,
                "index": int(index + 1),
                "d": float(d_cross),
            }
        )
    return rows


def _interpolated_gate_time(segment: pd.DataFrame, gate_s: float) -> pd.Timestamp | pd.NaT:
    events = _crossings(segment.reset_index(drop=True), gate_s, AnalysisGeometry(
        origin_x=0, origin_y=0, g0_approach_entry_s=-1, g1_response_entry_s=-.5,
        g2_route_s=0, g3_recovery_s=1, core_start_s=-.25, core_end_s=.25,
        gate_half_width_m=1e6,
    ))
    return events[0]["time"] if events else pd.NaT


def _path_length(segment: pd.DataFrame) -> float:
    xy = segment[["x", "y"]].dropna().to_numpy(float)
    return float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()) if len(xy) > 1 else np.nan


def _route_from_core(
    segment: pd.DataFrame, geometry: AnalysisGeometry, config: PassageConfig
) -> tuple[str, float, float, int]:
    core = segment[
        segment["s"].between(geometry.core_start_s, geometry.core_end_s)
        & segment.get("valid_pose", True)
    ]
    if len(core) < config.minimum_core_samples:
        return "unknown", np.nan, np.nan, len(core)
    score_samples = core["d"].to_numpy(float) - geometry.splitter_d(core["s"].to_numpy(float))
    score = float(np.median(score_samples))
    sign_consistency = float(max(np.mean(score_samples > 0), np.mean(score_samples < 0)))
    if abs(score) <= geometry.ambiguous_half_width_m or sign_consistency < config.minimum_sign_consistency:
        label = "ambiguous"
    else:
        label = "outer" if score > 0 else "inner"
    return label, score, sign_consistency, len(core)


def _forward_valid(segment: pd.DataFrame, config: PassageConfig) -> float:
    if not {"v_s", "v_d"}.issubset(segment.columns):
        return np.nan
    valid = segment.get("valid_kinematics", pd.Series(True, index=segment.index)).astype(bool)
    limit = np.tan(np.deg2rad(config.forward_heading_limit_deg))
    forward = valid & (segment["v_s"] > config.forward_speed_min_mps) & (
        segment["v_d"].abs() <= limit * segment["v_s"]
    )
    return float(forward.mean()) if len(segment) else np.nan


def detect_passages(
    trajectories: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: PassageConfig | None = None,
) -> pd.DataFrame:
    """Pair G0/G3 crossings and classify the median route through the core."""

    config = config or PassageConfig()
    aligned = trajectories if {"s", "d"}.issubset(trajectories) else world_to_corridor(trajectories, geometry)
    rows: list[dict[str, object]] = []
    for keys, group in aligned.groupby(["condition", "run_name", "robot_id"], sort=True):
        condition, run_name, robot_id = keys
        group = group.sort_values("time").reset_index(drop=True)
        entries = _crossings(group, geometry.g0_approach_entry_s, geometry)
        exits = _crossings(group, geometry.g3_recovery_s, geometry)
        exit_cursor = 0
        for passage_id, entry in enumerate(entries, start=1):
            while exit_cursor < len(exits) and exits[exit_cursor]["time"] <= entry["time"]:
                exit_cursor += 1
            exit_event = exits[exit_cursor] if exit_cursor < len(exits) else None
            valid = exit_event is not None
            reason = "" if valid else "missing_G3_before_run_end"
            if valid:
                duration = (exit_event["time"] - entry["time"]).total_seconds()
                if not config.minimum_duration_s <= duration <= config.maximum_duration_s:
                    valid, reason = False, "duration_out_of_range"
                segment = group.iloc[entry["index"] : exit_event["index"] + 1].copy()
                exit_cursor += 1
            else:
                duration = np.nan
                segment = group.iloc[entry["index"] :].copy()
            route_label, route_score, consistency, core_n = _route_from_core(segment, geometry, config)
            if valid and route_label == "unknown":
                valid, reason = False, "insufficient_core_samples"
            path_length = _path_length(segment)
            rows.append(
                {
                    "condition": condition, "run_name": run_name, "robot_id": int(robot_id),
                    "passage_id": passage_id, "valid": valid, "invalid_reason": reason,
                    "route_label": route_label, "route_score_m": route_score,
                    "core_sign_consistency": consistency, "core_sample_count": core_n,
                    "t_g0": entry["time"],
                    "t_g1": _interpolated_gate_time(segment, geometry.g1_response_entry_s),
                    "t_g2": _interpolated_gate_time(segment, geometry.g2_route_s),
                    "t_g3": exit_event["time"] if exit_event else pd.NaT,
                    "t_in": entry["time"], "t_out": exit_event["time"] if exit_event else pd.NaT,
                    "passage_time_s": duration, "path_length_m": path_length,
                    "excess_path_length_m": path_length - (geometry.g3_recovery_s - geometry.g0_approach_entry_s) if np.isfinite(path_length) else np.nan,
                    "d_g0": entry["d"], "d_g3": exit_event["d"] if exit_event else np.nan,
                    "forward_motion_valid_fraction": _forward_valid(segment, config),
                    "start_sample_index": entry["index"],
                    "end_sample_index": exit_event["index"] if exit_event else pd.NA,
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["condition", "run_name", "robot_id", "passage_id"]
    ).reset_index(drop=True)


def extract_passage_samples(
    trajectories: pd.DataFrame, passages: pd.DataFrame, valid_only: bool = True
) -> pd.DataFrame:
    selected = passages[passages["valid"]] if valid_only else passages
    frames: list[pd.DataFrame] = []
    for passage in selected.itertuples(index=False):
        mask = (
            (trajectories["condition"] == passage.condition)
            & (trajectories["run_name"] == passage.run_name)
            & (trajectories["robot_id"] == passage.robot_id)
            & (trajectories["time"] >= passage.t_g0)
            & (trajectories["time"] <= passage.t_g3)
        )
        frame = trajectories.loc[mask].copy()
        frame["passage_id"] = passage.passage_id
        frame["route_label"] = passage.route_label
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
