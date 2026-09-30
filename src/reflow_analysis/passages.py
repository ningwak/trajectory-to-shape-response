"""Detect local passages and label routes against the inferred separator."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry, world_to_corridor
from .io import sort_by_condition


@dataclass(frozen=True)
class PassageConfig:
    """Basic support and duration checks for local G0-to-G3 passages."""

    minimum_duration_s: float = 0.5
    maximum_duration_s: float = 15.0
    minimum_core_samples: int = 3


def _crossings(
    group: pd.DataFrame, gate_s: float, geometry: AnalysisGeometry
) -> list[dict]:
    """Linearly interpolate all forward crossings of one transverse gate."""

    rows: list[dict] = []
    s, d = group["s"].to_numpy(float), group["d"].to_numpy(float)
    valid = group["valid_pose"].to_numpy(bool)
    for index in np.where(
        valid[:-1] & valid[1:] & (s[:-1] < gate_s) & (s[1:] >= gate_s)
    )[0]:
        alpha = (gate_s - s[index]) / (s[index + 1] - s[index])
        d_cross = d[index] + alpha * (d[index + 1] - d[index])
        if abs(d_cross) > geometry.gate_half_width_m:
            continue
        t0, t1 = pd.Timestamp(group.iloc[index]["time"]), pd.Timestamp(
            group.iloc[index + 1]["time"]
        )
        rows.append(
            {
                "time": t0 + (t1 - t0) * alpha,
                "index": int(index + 1),
                "d": float(d_cross),
                "x": float(
                    group.iloc[index]["x"]
                    + alpha * (group.iloc[index + 1]["x"] - group.iloc[index]["x"])
                ),
                "y": float(
                    group.iloc[index]["y"]
                    + alpha * (group.iloc[index + 1]["y"] - group.iloc[index]["y"])
                ),
            }
        )
    return rows


def _interpolated_gate_time(
    segment: pd.DataFrame, gate_s: float
) -> pd.Timestamp | pd.NaT:
    """Return the first forward crossing time of a gate inside one passage."""

    ordered = segment.sort_values("time").reset_index(drop=True)
    s = ordered["s"].to_numpy(float)
    candidates = np.where((s[:-1] < gate_s) & (s[1:] >= gate_s))[0]
    if not len(candidates):
        return pd.NaT
    index = int(candidates[0])
    alpha = (gate_s - s[index]) / (s[index + 1] - s[index])
    t0, t1 = pd.Timestamp(ordered.iloc[index]["time"]), pd.Timestamp(
        ordered.iloc[index + 1]["time"]
    )
    return t0 + (t1 - t0) * alpha


def _sampled_path_length(segment: pd.DataFrame) -> float:
    """Compute Euclidean length when an invalid passage lacks two endpoints."""

    xy = segment[["x", "y"]].dropna().to_numpy(float)
    return (
        float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        if len(xy) > 1
        else np.nan
    )


def _path_length_between_crossings(
    group: pd.DataFrame, entry: dict, exit_event: dict
) -> float:
    """Integrate path length between the interpolated G0 and G3 positions."""

    interior = group[
        (group["time"] > entry["time"])
        & (group["time"] < exit_event["time"])
        & group["valid_pose"]
    ][["x", "y"]].to_numpy(float)
    points = np.vstack(
        [
            np.array([[entry["x"], entry["y"]]]),
            interior,
            np.array([[exit_event["x"], exit_event["y"]]]),
        ]
    )
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def _route_from_core(
    segment: pd.DataFrame, geometry: AnalysisGeometry, config: PassageConfig
) -> tuple[str, float, int]:
    """Classify a route using the median offset over the full stable core.

    A core median is less sensitive than a single gate sample to tracking noise
    or a brief lateral reversal. Near-zero offsets are explicitly ambiguous;
    missing core coverage is unknown rather than forcing a route label.
    """

    core = segment[
        segment["s"].between(geometry.core_start_s, geometry.core_end_s)
        & segment["valid_pose"]
    ]
    if len(core) < config.minimum_core_samples:
        return "unknown", np.nan, len(core)
    score_samples = core["d"].to_numpy(float) - geometry.splitter_d(
        core["s"].to_numpy(float)
    )
    score = float(np.median(score_samples))
    if abs(score) <= geometry.ambiguous_half_width_m:
        label = "ambiguous"
    else:
        label = "outer" if score > 0 else "inner"
    return label, score, len(core)


def detect_passages(
    trajectories: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: PassageConfig | None = None,
) -> pd.DataFrame:
    """Pair G0/G3 crossings and classify the median route through the core.

    A new G0 replaces an unmatched older G0. This prevents one missed G3 in a
    closed-loop run from shifting every later entry/exit pair by one lap.
    """

    config = config or PassageConfig()
    aligned = (
        trajectories
        if {"s", "d"}.issubset(trajectories)
        else world_to_corridor(trajectories, geometry)
    )
    rows: list[dict[str, object]] = []
    for keys, group in aligned.groupby(
        ["condition", "run_name", "robot_id"], sort=True
    ):
        condition, run_name, robot_id = keys
        group = group.sort_values("time").reset_index(drop=True)
        entries = _crossings(group, geometry.g0_approach_entry_s, geometry)
        exits = _crossings(group, geometry.g3_recovery_s, geometry)
        for passage_id, entry in enumerate(entries, start=1):
            entry["passage_id"] = passage_id

        events = sorted(
            [("entry", event) for event in entries]
            + [("exit", event) for event in exits],
            key=lambda item: item[1]["time"],
        )
        paired_events: list[tuple[dict, dict | None, str, int]] = []
        active_entry: dict | None = None
        for event_type, event in events:
            if event_type == "entry":
                if active_entry is not None:
                    paired_events.append(
                        (
                            active_entry,
                            None,
                            "superseded_by_next_G0",
                            event["index"],
                        )
                    )
                active_entry = event
            elif active_entry is not None:
                paired_events.append((active_entry, event, "", event["index"] + 1))
                active_entry = None
        if active_entry is not None:
            paired_events.append(
                (active_entry, None, "missing_G3_before_run_end", len(group))
            )

        for entry, exit_event, reason, segment_end in paired_events:
            valid = exit_event is not None
            if valid:
                duration = (exit_event["time"] - entry["time"]).total_seconds()
                if not (
                    config.minimum_duration_s <= duration <= config.maximum_duration_s
                ):
                    valid, reason = False, "duration_out_of_range"
            else:
                duration = np.nan
            segment = group.iloc[entry["index"] : segment_end].copy()
            route_label, route_score, core_n = _route_from_core(
                segment, geometry, config
            )
            if valid and route_label == "unknown":
                valid, reason = False, "insufficient_core_samples"
            path_length = (
                _path_length_between_crossings(group, entry, exit_event)
                if exit_event is not None
                else _sampled_path_length(segment)
            )
            rows.append(
                {
                    "condition": condition,
                    "run_name": run_name,
                    "robot_id": int(robot_id),
                    "passage_id": entry["passage_id"],
                    "valid": valid,
                    "invalid_reason": reason,
                    "route_label": route_label,
                    "route_score_m": route_score,
                    "core_sample_count": core_n,
                    "t_g0": entry["time"],
                    "t_g1": _interpolated_gate_time(
                        segment, geometry.g1_response_entry_s
                    ),
                    "t_g2": _interpolated_gate_time(segment, geometry.g2_route_s),
                    "t_g3": exit_event["time"] if exit_event else pd.NaT,
                    "passage_time_s": duration,
                    "path_length_m": path_length,
                    "excess_path_length_m": (
                        path_length
                        - (geometry.g3_recovery_s - geometry.g0_approach_entry_s)
                        if np.isfinite(path_length)
                        else np.nan
                    ),
                    "d_g0": entry["d"],
                    "d_g3": exit_event["d"] if exit_event else np.nan,
                }
            )
    return sort_by_condition(pd.DataFrame(rows), "run_name", "robot_id", "passage_id")


def extract_passage_samples(
    trajectories: pd.DataFrame, passages: pd.DataFrame, valid_only: bool = True
) -> pd.DataFrame:
    """Collect local trajectory samples belonging to valid detected passages."""

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
