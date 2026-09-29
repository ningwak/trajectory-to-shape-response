"""Convert cyclic robot trajectories into structured local flow passages.

The raw run is a continuous closed-loop log, whereas the interview analysis
needs repeatable local units.  A passage begins at a forward G0 crossing and
ends at the next forward G3 crossing.  A chronological state machine prevents
one missed exit from shifting every later lap.  Completed passages receive
timing, path-length, and barrier-route information.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry, world_to_corridor
from .io import sort_by_condition


@dataclass(frozen=True)
class PassageConfig:
    """Small set of duration, support, and route-consistency requirements."""

    minimum_duration_s: float = 0.5
    maximum_duration_s: float = 15.0
    minimum_core_samples: int = 3
    minimum_sign_consistency: float = 0.70


def _crossings(
    group: pd.DataFrame, gate_s: float, geometry: AnalysisGeometry
) -> list[dict]:
    """Find forward crossings and interpolate their time and lateral position.

    The test ``s[i] < gate <= s[i+1]`` counts only motion toward increasing s.
    It also gives each passage one continuous crossing estimate rather than the
    first discrete sample that happens to fall beyond the gate.
    """

    rows: list[dict] = []
    s, d = group["s"].to_numpy(float), group["d"].to_numpy(float)
    valid = group["valid_pose"].to_numpy(bool)
    for index in np.where(
        valid[:-1] & valid[1:] & (s[:-1] < gate_s) & (s[1:] >= gate_s)
    )[0]:
        # alpha is the fractional location of the gate between adjacent samples.
        alpha = (gate_s - s[index]) / (s[index + 1] - s[index])
        d_cross = d[index] + alpha * (d[index + 1] - d[index])
        # Exclude geometrically unrelated parts of the closed loop that project
        # onto the same longitudinal coordinate outside the local corridor.
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
                # Store the continuous world-frame gate point as well.  These
                # endpoints make path length consistent with interpolated time.
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
    """Interpolate the first internal forward gate crossing of one passage."""

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
    """Sum samples when an invalid passage has no complete pair of gate points."""

    xy = segment[["x", "y"]].dropna().to_numpy(float)
    return (
        float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
        if len(xy) > 1
        else np.nan
    )


def _path_length_between_crossings(
    group: pd.DataFrame, entry: dict, exit_event: dict
) -> float:
    """Measure a completed path exactly between interpolated G0 and G3 points.

    The grid sample after G0 is downstream of the true entry, and the grid
    sample after G3 is downstream of the true exit.  Using only those samples
    would omit one partial segment and add another.  Explicit interpolated
    endpoints remove that discretization bias.
    """

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
) -> tuple[str, float, float, int]:
    """Classify a barrier route from signed offsets over the stable core.

    A core median is less sensitive than a single gate sample to tracking noise
    and short lateral reversals. Sign consistency creates a separate ambiguous
    class instead of forcing a low-confidence inner/outer decision.
    """

    core = segment[
        segment["s"].between(geometry.core_start_s, geometry.core_end_s)
        & segment["valid_pose"]
    ]
    if len(core) < config.minimum_core_samples:
        return "unknown", np.nan, np.nan, len(core)
    # Positive and negative offsets denote opposite sides of the fitted
    # effective separator.  The signed median is the passage-level route score.
    score_samples = core["d"].to_numpy(float) - geometry.splitter_d(
        core["s"].to_numpy(float)
    )
    score = float(np.median(score_samples))
    # A large median alone can hide repeated side changes.  Sign consistency
    # measures the dominant fraction on either side across the entire core.
    sign_consistency = float(
        max(np.mean(score_samples > 0), np.mean(score_samples < 0))
    )
    if (
        abs(score) <= geometry.ambiguous_half_width_m
        or sign_consistency < config.minimum_sign_consistency
    ):
        label = "ambiguous"
    else:
        label = "outer" if score > 0 else "inner"
    return label, score, sign_consistency, len(core)


def detect_passages(
    trajectories: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: PassageConfig | None = None,
) -> pd.DataFrame:
    """Pair ordered G0/G3 events and build one row per local passage.

    A newly observed G0 replaces any older unmatched G0.  That old entry is
    retained as ``superseded_by_next_G0`` rather than paired to an exit on a
    later lap.  Incomplete or implausibly timed passages remain visible with an
    explicit rejection reason.
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
            # Preserve the original G0 ordering even when an entry is later
            # superseded; online arrival-state construction still needs it.
            entry["passage_id"] = passage_id

        # Merge the two already ordered event streams.  ``active_entry`` is the
        # only passage currently travelling through the local analysis region.
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
                    # Seeing another G0 means the old robot circuit reached G0
                    # again without a valid local G3.  Replacing it prevents a
                    # one-lap offset from cascading through all later pairs.
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
                # Extremely short/long pairs are retained as rejected records
                # rather than allowed to distort passage-time evaluation.
                if (
                    not config.minimum_duration_s
                    <= duration
                    <= config.maximum_duration_s
                ):
                    valid, reason = False, "duration_out_of_range"
            else:
                duration = np.nan
            segment = group.iloc[entry["index"] : segment_end].copy()
            # Route classification is computed even for an incomplete segment
            # when enough core samples exist, but only valid passages enter final
            # evaluation tables.
            route_label, route_score, consistency, core_n = _route_from_core(
                segment, geometry, config
            )
            if valid and route_label == "unknown":
                valid, reason = False, "insufficient_core_samples"
            # Compare travelled polyline length with the straight local G0-to-G3
            # longitudinal distance to obtain a simple detour measure.
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
                    "core_sign_consistency": consistency,
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
    """Recover sample-level passage segments for Figure 2 only.

    The passage table stays the primary structured representation; this helper
    attaches passage IDs/route labels back to trajectory samples for plotting.
    """

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
