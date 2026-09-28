"""Compact outcome evaluation and transparent supporting diagnostics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry, world_to_corridor


ROUTE_LABELS = ("inner", "outer", "ambiguous")


@dataclass(frozen=True)
class MetricConfig:
    pair_distance_rate_hz: float = 2.0
    pair_distance_quantile: float = 0.10
    speed_bin_width_m: float = 0.05


def route_distribution(passages: pd.DataFrame) -> pd.DataFrame:
    valid = passages[passages["valid"] & passages["route_label"].isin(ROUTE_LABELS)]
    conditions = sorted(passages["condition"].unique())
    index = pd.MultiIndex.from_product([conditions, ROUTE_LABELS], names=["condition", "route_label"])
    counts = valid.groupby(["condition", "route_label"]).size().reindex(index, fill_value=0)
    result = counts.rename("passage_count").reset_index()
    totals = result.groupby("condition")["passage_count"].transform("sum")
    result["route_proportion"] = np.where(totals > 0, result["passage_count"] / totals, np.nan)
    result["total_valid_passages"] = totals
    return result


def passage_time_tables(passages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = passages[passages["valid"]].copy()
    per_robot = (
        valid.groupby(["condition", "run_name", "robot_id"], as_index=False)
        .agg(
            passage_count=("passage_time_s", "size"),
            median_passage_time_s=("passage_time_s", "median"),
            mean_passage_time_s=("passage_time_s", "mean"),
            median_path_length_m=("path_length_m", "median"),
            median_excess_path_length_m=("excess_path_length_m", "median"),
        )
        .sort_values(["condition", "robot_id"])
    )
    summary = (
        valid.groupby("condition", as_index=False)
        .agg(
            valid_passages=("passage_id", "size"),
            median_passage_time_s=("passage_time_s", "median"),
            median_path_length_m=("path_length_m", "median"),
            median_excess_path_length_m=("excess_path_length_m", "median"),
        )
    )
    summary["passage_count"] = summary["valid_passages"]
    baseline_time = summary.loc[summary["condition"] == "baseline", "median_passage_time_s"]
    reference = float(baseline_time.iloc[0]) if len(baseline_time) else np.nan
    summary["relative_passage_time"] = summary["median_passage_time_s"] / reference
    return per_robot, summary


def paired_robot_passage_times(per_robot: pd.DataFrame) -> pd.DataFrame:
    pivot = per_robot.pivot_table(index="robot_id", columns="condition", values="median_passage_time_s").reset_index()
    changed = [column for column in pivot.columns if column not in ("robot_id", "baseline")]
    if "baseline" in pivot and changed:
        pivot["changed_minus_baseline_s"] = pivot[changed[0]] - pivot["baseline"]
        pivot["relative_change"] = pivot["changed_minus_baseline_s"] / pivot["baseline"]
    return pivot


def pair_distance_timeseries(
    trajectories: pd.DataFrame, geometry: AnalysisGeometry, config: MetricConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """At 2 Hz, take the closest robot-center pair inside G1--G3."""

    aligned = trajectories if {"s", "d"}.issubset(trajectories) else world_to_corridor(trajectories, geometry)
    rows: list[dict[str, object]] = []
    for (condition, run_name), run in aligned.groupby(["condition", "run_name"], sort=True):
        times = pd.DatetimeIndex(sorted(run["time"].unique()))
        if len(times) < 2:
            continue
        base_rate = 1 / pd.Series(times).diff().dropna().dt.total_seconds().median()
        stride = max(1, int(round(base_rate / config.pair_distance_rate_hz)))
        for time in times[::stride]:
            group = run[(run["time"] == time) & run.get("valid_pose", True)]
            group = group[group["s"].between(geometry.g1_response_entry_s, geometry.g3_recovery_s)]
            xy = group[["x", "y"]].to_numpy(float)
            if len(xy) < 2:
                continue
            distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
            distances[np.eye(len(xy), dtype=bool)] = np.inf
            rows.append(
                {
                    "condition": condition, "run_name": run_name, "time": time,
                    "n_robots_in_roi": len(xy), "minimum_pair_distance_m": float(distances.min()),
                }
            )
    series = pd.DataFrame(rows)
    if series.empty:
        return series, pd.DataFrame()
    summary = (
        series.groupby("condition", as_index=False)
        .agg(
            pair_distance_q10_m=("minimum_pair_distance_m", lambda x: x.quantile(config.pair_distance_quantile)),
            pair_distance_median_m=("minimum_pair_distance_m", "median"),
            pair_distance_min_m=("minimum_pair_distance_m", "min"),
            covered_timestamps=("minimum_pair_distance_m", "size"),
        )
    )
    return series, summary


def spatial_speed_profile(
    trajectories: pd.DataFrame, geometry: AnalysisGeometry, config: MetricConfig
) -> pd.DataFrame:
    aligned = trajectories if {"s", "v_s"}.issubset(trajectories) else world_to_corridor(trajectories, geometry)
    valid = aligned[aligned.get("valid_kinematics", True)].copy()
    valid = valid[
        valid["s"].between(geometry.g0_approach_entry_s, geometry.g3_recovery_s)
        & (valid["d"].abs() <= geometry.gate_half_width_m)
    ]
    edges = np.arange(geometry.g0_approach_entry_s, geometry.g3_recovery_s + config.speed_bin_width_m, config.speed_bin_width_m)
    valid["s_bin"] = pd.cut(valid["s"], edges, include_lowest=True)
    profile = (
        valid.groupby(["condition", "s_bin"], observed=True)
        .agg(s_mean_m=("s", "mean"), median_v_s_mps=("v_s", "median"),
             q25_v_s_mps=("v_s", lambda x: x.quantile(.25)),
             q75_v_s_mps=("v_s", lambda x: x.quantile(.75)), sample_count=("v_s", "size"))
        .reset_index(drop=False).drop(columns="s_bin")
    )
    return profile


def build_evaluation_tables(
    trajectories: pd.DataFrame,
    passages: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: MetricConfig | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or MetricConfig()
    distribution = route_distribution(passages)
    per_robot, condition = passage_time_tables(passages)
    pair_series, pair_summary = pair_distance_timeseries(trajectories, geometry, config)
    route_wide = distribution.pivot(index="condition", columns="route_label", values="route_proportion").reset_index()
    compact = condition.merge(route_wide, on="condition", how="left").merge(pair_summary, on="condition", how="left")
    compact = compact[
        ["condition", "inner", "outer", "ambiguous", "relative_passage_time",
         "median_excess_path_length_m", "pair_distance_q10_m"]
    ]
    return {
        "compact_evaluation": compact,
        "route_distribution": distribution,
        "robot_passage_time": per_robot,
        "condition_passage_time": condition,
        "paired_robot_passage_time": paired_robot_passage_times(per_robot),
        "pair_distance_timeseries": pair_series,
        "pair_distance_summary": pair_summary,
        "spatial_speed_profile": spatial_speed_profile(trajectories, geometry, config),
        "invalid_passage_summary": passages.groupby(["condition", "valid", "invalid_reason"], dropna=False).size().rename("count").reset_index(),
    }
