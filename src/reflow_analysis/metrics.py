"""Small set of post-action metrics used in the interview analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry, world_to_corridor
from .io import condition_order, sort_by_condition

ROUTE_LABELS = ("inner", "outer", "ambiguous")


@dataclass(frozen=True)
class MetricConfig:
    """Configuration for the nearest-pair spacing proxy."""

    pair_distance_rate_hz: float = 2.0
    pair_distance_quantile: float = 0.10


def route_distribution(passages: pd.DataFrame) -> pd.DataFrame:
    """Count valid inner, outer, and ambiguous passages by condition."""

    valid = passages[passages["valid"] & passages["route_label"].isin(ROUTE_LABELS)]
    conditions = condition_order(passages["condition"])
    index = pd.MultiIndex.from_product(
        [conditions, ROUTE_LABELS], names=["condition", "route_label"]
    )
    counts = (
        valid.groupby(["condition", "route_label"]).size().reindex(index, fill_value=0)
    )
    result = counts.rename("passage_count").reset_index()
    totals = result.groupby("condition")["passage_count"].transform("sum")
    result["route_proportion"] = np.where(
        totals > 0, result["passage_count"] / totals, np.nan
    )
    result["total_valid_passages"] = totals
    return result


def passage_time_tables(passages: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate within robot before making condition-level comparisons.

    Repeated laps from one robot are summarized first. Relative passage time is
    then calculated within each robot and finally summarized across robots, so
    robots with more completed laps do not receive more weight.
    """

    valid = passages[passages["valid"]].copy()
    per_robot = valid.groupby(
        ["condition", "run_name", "robot_id"], as_index=False
    ).agg(
        passage_count=("passage_time_s", "size"),
        median_passage_time_s=("passage_time_s", "median"),
        median_path_length_m=("path_length_m", "median"),
        median_excess_path_length_m=("excess_path_length_m", "median"),
    )
    per_robot = sort_by_condition(per_robot, "robot_id")
    per_robot_condition = per_robot.groupby(
        ["condition", "robot_id"], as_index=False
    ).agg(
        median_passage_time_s=("median_passage_time_s", "median"),
        median_excess_path_length_m=("median_excess_path_length_m", "median"),
    )
    condition = per_robot_condition.groupby("condition", as_index=False).agg(
        median_passage_time_s=("median_passage_time_s", "median"),
        median_excess_path_length_m=("median_excess_path_length_m", "median"),
    )
    passage_counts = per_robot.groupby("condition", as_index=False).agg(
        valid_passages=("passage_count", "sum")
    )
    condition = condition.merge(passage_counts, on="condition", how="left")
    condition = sort_by_condition(condition)

    baseline = per_robot_condition[per_robot_condition["condition"] == "baseline"][
        ["robot_id", "median_passage_time_s"]
    ].rename(columns={"median_passage_time_s": "baseline_passage_time_s"})
    relative_by_condition: dict[str, float] = {}
    for condition_name, group in per_robot_condition.groupby("condition"):
        matched = group.merge(baseline, on="robot_id", how="inner")
        ratios = matched["median_passage_time_s"] / matched["baseline_passage_time_s"]
        relative_by_condition[str(condition_name)] = (
            float(ratios.median()) if len(ratios) else np.nan
        )
    condition["relative_passage_time"] = condition["condition"].map(
        relative_by_condition
    )
    return per_robot, condition


def _pair_distance_summary(
    trajectories: pd.DataFrame, geometry: AnalysisGeometry, config: MetricConfig
) -> pd.DataFrame:
    """Summarize the lower tail of nearest robot-center distance in G1--G3.

    At 2 Hz, the minimum pairwise center distance is computed among valid robots
    inside the response region. The lower quantile is retained as a compact
    crowding proxy; it is not physical clearance because robot radius is absent.
    """

    aligned = (
        trajectories
        if {"s", "d"}.issubset(trajectories)
        else world_to_corridor(trajectories, geometry)
    )
    rows: list[dict[str, object]] = []
    for (condition, run_name), run in aligned.groupby(
        ["condition", "run_name"], sort=True
    ):
        times = pd.DatetimeIndex(sorted(run["time"].unique()))
        if len(times) < 2:
            continue
        base_rate = 1 / pd.Series(times).diff().dropna().dt.total_seconds().median()
        stride = max(1, int(round(base_rate / config.pair_distance_rate_hz)))
        for time in times[::stride]:
            group = run[(run["time"] == time) & run["valid_pose"]]
            group = group[
                group["s"].between(geometry.g1_response_entry_s, geometry.g3_recovery_s)
            ]
            xy = group[["x", "y"]].to_numpy(float)
            if len(xy) < 2:
                continue
            distances = np.linalg.norm(xy[:, None, :] - xy[None, :, :], axis=2)
            distances[np.eye(len(xy), dtype=bool)] = np.inf
            rows.append(
                {
                    "condition": condition,
                    "run_name": run_name,
                    "minimum_pair_distance_m": float(distances.min()),
                }
            )
    samples = pd.DataFrame(rows)
    if samples.empty:
        return pd.DataFrame(columns=["condition", "pair_distance_q10_m"])
    return samples.groupby("condition", as_index=False).agg(
        pair_distance_q10_m=(
            "minimum_pair_distance_m",
            lambda values: values.quantile(config.pair_distance_quantile),
        )
    )


def build_evaluation_tables(
    trajectories: pd.DataFrame,
    passages: pd.DataFrame,
    geometry: AnalysisGeometry,
    config: MetricConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """Build only the compact evaluation and its two supporting tables."""

    config = config or MetricConfig()
    distribution = route_distribution(passages)
    per_robot, condition = passage_time_tables(passages)
    pair_summary = _pair_distance_summary(trajectories, geometry, config)
    route_wide = distribution.pivot(
        index="condition", columns="route_label", values="route_proportion"
    ).reset_index()
    compact = condition.merge(route_wide, on="condition", how="left").merge(
        pair_summary, on="condition", how="left"
    )[
        [
            "condition",
            "inner",
            "outer",
            "ambiguous",
            "relative_passage_time",
            "median_excess_path_length_m",
            "pair_distance_q10_m",
        ]
    ]
    # Route sides are defined by the separator inferred from the changed
    # condition. Reporting baseline inner/outer shares would suggest a split
    # that did not physically exist, so they are intentionally not applicable.
    compact.loc[compact["condition"] == "baseline", list(ROUTE_LABELS)] = np.nan
    invalid = (
        passages.groupby(["condition", "valid", "invalid_reason"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    invalid = sort_by_condition(invalid, "valid", "invalid_reason")
    return {
        "compact_evaluation": compact,
        "route_distribution": distribution,
        "robot_passage_time": per_robot,
        "invalid_passage_summary": invalid,
    }
