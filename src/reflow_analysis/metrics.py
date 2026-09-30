"""Small set of *post-action* metrics used in the interview analysis.

These describe what happened after an intervention and are not inputs to a
causal geometry-selection policy. The headline passage time is the arithmetic
mean over valid passage rows, not the average of six robot-level averages.
The per-robot view remains a useful check for uneven behaviour across robots.
"""

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
    """Report per-robot medians, but compare conditions by pooled passage mean.

    The main question here is the mean duration of a valid traversal in each
    observed run. Therefore every valid passage has one vote in that estimate;
    the changed/baseline ratio compares those two pooled means. Per-robot
    medians remain visible as a heterogeneity check. Neither the passages nor
    the two runs are independent experimental replications.
    """

    # Only completed, valid G0-to-G3 traversals contribute. An invalid lap is
    # still visible in passages.csv together with its reason, but cannot be
    # assigned a comparable local traversal duration.
    valid = passages[passages["valid"]].copy()
    # Supporting table: summarize repeat laps *within* each robot. These
    # medians drive the paired-robot figure, not the condition comparison.
    per_robot = valid.groupby(
        ["condition", "run_name", "robot_id"], as_index=False
    ).agg(
        passage_count=("passage_time_s", "size"),
        median_passage_time_s=("passage_time_s", "median"),
        median_path_length_m=("path_length_m", "median"),
        median_excess_path_length_m=("excess_path_length_m", "median"),
    )
    per_robot = sort_by_condition(per_robot, "robot_id")
    # Main comparison: every valid passage is one observation. This answers
    # the directly interpretable question, "How long was an observed local
    # traversal on average in this condition?" It does not estimate a
    # population causal effect from these two nonreplicated runs.
    condition = valid.groupby("condition", as_index=False).agg(
        mean_passage_time_s=("passage_time_s", "mean"),
        valid_passages=("passage_time_s", "size"),
    )
    # Excess distance is deliberately a median: occasional tracking/path
    # outliers should not dominate this separate geometric descriptor.
    excess = valid.groupby("condition", as_index=False).agg(
        median_excess_path_length_m=("excess_path_length_m", "median")
    )
    condition = condition.merge(excess, on="condition", how="left")
    condition = sort_by_condition(condition)
    baseline = condition.loc[
        condition["condition"] == "baseline", "mean_passage_time_s"
    ]
    if len(baseline) != 1:
        raise ValueError("Passage-time comparison needs one baseline condition")
    condition["relative_passage_time"] = (
        condition["mean_passage_time_s"] / float(baseline.iloc[0])
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
        "passage_time_summary": condition,
        "invalid_passage_summary": invalid,
    }
