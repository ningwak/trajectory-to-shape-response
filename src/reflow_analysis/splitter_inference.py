"""Infer an effective route separator on the horizontal upper corridor leg.

The JSON supplies a nominal rectangular route, but not the physical barrier
pose. The upper reference edge defines our analysis frame; the separator is
inferred from where the recorded passages actually travel.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

from .geometry import AnalysisGeometry, world_to_corridor


@dataclass(frozen=True)
class SplitterInferenceConfig:
    """Transparent cross-section settings for this near-horizontal experiment."""

    section_step_m: float = 0.04
    section_half_width_m: float = 0.75
    edge_inset_fraction: float = 0.10
    minimum_cluster_samples: int = 3
    minimum_core_separation_m: float = 0.25
    minimum_supported_fraction: float = 0.75
    response_fraction_of_peak: float = 0.20
    baseline_split_repeats: int = 100
    baseline_reference_quantile: float = 0.95
    random_seed: int = 7


@dataclass(frozen=True)
class SplitterInferenceResult:
    """Inferred geometry and the cross-sections that support it."""

    geometry: AnalysisGeometry
    section_diagnostics: pd.DataFrame
    crossing_samples: pd.DataFrame


def _changed_condition(values) -> str:
    """Return the single non-baseline condition required by this comparison."""

    changed = sorted(set(values) - {"baseline"})
    if len(changed) != 1:
        raise ValueError("Splitter inference expects one baseline and one changed run")
    return changed[0]


def _upper_reference_edge(
    rectangle_corners: tuple[tuple[float, float], ...],
) -> tuple[float, float, float]:
    """Select the upper nominal route edge, not the physical splitter.

    This experiment has one visibly affected, near-horizontal upper leg. The
    supplied rectangle defines its reproducible x range and y origin without
    using the accompanying video or manually entering a barrier position.
    """

    corners = np.asarray(rectangle_corners, dtype=float)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        raise ValueError("Expected four finite nominal-route corners")
    top = corners[np.argsort(corners[:, 1])[-2:]]
    if abs(top[0, 1] - top[1, 1]) > 0.05:
        raise ValueError("This analysis requires a nearly horizontal upper edge")
    x_left, x_right = sorted(top[:, 0])
    if x_right <= x_left:
        raise ValueError("The upper edge must have positive length")
    return float(x_left), float(x_right), float(top[:, 1].mean())


def _interpolate_sections(
    aligned: pd.DataFrame, s_grid: np.ndarray, lateral_limit: float
) -> pd.DataFrame:
    """Interpolate forward crossings; each row is one passage-level section sample."""

    rows: list[dict[str, object]] = []
    keys = ["condition", "run_name", "robot_id"]
    for (condition, run_name, robot_id), group in aligned.groupby(keys, sort=True):
        group = group.sort_values("time").reset_index(drop=True)
        s = group["s"].to_numpy(dtype=float)
        d = group["d"].to_numpy(dtype=float)
        times = pd.DatetimeIndex(group["time"])
        vs = (
            group["v_s"].to_numpy(dtype=float)
            if "v_s" in group
            else np.full(len(group), np.nan)
        )
        vd = (
            group["v_d"].to_numpy(dtype=float)
            if "v_d" in group
            else np.full(len(group), np.nan)
        )
        valid = group["valid_pose"].to_numpy(dtype=bool)
        for section_index, section_s in enumerate(s_grid):
            crossings = np.where(
                valid[:-1] & valid[1:] & (s[:-1] < section_s) & (s[1:] >= section_s)
            )[0]
            passage_number = 0
            for index in crossings:
                denominator = s[index + 1] - s[index]
                if denominator <= 0:
                    continue
                alpha = (section_s - s[index]) / denominator
                d_cross = d[index] + alpha * (d[index + 1] - d[index])
                if not np.isfinite(d_cross) or abs(d_cross) > lateral_limit:
                    continue
                passage_number += 1
                time = times[index] + (times[index + 1] - times[index]) * alpha
                rows.append(
                    {
                        "condition": condition,
                        "run_name": run_name,
                        "robot_id": int(robot_id),
                        "section_index": section_index,
                        "s": float(section_s),
                        "passage_number": passage_number,
                        "time": time,
                        "d": float(d_cross),
                        "v_s": (
                            float(vs[index] + alpha * (vs[index + 1] - vs[index]))
                            if np.isfinite(vs[index : index + 2]).all()
                            else np.nan
                        ),
                        "v_d": (
                            float(vd[index] + alpha * (vd[index + 1] - vd[index]))
                            if np.isfinite(vd[index : index + 2]).all()
                            else np.nan
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _kmeans_1d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit the globally optimal two-cluster K-means split in one dimension.

    K=2 describes the two observed route bands. In one dimension, every
    optimum partitions the sorted samples at one split point. We enumerate
    these points to avoid initialization; silhouette remains a diagnostic,
    not another threshold tuned on this small dataset.
    """

    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("1-D K-means needs at least two finite values")

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    best_sse = np.inf
    best_split = 1
    best_centers = np.array([sorted_values[0], sorted_values[1]])
    for split in range(1, len(sorted_values)):
        left, right = sorted_values[:split], sorted_values[split:]
        centers = np.array([left.mean(), right.mean()])
        sse = float(
            np.sum((left - centers[0]) ** 2) + np.sum((right - centers[1]) ** 2)
        )
        if sse < best_sse:
            best_sse = sse
            best_split = split
            best_centers = centers

    sorted_labels = np.r_[
        np.zeros(best_split, dtype=int),
        np.ones(len(sorted_values) - best_split, dtype=int),
    ]
    labels_ = np.empty(len(values), dtype=int)
    labels_[order] = sorted_labels
    centers = best_centers
    silhouettes = []
    for index, value in enumerate(values):
        same_indices = np.flatnonzero(labels_ == labels_[index])
        same = values[same_indices[same_indices != index]]
        other = values[labels_ != labels_[index]]
        if not len(same):
            silhouettes.append(0.0)
            continue
        a = np.mean(np.abs(same - value))
        b = np.mean(np.abs(other - value)) if len(other) else np.nan
        silhouettes.append(
            (b - a) / max(a, b) if np.isfinite(b) and max(a, b) > 0 else 0.0
        )
    return centers, labels_, float(np.mean(silhouettes))


def _baseline_variability_reference(
    crossings: pd.DataFrame, s_grid: np.ndarray, config: SplitterInferenceConfig
) -> float:
    """Estimate a baseline-only Wasserstein *reference* by random splits.

    At each section, randomly divide the baseline passage crossings into two
    halves and compare them. Pool those distances over sections and report a
    high quantile on the plot. Repeated laps/adjacent sections are correlated:
    this is NOT a formal test and does NOT determine gates or the response core.
    """

    rng = np.random.default_rng(config.random_seed)
    baseline = crossings[crossings["condition"] == "baseline"]
    distances: list[float] = []
    for section_s in s_grid:
        values = baseline.loc[np.isclose(baseline["s"], section_s), "d"].to_numpy(
            dtype=float
        )
        if len(values) < 2 * config.minimum_cluster_samples:
            continue
        for _ in range(config.baseline_split_repeats):
            shuffled = rng.permutation(values)
            middle = len(shuffled) // 2
            distances.append(
                float(wasserstein_distance(shuffled[:middle], shuffled[middle:]))
            )
    return (
        float(np.quantile(distances, config.baseline_reference_quantile))
        if distances
        else np.nan
    )


def _interval_near_peak(mask: np.ndarray, peak_index: int) -> tuple[int, int]:
    """Choose a contiguous accepted interval containing the profile peak."""

    intervals: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if not intervals:
        raise RuntimeError("No supported response interval was found")
    containing = [
        interval for interval in intervals if interval[0] <= peak_index <= interval[1]
    ]
    return (
        containing[0]
        if containing
        else max(intervals, key=lambda item: item[1] - item[0])
    )


def infer_effective_splitter(
    trajectories: pd.DataFrame,
    rectangle_corners: tuple[tuple[float, float], ...],
    config: SplitterInferenceConfig | None = None,
) -> SplitterInferenceResult:
    """Infer observed routes on the nominal upper edge, not barrier pose.

    Here s=x-x_midpoint and d=y-y_upper_edge. Gate placement uses the sustained
    Wasserstein response relative to its own peak; the baseline-only random
    split is displayed for context but never used as a detection threshold.
    """

    config = config or SplitterInferenceConfig()
    x_left, x_right, y_top = _upper_reference_edge(rectangle_corners)
    origin_x = 0.5 * (x_left + x_right)
    provisional = AnalysisGeometry(origin_x=origin_x, origin_y=y_top)
    aligned = world_to_corridor(trajectories, provisional)
    # The JSON vertices are turns, not useful passage gates. Keep the analysis
    # on the straight interior of the top edge to avoid corner recrossings.
    if not 0 <= config.edge_inset_fraction < 0.5:
        raise ValueError("edge_inset_fraction must be in [0, 0.5)")
    inset = config.edge_inset_fraction * (x_right - x_left)
    s_grid = np.arange(
        x_left + inset - origin_x,
        x_right - inset - origin_x + config.section_step_m / 2,
        config.section_step_m,
    )
    crossings = _interpolate_sections(aligned, s_grid, config.section_half_width_m)
    changed_name = _changed_condition(crossings["condition"])
    reference = _baseline_variability_reference(crossings, s_grid, config)

    rows: list[dict[str, object]] = []
    for section_s in s_grid:
        base = crossings[
            (crossings["condition"] == "baseline")
            & np.isclose(crossings["s"], section_s)
        ]["d"].to_numpy(dtype=float)
        changed = crossings[
            (crossings["condition"] == changed_name)
            & np.isclose(crossings["s"], section_s)
        ]["d"].to_numpy(dtype=float)
        row: dict[str, object] = {
            "s": section_s,
            "baseline_n": len(base),
            "changed_n": len(changed),
            "wasserstein_m": (
                wasserstein_distance(base, changed)
                if len(base) and len(changed)
                else np.nan
            ),
            "baseline_reference_m": reference,
            "cluster_0_center_m": np.nan,
            "cluster_1_center_m": np.nan,
            "separator_midpoint_m": np.nan,
            "cluster_separation_m": np.nan,
            "silhouette": np.nan,
            "minimum_cluster_n": 0,
            "baseline_median_m": np.nan,
            "stable_core_valid": False,
        }
        if len(changed) >= 2 * config.minimum_cluster_samples:
            centers, labels_, silhouette = _kmeans_1d(changed)
            counts = np.bincount(labels_, minlength=2)
            separation = centers[1] - centers[0]
            baseline_median = float(np.median(base)) if len(base) else np.nan
            midpoint = float(centers.mean())
            # K=2 always returns two centers, even for one broad path. Require
            # two populated, separated sides around the natural baseline path.
            stable_core = (
                counts.min() >= config.minimum_cluster_samples
                and separation >= config.minimum_core_separation_m
                and centers[0] < baseline_median < centers[1]
            )
            row.update(
                {
                    "cluster_0_center_m": centers[0],
                    "cluster_1_center_m": centers[1],
                    "separator_midpoint_m": centers.mean(),
                    "cluster_separation_m": separation,
                    "silhouette": silhouette,
                    "minimum_cluster_n": int(counts.min()),
                    "baseline_median_m": baseline_median,
                    "stable_core_valid": bool(stable_core),
                }
            )
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    baseline_support = (
        diagnostics["baseline_n"].to_numpy()
        >= config.minimum_supported_fraction
        * diagnostics.loc[diagnostics["baseline_n"] > 0, "baseline_n"].median()
    )
    changed_support = (
        diagnostics["changed_n"].to_numpy()
        >= config.minimum_supported_fraction
        * diagnostics.loc[diagnostics["changed_n"] > 0, "changed_n"].median()
    )
    supported = baseline_support & changed_support
    profile = diagnostics["wasserstein_m"].fillna(0).to_numpy()
    max_index = int(np.argmax(np.where(supported, profile, -np.inf)))
    peak = profile[max_index]
    if not np.isfinite(peak) or peak <= 0:
        raise RuntimeError("No supported trajectory response was found")
    # A simple peak-relative interval describes where the observed difference
    # persists. It does not claim a statistically exact intervention onset.
    response = supported & (profile >= config.response_fraction_of_peak * peak)
    response_start, response_end = _interval_near_peak(response, max_index)
    core_mask = response & diagnostics["stable_core_valid"].to_numpy(dtype=bool)
    core_start, core_end = _interval_near_peak(core_mask, max_index)
    selected_core_mask = np.zeros(len(diagnostics), dtype=bool)
    selected_core_mask[core_start : core_end + 1] = True

    core = diagnostics.iloc[core_start : core_end + 1].dropna(
        subset=["separator_midpoint_m"]
    )
    if len(core) < 2:
        raise RuntimeError(
            "At least two stable sections are required to fit a separator"
        )
    coefficients = np.polyfit(core["s"], core["separator_midpoint_m"], deg=1)
    slope, intercept = map(float, coefficients)
    median_separation = float(core["cluster_separation_m"].median())
    ambiguity = max(0.02, 0.10 * median_separation)

    supported_upstream = np.flatnonzero(
        supported & (np.arange(len(s_grid)) < response_start)
    )
    if not len(supported_upstream):
        raise RuntimeError("No supported approach section before the response")
    g0 = float(s_grid[supported_upstream[0]])
    g1 = float(diagnostics["s"].iloc[response_start])
    g3 = float(diagnostics["s"].iloc[response_end])
    c0, c1 = float(diagnostics["s"].iloc[core_start]), float(
        diagnostics["s"].iloc[core_end]
    )
    g2 = 0.5 * (c0 + c1)
    if not g0 < g1:
        raise RuntimeError(
            "No supported approach section was found before the response"
        )
    geometry = AnalysisGeometry(
        origin_x=origin_x,
        origin_y=y_top,
        g0_approach_entry_s=g0,
        g1_response_entry_s=g1,
        g2_route_s=g2,
        g3_recovery_s=g3,
        core_start_s=c0,
        core_end_s=c1,
        splitter_intercept_m=intercept,
        splitter_slope=slope,
        ambiguous_half_width_m=ambiguity,
        gate_half_width_m=config.section_half_width_m,
        inference_method="horizontal JSON route edge; trajectory-derived separator",
    )
    diagnostics["section_support_valid"] = supported
    diagnostics["response_valid"] = response
    diagnostics["core_valid"] = selected_core_mask
    return SplitterInferenceResult(geometry, diagnostics, crossings)
