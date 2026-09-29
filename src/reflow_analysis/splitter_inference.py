"""Infer a local flow frame and an effective splitter from trajectory changes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label
from scipy.stats import wasserstein_distance

from .geometry import AnalysisGeometry, world_to_corridor


@dataclass(frozen=True)
class SplitterInferenceConfig:
    """Thresholds for locating one response region and two stable route bands."""

    spatial_step_m: float = 0.02
    occupancy_bin_m: float = 0.04
    occupancy_smoothing_bins: float = 1.2
    component_relative_threshold: float = 0.35
    section_step_m: float = 0.04
    section_half_width_m: float = 0.75
    minimum_cluster_samples: int = 3
    minimum_core_separation_m: float = 0.25
    minimum_core_silhouette: float = 0.55
    maximum_core_gap_fraction: float = 0.20
    section_upstream_margin_m: float = 0.45
    section_downstream_margin_m: float = 0.25
    null_bootstrap_repeats: int = 100
    null_quantile: float = 0.95
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


def _spatially_resample(group: pd.DataFrame, step_m: float) -> np.ndarray:
    """Sample a trajectory by arc length so pauses do not dominate occupancy."""

    xy = group.sort_values("time")[["x", "y"]].dropna().to_numpy(dtype=float)
    if len(xy) < 2:
        return np.empty((0, 2))
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    unique_arc, index = np.unique(arc, return_index=True)
    xy = xy[index]
    if unique_arc[-1] < step_m:
        return xy[[0]]
    query = np.arange(0.0, unique_arc[-1], step_m)
    return np.column_stack(
        [np.interp(query, unique_arc, xy[:, 0]), np.interp(query, unique_arc, xy[:, 1])]
    )


def _condition_points(
    trajectories: pd.DataFrame, step_m: float
) -> dict[str, np.ndarray]:
    """Create equal-arc-length point clouds for each experimental condition."""

    valid = trajectories[trajectories["valid_pose"]].copy()
    output: dict[str, list[np.ndarray]] = {}
    for keys, group in valid.groupby(["condition", "run_name", "robot_id"], sort=True):
        points = _spatially_resample(group, step_m)
        if len(points):
            output.setdefault(str(keys[0]), []).append(points)
    return {condition: np.vstack(groups) for condition, groups in output.items()}


def _response_component(
    points: dict[str, np.ndarray], config: SplitterInferenceConfig
) -> tuple[np.ndarray, pd.DataFrame]:
    """Locate the strongest connected baseline-depletion component in 2-D."""

    if "baseline" not in points:
        raise ValueError("Splitter inference needs a baseline run")
    changed_name = _changed_condition(points)
    all_points = np.vstack([points["baseline"], points[changed_name]])
    margin = 2 * config.occupancy_bin_m
    x_edges = np.arange(
        all_points[:, 0].min() - margin,
        all_points[:, 0].max() + margin,
        config.occupancy_bin_m,
    )
    y_edges = np.arange(
        all_points[:, 1].min() - margin,
        all_points[:, 1].max() + margin,
        config.occupancy_bin_m,
    )
    densities: dict[str, np.ndarray] = {}
    for name in ("baseline", changed_name):
        hist, _, _ = np.histogram2d(
            points[name][:, 0], points[name][:, 1], bins=(x_edges, y_edges)
        )
        smooth = gaussian_filter(hist, config.occupancy_smoothing_bins)
        densities[name] = smooth / max(smooth.sum(), 1.0)
    difference = densities["baseline"] - densities[changed_name]
    positive_baseline = densities["baseline"][densities["baseline"] > 0]
    density_floor = np.quantile(positive_baseline, 0.50)
    mask = (difference >= config.component_relative_threshold * difference.max()) & (
        densities["baseline"] >= density_floor
    )
    labels, count = label(mask)
    if count == 0:
        raise RuntimeError("No connected trajectory-response region was detected")
    scores = [
        difference[labels == component].sum() for component in range(1, count + 1)
    ]
    selected = int(np.argmax(scores)) + 1
    ii, jj = np.where(labels == selected)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    weights = difference[ii, jj]
    component = pd.DataFrame(
        {
            "x": x_centers[ii],
            "y": y_centers[jj],
            "occupancy_difference": weights,
            "baseline_density": densities["baseline"][ii, jj],
            "changed_density": densities[changed_name][ii, jj],
        }
    )
    origin = np.array(
        [
            np.average(component["x"], weights=weights),
            np.average(component["y"], weights=weights),
        ]
    )
    return origin, component


def _local_frame(
    trajectories: pd.DataFrame, origin: np.ndarray, component: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and orient the local baseline tangent from valid velocities."""

    radius = max(
        0.25,
        float(
            np.hypot(component["x"] - origin[0], component["y"] - origin[1]).quantile(
                0.90
            )
        ),
    )
    baseline = trajectories[trajectories["condition"] == "baseline"].copy()
    near = baseline[
        baseline["valid_kinematics"]
        & (
            ((baseline["x"] - origin[0]) ** 2 + (baseline["y"] - origin[1]) ** 2)
            <= radius**2
        )
    ]
    velocities = near[["vx", "vy"]].dropna().to_numpy(dtype=float)
    speeds = np.linalg.norm(velocities, axis=1)
    velocities = velocities[speeds > 0.04]
    if len(velocities) < 10:
        raise RuntimeError(
            "Too few valid baseline velocities in the inferred response region"
        )
    unit = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)
    tangent = np.median(unit, axis=0)
    tangent /= np.linalg.norm(tangent)
    if np.dot(tangent, np.mean(velocities, axis=0)) < 0:
        tangent *= -1
    normal = np.array([-tangent[1], tangent[0]])
    return tangent, normal


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

    K=2 is used as a transparent baseline because the physical hypothesis is
    specifically two routes around one splitter. In one dimension, every
    K-means solution is a split of the sorted samples. Enumerating those splits
    is deterministic and avoids the local minima of Lloyd iteration.
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


def _null_threshold(
    crossings: pd.DataFrame, s_grid: np.ndarray, config: SplitterInferenceConfig
) -> float:
    """Estimate a baseline-only Wasserstein threshold by repeated random splits.

    This empirical null avoids declaring ordinary baseline variation to be a
    barrier response and is easier to explain than a fitted parametric model.
    """

    rng = np.random.default_rng(config.random_seed)
    baseline = crossings[crossings["condition"] == "baseline"]
    null: list[float] = []
    for section_s in s_grid:
        values = baseline.loc[np.isclose(baseline["s"], section_s), "d"].to_numpy(
            dtype=float
        )
        if len(values) < 2 * config.minimum_cluster_samples:
            continue
        for _ in range(config.null_bootstrap_repeats):
            shuffled = rng.permutation(values)
            middle = len(shuffled) // 2
            null.append(
                float(wasserstein_distance(shuffled[:middle], shuffled[middle:]))
            )
    return float(np.quantile(null, config.null_quantile)) if null else 0.05


def _interval_near_peak(mask: np.ndarray, peak_index: int) -> tuple[int, int]:
    """Choose the true interval containing the response peak, or the longest one."""

    intervals: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if not intervals:
        return peak_index, peak_index
    containing = [
        interval for interval in intervals if interval[0] <= peak_index <= interval[1]
    ]
    return (
        containing[0]
        if containing
        else max(intervals, key=lambda item: item[1] - item[0])
    )


def infer_effective_splitter(
    trajectories: pd.DataFrame, config: SplitterInferenceConfig | None = None
) -> SplitterInferenceResult:
    """Infer local geometry without using videos or a manually entered splitter pose."""

    config = config or SplitterInferenceConfig()
    points = _condition_points(trajectories, config.spatial_step_m)
    origin, component = _response_component(points, config)
    tangent, normal = _local_frame(trajectories, origin, component)
    provisional = AnalysisGeometry(
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        tangent_x=float(tangent[0]),
        tangent_y=float(tangent[1]),
        normal_x=float(normal[0]),
        normal_y=float(normal[1]),
    )
    aligned = world_to_corridor(trajectories, provisional)
    component_s = (component["x"] - origin[0]) * tangent[0] + (
        component["y"] - origin[1]
    ) * tangent[1]
    grid_min = float(component_s.min() - config.section_upstream_margin_m)
    grid_max = float(component_s.max() + config.section_downstream_margin_m)
    s_grid = np.arange(
        grid_min, grid_max + config.section_step_m / 2, config.section_step_m
    )
    crossings = _interpolate_sections(aligned, s_grid, config.section_half_width_m)
    changed_name = _changed_condition(crossings["condition"])
    threshold = _null_threshold(crossings, s_grid, config)

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
            "null_threshold_m": threshold,
            "cluster_0_center_m": np.nan,
            "cluster_1_center_m": np.nan,
            "separator_midpoint_m": np.nan,
            "cluster_separation_m": np.nan,
            "silhouette": np.nan,
            "minimum_cluster_n": 0,
            "baseline_median_m": np.nan,
            "gap_fraction": np.nan,
            "stable_core_valid": False,
        }
        if len(changed) >= 2 * config.minimum_cluster_samples:
            centers, labels_, silhouette = _kmeans_1d(changed)
            counts = np.bincount(labels_, minlength=2)
            separation = centers[1] - centers[0]
            baseline_median = float(np.median(base)) if len(base) else np.nan
            midpoint = float(centers.mean())
            gap_fraction = float(
                np.mean(np.abs(changed - midpoint) < 0.15 * separation)
            )
            # The physical hypothesis is two supported route bands with the
            # natural baseline path lying in the depleted gap between them.
            stable_core = (
                counts.min() >= config.minimum_cluster_samples
                and separation >= config.minimum_core_separation_m
                and silhouette >= config.minimum_core_silhouette
                and gap_fraction <= config.maximum_core_gap_fraction
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
                    "gap_fraction": gap_fraction,
                    "stable_core_valid": bool(stable_core),
                }
            )
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    baseline_support = (
        diagnostics["baseline_n"].to_numpy()
        >= 0.75 * diagnostics.loc[diagnostics["baseline_n"] > 0, "baseline_n"].median()
    )
    changed_support = (
        diagnostics["changed_n"].to_numpy()
        >= 0.75 * diagnostics.loc[diagnostics["changed_n"] > 0, "changed_n"].median()
    )
    supported = baseline_support & changed_support
    response = (
        diagnostics["wasserstein_m"].fillna(0).to_numpy() > threshold
    ) & supported
    max_index = int(diagnostics["wasserstein_m"].fillna(-np.inf).to_numpy().argmax())
    response_start, response_end = _interval_near_peak(response, max_index)
    core_mask = response & diagnostics["stable_core_valid"].to_numpy(dtype=bool)
    core_start, core_end = _interval_near_peak(core_mask, max_index)
    if core_start == core_end and not core_mask[core_start]:
        raise RuntimeError(
            "The changed trajectories do not support a stable two-route separator"
        )

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

    upstream = diagnostics.iloc[: response_start + 1]
    supported_upstream = upstream.loc[supported[: response_start + 1], "s"]
    g0 = float(supported_upstream.iloc[0])
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
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        tangent_x=float(tangent[0]),
        tangent_y=float(tangent[1]),
        normal_x=float(normal[0]),
        normal_y=float(normal[1]),
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
    )
    diagnostics["section_support_valid"] = supported
    diagnostics["response_valid"] = response
    diagnostics["core_valid"] = selected_core_mask
    return SplitterInferenceResult(geometry, diagnostics, crossings)
