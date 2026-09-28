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
    """Conservative settings for the small six-robot demonstration dataset."""

    spatial_step_m: float = 0.02
    occupancy_bin_m: float = 0.04
    occupancy_smoothing_bins: float = 1.2
    component_relative_threshold: float = 0.35
    section_step_m: float = 0.04
    section_half_width_m: float = 0.75
    minimum_section_samples: int = 6
    minimum_cluster_samples: int = 3
    minimum_cluster_separation_m: float = 0.08
    minimum_silhouette: float = 0.35
    null_bootstrap_repeats: int = 100
    null_quantile: float = 0.95
    random_seed: int = 7


@dataclass(frozen=True)
class SplitterInferenceResult:
    geometry: AnalysisGeometry
    section_diagnostics: pd.DataFrame
    crossing_samples: pd.DataFrame
    occupancy_component: pd.DataFrame
    summary: pd.DataFrame


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


def _condition_points(trajectories: pd.DataFrame, step_m: float) -> dict[str, np.ndarray]:
    valid = trajectories[trajectories.get("valid_pose", True)].copy()
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

    if "baseline" not in points or len(points) < 2:
        raise ValueError("Splitter inference needs one baseline and one changed condition")
    changed_name = next(name for name in sorted(points) if name != "baseline")
    all_points = np.vstack([points["baseline"], points[changed_name]])
    margin = 2 * config.occupancy_bin_m
    x_edges = np.arange(all_points[:, 0].min() - margin, all_points[:, 0].max() + margin, config.occupancy_bin_m)
    y_edges = np.arange(all_points[:, 1].min() - margin, all_points[:, 1].max() + margin, config.occupancy_bin_m)
    densities: dict[str, np.ndarray] = {}
    for name in ("baseline", changed_name):
        hist, _, _ = np.histogram2d(points[name][:, 0], points[name][:, 1], bins=(x_edges, y_edges))
        smooth = gaussian_filter(hist, config.occupancy_smoothing_bins)
        densities[name] = smooth / max(smooth.sum(), 1.0)
    difference = densities["baseline"] - densities[changed_name]
    positive_baseline = densities["baseline"][densities["baseline"] > 0]
    density_floor = np.quantile(positive_baseline, 0.50)
    mask = (
        (difference >= config.component_relative_threshold * difference.max())
        & (densities["baseline"] >= density_floor)
    )
    labels, count = label(mask)
    if count == 0:
        raise RuntimeError("No connected trajectory-response region was detected")
    scores = [difference[labels == component].sum() for component in range(1, count + 1)]
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
        [np.average(component["x"], weights=weights), np.average(component["y"], weights=weights)]
    )
    return origin, component


def _local_frame(
    trajectories: pd.DataFrame, origin: np.ndarray, component: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate and orient the local baseline tangent from valid velocities."""

    radius = max(
        0.25,
        float(np.hypot(component["x"] - origin[0], component["y"] - origin[1]).quantile(0.90)),
    )
    baseline = trajectories[trajectories["condition"] == "baseline"].copy()
    near = baseline[
        baseline.get("valid_kinematics", True)
        & (((baseline["x"] - origin[0]) ** 2 + (baseline["y"] - origin[1]) ** 2) <= radius**2)
    ]
    velocities = near[["vx", "vy"]].dropna().to_numpy(dtype=float)
    speeds = np.linalg.norm(velocities, axis=1)
    velocities = velocities[speeds > 0.04]
    if len(velocities) < 10:
        raise RuntimeError("Too few valid baseline velocities in the inferred response region")
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
        vs = group["v_s"].to_numpy(dtype=float) if "v_s" in group else np.full(len(group), np.nan)
        vd = group["v_d"].to_numpy(dtype=float) if "v_d" in group else np.full(len(group), np.nan)
        valid = group.get("valid_pose", pd.Series(True, index=group.index)).to_numpy(dtype=bool)
        for section_index, section_s in enumerate(s_grid):
            crossings = np.where(valid[:-1] & valid[1:] & (s[:-1] < section_s) & (s[1:] >= section_s))[0]
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
                        "v_s": float(vs[index] + alpha * (vs[index + 1] - vs[index])) if np.isfinite(vs[index:index+2]).all() else np.nan,
                        "v_d": float(vd[index] + alpha * (vd[index + 1] - vd[index])) if np.isfinite(vd[index:index+2]).all() else np.nan,
                    }
                )
    return pd.DataFrame(rows)


def _kmeans_1d(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    values = np.asarray(values, dtype=float)
    centers = np.quantile(values, [0.25, 0.75])
    labels_ = np.zeros(len(values), dtype=int)
    for _ in range(50):
        labels_ = np.argmin(np.abs(values[:, None] - centers[None, :]), axis=1)
        if len(np.unique(labels_)) < 2:
            break
        updated = np.array([values[labels_ == index].mean() for index in (0, 1)])
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated
    order = np.argsort(centers)
    remap = np.zeros(2, dtype=int)
    remap[order] = np.arange(2)
    labels_ = remap[labels_]
    centers = centers[order]
    silhouettes = []
    for index, value in enumerate(values):
        same = values[labels_ == labels_[index]]
        other = values[labels_ != labels_[index]]
        a = np.mean(np.abs(same - value)) if len(same) > 1 else 0.0
        b = np.mean(np.abs(other - value)) if len(other) else np.nan
        silhouettes.append((b - a) / max(a, b) if np.isfinite(b) and max(a, b) > 0 else 0.0)
    return centers, labels_, float(np.mean(silhouettes))


def _null_threshold(
    crossings: pd.DataFrame, s_grid: np.ndarray, config: SplitterInferenceConfig
) -> float:
    rng = np.random.default_rng(config.random_seed)
    baseline = crossings[crossings["condition"] == "baseline"]
    null: list[float] = []
    for section_s in s_grid:
        values = baseline.loc[np.isclose(baseline["s"], section_s), "d"].to_numpy(dtype=float)
        if len(values) < 2 * config.minimum_cluster_samples:
            continue
        for _ in range(config.null_bootstrap_repeats):
            shuffled = rng.permutation(values)
            middle = len(shuffled) // 2
            null.append(float(wasserstein_distance(shuffled[:middle], shuffled[middle:])))
    return float(np.quantile(null, config.null_quantile)) if null else 0.05


def _longest_true_interval(mask: np.ndarray, preferred: int) -> tuple[int, int]:
    intervals: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(np.r_[mask, False]):
        if value and start is None:
            start = index
        elif not value and start is not None:
            intervals.append((start, index - 1))
            start = None
    if not intervals:
        return preferred, preferred
    containing = [interval for interval in intervals if interval[0] <= preferred <= interval[1]]
    return containing[0] if containing else max(intervals, key=lambda item: item[1] - item[0])


def infer_effective_splitter(
    trajectories: pd.DataFrame, config: SplitterInferenceConfig | None = None
) -> SplitterInferenceResult:
    """Infer local geometry without using videos or a manually entered splitter pose."""

    config = config or SplitterInferenceConfig()
    points = _condition_points(trajectories, config.spatial_step_m)
    origin, component = _response_component(points, config)
    tangent, normal = _local_frame(trajectories, origin, component)
    provisional = AnalysisGeometry(
        origin_x=float(origin[0]), origin_y=float(origin[1]),
        tangent_x=float(tangent[0]), tangent_y=float(tangent[1]),
        normal_x=float(normal[0]), normal_y=float(normal[1]),
    )
    aligned = world_to_corridor(trajectories, provisional)
    component_s = (
        (component["x"] - origin[0]) * tangent[0]
        + (component["y"] - origin[1]) * tangent[1]
    )
    grid_min = float(component_s.min() - 0.45)
    grid_max = float(component_s.max() + 0.25)
    s_grid = np.arange(grid_min, grid_max + config.section_step_m / 2, config.section_step_m)
    crossings = _interpolate_sections(aligned, s_grid, config.section_half_width_m)
    changed_name = next(name for name in sorted(crossings["condition"].unique()) if name != "baseline")
    threshold = _null_threshold(crossings, s_grid, config)

    rows: list[dict[str, object]] = []
    for section_s in s_grid:
        base = crossings[(crossings["condition"] == "baseline") & np.isclose(crossings["s"], section_s)]["d"].to_numpy(dtype=float)
        changed = crossings[(crossings["condition"] == changed_name) & np.isclose(crossings["s"], section_s)]["d"].to_numpy(dtype=float)
        row: dict[str, object] = {
            "s": section_s, "baseline_n": len(base), "changed_n": len(changed),
            "wasserstein_m": wasserstein_distance(base, changed) if len(base) and len(changed) else np.nan,
            "null_threshold_m": threshold,
            "cluster_0_center_m": np.nan, "cluster_1_center_m": np.nan,
            "separator_midpoint_m": np.nan, "cluster_separation_m": np.nan,
            "silhouette": np.nan, "minimum_cluster_n": 0, "bimodal_valid": False,
        }
        if len(changed) >= config.minimum_section_samples:
            centers, labels_, silhouette = _kmeans_1d(changed)
            counts = np.bincount(labels_, minlength=2)
            separation = centers[1] - centers[0]
            valid = (
                counts.min() >= config.minimum_cluster_samples
                and separation >= config.minimum_cluster_separation_m
                and silhouette >= config.minimum_silhouette
            )
            row.update(
                {
                    "cluster_0_center_m": centers[0], "cluster_1_center_m": centers[1],
                    "separator_midpoint_m": centers.mean(), "cluster_separation_m": separation,
                    "silhouette": silhouette, "minimum_cluster_n": int(counts.min()),
                    "bimodal_valid": bool(valid),
                }
            )
        rows.append(row)
    diagnostics = pd.DataFrame(rows)
    baseline_support = diagnostics["baseline_n"].to_numpy() >= 0.75 * diagnostics.loc[diagnostics["baseline_n"] > 0, "baseline_n"].median()
    changed_support = diagnostics["changed_n"].to_numpy() >= 0.75 * diagnostics.loc[diagnostics["changed_n"] > 0, "changed_n"].median()
    supported = baseline_support & changed_support
    response = (diagnostics["wasserstein_m"].fillna(0).to_numpy() > threshold) & supported
    max_index = int(diagnostics["wasserstein_m"].fillna(-np.inf).to_numpy().argmax())
    response_start, response_end = _longest_true_interval(response, max_index)
    core_mask = response & diagnostics["bimodal_valid"].to_numpy(dtype=bool)
    core_start, core_end = _longest_true_interval(core_mask, max_index)
    if core_start == core_end and not core_mask[core_start]:
        valid_indices = np.flatnonzero(diagnostics["bimodal_valid"].to_numpy(dtype=bool))
        if not len(valid_indices):
            raise RuntimeError("The changed trajectories do not support a stable two-route separator")
        core_start, core_end = int(valid_indices.min()), int(valid_indices.max())

    core = diagnostics.iloc[core_start : core_end + 1].dropna(subset=["separator_midpoint_m"])
    coefficients = np.polyfit(core["s"], core["separator_midpoint_m"], deg=1)
    for _ in range(3):
        residual = core["separator_midpoint_m"] - np.polyval(coefficients, core["s"])
        mad = np.median(np.abs(residual - np.median(residual)))
        keep = np.abs(residual) <= max(0.02, 3.0 * 1.4826 * mad)
        if keep.sum() < 2:
            break
        coefficients = np.polyfit(core.loc[keep, "s"], core.loc[keep, "separator_midpoint_m"], deg=1)
        core = core.loc[keep]
    slope, intercept = map(float, coefficients)
    fit_residual = core["separator_midpoint_m"] - (intercept + slope * core["s"])
    median_separation = float(core["cluster_separation_m"].median())
    ambiguity = max(0.02, 0.10 * median_separation, 2.0 * float(np.sqrt(np.mean(fit_residual**2))))

    upstream = diagnostics.iloc[: response_start + 1]
    stable_candidates: list[float] = []
    for section_s in upstream["s"]:
        section = crossings[(crossings["condition"] == "baseline") & np.isclose(crossings["s"], section_s)]
        denominator = section["v_s"].abs() + section["v_d"].abs()
        aligned_fraction = ((section["v_s"] > 0.05) & ((section["v_s"].abs() / denominator) >= 0.70)).mean()
        if len(section) >= config.minimum_section_samples and aligned_fraction >= 0.70:
            stable_candidates.append(float(section_s))
    g0 = stable_candidates[0] if stable_candidates else float(diagnostics["s"].iloc[0])
    g1 = float(diagnostics["s"].iloc[response_start])
    g3 = float(diagnostics["s"].iloc[response_end])
    c0, c1 = float(diagnostics["s"].iloc[core_start]), float(diagnostics["s"].iloc[core_end])
    g2 = 0.5 * (c0 + c1)
    if not g0 < g1:
        g0 = float(diagnostics["s"].iloc[max(0, response_start - 3)])
    if g3 < c1:
        g3 = c1
    lateral = crossings[crossings["s"].between(g0, g3)]["d"].abs()
    gate_half_width = max(config.section_half_width_m, float(lateral.quantile(0.99) + 0.05))
    geometry = AnalysisGeometry(
        origin_x=float(origin[0]), origin_y=float(origin[1]),
        tangent_x=float(tangent[0]), tangent_y=float(tangent[1]),
        normal_x=float(normal[0]), normal_y=float(normal[1]),
        g0_approach_entry_s=g0, g1_response_entry_s=g1, g2_route_s=g2,
        g3_recovery_s=g3, core_start_s=c0, core_end_s=c1,
        splitter_intercept_m=intercept, splitter_slope=slope,
        ambiguous_half_width_m=ambiguity, gate_half_width_m=gate_half_width,
    )
    diagnostics["section_support_valid"] = supported
    diagnostics["response_valid"] = response
    diagnostics["core_valid"] = core_mask
    supported_after = supported[response_end + 1 :]
    below_after = diagnostics["wasserstein_m"].fillna(np.inf).to_numpy()[response_end + 1 :] <= threshold
    recovery_observed = bool(np.any(supported_after & below_after))
    summary = pd.DataFrame(
        [
            {"quantity": "changed_condition", "value": changed_name, "unit": ""},
            {"quantity": "occupancy_origin_x", "value": origin[0], "unit": "m"},
            {"quantity": "occupancy_origin_y", "value": origin[1], "unit": "m"},
            {"quantity": "flow_tangent_x", "value": tangent[0], "unit": ""},
            {"quantity": "flow_tangent_y", "value": tangent[1], "unit": ""},
            {"quantity": "W1_null_threshold", "value": threshold, "unit": "m"},
            {"quantity": "median_cluster_separation", "value": median_separation, "unit": "m"},
            {"quantity": "ambiguity_half_width", "value": ambiguity, "unit": "m"},
            {"quantity": "splitter_intercept", "value": intercept, "unit": "m"},
            {"quantity": "splitter_slope", "value": slope, "unit": "m/m"},
            {"quantity": "G0_approach_entry", "value": g0, "unit": "m"},
            {"quantity": "G1_response_onset", "value": g1, "unit": "m"},
            {"quantity": "G2_route_section", "value": g2, "unit": "m"},
            {"quantity": "G3_analysis_exit", "value": g3, "unit": "m"},
            {"quantity": "recovery_observed_before_G3", "value": recovery_observed, "unit": "boolean"},
        ]
    )
    return SplitterInferenceResult(geometry, diagnostics, crossings, component, summary)
