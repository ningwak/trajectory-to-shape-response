"""Infer a response-centred local frame and effective trajectory separator.

This module contains the main data-derived method.  The JSON files do not give
the physical barrier pose, so the code asks a narrower, observable question:

    Where does the barrier condition remove occupancy from the natural path,
    where do two stable route bands appear, and what line separates them?

The resulting line is an *effective separator of trajectories*.  It is useful
for route classification and response geometry, but it is not presented as a
computer-vision reconstruction of the barrier body.

Algorithm outline:

1. resample paths by arc length;
2. locate the strongest baseline-occupancy depletion under the barrier;
3. orient a local frame using nearby baseline velocity;
4. interpolate passage-level lateral crossings on transverse sections;
5. compare conditions with 1-D Wasserstein distance;
6. identify two stable barrier route bands using deterministic K=2 K-means;
7. fit a line through their section-wise midpoints and derive four gates.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, label
from scipy.stats import wasserstein_distance

from .geometry import AnalysisGeometry, world_to_corridor


@dataclass(frozen=True)
class SplitterInferenceConfig:
    """Visible thresholds for response localization and stable route evidence.

    Spatial and occupancy settings control the coarse 2-D localization stage.
    Section settings control the later 1-D lateral analysis.  Core thresholds
    deliberately express simple physical/statistical requirements rather than
    a highly tuned classifier for this one small dataset.
    """

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
    """Final geometry plus the evidence needed to inspect how it was obtained."""

    geometry: AnalysisGeometry
    section_diagnostics: pd.DataFrame
    crossing_samples: pd.DataFrame


def _changed_condition(values) -> str:
    """Return the single non-baseline condition required by this method.

    The current experiment supplies one baseline and one barrier geometry.  A
    multi-geometry study should call/extend the comparison explicitly rather
    than silently selecting the first changed condition.
    """

    changed = sorted(set(values) - {"baseline"})
    if len(changed) != 1:
        raise ValueError("Splitter inference expects one baseline and one changed run")
    return changed[0]


def _spatially_resample(group: pd.DataFrame, step_m: float) -> np.ndarray:
    """Interpolate one trajectory at approximately equal arc-length intervals.

    Occupancy built from raw timestamps would assign more mass to slow motion
    and sample-and-hold repetitions.  Arc-length sampling instead gives roughly
    equal weight per travelled metre, which better exposes path geometry.
    """

    xy = group.sort_values("time")[["x", "y"]].dropna().to_numpy(dtype=float)
    if len(xy) < 2:
        return np.empty((0, 2))
    # Cumulative Euclidean distance parameterizes the polyline independently of
    # the non-uniform time spacing of the source log.
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    # Repeated poses create repeated arc coordinates.  np.interp requires an
    # increasing coordinate, so retain the first point at each unique distance.
    unique_arc, index = np.unique(arc, return_index=True)
    xy = xy[index]
    if unique_arc[-1] < step_m:
        return xy[[0]]
    # Interpolate x and y on the same distance queries to stay on the original
    # piecewise-linear trajectory.
    query = np.arange(0.0, unique_arc[-1], step_m)
    return np.column_stack(
        [np.interp(query, unique_arc, xy[:, 0]), np.interp(query, unique_arc, xy[:, 1])]
    )


def _condition_points(
    trajectories: pd.DataFrame, step_m: float
) -> dict[str, np.ndarray]:
    """Combine spatially resampled robot paths into one cloud per condition."""

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
    """Locate the strongest connected loss of natural-path occupancy.

    The comparison uses normalized smoothed histograms so runs with different
    durations or total travelled distance remain comparable.  Positive values
    of ``baseline - barrier`` indicate world locations used more by the natural
    path than after the intervention.
    """

    if "baseline" not in points:
        raise ValueError("Splitter inference needs a baseline run")
    changed_name = _changed_condition(points)
    # Both conditions must use identical grid edges; otherwise cellwise density
    # subtraction would have no geometric meaning.
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
        # Gaussian smoothing connects nearby path samples and avoids making the
        # connected component depend on single empty histogram cells.
        smooth = gaussian_filter(hist, config.occupancy_smoothing_bins)
        # Normalize each condition independently: the field now represents a
        # spatial distribution rather than an exposure-time/sample-count total.
        densities[name] = smooth / max(smooth.sum(), 1.0)
    difference = densities["baseline"] - densities[changed_name]
    # The baseline-density floor suppresses numerical tails introduced by the
    # Gaussian filter.  The relative depletion threshold retains strong change.
    positive_baseline = densities["baseline"][densities["baseline"] > 0]
    density_floor = np.quantile(positive_baseline, 0.50)
    mask = (difference >= config.component_relative_threshold * difference.max()) & (
        densities["baseline"] >= density_floor
    )
    # Connected components turn a scattered cell mask into candidate physical
    # response regions.  Select the component with the most removed probability
    # mass, not simply the largest number of pixels.
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
    # The depletion-weighted centroid initializes the local frame near the most
    # prominent centre of the condition response.
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
    """Estimate a motion-aligned orthonormal frame near the response component.

    Median unit directions reduce the influence of speed magnitude and isolated
    velocity outliers.  The final dot-product sign check makes increasing ``s``
    agree with the mean observed baseline motion.
    """

    # Cover most of the detected component but never use a tiny neighbourhood
    # that would contain too few valid baseline velocities.
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
    # Normalize before the median so fast samples do not dominate orientation.
    unit = velocities / np.linalg.norm(velocities, axis=1, keepdims=True)
    tangent = np.median(unit, axis=0)
    tangent /= np.linalg.norm(tangent)
    # A direction axis has a sign ambiguity.  Resolve it using actual motion so
    # forward gate crossings consistently move from lower to higher s.
    if np.dot(tangent, np.mean(velocities, axis=0)) < 0:
        tangent *= -1
    normal = np.array([-tangent[1], tangent[0]])
    return tangent, normal


def _interpolate_sections(
    aligned: pd.DataFrame, s_grid: np.ndarray, lateral_limit: float
) -> pd.DataFrame:
    """Interpolate every forward crossing of every transverse ``s`` section.

    A robot travels around a closed loop and can cross the same local section
    several times during a run.  Each detected negative-to-positive crossing is
    therefore retained as a passage-level observation.  Linear interpolation
    removes dependence on whether a 20 Hz sample lands exactly on the section.
    """

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
            # Strictly lower at i and greater/equal at i+1 defines a forward
            # crossing and avoids also counting motion in the reverse direction.
            crossings = np.where(
                valid[:-1] & valid[1:] & (s[:-1] < section_s) & (s[1:] >= section_s)
            )[0]
            passage_number = 0
            for index in crossings:
                denominator = s[index + 1] - s[index]
                if denominator <= 0:
                    continue
                # alpha in [0,1] locates the continuous crossing between the two
                # discrete samples.  The same alpha is used for d, time, and
                # velocity to keep the interpolated record self-consistent.
                alpha = (section_s - s[index]) / denominator
                d_cross = d[index] + alpha * (d[index + 1] - d[index])
                # The lateral window rejects crossings on unrelated parts of
                # the closed loop that happen to have the same projected s.
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
    """Fit globally optimal K=2 K-means and standard mean silhouette.

    K=2 is used as a transparent baseline because the physical hypothesis is
    specifically two routes around one splitter.  In one dimension, every
    squared-error K-means partition consists of two contiguous blocks after
    sorting.  Enumerating every non-empty split therefore gives the global
    optimum and remains simpler than importing a general clustering package.
    """

    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("1-D K-means needs at least two finite values")

    # stable sorting makes ties repeatable and lets us restore labels to the
    # original crossing order after choosing the best sorted split.
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    best_sse = np.inf
    best_split = 1
    best_centers = np.array([sorted_values[0], sorted_values[1]])
    for split in range(1, len(sorted_values)):
        left, right = sorted_values[:split], sorted_values[split:]
        centers = np.array([left.mean(), right.mean()])
        # Each mean minimizes squared error within its proposed block.  Testing
        # every boundary avoids the local-minimum failure possible with Lloyd
        # iteration and quartile initialization.
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

    # Standard silhouette excludes the point itself from its within-cluster
    # distance a(i).  A singleton cluster has silhouette zero by convention.
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
    """Estimate ordinary baseline W1 variability by repeated random half-splits.

    This empirical null avoids declaring ordinary baseline variation to be a
    barrier response and is easier to explain than a fitted parametric model.
    All sections contribute to one pooled null distribution, which is adequate
    for this preliminary two-run comparison and yields one readable threshold
    line in Figure 3.
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
            # Both halves come from the same no-barrier section, so their W1
            # distance represents finite-sample variation under the null.
            shuffled = rng.permutation(values)
            middle = len(shuffled) // 2
            null.append(
                float(wasserstein_distance(shuffled[:middle], shuffled[middle:]))
            )
    return float(np.quantile(null, config.null_quantile)) if null else 0.05


def _interval_near_peak(mask: np.ndarray, peak_index: int) -> tuple[int, int]:
    """Select a contiguous valid interval instead of disconnected sections.

    Prefer the interval containing the strongest Wasserstein response.  If the
    stricter mask (for example stable-core evidence) does not contain that exact
    peak, choose its longest supported interval.  Returning indices preserves
    direct alignment with the ordered section diagnostics table.
    """

    # Appending one False sentinel closes an interval that reaches the final
    # element without needing a special post-loop case.
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
    """Execute the complete data-derived local response-geometry algorithm.

    The returned diagnostics deliberately expose intermediate evidence such as
    sample support, W1, cluster centers, separation, silhouette, and gap
    fraction.  This makes the final gates/separator inspectable instead of a
    black-box coordinate estimate.
    """

    config = config or SplitterInferenceConfig()
    # Step 1: remove time-density bias and find a world-frame response origin.
    points = _condition_points(trajectories, config.spatial_step_m)
    origin, component = _response_component(points, config)

    # Step 2: estimate the local longitudinal direction from natural baseline
    # motion near that response origin.  The normal defines signed lateral d.
    tangent, normal = _local_frame(trajectories, origin, component)

    # A provisional geometry is sufficient to rotate every trajectory into the
    # local frame.  Its default gates are not used for the inference below; the
    # final gates are reconstructed after section analysis.
    provisional = AnalysisGeometry(
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        tangent_x=float(tangent[0]),
        tangent_y=float(tangent[1]),
        normal_x=float(normal[0]),
        normal_y=float(normal[1]),
    )
    aligned = world_to_corridor(trajectories, provisional)
    # Project the selected occupancy component onto s and extend the section
    # grid upstream/downstream so response onset and exit can be observed around
    # the initially detected region.
    component_s = (component["x"] - origin[0]) * tangent[0] + (
        component["y"] - origin[1]
    ) * tangent[1]
    grid_min = float(component_s.min() - config.section_upstream_margin_m)
    grid_max = float(component_s.max() + config.section_downstream_margin_m)
    s_grid = np.arange(
        grid_min, grid_max + config.section_step_m / 2, config.section_step_m
    )
    # Step 3: each row is a passage crossing, not a raw frame.  This prevents a
    # slow robot from contributing many more observations at one section.
    crossings = _interpolate_sections(aligned, s_grid, config.section_half_width_m)
    changed_name = _changed_condition(crossings["condition"])

    # Step 4: quantify how much W1 can arise when baseline samples are randomly
    # split into two groups at the same sections.
    threshold = _null_threshold(crossings, s_grid, config)

    rows: list[dict[str, object]] = []
    # Step 5: compare the two conditions and test the explicit two-route
    # hypothesis independently at each longitudinal section.
    for section_s in s_grid:
        base = crossings[
            (crossings["condition"] == "baseline")
            & np.isclose(crossings["s"], section_s)
        ]["d"].to_numpy(dtype=float)
        changed = crossings[
            (crossings["condition"] == changed_name)
            & np.isclose(crossings["s"], section_s)
        ]["d"].to_numpy(dtype=float)
        # Initialize unavailable cluster diagnostics as NaN/False.  A section
        # with insufficient barrier crossings still has useful W1/support data.
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
            # The midpoint band spans 30% of center separation in total.  A low
            # fraction there is evidence of an actual depleted lane between the
            # two route families rather than an arbitrary split of one broad mode.
            gap_fraction = float(
                np.mean(np.abs(changed - midpoint) < 0.15 * separation)
            )
            # Stable core = two sufficiently populated, separated, internally
            # coherent route bands with the natural baseline path between them.
            # These requirements are stricter than merely obtaining two K-means
            # labels, because K-means always returns two clusters when possible.
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
    # Step 6: require both conditions to have at least 75% of their typical
    # non-zero crossing count.  This prevents a large W1 based on a nearly empty
    # section from defining the response interval.
    baseline_support = (
        diagnostics["baseline_n"].to_numpy()
        >= 0.75 * diagnostics.loc[diagnostics["baseline_n"] > 0, "baseline_n"].median()
    )
    changed_support = (
        diagnostics["changed_n"].to_numpy()
        >= 0.75 * diagnostics.loc[diagnostics["changed_n"] > 0, "changed_n"].median()
    )
    supported = baseline_support & changed_support
    # A supported section becomes a response section only when the observed
    # cross-condition W1 exceeds ordinary baseline split variability.
    response = (
        diagnostics["wasserstein_m"].fillna(0).to_numpy() > threshold
    ) & supported
    # Anchor interval selection at the strongest observed distribution change.
    max_index = int(diagnostics["wasserstein_m"].fillna(-np.inf).to_numpy().argmax())
    response_start, response_end = _interval_near_peak(response, max_index)
    # The effective splitter is only fitted where distribution change and the
    # stronger two-route evidence are simultaneously present.
    core_mask = response & diagnostics["stable_core_valid"].to_numpy(dtype=bool)
    core_start, core_end = _interval_near_peak(core_mask, max_index)
    if core_start == core_end and not core_mask[core_start]:
        raise RuntimeError(
            "The changed trajectories do not support a stable two-route separator"
        )

    # Record only the selected contiguous core, not every isolated section that
    # individually passes the threshold.
    selected_core_mask = np.zeros(len(diagnostics), dtype=bool)
    selected_core_mask[core_start : core_end + 1] = True

    core = diagnostics.iloc[core_start : core_end + 1].dropna(
        subset=["separator_midpoint_m"]
    )
    if len(core) < 2:
        raise RuntimeError(
            "At least two stable sections are required to fit a separator"
        )
    # Step 7: each section contributes one midpoint between its two route
    # centers.  A first-degree least-squares fit gives an easy-to-explain local
    # separator that can be evaluated at any core s coordinate.
    coefficients = np.polyfit(core["s"], core["separator_midpoint_m"], deg=1)
    slope, intercept = map(float, coefficients)
    median_separation = float(core["cluster_separation_m"].median())
    # The uncertainty band is intentionally simple for this preliminary dataset:
    # never narrower than 2 cm, otherwise 10% of typical route separation.
    ambiguity = max(0.02, 0.10 * median_separation)

    # Step 8: convert intervals into semantically named virtual gates.
    # G0 is the earliest well-supported section available before/on response
    # onset; G1 is the first significant response section.
    upstream = diagnostics.iloc[: response_start + 1]
    supported_upstream = upstream.loc[supported[: response_start + 1], "s"]
    g0 = float(supported_upstream.iloc[0])
    g1 = float(diagnostics["s"].iloc[response_start])
    g3 = float(diagnostics["s"].iloc[response_end])
    # G2 is a representative section at the core midpoint, while G3 is the last
    # supported significant-response section.  G3 is an analysis exit and does
    # not automatically prove that the flow has fully recovered there.
    c0, c1 = float(diagnostics["s"].iloc[core_start]), float(
        diagnostics["s"].iloc[core_end]
    )
    g2 = 0.5 * (c0 + c1)
    if not g0 < g1:
        raise RuntimeError(
            "No supported approach section was found before the response"
        )
    # Constructing AnalysisGeometry re-checks orthonormality and gate ordering
    # before any passage or state extraction can consume the result.
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
    # Attach the final boolean masks so Figure 3 and CSV inspection use exactly
    # the same decisions that produced the geometry.
    diagnostics["section_support_valid"] = supported
    diagnostics["response_valid"] = response
    diagnostics["core_valid"] = selected_core_mask
    return SplitterInferenceResult(geometry, diagnostics, crossings)
