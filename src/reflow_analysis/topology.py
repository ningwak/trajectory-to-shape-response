"""Optional, geometry-agnostic global path-topology baseline.

This module is deliberately outside the main splitter pipeline.  It turns a
baseline occupancy image into a directed skeleton graph and therefore supports
open, closed, and branched paths without assuming a rectangle.

It is a concrete extensibility baseline, not a prerequisite for the supplied
two-condition local analysis.  Optional ``networkx`` and ``scikit-image``
dependencies are imported only when this module is executed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial import cKDTree

@dataclass(frozen=True)
class TopologyConfig:
    """Occupancy-grid resolution, cleanup, and graph-pruning settings."""

    grid_size_m: float = 0.04
    smoothing_bins: float = 1.2
    occupancy_quantile: float = 0.55
    minimum_component_pixels: int = 12
    spur_length_m: float = 0.12


@dataclass
class TopologyResult:
    """Directed path graph plus the raster evidence used to construct it."""

    graph: object
    occupancy: np.ndarray
    skeleton: np.ndarray
    x_centers: np.ndarray
    y_centers: np.ndarray
    node_table: pd.DataFrame
    edge_table: pd.DataFrame


def _require_optional_dependencies():
    """Keep optional topology packages out of the main installation path."""

    try:
        import networkx as nx
        from skimage.morphology import closing, disk, skeletonize
    except ImportError as exc:
        raise ImportError(
            "Global topology needs optional dependencies: pip install -e '.[topology]'"
        ) from exc
    return nx, closing, disk, skeletonize


def _spatially_resample(group: pd.DataFrame, step_m: float) -> np.ndarray:
    """Sample one trajectory at equal travelled-distance intervals.

    This helper belongs to the *optional global topology baseline*. Its
    occupancy image would otherwise give stationary portions more weight than
    moving portions. It is intentionally separate from horizontal sections in
    the main splitter analysis, which use one sample per forward crossing.
    """

    xy = group.sort_values("time")[["x", "y"]].dropna().to_numpy(dtype=float)
    if len(xy) < 2:
        return np.empty((0, 2))
    # Repeated held poses give duplicate distance coordinates, unsuitable for
    # interpolation. Keep their first location at each unique arc length.
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    unique_arc, index = np.unique(arc, return_index=True)
    xy = xy[index]
    if unique_arc[-1] < step_m:
        return xy[[0]]
    query = np.arange(0.0, unique_arc[-1], step_m)
    return np.column_stack(
        [np.interp(query, unique_arc, xy[:, 0]), np.interp(query, unique_arc, xy[:, 1])]
    )


def _prune_short_spurs(graph, positions: dict, maximum_length: float) -> None:
    """Remove short skeleton artefacts that terminate at a real junction.

    Skeletonization often creates tiny leaf branches from raster noise.  Starting
    at each degree-one leaf, the code walks through degree-two nodes until a
    termination or junction is reached.  A short leaf-to-junction path is pruned
    while the junction itself is preserved.
    """

    changed = True
    # Removing one branch changes graph degrees and may expose another spur, so
    # restart scanning after every deletion until a full pass changes nothing.
    while changed:
        changed = False
        for leaf in [node for node in graph.nodes if graph.degree(node) == 1]:
            path = [leaf]
            previous = None
            current = leaf
            length = 0.0
            while graph.degree(current) <= 2:
                candidates = [
                    node for node in graph.neighbors(current) if node != previous
                ]
                if not candidates:
                    break
                nxt = candidates[0]
                length += float(np.linalg.norm(positions[current] - positions[nxt]))
                path.append(nxt)
                previous, current = current, nxt
                if graph.degree(current) != 2:
                    break
            # Only prune a short branch attached to a junction.  A legitimate
            # short path between two endpoints is not removed by this rule.
            if graph.degree(current) >= 3 and length < maximum_length:
                graph.remove_nodes_from(path[:-1])
                changed = True
                break


def infer_global_path_topology(
    trajectories: pd.DataFrame, config: TopologyConfig | None = None
) -> TopologyResult:
    """Convert baseline occupancy into a cleaned, directed skeleton graph.

    This method makes no closed-loop or quadrilateral assumption.  Its limits
    are instead those of a raster skeleton baseline: topology depends on grid,
    smoothing, threshold, and spur-pruning settings.
    """

    nx, closing, disk, skeletonize = _require_optional_dependencies()
    config = config or TopologyConfig()
    # Only natural no-barrier motion defines global path structure.  Barrier
    # detours should not become permanent branches in this baseline map.
    baseline = trajectories[trajectories["condition"] == "baseline"]
    point_groups = []
    # Equal-arc-length sampling prevents slow portions from dominating the
    # occupancy image. The main local method instead samples section crossings.
    for _, group in baseline.groupby(["run_name", "robot_id"]):
        points = _spatially_resample(group[group["valid_pose"]], config.grid_size_m / 2)
        if len(points):
            point_groups.append(points)
    points = np.vstack(point_groups)
    margin = 2 * config.grid_size_m
    x_edges = np.arange(
        points[:, 0].min() - margin, points[:, 0].max() + margin, config.grid_size_m
    )
    y_edges = np.arange(
        points[:, 1].min() - margin, points[:, 1].max() + margin, config.grid_size_m
    )
    occupancy, _, _ = np.histogram2d(
        points[:, 0], points[:, 1], bins=(x_edges, y_edges)
    )
    # Smoothing fills sub-cell sampling gaps; quantile thresholding retains the
    # more consistently occupied corridor footprint.
    occupancy = gaussian_filter(occupancy, config.smoothing_bins)
    positive = occupancy[occupancy > 0]
    binary = occupancy >= np.quantile(positive, config.occupancy_quantile)
    # Morphological closing bridges small holes before skeletonization.
    binary = closing(binary, disk(1))
    from scipy.ndimage import label as component_label

    component_ids, component_count = component_label(binary)
    # Remove isolated pixel islands that are too small to represent a supported
    # global path component.
    for component_id in range(1, component_count + 1):
        if np.sum(component_ids == component_id) < config.minimum_component_pixels:
            binary[component_ids == component_id] = False
    # Reduce the occupied corridor mask to a one-pixel-wide topological centreline.
    skeleton = skeletonize(binary)
    x_centers, y_centers = (x_edges[:-1] + x_edges[1:]) / 2, (
        y_edges[:-1] + y_edges[1:]
    ) / 2

    # Initially create an undirected 8-neighbour pixel graph.  Direction is added
    # later from measured velocities rather than inferred from pixel ordering.
    graph = nx.Graph()
    pixel_to_node = {}
    positions = {}
    for node_id, (i, j) in enumerate(zip(*np.where(skeleton))):
        pixel_to_node[(i, j)] = node_id
        positions[node_id] = np.array([x_centers[i], y_centers[j]])
        graph.add_node(node_id, x=x_centers[i], y=y_centers[j])
    for (i, j), node in pixel_to_node.items():
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                neighbor = pixel_to_node.get((i + di, j + dj))
                # neighbor > node adds every undirected edge exactly once.
                if neighbor is not None and neighbor > node:
                    graph.add_edge(
                        node,
                        neighbor,
                        length=float(
                            np.linalg.norm(positions[node] - positions[neighbor])
                        ),
                    )
    _prune_short_spurs(graph, positions, config.spur_length_m)

    # Map every valid baseline velocity to its nearest remaining skeleton node.
    active_nodes = list(graph.nodes)
    tree = cKDTree(np.vstack([positions[node] for node in active_nodes]))
    # Store votes under a canonical sorted edge key.  The sign later decides
    # which endpoint becomes source versus target.
    velocity_votes = {tuple(sorted(edge)): 0.0 for edge in graph.edges}
    valid_velocity = baseline[
        baseline["valid_kinematics"] & baseline[["vx", "vy"]].notna().all(axis=1)
    ]
    _, nearest = tree.query(valid_velocity[["x", "y"]].to_numpy(float))
    for node, velocity in zip(
        np.asarray(active_nodes)[nearest], valid_velocity[["vx", "vy"]].to_numpy(float)
    ):
        for neighbor in graph.neighbors(int(node)):
            displacement = positions[neighbor] - positions[int(node)]
            # Positive dot product supports motion from node toward neighbour;
            # the endpoint-order factor expresses this in the canonical key's
            # low-to-high orientation.
            velocity_votes[tuple(sorted((int(node), neighbor)))] += float(
                np.dot(velocity, displacement)
            ) * (1 if int(node) < neighbor else -1)

    # Convert each edge once using the sign of accumulated local velocity votes.
    directed = nx.DiGraph()
    directed.add_nodes_from(graph.nodes(data=True))
    edge_rows = []
    for edge, vote in velocity_votes.items():
        low, high = edge
        source, target = (low, high) if vote >= 0 else (high, low)
        directed.add_edge(
            source, target, length=graph.edges[edge]["length"], direction_vote=abs(vote)
        )
        edge_rows.append(
            {
                "source": source,
                "target": target,
                "length_m": graph.edges[edge]["length"],
                "direction_vote": abs(vote),
            }
        )
    node_table = pd.DataFrame(
        [
            {
                "node_id": node,
                "x": positions[node][0],
                "y": positions[node][1],
                "degree": graph.degree(node),
            }
            for node in graph.nodes
        ]
    )
    return TopologyResult(
        directed,
        occupancy,
        skeleton,
        x_centers,
        y_centers,
        node_table,
        pd.DataFrame(edge_rows),
    )


def map_match_trajectory(
    trajectory: pd.DataFrame, topology: TopologyResult, heading_weight: float = 0.15
) -> pd.DataFrame:
    """Greedily map samples to nearby graph nodes using three transparent costs.

    Cost components are Euclidean point-to-node distance, a fixed penalty for a
    non-adjacent jump from the previous match, and optional heading disagreement
    with outgoing directed edges.  This is intentionally simpler than an HMM or
    globally optimal dynamic-programming map matcher.
    """

    nodes = topology.node_table.sort_values("node_id")
    tree = cKDTree(nodes[["x", "y"]].to_numpy(float))
    # Five nearby nodes provide alternatives around junctions without scoring
    # the complete skeleton for every trajectory sample.
    distances, indices = tree.query(
        trajectory[["x", "y"]].to_numpy(float), k=min(5, len(nodes))
    )
    if distances.ndim == 1:
        distances, indices = distances[:, None], indices[:, None]
    selected = []
    previous = None
    velocity = (
        trajectory[["vx", "vy"]].to_numpy(float)
        if {"vx", "vy"}.issubset(trajectory)
        else np.full((len(trajectory), 2), np.nan)
    )
    positions = nodes[["x", "y"]].to_numpy(float)
    node_ids = nodes["node_id"].to_numpy(int)
    # Never use a DataFrame row index as a graph-node identifier.  This explicit
    # mapping remains correct if node_table was shuffled or reindexed after
    # topology construction.
    position_by_id = dict(zip(node_ids, positions))
    for row, candidates in enumerate(indices):
        costs = distances[row].astype(float)
        # Continuity cost allows staying at the current node or moving to an
        # adjacent node; larger graph jumps receive a fixed penalty.
        if previous is not None:
            costs += np.array(
                [
                    (
                        0
                        if node_ids[index] == previous
                        or topology.graph.has_edge(previous, node_ids[index])
                        else 0.25
                    )
                    for index in candidates
                ]
            )
        # When velocity exists, prefer a candidate whose best outgoing edge is
        # aligned with the observed motion direction.
        if np.isfinite(velocity[row]).all() and np.linalg.norm(velocity[row]) > 0:
            direction = velocity[row] / np.linalg.norm(velocity[row])
            for k, index in enumerate(candidates):
                node = node_ids[index]
                successors = list(topology.graph.successors(node))
                if successors:
                    tangents = [
                        position_by_id[nxt] - positions[index] for nxt in successors
                    ]
                    alignment = max(
                        np.dot(direction, tangent / np.linalg.norm(tangent))
                        for tangent in tangents
                    )
                    costs[k] += heading_weight * (1 - alignment)
        previous = int(node_ids[candidates[int(np.argmin(costs))]])
        selected.append(previous)
    result = trajectory.copy()
    result["matched_node_id"] = selected
    return result
