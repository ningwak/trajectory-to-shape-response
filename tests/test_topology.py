from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

nx = pytest.importorskip("networkx")

from reflow_analysis.topology import TopologyResult, map_match_trajectory


def test_map_matching_uses_node_ids_instead_of_dataframe_index():
    graph = nx.DiGraph()
    graph.add_edges_from([(10, 20), (20, 30)])
    node_table = pd.DataFrame(
        {
            "node_id": [20, 10, 30],
            "x": [1.0, 0.0, 2.0],
            "y": [0.0, 0.0, 0.0],
            "degree": [2, 1, 1],
        }
    ).reset_index(drop=True)
    topology = TopologyResult(
        graph=graph,
        occupancy=np.ones((1, 1)),
        skeleton=np.ones((1, 1), dtype=bool),
        x_centers=np.array([0.0]),
        y_centers=np.array([0.0]),
        node_table=node_table,
        edge_table=pd.DataFrame(),
    )
    trajectory = pd.DataFrame(
        {"x": [0.0, 1.0, 2.0], "y": [0.0, 0.0, 0.0], "vx": [1.0] * 3, "vy": [0.0] * 3}
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        matched = map_match_trajectory(trajectory, topology)

    assert matched["matched_node_id"].tolist() == [10, 20, 30]
