"""Standalone CLI for the optional global path-topology baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from .io import load_dataset
from .preprocessing import preprocess_dataset
from .topology import infer_global_path_topology


def main() -> None:
    """Run the optional global skeleton-graph baseline and export its graph."""

    parser = argparse.ArgumentParser(
        description="Infer a general baseline path skeleton graph."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trajectories, _ = preprocess_dataset(load_dataset(args.data_root))
    result = infer_global_path_topology(trajectories)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result.node_table.to_csv(args.output_dir / "topology_nodes.csv", index=False)
    result.edge_table.to_csv(args.output_dir / "topology_edges.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(
        result.occupancy.T,
        origin="lower",
        extent=[
            result.x_centers.min(),
            result.x_centers.max(),
            result.y_centers.min(),
            result.y_centers.max(),
        ],
        cmap="Greys",
        alpha=0.55,
    )
    for edge in result.edge_table.itertuples(index=False):
        nodes = result.node_table.set_index("node_id").loc[[edge.source, edge.target]]
        ax.plot(nodes["x"], nodes["y"], color="#C00000", lw=0.8)
    ax.set_aspect("equal")
    ax.set(
        title="Optional baseline path-topology graph", xlabel="x [m]", ylabel="y [m]"
    )
    fig.savefig(args.output_dir / "topology_graph.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(result.node_table.to_string(index=False))
    print(result.edge_table.to_string(index=False))


if __name__ == "__main__":
    main()
