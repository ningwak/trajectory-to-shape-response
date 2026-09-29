"""Translate already-computed results into six interview-focused figures.

Plotting is intentionally downstream-only: no function in this module changes
gate positions, reclassifies a route, or recomputes an evaluation metric.  This
keeps visual design separate from scientific decisions and makes every plotted
number traceable to a table or inferred-geometry field.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry
from .io import condition_order
from .passages import extract_passage_samples
from .splitter_inference import SplitterInferenceResult

ROUTE_COLORS = {"inner": "#2F5597", "outer": "#ED7D31", "ambiguous": "#7F7F7F"}
CONDITION_COLORS = {"baseline": "#4472C4", "barrier_v01": "#C00000"}


def _save(fig: plt.Figure, path: Path) -> Path:
    """Apply the same layout, DPI, bounding-box, and close behaviour to figures."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_full_trajectories(
    trajectories: pd.DataFrame, manifest: pd.DataFrame, path: Path
) -> Path:
    """Figure 1: world-frame overview with directly comparable panel scales.

    Both conditions use limits computed from the union of all valid trajectories
    and an equal physical aspect ratio.  A metre therefore has the same visual
    length in both panels; condition differences cannot be amplified by separate
    autoscaling.
    """

    conditions = condition_order(trajectories["condition"])
    # One global bounding box, padded by 4%, is shared by both condition axes.
    valid = trajectories[trajectories["valid_pose"]]
    x_padding = 0.04 * (valid["x"].max() - valid["x"].min())
    y_padding = 0.04 * (valid["y"].max() - valid["y"].min())
    x_limits = (valid["x"].min() - x_padding, valid["x"].max() + x_padding)
    y_limits = (valid["y"].min() - y_padding, valid["y"].max() + y_padding)
    fig, axes = plt.subplots(
        1,
        len(conditions),
        figsize=(6.2 * len(conditions), 5.0),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for col, condition in enumerate(conditions):
        ax = axes[0, col]
        data = trajectories[
            (trajectories["condition"] == condition) & trajectories["valid_pose"]
        ]
        # Include run_name in the grouping so a future second run with reused
        # robot IDs cannot be connected to the first run by one false line.
        for index, ((_, robot_id), group) in enumerate(
            data.groupby(["run_name", "robot_id"], sort=True)
        ):
            ax.plot(
                group["x"],
                group["y"],
                lw=0.65,
                alpha=0.65,
                color=plt.get_cmap("tab10")(index),
                label=f"R{robot_id}",
            )
        # The dashed JSON polygon is contextual metadata only.  No later
        # splitter-inference function reads its plotted coordinates.
        row = manifest[manifest["condition"] == condition].iloc[0]
        corners = np.asarray(json.loads(row["rectangle_corners"]), dtype=float)
        corners = np.vstack([corners, corners[0]])
        ax.plot(
            corners[:, 0],
            corners[:, 1],
            "--",
            color="#333333",
            lw=1.2,
            label="nominal route from JSON",
        )
        ax.set(
            title=condition.replace("_", " "),
            xlabel="x [m]",
            ylabel="y [m]",
            xlim=x_limits,
            ylim=y_limits,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.18)
        ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Complete trajectories (no manually entered splitter geometry)")
    return _save(fig, path)


def plot_inferred_local_geometry(
    trajectories: pd.DataFrame,
    passages: pd.DataFrame,
    geometry: AnalysisGeometry,
    path: Path,
) -> Path:
    """Figure 2: connect passage paths, gates, and the stable separator core.

    The separator is drawn only in the barrier panel and only from core_start to
    core_end.  Baseline paths are neutral gray because the barrier-derived
    inner/outer categories are not physically present there.
    """

    segments = extract_passage_samples(trajectories, passages, valid_only=True)
    conditions = condition_order(trajectories["condition"])
    fig, axes = plt.subplots(
        1, len(conditions), figsize=(6.4 * len(conditions), 4.5), squeeze=False
    )
    for col, condition in enumerate(conditions):
        ax = axes[0, col]
        subset = segments[segments["condition"] == condition]
        # run_name prevents repeated IDs from separate future runs being joined.
        for _, group in subset.groupby(
            ["run_name", "robot_id", "passage_id"], sort=False
        ):
            route = group["route_label"].iloc[0]
            color = (
                "#888888"
                if condition == "baseline"
                else ROUTE_COLORS.get(route, "#aaaaaa")
            )
            ax.plot(
                group["s"],
                group["d"],
                color=color,
                alpha=0.30,
                lw=0.8,
            )
        for gate, label, color in [
            (geometry.g0_approach_entry_s, "G0", "#70AD47"),
            (geometry.g1_response_entry_s, "G1", "#FFC000"),
            (geometry.g2_route_s, "G2", "#7030A0"),
            (geometry.g3_recovery_s, "G3", "#5B9BD5"),
        ]:
            ax.axvline(gate, ls="--", lw=1, color=color, label=label)
        # Inner/outer separation is not a physical object in the baseline run,
        # so the black line and ambiguity band are intentionally barrier-only.
        if condition != "baseline":
            core_s = np.linspace(geometry.core_start_s, geometry.core_end_s, 100)
            ax.plot(
                core_s,
                geometry.splitter_d(core_s),
                color="#111111",
                lw=3,
                label="effective splitter",
            )
            ax.fill_between(
                core_s,
                geometry.splitter_d(core_s) - geometry.ambiguous_half_width_m,
                geometry.splitter_d(core_s) + geometry.ambiguous_half_width_m,
                color="#888888",
                alpha=0.18,
                label="ambiguous band",
            )
        ax.axvspan(
            geometry.g0_approach_entry_s,
            geometry.g1_response_entry_s,
            color="#70AD47",
            alpha=0.08,
        )
        ax.set(
            title=condition.replace("_", " "),
            xlabel="local longitudinal s [m]",
            ylabel="local lateral d [m]",
            xlim=(geometry.g0_approach_entry_s - 0.05, geometry.g3_recovery_s + 0.05),
            ylim=(-geometry.gate_half_width_m, geometry.gate_half_width_m),
        )
        ax.grid(alpha=0.18)
    handles, labels = [], []
    for ax in axes[0]:
        current_handles, current_labels = ax.get_legend_handles_labels()
        handles.extend(current_handles)
        labels.extend(current_labels)
    unique = dict(zip(labels, handles))
    fig.legend(unique.values(), unique.keys(), ncol=4, fontsize=8, loc="lower center")
    fig.suptitle("Data-derived local response geometry and passage labels")
    fig.subplots_adjust(bottom=0.18)
    return _save(fig, path)


def plot_wasserstein_profile(
    diagnostics: pd.DataFrame, geometry: AnalysisGeometry, path: Path
) -> Path:
    """Figure 3: show response detection and stable-core evidence along s.

    W1 above the empirical null indicates a detectable condition response when
    section support is adequate.  The two shaded intervals distinguish that
    response from the stricter region used to fit the separator.
    """

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(
        diagnostics["s"],
        diagnostics["wasserstein_m"],
        color="#4472C4",
        lw=2,
        label="baseline vs barrier W1",
    )
    ax.plot(
        diagnostics["s"],
        diagnostics["null_threshold_m"],
        "--",
        color="#C00000",
        label="baseline split-null 95%",
    )
    ax.axvspan(
        geometry.g1_response_entry_s,
        geometry.g3_recovery_s,
        color="#FFC000",
        alpha=0.12,
        label="supported response interval",
    )
    ax.axvspan(
        geometry.core_start_s,
        geometry.core_end_s,
        color="#7030A0",
        alpha=0.10,
        label="bimodal core",
    )
    ax.set(
        xlabel="local longitudinal s [m]",
        ylabel="1-D Wasserstein distance [m]",
        title="Where a trajectory response becomes detectable",
    )
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_cross_section(result: SplitterInferenceResult, path: Path) -> Path:
    """Figure 4: show actual lateral crossing samples at the section nearest G2.

    Categorical y positions are exact: all baseline points lie on one row and all
    barrier points on the other.  No jitter is added because a displaced y value
    would look like a second physical coordinate.
    """

    geometry, crossings = result.geometry, result.crossing_samples
    # G2 is derived from a core midpoint and need not equal a floating-point grid
    # value exactly; select the stored section with minimum absolute difference.
    available_s = np.sort(crossings["s"].unique())
    section_s = float(available_s[np.argmin(np.abs(available_s - geometry.g2_route_s))])
    section = crossings[np.isclose(crossings["s"], section_s)]
    changed = condition_order(section["condition"])[1]
    diag = result.section_diagnostics.iloc[
        (result.section_diagnostics["s"] - geometry.g2_route_s).abs().argmin()
    ]
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    for index, condition in enumerate(["baseline", changed]):
        values = section[section["condition"] == condition]["d"].to_numpy(float)
        ax.scatter(
            values,
            np.full(len(values), index),
            alpha=0.45,
            s=22,
            color=CONDITION_COLORS.get(condition),
            label=condition,
        )
    # Centers come from barrier samples only.  Their midpoint is the section-wise
    # separator observation; the black line uses the fitted separator at G2.
    for index, center in enumerate(
        [diag["cluster_0_center_m"], diag["cluster_1_center_m"]]
    ):
        ax.axvline(
            center,
            color="#ED7D31",
            lw=1.5,
            ls="--",
            label="K=2 route centers" if index == 0 else None,
        )
    separator = float(geometry.splitter_d(geometry.g2_route_s))
    ax.axvline(
        separator, color="#111111", lw=2.5, label="K=2 midpoint / fitted separator"
    )
    ax.set_yticks([0, 1], ["baseline", changed])
    ax.invert_yaxis()
    ax.set_xlabel("lateral crossing position d [m]")
    ax.set_title(
        f"Representative G2 cross-section: separation={diag['cluster_separation_m']:.3f} m, silhouette={diag['silhouette']:.2f}"
    )
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    return _save(fig, path)


def plot_route_distribution(distribution: pd.DataFrame, path: Path) -> Path:
    """Figure 5: plot the barrier-only route-composition vector.

    Inner/outer labels are defined by the splitter inferred from the changed
    condition. Applying those labels to the unsplit baseline would be visually
    misleading, so baseline is deliberately omitted from this figure.
    """

    changed = distribution[distribution["condition"] != "baseline"]
    pivot = (
        changed.pivot(
            index="condition", columns="route_label", values="route_proportion"
        )
        .reindex(columns=list(ROUTE_COLORS))
        .fillna(0)
    )
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    left = np.zeros(len(pivot))
    for route in pivot:
        values = pivot[route].to_numpy()
        ax.barh(pivot.index, values, left=left, color=ROUTE_COLORS[route], label=route)
        for row, value in enumerate(values):
            if value >= 0.04:
                display_route = "ambig." if route == "ambiguous" else route
                ax.text(
                    left[row] + value / 2,
                    row,
                    f"{display_route}\n{100*value:.1f}%",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        left += values
    condition_name = str(pivot.index[0]).replace("_", " ")
    ax.set(
        xlabel=f"proportion of valid {condition_name} passages",
        xlim=(0, 1),
        title=f"{condition_name.title()} route allocation",
    )
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, path)


def plot_passage_time(per_robot: pd.DataFrame, path: Path) -> Path:
    """Figure 6: connect each robot's baseline and barrier median passage time.

    Lines encode paired robot identity, not statistical confidence intervals.
    Colour and legend avoid overlapping text labels at similar barrier values.
    """

    # Pivoting gives one row per physical robot and one column per condition, so
    # every line joins like with like rather than two unrelated condition means.
    pivot = per_robot.pivot_table(
        index="robot_id", columns="condition", values="median_passage_time_s"
    )
    conditions = condition_order(pivot.columns)
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for index, (robot_id, row) in enumerate(pivot.iterrows()):
        ax.plot(
            range(len(conditions)),
            row[conditions],
            "o-",
            color=plt.get_cmap("tab10")(index),
            alpha=0.75,
            lw=1.2,
            label=f"R{robot_id}",
        )
    ax.set_xticks(range(len(conditions)), [c.replace("_", " ") for c in conditions])
    ax.set(
        ylabel="median G0-to-G3 passage time [s]",
        title="Robot-matched local passage time",
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(title="robot", ncol=2, frameon=False)
    return _save(fig, path)


def generate_all_plots(
    trajectories: pd.DataFrame,
    manifest: pd.DataFrame,
    passages: pd.DataFrame,
    evaluation: dict[str, pd.DataFrame],
    inference: SplitterInferenceResult,
    output_dir: Path,
) -> dict[str, Path]:
    """Generate the fixed six-figure set with numbered, stable filenames."""

    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = inference.geometry
    return {
        "full_trajectories": plot_full_trajectories(
            trajectories, manifest, output_dir / "01_full_trajectories.png"
        ),
        "inferred_local_geometry": plot_inferred_local_geometry(
            trajectories,
            passages,
            geometry,
            output_dir / "02_inferred_local_geometry.png",
        ),
        "wasserstein_profile": plot_wasserstein_profile(
            inference.section_diagnostics,
            geometry,
            output_dir / "03_wasserstein_profile.png",
        ),
        "cross_section": plot_cross_section(
            inference, output_dir / "04_cross_section_kmeans.png"
        ),
        "route_distribution": plot_route_distribution(
            evaluation["route_distribution"], output_dir / "05_route_distribution.png"
        ),
        "passage_time": plot_passage_time(
            evaluation["robot_passage_time"], output_dir / "06_passage_time.png"
        ),
    }
