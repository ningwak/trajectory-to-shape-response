"""Figures separated from printed/CSV tables for interview use."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .geometry import AnalysisGeometry
from .passages import extract_passage_samples
from .splitter_inference import SplitterInferenceResult


ROUTE_COLORS = {"inner": "#2F5597", "outer": "#ED7D31", "ambiguous": "#7F7F7F"}
CONDITION_COLORS = {"baseline": "#4472C4", "barrier_v01": "#C00000"}


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_full_trajectories(trajectories: pd.DataFrame, manifest: pd.DataFrame, path: Path) -> Path:
    """Raw overview only: trajectories and the JSON reference rectangle."""

    conditions = sorted(trajectories["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(6.2 * len(conditions), 5.0), squeeze=False)
    for col, condition in enumerate(conditions):
        ax = axes[0, col]
        data = trajectories[(trajectories["condition"] == condition) & trajectories.get("valid_pose", True)]
        for index, (robot_id, group) in enumerate(data.groupby("robot_id", sort=True)):
            ax.plot(group["x"], group["y"], lw=.65, alpha=.65, color=plt.get_cmap("tab10")(index), label=f"R{robot_id}")
        row = manifest[manifest["condition"] == condition].iloc[0]
        corners = np.asarray(json.loads(row["rectangle_corners"]), dtype=float)
        corners = np.vstack([corners, corners[0]])
        ax.plot(corners[:, 0], corners[:, 1], "--", color="#333333", lw=1.2, label="JSON reference rectangle")
        ax.set(title=condition.replace("_", " "), xlabel="x [m]", ylabel="y [m]")
        ax.set_aspect("equal", adjustable="box"); ax.grid(alpha=.18); ax.legend(fontsize=7, ncol=2)
    fig.suptitle("Complete trajectories (no manually entered splitter geometry)")
    return _save(fig, path)


def plot_inferred_local_geometry(
    trajectories: pd.DataFrame, passages: pd.DataFrame, geometry: AnalysisGeometry, path: Path
) -> Path:
    segments = extract_passage_samples(trajectories, passages, valid_only=True)
    conditions = sorted(trajectories["condition"].unique())
    fig, axes = plt.subplots(1, len(conditions), figsize=(6.4 * len(conditions), 4.5), squeeze=False)
    for col, condition in enumerate(conditions):
        ax = axes[0, col]
        subset = segments[segments["condition"] == condition]
        for _, group in subset.groupby(["robot_id", "passage_id"], sort=False):
            route = group["route_label"].iloc[0]
            ax.plot(group["s"], group["d"], color=ROUTE_COLORS.get(route, "#aaaaaa"), alpha=.30, lw=.8)
        for gate, label, color in [
            (geometry.g0_approach_entry_s, "G0", "#70AD47"),
            (geometry.g1_response_entry_s, "G1", "#FFC000"),
            (geometry.g2_route_s, "G2", "#7030A0"),
            (geometry.g3_recovery_s, "G3", "#5B9BD5"),
        ]:
            ax.axvline(gate, ls="--", lw=1, color=color, label=label)
        core_s = np.linspace(geometry.core_start_s, geometry.core_end_s, 100)
        ax.plot(core_s, geometry.splitter_d(core_s), color="#111111", lw=3, label="effective splitter")
        ax.fill_between(core_s, geometry.splitter_d(core_s)-geometry.ambiguous_half_width_m,
                        geometry.splitter_d(core_s)+geometry.ambiguous_half_width_m,
                        color="#888888", alpha=.18, label="ambiguous band")
        ax.axvspan(geometry.g0_approach_entry_s, geometry.g1_response_entry_s, color="#70AD47", alpha=.08)
        ax.set(title=condition.replace("_", " "), xlabel="local longitudinal s [m]", ylabel="local lateral d [m]",
               xlim=(geometry.g0_approach_entry_s-.05, geometry.g3_recovery_s+.05),
               ylim=(-geometry.gate_half_width_m, geometry.gate_half_width_m))
        ax.grid(alpha=.18)
    handles, labels = axes[0, -1].get_legend_handles_labels()
    unique = dict(zip(labels, handles)); fig.legend(unique.values(), unique.keys(), ncol=4, fontsize=8, loc="lower center")
    fig.suptitle("Data-derived local response geometry and passage labels")
    fig.subplots_adjust(bottom=.18)
    return _save(fig, path)


def plot_wasserstein_profile(diagnostics: pd.DataFrame, geometry: AnalysisGeometry, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    ax.plot(diagnostics["s"], diagnostics["wasserstein_m"], color="#4472C4", lw=2, label="baseline vs changed W1")
    ax.plot(diagnostics["s"], diagnostics["null_threshold_m"], "--", color="#C00000", label="baseline split-null 95%")
    ax.axvspan(geometry.g1_response_entry_s, geometry.g3_recovery_s, color="#FFC000", alpha=.12, label="supported response interval")
    ax.axvspan(geometry.core_start_s, geometry.core_end_s, color="#7030A0", alpha=.10, label="bimodal core")
    ax.set(xlabel="local longitudinal s [m]", ylabel="1-D Wasserstein distance [m]", title="Where a trajectory response becomes detectable")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    return _save(fig, path)


def plot_cross_section(result: SplitterInferenceResult, path: Path) -> Path:
    geometry, crossings = result.geometry, result.crossing_samples
    available_s = np.sort(crossings["s"].unique())
    section_s = float(available_s[np.argmin(np.abs(available_s - geometry.g2_route_s))])
    section = crossings[np.isclose(crossings["s"], section_s)]
    changed = next(name for name in sorted(section["condition"].unique()) if name != "baseline")
    diag = result.section_diagnostics.iloc[(result.section_diagnostics["s"] - geometry.g2_route_s).abs().argmin()]
    fig, ax = plt.subplots(figsize=(7.3, 4.2))
    for index, condition in enumerate(["baseline", changed]):
        values = section[section["condition"] == condition]["d"].to_numpy(float)
        jitter = np.linspace(-.07, .07, len(values)) if len(values) else []
        ax.scatter(values, np.full(len(values), index) + jitter, alpha=.55, s=18,
                   color=CONDITION_COLORS.get(condition), label=condition)
    for center in [diag["cluster_0_center_m"], diag["cluster_1_center_m"]]:
        ax.axvline(center, color="#ED7D31", lw=1.5, ls="--")
    separator = float(geometry.splitter_d(geometry.g2_route_s))
    ax.axvline(separator, color="#111111", lw=2.5, label="K=2 midpoint / fitted separator")
    ax.set_yticks([0, 1], ["baseline", changed]); ax.set_xlabel("lateral crossing position d [m]")
    ax.set_title(f"Representative G2 cross-section: separation={diag['cluster_separation_m']:.3f} m, silhouette={diag['silhouette']:.2f}")
    ax.grid(axis="x", alpha=.2); ax.legend(frameon=False)
    return _save(fig, path)


def plot_route_distribution(distribution: pd.DataFrame, path: Path) -> Path:
    pivot = distribution.pivot(index="condition", columns="route_label", values="route_proportion").reindex(columns=list(ROUTE_COLORS)).fillna(0)
    fig, ax = plt.subplots(figsize=(7.0, 4.2)); bottom = np.zeros(len(pivot))
    for route in pivot:
        ax.bar(pivot.index, pivot[route], bottom=bottom, color=ROUTE_COLORS[route], label=route)
        bottom += pivot[route].to_numpy()
    ax.set(ylabel="valid-passage proportion", ylim=(0, 1), title="Route allocation around the inferred separator")
    ax.grid(axis="y", alpha=.2); ax.legend(frameon=False, ncol=3)
    return _save(fig, path)


def plot_passage_time(per_robot: pd.DataFrame, path: Path) -> Path:
    pivot = per_robot.pivot_table(index="robot_id", columns="condition", values="median_passage_time_s")
    conditions = list(pivot.columns); fig, ax = plt.subplots(figsize=(7.0, 4.3))
    for robot_id, row in pivot.iterrows():
        ax.plot(range(len(conditions)), row[conditions], "o-", color="#999999", alpha=.65, lw=1)
        if pd.notna(row.iloc[-1]): ax.text(len(conditions)-1+.03, row.iloc[-1], f"R{robot_id}", fontsize=7)
    ax.set_xticks(range(len(conditions)), [c.replace("_", " ") for c in conditions])
    ax.set(ylabel="median G0-to-G3 passage time [s]", title="Robot-matched local passage time")
    ax.grid(axis="y", alpha=.2)
    return _save(fig, path)


def plot_online_state(summary: pd.DataFrame, path: Path) -> Path:
    condition = next((c for c in sorted(summary["condition"].unique()) if c != "baseline"), "baseline")
    data = summary[summary["condition"] == condition].sort_values("elapsed_s")
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 7.2), sharex=True)
    axes[0].step(data["elapsed_s"], data["crossing_count_window"], where="post", label="N in 15 s")
    axes[0].step(data["elapsed_s"], data["n_approach"], where="post", label="N in approach")
    for q in (33, 50, 67): axes[1].plot(data["elapsed_s"], data[f"v_s_q{q}_mps"], label=f"q{q}")
    for q in (33, 50, 67): axes[2].plot(data["elapsed_s"], data[f"d_q{q}_m"], label=f"q{q}")
    axes[0].set_ylabel("count"); axes[1].set_ylabel("v_s [m/s]"); axes[2].set_ylabel("d [m]"); axes[2].set_xlabel("run elapsed time [s]")
    for ax in axes: ax.grid(alpha=.18); ax.legend(frameon=False, ncol=3, fontsize=8)
    fig.suptitle(f"Example causal decision state ({condition})")
    return _save(fig, path)


def plot_spatial_speed(profile: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for condition, group in profile.groupby("condition"):
        group = group.sort_values("s_mean_m"); color = CONDITION_COLORS.get(condition)
        ax.plot(group["s_mean_m"], group["median_v_s_mps"], lw=2, color=color, label=condition)
        ax.fill_between(group["s_mean_m"], group["q25_v_s_mps"], group["q75_v_s_mps"], color=color, alpha=.15)
    ax.set(xlabel="local longitudinal s [m]", ylabel="v_s [m/s]", title="Supporting spatial speed diagnostic")
    ax.grid(alpha=.2); ax.legend(frameon=False)
    return _save(fig, path)


def plot_pair_distance(series: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for condition, group in series.groupby("condition"):
        ax.hist(group["minimum_pair_distance_m"], bins=24, density=True, alpha=.35,
                color=CONDITION_COLORS.get(condition), label=condition)
    ax.set(xlabel="minimum robot-center pair distance [m]", ylabel="density", title="Supporting spacing-proxy distribution")
    ax.legend(frameon=False); ax.grid(axis="x", alpha=.2)
    return _save(fig, path)


def generate_all_plots(
    trajectories: pd.DataFrame, manifest: pd.DataFrame, passages: pd.DataFrame,
    online_summary: pd.DataFrame, evaluation: dict[str, pd.DataFrame],
    inference: SplitterInferenceResult, output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    geometry = inference.geometry
    return {
        "full_trajectories": plot_full_trajectories(trajectories, manifest, output_dir / "01_full_trajectories.png"),
        "inferred_local_geometry": plot_inferred_local_geometry(trajectories, passages, geometry, output_dir / "02_inferred_local_geometry.png"),
        "wasserstein_profile": plot_wasserstein_profile(inference.section_diagnostics, geometry, output_dir / "03_wasserstein_profile.png"),
        "cross_section": plot_cross_section(inference, output_dir / "04_cross_section_kmeans.png"),
        "route_distribution": plot_route_distribution(evaluation["route_distribution"], output_dir / "05_route_distribution.png"),
        "passage_time": plot_passage_time(evaluation["robot_passage_time"], output_dir / "06_passage_time.png"),
        "online_state": plot_online_state(online_summary, output_dir / "07_online_state.png"),
        "spatial_speed": plot_spatial_speed(evaluation["spatial_speed_profile"], output_dir / "08_spatial_speed_profile.png"),
        "pair_distance": plot_pair_distance(evaluation["pair_distance_timeseries"], output_dir / "09_pair_distance.png"),
    }
