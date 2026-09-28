"""End-to-end JSON-only trajectory-to-shape-response pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .geometry import AnalysisGeometry, save_geometry, world_to_corridor
from .io import build_run_manifest, load_dataset
from .metrics import MetricConfig, build_evaluation_tables
from .online_state import OnlineStateConfig, extract_online_state
from .passages import PassageConfig, detect_passages
from .plots import generate_all_plots
from .preprocessing import PreprocessingConfig, preprocess_dataset
from .splitter_inference import (
    SplitterInferenceConfig, SplitterInferenceResult, infer_effective_splitter,
)


@dataclass(frozen=True)
class FullAnalysisConfig:
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    splitter: SplitterInferenceConfig = field(default_factory=SplitterInferenceConfig)
    passages: PassageConfig = field(default_factory=PassageConfig)
    online_state: OnlineStateConfig = field(default_factory=OnlineStateConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)
    geometry_override: AnalysisGeometry | None = None


@dataclass(frozen=True)
class AnalysisResult:
    trajectories: pd.DataFrame
    quality: pd.DataFrame
    manifest: pd.DataFrame
    inference: SplitterInferenceResult
    passages: pd.DataFrame
    online_robot_states: pd.DataFrame
    crossing_events: pd.DataFrame
    online_state_summary: pd.DataFrame
    evaluation_tables: dict[str, pd.DataFrame]
    figure_paths: dict[str, Path]
    summary: dict[str, Any]


def _write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _definition_tables() -> dict[str, pd.DataFrame]:
    state = pd.DataFrame(
        [
            ("crossing_count_window", "scalar", "G0 crossings during the past 15 s"),
            ("time_since_last_crossing_s", "scalar", "Recency of the last G0 crossing"),
            ("n_approach", "scalar", "Valid robots currently between G0 and G1"),
            ("v_s_q33/q50/q67_mps", "3-vector", "Longitudinal speeds at recent G0 crossings"),
            ("d_q33/q50/q67_m", "3-vector", "Lateral positions at recent G0 crossings"),
        ], columns=["state_block", "shape", "meaning"]
    )
    evaluation = pd.DataFrame(
        [
            ("inner/outer/ambiguous", "3-vector", "Route proportions"),
            ("relative_passage_time", "scalar", "Median G0-G3 time divided by baseline"),
            ("median_excess_path_length_m", "scalar", "Path length beyond straight G0-G3 distance"),
            ("pair_distance_q10_m", "scalar", "10th percentile of timestamp-wise nearest-pair distance"),
        ], columns=["metric_block", "shape", "meaning"]
    )
    return {"decision_state_definition": state, "evaluation_definition": evaluation}


def _summary(result_tables: dict[str, pd.DataFrame], quality: pd.DataFrame, passages: pd.DataFrame) -> dict[str, Any]:
    return {
        "robot_json_files": int(len(quality)),
        "valid_passages": int(passages["valid"].sum()),
        "invalid_passages": int((~passages["valid"]).sum()),
        "decision_state_components": 9,
        "evaluation_components": 6,
        "compact_evaluation": result_tables["compact_evaluation"].to_dict(orient="records"),
    }


def run_full_analysis(
    data_root: str | Path, output_dir: str | Path, config: FullAnalysisConfig | None = None
) -> AnalysisResult:
    """Run the main pipeline. Discovery is restricted to ``R*.json`` files."""

    config = config or FullAnalysisConfig()
    output = Path(output_dir).expanduser().resolve()
    tables_dir, figures_dir, config_dir = output / "tables", output / "figures", output / "config"
    for directory in (tables_dir, figures_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    runs = load_dataset(data_root)
    trajectories, quality = preprocess_dataset(runs, config.preprocessing)
    manifest = build_run_manifest(runs)
    inferred = infer_effective_splitter(trajectories, config.splitter)
    if config.geometry_override is not None:
        inferred = SplitterInferenceResult(
            config.geometry_override, inferred.section_diagnostics,
            inferred.crossing_samples, inferred.occupancy_component, inferred.summary,
        )
    geometry = inferred.geometry
    trajectories = world_to_corridor(trajectories, geometry)
    passages = detect_passages(trajectories, geometry, config.passages)
    robot_states, crossing_events, online_summary = extract_online_state(
        trajectories, passages, geometry, config.online_state
    )
    evaluation = build_evaluation_tables(trajectories, passages, geometry, config.metrics)
    evaluation.update(_definition_tables())

    base_tables = {
        "trajectory_samples": trajectories,
        "quality_report": quality,
        "run_manifest": manifest,
        "splitter_inference_summary": inferred.summary,
        "section_diagnostics": inferred.section_diagnostics,
        "section_crossing_samples": inferred.crossing_samples,
        "passages": passages,
        "online_robot_states": robot_states,
        "g0_crossing_events": crossing_events,
        "online_state_summary": online_summary,
    }
    for name, table in {**base_tables, **evaluation}.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    save_geometry(geometry, config_dir / "inferred_geometry.json")
    _write_json(asdict(config), config_dir / "analysis_config.json")
    figures = generate_all_plots(
        trajectories, manifest, passages, online_summary, evaluation, inferred, figures_dir
    )
    summary = _summary(evaluation, quality, passages)
    _write_json(summary, output / "analysis_summary.json")
    return AnalysisResult(
        trajectories, quality, manifest, inferred, passages, robot_states,
        crossing_events, online_summary, evaluation, figures, summary,
    )
