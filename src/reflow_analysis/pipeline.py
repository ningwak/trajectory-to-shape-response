"""End-to-end JSON-only trajectory-to-shape-response pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .geometry import save_geometry, world_to_corridor
from .io import build_run_manifest, load_dataset
from .metrics import MetricConfig, build_evaluation_tables
from .online_state import OnlineStateConfig, extract_online_state
from .passages import PassageConfig, detect_passages
from .plots import generate_all_plots
from .preprocessing import PreprocessingConfig, preprocess_dataset
from .splitter_inference import (
    SplitterInferenceConfig,
    SplitterInferenceResult,
    infer_effective_splitter,
)


@dataclass(frozen=True)
class FullAnalysisConfig:
    """Bundle the small number of configurations used by the main pipeline."""

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    splitter: SplitterInferenceConfig = field(default_factory=SplitterInferenceConfig)
    passages: PassageConfig = field(default_factory=PassageConfig)
    online_state: OnlineStateConfig = field(default_factory=OnlineStateConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)


@dataclass(frozen=True)
class AnalysisResult:
    """Keep exported results and useful in-memory diagnostics for notebooks."""

    trajectories: pd.DataFrame
    quality: pd.DataFrame
    manifest: pd.DataFrame
    inference: SplitterInferenceResult
    passages: pd.DataFrame
    online_state_summary: pd.DataFrame
    evaluation_tables: dict[str, pd.DataFrame]
    figure_paths: dict[str, Path]
    summary: dict[str, Any]


def _write_json(payload: dict[str, Any], path: Path) -> None:
    """Write a JSON configuration or summary with parent-directory creation."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _summary(
    result_tables: dict[str, pd.DataFrame],
    quality: pd.DataFrame,
    passages: pd.DataFrame,
) -> dict[str, Any]:
    """Create the concise machine-readable run summary."""

    return {
        "robot_json_files": int(len(quality)),
        "valid_passages": int(passages["valid"].sum()),
        "invalid_passages": int((~passages["valid"]).sum()),
        "decision_state_components": 9,
        "evaluation_components": 6,
        "compact_evaluation": result_tables["compact_evaluation"].to_dict(
            orient="records"
        ),
    }


def run_full_analysis(
    data_root: str | Path,
    output_dir: str | Path,
    config: FullAnalysisConfig | None = None,
) -> AnalysisResult:
    """Run the main pipeline. Discovery is restricted to ``R*.json`` files."""

    config = config or FullAnalysisConfig()
    output = Path(output_dir).expanduser().resolve()
    tables_dir, figures_dir, config_dir = (
        output / "tables",
        output / "figures",
        output / "config",
    )
    for directory in (tables_dir, figures_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    runs = load_dataset(data_root)
    trajectories, quality = preprocess_dataset(runs, config.preprocessing)
    manifest = build_run_manifest(runs)
    # The upper nominal edge gives a reproducible horizontal analysis frame;
    # these corners do not supply (or imply) a physical barrier location.
    inferred = infer_effective_splitter(
        trajectories, runs[0].robots[0].rectangle_corners, config.splitter
    )
    geometry = inferred.geometry
    trajectories = world_to_corridor(trajectories, geometry)
    passages = detect_passages(trajectories, geometry, config.passages)
    online_summary = extract_online_state(
        trajectories, passages, geometry, config.online_state
    )
    evaluation = build_evaluation_tables(
        trajectories, passages, geometry, config.metrics
    )

    # Keep the default output small enough to inspect during interview preparation.
    base_tables = {
        "quality_report": quality,
        "section_diagnostics": inferred.section_diagnostics,
        "passages": passages,
        "online_state_summary": online_summary,
        "compact_evaluation": evaluation["compact_evaluation"],
        "robot_passage_time": evaluation["robot_passage_time"],
    }
    for name, table in base_tables.items():
        table.to_csv(tables_dir / f"{name}.csv", index=False)
    save_geometry(geometry, config_dir / "inferred_geometry.json")
    _write_json(asdict(config), config_dir / "analysis_config.json")
    figures = generate_all_plots(
        trajectories, manifest, passages, evaluation, inferred, figures_dir
    )
    summary = _summary(evaluation, quality, passages)
    _write_json(summary, output / "analysis_summary.json")
    return AnalysisResult(
        trajectories,
        quality,
        manifest,
        inferred,
        passages,
        online_summary,
        evaluation,
        figures,
        summary,
    )
