"""Connect the analysis modules into one reproducible JSON-only pipeline.

This is the best source file to read first after the pseudocode document.  It
contains almost no numerical method of its own; its job is to make the order of
operations explicit and to keep file writing at the outer edge of the program.

The central data flow is:

raw JSON -> causal trajectories -> inferred local geometry -> passages
         -> online state + post-action metrics -> tables and figures

The videos supplied with the interview dataset are deliberately absent from
this flow.  Data discovery in :mod:`reflow_analysis.io` only matches robot JSON
filenames.
"""

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
    """Group the five stage-specific configuration objects in one place.

    Each numerical threshold still belongs to the module that uses it.  This
    wrapper simply lets a notebook or CLI pass one reproducible configuration
    through the complete pipeline.
    """

    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    splitter: SplitterInferenceConfig = field(default_factory=SplitterInferenceConfig)
    passages: PassageConfig = field(default_factory=PassageConfig)
    online_state: OnlineStateConfig = field(default_factory=OnlineStateConfig)
    metrics: MetricConfig = field(default_factory=MetricConfig)


@dataclass(frozen=True)
class AnalysisResult:
    """Return the important outputs without exposing every temporary array.

    The six CSV tables are only a compact on-disk view.  This object also keeps
    the transformed trajectories, cross-sectional inference evidence, and plot
    paths available to an interactive notebook.
    """

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
    """Write readable JSON and create only the required parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _summary(
    result_tables: dict[str, pd.DataFrame],
    quality: pd.DataFrame,
    passages: pd.DataFrame,
) -> dict[str, Any]:
    """Create a small index of run size and final compact outcomes.

    This is not another scientific result table.  It is a convenient entry
    point for checking how many files/passages were retained and for reading
    the two final condition rows without opening several CSV files.
    """

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
    """Run every main analysis stage and write the compact deliverables.

    Important ordering dependency: splitter inference operates in the world
    frame and returns the local frame.  Only after that step can the complete
    trajectory table be augmented with ``s``, ``d``, ``v_s``, and ``v_d``.
    Passage, state, and metric functions then share exactly the same geometry.
    """

    config = config or FullAnalysisConfig()
    output = Path(output_dir).expanduser().resolve()
    tables_dir, figures_dir, config_dir = (
        output / "tables",
        output / "figures",
        output / "config",
    )
    for directory in (tables_dir, figures_dir, config_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # Stage A: parse and causally preprocess the two experimental runs.  The
    # quality table records whether the chosen hold/velocity windows are well
    # supported; validity flags stay separate from physical state variables.
    runs = load_dataset(data_root)
    trajectories, quality = preprocess_dataset(runs, config.preprocessing)
    manifest = build_run_manifest(runs)

    # Stages B/C: infer a response-centred frame and effective separator from
    # the baseline/barrier trajectory difference.  No barrier coordinate is
    # read from the JSON metadata.
    inferred = infer_effective_splitter(trajectories, config.splitter)
    geometry = inferred.geometry
    trajectories = world_to_corridor(trajectories, geometry)

    # Stage C continued: turn a cyclic trajectory log into repeated, structured
    # G0-to-G3 passages, then assign barrier routes from the stable core.
    passages = detect_passages(trajectories, geometry, config.passages)

    # Stages D/E intentionally branch here.  The online summary is causal and
    # intended as future decision-agent input.  Evaluation metrics use completed
    # passages and are intended to judge an already executed geometry.
    online_summary = extract_online_state(
        trajectories, passages, geometry, config.online_state
    )
    evaluation = build_evaluation_tables(
        trajectories, passages, geometry, config.metrics
    )

    # Keep the default output small enough to inspect during interview
    # preparation.  Large aligned samples and internal crossing events are
    # reproducible from the raw JSON and therefore are not duplicated as CSVs.
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
    # Save numerical configuration and inferred geometry next to the results so
    # that a figure can be traced back to the thresholds/gates that produced it.
    save_geometry(geometry, config_dir / "inferred_geometry.json")
    _write_json(asdict(config), config_dir / "analysis_config.json")
    # Plotting consumes already-computed tables.  It does not re-estimate the
    # splitter or silently apply a second set of thresholds.
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
