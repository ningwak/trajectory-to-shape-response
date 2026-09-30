"""End-to-end JSON-only trajectory-to-shape-response pipeline."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .geometry import save_geometry, world_to_corridor
from .io import build_run_manifest, load_dataset, sort_by_condition
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
    path.write_text(
        json.dumps(payload, indent=2, default=str, allow_nan=False), encoding="utf-8"
    )


def _json_records(table: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn table rows into strict JSON, representing unavailable metrics as null."""

    return json.loads(table.to_json(orient="records", double_precision=15))


def _summary(
    result_tables: dict[str, pd.DataFrame],
    quality: pd.DataFrame,
    passages: pd.DataFrame,
    online_state: pd.DataFrame,
) -> dict[str, Any]:
    """Collect the outcome and a descriptive history mean of the causal state.

    Each online-state row is one equally spaced decision timestamp. The mean
    is taken across the complete recorded history of each condition, including
    zeros for arrival/occupancy counts. Missing quantiles and the undefined
    pre-first-arrival clock are omitted from their respective means; the
    number of usable timestamps is reported alongside them.
    """

    state_columns = [
        "crossing_count_window",
        "time_since_last_crossing_s",
        "n_approach",
        "v_s_q33_mps",
        "v_s_q50_mps",
        "v_s_q67_mps",
        "d_q33_m",
        "d_q50_m",
        "d_q67_m",
    ]
    history = online_state.groupby("condition", as_index=False)[state_columns].mean()
    history = history.rename(columns={name: f"{name}_mean" for name in state_columns})
    support = online_state.groupby("condition", as_index=False).agg(
        state_updates=("time", "size"),
        run_count=("run_name", "nunique"),
        quantiles_valid_updates=("quantiles_valid", "sum"),
        last_crossing_clock_valid_updates=("time_since_last_crossing_s", "count"),
    )
    history = history.merge(support, on="condition")
    history["quantiles_valid_fraction"] = (
        history["quantiles_valid_updates"] / history["state_updates"]
    )
    history = sort_by_condition(history)

    # An ambiguous *valid* passage has a measured route score near the fitted
    # separator. Its passage_id counts that robot's G0 entries from one in its
    # run, including any earlier entry later rejected as an invalid passage.
    ambiguous = passages[
        passages["valid"]
        & (passages["condition"] != "baseline")
        & (passages["route_label"] == "ambiguous")
    ][
        [
            "condition",
            "run_name",
            "robot_id",
            "passage_id",
            "t_g0",
            "route_score_m",
            "passage_time_s",
        ]
    ].sort_values(["condition", "run_name", "t_g0"])
    ambiguous = ambiguous.copy()
    ambiguous["t_g0"] = ambiguous["t_g0"].astype(str)

    route_counts = result_tables["route_distribution"]
    route_counts = route_counts[route_counts["condition"] != "baseline"]

    return {
        "robot_json_files": int(len(quality)),
        "valid_passages": int(passages["valid"].sum()),
        "invalid_passages": int((~passages["valid"]).sum()),
        "decision_state_components": 9,
        "evaluation_components": 6,
        "online_state_history_means": _json_records(history),
        "compact_evaluation": _json_records(result_tables["compact_evaluation"]),
        "passage_time_summary": _json_records(result_tables["passage_time_summary"]),
        "route_counts_changed_condition": _json_records(route_counts),
        "passage_id_definition": (
            "1-based G0 entry count for one robot in one run; "
            "it may include invalid entries"
        ),
        "ambiguous_passages": _json_records(ambiguous),
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
    # The upper nominal edge gives a reproducible horizontal analysis frame.
    # The video and full-path plot motivate analysing this leg, but neither a
    # video-derived coordinate nor a physical barrier annotation enters here.
    # The separator and response interval are inferred from *trajectory*
    # crossings afterward; the JSON corners do not reveal a barrier location.
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
    summary = _summary(evaluation, quality, passages, online_summary)
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
