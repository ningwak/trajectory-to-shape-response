"""Command-line interface for the complete six-table/six-figure analysis.

The CLI exposes only settings that are plausible to adjust during data loading
or interview preparation.  Less frequently changed algorithm thresholds remain
visible in their module configuration dataclasses and are serialized with each
run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .metrics import MetricConfig
from .online_state import OnlineStateConfig
from .passages import PassageConfig
from .pipeline import FullAnalysisConfig, run_full_analysis
from .preprocessing import PreprocessingConfig
from .splitter_inference import SplitterInferenceConfig


def build_parser() -> argparse.ArgumentParser:
    """Define explicit input/output paths and the main temporal settings."""

    parser = argparse.ArgumentParser(
        description="Infer an effective splitter and evaluate its flow response."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--max-hold-gap-s", type=float, default=0.25)
    parser.add_argument("--velocity-window-s", type=float, default=0.25)
    parser.add_argument("--state-rate-hz", type=float, default=2.0)
    parser.add_argument("--crossing-window-s", type=float, default=15.0)
    parser.add_argument("--maximum-passage-s", type=float, default=15.0)
    return parser


def _print_table(title: str, table: pd.DataFrame, digits: int = 3) -> None:
    """Print DataFrames consistently without changing the saved CSV precision."""

    print(f"\n{title}\n{'-' * len(title)}")
    print(table.round(digits).to_string(index=False))


def main() -> None:
    """Translate CLI arguments to configuration, run, and print key evidence."""

    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (
        (args.output_dir or data_root / "analysis_output").expanduser().resolve()
    )
    # Constructing the nested configuration here keeps the CLI thin: every
    # numerical method remains implemented in its domain module.
    config = FullAnalysisConfig(
        preprocessing=PreprocessingConfig(
            sample_rate_hz=args.sample_rate_hz,
            max_hold_gap_s=args.max_hold_gap_s,
            velocity_window_s=args.velocity_window_s,
            alignment="intersection",
        ),
        splitter=SplitterInferenceConfig(),
        passages=PassageConfig(maximum_duration_s=args.maximum_passage_s),
        online_state=OnlineStateConfig(
            state_rate_hz=args.state_rate_hz, crossing_window_s=args.crossing_window_s
        ),
        metrics=MetricConfig(),
    )
    result = run_full_analysis(data_root, output_dir, config)
    geometry_rows = pd.DataFrame(
        [
            ("G0 approach entry", result.inference.geometry.g0_approach_entry_s),
            ("G1 response onset", result.inference.geometry.g1_response_entry_s),
            ("G2 route section", result.inference.geometry.g2_route_s),
            ("G3 analysis exit", result.inference.geometry.g3_recovery_s),
            ("core start", result.inference.geometry.core_start_s),
            ("core end", result.inference.geometry.core_end_s),
            ("ambiguity half-width", result.inference.geometry.ambiguous_half_width_m),
        ],
        columns=["landmark", "s_or_width_m"],
    )
    _print_table("Data quality", result.quality, digits=4)
    _print_table("Inferred effective splitter and gates", geometry_rows)
    # G2 may lie between grid coordinates, so show the stored cross-section
    # nearest to it as one compact piece of splitter evidence.
    representative = result.inference.section_diagnostics.iloc[
        (
            result.inference.section_diagnostics["s"]
            - result.inference.geometry.g2_route_s
        )
        .abs()
        .argmin()
    ]
    diagnostic_columns = [
        "s",
        "baseline_n",
        "changed_n",
        "wasserstein_m",
        "null_threshold_m",
        "cluster_0_center_m",
        "cluster_1_center_m",
        "cluster_separation_m",
        "silhouette",
        "minimum_cluster_n",
    ]
    _print_table(
        "Representative G2 cross-section diagnostic",
        pd.DataFrame([{name: representative[name] for name in diagnostic_columns}]),
    )
    _print_table(
        "Compact evaluation (6 numerical entries per condition)",
        result.evaluation_tables["compact_evaluation"],
    )
    _print_table(
        "Per-robot passage time and path length",
        result.evaluation_tables["robot_passage_time"],
    )
    _print_table(
        "Invalid / retained passage counts",
        result.evaluation_tables["invalid_passage_summary"],
    )
    print(f"\nSaved six focused CSV tables and six figures to: {output_dir}")


if __name__ == "__main__":
    main()
