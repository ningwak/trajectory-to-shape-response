"""Command-line entry point for dataset loading and preprocessing."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .io import build_run_manifest, load_dataset
from .preprocessing import PreprocessingConfig, preprocess_dataset


DEFAULT_COLAB_DATA_ROOT = Path("/content/drive/MyDrive/Roboflow-dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load and preprocess the RE-FLOW RoboFlow trajectory dataset."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_COLAB_DATA_ROOT,
        help=f"Directory containing the two raw run folders (default: {DEFAULT_COLAB_DATA_ROOT}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <data-root>/processed.",
    )
    parser.add_argument("--sample-rate-hz", type=float, default=20.0)
    parser.add_argument("--max-hold-gap-s", type=float, default=0.25)
    parser.add_argument("--velocity-window-s", type=float, default=0.25)
    parser.add_argument(
        "--alignment",
        choices=("intersection", "union"),
        default="intersection",
    )
    parser.add_argument(
        "--format",
        choices=("csv", "parquet"),
        default="csv",
        dest="output_format",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_root = args.data_root.expanduser().resolve()
    output_dir = (args.output_dir or data_root / "processed").expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    config = PreprocessingConfig(
        sample_rate_hz=args.sample_rate_hz,
        max_hold_gap_s=args.max_hold_gap_s,
        velocity_window_s=args.velocity_window_s,
        alignment=args.alignment,
    )
    runs = load_dataset(data_root)
    trajectories, quality = preprocess_dataset(runs, config)
    manifest = build_run_manifest(runs)

    if args.output_format == "csv":
        trajectories.to_csv(output_dir / "trajectory_samples.csv", index=False)
        quality.to_csv(output_dir / "quality_report.csv", index=False)
        manifest.to_csv(output_dir / "run_manifest.csv", index=False)
    else:
        trajectories.to_parquet(output_dir / "trajectory_samples.parquet", index=False)
        quality.to_parquet(output_dir / "quality_report.parquet", index=False)
        manifest.to_parquet(output_dir / "run_manifest.parquet", index=False)

    with (output_dir / "preprocessing_config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    print(f"Loaded {len(runs)} runs and {quality['robot_id'].count()} robot files.")
    print(f"Wrote {len(trajectories):,} aligned samples to {output_dir}.")


if __name__ == "__main__":
    main()
