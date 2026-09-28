"""Trajectory-to-shape-response analysis for the RE-FLOW dataset."""

from .geometry import AnalysisGeometry, world_to_corridor
from .io import RobotTrajectory, RunData, load_dataset, load_robot_file, load_run
from .metrics import MetricConfig, build_evaluation_tables
from .online_state import OnlineStateConfig, extract_online_state
from .passages import PassageConfig, detect_passages
from .pipeline import FullAnalysisConfig, run_full_analysis
from .preprocessing import PreprocessingConfig, preprocess_dataset, preprocess_run
from .splitter_inference import SplitterInferenceConfig, infer_effective_splitter

__all__ = [
    "AnalysisGeometry",
    "FullAnalysisConfig",
    "MetricConfig",
    "OnlineStateConfig",
    "PassageConfig",
    "PreprocessingConfig",
    "RobotTrajectory",
    "RunData",
    "SplitterInferenceConfig",
    "build_evaluation_tables",
    "detect_passages",
    "extract_online_state",
    "infer_effective_splitter",
    "load_dataset",
    "load_robot_file",
    "load_run",
    "preprocess_dataset",
    "preprocess_run",
    "run_full_analysis",
    "world_to_corridor",
]
