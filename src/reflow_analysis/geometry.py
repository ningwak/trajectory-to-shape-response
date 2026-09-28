"""Data-derived local flow coordinates and inferred analysis landmarks."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AnalysisGeometry:
    """Local flow frame and four virtual gates inferred from trajectories.

    ``s`` follows the baseline flow and ``d`` is its left normal. The line
    ``d = splitter_intercept_m + splitter_slope * s`` is an *effective*
    trajectory separator; it is not presented as a measured physical pose.
    """

    origin_x: float = 0.0
    origin_y: float = 0.0
    tangent_x: float = 1.0
    tangent_y: float = 0.0
    normal_x: float = 0.0
    normal_y: float = 1.0
    g0_approach_entry_s: float = -0.75
    g1_response_entry_s: float = -0.40
    g2_route_s: float = 0.0
    g3_recovery_s: float = 0.50
    core_start_s: float = -0.20
    core_end_s: float = 0.20
    splitter_intercept_m: float = 0.0
    splitter_slope: float = 0.0
    ambiguous_half_width_m: float = 0.04
    gate_half_width_m: float = 0.65
    inference_method: str = "data-derived effective separator"

    def __post_init__(self) -> None:
        """Reject inconsistent local frames and unordered inferred landmarks."""

        tnorm = math.hypot(self.tangent_x, self.tangent_y)
        nnorm = math.hypot(self.normal_x, self.normal_y)
        dot = self.tangent_x * self.normal_x + self.tangent_y * self.normal_y
        if not math.isclose(tnorm, 1.0, abs_tol=1e-6):
            raise ValueError("The tangent vector must have unit length")
        if not math.isclose(nnorm, 1.0, abs_tol=1e-6):
            raise ValueError("The normal vector must have unit length")
        if not math.isclose(dot, 0.0, abs_tol=1e-6):
            raise ValueError("The tangent and normal vectors must be orthogonal")
        if not (
            self.g0_approach_entry_s
            < self.g1_response_entry_s
            <= self.core_start_s
            <= self.g2_route_s
            <= self.core_end_s
            <= self.g3_recovery_s
        ):
            raise ValueError("Virtual gates and inferred core must be ordered")
        if self.gate_half_width_m <= 0 or self.ambiguous_half_width_m < 0:
            raise ValueError("Gate width parameters must be non-negative")

    def splitter_d(self, s) -> np.ndarray:
        """Evaluate the fitted effective-separator line at local coordinate s."""

        values = np.asarray(s, dtype=float)
        return self.splitter_intercept_m + self.splitter_slope * values


def world_to_corridor(
    samples: pd.DataFrame, geometry: AnalysisGeometry
) -> pd.DataFrame:
    """Add local longitudinal/lateral position and velocity columns."""

    missing = {"x", "y"}.difference(samples.columns)
    if missing:
        raise ValueError(f"Missing world-coordinate columns: {sorted(missing)}")
    result = samples.copy()
    dx = result["x"] - geometry.origin_x
    dy = result["y"] - geometry.origin_y
    result["s"] = dx * geometry.tangent_x + dy * geometry.tangent_y
    result["d"] = dx * geometry.normal_x + dy * geometry.normal_y
    if {"vx", "vy"}.issubset(result.columns):
        result["v_s"] = (
            result["vx"] * geometry.tangent_x + result["vy"] * geometry.tangent_y
        )
        result["v_d"] = (
            result["vx"] * geometry.normal_x + result["vy"] * geometry.normal_y
        )
    return result


def save_geometry(geometry: AnalysisGeometry, path: str | Path) -> Path:
    """Serialize inferred geometry to a readable JSON file."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(asdict(geometry), indent=2), encoding="utf-8")
    return output
