"""Load and validate RoboFlow robot trajectory files."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

REQUIRED_TOP_LEVEL_KEYS = {
    "robot_id",
    "function",
    "run_id",
    "run_timestamp",
    "rectangle_corners",
    "corridor_width_m",
    "trajectory",
}
REQUIRED_SAMPLE_KEYS = {"time", "x", "y", "theta"}


@dataclass(frozen=True)
class RobotTrajectory:
    """One robot trajectory and the metadata stored in its JSON file."""

    source_file: Path
    run_name: str
    condition: str
    robot_id: int
    source_run_id: str
    run_timestamp: str
    function: str
    rectangle_corners: tuple[tuple[float, float], ...]
    corridor_width_m: float
    samples: pd.DataFrame


@dataclass(frozen=True)
class RunData:
    """All robot files belonging to one experiment directory."""

    directory: Path
    run_name: str
    condition: str
    robots: tuple[RobotTrajectory, ...]


def infer_condition(run_name: str) -> str:
    """Infer a stable condition label from an experiment directory name."""

    name = run_name.lower()
    if "baseline" in name or "no_barrier" in name:
        return "baseline"

    match = re.search(r"barrier[_-]?(v\d+)", name)
    if match:
        return f"barrier_{match.group(1)}"
    if "barrier" in name:
        return "barrier"
    return run_name


def condition_order(values: Iterable[str]) -> list[str]:
    """Return condition names with baseline first and changed runs afterwards."""

    return sorted(set(values), key=lambda name: (name != "baseline", name))


def sort_by_condition(table: pd.DataFrame, *within_condition: str) -> pd.DataFrame:
    """Sort a result table as baseline, then barrier, then requested columns."""

    order = {
        name: index for index, name in enumerate(condition_order(table["condition"]))
    }
    return (
        table.assign(_condition_order=table["condition"].map(order))
        .sort_values(["_condition_order", *within_condition], kind="stable")
        .drop(columns="_condition_order")
        .reset_index(drop=True)
    )


def _parse_utc_timestamps(values: pd.Series, source_file: Path) -> pd.Series:
    """Parse mixed ISO-like timestamps and normalize them to UTC."""

    try:
        parsed = pd.to_datetime(values, utc=True, format="mixed", errors="raise")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp in {source_file}: {exc}") from exc
    return pd.Series(parsed, index=values.index, name="time")


def _validate_top_level(payload: dict[str, Any], source_file: Path) -> None:
    """Check required metadata and ensure that trajectory data are present."""

    missing = REQUIRED_TOP_LEVEL_KEYS.difference(payload)
    if missing:
        raise ValueError(f"{source_file} is missing top-level keys: {sorted(missing)}")
    if not isinstance(payload["trajectory"], list) or not payload["trajectory"]:
        raise ValueError(f"{source_file} contains an empty or invalid trajectory list")


def _parse_corners(
    raw_corners: Iterable[dict[str, Any]], source_file: Path
) -> tuple[tuple[float, float], ...]:
    """Convert the four nominal JSON rectangle corners to numeric tuples."""

    corners: list[tuple[float, float]] = []
    for corner in raw_corners:
        if not isinstance(corner, dict) or not {"x", "y"}.issubset(corner):
            raise ValueError(f"Invalid rectangle corner in {source_file}")
        corners.append((float(corner["x"]), float(corner["y"])))
    if len(corners) != 4:
        raise ValueError(
            f"Expected four rectangle corners in {source_file}, found {len(corners)}"
        )
    return tuple(corners)


def load_robot_file(path: str | Path, condition: str | None = None) -> RobotTrajectory:
    """Load one JSON file into a validated, time-sorted trajectory object."""

    source_file = Path(path).expanduser().resolve()
    with source_file.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {source_file}")
    _validate_top_level(payload, source_file)

    records = payload["trajectory"]
    first_missing = [
        index
        for index, row in enumerate(records)
        if not REQUIRED_SAMPLE_KEYS.issubset(row)
    ]
    if first_missing:
        raise ValueError(
            f"Trajectory samples are missing required fields in {source_file}; first index: {first_missing[0]}"
        )

    samples = pd.DataFrame.from_records(records, columns=["time", "x", "y", "theta"])
    samples["time"] = _parse_utc_timestamps(samples["time"], source_file)
    for column in ("x", "y", "theta"):
        samples[column] = pd.to_numeric(samples[column], errors="raise").astype(float)
        if not samples[column].map(math.isfinite).all():
            raise ValueError(
                f"Non-finite values found in column '{column}' of {source_file}"
            )

    samples = (
        samples.sort_values("time", kind="stable")
        .drop_duplicates(subset="time", keep="last")
        .reset_index(drop=True)
    )
    if not samples["time"].is_monotonic_increasing:
        raise ValueError(f"Timestamps could not be ordered in {source_file}")

    run_name = source_file.parent.name
    return RobotTrajectory(
        source_file=source_file,
        run_name=run_name,
        condition=condition or infer_condition(run_name),
        robot_id=int(payload["robot_id"]),
        source_run_id=str(payload["run_id"]),
        run_timestamp=str(payload["run_timestamp"]),
        function=str(payload["function"]),
        rectangle_corners=_parse_corners(payload["rectangle_corners"], source_file),
        corridor_width_m=float(payload["corridor_width_m"]),
        samples=samples,
    )


def _check_run_consistency(robots: tuple[RobotTrajectory, ...], run_dir: Path) -> None:
    """Verify that robot files in one directory describe the same experiment."""

    robot_ids = [robot.robot_id for robot in robots]
    if len(set(robot_ids)) != len(robot_ids):
        raise ValueError(f"Duplicate robot IDs in {run_dir}: {robot_ids}")

    reference = robots[0]
    for robot in robots[1:]:
        if robot.function != reference.function:
            raise ValueError(f"Inconsistent experiment function in {run_dir}")
        if robot.rectangle_corners != reference.rectangle_corners:
            raise ValueError(f"Inconsistent rectangle geometry in {run_dir}")
        if not math.isclose(robot.corridor_width_m, reference.corridor_width_m):
            raise ValueError(f"Inconsistent corridor width in {run_dir}")


def load_run(run_dir: str | Path) -> RunData:
    """Load every robot JSON file in one experiment directory."""

    directory = Path(run_dir).expanduser().resolve()
    json_files = sorted(
        path for path in directory.glob("R*.json") if re.match(r"R\d+_", path.name)
    )
    if not json_files:
        raise FileNotFoundError(f"No JSON trajectory files found in {directory}")

    condition = infer_condition(directory.name)
    robots = tuple(load_robot_file(path, condition=condition) for path in json_files)
    robots = tuple(sorted(robots, key=lambda robot: robot.robot_id))
    _check_run_consistency(robots, directory)

    return RunData(
        directory=directory,
        run_name=directory.name,
        condition=condition,
        robots=robots,
    )


def discover_run_directories(dataset_root: str | Path) -> tuple[Path, ...]:
    """Find directories containing robot JSON files below the dataset root."""

    root = Path(dataset_root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(
            f"Dataset root does not exist or is not a directory: {root}"
        )
    robot_files = (
        path for path in root.rglob("R*.json") if re.match(r"R\d+_", path.name)
    )
    directories = sorted({path.parent for path in robot_files})
    if not directories:
        raise FileNotFoundError(f"No JSON trajectory files found below {root}")
    return tuple(directories)


def load_dataset(dataset_root: str | Path) -> tuple[RunData, ...]:
    """Discover and load all experiment runs below the dataset root."""

    return tuple(
        load_run(directory) for directory in discover_run_directories(dataset_root)
    )


def build_run_manifest(runs: Iterable[RunData]) -> pd.DataFrame:
    """Create a compact table describing the discovered experiment runs."""

    rows: list[dict[str, Any]] = []
    for run in runs:
        reference = run.robots[0]
        rows.append(
            {
                "condition": run.condition,
                "run_name": run.run_name,
                "run_directory": str(run.directory),
                "robot_count": len(run.robots),
                "robot_ids": ",".join(str(robot.robot_id) for robot in run.robots),
                "function": reference.function,
                "corridor_width_m": reference.corridor_width_m,
                "rectangle_corners": json.dumps(reference.rectangle_corners),
            }
        )
    return sort_by_condition(pd.DataFrame(rows), "run_name")
