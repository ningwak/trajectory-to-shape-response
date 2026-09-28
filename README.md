# Trajectory to Shape Response

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

This repository develops a compact and auditable analysis pipeline for the RE-FLOW RoboFlow trajectory dataset. The long-term objective is to connect observed robot motion to decision support for reconfigurable collective geometries:

```text
raw trajectories
    -> validated and synchronized robot states
    -> passage events and route responses
    -> causal online flow-state summaries
    -> post-action shape evaluation
    -> state-geometry-response data for future decision models
```

The current implementation covers the first stage: dataset I/O, validation, synchronization, causal resampling, kinematic preprocessing, quality reporting, and a command-line interface. Geometry reconstruction, passage detection, online state extraction, and evaluation metrics are planned as later modules.

## Repository

- GitHub repository: [`ningwak/trajectory-to-shape-response`](https://github.com/ningwak/trajectory-to-shape-response)
- Python package: `reflow_analysis`
- Primary execution environment: Google Colab
- Raw data location: Google Drive

## Current functionality

### 1. Dataset discovery

The loader recursively searches the selected dataset root for experiment directories containing robot files with names matching `R<robot_id>_*.json`.

This restriction prevents generated files such as `processed/preprocessing_config.json` from being mistaken for raw robot trajectories when the pipeline is run repeatedly.

The supplied dataset is expected to contain two experiment directories:

```text
Roboflow-dataset/
├── closed_loop_rectangle_corridor_drive_with_lane_change_20260903T153210_baseline/
│   ├── R20_....json
│   ├── R24_....json
│   ├── R25_....json
│   ├── R32_....json
│   ├── R35_....json
│   ├── R50_....json
│   └── data_collection_no_barrier.mp4
└── closed_loop_rectangle_corridor_drive_with_lane_change_20260903T155810_barrier_v01/
    ├── R20_....json
    ├── R24_....json
    ├── R25_....json
    ├── R32_....json
    ├── R35_....json
    ├── R50_....json
    └── data_collection_with_barrier.mp4
```

### 2. JSON loading and validation

Each robot file is checked for the following top-level fields:

```text
robot_id
function
run_id
run_timestamp
rectangle_corners
corridor_width_m
trajectory
```

Each trajectory sample must contain:

```text
time
x
y
theta
```

The loader also:

- parses timestamps with or without fractional seconds;
- normalizes all timestamps to UTC;
- converts `x`, `y`, and `theta` to finite floating-point values;
- sorts samples by timestamp;
- removes exact duplicate timestamps while retaining the last record;
- verifies that robot IDs are unique within a run;
- verifies that experiment function, rectangle geometry, and corridor width are consistent across robots;
- records the associated MP4 file when exactly one video is present.

### 3. Condition inference

The experiment condition is inferred from the directory name:

- a name containing `baseline` or `no_barrier` becomes `baseline`;
- a name containing `barrier_v01` becomes `barrier_v01`;
- other barrier directories receive a corresponding barrier label.

The original per-robot `run_id` is retained as `source_run_id`. The experiment directory name is used as the shared `run_name`, because the supplied JSON files use different UUIDs for different robots within the same physical experiment.

### 4. Run-level synchronization

Robot recordings do not start and stop at exactly the same time. The default `intersection` alignment keeps only the interval in which every robot in the run has data.

This is important for flow analysis: a robot that has not started logging must not be interpreted as an absent pedestrian or a change in crowd density.

An optional `union` mode is available for diagnostics, but it will naturally contain invalid samples near the beginning or end of some robot recordings.

### 5. Causal resampling

The raw logs are highly irregular. Many timestamps are only milliseconds apart, and the same pose is often recorded repeatedly. The pipeline resamples each robot to a regular 20 Hz grid using causal last-observation-held interpolation.

At grid time `t`, the state is constructed only from the latest raw observation at or before `t`. No future sample is used. This makes the processed trajectory suitable for later online-state extraction.

The default maximum hold duration is 0.25 seconds. If the most recent observation is older than this limit, the pose is marked invalid rather than silently extrapolated.

Repeated poses are retained in the raw trajectory. They provide evidence that a pose remained available and should not be removed as if they were redundant statistical samples.

### 6. Causal velocity estimation

The heading angle is unwrapped internally before angular velocity is calculated. Linear and angular velocities are estimated using a past-only difference window:

```text
vx(t)    = [x(t)     - x(t - T)]     / T
vy(t)    = [y(t)     - y(t - T)]     / T
omega(t) = [theta(t) - theta(t - T)] / T
```

The default window is `T = 0.25 s`. This simple estimator is intentionally chosen because it is causal, transparent, robust to high-frequency repeated poses, easy to explain, and sufficient for the first analysis stage.

Velocity values are marked invalid when the full past window does not contain valid poses.

### 7. Quality reporting

The preprocessing pipeline produces a one-row-per-robot quality report containing:

- raw sample count;
- number of actual pose changes;
- raw start and end times;
- raw recording duration;
- median positive timestamp interval;
- maximum raw timestamp gap;
- number of aligned output samples;
- valid-pose fraction;
- valid-kinematics fraction.

This report makes data-cleaning decisions visible and prevents preprocessing assumptions from being hidden inside notebook cells.

### 8. Run manifest

The run manifest records the condition label, run directory, run name, robot count and IDs, experiment function, corridor width, rectangle corners, and associated video path.

### 9. Command-line interface

The complete first-stage pipeline can be run with:

```bash
python -m reflow_analysis.cli
```

Available options are:

```text
--data-root
--output-dir
--sample-rate-hz
--max-hold-gap-s
--velocity-window-s
--alignment {intersection,union}
--format {csv,parquet}
```

The defaults are:

```text
data root:          /content/drive/MyDrive/Roboflow-dataset
output directory:   <data-root>/processed
sample rate:        20 Hz
maximum hold gap:   0.25 s
velocity window:    0.25 s
alignment:          intersection
output format:      CSV
```

## Code structure

```text
trajectory-to-shape-response/
├── README.md
├── pyproject.toml
├── src/
│   └── reflow_analysis/
│       ├── __init__.py
│       ├── io.py
│       ├── preprocessing.py
│       └── cli.py
├── notebooks/
│   └── README.md
└── tests/
    ├── test_io.py
    └── test_preprocessing.py
```

All reusable Python logic is placed in the `reflow_analysis` package. Notebooks should contain only orchestration, exploratory checks, and presentation figures. A function that is reused or affects a reported result should be moved into the package and tested.

At the current project size, modules remain at one package level. Additional subpackages such as `reflow_analysis/io/` are unnecessary until a module grows into several files with a clear internal interface.

## Module reference

### `reflow_analysis.io`

#### Data classes

- `RobotTrajectory`: stores one validated robot trajectory and its source metadata.
- `RunData`: stores all robot trajectories belonging to one experiment directory and its video path.

#### Public functions

- `infer_condition(run_name)`: converts an experiment directory name into a stable condition label.
- `load_robot_file(path, condition=None)`: loads and validates one robot JSON file.
- `load_run(run_dir)`: loads all robot files in one experiment directory and checks consistency.
- `discover_run_directories(dataset_root)`: finds experiment directories below the dataset root.
- `load_dataset(dataset_root)`: discovers and loads all experiment runs.
- `build_run_manifest(runs)`: converts run metadata into a compact pandas DataFrame.

### `reflow_analysis.preprocessing`

- `PreprocessingConfig`: stores and validates preprocessing parameters.
- `preprocess_run(run, config=None)`: synchronizes and preprocesses all robots in one experiment.
- `preprocess_dataset(runs, config=None)`: preprocesses multiple experiments and concatenates the results.

### `reflow_analysis.cli`

- `build_parser()`: defines command-line arguments and defaults.
- `main()`: loads data, runs preprocessing, and writes all output files.

## Output files

With the default CSV format, the pipeline writes:

```text
Roboflow-dataset/processed/
├── trajectory_samples.csv
├── quality_report.csv
├── run_manifest.csv
└── preprocessing_config.json
```

### Trajectory table

| Column | Description |
|---|---|
| `condition` | Inferred experiment condition, such as `baseline` or `barrier_v01`. |
| `run_name` | Shared experiment directory name. |
| `robot_id` | Physical robot identifier. |
| `source_run_id` | Original UUID stored in the robot JSON file. |
| `time` | UTC timestamp on the common regular grid. |
| `elapsed_s` | Time from the beginning of the aligned run interval. |
| `source_time` | Timestamp of the raw observation held at the current grid time. |
| `observation_age_s` | Age of the held raw observation. |
| `x`, `y` | Planar robot position. |
| `theta` | Wrapped heading angle. |
| `vx`, `vy` | Causal Cartesian velocity estimate. |
| `speed` | Magnitude of the Cartesian velocity. |
| `omega` | Causal angular velocity estimate. |
| `valid_pose` | Whether a sufficiently recent pose was available. |
| `valid_kinematics` | Whether the complete past velocity window was valid. |

`quality_report.csv` contains per-robot sampling, gap, output-count, and validity diagnostics. `run_manifest.csv` contains condition, robot, geometry, source-directory, and video metadata. `preprocessing_config.json` records the exact parameters used to generate the processed dataset.

## Complete Google Colab guide

The following instructions are designed for a private GitHub repository and a dataset stored in Google Drive.

### Step 1: Create a GitHub personal access token

For a private repository, create a fine-grained personal access token in GitHub:

1. Open GitHub settings.
2. Go to **Developer settings**.
3. Select **Personal access tokens** and **Fine-grained tokens**.
4. Create a token restricted to `ningwak/trajectory-to-shape-response`.
5. For cloning and reading, grant **Contents: Read-only** access.
6. Copy the token when GitHub displays it.

Do not write the token directly into a notebook cell, repository file, or permanent Git remote URL.

### Step 2: Store the token in Colab Secrets

In the Colab notebook:

1. Open the key-shaped **Secrets** panel in the left sidebar.
2. Add a secret named `GITHUB_TOKEN`.
3. Paste the token as its value.
4. Enable notebook access for the secret.

### Step 3: Clone or update the private repository

Run the following as a Python cell. Do not prefix Python statements with `!`.

```python
from pathlib import Path
import subprocess

from google.colab import userdata

repo_dir = Path("/content/trajectory-to-shape-response")
token = userdata.get("GITHUB_TOKEN")
if not token:
    raise RuntimeError("GITHUB_TOKEN is missing or notebook access is disabled.")

authenticated_url = (
    f"https://x-access-token:{token}@github.com/"
    "ningwak/trajectory-to-shape-response.git"
)

if repo_dir.exists():
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "pull", authenticated_url, "main"],
        text=True,
        capture_output=True,
    )
else:
    result = subprocess.run(
        ["git", "clone", authenticated_url, str(repo_dir)],
        text=True,
        capture_output=True,
    )

if result.returncode != 0:
    safe_error = result.stderr.replace(token, "***")
    raise RuntimeError(f"Git operation failed:\n{safe_error}")

# Ensure that the stored remote URL never contains the token.
subprocess.run(
    [
        "git",
        "-C",
        str(repo_dir),
        "remote",
        "set-url",
        "origin",
        "https://github.com/ningwak/trajectory-to-shape-response.git",
    ],
    check=True,
)

del token, authenticated_url
print(f"Repository ready at: {repo_dir}")
```

If the repository later becomes public, use:

```python
!git clone https://github.com/ningwak/trajectory-to-shape-response.git
```

### Step 4: Mount Google Drive

Run this as a Python cell:

```python
from google.colab import drive

drive.mount("/content/drive")
```

Do not write `!from google.colab import drive`. The `!` prefix sends a line to the Linux shell, while `from ... import ...` and `drive.mount(...)` are Python statements.

### Step 5: Verify the dataset path

```python
from pathlib import Path

data_root = Path("/content/drive/MyDrive/Roboflow-dataset")
if not data_root.is_dir():
    raise FileNotFoundError(f"Dataset directory not found: {data_root}")

for path in sorted(data_root.iterdir()):
    print(path.name)
```

The output should include the baseline and barrier experiment directories. If the Drive folder has a different name, use that exact spelling in both `data_root` and the command-line arguments below.

### Step 6: Install the repository package

```python
%cd /content/trajectory-to-shape-response
```

```python
%pip install -e ".[dev]"
```

The editable installation is required because the code uses a `src/` package layout. It makes `reflow_analysis` importable from notebooks and tests.

If `ModuleNotFoundError: No module named 'reflow_analysis'` appears, rerun the installation cell and confirm that the current directory is the repository root.

### Step 7: Run the tests

```python
!python -m pytest -q
```

Expected result:

```text
2 passed
```

Do not run `python tests/test_io.py` directly. The test files are designed for pytest, and direct execution may bypass package discovery or test fixtures.

### Step 8: Run preprocessing

```python
!python -m reflow_analysis.cli \
  --data-root "/content/drive/MyDrive/Roboflow-dataset" \
  --output-dir "/content/drive/MyDrive/Roboflow-dataset/processed" \
  --sample-rate-hz 20 \
  --max-hold-gap-s 0.25 \
  --velocity-window-s 0.25 \
  --alignment intersection \
  --format csv
```

For the supplied dataset, the current implementation should report two runs, twelve robot files, and approximately 78,834 aligned samples. Small differences are possible if the source files are replaced or modified.

### Step 9: Inspect generated results

```python
from pathlib import Path
import pandas as pd

output_dir = Path("/content/drive/MyDrive/Roboflow-dataset/processed")

trajectories = pd.read_csv(
    output_dir / "trajectory_samples.csv",
    parse_dates=["time", "source_time"],
)
quality = pd.read_csv(
    output_dir / "quality_report.csv",
    parse_dates=["raw_start_time", "raw_end_time"],
)
manifest = pd.read_csv(output_dir / "run_manifest.csv")

print(trajectories.shape)
display(quality)
display(manifest)
display(trajectories.head())
```

Check that each condition contains six robots at every aligned timestamp:

```python
robots_per_time = trajectories.groupby(["condition", "time"])["robot_id"].nunique()
print(robots_per_time.groupby("condition").agg(["min", "max"]))
```

With `intersection` alignment, both the minimum and maximum should be six for each condition.

Inspect validity fractions:

```python
display(
    quality[
        [
            "condition",
            "robot_id",
            "raw_sample_count",
            "raw_pose_update_count",
            "raw_max_gap_s",
            "valid_pose_fraction",
            "valid_kinematics_fraction",
        ]
    ]
)
```

### Step 10: Use the API from a notebook

```python
from pathlib import Path

from reflow_analysis.io import build_run_manifest, load_dataset
from reflow_analysis.preprocessing import PreprocessingConfig, preprocess_dataset

data_root = Path("/content/drive/MyDrive/Roboflow-dataset")
runs = load_dataset(data_root)

config = PreprocessingConfig(
    sample_rate_hz=20.0,
    max_hold_gap_s=0.25,
    velocity_window_s=0.25,
    alignment="intersection",
)

trajectories, quality = preprocess_dataset(runs, config)
manifest = build_run_manifest(runs)
```

### Step 11: Optional Parquet output

Parquet files are smaller and preserve data types more reliably than CSV files. Install the optional dependency:

```python
%pip install -e ".[parquet]"
```

Then run:

```python
!python -m reflow_analysis.cli \
  --data-root "/content/drive/MyDrive/Roboflow-dataset" \
  --output-dir "/content/drive/MyDrive/Roboflow-dataset/processed_parquet" \
  --format parquet
```

## Data and repository safety

The raw dataset and videos are not included in this repository. The `.gitignore` excludes `data/`, `outputs/`, `processed/`, MP4 videos, Parquet outputs, Python caches, and notebook checkpoints.

Before any commit, verify the staged files:

```bash
git status
git diff --cached
```

Never commit the GitHub personal access token, private dataset, generated data tables, or original videos.

## Tests

`tests/test_io.py` verifies mixed timestamp precision, UTC normalization, and basic robot metadata loading.

`tests/test_preprocessing.py` verifies run-level intersection alignment, causal resampling, synthetic multi-robot synchronization, and valid kinematic output after the past window is available.

Run the full test suite from the repository root:

```bash
python -m pytest -q
```

## Planned analysis modules

The next stages should remain simple, geometry-aware, and easy to audit:

1. `geometry.py`: corridor-aligned coordinates, verified barrier pose, local analysis region, and virtual gates.
2. `plots.py`: full closed-loop trajectories and local splitter-region views.
3. `passages.py`: finite-state passage detection and `inner`/`outer`/`ambiguous` route labels.
4. `online_state.py`: causal per-robot states and coarse rank-based flow summaries.
5. `metrics.py`: route distribution, local passage time, and spatial speed profiles.

The current baseline and barrier runs are sufficient to validate this feature pipeline and produce preliminary descriptive comparisons. They are not sufficient to train or claim a general geometry-selection policy. Future experiments should vary splitter offset, orientation, length, and activation under repeated comparable flow conditions to build state-geometry-response data.

