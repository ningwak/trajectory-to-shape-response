# Trajectory to Shape Response

A compact, control-oriented analysis pipeline for the RE-FLOW RoboFlow interview dataset. It converts JSON robot trajectories into:

1. a data-derived local flow frame and **effective splitter**;
2. a 9-component causal state that a future shape-decision agent could observe; and
3. a separate 6-component post-action evaluation vector.

The methods are deliberately transparent: causal finite differences, spatial occupancy, 1-D Wasserstein distance, two-cluster cross-sections, virtual-gate events, quantiles, and descriptive robot-level metrics. The current two runs demonstrate state construction and response evaluation; they are not sufficient to train a general geometry-selection policy.

Repository: [ningwak/trajectory-to-shape-response](https://github.com/ningwak/trajectory-to-shape-response)

## Important interpretation rules

- The JSON `rectangle_corners` field is drawn only as a reference rectangle. It is **not** treated as the splitter.
- `corridor_width_m` is retained as source metadata. Its exact experiment-specific semantics are not documented in the supplied JSON, so it is **not** used to place the splitter.
- The inferred line is called an **effective trajectory separator**. It estimates the boundary between observed route families; it does not claim to recover the physical barrier pose.
- MP4 files are not inputs. Every Python discovery path is restricted to files named `R<robot_id>_*.json`; no script discovers, enumerates, opens, or decodes video.
- Global route topology is optional and outside the main analysis. The main splitter analysis does not assume a rectangle or require global path recognition.

## Main pipeline

```text
JSON validation and causal preprocessing
        -> full-trajectory visual check
        -> 2-D changed-occupancy region
        -> baseline local flow frame
        -> cross-section response and effective splitter
        -> four virtual gates and structured passages
        -> causal decision state | post-action evaluation
        -> printed tables, CSV files, figures, and JSON config
```

### A. JSON I/O and preprocessing

The loader validates required metadata and `time, x, y, theta` samples, converts time to UTC, sorts samples, removes duplicate timestamps, and checks within-run consistency. The default preprocessing then:

- aligns robots over their shared time interval;
- resamples at 20 Hz by causal last-observation hold;
- marks a pose valid only when the newest past observation is no more than 0.25 s old;
- estimates `vx`, `vy`, and `omega` with a past-only 0.25 s finite-difference window;
- marks kinematics valid only when the complete lag window contains valid poses.

`valid_pose`, `valid_kinematics`, and `forward_motion_valid` are quality masks, not learned state features. `forward_motion_valid` requires valid kinematics, positive local longitudinal speed, and a bounded lateral-to-longitudinal velocity ratio.

### B. Full trajectories and local response discovery

`01_full_trajectories.png` draws every robot path for baseline and barrier runs on equal world-coordinate axes. The dashed polygon is the JSON reference rectangle; no splitter is drawn at this stage.

To locate the local response region without video or manual coordinates:

1. each trajectory is resampled by arc length so pauses do not dominate occupancy;
2. smoothed 2-D occupancy is computed separately for baseline and changed conditions;
3. the strongest connected baseline-depletion component identifies a candidate response region;
4. valid baseline velocities in that region determine the local tangent `s`; its left normal defines lateral coordinate `d`.

### C. Effective splitter, approach, and gates

Passage-level lateral crossings are interpolated on a sequence of sections perpendicular to the inferred flow. At each section:

- 1-D Wasserstein distance compares baseline and changed lateral distributions;
- a baseline-vs-baseline random split supplies a 95% null threshold;
- deterministic `K=2` 1-D K-means summarizes the two changed-condition route families;
- minimum cluster support, center separation, silhouette, and section support reject weak sections.

A robust line through valid cluster midpoints gives

```text
d_split(s) = intercept + slope * s
```

Four virtual gates organize the local control problem:

| Gate | Meaning | Data-derived rule |
|---|---|---|
| G0 | approach observation entry | earliest well-supported upstream section with stable forward baseline motion |
| G1 | detectable response onset | first supported section whose Wasserstein distance exceeds the null threshold |
| G2 | route classification section | midpoint of the stable bimodal core |
| G3 | analysis exit / recovery observation | last well-supported response section |

If the response remains significant until trajectories leave the well-supported straight section, G3 is an **analysis exit**, not evidence that recovery occurred. The exact values are exported to `config/inferred_geometry.json`.

For each completed G0-to-G3 passage, the route score is the median

```text
z_i = median[d_i(s) - d_split(s)]
```

over the inferred core. A route is `ambiguous` when the score lies inside the uncertainty band or when signs are insufficiently consistent. Missing core coverage is `unknown` and invalid, not ambiguous.

### D. Causal decision state: 9 numerical components

At 2 Hz, the online state uses only current/past observations:

```text
x_dec(t) = [N_W, delta_t_last, N_approach, q_v(3), q_d(3)]
```

| Block | Size | Definition |
|---|---:|---|
| `crossing_count_window` | 1 | G0 crossings in the previous 15 s |
| `time_since_last_crossing_s` | 1 | time since the most recent G0 crossing |
| `n_approach` | 1 | robots currently between G0 and G1 |
| `v_s_q33/q50/q67_mps` | 3 | longitudinal-speed quantiles of recent G0 crossings |
| `d_q33/q50/q67_m` | 3 | lateral-position quantiles of recent G0 crossings |

Quantiles are missing when fewer than three crossings are available, and `crossing_count_window` exposes that support explicitly. The quantiles aggregate agents that crossed the same virtual gate during a recent window; they do not require robots to occupy the gate at the same timestamp.

### E. Post-action evaluation: 6 numerical components

Evaluation is intentionally separate from controller input:

| Block | Size | Definition |
|---|---:|---|
| route proportions | 3 | inner, outer, ambiguous fractions |
| relative passage time | 1 | condition median G0-to-G3 time divided by baseline median |
| excess path length | 1 | median path length beyond straight G0-to-G3 distance |
| lower-tail pair distance | 1 | 10th percentile of timestamp-wise nearest robot-center distance in G1-to-G3 ROI |

Pair distance is evaluated at 2 Hz. At every timestamp with at least two valid robots in the ROI, the minimum Euclidean center distance is found; the reported metric is the 10th percentile across those minima. Because robot radius is unavailable, this is a spacing proxy, not physical clearance.

The repository also exports the complete per-robot passage-time/path-length table and supporting speed and spacing diagnostics. No frame-level significance test is applied because there is only one run per condition.

## Generated plots

| File | Purpose |
|---|---|
| `01_full_trajectories.png` | raw path overview, identical axes, JSON reference rectangle |
| `02_inferred_local_geometry.png` | inferred frame, gates, approach, effective separator, routes |
| `03_wasserstein_profile.png` | response profile, null threshold, supported interval and core |
| `04_cross_section_kmeans.png` | representative lateral samples and K-means centers |
| `05_route_distribution.png` | stacked route proportions |
| `06_passage_time.png` | robot-matched passage-time comparison |
| `07_online_state.png` | example causal count and quantile state history |
| `08_spatial_speed_profile.png` | supporting local longitudinal-speed profile |
| `09_pair_distance.png` | supporting spacing-proxy distribution |

## Printed and CSV results

The complete CLI prints concise tables for:

- data quality per robot JSON;
- inferred splitter and gate landmarks;
- the compact 6-component evaluation;
- per-robot passage time and path length;
- invalid and retained passage counts.

All detailed tables are also saved under `analysis_output/tables`, including section diagnostics, K-means results, G0 events, online state history, passage records, pair-distance samples, and explicit state/metric definition tables.

## Google Colab: complete run

The two raw run folders should be stored directly under:

```text
My Drive/
└── Roboflow-dataset/
    ├── closed_loop_..._baseline/
    │   ├── R20_....json
    │   └── ...
    └── closed_loop_..._barrier_v01/
        ├── R20_....json
        └── ...
```

MP4 files may remain in those folders; the code ignores them.

Run these cells in a fresh Colab notebook.

**1. Mount Drive**

```python
from google.colab import drive
drive.mount("/content/drive")
```

**2. Clone and install**

```bash
%cd /content
!git clone https://github.com/ningwak/trajectory-to-shape-response.git
%cd /content/trajectory-to-shape-response
!python -m pip install -e ".[dev]"
```

If the repository already exists in the current Colab runtime:

```bash
%cd /content/trajectory-to-shape-response
!git pull --ff-only
!python -m pip install -e ".[dev]"
```

**3. Run tests**

```bash
!python -m pytest -q
```

**4. Run the complete main pipeline**

```bash
!python -m reflow_analysis.analysis_cli \
  --data-root "/content/drive/MyDrive/Roboflow-dataset" \
  --output-dir "/content/drive/MyDrive/Roboflow-dataset/analysis_output"
```

**5. Display the main figures**

```python
from pathlib import Path
from IPython.display import display, Image

figure_dir = Path("/content/drive/MyDrive/Roboflow-dataset/analysis_output/figures")
for path in sorted(figure_dir.glob("*.png")):
    print(path.name)
    display(Image(filename=str(path)))
```

**6. Inspect exact tables**

```python
import pandas as pd

table_dir = Path("/content/drive/MyDrive/Roboflow-dataset/analysis_output/tables")
display(pd.read_csv(table_dir / "compact_evaluation.csv"))
display(pd.read_csv(table_dir / "robot_passage_time.csv"))
display(pd.read_csv(table_dir / "section_diagnostics.csv"))
```

## Optional global path-topology baseline

This baseline is implemented, but the main pipeline does not call it. It makes no rectangle, closed-loop, or fixed-lane assumption:

1. spatially resample baseline paths;
2. create and smooth an occupancy grid;
3. threshold and morphologically clean occupied regions;
4. skeletonize the result;
5. convert 8-neighbor skeleton pixels into a graph;
6. prune short leaf spurs;
7. infer edge direction from nearby velocity votes;
8. provide distance-plus-heading map matching with an adjacency penalty.

Install and run it separately:

```bash
!python -m pip install -e ".[topology]"
!python -m reflow_analysis.path_cli \
  --data-root "/content/drive/MyDrive/Roboflow-dataset" \
  --output-dir "/content/drive/MyDrive/Roboflow-dataset/topology_output"
```

The output contains `topology_nodes.csv`, `topology_edges.csv`, and `topology_graph.png`. This is a baseline interface for more complex scenes, not a prerequisite for splitter inference.

## Repository structure

```text
trajectory-to-shape-response/
├── README.md
├── METHODS.md
├── pyproject.toml
├── src/reflow_analysis/
│   ├── io.py
│   ├── preprocessing.py
│   ├── geometry.py
│   ├── splitter_inference.py
│   ├── passages.py
│   ├── online_state.py
│   ├── metrics.py
│   ├── plots.py
│   ├── topology.py
│   ├── pipeline.py
│   ├── cli.py
│   ├── analysis_cli.py
│   └── path_cli.py
└── tests/
```

One flat Python package is sufficient at this size. The optional topology code is isolated by module and dependency extra rather than by another directory level.

## Local run

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m reflow_analysis.analysis_cli \
  --data-root "/path/to/Roboflow-dataset" \
  --output-dir "/path/to/analysis_output"
```

See [METHODS.md](METHODS.md) for algorithm details, assumptions, failure modes, and the output schema.
