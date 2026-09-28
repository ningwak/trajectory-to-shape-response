# Trajectory to Shape Response

A compact, control-oriented analysis pipeline for the RE-FLOW RoboFlow interview dataset. It converts JSON robot trajectories into:

1. a data-derived local flow frame and **effective splitter**;
2. a 9-component causal state that a future shape-decision agent could observe; and
3. a separate 6-component post-action evaluation vector.

The methods are deliberately transparent: causal finite differences, spatial occupancy, 1-D Wasserstein distance, two-cluster cross-sections, virtual-gate events, quantiles, and descriptive robot-level metrics. The current two runs demonstrate state construction and response evaluation; they are not sufficient to train a general geometry-selection policy.

Repository: [ningwak/trajectory-to-shape-response](https://github.com/ningwak/trajectory-to-shape-response)

## Important interpretation rules

- The JSON `rectangle_corners` field is drawn only as the nominal route boundary. It is **not** treated as the splitter.
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

`valid_pose` and `valid_kinematics` are quality masks, not learned state features. No additional motion-validity flag is calculated because it was not used by the selected state or evaluation metrics.

### B. Full trajectories and local response discovery

`01_full_trajectories.png` draws every robot path for baseline and barrier runs with shared global `x/y` limits and the same 1:1 physical scale. The dashed polygon is the nominal route stored in the JSON; no splitter is drawn at this stage.

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
- the stable splitter core requires at least three samples per route, at least 0.25 m center separation, silhouette at least 0.55, a depleted midpoint gap, and centers on opposite sides of the baseline median path.

A robust line through valid cluster midpoints gives

```text
d_split(s) = intercept + slope * s
```

Four virtual gates organize the local control problem:

| Gate | Meaning | Data-derived rule |
|---|---|---|
| G0 | approach observation entry | earliest well-supported section upstream of the detected response |
| G1 | detectable response onset | first supported section whose Wasserstein distance exceeds the null threshold |
| G2 | route classification section | midpoint of the stable bimodal core |
| G3 | analysis exit / recovery observation | last well-supported response section |

If the response remains significant until trajectories leave the well-supported straight section, G3 is an **analysis exit**, not evidence that recovery occurred. The exact values are exported to `config/inferred_geometry.json`.

The effective-splitter line is drawn only over the contiguous **stable core**, not from the earlier response onset. G1 says that trajectories have begun reacting; the core begins only after two separated routes and a low-occupancy gap are established.

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

The repository also exports the complete per-robot passage-time/path-length table. No frame-level significance test is applied because there is only one run per condition.

Inner/outer labels are meaningful only after a separator has been inferred from the changed condition. Baseline route shares are therefore marked not applicable and omitted from the route-allocation figure.

## Generated plots

| File | Purpose |
|---|---|
| `01_full_trajectories.png` | raw path overview, identical axes, nominal route from JSON |
| `02_inferred_local_geometry.png` | inferred frame, gates, approach, effective separator, routes |
| `03_wasserstein_profile.png` | response profile, null threshold, supported interval and core |
| `04_cross_section_kmeans.png` | representative lateral samples and K-means centers |
| `05_route_distribution.png` | stacked route proportions |
| `06_passage_time.png` | robot-matched passage-time comparison |

## Printed and CSV results

The complete CLI prints concise tables for:

- data quality per robot JSON;
- inferred splitter and gate landmarks;
- the compact 6-component evaluation;
- per-robot passage time and path length;
- invalid and retained passage counts.

Only six reviewable CSV files are saved:

| File | Contents |
|---|---|
| `quality_report.csv` | one row per robot JSON |
| `section_diagnostics.csv` | Wasserstein and K-means evidence by cross-section |
| `passages.csv` | structured G0-to-G3 passage records |
| `online_state_summary.csv` | causal 9-component state history |
| `compact_evaluation.csv` | compact 6-component evaluation |
| `robot_passage_time.csv` | robot-level timing and path-length summary |

Large aligned-sample, per-timestamp spacing, G0-event, and intermediate crossing tables are intentionally not written. They can be recomputed from the raw JSON when needed. Feature definitions belong in this README and `METHODS.md`, not in additional CSV files.

All figures and result tables use the same presentation order: `baseline` first and `barrier_v01` second. The route-allocation figure is the exception because inner/outer labels are not applicable to the unsplit baseline.

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

The data path is supplied explicitly in the command below; the package contains no hard-coded Google Drive path.

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
  --output-dir "/content/drive/MyDrive/Roboflow-dataset/analysis_output_compact"
```

**5. Display the main figures**

```python
from pathlib import Path
from IPython.display import display, Image

figure_dir = Path("/content/drive/MyDrive/Roboflow-dataset/analysis_output_compact/figures")
for path in sorted(figure_dir.glob("*.png")):
    print(path.name)
    display(Image(filename=str(path)))
```

**6. Inspect exact tables**

```python
import pandas as pd

table_dir = Path("/content/drive/MyDrive/Roboflow-dataset/analysis_output_compact/tables")
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
