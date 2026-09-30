# Trajectory to Shape Response

A compact, control-oriented analysis pipeline for the RE-FLOW RoboFlow interview dataset. It converts JSON robot trajectories into:

1. a horizontal analysis frame from the JSON route edge and a trajectory-derived **effective splitter**;
2. a 9-component causal state that a future shape-decision agent could observe; and
3. a separate 6-component post-action evaluation vector.

The methods are deliberately transparent: causal finite differences, horizontal cross-section crossings, 1-D Wasserstein distance, two-route clustering, virtual-gate events, quantiles, and descriptive passage-level metrics. The two runs demonstrate state construction and response evaluation; they are not sufficient to train a general geometry-selection policy.

Repository: [ningwak/trajectory-to-shape-response](https://github.com/ningwak/trajectory-to-shape-response)

## Important interpretation rules

- The JSON `rectangle_corners` field defines the nominal route. Its upper straight edge supplies the horizontal analysis frame; it is **not** treated as the splitter or a barrier-pose measurement.
- `corridor_width_m` is retained as source metadata. Its exact experiment-specific semantics are not documented in the supplied JSON, so it is **not** used to place the splitter.
- The inferred line is called an **effective trajectory separator**. It estimates the boundary between observed route families; it does not claim to recover the physical barrier pose.
- MP4 files are not inputs. Every Python discovery path is restricted to files named `R<robot_id>_*.json`; no script discovers, enumerates, opens, or decodes video.
- Global route topology is optional and outside the main analysis. This interview-specific mainline explicitly selects the upper nominal rectangle edge; another path requires selecting a different segment or using the optional topology baseline.

## Main pipeline

```text
JSON validation and causal preprocessing
        -> full-trajectory visual check
        -> horizontal frame from the JSON upper route edge
        -> passage-level cross-sections and Wasserstein response
        -> two-route centres and effective trajectory separator
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

### B. Full trajectories and horizontal analysis frame

`01_full_trajectories.png` draws every robot path for baseline and barrier runs with shared global `x/y` limits and the same 1:1 physical scale. The dashed polygon is the nominal route stored in the JSON; no splitter is drawn at this stage.

The complete paths show that this experiment's altered motion is on the upper, nearly horizontal leg of the nominal JSON rectangle. That known route edge defines `s = x - x_mid` and `d = y - y_top`, with `v_s = v_x`. Its vertices are turns, so the analysis grid excludes the outer 10% of the edge at each end. This is an experiment-specific analysis-region choice, **not** a manually specified barrier location. No 2-D occupancy map, arc-length resampling or estimated tilted flow frame is needed in the mainline.

### C. Effective splitter, approach, and gates

Each forward passage contributes one interpolated lateral position at each evenly spaced horizontal (`x`) section. This avoids giving slow-moving robots or dense raw timestamps extra weight. At each section:

- 1-D Wasserstein distance compares baseline and changed lateral distributions;
- baseline-vs-baseline random half-splits supply an **auxiliary, descriptive** variability line on the plot; they do not select the response interval or claim statistical significance;
- exact `K=2` 1-D K-means enumerates sorted split points and selects the global minimum-SSE partition;
- the response interval is the contiguous supported region around the W1 peak where W1 is at least 20% of that peak;
- the stable two-route core requires at least three crossings per side, at least 0.25 m centre separation, the baseline median path between the two route centres, and at most 10% of changed crossings within a narrow natural-path band (±15% of the route-centre separation around the baseline median).

Silhouette is retained as a diagnostic, not another tuning threshold; a sample is excluded from its own within-cluster distance. An ordinary least-squares line through valid cluster
midpoints gives

```text
d_split(s) = intercept + slope * s
```

Four virtual gates organize the local control problem:

| Gate | Meaning | Data-derived rule |
|---|---|---|
| G0 | approach observation entry | earliest well-supported interior section upstream of the response |
| G1 | approximate response onset | first section of the sustained peak-relative W1 interval |
| G2 | route classification section | midpoint of the stable bimodal core |
| G3 | analysis exit | last section of the sustained peak-relative W1 interval |

When the response persists to the end of the straight-edge grid, G3 is an **analysis exit**, not evidence that recovery occurred. The exact values are exported to `config/inferred_geometry.json`.

G0 is an observation boundary, not a detected turning instant. The 10% inset avoids placing it on a corner, where a robot may cross the same `x` value more than once. The method does not use `|v_d| > |v_s|` to define approach onset.

The effective-separator line is drawn only over the contiguous **stable core**, not from the earlier response onset. G1 marks where distributions begin to differ noticeably; the core begins once the natural path is mostly vacated and two supported routes are separated. The line is a descriptive boundary between observed paths, so an ambiguous passage or occasional deviation can still cross it.

For each completed G0-to-G3 passage, the route score is the median

```text
z_i = median[d_i(s) - d_split(s)]
```

over the inferred core. A route is `ambiguous` when its score lies inside the narrow band around the separator. Missing core coverage is `unknown` and invalid, not ambiguous. A separate sign-consistency threshold is no longer used.

Passages are paired by a small chronological state machine. A new G0 replaces
an older unmatched G0, preventing one missed G3 in a closed-loop run from
shifting every later lap. Replaced entries remain visible with the reason
`superseded_by_next_G0`. Completed path length starts and ends at the linearly
interpolated G0/G3 positions rather than at the nearest grid samples.

For this experiment, negative separator offset is named `inner` and positive
offset `outer`. This name relies on the observed direction around the supplied
loop; a general scene should first infer inward orientation or retain neutral
negative-side/positive-side labels.

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

An interpolated crossing becomes observable when its following 20 Hz sample
arrives. Because state timestamps are a subset of that same regular grid, the
crossing sample is always available by the first state update that uses it;
future passage completion and route labels are never used.

### E. Post-action evaluation: 6 numerical components

Evaluation is intentionally separate from controller input:

| Block | Size | Definition |
|---|---:|---|
| route proportions | 3 | inner, outer, ambiguous fractions |
| relative passage time | 1 | mean of all valid barrier passages divided by mean of all valid baseline passages |
| excess path length | 1 | median across all valid passage excess path lengths |
| lower-tail pair distance | 1 | 10th percentile of timestamp-wise nearest robot-center distance in G1-to-G3 ROI |

Pair distance is evaluated at 2 Hz. At every timestamp with at least two valid robots in the ROI, the minimum Euclidean center distance is found; the reported metric is the 10th percentile across those minima. Because robot radius is unavailable, this is a spacing proxy, not physical clearance.

The main passage-time comparison pools valid G0-to-G3 passages within each condition. It answers: *How long did one observed valid passage take on average?* The CLI prints both means, their ratio and the counts. A separate per-robot median table/paired plot exposes heterogeneity but does not determine the headline ratio. Repeated laps are **not independent experimental runs**; no significance test is claimed from them.

Inner/outer labels are meaningful only after a separator has been inferred from the changed condition. Baseline route shares are therefore marked not applicable and omitted from the route-allocation figure.

## Generated plots

| File | Purpose |
|---|---|
| `01_full_trajectories.png` | raw path overview, identical axes, nominal route from JSON |
| `02_inferred_local_geometry.png` | horizontal frame, gates, approach and barrier routes; baseline paths are neutral gray |
| `03_wasserstein_profile.png` | response profile, descriptive baseline reference, peak-relative interval and core |
| `04_cross_section_kmeans.png` | representative lateral samples and K-means centers |
| `05_route_distribution.png` | stacked route proportions |
| `06_passage_time.png` | paired robot-level median times (supporting view, not the pooled mean) |

## Printed and CSV results

The complete CLI prints concise tables for:

- data quality per robot JSON;
- inferred splitter and gate landmarks;
- the compact 6-component evaluation;
- pooled mean time and valid-passage count per condition;
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
├── ALGORITHMS_PSEUDOCODE.md
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

See [METHODS.md](METHODS.md) for assumptions, failure modes, and the output
schema. See [ALGORITHMS_PSEUDOCODE.md](ALGORITHMS_PSEUDOCODE.md) for
implementation-aligned pseudocode covering the main pipeline and the optional
global topology baseline.
