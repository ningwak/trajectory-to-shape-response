# Methods and assumptions

## 1. Scope

The repository answers a limited interview question: how can trajectory observations be converted into compact evidence that supports later shape formation? It does not learn the mapping from state to geometry because the supplied dataset contains only two conditions and one experimental run per condition.

The implemented analysis separates:

- **prior / online information** available before or during a future decision; and
- **posterior / evaluation information** used to judge the response afterward.

## 2. Kinematic estimation and validity

At a regular time `t`, causal last-observation hold selects the newest raw sample at or before `t`. `valid_pose` is true only when this source sample exists and its age is at most `max_hold_gap_s`.

With lag `T`, velocity is:

```text
v(t) = [p(t) - p(t-T)] / T
```

`valid_kinematics` requires every resampled pose in the complete lag window to be valid. The first lag window of each run is therefore invalid by construction.

These two masks are retained because they have direct meanings in the preprocessing step. The main analysis does not add a separate `forward_motion_valid` flag: that quantity was not used by the compact state or metrics and would only enlarge the output schema.

## 3. Horizontal analysis region

The full-trajectory plot identifies the affected segment as the nearly horizontal upper leg of the nominal JSON rectangle. The two upper corners supply its reproducible x range and mean y coordinate; they are **route references, not barrier endpoints**. The main analysis uses `s = x - x_mid`, `d = y - y_top`, `v_s = v_x`. A 10% inset at each end keeps the section grid away from the turns, where repeated x crossings would confuse local passage counting. This is a scene-specific selection of the analysis region; the optional, separate topology baseline addresses more general paths.

The mainline does not locate a 2-D changed-occupancy component, spatially resample by arc length, or fit a tilted local flow direction. Selecting another scene's corridor segment would require an explicit new analysis-region rule.

## 4. Cross-sectional inference

For every section `s_j`, forward trajectory crossings are linearly interpolated. Each crossing is one passage-level lateral sample, avoiding raw timestamp-density bias.

The response statistic is the 1-D Wasserstein distance between baseline and changed lateral samples. The supported contiguous interval around its peak with at least 20% of that peak is an approximate **observed response region**, not a statistical detection interval. Sections need crossing counts at least 75% of the corresponding median nonzero section count in both conditions. Random baseline half-splits produce a 95th-percentile **descriptive reference line only**; changing their seed or repeat count does not move the gates or separator.

At each supported section, exact 1-D `K=2` K-means sorts the samples, enumerates every non-empty split point, and selects the partition with minimum within-cluster squared error. This is deterministic and, unlike initialized Lloyd iteration, cannot stop at a local minimum. Silhouette excludes each sample from its own within-cluster distance. A section belongs to the **stable core** only when:

- each cluster contains at least three crossings;
- route-center separation of at least 0.25 m;
- the two changed-condition centers to lie on opposite sides of the baseline median path.

This prevents the early, still-unimodal response region from being drawn as if a separator were already established. Silhouette remains a descriptive column, not an additional acceptance threshold. Response onset and stable splitter core are therefore different spatial concepts.

K-means is the primary baseline because the exact one-dimensional solution is deterministic, transparent, stable for two visibly separated route bands, and needs few parameters. A Gaussian mixture would add unequal variances and probabilistic assignments, but with six robots it is more initialization-sensitive and can overinterpret tails. GMM is therefore more suitable as a later sensitivity analysis than as the interview baseline.

The separator observation at section `s_j` is the midpoint of the two centers. An ordinary least-squares line through the contiguous stable-core midpoints produces the effective separator. For this preliminary dataset, the ambiguity half-width is the larger of 0.02 m and 10% of the median route-center separation. This simple rule is explicit and avoids presenting a more elaborate uncertainty model than the data support.

G0 is the earliest adequately supported interior section upstream of the response; it is an observation boundary, not a claim that turning begins there. G1 is the response start, G2 the midpoint of the stable core, and G3 the response end. G3 is an analysis exit, not a recovery claim when the response reaches the grid boundary.

## 5. Passage and route logic

A passage begins at a forward G0 crossing and ends at the next forward G3 crossing. Pairing uses an event state machine: if another G0 appears before an exit, the older unmatched entry is retained as invalid with `superseded_by_next_G0`, and the newer G0 becomes active. This prevents a missing exit on one closed-loop lap from shifting every later pair. A passage is also invalid when G3 is absent, duration is out of range, or the inferred core has insufficient samples. Completed path length includes the interpolated G0 and G3 positions rather than approximating them by adjacent samples.

Route classification uses the full core, not a single instantaneous lateral coordinate. Let

```text
r_i(s) = d_i(s) - d_split(s)
z_i    = median r_i(s)
```

Then `z_i < -epsilon` is inner, `z_i > epsilon` is outer, and `|z_i| <= epsilon` is ambiguous. Missing adequate core coverage is unknown and invalid; no separate sign-consistency rule is applied. The inner/outer names are a convention for the observed loop direction; in a general path, the two signs should remain neutral side labels until inward orientation is known.

## 6. Compact states and metrics

The decision state has three scalars and two 3-vectors, totaling nine numerical components. The evaluation vector has one 3-vector and three scalars, totaling six. Highly correlated alternatives are left in supporting tables rather than added to the core vectors.

The main passage-time mean pools **all valid passages within a condition**, then divides the changed-condition mean by the baseline mean. The median excess path length also pools valid passages. Per-robot passage medians remain in a supporting table/figure to show heterogeneity, but they do not set the headline ratio. Repeated laps and frames are not independent experimental replicates.

## 7. Output schema

The six exported tables are:

- `section_diagnostics.csv`: W1, auxiliary baseline reference, K-means centers/support/separation/silhouette;
- `passages.csv`: G0-G3 events, labels, duration, length, and rejection reason;
- `online_state_summary.csv`: 9-component state plus validity metadata;
- `compact_evaluation.csv`: 6-component outcome vector by condition;
- `robot_passage_time.csv`: supporting per-robot timing and path-length summaries;
- `quality_report.csv`: data validity and sampling quality per robot JSON.

Intermediate aligned samples, crossing events, pair-distance samples, and cross-sectional samples remain in memory and are not exported by default. Their definitions are documented here instead of generating schema/definition CSV files.

## 8. Statistical limitations

Repeated frames and repeated laps are useful descriptive observations but are not independent experimental replicates. The pipeline does not report a p-value from frame-level tests. General claims require more runs, geometries, demand levels, hazards, and randomized condition order.
