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

## 3. Response-region localization

Time-sampled occupancy would overweight stops. Each robot path is therefore resampled at approximately equal arc-length intervals. Baseline and changed-condition point clouds are binned on the same grid and Gaussian-smoothed. The strongest connected component of positive `baseline_density - changed_density`, restricted to non-negligible baseline occupancy, initializes the local response region.

The component's weighted centroid defines the frame origin. The normalized median of nearby valid baseline velocity directions defines the tangent. The normal is the tangent rotated by 90 degrees.

Failure mode: unrelated condition changes may create a stronger occupancy difference elsewhere. The exported full trajectories and response component should therefore be visually checked. In a larger study, multiple components should be scored using persistence, support, and intervention priors.

## 4. Cross-sectional inference

For every section `s_j`, forward trajectory crossings are linearly interpolated. Each crossing is one passage-level lateral sample, avoiding raw timestamp-density bias.

The response statistic is the 1-D Wasserstein distance between baseline and changed lateral samples. A null distribution is produced by repeatedly splitting baseline samples into two random halves. The 95th percentile is the detection threshold.

At each supported section, exact 1-D `K=2` K-means sorts the samples, enumerates every non-empty split point, and selects the partition with minimum within-cluster squared error. This is deterministic and, unlike initialized Lloyd iteration, cannot stop at a local minimum. Silhouette excludes each sample from its own within-cluster distance. A section belongs to the **stable core** only when:

- each cluster contains at least three crossings;
- route-center separation of at least 0.25 m;
- silhouette of at least 0.55;
- no more than 20% of crossings in a narrow midpoint gap; and
- the two changed-condition centers to lie on opposite sides of the baseline median path.

This prevents the early, still-unimodal response region from being drawn as if a separator were already established. Response onset and stable splitter core are therefore different spatial concepts.

K-means is the primary baseline because the exact one-dimensional solution is deterministic, transparent, stable for two visibly separated route bands, and needs few parameters. A Gaussian mixture would add unequal variances and probabilistic assignments, but with six robots it is more initialization-sensitive and can overinterpret tails. GMM is therefore more suitable as a later sensitivity analysis than as the interview baseline.

The separator observation at section `s_j` is the midpoint of the two centers. An ordinary least-squares line through the contiguous stable-core midpoints produces the effective separator. For this preliminary dataset, the ambiguity half-width is the larger of 0.02 m and 10% of the median route-center separation. This simple rule is explicit and avoids presenting a more elaborate uncertainty model than the data support.

G0 is the earliest adequately supported section on a grid extended upstream of
the response component by `section_upstream_margin_m`; it is an observation
boundary, not a claim that turning begins there. G3 is the last supported
response section and is reported as an analysis exit when recovery is not seen.

## 5. Passage and route logic

A passage begins at a forward G0 crossing and ends at the next forward G3 crossing. Pairing uses an event state machine: if another G0 appears before an exit, the older unmatched entry is retained as invalid with `superseded_by_next_G0`, and the newer G0 becomes active. This prevents a missing exit on one closed-loop lap from shifting every later pair. A passage is also invalid when G3 is absent, duration is out of range, or the inferred core has insufficient samples. Completed path length includes the interpolated G0 and G3 positions rather than approximating them by adjacent samples.

Route classification uses the full core, not a single instantaneous lateral coordinate. Let

```text
r_i(s) = d_i(s) - d_split(s)
z_i    = median r_i(s)
```

Then `z_i < -epsilon` is inner, `z_i > epsilon` is outer, and `|z_i| <= epsilon` is ambiguous. A passage is also ambiguous when fewer than 70% of core samples agree in sign. This reduces sensitivity to noise at one gate. The inner/outer names are a convention for the observed loop direction; in a general path, the two signs should remain neutral side labels until inward orientation is known.

## 6. Compact states and metrics

The decision state has three scalars and two 3-vectors, totaling nine numerical components. The evaluation vector has one 3-vector and three scalars, totaling six. Highly correlated alternatives are left in supporting tables rather than added to the core vectors.

Per-robot passage tables are always exported because they expose heterogeneity and make the small effective sample size explicit. Repeated laps are summarized within robot first. Relative passage time is computed within matched robot IDs across conditions and then summarized across robots; excess length is likewise summarized from robot-level medians. This prevents robots with more retained laps from receiving extra weight.

## 7. Output schema

The six exported tables are:

- `section_diagnostics.csv`: W1, null threshold, K-means centers/support/separation/silhouette;
- `passages.csv`: G0-G3 events, labels, duration, length, and rejection reason;
- `online_state_summary.csv`: 9-component state plus validity metadata;
- `compact_evaluation.csv`: 6-component outcome vector by condition;
- `robot_passage_time.csv`: robot-first timing and path-length summaries;
- `quality_report.csv`: data validity and sampling quality per robot JSON.

Intermediate aligned samples, crossing events, pair-distance samples, and cross-sectional samples remain in memory and are not exported by default. Their definitions are documented here instead of generating schema/definition CSV files.

## 8. Statistical limitations

Repeated frames and repeated laps are useful descriptive observations but are not independent experimental replicates. The pipeline does not report a p-value from frame-level tests. General claims require more runs, geometries, demand levels, hazards, and randomized condition order.
