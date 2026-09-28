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

After the local frame is inferred:

```text
forward_motion_valid = valid_kinematics
                       and v_s > v_min
                       and |v_d| <= tan(theta_max) v_s
```

This mask protects crossings and diagnostics from backward or strongly transverse motion. It is not included among the nine policy-state numbers.

## 3. Response-region localization

Time-sampled occupancy would overweight stops. Each robot path is therefore resampled at approximately equal arc-length intervals. Baseline and changed-condition point clouds are binned on the same grid and Gaussian-smoothed. The strongest connected component of positive `baseline_density - changed_density`, restricted to non-negligible baseline occupancy, initializes the local response region.

The component's weighted centroid defines the frame origin. The normalized median of nearby valid baseline velocity directions defines the tangent. The normal is the tangent rotated by 90 degrees.

Failure mode: unrelated condition changes may create a stronger occupancy difference elsewhere. The exported full trajectories and response component should therefore be visually checked. In a larger study, multiple components should be scored using persistence, support, and intervention priors.

## 4. Cross-sectional inference

For every section `s_j`, forward trajectory crossings are linearly interpolated. Each crossing is one passage-level lateral sample, avoiding raw timestamp-density bias.

The response statistic is the 1-D Wasserstein distance between baseline and changed lateral samples. A null distribution is produced by repeatedly splitting baseline samples into two random halves. The 95th percentile is the detection threshold.

At each supported section, deterministic 1-D `K=2` K-means is initialized at the first and third quartiles. It is retained when:

- both clusters meet minimum sample support;
- center separation exceeds a minimum physical distance; and
- mean silhouette exceeds a minimum value.

K-means is the primary baseline because it is deterministic, transparent, stable for two visibly separated route bands, and needs few parameters. A Gaussian mixture would add unequal variances and probabilistic assignments, but with six robots it is more initialization-sensitive and can overinterpret tails. GMM is therefore more suitable as a later sensitivity analysis than as the interview baseline.

The separator observation at section `s_j` is the midpoint of the two centers. Iterative MAD-based outlier removal followed by a line fit produces the effective separator. The ambiguity half-width is the maximum of a 0.02 m floor, 10% of median cluster separation, and twice the separator-fit RMSE.

## 5. Passage and route logic

A passage begins at a forward G0 crossing and ends at the next forward G3 crossing. It is invalid when G3 is absent, duration is out of range, or the inferred core has insufficient samples.

Route classification uses the full core, not a single instantaneous lateral coordinate. Let

```text
r_i(s) = d_i(s) - d_split(s)
z_i    = median r_i(s)
```

Then `z_i < -epsilon` is inner, `z_i > epsilon` is outer, and `|z_i| <= epsilon` is ambiguous. A passage is also ambiguous when fewer than 70% of core samples agree in sign. This reduces sensitivity to noise at one gate.

## 6. Compact states and metrics

The decision state has three scalars and two 3-vectors, totaling nine numerical components. The evaluation vector has one 3-vector and three scalars, totaling six. Highly correlated alternatives are left in supporting tables rather than added to the core vectors.

Per-robot passage tables are always exported because they expose heterogeneity and make the small effective sample size explicit.

## 7. Output schema

Key tables:

- `trajectory_samples.csv`: aligned poses, kinematics, validity masks, and local coordinates;
- `section_diagnostics.csv`: W1, null threshold, K-means centers/support/separation/silhouette;
- `section_crossing_samples.csv`: passage-level cross-sectional lateral samples;
- `passages.csv`: G0-G3 events, labels, duration, length, and rejection reason;
- `online_state_summary.csv`: 9-component state plus validity metadata;
- `compact_evaluation.csv`: 6-component outcome vector by condition;
- `robot_passage_time.csv`: robot-first timing and path-length summaries;
- `pair_distance_timeseries.csv`: timestamp-wise spacing proxy.

## 8. Statistical limitations

Repeated frames and repeated laps are useful descriptive observations but are not independent experimental replicates. The pipeline does not report a p-value from frame-level tests. General claims require more runs, geometries, demand levels, hazards, and randomized condition order.
