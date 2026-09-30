# Algorithm Pseudocode Reference

This document mirrors the implementation in `src/reflow_analysis/`. It is a
reading aid, not an alternative specification: when a detail matters for exact
reproduction, the Python source and the saved `analysis_config.json` remain
authoritative.

## 1. Notation and data model

- `baseline`: run without the physical barrier.
- `barrier_v01`: run with the barrier.
- `(x, y)`: world-frame robot position.
- `(s, d)`: horizontal analysis coordinates on the JSON upper route edge:
  `s = x - x_mid`, `d = y - y_top`, `v_s = v_x`.
- `G0`: approach-entry gate.
- `G1`: first detectable response gate.
- `G2`: representative route-classification section.
- `G3`: analysis-exit gate.
- `d_split(s)`: fitted effective trajectory separator. It is not a direct
  estimate of the physical barrier outline.
- A *section sample* is one interpolated forward crossing of a transverse
  section by one robot passage. It is not one raw timestamp.

The main analysis assumes exactly one baseline run and one changed/barrier run.
The optional topology algorithm is separate and uses baseline trajectories only.

## 2. End-to-end main pipeline

Corresponding function: `pipeline.run_full_analysis`

```text
ALGORITHM RunFullAnalysis(data_root, output_directory, configuration)
    runs <- discover and load every R<id>_*.json file below data_root

    trajectories, quality_report <- PreprocessDataset(runs)
    manifest <- one metadata row per run

    splitter_result <- InferEffectiveSplitter(trajectories, rectangle_corners)
    geometry <- splitter_result.geometry

    trajectories <- TransformWorldToLocalFrame(trajectories, geometry)
    passages <- DetectPassages(trajectories, geometry)
    online_state <- ExtractOnlineState(trajectories, passages, geometry)
    evaluation_tables <- BuildEvaluationTables(
        trajectories, passages, geometry
    )

    write exactly six reviewable CSV tables
    write inferred geometry and analysis configuration as JSON
    generate exactly six figures
    write one compact machine-readable summary

    RETURN all principal tables, figures, geometry, and summary in memory
END ALGORITHM
```

Condition-dependent tables and comparison figures always place `baseline`
before `barrier_v01`. The route-allocation figure contains only the barrier
condition because inner/outer labels are not meaningful for the unsplit run.

## 3. JSON discovery, validation, and loading

Corresponding functions: `io.discover_run_directories`, `io.load_robot_file`,
`io.load_run`, `io.load_dataset`

```text
ALGORITHM DiscoverRunDirectories(dataset_root)
    recursively find files whose names match R<digits>_*.json
    collect their parent directories
    sort and return the unique directories
END ALGORITHM
```

```text
ALGORITHM LoadRobotFile(json_path, optional_condition)
    payload <- parse JSON object

    REQUIRE top-level keys:
        robot_id, function, run_id, run_timestamp,
        rectangle_corners, corridor_width_m, trajectory
    REQUIRE trajectory is a non-empty list
    REQUIRE every trajectory record has time, x, y, theta

    samples <- table of [time, x, y, theta]
    parse all timestamps as UTC
    convert x, y, theta to finite floating-point values
    stable-sort samples by time
    for duplicate timestamps, retain the last record

    parse exactly four JSON rectangle corners
    infer condition from directory name unless condition was supplied

    RETURN RobotTrajectory(metadata, samples)
END ALGORITHM
```

```text
ALGORITHM LoadRun(run_directory)
    files <- sorted files matching R<digits>_*.json in this directory
    condition <- infer from directory name
    robots <- LoadRobotFile(file, condition) for every file
    sort robots by robot_id

    REQUIRE unique robot IDs
    REQUIRE equal function metadata across robot files
    REQUIRE equal rectangle corners across robot files
    REQUIRE equal corridor_width_m across robot files

    RETURN RunData(directory, run name, condition, robots)
END ALGORITHM
```

Only JSON files matching the robot naming pattern are discovered. MP4 files are
never enumerated or read.

## 4. Causal preprocessing and kinematic estimation

Corresponding functions: `preprocessing._build_time_grid`,
`preprocessing._resample_robot`, `preprocessing.preprocess_run`,
`preprocessing.preprocess_dataset`

### 4.1 Shared time grid

```text
ALGORITHM BuildTimeGrid(robots, sample_rate, alignment_mode)
    starts <- first raw timestamp of each robot
    ends   <- last raw timestamp of each robot

    IF alignment_mode is "intersection" THEN
        grid_start <- latest(starts)
        grid_end   <- earliest(ends)
    ELSE
        grid_start <- earliest(starts)
        grid_end   <- latest(ends)
    END IF

    period <- 1 / sample_rate
    round grid_start upward to the next period boundary
    round grid_end downward to the previous period boundary
    RETURN regular UTC timestamps from grid_start through grid_end
END ALGORITHM
```

The main pipeline uses intersection alignment so all robots in a run share the
same analysed interval.

### 4.2 Causal last-observation hold

```text
ALGORITHM CausalResampleRobot(raw_samples, regular_grid, max_hold_gap)
    theta_unwrapped <- unwrap raw theta to remove +/-pi discontinuities

    FOR each grid time t in chronological order DO
        source <- newest raw observation with source.time <= t

        IF source exists AND t - source.time <= max_hold_gap THEN
            x(t), y(t), theta_unwrapped(t) <- source values
            observation_age(t) <- t - source.time
            valid_pose(t) <- TRUE
        ELSE
            pose values <- missing
            valid_pose(t) <- FALSE
        END IF
    END FOR

    theta(t) <- wrap theta_unwrapped(t) back to [-pi, pi]
    RETURN aligned poses, source timestamps, ages, and valid_pose
END ALGORITHM
```

The lookup is backward-only. Therefore a state at time `t` never uses a future
measurement.

### 4.3 Past-window finite-difference kinematics

For a velocity window `T` represented by `L` grid steps:

```text
ALGORITHM EstimateKinematics(aligned_samples, lag_steps L)
    FOR each time index k DO
        IF k < L THEN
            valid_kinematics(k) <- FALSE
            vx(k), vy(k), omega(k) <- missing
            CONTINUE
        END IF

        delta_t <- time(k) - time(k-L)
        vx(k) <- [x(k) - x(k-L)] / delta_t
        vy(k) <- [y(k) - y(k-L)] / delta_t
        omega(k) <- [theta_unwrapped(k) - theta_unwrapped(k-L)] / delta_t

        valid_kinematics(k) <- every valid_pose value from k-L through k is TRUE
        IF valid_kinematics(k) is FALSE THEN
            vx(k), vy(k), omega(k) <- missing
        END IF

        speed(k) <- sqrt(vx(k)^2 + vy(k)^2)
    END FOR
END ALGORITHM
```

`valid_pose` and `valid_kinematics` are quality masks, not decision features.

## 5. Local flow coordinates

Corresponding function: `geometry.world_to_corridor`

The transformation utility supports any origin `o`, unit tangent `t`, and
normal `n = [-t_y, t_x]`. The mainline uses the upper JSON edge's midpoint as
`o`, with `t = (1, 0)` and `n = (0, 1)`.

```text
ALGORITHM TransformWorldToLocalFrame(samples, geometry)
    FOR each sample DO
        displacement <- [x, y] - origin
        s <- dot(displacement, tangent)
        d <- dot(displacement, normal)

        IF world-frame velocity exists THEN
            v_s <- dot([vx, vy], tangent)
            v_d <- dot([vx, vy], normal)
        END IF
    END FOR
    RETURN samples with s, d, and optional v_s, v_d
END ALGORITHM
```

## 6. Effective-splitter inference

Corresponding module: `splitter_inference.py`

### 6.1 Select the horizontal analysis edge

```text
ALGORITHM UpperReferenceEdge(rectangle_corners)
    REQUIRE four finite corners
    top <- the two corners with largest y
    REQUIRE abs(top[0].y - top[1].y) <= 0.05 m
    RETURN left x, right x, mean top y
END ALGORITHM
```

The route edge is a reproducible analysis reference, **not** a barrier-pose
annotation. The full trajectories identify the upper leg as the affected one.

### 6.2 Place interior sections

```text
origin_x <- midpoint(left x, right x)
inset <- 0.10 * (right x - left x)
section_grid <- x positions from left x + inset to right x - inset
                in steps of 0.04 m, expressed relative to origin_x
s <- x - origin_x; d <- y - top y
```

Excluding the two nominal corner regions avoids treating turn recrossings as
additional passages. No arc-length resampling, 2-D occupancy detection, or
tilted frame estimation occurs in the mainline.

### 6.3 Interpretation limit

The upper segment is specific to this interview run. A different geometry
would need its own analysis-region selection; the optional topology baseline
does not currently replace this mainline choice.

### 6.4 Interpolate forward section crossings

```text
ALGORITHM InterpolateSections(local_trajectories, section_grid, lateral_limit)
    FOR each robot trajectory DO
        sort samples by time

        FOR each section position s_section DO
            find consecutive valid samples i, i+1 satisfying
                s(i) < s_section <= s(i+1)

            FOR each such forward crossing DO
                alpha <- [s_section - s(i)] / [s(i+1) - s(i)]
                d_cross <- d(i) + alpha * [d(i+1) - d(i)]

                IF abs(d_cross) <= lateral_limit THEN
                    interpolate crossing time, v_s, and v_d with alpha
                    append one section-crossing record
                END IF
            END FOR
        END FOR
    END FOR
    RETURN crossing table
END ALGORITHM
```

### 6.5 Exact one-dimensional K-means with K = 2

```text
ALGORITHM KMeans1D(values)
    REQUIRE at least two finite values
    sorted_values, original_order <- stable sort(values)
    best_SSE <- infinity

    FOR split = 1 TO count(values)-1 DO
        left <- sorted_values before split
        right <- sorted_values from split onward
        centers <- [mean(left), mean(right)]
        SSE <- sum squared deviations from the corresponding center
        IF SSE < best_SSE THEN
            retain split and centers
        END IF
    END FOR

    labels <- lower cluster before retained split, upper cluster afterward
    map labels back to original sample order

    FOR each value i DO
        same <- same-cluster values excluding value i itself
        IF same is empty THEN silhouette(i) <- 0 AND CONTINUE
        a(i) <- mean absolute distance to same
        b(i) <- mean absolute distance to values in the other cluster
        silhouette(i) <- [b(i) - a(i)] / max[a(i), b(i)]
    END FOR

    RETURN ordered centers, labels, mean silhouette
END ALGORITHM
```

Every one-dimensional K-means partition is a split of the sorted samples.
Enumerating those splits is globally optimal for K=2, deterministic, and avoids
both a machine-learning dependency and Lloyd-iteration local minima.

### 6.6 Auxiliary baseline-variability reference

```text
ALGORITHM BaselineVariabilityReference(baseline_crossings, section_grid)
    initialize deterministic random generator
    reference_distances <- empty list

    FOR each section DO
        values <- baseline lateral crossings at this section
        IF count(values) < 2 * minimum_cluster_samples THEN CONTINUE

        REPEAT baseline_split_repeats times
            shuffled <- random permutation(values)
            split shuffled values into two nearly equal halves
            distance <- Wasserstein1D(first_half, second_half)
            append distance to reference_distances
        END REPEAT
    END FOR

    RETURN 95th percentile(reference_distances)
END ALGORITHM
```

This baseline-versus-baseline line helps read the plot. It is **not** a formal
null test, a significance threshold, or a gate-selection input: repeated laps
and adjacent sections are correlated.

### 6.7 Section diagnostics and stable core

At every section:

```text
ALGORITHM DiagnoseSection(baseline_d, barrier_d)
    W1 <- Wasserstein1D(baseline_d, barrier_d)

    IF count(barrier_d) >= 2 * minimum_cluster_samples THEN
        centers, labels, silhouette <- KMeans1D(barrier_d)
        counts <- samples in each cluster
        separation <- centers(1) - centers(0)
        midpoint <- mean(centers)
        baseline_median <- median(baseline_d)
        natural_path_fraction <- fraction of barrier_d within
            0.15 * separation of baseline_median

        stable_core_valid <-
            min(counts) >= 3
            AND separation >= 0.25 m
            AND centers(0) < baseline_median < centers(1)
            AND natural_path_fraction <= 0.10
    ELSE
        stable_core_valid <- FALSE
    END IF

    RETURN W1 and all cluster diagnostics
END ALGORITHM
```

A section has adequate trajectory support when both its baseline and barrier
crossing counts are at least 75% of their respective median non-zero section
counts.

```text
peak <- maximum W1 among supported sections
response_valid(section) <-
    section_support_valid AND W1(section) >= 0.20 * peak

core_candidate(section) <-
    response_valid(section)
    AND stable_core_valid(section)
```

### 6.8 Select contiguous intervals, fit separator, and place gates

```text
ALGORITHM SelectIntervalNearPeak(boolean_mask, response_peak_index)
    find all contiguous TRUE intervals
    IF an interval contains response_peak_index THEN
        RETURN that interval
    ELSE
        RETURN the longest TRUE interval
    END IF
END ALGORITHM
```

```text
ALGORITHM FitSeparatorAndGates(section_diagnostics)
    peak <- supported section with maximum W1
    response_interval <- SelectIntervalNearPeak(response_valid, peak)
    core_interval <- SelectIntervalNearPeak(core_candidate, peak)

    REQUIRE core_interval contains at least two stable sections

    FOR every section j in core_interval DO
        separator_observation(j) <- mean of its two K-means centers
    END FOR

    slope, intercept <- ordinary least-squares line through
        [section_s(j), separator_observation(j)]

    d_split(s) <- intercept + slope * s
    median_separation <- median cluster separation over core_interval
    ambiguity_half_width <- max(0.02 m, 0.10 * median_separation)

    G0 <- earliest adequately supported interior section before response start
    G1 <- first section of response_interval
    core_start <- first section of core_interval
    core_end <- last section of core_interval
    G2 <- midpoint(core_start, core_end)
    G3 <- last section of response_interval

    REQUIRE G0 < G1 <= core_start <= G2 <= core_end <= G3
    RETURN local frame, d_split, ambiguity width, G0, G1, G2, G3
END ALGORITHM
```

`G1` and `core_start` are intentionally different. `G1` marks an approximate
peak-relative response onset; `core_start` marks two separated routes after the
natural-path region is mostly vacated.
Neither locates the physical barrier endpoint or establishes statistical
significance.

## 7. Passage extraction and route classification

Corresponding module: `passages.py`

### 7.1 Forward gate crossing

```text
ALGORITHM FindForwardCrossings(trajectory, gate_s)
    FOR consecutive valid samples i, i+1 DO
        IF s(i) < gate_s <= s(i+1) THEN
            alpha <- [gate_s - s(i)] / [s(i+1) - s(i)]
            d_cross <- linear interpolation of d with alpha
            t_cross <- linear interpolation of time with alpha
            x_cross, y_cross <- linear interpolation of world position

            IF abs(d_cross) <= gate_half_width THEN
                append crossing(t_cross, following sample index,
                                d_cross, x_cross, y_cross)
            END IF
        END IF
    END FOR
    RETURN crossings in time order
END ALGORITHM
```

### 7.2 Pair G0 entries with G3 exits

```text
ALGORITHM DetectPassages(trajectories, geometry)
    FOR each condition/run/robot trajectory DO
        entries <- FindForwardCrossings(trajectory, G0)
        exits <- FindForwardCrossings(trajectory, G3)
        events <- chronological union of entries and exits
        active_entry <- none

        FOR each event DO
            IF event is G0 entry THEN
                IF active_entry exists THEN
                    retain old entry as invalid: superseded_by_next_G0
                END IF
                active_entry <- current entry
            ELSE IF event is G3 exit AND active_entry exists THEN
                pair active_entry with current exit
                active_entry <- none
            END IF
        END FOR

        IF active_entry remains at run end THEN
            retain it as invalid: missing_G3_before_run_end
        END IF

        FOR each retained passage record DO
            duration <- exit_time - entry_time when an exit exists
            IF duration outside [minimum_duration, maximum_duration] THEN
                passage is invalid: duration_out_of_range
            END IF

            interpolate G1 and G2 crossing times within segment
            classify route from the stable core
            compute completed path length from interpolated G0 position,
                through interior samples, to interpolated G3 position
            append one structured passage row
        END FOR
    END FOR
    RETURN passage table, ordered baseline then barrier
END ALGORITHM
```

### 7.3 Core-median route label

```text
ALGORITHM ClassifyRoute(segment, separator, core_interval)
    core_samples <- valid samples with core_start <= s <= core_end

    IF count(core_samples) < minimum_core_samples THEN
        RETURN unknown
    END IF

    offsets <- d(sample) - d_split(s(sample)) for every core sample
    route_score <- median(offsets)
    IF abs(route_score) <= ambiguity_half_width THEN
        label <- ambiguous
    ELSE IF route_score > 0 THEN
        label <- outer
    ELSE
        label <- inner
    END IF

    RETURN label, route_score, sample count
END ALGORITHM
```

The median uses the complete stable core rather than one instantaneous G2
sample, reducing sensitivity to tracking noise and brief lateral reversals.

### 7.4 Path length

```text
path_points <- [interpolated G0 position,
                valid samples strictly between G0 and G3,
                interpolated G3 position]

path_length <- sum over consecutive path_points of
    sqrt([x(k)-x(k-1)]^2 + [y(k)-y(k-1)]^2)

excess_path_length <- path_length - (G3 - G0)
```

## 8. Causal online decision state

Corresponding module: `online_state.py`

The output contains nine numerical components:

```text
x_dec(t) = [N_window, delta_t_last, N_approach,
            v_s_q33, v_s_q50, v_s_q67,
            d_q33, d_q50, d_q67]
```

### 8.1 Downsample the regular grid

```text
ALGORITHM DownsampleRun(preprocessed_run, target_rate)
    base_period <- median difference between consecutive grid timestamps
    stride <- round[1 / (target_rate * base_period)]
    retain every stride-th timestamp for all robots
    RETURN selected samples
END ALGORITHM
```

### 8.2 Build G0 arrival events

```text
ALGORITHM BuildCrossingEvents(trajectories, passages)
    FOR each passage entry at time t_G0 DO
        d_cross <- passage.d_G0
        nearest_sample <- same robot sample nearest to t_G0
        IF nearest_sample has valid kinematics THEN
            v_s_cross <- nearest_sample.v_s
        ELSE
            v_s_cross <- missing
        END IF
        append event(time=t_G0, d_cross, v_s_cross)
    END FOR
    RETURN events
END ALGORITHM
```

Using all detected G0 entries is causal: passage completion or its later route
label is not used to construct the event. The interpolated event is confirmed by
the following regular-grid sample; every decision-state timestamp that can use
the event is at or after that confirming sample.

### 8.3 Summarize state at time t

```text
ALGORITHM SummarizeOnlineState(robot_samples_at_t, all_G0_events, window)
    recent_events <- events satisfying t-window < event.time <= t
    prior_events  <- events satisfying event.time <= t

    N_window <- count(recent_events)
    delta_t_last <- t - time of most recent prior event, or missing

    N_approach <- number of valid robot poses satisfying
        G0 <= s < G1 AND abs(d) <= gate_half_width

    quantile_events <- recent events with valid d_cross and v_s_cross

    IF count(quantile_events) >= 3 THEN
        [v_s_q33, v_s_q50, v_s_q67] <- speed quantiles
        [d_q33, d_q50, d_q67] <- lateral-position quantiles
        quantiles_valid <- TRUE
    ELSE
        all six quantiles <- missing
        quantiles_valid <- FALSE
    END IF

    RETURN nine numerical state values plus timestamp/support metadata
END ALGORITHM
```

Every filter uses only events whose timestamps are no later than `t`.

## 9. Post-action evaluation metrics

Corresponding module: `metrics.py`

### 9.1 Route distribution

```text
ALGORITHM RouteDistribution(passages)
    valid <- passages that are valid and labelled inner/outer/ambiguous
    FOR each condition DO
        count each of the three labels
        proportion(label) <- count(label) / total valid labelled passages
    END FOR
    RETURN complete condition-by-label table, including zero counts
END ALGORITHM
```

Baseline proportions are set to not applicable in the compact evaluation because
the separator is inferred from the barrier condition.

### 9.2 Pooled passage time, with per-robot supporting view

```text
ALGORITHM PassageTimeTables(valid_passages)
    FOR each condition/run/robot DO
        passage_count <- number of valid passages
        median_passage_time <- median passage duration
        median_path_length <- median path length
        median_excess_length <- median excess path length
    END FOR

    FOR each condition DO
        condition_mean_time <- mean duration across all valid passages
        condition_median_excess_length <- median excess length across all
                                          valid passages
    END FOR
    relative_passage_time(changed) <- changed condition mean time /
                                      baseline condition mean time

    RETURN per-robot table and per-condition table
END ALGORITHM
```

The per-robot table exposes heterogeneity; it does not determine the pooled
headline comparison. No significance test is performed because the dataset
has only one run per condition.

### 9.3 Lower-tail nearest-pair distance

```text
ALGORITHM PairDistanceSummary(trajectories, target_rate, quantile=0.10)
    FOR each run DO
        downsample regular timestamps to target_rate

        FOR each retained timestamp DO
            robots <- valid poses with G1 <= s <= G3
            IF fewer than two robots THEN CONTINUE

            compute all pairwise Euclidean center distances
            ignore diagonal self-distances
            append minimum pair distance at this timestamp
        END FOR
    END FOR

    FOR each condition DO
        pair_distance_q10 <- 10th percentile of timestamp minima
    END FOR
    RETURN condition summary
END ALGORITHM
```

This is a robot-center spacing proxy, not physical clearance, because robot
radius is not supplied.

### 9.4 Compact six-component outcome vector

```text
y_eval = [inner_fraction, outer_fraction, ambiguous_fraction,
          relative_passage_time,
          median_excess_path_length,
          pair_distance_q10]
```

## 10. Figure-generation logic

Corresponding module: `plots.py`

```text
FIGURE 1: Full trajectories
    plot every valid world-frame path by robot and condition
    overlay nominal JSON route boundary
    use shared global x/y limits and equal 1:1 physical aspect
    place baseline panel before barrier panel

FIGURE 2: Inferred local geometry
    extract valid G0-to-G3 samples
    plot paths in local (s,d) coordinates, coloured by route label
    draw G0, G1, G2, G3
    draw effective separator and ambiguity band only in barrier panel
    restrict separator line to the stable core

FIGURE 3: Wasserstein response profile
    plot W1(section) and the auxiliary baseline-split reference
    shade peak-relative response interval and stable core

FIGURE 4: Representative G2 cross-section
    select sampled section nearest inferred G2
    place baseline and barrier samples on two exact categorical rows
    do not add vertical jitter
    draw K=2 barrier route centers and fitted separator

FIGURE 5: Barrier route allocation
    plot one horizontal stacked bar for inner/outer/ambiguous proportions
    omit baseline because route sides are not applicable

FIGURE 6: Per-robot median passage time (supporting view)
    pivot robot-level medians by condition
    connect the same robot from baseline to barrier
    preserve baseline-left, barrier-right order
```

## 11. Optional global path-topology baseline

Corresponding module: `topology.py`. This module is not called by the main
splitter pipeline.

### 11.1 Occupancy skeleton to directed graph

```text
ALGORITHM InferGlobalPathTopology(baseline_trajectories)
    spatially resample every baseline robot path
    combine points into one cloud

    occupancy <- 2-D histogram on a regular world-frame grid
    occupancy <- GaussianSmooth(occupancy)
    binary <- occupancy >= selected positive-occupancy quantile
    binary <- morphological closing(binary, radius-one disk)

    label connected components
    remove components smaller than minimum_component_pixels
    skeleton <- one-pixel-wide skeletonization(binary)

    create one undirected graph node per skeleton pixel
    connect every pair of 8-neighbour skeleton pixels
    edge length <- Euclidean distance between pixel centers

    PruneShortSpurs(graph)
    DirectEdgesFromVelocityVotes(graph, baseline velocities)

    RETURN directed graph, occupancy, skeleton, node table, edge table
END ALGORITHM
```

### 11.2 Short-spur pruning

```text
ALGORITHM PruneShortSpurs(graph, maximum_length)
    REPEAT until no deletion occurs
        FOR each degree-one leaf DO
            walk from leaf through degree-two nodes
            stop at termination or degree-three-plus junction
            accumulate branch length

            IF walk reaches a junction AND branch length < maximum_length THEN
                delete branch nodes but preserve the junction
                restart the scan
            END IF
        END FOR
    END REPEAT
END ALGORITHM
```

### 11.3 Direction from nearby velocity votes

```text
ALGORITHM DirectEdgesFromVelocityVotes(graph, baseline_samples)
    build KD-tree of active graph-node positions

    FOR each valid baseline velocity sample DO
        node <- nearest graph node to sample position

        FOR each neighbour of node DO
            displacement <- neighbour_position - node_position
            vote for canonical undirected edge +=
                signed dot(velocity, displacement)
        END FOR
    END FOR

    FOR each undirected edge DO
        orient edge according to sign of accumulated vote
        save absolute vote magnitude as direction confidence evidence
    END FOR
    RETURN directed graph
END ALGORITHM
```

### 11.4 Greedy distance-plus-heading map match

```text
ALGORITHM MapMatchTrajectory(trajectory, topology_graph)
    build KD-tree of graph-node positions
    previous_node <- none

    FOR each trajectory sample in time order DO
        candidates <- five nearest graph nodes
        cost(candidate) <- Euclidean distance to candidate

        IF previous_node exists AND candidate is neither previous_node
           nor adjacent to previous_node THEN
            cost(candidate) += continuity_penalty
        END IF

        IF velocity is valid AND candidate has directed successors THEN
            candidate_alignment <- maximum cosine alignment between
                velocity direction and outgoing successor directions
            cost(candidate) += heading_weight * (1 - candidate_alignment)
        END IF

        selected <- candidate with minimum cost
        previous_node <- selected
        append selected node ID
    END FOR
    RETURN trajectory with matched_node_id
END ALGORITHM
```

This map matcher is deliberately a transparent baseline. It is not a hidden
Markov model and does not claim globally optimal sequence matching.

## 12. Function-to-section index

| Python function | Pseudocode section |
|---|---:|
| `io.load_robot_file`, `load_run`, `load_dataset` | 3 |
| `preprocessing._build_time_grid`, `_resample_robot` | 4 |
| `geometry.world_to_corridor` | 5 |
| `splitter_inference._upper_reference_edge` | 6.1 |
| `splitter_inference._interpolate_sections` | 6.4 |
| `splitter_inference._kmeans_1d` | 6.5 |
| `splitter_inference._baseline_variability_reference` | 6.6 |
| `splitter_inference._interval_near_peak` | 6.8 |
| `splitter_inference.infer_effective_splitter` | 6 |
| `passages._crossings`, `detect_passages` | 7 |
| `passages._route_from_core` | 7.3 |
| `online_state.extract_online_state` and helpers | 8 |
| `metrics.route_distribution` | 9.1 |
| `metrics.passage_time_tables` | 9.2 |
| `metrics._pair_distance_summary` | 9.3 |
| `plots.generate_all_plots` and plot functions | 10 |
| `topology.infer_global_path_topology` | 11.1--11.3 |
| `topology._spatially_resample` | 11.1 |
| `topology.map_match_trajectory` | 11.4 |
| `pipeline.run_full_analysis` | 2 |
