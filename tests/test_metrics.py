from __future__ import annotations

import pandas as pd

from reflow_analysis.metrics import passage_time_tables, route_distribution


def test_route_distribution_and_robot_level_passage_aggregation():
    passages = pd.DataFrame(
        {
            "condition": ["baseline"] * 4 + ["barrier_v01"] * 4,
            "run_name": ["run_a"] * 4 + ["run_b"] * 4,
            "robot_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "valid": [True, True, True, False, True, True, True, True],
            "route_label": [
                "inner",
                "outer",
                "ambiguous",
                "unknown",
                "inner",
                "inner",
                "inner",
                "outer",
            ],
            "passage_time_s": [2.0, 4.0, 3.0, None, 5.0, 7.0, 6.0, 8.0],
            "path_length_m": [1.0] * 8,
            "excess_path_length_m": [0.1] * 8,
            "passage_id": range(8),
        }
    )

    distribution = route_distribution(passages)
    per_robot, condition = passage_time_tables(passages)

    baseline = distribution[distribution["condition"] == "baseline"].set_index("route_label")
    assert baseline.loc["inner", "passage_count"] == 1
    assert baseline.loc["inner", "route_proportion"] == 1 / 3
    robot_one = per_robot[
        (per_robot["condition"] == "baseline") & (per_robot["robot_id"] == 1)
    ].iloc[0]
    assert robot_one["mean_passage_time_s"] == 3.0
    assert condition.loc[condition["condition"] == "barrier_v01", "passage_count"].iloc[0] == 4
