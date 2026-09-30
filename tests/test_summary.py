"""Check the interview-facing summary without depending on private JSON data."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from reflow_analysis.pipeline import _summary


def test_history_means_keep_support_and_list_ambiguous_passages():
    time = pd.Timestamp("2026-01-01", tz="UTC")
    states = pd.DataFrame(
        {
            "condition": ["baseline", "baseline", "barrier_v01"],
            "run_name": ["base", "base", "changed"],
            "time": [time, time + pd.Timedelta(seconds=0.5), time],
            "crossing_count_window": [0, 2, 1],
            "time_since_last_crossing_s": [np.nan, 3.0, np.nan],
            "n_approach": [0, 1, 1],
            "quantiles_valid": [False, True, False],
        }
    )
    for name in (
        "v_s_q33_mps",
        "v_s_q50_mps",
        "v_s_q67_mps",
        "d_q33_m",
        "d_q50_m",
        "d_q67_m",
    ):
        states[name] = [np.nan, 0.2, np.nan]

    passages = pd.DataFrame(
        {
            "condition": ["barrier_v01", "barrier_v01"],
            "run_name": ["changed", "changed"],
            "robot_id": [25, 50],
            "passage_id": [2, 1],
            "valid": [True, False],
            "route_label": ["ambiguous", "ambiguous"],
            "t_g0": [time, time],
            "route_score_m": [0.01, 0.02],
            "passage_time_s": [5.0, np.nan],
        }
    )
    evaluation = {
        "compact_evaluation": pd.DataFrame(
            {"condition": ["baseline", "barrier_v01"], "ambiguous": [np.nan, 0.1]}
        ),
        "passage_time_summary": pd.DataFrame(
            {
                "condition": ["baseline", "barrier_v01"],
                "mean_passage_time_s": [5.0, 6.0],
            }
        ),
        "route_distribution": pd.DataFrame(
            {
                "condition": ["barrier_v01"],
                "route_label": ["ambiguous"],
                "passage_count": [1],
                "route_proportion": [1.0],
            }
        ),
    }

    summary = _summary(evaluation, pd.DataFrame(index=range(2)), passages, states)
    base, barrier = summary["online_state_history_means"]
    assert base["crossing_count_window_mean"] == 1.0
    assert base["time_since_last_crossing_s_mean"] == 3.0
    assert base["quantiles_valid_fraction"] == 0.5
    assert base["v_s_q50_mps_mean"] == 0.2
    assert barrier["v_s_q50_mps_mean"] is None
    assert summary["compact_evaluation"][0]["ambiguous"] is None
    assert [
        (item["robot_id"], item["passage_id"])
        for item in summary["ambiguous_passages"]
    ] == [(25, 2)]
    json.dumps(summary, allow_nan=False)
