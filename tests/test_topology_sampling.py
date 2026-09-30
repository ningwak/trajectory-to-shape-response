from __future__ import annotations

import numpy as np
import pandas as pd

from reflow_analysis.topology import _spatially_resample


def test_optional_topology_sampler_handles_repeated_poses():
    """The optional module imports independently of splitter inference."""

    path = pd.DataFrame(
        {
            "time": pd.date_range("2026-01-01", periods=3, freq="s", tz="UTC"),
            "x": [0.0, 0.0, 1.0],
            "y": [0.0, 0.0, 0.0],
        }
    )

    samples = _spatially_resample(path, step_m=0.25)

    np.testing.assert_allclose(samples[:, 0], [0.0, 0.25, 0.5, 0.75])
    np.testing.assert_allclose(samples[:, 1], 0.0)
