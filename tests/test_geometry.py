"""Executable example of projecting a rotated/translated world frame to (s,d)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from reflow_analysis.geometry import AnalysisGeometry, world_to_corridor


def test_world_to_corridor_uses_the_inferred_frame():
    # Tangent points along world +y and normal along world -x.  Constructing
    # world points analytically from known s/d values makes expected projections
    # independent of the implementation under test.
    geometry = AnalysisGeometry(
        origin_x=1.0,
        origin_y=2.0,
        tangent_x=0.0,
        tangent_y=1.0,
        normal_x=-1.0,
        normal_y=0.0,
    )
    source_s = np.array([-0.6, 0.0, 0.45])
    source_d = np.array([-0.2, 0.0, 0.3])
    x = geometry.origin_x - source_d
    y = geometry.origin_y + source_s
    transformed = world_to_corridor(pd.DataFrame({"x": x, "y": y}), geometry)

    np.testing.assert_allclose(transformed["s"], source_s)
    np.testing.assert_allclose(transformed["d"], source_d)
