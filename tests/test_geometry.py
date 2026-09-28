from __future__ import annotations

import numpy as np
import pandas as pd

from reflow_analysis.geometry import AnalysisGeometry, corridor_to_world, world_to_corridor


def test_corridor_coordinate_round_trip():
    geometry = AnalysisGeometry()
    source_s = np.array([-0.6, 0.0, 0.45])
    source_d = np.array([-0.2, 0.0, 0.3])
    x, y = corridor_to_world(source_s, source_d, geometry)
    transformed = world_to_corridor(pd.DataFrame({"x": x, "y": y}), geometry)

    np.testing.assert_allclose(transformed["s"], source_s)
    np.testing.assert_allclose(transformed["d"], source_d)

