"""Row-level feature derivations shared by training and serving.

Pure Python with no pandas or scikit-learn, so the slim serving image can
import it. Training applies the same functions (vectorized in
transform.add_row_features), so a feature is computed one way everywhere.
"""

from __future__ import annotations

import math

DENSITY_FLOOR = 1.0


def log_density(density: float | None) -> float:
    """log of population density, with zero, negative or missing values clipped to 1."""
    if density is None or (isinstance(density, float) and math.isnan(density)):
        return math.log(DENSITY_FLOOR)
    return math.log(max(float(density), DENSITY_FLOOR))
