"""The 85 decision-time features per entry (the CPU baselines' compact set) as MLP input.

One row per entry instead of a candle window: indicators on M1/M5/M15/M30 at the last closed minute,
returns over 1 minute to 24 hours, realized volatility, the 18 main anchors now and in the last 120 minutes,
and hour / weekday as sine-cosine pairs. Computed by trade_baselines.decision_features, so the MLP and the
baselines see exactly the same numbers.
"""
import sys
from pathlib import Path

import numpy as np

from .data import mmap_array
from .schema import WIDTH

BASELINES_DIR = Path(__file__).resolve().parents[3] / "Models" / "Baselines"


def compact_matrix(data):
    """(rows for every entry, feature names); only valid rows are filled."""
    if str(BASELINES_DIR) not in sys.path:
        sys.path.insert(0, str(BASELINES_DIR))
    from trade_baselines import decision_features

    timeline = mmap_array(data.paths.timeline, "timeline", (WIDTH,))
    bounded = data.schema.get("macd_normalization") == "x / sqrt(1 + x^2)"
    rows = data.valid
    features, names = decision_features(timeline, data.history_rows[rows], data.times[rows], bounded)
    matrix = np.zeros((len(data.ids), features.shape[1]), dtype=np.float32)
    matrix[rows] = features
    return matrix, names
