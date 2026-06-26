#!/usr/bin/env python3
"""Stimulus difficulty labels from distance to the true category bound."""

from __future__ import annotations

import numpy as np


def add_bound_difficulty(beh):
    missing = [col for col in ["x", "y"] if col not in beh.columns]
    if missing:
        raise ValueError(f"Missing stimulus coordinate columns: {missing}")
    d = beh.copy()
    d["bound_distance"] = np.abs(d["x"].astype(float) - d["y"].astype(float)) / np.sqrt(2.0)
    lo = float(d["bound_distance"].quantile(1.0 / 3.0))
    hi = float(d["bound_distance"].quantile(2.0 / 3.0))
    d["difficulty"] = ""
    d.loc[d["bound_distance"] <= lo, "difficulty"] = "difficult"
    d.loc[d["bound_distance"] >= hi, "difficulty"] = "easy"
    return d
