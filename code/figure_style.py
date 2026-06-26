#!/usr/bin/env python3
"""Shared figure styling for project plots."""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

DAYS = [1, 2, 3, 4, 5]
DAY_COLORS = {
    1: "#440154",
    2: "#3b528b",
    3: "#21918c",
    4: "#5ec962",
    5: "#fde725",
}


def setup_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def sem(values):
    arr = values.dropna().to_numpy(float)
    if len(arr) <= 1:
        return float("nan")
    return float(arr.std(ddof=1) / (len(arr) ** 0.5))
