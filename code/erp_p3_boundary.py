#!/usr/bin/env python3
"""Parietal P3 × boundary distance analysis."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

P3_CHANNELS = ["Pz", "P3", "P4"]
P3_TMIN = 0.300
P3_TMAX = 0.600


def run_p3_boundary_analysis(
    p3_channels: list[str] = P3_CHANNELS,
    p3_tmin: float = P3_TMIN,
    p3_tmax: float = P3_TMAX,
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    raise NotImplementedError("erp_p3_boundary analysis is not yet implemented.")


if __name__ == "__main__":
    run_p3_boundary_analysis()
