#!/usr/bin/env python3
"""MVPA classifier confidence (decision-boundary distance) time-resolved analysis."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "mvpa_classifier_confidence"
FIGURES_DIR = PROJECT_DIR / "figures" / "mvpa_classifier_confidence"


def run_classifier_confidence_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    raise NotImplementedError("mvpa_classifier_confidence analysis is not yet implemented.")


if __name__ == "__main__":
    run_classifier_confidence_analysis()
