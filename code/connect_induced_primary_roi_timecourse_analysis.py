#!/usr/bin/env python3
"""Collapse total-vs-induced primary dwPLI edges into ROI time courses."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

from connect_induced_primary_edge_timecourse_analysis import OUTPUT_PREFIX
from connect_multimeasure_roi_timecourse_analysis import (
    run_connect_multimeasure_roi_timecourse,
)
from connect_multimeasure_utils import OUTPUT_DIR


def run_connect_induced_primary_roi_timecourse(
    output_dir=OUTPUT_DIR,
    input_prefix=OUTPUT_PREFIX,
    output_prefix=OUTPUT_PREFIX,
):
    return run_connect_multimeasure_roi_timecourse(
        output_dir=Path(output_dir),
        input_prefix=input_prefix,
        output_prefix=output_prefix,
        extra_group_cols=["signal_estimate", "lock_type", "band", "measure"],
    )


if __name__ == "__main__":
    run_connect_induced_primary_roi_timecourse()
