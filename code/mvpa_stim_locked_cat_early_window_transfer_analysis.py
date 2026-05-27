#!/usr/bin/env python3
"""Early-window stimulus-locked category transfer across training days."""

from __future__ import annotations

from pathlib import Path

from mvpa_stim_locked_cat_late_window_analysis import OUTPUT_DIR, RANDOM_STATE
from mvpa_stim_locked_cat_late_window_transfer_analysis import (
    run_mvpa_stim_locked_cat_late_window_transfer,
)
from mvpa_stim_locked_cat_tg_window_structure_analysis import MVPA_CAT_TG_WINDOWS

WINDOW = "early"
WINDOW_START_SEC = MVPA_CAT_TG_WINDOWS[WINDOW][0]
WINDOW_END_SEC = MVPA_CAT_TG_WINDOWS[WINDOW][1]


def run_mvpa_stim_locked_cat_early_window_transfer(
    output_dir: Path | str = OUTPUT_DIR,
    min_epochs: int = 20,
    random_state: int = RANDOM_STATE,
    n_workers: int | None = None,
):
    return run_mvpa_stim_locked_cat_late_window_transfer(
        output_dir=output_dir,
        min_epochs=min_epochs,
        random_state=random_state,
        n_workers=n_workers,
        window=WINDOW,
        window_start_sec=WINDOW_START_SEC,
        window_end_sec=WINDOW_END_SEC,
        file_window=WINDOW,
    )


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_early_window_transfer()
