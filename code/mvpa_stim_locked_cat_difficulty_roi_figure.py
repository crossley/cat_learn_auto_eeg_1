#!/usr/bin/env python3
"""Plot strict-ROI category MVPA by stimulus difficulty."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import DAYS, DAY_COLORS, FIGURES_DIR, setup_axis
from mvpa_stim_locked_cat_difficulty_roi_analysis import OUTPUT_DIR
from sensor_rois import ROI_LABELS

DIFFICULTIES = ["easy", "difficult"]
ROIS = ["frontal", "central", "visual"]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MVPA difficulty ROI output: {path}. "
            "Run mvpa_stim_locked_cat_difficulty_roi_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty MVPA difficulty ROI output: {path}")
    return d


def save_fig_mvpa_stim_locked_cat_difficulty_roi(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "mvpa_stim_locked_cat_difficulty_roi_day_means_timecourse.csv")
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 8.6), sharex=True, sharey=True)
    for row_i, roi in enumerate(ROIS):
        for col_i, difficulty in enumerate(DIFFICULTIES):
            ax = axes[row_i, col_i]
            for day in DAYS:
                g = d[
                    (d["roi"] == roi)
                    & (d["difficulty"] == difficulty)
                    & (d["day"] == day)
                ].sort_values("time_sec")
                if g.empty:
                    continue
                x = g["time_sec"].to_numpy(float)
                y = g["auc_mean"].to_numpy(float)
                err = g["auc_sem"].to_numpy(float)
                ax.plot(x, y, color=DAY_COLORS[day], linewidth=1.8, label=f"D{day}")
                good = np.isfinite(err)
                if np.any(good):
                    ax.fill_between(
                        x[good],
                        y[good] - err[good],
                        y[good] + err[good],
                        color=DAY_COLORS[day],
                        alpha=0.10,
                        linewidth=0,
                    )
            ax.axhline(0.5, color="0.25", linewidth=0.8)
            if row_i == 0:
                ax.set_title(difficulty.title())
            if col_i == 0:
                ax.set_ylabel(f"{ROI_LABELS[roi].title()}\nAUC")
            if row_i == len(ROIS) - 1:
                ax.set_xlabel("time from stimulus (s)")
            setup_axis(ax)
    axes[0, -1].legend(frameon=False, ncol=1, loc="upper left")
    fig.suptitle("Time-Resolved Category Decoding by Sensor Region and Difficulty")
    fig.tight_layout()
    path = figures_dir / "mvpa_stim_locked_cat_difficulty_roi_time_resolved_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[MVPA difficulty ROI figure] wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_difficulty_roi()
