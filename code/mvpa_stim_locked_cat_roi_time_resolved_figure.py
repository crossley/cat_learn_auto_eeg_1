#!/usr/bin/env python3
"""Plot stimulus-locked category decoding by strict sensor ROI."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mvpa_stim_locked_cat_roi_time_resolved_analysis import OUTPUT_DIR
from presentation_figure import DAYS, DAY_COLORS, FIGURES_DIR, setup_axis
from sensor_rois import ROI_LABELS


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MVPA ROI output: {path}. "
            "Run mvpa_stim_locked_cat_roi_time_resolved_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty MVPA ROI output: {path}")
    return d


def save_fig_mvpa_stim_locked_cat_roi_time_resolved(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "mvpa_stim_locked_cat_roi_day_means_timecourse.csv")
    rois = ["frontal", "central", "visual"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharey=True)
    for ax, roi in zip(axes, rois):
        for day in DAYS:
            g = d[(d["roi"] == roi) & (d["day"] == day)].sort_values("time_sec")
            if g.empty:
                continue
            x = g["time_sec"].to_numpy(dtype=float)
            y = g["auc_mean"].to_numpy(dtype=float)
            err = g["auc_sem"].to_numpy(dtype=float)
            ax.plot(x, y, color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
            good = np.isfinite(err)
            if np.any(good):
                ax.fill_between(
                    x[good],
                    y[good] - err[good],
                    y[good] + err[good],
                    color=DAY_COLORS[day],
                    alpha=0.12,
                    linewidth=0,
                )
        ax.axhline(0.5, color="0.25", linewidth=0.8)
        ax.set_xlabel("time from stimulus (s)")
        ax.set_title(ROI_LABELS[roi].title())
        setup_axis(ax)
    axes[0].set_ylabel("AUC")
    axes[-1].legend(frameon=False, ncol=1, loc="upper left")
    fig.suptitle("Time-Resolved Category Decoding by Sensor Region")
    fig.tight_layout()
    path = figures_dir / "mvpa_stim_locked_cat_roi_time_resolved_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[MVPA ROI] wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_roi_time_resolved()
