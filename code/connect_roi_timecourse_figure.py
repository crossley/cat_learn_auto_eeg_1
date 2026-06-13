#!/usr/bin/env python3
"""Plot strict cross-region connectivity time courses."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connect_sensorwide_analysis import OUTPUT_DIR, FIGURES_DIR
from presentation_figure import DAYS, DAY_COLORS, setup_axis
from sensor_rois import ROI_LABELS


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing connectivity ROI output: {path}. "
            "Run connect_roi_timecourse_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty connectivity ROI output: {path}")
    return d


def save_fig_connect_roi_timecourse(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "connect_roi_timecourse_day_mean.csv")
    roi_pairs = ["visual_frontal", "visual_central"]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    for ax, roi_pair in zip(axes, roi_pairs):
        for day in DAYS:
            g = d[(d["roi_pair"] == roi_pair) & (d["day"] == day)].sort_values("lock_time")
            if g.empty:
                continue
            x = g["lock_time"].to_numpy(dtype=float)
            y = g["conn_mean"].to_numpy(dtype=float)
            err = g["conn_sem"].to_numpy(dtype=float)
            ax.plot(x, y, color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
            good = np.isfinite(err)
            if np.any(good):
                ax.fill_between(
                    x[good],
                    y[good] - err[good],
                    y[good] + err[good],
                    color=DAY_COLORS[day],
                    alpha=0.13,
                    linewidth=0,
                )
        ax.axvline(0.0, color="0.25", linewidth=0.8)
        ax.set_xlabel("time from stimulus (s)")
        ax.set_title(ROI_LABELS[roi_pair].title())
        setup_axis(ax)
    axes[0].set_ylabel("mean connectivity")
    axes[1].legend(frameon=False, ncol=1, loc="upper right")
    fig.suptitle("Functional Connectivity Time Series by Sensor Region")
    fig.tight_layout()
    path = figures_dir / "connect_roi_timecourse_visual_frontal_central.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[connect ROI] wrote {path}", flush=True)
    return path


def save_fig_connect_roi_timecourse_by_day(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "connect_roi_timecourse_day_mean.csv")
    roi_specs = [
        ("visual_frontal", "#7b3294"),
        ("visual_central", "#1b9e77"),
    ]
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.4), sharex=True, sharey=True)
    for ax, day in zip(axes, DAYS):
        for roi_pair, color in roi_specs:
            g = d[(d["roi_pair"] == roi_pair) & (d["day"] == day)].sort_values("lock_time")
            if g.empty:
                continue
            x = g["lock_time"].to_numpy(dtype=float)
            y = g["conn_mean"].to_numpy(dtype=float)
            err = g["conn_sem"].to_numpy(dtype=float)
            ax.plot(
                x,
                y,
                color=color,
                linewidth=2.0,
                label=ROI_LABELS[roi_pair].title(),
            )
            good = np.isfinite(err)
            if np.any(good):
                ax.fill_between(
                    x[good],
                    y[good] - err[good],
                    y[good] + err[good],
                    color=color,
                    alpha=0.12,
                    linewidth=0,
                )
        ax.axvline(0.0, color="0.25", linewidth=0.8)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("time from stimulus (s)")
        setup_axis(ax)
    axes[0].set_ylabel("mean connectivity")
    axes[-1].legend(frameon=False, loc="upper right", fontsize=8)
    fig.suptitle("Functional Connectivity Time Series by Day and Sensor Region")
    fig.tight_layout()
    path = figures_dir / "connect_roi_timecourse_by_day_visual_frontal_central.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[connect ROI] wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    save_fig_connect_roi_timecourse()
    save_fig_connect_roi_timecourse_by_day()
