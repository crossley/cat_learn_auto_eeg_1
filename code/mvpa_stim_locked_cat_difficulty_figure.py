#!/usr/bin/env python3
"""Plot stimulus-locked category MVPA by stimulus difficulty."""

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
from mvpa_stim_locked_cat_difficulty_analysis import OUTPUT_DIR

DIFFICULTIES = ["easy", "difficult"]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MVPA difficulty output: {path}. "
            "Run mvpa_stim_locked_cat_difficulty_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty MVPA difficulty output: {path}")
    return d


def save_fig_mvpa_stim_locked_cat_difficulty(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "mvpa_stim_locked_cat_difficulty_day_means_timecourse.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True)
    for ax, difficulty in zip(axes, DIFFICULTIES):
        for day in DAYS:
            g = d[
                (d["difficulty"] == difficulty)
                & (d["day"] == day)
            ].sort_values("time_sec")
            if g.empty:
                continue
            x = g["time_sec"].to_numpy(float)
            y = g["auc_mean"].to_numpy(float)
            err = g["auc_sem"].to_numpy(float)
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
        ax.set_title(difficulty.title())
        setup_axis(ax)
    axes[0].set_ylabel("AUC")
    axes[-1].legend(frameon=False, ncol=1, loc="upper left")
    fig.suptitle("Time-Resolved Category Decoding by Stimulus Difficulty")
    fig.tight_layout()
    path = figures_dir / "mvpa_stim_locked_cat_difficulty_time_resolved_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[MVPA difficulty figure] wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_difficulty()
