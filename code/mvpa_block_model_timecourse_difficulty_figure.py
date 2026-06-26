#!/usr/bin/env python3
"""Plot MVPA block-model timecourses by stimulus difficulty."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from block_model_timecourse_figure import split_block_color
from connect_sensorwide_analysis import FIGURES_DIR
from figure_style import setup_axis
from mvpa_block_model_timecourse_difficulty_analysis import OUTPUT_DIR

DIFFICULTIES = ["easy", "difficult"]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MVPA block difficulty output: {path}. "
            "Run mvpa_block_model_timecourse_difficulty_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty MVPA block difficulty output: {path}")
    return d


def plot_panel(ax, d, difficulty):
    d = d[d["difficulty"] == difficulty].copy()
    continuous = d[d["model_label"] == "Continuous Restructuring"].sort_values("time_sec")
    if not continuous.empty:
        x = continuous["time_sec"].to_numpy(float)
        y = -continuous["delta_bic_baseline_mean"].to_numpy(float)
        err = continuous["delta_bic_baseline_sem"].to_numpy(float)
        ax.plot(x, y, color="#1f1f1f", linewidth=2.1, label="Continuous")
        good = np.isfinite(err)
        if np.any(good):
            ax.fill_between(
                x[good],
                y[good] - err[good],
                y[good] + err[good],
                color="#1f1f1f",
                alpha=0.10,
                linewidth=0,
            )
    split_rows = d[d["model"] == "discrete"].copy()
    split_rows = split_rows[np.isfinite(split_rows["split_block"].astype(float))]
    for split_block in sorted(split_rows["split_block"].astype(float).dropna().unique()):
        g = split_rows[
            np.isclose(split_rows["split_block"].astype(float), float(split_block))
        ].sort_values("time_sec")
        x = g["time_sec"].to_numpy(float)
        y = -g["delta_bic_baseline_mean"].to_numpy(float)
        ax.plot(
            x,
            y,
            color=split_block_color(split_block),
            linewidth=1.0,
            alpha=0.78,
            label=f"B{int(split_block)}",
        )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    ax.set_xlabel("time from stimulus (s)")
    ax.set_title(difficulty.title())
    setup_axis(ax)


def save_fig_mvpa_block_model_timecourse_difficulty(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "mvpa_block_model_timecourse_difficulty_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), sharey=True)
    for ax, difficulty in zip(axes, DIFFICULTIES):
        plot_panel(ax, d, difficulty)
    axes[0].set_ylabel("Evidence above baseline model")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        title="Continuous and transition after block",
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        ncol=9,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
    )
    fig.suptitle("MVPA Block Model Evidence by Stimulus Difficulty")
    fig.tight_layout(rect=[0.0, 0.14, 1.0, 1.0])
    path = figures_dir / "mvpa_block_model_timecourse_difficulty.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[MVPA block difficulty figure] wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    save_fig_mvpa_block_model_timecourse_difficulty()
