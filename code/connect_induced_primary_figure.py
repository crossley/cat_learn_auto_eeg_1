#!/usr/bin/env python3
"""Figure for total-vs-induced primary dwPLI connectivity."""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connect_induced_primary_edge_timecourse_analysis import OUTPUT_PREFIX
from connect_multimeasure_utils import (
    FIGURES_DIR,
    OUTPUT_DIR,
    PRIMARY_BAND,
    PRIMARY_DECISION_WINDOW,
    PRIMARY_MEASURE,
)
from figure_style import DAYS, DAY_COLORS, setup_axis
from sensor_rois import ROI_LABELS

SIGNAL_STYLES = {
    "total": "-",
    "induced": "--",
}


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing induced-primary figure input: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty induced-primary figure input: {path}")
    return d


def save_connect_induced_primary_figure(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
    output_prefix=OUTPUT_PREFIX,
    figure_prefix=OUTPUT_PREFIX,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    roi_df = require_csv(output_dir / f"{output_prefix}_roi_timecourse_day_mean.csv")
    block_df = require_csv(
        output_dir / f"{output_prefix}_block_model_timecourse_summary.csv"
    )
    roi_df = roi_df[
        (roi_df["band"] == PRIMARY_BAND)
        & (roi_df["measure"] == PRIMARY_MEASURE)
        & (roi_df["lock_type"] == "stim")
    ].copy()
    block_df = block_df[
        (block_df["band"] == PRIMARY_BAND)
        & (block_df["measure"] == PRIMARY_MEASURE)
        & (block_df["model_label"] == "Continuous Restructuring")
    ].copy()
    if roi_df.empty:
        raise ValueError("No induced-primary ROI rows found")
    if block_df.empty:
        raise ValueError("No induced-primary continuous block-model rows found")

    fig = plt.figure(figsize=(11.0, 7.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.9])
    roi_pairs = ["visual_frontal", "visual_central"]
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    ax_block = fig.add_subplot(grid[1, :])

    for ax, roi_pair in zip(axes, roi_pairs):
        for signal_estimate, linestyle in SIGNAL_STYLES.items():
            for day in DAYS:
                g = roi_df[
                    (roi_df["roi_pair"] == roi_pair)
                    & (roi_df["signal_estimate"] == signal_estimate)
                    & (roi_df["day"] == day)
                ].sort_values("lock_time")
                if g.empty:
                    continue
                alpha = 0.95 if signal_estimate == "total" else 0.75
                ax.plot(
                    g["lock_time"].to_numpy(float),
                    g["conn_mean"].to_numpy(float),
                    color=DAY_COLORS[day],
                    linestyle=linestyle,
                    linewidth=1.8,
                    alpha=alpha,
                )
        ax.axvspan(
            PRIMARY_DECISION_WINDOW[0],
            PRIMARY_DECISION_WINDOW[1],
            color="0.5",
            alpha=0.10,
            linewidth=0,
        )
        ax.axvline(0.0, color="0.25", linewidth=0.8)
        ax.set_title(ROI_LABELS[roi_pair].title())
        ax.set_xlabel("time from stimulus (s)")
        setup_axis(ax)
    axes[0].set_ylabel("mean dwPLI")

    for signal_estimate, linestyle in SIGNAL_STYLES.items():
        g = block_df[block_df["signal_estimate"] == signal_estimate].sort_values(
            "time_sec"
        )
        if g.empty:
            continue
        x = g["time_sec"].to_numpy(float)
        y = -g["delta_bic_baseline_mean"].to_numpy(float)
        err = g["delta_bic_baseline_sem"].to_numpy(float)
        color = "#1f1f1f" if signal_estimate == "total" else "#d55e00"
        ax_block.plot(
            x,
            y,
            color=color,
            linestyle=linestyle,
            linewidth=2.2,
            label=signal_estimate.title(),
        )
        good = np.isfinite(err)
        if np.any(good):
            ax_block.fill_between(
                x[good],
                y[good] - err[good],
                y[good] + err[good],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
    ax_block.axhline(0.0, color="0.25", linewidth=0.8)
    ax_block.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    ax_block.axvspan(
        PRIMARY_DECISION_WINDOW[0],
        PRIMARY_DECISION_WINDOW[1],
        color="0.5",
        alpha=0.10,
        linewidth=0,
    )
    ax_block.set_xlabel("time from stimulus (s)")
    ax_block.set_ylabel("continuous evidence above baseline")
    ax_block.legend(frameon=False, loc="upper right")
    setup_axis(ax_block)

    day_handles = [
        plt.Line2D([0], [0], color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
        for day in DAYS
    ]
    signal_handles = [
        plt.Line2D([0], [0], color="0.2", linestyle=style, linewidth=2.0, label=label.title())
        for label, style in SIGNAL_STYLES.items()
    ]
    axes[1].legend(
        handles=day_handles + signal_handles,
        frameon=False,
        fontsize=8,
        loc="upper right",
    )
    fig.suptitle("Total vs Induced Primary Broadband dwPLI")
    fig.text(
        0.5,
        0.02,
        "Induced dwPLI subtracts the per-subject, per-condition ERP before connectivity; "
        "a surviving effect is not merely evoked field spread.",
        ha="center",
        va="bottom",
        fontsize=9,
    )
    fig.tight_layout(rect=[0.0, 0.05, 1.0, 0.96])
    path = figures_dir / f"{figure_prefix}_total_vs_induced_dwpli.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[connect induced primary figure] wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    save_connect_induced_primary_figure()
