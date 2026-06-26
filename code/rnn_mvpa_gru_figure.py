#!/usr/bin/env python3
"""Plot GRU early-exit MVPA transfer outputs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import DAYS, DAY_COLORS, FIGURES_DIR, setup_axis
from rnn_mvpa_gru_analysis import OUTPUT_DIR

SELECTED_MATRIX_END_TIMES = [0.20, 0.40, 0.60, 0.80]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing RNN MVPA output: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty RNN MVPA output: {path}")
    return d


def save_early_exit_auc(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "rnn_mvpa_gru_transfer_day_mean.csv")
    d = d[d["train_day"] == d["test_day"]].copy()

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for day in DAYS:
        g = d[d["train_day"] == day].sort_values("end_time_sec")
        if g.empty:
            continue
        x = g["end_time_sec"].to_numpy(float)
        y = g["auc_mean"].to_numpy(float)
        err = g["auc_sem"].to_numpy(float)
        ax.plot(x, y, color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
        ax.fill_between(
            x,
            y - err,
            y + err,
            color=DAY_COLORS[day],
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(0.5, color="0.25", linewidth=0.8)
    ax.set_xlabel("sequence end time from stimulus (s)")
    ax.set_ylabel("AUC")
    ax.set_title("GRU Early-Exit Category Decoding")
    ax.legend(frameon=False, loc="upper left")
    setup_axis(ax)
    fig.tight_layout()
    path = figures_dir / "rnn_mvpa_gru_early_exit_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RNN MVPA figure] wrote {path}", flush=True)
    return path


def matrix_for_end_time(d, end_time_sec):
    g = d[np.isclose(d["end_time_sec"], end_time_sec)].copy()
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["train_day"]))
        j = DAYS.index(int(row["test_day"]))
        mat[i, j] = float(row["auc_mean"])
    return mat


def save_transfer_matrices(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "rnn_mvpa_gru_transfer_day_mean.csv")

    fig, axes = plt.subplots(1, len(SELECTED_MATRIX_END_TIMES), figsize=(11.5, 3.1))
    if len(SELECTED_MATRIX_END_TIMES) == 1:
        axes = [axes]
    vals = d["auc_mean"].to_numpy(float)
    vmin = max(0.45, float(np.nanpercentile(vals, 5)))
    vmax = min(0.75, float(np.nanpercentile(vals, 95)))
    if vmax <= vmin:
        vmin, vmax = 0.45, 0.75
    last_im = None
    for ax, end_time_sec in zip(axes, SELECTED_MATRIX_END_TIMES):
        mat = matrix_for_end_time(d, end_time_sec)
        last_im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"through {end_time_sec:.1f} s")
        ax.set_xticks(range(len(DAYS)))
        ax.set_yticks(range(len(DAYS)))
        ax.set_xticklabels([f"D{d}" for d in DAYS])
        ax.set_yticklabels([f"D{d}" for d in DAYS])
        ax.set_xlabel("test day")
        ax.set_ylabel("train day")
        for i in range(len(DAYS)):
            for j in range(len(DAYS)):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if mat[i, j] < (vmin + vmax) / 2 else "black",
                    )
    fig.suptitle("GRU Matched-Time Cross-Day Transfer")
    fig.tight_layout(rect=[0, 0, 0.95, 0.92])
    cax = fig.add_axes([0.96, 0.18, 0.012, 0.62])
    fig.colorbar(last_im, cax=cax, label="AUC")
    path = figures_dir / "rnn_mvpa_gru_transfer_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RNN MVPA figure] wrote {path}", flush=True)
    return path


def save_model_timecourse(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "rnn_mvpa_gru_model_timecourse_summary.csv")

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = {
        "Continuous Restructuring": "#1b9e77",
        "Discrete Restructuring D1": "#440154",
        "Discrete Restructuring D2": "#3b528b",
        "Discrete Restructuring D3": "#21918c",
        "Discrete Restructuring D4": "#5ec962",
    }
    labels = list(colors)
    for label in labels:
        g = d[d["model_label"] == label].sort_values("end_time_sec")
        if g.empty:
            continue
        x = g["end_time_sec"].to_numpy(float)
        y = -g["delta_bic_baseline_mean"].to_numpy(float)
        err = g["delta_bic_baseline_sem"].to_numpy(float)
        ax.plot(x, y, color=colors[label], linewidth=2.0, label=label)
        ax.fill_between(
            x,
            y - err,
            y + err,
            color=colors[label],
            alpha=0.10,
            linewidth=0,
        )
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("sequence end time from stimulus (s)")
    ax.set_ylabel("BIC support over baseline")
    ax.set_title("GRU Transfer Model Evidence Over Time")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    setup_axis(ax)
    fig.tight_layout()
    path = figures_dir / "rnn_mvpa_gru_model_timecourse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[RNN MVPA figure] wrote {path}", flush=True)
    return path


def save_rnn_mvpa_gru_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    return {
        "early_exit_auc": save_early_exit_auc(output_dir, figures_dir),
        "transfer_matrices": save_transfer_matrices(output_dir, figures_dir),
        "model_timecourse": save_model_timecourse(output_dir, figures_dir),
    }


if __name__ == "__main__":
    save_rnn_mvpa_gru_figures()
