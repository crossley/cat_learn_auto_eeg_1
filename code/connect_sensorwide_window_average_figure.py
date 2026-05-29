#!/usr/bin/env python3
"""Plot window-averaged sensorwide connectivity day-distance matrices."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connect_sensorwide_analysis import FIGURES_DIR, OUTPUT_DIR

DAYS = [1, 2, 3, 4, 5]
WINDOWS = ["early", "late"]
METRIC = "z_euclidean"


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing window-averaged connectivity output: {path}. "
            "Run connect_sensorwide_window_average_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty window-averaged connectivity output: {path}")
    return d


def matrix_offdiag_minmax_scaled(mat):
    out = np.full(mat.shape, np.nan, dtype=float)
    vals = []
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            if r != c and np.isfinite(val):
                vals.append(float(val))
    if len(vals) == 0:
        raise ValueError("No finite off-diagonal values to scale")
    arr = np.asarray(vals, dtype=float)
    val_min = float(np.min(arr))
    val_max = float(np.max(arr))
    denom = val_max - val_min
    for r in range(mat.shape[0]):
        for c in range(mat.shape[1]):
            val = mat[r, c]
            if not np.isfinite(val):
                continue
            if r == c:
                out[r, c] = 0.0
            elif denom <= np.finfo(float).eps:
                out[r, c] = 0.5
            else:
                out[r, c] = (float(val) - val_min) / denom
    return out


def group_matrix(d, window):
    g = d[(d["window"] == window) & (d["metric"] == METRIC)]
    if g.empty:
        raise ValueError(f"Missing {METRIC} group rows for window={window}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for row in g.itertuples(index=False):
        train_idx = DAYS.index(int(row.train_day))
        test_idx = DAYS.index(int(row.test_day))
        mat[train_idx, test_idx] = float(row.distance_mean)
    missing = []
    for train_day in DAYS:
        for test_day in DAYS:
            val = mat[DAYS.index(train_day), DAYS.index(test_day)]
            if not np.isfinite(val):
                missing.append(f"D{train_day}->D{test_day}")
    if len(missing) > 0:
        raise ValueError(
            f"Missing group distance cells for window={window}: "
            + ", ".join(missing)
        )
    return mat


def plot_window_average_z_euclidean(group_df, figures_dir):
    mats = {}
    display_mats = {}
    for window in WINDOWS:
        mat = group_matrix(group_df, window)
        mats[window] = mat
        display_mats[window] = matrix_offdiag_minmax_scaled(mat)

    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(8.8, 4.2), squeeze=False)
    for col, window in enumerate(WINDOWS):
        ax = axes[0, col]
        display_mat = display_mats[window]
        raw_mat = mats[window]
        im = ax.imshow(display_mat, vmin=0, vmax=1, cmap="viridis")
        ax.set_title(window)
        ax.set_xticks(np.arange(len(DAYS)))
        ax.set_yticks(np.arange(len(DAYS)))
        day_labels = []
        for day in DAYS:
            day_labels.append(f"D{day}")
        ax.set_xticklabels(day_labels)
        ax.set_yticklabels(day_labels)
        ax.set_xlabel("test day")
        if col == 0:
            ax.set_ylabel("train day")
        for r in range(display_mat.shape[0]):
            for c in range(display_mat.shape[1]):
                val = display_mat[r, c]
                raw_val = raw_mat[r, c]
                if not np.isfinite(val):
                    continue
                text_color = "white"
                if val > 0.65:
                    text_color = "black"
                ax.text(
                    c,
                    r,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    color=text_color,
                    fontsize=8,
                )
                if r == c and abs(raw_val) > np.finfo(float).eps:
                    raise ValueError("Expected zero same-day distance on diagonal")
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.04)
    cbar.set_label("off-diagonal scaled z-Euclidean distance")
    fig.suptitle("Window-Averaged Connectivity Day Distances")
    fig.tight_layout(rect=[0, 0, 0.94, 0.92])
    fig_path = (
        figures_dir
        / "connect_sensorwide_window_average_z_euclidean_scaled_matrices_"
        "top20_stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_connect_sensorwide_window_average(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    group_df = require_csv(
        output_dir / "connect_sensorwide_window_average_group_distances.csv"
    )
    fig_path = plot_window_average_z_euclidean(group_df, figures_dir)
    print(f"[connect window-average] wrote {fig_path}", flush=True)
    return {"z_euclidean_scaled_matrices": fig_path}


if __name__ == "__main__":
    save_fig_connect_sensorwide_window_average()
