#!/usr/bin/env python3
"""Plot stimulus-locked cross-day temporal generalization figures."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from figure_style import DAYS
from mvpa_stim_locked_cat_tg_analysis import FIGURES_DIR, OUTPUT_DIR


def template_matrix(kind, split_day=None):
    mat = np.full((5, 5), np.nan)
    for train_day in DAYS:
        for test_day in DAYS:
            if kind == "gradual":
                val = 0.65 * min(train_day, test_day) / float(max(DAYS))
                if train_day == test_day:
                    val = train_day / float(max(DAYS))
            elif kind == "split_gradual":
                if split_day is None:
                    raise ValueError("split_gradual requires split_day")
                train_late = train_day > split_day
                test_late = test_day > split_day
                if train_late != test_late:
                    val = 0.0
                else:
                    val = 0.65 * min(train_day, test_day) / float(max(DAYS))
                    if train_day == test_day:
                        val = train_day / float(max(DAYS))
            else:
                raise ValueError(f"Unknown template: {kind}")
            mat[train_day - 1, test_day - 1] = val
    return mat


def plot_matrix(ax, mat, title):
    image = ax.imshow(mat, origin="upper", cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(["D1", "D2", "D3", "D4", "D5"], fontsize=8)
    ax.set_yticklabels(["D1", "D2", "D3", "D4", "D5"], fontsize=8)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    return image


def save_fig_mvpa_window_transfer_model_predictions(figures_dir=FIGURES_DIR):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(11.0, 5.5))
    gs = gridspec.GridSpec(
        2,
        8,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        left=0.05,
        right=0.99,
        top=0.88,
        bottom=0.07,
    )
    ax_top = fig.add_subplot(gs[0, 3:5])
    image = plot_matrix(
        ax_top,
        template_matrix("gradual"),
        "Continuous Restructuring",
    )
    for idx, split_day in enumerate([1, 2, 3, 4]):
        ax = fig.add_subplot(gs[1, idx * 2: idx * 2 + 2])
        plot_matrix(
            ax,
            template_matrix("split_gradual", split_day=split_day),
            f"Discrete Restructuring (D{split_day})",
        )
    fig.colorbar(image, ax=fig.axes, fraction=0.022, pad=0.01)
    fig.suptitle("MVPA Transfer Model Predictions", y=0.97)
    path = figures_dir / "mvpa_stim_locked_cat_window_transfer_model_predictions.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[TG figure] wrote {path}", flush=True)
    return path


def save_fig_mvpa_stim_locked_cat_tg(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cross_matrix_day_mean_csv = output_dir / "mvpa_stim_locked_cat_tg_timegen_day_mean.csv"
    if not cross_matrix_day_mean_csv.exists():
        raise FileNotFoundError(
            f"Missing TG cross-day output in {output_dir}. "
            "Run mvpa_stim_locked_cat_tg_analysis.py first."
        )
    d_mat = pd.read_csv(cross_matrix_day_mean_csv)
    if d_mat.empty:
        raise ValueError(f"Empty TG cross-day timegen output table: {cross_matrix_day_mean_csv}")
    fig_cross_timegen = figures_dir / "mvpa_stim_locked_cat_tg_timegen_matrices_5x5.png"

    day_grid = sorted({1, 2, 3, 4, 5})
    fig, axes = plt.subplots(5, 5, figsize=(18, 16), squeeze=False)
    vmin = float(d_mat["auc_mean"].min())
    vmax = float(d_mat["auc_mean"].max())
    im = None
    for i, train_day in enumerate(day_grid):
        for j, test_day in enumerate(day_grid):
            ax = axes[i, j]
            g = d_mat[
                (d_mat["train_day"] == train_day)
                & (d_mat["test_day"] == test_day)
            ]
            if g.empty:
                raise ValueError(
                    f"Missing TG matrix day pair in {cross_matrix_day_mean_csv}: "
                    f"train_day={train_day}, test_day={test_day}"
                )
            pivot = g.pivot(
                index="train_time_sec", columns="test_time_sec", values="auc_mean"
            )
            im = ax.imshow(
                pivot.to_numpy(),
                origin="lower",
                aspect="auto",
                extent=[
                    float(pivot.columns.min()),
                    float(pivot.columns.max()),
                    float(pivot.index.min()),
                    float(pivot.index.max()),
                ],
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
            ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
            ax.set_title(f"Train D{train_day} -> Test D{test_day}", fontsize=9)
            if i == len(day_grid) - 1:
                ax.set_xlabel("Test Time (s)")
            if j == 0:
                ax.set_ylabel("Train Time (s)")
    fig.suptitle("Cross-Day Temporal Generalization by Day Pair (AUC)")
    fig.subplots_adjust(
        top=0.94,
        bottom=0.05,
        left=0.05,
        right=0.90,
        wspace=0.30,
        hspace=0.35,
    )
    cax = fig.add_axes([0.92, 0.12, 0.015, 0.74])
    fig.colorbar(im, cax=cax, label="AUC")
    fig.savefig(fig_cross_timegen, dpi=150, bbox_inches="tight")
    plt.close(fig)

    model_path = save_fig_mvpa_window_transfer_model_predictions(figures_dir)
    return {
        "timegen_figure_path": fig_cross_timegen,
        "model_prediction_figure_path": model_path,
    }


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_tg()
