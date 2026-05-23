#!/usr/bin/env python3
"""Plot schematic one-stage and two-stage day-transfer predictions."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
DAYS = [1, 2, 3, 4, 5]
STAGE_A_DAYS = [1, 2, 3]
STAGE_B_DAYS = [4, 5]


def same_stage(day_i, day_j):
    if day_i in STAGE_A_DAYS and day_j in STAGE_A_DAYS:
        return True
    if day_i in STAGE_B_DAYS and day_j in STAGE_B_DAYS:
        return True
    return False


def make_one_stage_binary():
    mat = np.ones((len(DAYS), len(DAYS)), dtype=float)
    return mat


def make_two_stage_binary():
    mat = np.zeros((len(DAYS), len(DAYS)), dtype=float)
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            if same_stage(day_i, day_j):
                mat[i, j] = 1.0
    return mat


def make_one_stage_gradient():
    mat = np.zeros((len(DAYS), len(DAYS)), dtype=float)
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            mat[i, j] = min(day_i, day_j) / float(max(DAYS))
    return mat


def make_two_stage_gradient():
    mat = np.zeros((len(DAYS), len(DAYS)), dtype=float)
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            if same_stage(day_i, day_j):
                mat[i, j] = min(day_i, day_j) / float(max(DAYS))
    return mat


def format_axis(ax, title):
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    ax.set_title(title)
    ax.set_xticks(range(len(DAYS)))
    ax.set_yticks(range(len(DAYS)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Test day")
    ax.set_ylabel("Train day")


def add_cell_labels(ax, mat):
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = float(mat[i, j])
            color = "white"
            if val > 0.65:
                color = "black"
            ax.text(
                j,
                i,
                f"{val:.1f}",
                ha="center",
                va="center",
                color=color,
                fontsize=8,
            )


def save_fig_mvpa_stim_locked_cat_transfer_prediction(figures_dir=FIGURES_DIR):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "mvpa_stim_locked_cat_transfer_prediction_models.png"

    matrices = [
        make_one_stage_binary(),
        make_two_stage_binary(),
        make_one_stage_gradient(),
        make_two_stage_gradient(),
    ]
    titles = [
        "One-stage: binary",
        "Two-stage: binary",
        "One-stage: graded strength",
        "Two-stage: graded within stage",
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 7.6), squeeze=False)
    binary_cmap = plt.get_cmap("gray")
    graded_cmap = plt.get_cmap("viridis")
    ims = []
    for idx, mat in enumerate(matrices):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        cmap = binary_cmap
        if row == 1:
            cmap = graded_cmap
        im = ax.imshow(mat, origin="upper", cmap=cmap, vmin=0.0, vmax=1.0)
        ims.append(im)
        format_axis(ax, titles[idx])
        add_cell_labels(ax, mat)

    fig.suptitle("A Priori Day-Generalisation Predictions")
    fig.subplots_adjust(top=0.90, bottom=0.08, left=0.08, right=0.86, wspace=0.32, hspace=0.38)
    cax = fig.add_axes([0.89, 0.18, 0.018, 0.64])
    fig.colorbar(ims[-1], cax=cax, label="Predicted transfer")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[MVPA prediction] Wrote {fig_path}")
    return {"prediction_models": fig_path}


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_transfer_prediction()
