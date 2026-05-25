#!/usr/bin/env python3
"""Plot schematic one-stage and two-stage connectivity distance predictions."""

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


def make_one_stage_gradient_distance():
    mat = np.zeros((len(DAYS), len(DAYS)), dtype=float)
    max_dist = float(max(DAYS) - min(DAYS))
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            mat[i, j] = abs(float(day_i) - float(day_j)) / max_dist
    return mat


def make_two_stage_gradient_distance():
    mat = np.zeros((len(DAYS), len(DAYS)), dtype=float)
    max_dist = float(max(DAYS) - min(DAYS))
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            day_dist = abs(float(day_i) - float(day_j)) / max_dist
            if same_stage(day_i, day_j):
                mat[i, j] = 0.25 * day_dist
            else:
                mat[i, j] = 0.70 + 0.30 * day_dist
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
    ax.set_xlabel("Day")
    ax.set_ylabel("Day")


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


def save_fig_connect_network_distance_prediction(figures_dir=FIGURES_DIR):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "connect_network_distance_prediction_models.png"

    matrices = [
        make_one_stage_gradient_distance(),
        make_two_stage_gradient_distance(),
    ]
    titles = [
        "One-stage: graded distance",
        "Two-stage: graded distance",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), squeeze=False)
    graded_cmap = plt.get_cmap("viridis")
    ims = []
    for idx, mat in enumerate(matrices):
        ax = axes[0, idx]
        im = ax.imshow(
            mat,
            origin="upper",
            cmap=graded_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        ims.append(im)
        format_axis(ax, titles[idx])
        add_cell_labels(ax, mat)

    fig.suptitle("A Priori Connectivity Network-Distance Predictions")
    fig.subplots_adjust(
        top=0.90,
        bottom=0.14,
        left=0.08,
        right=0.86,
        wspace=0.32,
    )
    cax = fig.add_axes([0.89, 0.20, 0.018, 0.58])
    fig.colorbar(ims[-1], cax=cax, label="Predicted distance")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Connectivity prediction] Wrote {fig_path}")
    return {"network_distance_prediction_models": fig_path}


if __name__ == "__main__":
    save_fig_connect_network_distance_prediction()
