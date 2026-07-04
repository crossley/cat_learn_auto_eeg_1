#!/usr/bin/env python3
"""Figures for connectivity latent trajectory analysis."""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style import DAYS, DAY_COLORS, FIGURES_DIR, OUTPUT_DIR, setup_axis
from latent_dynamics_utils import require_csv

MODEL_ORDER = ["Gradual", "Discrete D1", "Discrete D2", "Discrete D3", "Discrete D4"]


def save_latent_connectivity_trajectory_plot(output_dir, figures_dir):
    d = require_csv(output_dir / "latent_connectivity_trajectory_points.csv")
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for day in DAYS:
        g = (
            d[d["day"] == day]
            .groupby("time_sec", as_index=False)
            .agg(latent_1=("latent_1", "mean"), latent_2=("latent_2", "mean"))
            .sort_values("time_sec")
        )
        if g.empty:
            continue
        ax.plot(
            g["latent_1"],
            g["latent_2"],
            color=DAY_COLORS[day],
            linewidth=2.2,
            label=f"D{day}",
        )
        ax.scatter(g["latent_1"].iloc[0], g["latent_2"].iloc[0], color=DAY_COLORS[day], s=26)
        ax.scatter(
            g["latent_1"].iloc[-1],
            g["latent_2"].iloc[-1],
            color=DAY_COLORS[day],
            s=34,
            marker="s",
        )
    ax.set_title("Connectivity Latent Trajectories")
    ax.set_xlabel("latent 1")
    ax.set_ylabel("latent 2")
    ax.legend(frameon=False, loc="best")
    setup_axis(ax)
    fig.tight_layout()
    path = figures_dir / "latent_connectivity_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent connectivity figure] wrote {path}", flush=True)


def distance_matrix(d):
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    np.fill_diagonal(mat, np.nan)
    for row in d.itertuples(index=False):
        i = DAYS.index(int(row.day_i))
        j = DAYS.index(int(row.day_j))
        mat[i, j] = float(row.distance_mean)
        mat[j, i] = float(row.distance_mean)
    return mat


def save_latent_connectivity_matrix_plot(output_dir, figures_dir):
    d = require_csv(output_dir / "latent_connectivity_trajectory_group_distances.csv")
    mat = distance_matrix(d)
    finite = mat[np.isfinite(mat)]
    fig, ax = plt.subplots(figsize=(4.8, 4.1))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")
    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=np.min(finite), vmax=np.max(finite))
    ax.set_title("Connectivity Latent Day Distances")
    ax.set_xticks(range(len(DAYS)))
    ax.set_yticks(range(len(DAYS)))
    ax.set_xticklabels([f"D{x}" for x in DAYS])
    ax.set_yticklabels([f"D{x}" for x in DAYS])
    for i in range(len(DAYS)):
        for j in range(len(DAYS)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=8)
            else:
                ax.text(j, i, "-", ha="center", va="center", fontsize=9)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("mean z-Euclidean distance")
    fig.tight_layout()
    path = figures_dir / "latent_connectivity_day_geometry_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent connectivity figure] wrote {path}", flush=True)


def save_latent_connectivity_model_evidence(output_dir, figures_dir):
    d = require_csv(output_dir / "latent_connectivity_trajectory_model_subject.csv")
    d = d[d["model_label"] != "Baseline"].copy()
    box_vals = []
    for label in MODEL_ORDER:
        vals = -d[d["model_label"] == label]["delta_bic_baseline"].to_numpy(float)
        box_vals.append(vals[np.isfinite(vals)])
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    x = np.arange(len(MODEL_ORDER))
    ax.boxplot(
        box_vals,
        positions=x,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        boxprops={"facecolor": "#4c78a8", "alpha": 0.55, "edgecolor": "black"},
    )
    for xi, vals in enumerate(box_vals):
        jitter = np.linspace(-0.14, 0.14, len(vals))
        ax.scatter(np.full(len(vals), xi) + jitter, vals, color="black", alpha=0.35, s=14)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_title("Connectivity Latent 5x5 Model Evidence")
    ax.set_ylabel("evidence above baseline model")
    ax.set_xlabel("model")
    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right")
    setup_axis(ax)
    fig.tight_layout()
    path = figures_dir / "latent_connectivity_model_evidence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent connectivity figure] wrote {path}", flush=True)


def save_latent_connectivity_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_latent_connectivity_trajectory_plot(output_dir, figures_dir)
    save_latent_connectivity_matrix_plot(output_dir, figures_dir)
    save_latent_connectivity_model_evidence(output_dir, figures_dir)


if __name__ == "__main__":
    save_latent_connectivity_figures()
