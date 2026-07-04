#!/usr/bin/env python3
"""Figures for category-linked latent trajectory analyses."""

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

from figure_style import DAYS, DAY_COLORS, FIGURES_DIR, OUTPUT_DIR, setup_axis
from latent_dynamics_utils import require_csv

ANALYSES = [
    ("category_centroid", "A-B ERP Contrast"),
    ("classifier_axis", "Classifier Axis"),
    ("category_evidence", "CV Category Evidence"),
]

MODEL_ORDER = ["Gradual", "Discrete D1", "Discrete D2", "Discrete D3", "Discrete D4"]


def save_trajectory_figure(output_dir, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.5), sharex=False, sharey=False)
    for ax, (analysis, title) in zip(axes, ANALYSES):
        d = require_csv(output_dir / f"latent_{analysis}_points.csv")
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
                linewidth=2.0,
                label=f"D{day}",
            )
            ax.scatter(
                g["latent_1"].iloc[0],
                g["latent_2"].iloc[0],
                color=DAY_COLORS[day],
                s=24,
            )
            ax.scatter(
                g["latent_1"].iloc[-1],
                g["latent_2"].iloc[-1],
                color=DAY_COLORS[day],
                s=32,
                marker="s",
            )
        ax.set_title(title)
        ax.set_xlabel("latent 1")
        ax.set_ylabel("latent 2")
        setup_axis(ax)
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Category-Linked Latent Trajectories")
    fig.tight_layout()
    path = figures_dir / "latent_category_encoding_trajectories.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent category figure] wrote {path}", flush=True)


def distance_matrix(d):
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    np.fill_diagonal(mat, np.nan)
    for row in d.itertuples(index=False):
        i = DAYS.index(int(row.day_i))
        j = DAYS.index(int(row.day_j))
        mat[i, j] = float(row.distance_mean)
        mat[j, i] = float(row.distance_mean)
    return mat


def save_matrix_figure(output_dir, figures_dir):
    mats = []
    for analysis, _title in ANALYSES:
        d = require_csv(output_dir / f"latent_{analysis}_group_distances.csv")
        mats.append(distance_matrix(d))
    finite = np.concatenate([mat[np.isfinite(mat)] for mat in mats])
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.88")

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 4.0), constrained_layout=True)
    for ax, mat, (_analysis, title) in zip(axes, mats, ANALYSES):
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
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
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("mean z-Euclidean distance")
    path = figures_dir / "latent_category_encoding_day_geometry_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent category figure] wrote {path}", flush=True)


def save_model_evidence_figure(output_dir, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), sharey=True)
    for ax, (analysis, title) in zip(axes, ANALYSES):
        d = require_csv(output_dir / f"latent_{analysis}_model_subject.csv")
        d = d[d["model_label"] != "Baseline"].copy()
        box_vals = []
        for label in MODEL_ORDER:
            vals = -d[d["model_label"] == label]["delta_bic_baseline"].to_numpy(float)
            box_vals.append(vals[np.isfinite(vals)])
        x = np.arange(len(MODEL_ORDER))
        ax.boxplot(
            box_vals,
            positions=x,
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.2},
            boxprops={"facecolor": "#4c78a8", "alpha": 0.55, "edgecolor": "black"},
            whiskerprops={"color": "black", "linewidth": 1.0},
            capprops={"color": "black", "linewidth": 1.0},
        )
        for xi, vals in enumerate(box_vals):
            jitter = np.linspace(-0.14, 0.14, len(vals))
            ax.scatter(
                np.full(len(vals), xi) + jitter,
                vals,
                color="black",
                alpha=0.35,
                s=14,
            )
        ax.axhline(0, color="0.25", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("model")
        ax.set_xticks(x)
        ax.set_xticklabels(MODEL_ORDER, rotation=35, ha="right")
        setup_axis(ax)
    axes[0].set_ylabel("evidence above baseline model")
    fig.suptitle("Category-Linked 5x5 Model Evidence")
    fig.tight_layout()
    path = figures_dir / "latent_category_encoding_model_evidence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent category figure] wrote {path}", flush=True)


def save_category_evidence_timecourse(output_dir, figures_dir):
    d = require_csv(output_dir / "latent_category_evidence_timecourse_day.csv")
    fig, ax = plt.subplots(figsize=(7.4, 4.0))
    for day in DAYS:
        g = d[d["day"] == day].sort_values("time_sec")
        if g.empty:
            continue
        x = g["time_sec"].to_numpy(float)
        y = g["signed_evidence_mean"].to_numpy(float)
        err = g["signed_evidence_sem"].to_numpy(float)
        ax.plot(x, y, color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
        ax.fill_between(x, y - err, y + err, color=DAY_COLORS[day], alpha=0.14)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.axvline(0, color="0.35", linewidth=0.8)
    ax.set_title("Cross-Validated Category Evidence")
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("mean signed decision value")
    ax.legend(frameon=False, loc="best")
    setup_axis(ax)
    fig.tight_layout()
    path = figures_dir / "latent_category_evidence_timecourse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent category figure] wrote {path}", flush=True)


def save_latent_category_encoding_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_trajectory_figure(output_dir, figures_dir)
    save_matrix_figure(output_dir, figures_dir)
    save_model_evidence_figure(output_dir, figures_dir)
    save_category_evidence_timecourse(output_dir, figures_dir)


if __name__ == "__main__":
    save_latent_category_encoding_figures()
