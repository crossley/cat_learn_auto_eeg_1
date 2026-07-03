#!/usr/bin/env python3
"""Figures for resting-state day-geometry controls."""

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

from figure_style import DAYS, FIGURES_DIR, OUTPUT_DIR, setup_axis


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing resting-state output: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty resting-state output: {path}")
    return d


def distance_matrix(d):
    mat = np.zeros((len(DAYS), len(DAYS)), dtype=float)
    mat[:] = np.nan
    np.fill_diagonal(mat, 0.0)
    for row in d.itertuples(index=False):
        i = DAYS.index(int(row.day_i))
        j = DAYS.index(int(row.day_j))
        mat[i, j] = float(row.distance_mean)
        mat[j, i] = float(row.distance_mean)
    return mat


def save_rest_distance_matrices(output_dir, figures_dir):
    d = require_csv(output_dir / "rest_day_geometry_group_distances.csv")
    d = d[d["band_group"] == "all_bands"].copy()
    panels = [
        ("connectivity", "Rest Connectivity"),
        ("spectral", "Rest Spectral Topography"),
    ]
    mats = []
    for feature_kind, _title in panels:
        g = d[d["feature_kind"] == feature_kind]
        if g.empty:
            raise ValueError(f"No group distances for {feature_kind}")
        mat = distance_matrix(g)
        for idx in range(len(DAYS)):
            mat[idx, idx] = np.nan
        mats.append(mat)
    finite_vals = np.concatenate([mat[np.isfinite(mat)] for mat in mats])
    vmin = float(np.nanmin(finite_vals))
    vmax = float(np.nanmax(finite_vals))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    for ax, mat, (_feature_kind, title) in zip(axes, mats, panels):
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="0.88")
        im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(len(DAYS)))
        ax.set_yticks(range(len(DAYS)))
        ax.set_xticklabels([f"D{day}" for day in DAYS])
        ax.set_yticklabels([f"D{day}" for day in DAYS])
        for i in range(len(DAYS)):
            for j in range(len(DAYS)):
                if not np.isfinite(mat[i, j]):
                    ax.text(j, i, "-", ha="center", va="center", fontsize=9)
                    continue
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if mat[i, j] > vmin + 0.55 * (vmax - vmin) else "black",
                )
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("mean z-Euclidean distance")
    path = figures_dir / "rest_day_geometry_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rest figure] wrote {path}", flush=True)


def save_rest_model_evidence(output_dir, figures_dir):
    d = require_csv(output_dir / "rest_day_geometry_model_summary.csv")
    s = require_csv(output_dir / "rest_day_geometry_model_subject.csv")
    d = d[d["band_group"] == "all_bands"].copy()
    d = d[d["model_label"] != "Baseline"].copy()
    s = s[s["band_group"] == "all_bands"].copy()
    s = s[s["model_label"] != "Baseline"].copy()
    order = ["Gradual", "Discrete D1", "Discrete D2", "Discrete D3", "Discrete D4"]
    panels = [
        ("connectivity", "Rest Connectivity"),
        ("spectral", "Rest Spectral Topography"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8), sharey=True)
    for ax, (feature_kind, title) in zip(axes, panels):
        g = d[d["feature_kind"] == feature_kind].copy()
        if g.empty:
            raise ValueError(f"No model evidence rows for {feature_kind}")
        box_vals = []
        for label in order:
            sub = s[
                (s["feature_kind"] == feature_kind)
                & (s["model_label"] == label)
            ].copy()
            y = -sub["delta_bic_baseline"].to_numpy(float)
            y = y[np.isfinite(y)]
            box_vals.append(y)
        x = np.arange(len(order))
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
        for xi, label in enumerate(order):
            sub = s[
                (s["feature_kind"] == feature_kind)
                & (s["model_label"] == label)
            ].copy()
            y = -sub["delta_bic_baseline"].to_numpy(float)
            y = y[np.isfinite(y)]
            if len(y) == 0:
                continue
            jitter = np.linspace(-0.14, 0.14, len(y))
            ax.scatter(
                np.full(len(y), xi) + jitter,
                y,
                s=14,
                color="black",
                alpha=0.35,
                linewidths=0,
                zorder=3,
            )
        ax.axhline(0, color="0.25", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels(order, rotation=35, ha="right")
        ax.set_xlabel("model")
        setup_axis(ax)
    axes[0].set_ylabel("evidence above baseline model")
    fig.suptitle("Resting-State 5x5 Model Evidence")
    fig.tight_layout()
    path = figures_dir / "rest_day_geometry_model_evidence.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rest figure] wrote {path}", flush=True)


def save_rest_day_geometry_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_rest_distance_matrices(output_dir, figures_dir)
    save_rest_model_evidence(output_dir, figures_dir)


if __name__ == "__main__":
    save_rest_day_geometry_figures()
