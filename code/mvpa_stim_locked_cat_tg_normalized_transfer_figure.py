#!/usr/bin/env python3
"""Plot TG day structure after within-day normalisation."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from mvpa_stim_locked_cat_tg_normalized_transfer_analysis import (
    DAYS,
    FIGURES_DIR,
    OUTPUT_DIR,
    SUMMARIES,
    WINDOWS,
)


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing TG normalized-transfer output: {path}. "
            "Run mvpa_stim_locked_cat_tg_normalized_transfer_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty TG normalized-transfer output: {path}")
    return d


def group_matrix(group_df, summary, window, value_col):
    g = group_df[
        (group_df["summary"] == summary)
        & (group_df["window"] == window)
    ]
    if g.empty:
        raise ValueError(f"Missing group rows: summary={summary}, window={window}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row[value_col])
        mat[j, i] = float(row[value_col])
    return mat


def save_matrix_figure(group_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_tg_normalized_transfer_similarity.png"
    value_col = "normalized_transfer_mean"
    all_vals = []
    for _, row in group_df.iterrows():
        val = float(row[value_col])
        if np.isfinite(val):
            all_vals.append(val)
    if len(all_vals) == 0:
        raise ValueError("No finite normalized-transfer values to plot")
    vmax = float(np.nanmax(np.abs(all_vals)))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Cannot determine normalized-transfer color scale")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.82")
    fig, axes = plt.subplots(
        len(SUMMARIES), len(WINDOWS), figsize=(7.6, 10.8), squeeze=False
    )
    im = None
    for r, summary in enumerate(SUMMARIES):
        for c, window in enumerate(WINDOWS):
            ax = axes[r, c]
            mat = group_matrix(group_df, summary, window, value_col)
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                origin="upper",
                cmap=cmap,
                vmin=-vmax,
                vmax=vmax,
            )
            ax.set_title(f"{summary} | {window}")
            labels = []
            for day in DAYS:
                labels.append(f"D{day}")
            ax.set_xticks(range(len(DAYS)))
            ax.set_yticks(range(len(DAYS)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Day")
            ax.set_ylabel("Day")
            for i in range(len(DAYS)):
                for j in range(len(DAYS)):
                    if np.isfinite(mat[i, j]):
                        color = "black"
                        if abs(float(mat[i, j])) > 0.55 * vmax:
                            color = "white"
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=7,
                        )
    fig.suptitle("Cross-Day TG Transfer Normalized by Within-Day Decoding")
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.86, wspace=0.35, hspace=0.35)
    cax = fig.add_axes([0.89, 0.16, 0.018, 0.68])
    fig.colorbar(im, cax=cax, label="Normalized transfer")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def linkage_matrix(clusters_df, summary, window):
    g = clusters_df[
        (clusters_df["row_type"] == "linkage")
        & (clusters_df["summary"] == summary)
        & (clusters_df["window"] == window)
    ].sort_values("merge_index")
    if len(g) != len(DAYS) - 1:
        raise ValueError(f"Missing linkage rows: summary={summary}, window={window}")
    z = np.zeros((len(g), 4), dtype=float)
    for i, row in enumerate(g.itertuples()):
        z[i, 0] = float(row.child_1)
        z[i, 1] = float(row.child_2)
        z[i, 2] = float(row.distance)
        z[i, 3] = float(row.n_members)
    return z


def save_cluster_figure(clusters_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_tg_normalized_transfer_clusters.png"
    fig, axes = plt.subplots(
        len(SUMMARIES), len(WINDOWS), figsize=(8.0, 9.8), squeeze=False
    )
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    for r, summary in enumerate(SUMMARIES):
        for c, window in enumerate(WINDOWS):
            ax = axes[r, c]
            z = linkage_matrix(clusters_df, summary, window)
            dendrogram(z, labels=labels, ax=ax, color_threshold=0.0)
            ax.set_title(f"{summary} | {window}")
            ax.set_ylabel("Distance")
    fig.suptitle("Clustering of Normalized Cross-Day TG Transfer")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_denominator_figure(group_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_tg_normalized_transfer_denominators.png"
    fig, axes = plt.subplots(
        len(SUMMARIES), len(WINDOWS), figsize=(7.6, 10.8), squeeze=False
    )
    max_failed = int(np.nanmax(group_df["n_denominator_failed"].to_numpy(dtype=float)))
    vmax = max(1, max_failed)
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="0.82")
    im = None
    for r, summary in enumerate(SUMMARIES):
        for c, window in enumerate(WINDOWS):
            ax = axes[r, c]
            mat = group_matrix(group_df, summary, window, "n_denominator_failed")
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                origin="upper",
                cmap=cmap,
                vmin=0,
                vmax=vmax,
            )
            ax.set_title(f"{summary} | {window}")
            labels = []
            for day in DAYS:
                labels.append(f"D{day}")
            ax.set_xticks(range(len(DAYS)))
            ax.set_yticks(range(len(DAYS)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            for i in range(len(DAYS)):
                for j in range(len(DAYS)):
                    if np.isfinite(mat[i, j]):
                        ax.text(
                            j,
                            i,
                            f"{int(mat[i, j])}",
                            ha="center",
                            va="center",
                            color="white" if mat[i, j] > 0.5 * vmax else "black",
                            fontsize=8,
                        )
    fig.suptitle("Subjects Excluded by Nonpositive Within-Day Signal")
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.86, wspace=0.35, hspace=0.35)
    cax = fig.add_axes([0.89, 0.16, 0.018, 0.68])
    fig.colorbar(im, cax=cax, label="Excluded subjects")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_mvpa_stim_locked_cat_tg_normalized_transfer(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    group_csv = output_dir / "mvpa_stim_locked_cat_tg_normalized_transfer_group_pairs.csv"
    clusters_csv = output_dir / "mvpa_stim_locked_cat_tg_normalized_transfer_clusters.csv"

    group_df = require_csv(group_csv)
    clusters_df = require_csv(clusters_csv)

    matrix_path = save_matrix_figure(group_df, figures_dir)
    cluster_path = save_cluster_figure(clusters_df, figures_dir)
    denominator_path = save_denominator_figure(group_df, figures_dir)
    print(f"[TG normalized transfer] Wrote {matrix_path}")
    print(f"[TG normalized transfer] Wrote {cluster_path}")
    print(f"[TG normalized transfer] Wrote {denominator_path}")
    return {
        "matrix": matrix_path,
        "clusters": cluster_path,
        "denominators": denominator_path,
    }


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_tg_normalized_transfer()
