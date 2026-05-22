#!/usr/bin/env python3
"""Plot day-structure analyses for stimulus-locked TG window summaries."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from mvpa_stim_locked_cat_tg_day_structure_analysis import (
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
            f"Missing TG day-structure output: {path}. "
            "Run mvpa_stim_locked_cat_tg_day_structure_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty TG day-structure output: {path}")
    return d


def group_matrix(sym_df, summary, window):
    g = sym_df[
        (sym_df["row_type"] == "group")
        & (sym_df["summary"] == summary)
        & (sym_df["window"] == window)
    ]
    if g.empty:
        raise ValueError(f"Missing symmetrised rows: summary={summary}, window={window}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row["similarity_mean"])
        mat[j, i] = float(row["similarity_mean"])
    return mat


def save_symmetrised_figure(sym_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_tg_day_structure_symmetrised.png"
    all_vals = []
    for _, row in sym_df[sym_df["row_type"] == "group"].iterrows():
        all_vals.append(float(row["similarity_mean"]))
    if len(all_vals) == 0:
        raise ValueError("No group symmetrised similarity values to plot")
    vmin = float(np.nanmin(all_vals))
    vmax = float(np.nanmax(all_vals))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.82")
    fig, axes = plt.subplots(
        len(SUMMARIES), len(WINDOWS), figsize=(7.6, 10.8), squeeze=False
    )
    im = None
    for r, summary in enumerate(SUMMARIES):
        for c, window in enumerate(WINDOWS):
            ax = axes[r, c]
            mat = group_matrix(sym_df, summary, window)
            masked = np.ma.masked_invalid(mat)
            im = ax.imshow(masked, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(f"{summary} | {window}")
            ax.set_xticks(range(len(DAYS)))
            ax.set_yticks(range(len(DAYS)))
            labels = []
            for day in DAYS:
                labels.append(f"D{day}")
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Day")
            ax.set_ylabel("Day")
            for i in range(len(DAYS)):
                for j in range(len(DAYS)):
                    if np.isfinite(mat[i, j]):
                        color = "white"
                        if mat[i, j] > (vmin + 0.65 * (vmax - vmin)):
                            color = "black"
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.3f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=7,
                        )
    fig.suptitle("Symmetrised Cross-Day TG Similarity")
    fig.subplots_adjust(top=0.93, bottom=0.06, left=0.08, right=0.86, wspace=0.35, hspace=0.35)
    cax = fig.add_axes([0.89, 0.16, 0.018, 0.68])
    fig.colorbar(im, cax=cax, label="AUC")
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
    fig_path = figures_dir / "mvpa_stim_locked_cat_tg_day_structure_clusters.png"
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
    fig.suptitle("Hierarchical Clustering of Symmetrised Day Similarity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_embedding_figure(embedding_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_tg_day_structure_embedding.png"
    fig, axes = plt.subplots(
        len(SUMMARIES), len(WINDOWS), figsize=(8.0, 10.0), squeeze=False
    )
    for r, summary in enumerate(SUMMARIES):
        for c, window in enumerate(WINDOWS):
            ax = axes[r, c]
            g = embedding_df[
                (embedding_df["summary"] == summary)
                & (embedding_df["window"] == window)
            ].sort_values("day")
            if len(g) != len(DAYS):
                raise ValueError(f"Missing embedding rows: summary={summary}, window={window}")
            ax.plot(g["x"], g["y"], color="0.6", linewidth=1.0, zorder=1)
            ax.scatter(g["x"], g["y"], s=70, color="tab:blue", zorder=2)
            for _, row in g.iterrows():
                ax.text(
                    float(row["x"]),
                    float(row["y"]),
                    f"D{int(row['day'])}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                    zorder=3,
                )
            explained = float(g["variance_explained_2d"].iloc[0])
            ax.set_title(f"{summary} | {window} ({explained:.2f} 2D)")
            ax.axhline(0.0, color="0.8", linewidth=0.8)
            ax.axvline(0.0, color="0.8", linewidth=0.8)
            ax.set_xlabel("MDS 1")
            ax.set_ylabel("MDS 2")
            ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("2D Embedding of Cross-Day TG Similarity")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def model_label(model):
    labels = {
        "day_distance_gradient": "distance",
        "day1_isolated_later_only": "day1 vs later",
        "adjacent_clusters_23_45": "2-3 / 4-5",
        "stage_blocks_1_23_45": "1 / 2-3 / 4-5",
        "late_cluster_45": "4-5",
    }
    if model not in labels:
        return model
    return labels[model]


def save_model_comparison_figure(model_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_tg_day_structure_model_comparison.png"
    d = model_df[model_df["row_type"] == "summary"].copy()
    if d.empty:
        raise ValueError("No candidate-model summary rows to plot")
    models = []
    for model in d["model"].dropna().unique():
        models.append(model)
    fig, axes = plt.subplots(
        len(SUMMARIES), len(WINDOWS), figsize=(11.0, 10.2), squeeze=False, sharey=True
    )
    for r, summary in enumerate(SUMMARIES):
        for c, window in enumerate(WINDOWS):
            ax = axes[r, c]
            g = d[
                (d["summary"] == summary)
                & (d["window"] == window)
            ].copy()
            if g.empty:
                raise ValueError(
                    f"Missing candidate model rows: summary={summary}, window={window}"
                )
            means = []
            sems = []
            labels = []
            for model in models:
                row = g[g["model"] == model]
                if row.empty:
                    raise ValueError(
                        f"Missing candidate model {model}: summary={summary}, window={window}"
                    )
                means.append(float(row["r_mean"].iloc[0]))
                sems.append(float(row["r_sem"].iloc[0]))
                labels.append(model_label(model))
            x = np.arange(len(models))
            ax.bar(x, means, yerr=sems, color="tab:blue", alpha=0.85, capsize=3)
            ax.axhline(0.0, color="0.3", linewidth=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            ax.set_title(f"{summary} | {window}")
            ax.set_ylabel("Mean subject r")
            ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Candidate Models of Cross-Day TG Day Structure")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_mvpa_stim_locked_cat_tg_day_structure(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    sym_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_symmetrised.csv"
    clusters_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_clusters.csv"
    embedding_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_embedding.csv"
    model_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_model_comparison.csv"

    sym_df = require_csv(sym_csv)
    clusters_df = require_csv(clusters_csv)
    embedding_df = require_csv(embedding_csv)
    model_df = require_csv(model_csv)

    paths = {
        "symmetrised": save_symmetrised_figure(sym_df, figures_dir),
        "clusters": save_cluster_figure(clusters_df, figures_dir),
        "embedding": save_embedding_figure(embedding_df, figures_dir),
        "model_comparison": save_model_comparison_figure(model_df, figures_dir),
    }
    print(f"[TG day-structure] Wrote {paths['symmetrised']}")
    print(f"[TG day-structure] Wrote {paths['clusters']}")
    print(f"[TG day-structure] Wrote {paths['embedding']}")
    print(f"[TG day-structure] Wrote {paths['model_comparison']}")
    return paths


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_tg_day_structure()
