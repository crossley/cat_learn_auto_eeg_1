#!/usr/bin/env python3
"""Plot late-window stimulus-locked category MVPA outputs."""

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

from mvpa_stim_locked_cat_late_window_analysis import (
    DAYS,
    FIGURES_DIR,
    OUTPUT_DIR,
    WINDOW_END_SEC,
    WINDOW_START_SEC,
)


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing late-window MVPA output: {path}. "
            "Run mvpa_stim_locked_cat_late_window_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty late-window MVPA output: {path}")
    return d


def make_haufe_info_from_pos_df(pos_df):
    ch_names = pos_df["channel"].tolist()
    ch_pos = {}
    for _, row in pos_df.iterrows():
        ch_pos[row["channel"]] = np.array([row["x"], row["y"], row["z"]], dtype=float)
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="ignore")
    return info, ch_names


def save_auc_figure(day_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_late_window_auc_by_day.png"
    g = day_df.sort_values("day")
    if len(g) != len(DAYS):
        raise ValueError("Late-window AUC day table does not contain all days")
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.errorbar(
        g["day"],
        g["auc_mean"],
        yerr=g["auc_sem"],
        color="black",
        marker="o",
        linewidth=1.8,
        capsize=3,
    )
    ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_xticks(DAYS)
    ax.set_xlabel("Day")
    ax.set_ylabel("AUC")
    ax.set_title(
        f"Late-Window Category Decoding ({WINDOW_START_SEC:.2f}-{WINDOW_END_SEC:.2f}s)"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def similarity_matrix(sim_df):
    g = sim_df[sim_df["row_type"] == "group"]
    if g.empty:
        raise ValueError("Missing late-window group similarity rows")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row["similarity_mean"])
        mat[j, i] = float(row["similarity_mean"])
    return mat


def save_similarity_figure(sim_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_late_window_haufe_similarity.png"
    mat = similarity_matrix(sim_df)
    vals = mat[np.isfinite(mat)]
    if len(vals) == 0:
        raise ValueError("No finite late-window Haufe similarity values")
    vmax = float(np.nanmax(np.abs(vals)))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Cannot determine late-window similarity color scale")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.82")
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(np.ma.masked_invalid(mat), origin="upper", cmap=cmap, vmin=-vmax, vmax=vmax)
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
                    fontsize=8,
                )
    ax.set_title("Late-Window Haufe Pattern Similarity")
    fig.colorbar(im, ax=ax, label="Pattern similarity r", fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def day_sensor_values(sensor_day_df, day, ch_names):
    g = sensor_day_df[sensor_day_df["day"] == int(day)]
    if g.empty:
        raise ValueError(f"Missing late-window sensor day rows: day={day}")
    vals = (
        g.set_index("channel")
        .reindex(ch_names)["pattern_mean"]
        .to_numpy(dtype=float)
    )
    if not np.all(np.isfinite(vals)):
        raise ValueError(f"Missing late-window topomap values: day={day}")
    return vals


def save_topomap_figure(sensor_day_df, pos_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_late_window_haufe_topomaps.png"
    info, ch_names = make_haufe_info_from_pos_df(pos_df)
    day_values = {}
    all_vals = []
    for day in DAYS:
        vals = day_sensor_values(sensor_day_df, int(day), ch_names)
        day_values[int(day)] = vals
        for val in vals:
            all_vals.append(float(val))
    vmax = float(np.nanmax(np.abs(all_vals)))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Cannot determine late-window topomap color scale")
    fig, axes = plt.subplots(1, len(DAYS), figsize=(10.0, 2.7), squeeze=False)
    im = None
    for c, day in enumerate(DAYS):
        ax = axes[0, c]
        im, _ = mne.viz.plot_topomap(
            day_values[int(day)],
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="RdBu_r",
            vlim=(-vmax, vmax),
        )
        ax.set_title(f"D{day}", fontsize=10)
    fig.suptitle("Late-Window Whole-Window Haufe Topographies")
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.03, right=0.90, wspace=0.08)
    cax = fig.add_axes([0.92, 0.24, 0.018, 0.44])
    fig.colorbar(im, cax=cax, label="Haufe pattern")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_normalized_topomap_figure(sensor_day_df, pos_df, figures_dir):
    fig_path = (
        figures_dir / "mvpa_stim_locked_cat_late_window_haufe_topomaps_normalized.png"
    )
    info, ch_names = make_haufe_info_from_pos_df(pos_df)
    day_values = {}
    for day in DAYS:
        vals = day_sensor_values(sensor_day_df, int(day), ch_names)
        vals = vals - np.mean(vals)
        denom = float(np.max(np.abs(vals)))
        if not np.isfinite(denom) or denom <= np.finfo(float).eps:
            raise ValueError(f"Cannot normalize flat late-window map: day={day}")
        day_values[int(day)] = vals / denom
    fig, axes = plt.subplots(1, len(DAYS), figsize=(10.0, 2.7), squeeze=False)
    im = None
    for c, day in enumerate(DAYS):
        ax = axes[0, c]
        im, _ = mne.viz.plot_topomap(
            day_values[int(day)],
            info,
            axes=ax,
            show=False,
            contours=0,
            cmap="RdBu_r",
            vlim=(-1.0, 1.0),
        )
        ax.set_title(f"D{day}", fontsize=10)
    fig.suptitle("Late-Window Shape-Normalized Haufe Topographies")
    fig.subplots_adjust(top=0.76, bottom=0.12, left=0.03, right=0.90, wspace=0.08)
    cax = fig.add_axes([0.92, 0.24, 0.018, 0.44])
    fig.colorbar(im, cax=cax, label="Demeaned / panel max abs")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def linkage_matrix(clusters_df):
    g = clusters_df[clusters_df["row_type"] == "linkage"].sort_values("merge_index")
    if len(g) != len(DAYS) - 1:
        raise ValueError("Missing late-window linkage rows")
    z = np.zeros((len(g), 4), dtype=float)
    for i, row in enumerate(g.itertuples()):
        z[i, 0] = float(row.child_1)
        z[i, 1] = float(row.child_2)
        z[i, 2] = float(row.distance)
        z[i, 3] = float(row.n_members)
    return z


def save_cluster_figure(clusters_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_late_window_haufe_clusters.png"
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    z = linkage_matrix(clusters_df)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    dendrogram(z, labels=labels, ax=ax, color_threshold=0.0)
    ax.set_title("Late-Window Haufe Day Clustering")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_mvpa_stim_locked_cat_late_window(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    day_csv = output_dir / "mvpa_stim_locked_cat_late_window_day_auc.csv"
    sensor_day_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_sensor_day_mean.csv"
    pos_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_channel_positions.csv"
    similarity_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_similarity.csv"
    clusters_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_clusters.csv"

    day_df = require_csv(day_csv)
    sensor_day_df = require_csv(sensor_day_csv)
    pos_df = require_csv(pos_csv)
    similarity_df = require_csv(similarity_csv)
    clusters_df = require_csv(clusters_csv)

    auc_path = save_auc_figure(day_df, figures_dir)
    similarity_path = save_similarity_figure(similarity_df, figures_dir)
    topomap_path = save_topomap_figure(sensor_day_df, pos_df, figures_dir)
    normalized_topomap_path = save_normalized_topomap_figure(
        sensor_day_df,
        pos_df,
        figures_dir,
    )
    cluster_path = save_cluster_figure(clusters_df, figures_dir)
    print(f"[MVPA late-window] Wrote {auc_path}")
    print(f"[MVPA late-window] Wrote {similarity_path}")
    print(f"[MVPA late-window] Wrote {topomap_path}")
    print(f"[MVPA late-window] Wrote {normalized_topomap_path}")
    print(f"[MVPA late-window] Wrote {cluster_path}")
    return {
        "auc": auc_path,
        "similarity": similarity_path,
        "topomaps": topomap_path,
        "topomaps_normalized": normalized_topomap_path,
        "clusters": cluster_path,
    }


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_late_window()
