#!/usr/bin/env python3
"""Plot late-window stimulus-locked category MVPA outputs."""

import json
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
    CLASSIFIERS,
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


def require_completed_progress(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing late-window MVPA progress file: {path}. "
            "Run mvpa_stim_locked_cat_late_window_analysis.py first."
        )
    payload = json.loads(path.read_text())
    stage = str(payload.get("stage", ""))
    done = int(payload.get("done", -1))
    total = int(payload.get("total", -2))
    if stage != "completed" or done != total:
        raise RuntimeError(
            f"Late-window MVPA outputs are incomplete: stage={stage}, "
            f"done={done}, total={total}. Re-run the analysis before plotting."
        )
    return payload


def validate_complete_classifier_outputs(session_df, day_df, sensor_day_df, similarity_df):
    missing_classifiers = []
    for classifier in CLASSIFIERS:
        if classifier not in set(session_df["classifier"].dropna()):
            missing_classifiers.append(classifier)
    if len(missing_classifiers) > 0:
        raise ValueError(
            "Late-window session output missing classifiers: "
            + ", ".join(missing_classifiers)
        )

    base_pairs = None
    for classifier in CLASSIFIERS:
        g = session_df[session_df["classifier"] == classifier]
        pairs = set()
        for _, row in g.iterrows():
            pairs.add((int(row["subject"]), int(row["day"])))
        if base_pairs is None:
            base_pairs = pairs
        elif pairs != base_pairs:
            raise ValueError(
                f"Classifier {classifier} has a different subject/day set. "
                "Refusing to plot partial classifier outputs."
            )

    for classifier in CLASSIFIERS:
        for day in DAYS:
            n_rows = int(
                np.sum(
                    (day_df["classifier"] == classifier)
                    & (day_df["day"] == int(day))
                )
            )
            if n_rows != 1:
                raise ValueError(
                    f"Expected one AUC day row for classifier={classifier}, "
                    f"day={day}; found {n_rows}"
                )
            n_channels = int(
                np.sum(
                    (sensor_day_df["classifier"] == classifier)
                    & (sensor_day_df["day"] == int(day))
                )
            )
            if n_channels < 3:
                raise ValueError(
                    f"Too few topomap channels for classifier={classifier}, "
                    f"day={day}; found {n_channels}"
                )
        g_sim = similarity_df[
            (similarity_df["row_type"] == "group")
            & (similarity_df["classifier"] == classifier)
        ]
        if len(g_sim) != 10:
            raise ValueError(
                f"Expected 10 group day-pair similarity rows for "
                f"classifier={classifier}; found {len(g_sim)}"
            )


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
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    colors = ["black", "tab:blue", "tab:orange"]
    for idx, classifier in enumerate(CLASSIFIERS):
        g = day_df[day_df["classifier"] == classifier].sort_values("day")
        if len(g) != len(DAYS):
            raise ValueError(
                f"Late-window AUC table missing days for classifier={classifier}"
            )
        ax.errorbar(
            g["day"],
            g["auc_mean"],
            yerr=g["auc_sem"],
            color=colors[idx],
            marker="o",
            linewidth=1.8,
            capsize=3,
            label=classifier,
        )
    ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_xticks(DAYS)
    ax.set_xlabel("Day")
    ax.set_ylabel("AUC")
    ax.set_title(
        f"Late-Window Category Decoding ({WINDOW_START_SEC:.2f}-{WINDOW_END_SEC:.2f}s)"
    )
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def similarity_matrix(sim_df, classifier):
    g = sim_df[
        (sim_df["row_type"] == "group")
        & (sim_df["classifier"] == classifier)
    ]
    if g.empty:
        raise ValueError(f"Missing late-window group similarity rows: {classifier}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row["similarity_mean"])
        mat[j, i] = float(row["similarity_mean"])
    return mat


def save_similarity_figure(sim_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_late_window_haufe_similarity.png"
    mats = {}
    vals_all = []
    for classifier in CLASSIFIERS:
        mat = similarity_matrix(sim_df, classifier)
        mats[classifier] = mat
        vals = mat[np.isfinite(mat)]
        for val in vals:
            vals_all.append(float(val))
    if len(vals_all) == 0:
        raise ValueError("No finite late-window Haufe similarity values")
    vmax = float(np.nanmax(np.abs(vals_all)))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Cannot determine late-window similarity color scale")
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.82")
    fig, axes = plt.subplots(1, len(CLASSIFIERS), figsize=(10.8, 3.7), squeeze=False)
    im = None
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    for c, classifier in enumerate(CLASSIFIERS):
        ax = axes[0, c]
        mat = mats[classifier]
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            origin="upper",
            cmap=cmap,
            vmin=-vmax,
            vmax=vmax,
        )
        ax.set_title(classifier)
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
    fig.suptitle("Late-Window Haufe Pattern Similarity")
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.06, right=0.90, wspace=0.35)
    cax = fig.add_axes([0.92, 0.24, 0.018, 0.46])
    fig.colorbar(im, cax=cax, label="Pattern similarity r")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def day_sensor_values(sensor_day_df, day, classifier, ch_names):
    g = sensor_day_df[
        (sensor_day_df["day"] == int(day))
        & (sensor_day_df["classifier"] == classifier)
    ]
    if g.empty:
        raise ValueError(
            f"Missing late-window sensor day rows: day={day}, classifier={classifier}"
        )
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
    for classifier in CLASSIFIERS:
        for day in DAYS:
            vals = day_sensor_values(sensor_day_df, int(day), classifier, ch_names)
            day_values[(classifier, int(day))] = vals
            for val in vals:
                all_vals.append(float(val))
    vmax = float(np.nanmax(np.abs(all_vals)))
    if not np.isfinite(vmax) or vmax <= 0:
        raise ValueError("Cannot determine late-window topomap color scale")
    fig, axes = plt.subplots(
        len(CLASSIFIERS),
        len(DAYS),
        figsize=(10.0, 6.8),
        squeeze=False,
    )
    im = None
    for r, classifier in enumerate(CLASSIFIERS):
        for c, day in enumerate(DAYS):
            ax = axes[r, c]
            im, _ = mne.viz.plot_topomap(
                day_values[(classifier, int(day))],
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-vmax, vmax),
            )
            if r == 0:
                ax.set_title(f"D{day}", fontsize=10)
            if c == 0:
                ax.set_ylabel(classifier, fontsize=9)
    fig.suptitle("Late-Window Whole-Window Haufe Topographies")
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.05, right=0.90, wspace=0.08, hspace=0.18)
    cax = fig.add_axes([0.92, 0.20, 0.018, 0.58])
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
    for classifier in CLASSIFIERS:
        for day in DAYS:
            vals = day_sensor_values(sensor_day_df, int(day), classifier, ch_names)
            vals = vals - np.mean(vals)
            denom = float(np.max(np.abs(vals)))
            if not np.isfinite(denom) or denom <= np.finfo(float).eps:
                raise ValueError(
                    f"Cannot normalize flat late-window map: "
                    f"classifier={classifier}, day={day}"
                )
            day_values[(classifier, int(day))] = vals / denom
    fig, axes = plt.subplots(
        len(CLASSIFIERS),
        len(DAYS),
        figsize=(10.0, 6.8),
        squeeze=False,
    )
    im = None
    for r, classifier in enumerate(CLASSIFIERS):
        for c, day in enumerate(DAYS):
            ax = axes[r, c]
            im, _ = mne.viz.plot_topomap(
                day_values[(classifier, int(day))],
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-1.0, 1.0),
            )
            if r == 0:
                ax.set_title(f"D{day}", fontsize=10)
            if c == 0:
                ax.set_ylabel(classifier, fontsize=9)
    fig.suptitle("Late-Window Shape-Normalized Haufe Topographies")
    fig.subplots_adjust(top=0.88, bottom=0.06, left=0.05, right=0.90, wspace=0.08, hspace=0.18)
    cax = fig.add_axes([0.92, 0.20, 0.018, 0.58])
    fig.colorbar(im, cax=cax, label="Demeaned / panel max abs")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def linkage_matrix(clusters_df, classifier):
    g = clusters_df[
        (clusters_df["row_type"] == "linkage")
        & (clusters_df["classifier"] == classifier)
    ].sort_values("merge_index")
    if len(g) != len(DAYS) - 1:
        raise ValueError(f"Missing late-window linkage rows: {classifier}")
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
    fig, axes = plt.subplots(1, len(CLASSIFIERS), figsize=(10.8, 3.4), squeeze=False)
    for c, classifier in enumerate(CLASSIFIERS):
        ax = axes[0, c]
        z = linkage_matrix(clusters_df, classifier)
        dendrogram(z, labels=labels, ax=ax, color_threshold=0.0)
        ax.set_title(classifier)
        ax.set_ylabel("Distance")
    fig.suptitle("Late-Window Haufe Day Clustering")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
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
    session_csv = output_dir / "mvpa_stim_locked_cat_late_window_session_auc.csv"
    progress_json = output_dir / "mvpa_stim_locked_cat_late_window_progress.json"

    require_completed_progress(progress_json)
    session_df = require_csv(session_csv)
    day_df = require_csv(day_csv)
    sensor_day_df = require_csv(sensor_day_csv)
    pos_df = require_csv(pos_csv)
    similarity_df = require_csv(similarity_csv)
    clusters_df = require_csv(clusters_csv)
    validate_complete_classifier_outputs(
        session_df,
        day_df,
        sensor_day_df,
        similarity_df,
    )

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
