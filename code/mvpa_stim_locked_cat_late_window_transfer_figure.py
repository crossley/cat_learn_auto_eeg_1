#!/usr/bin/env python3
"""Plot late-window stimulus-locked category transfer across days."""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            f"Missing late-window transfer output: {path}. "
            "Run mvpa_stim_locked_cat_late_window_transfer_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty late-window transfer output: {path}")
    return d


def require_completed_progress(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing late-window transfer progress file: {path}. "
            "Run mvpa_stim_locked_cat_late_window_transfer_analysis.py first."
        )
    payload = json.loads(path.read_text())
    stage = str(payload.get("stage", ""))
    done = int(payload.get("done", -1))
    total = int(payload.get("total", -2))
    if stage != "completed" or done != total:
        raise RuntimeError(
            f"Late-window transfer outputs are incomplete: stage={stage}, "
            f"done={done}, total={total}. Re-run the analysis before plotting."
        )
    return payload


def transfer_matrix(group_df, classifier):
    g = group_df[group_df["classifier"] == classifier]
    if g.empty:
        raise ValueError(f"Missing transfer rows for classifier={classifier}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    n_mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["train_day"]))
        j = DAYS.index(int(row["test_day"]))
        mat[i, j] = float(row["auc_mean"])
        n_mat[i, j] = float(row["n_subjects"])
    missing = []
    for i, train_day in enumerate(DAYS):
        for j, test_day in enumerate(DAYS):
            if not np.isfinite(mat[i, j]):
                missing.append(f"D{train_day}->D{test_day}")
    if len(missing) > 0:
        raise ValueError(
            f"Missing transfer cells for classifier={classifier}: "
            + ", ".join(missing)
        )
    return mat, n_mat


def save_transfer_figure(group_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_late_window_transfer_5x5.png"
    mats = {}
    all_vals = []
    for classifier in CLASSIFIERS:
        mat, n_mat = transfer_matrix(group_df, classifier)
        mats[classifier] = (mat, n_mat)
        vals = mat[np.isfinite(mat)]
        for val in vals:
            all_vals.append(float(val))
    if len(all_vals) == 0:
        raise ValueError("No finite late-window transfer values to plot")
    vmin = min(0.5, float(np.nanmin(all_vals)))
    vmax = float(np.nanmax(all_vals))
    if vmax <= vmin:
        vmax = vmin + 0.01
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.82")
    fig, axes = plt.subplots(1, len(CLASSIFIERS), figsize=(11.2, 3.8), squeeze=False)
    im = None
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    for c, classifier in enumerate(CLASSIFIERS):
        ax = axes[0, c]
        mat, n_mat = mats[classifier]
        im = ax.imshow(mat, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(classifier)
        ax.set_xticks(range(len(DAYS)))
        ax.set_yticks(range(len(DAYS)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Test day")
        ax.set_ylabel("Train day")
        for i in range(len(DAYS)):
            for j in range(len(DAYS)):
                val = float(mat[i, j])
                color = "white"
                if val > (vmin + 0.65 * (vmax - vmin)):
                    color = "black"
                ax.text(
                    j,
                    i,
                    f"{val:.3f}\nn={int(n_mat[i, j])}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
    fig.suptitle(
        f"Late-Window Day Transfer ({WINDOW_START_SEC:.2f}-{WINDOW_END_SEC:.2f}s)"
    )
    fig.subplots_adjust(top=0.80, bottom=0.13, left=0.06, right=0.90, wspace=0.35)
    cax = fig.add_axes([0.92, 0.24, 0.018, 0.46])
    fig.colorbar(im, cax=cax, label="AUC")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_offdiag_figure(group_df, figures_dir):
    fig_path = figures_dir / "mvpa_stim_locked_cat_late_window_transfer_offdiag.png"
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = ["black", "tab:blue", "tab:orange"]
    for idx, classifier in enumerate(CLASSIFIERS):
        g = group_df[
            (group_df["classifier"] == classifier)
            & (group_df["train_day"] != group_df["test_day"])
        ].copy()
        if g.empty:
            raise ValueError(f"Missing off-diagonal transfer rows: {classifier}")
        rows = []
        for distance, g_dist in g.groupby("day_distance"):
            rows.append(
                {
                    "day_distance": int(distance),
                    "auc_mean": float(np.mean(g_dist["auc_mean"])),
                    "auc_sem": (
                        float(np.std(g_dist["auc_mean"], ddof=1) / np.sqrt(len(g_dist)))
                        if len(g_dist) > 1 else np.nan
                    ),
                }
            )
        d = pd.DataFrame(rows).sort_values("day_distance")
        ax.errorbar(
            d["day_distance"],
            d["auc_mean"],
            yerr=d["auc_sem"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=colors[idx],
            label=classifier,
        )
    ax.axhline(0.5, color="0.55", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Day distance")
    ax.set_ylabel("Mean off-diagonal AUC")
    ax.set_title("Late-Window Transfer by Day Distance")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_mvpa_stim_locked_cat_late_window_transfer(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    progress_json = output_dir / "mvpa_stim_locked_cat_late_window_transfer_progress.json"
    group_csv = output_dir / "mvpa_stim_locked_cat_late_window_transfer_group_pairs.csv"

    require_completed_progress(progress_json)
    group_df = require_csv(group_csv)
    transfer_path = save_transfer_figure(group_df, figures_dir)
    offdiag_path = save_offdiag_figure(group_df, figures_dir)
    print(f"[MVPA late-window transfer] Wrote {transfer_path}")
    print(f"[MVPA late-window transfer] Wrote {offdiag_path}")
    return {
        "transfer": transfer_path,
        "offdiag": offdiag_path,
    }


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_late_window_transfer()
