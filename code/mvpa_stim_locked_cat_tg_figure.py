#!/usr/bin/env python3
"""Plot stimulus-locked cross-day temporal generalization figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mvpa_stim_locked_cat_tg_analysis import FIGURES_DIR, OUTPUT_DIR


def save_fig_mvpa_stim_locked_cat_tg(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cross_day_mean_csv = output_dir / "mvpa_stim_locked_cat_tg_day_mean.csv"
    cross_matrix_day_mean_csv = output_dir / "mvpa_stim_locked_cat_tg_timegen_day_mean.csv"
    if (not cross_day_mean_csv.exists()) or (not cross_matrix_day_mean_csv.exists()):
        raise FileNotFoundError(
            f"Missing TG cross-day output in {output_dir}. "
            "Run mvpa_stim_locked_cat_tg_analysis.py first."
        )
    cross_day_mean_df = pd.read_csv(cross_day_mean_csv)
    if cross_day_mean_df.empty:
        raise ValueError(f"Empty TG cross-day output table: {cross_day_mean_csv}")
    d_mat = pd.read_csv(cross_matrix_day_mean_csv)
    if d_mat.empty:
        raise ValueError(f"Empty TG cross-day timegen output table: {cross_matrix_day_mean_csv}")
    fig_cross = figures_dir / "mvpa_stim_locked_cat_tg_transfer_5x4.png"
    fig_cross_timegen = figures_dir / "mvpa_stim_locked_cat_tg_timegen_matrices_5x5.png"

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    day_grid = sorted({1, 2, 3, 4, 5})
    mat = np.full((len(day_grid), len(day_grid)), np.nan)
    for _, r in cross_day_mean_df.iterrows():
        i = day_grid.index(int(r["train_day"]))
        j = day_grid.index(int(r["test_day"]))
        mat[i, j] = float(r["auc_mean"])
    missing_pairs = []
    for train_day in day_grid:
        for test_day in day_grid:
            i = day_grid.index(train_day)
            j = day_grid.index(test_day)
            if not np.isfinite(mat[i, j]):
                missing_pairs.append(f"train_day={train_day}, test_day={test_day}")
    if len(missing_pairs) > 0:
        raise ValueError(
            f"Missing day-pair rows in {cross_day_mean_csv}:\n"
            + "\n".join(missing_pairs)
        )
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="magma", aspect="equal")
    ax.set_xticks(range(len(day_grid)))
    ax.set_yticks(range(len(day_grid)))
    x_labels = []
    for d in day_grid:
        x_labels.append(f"D{d}")
    y_labels = []
    for d in day_grid:
        y_labels.append(f"D{d}")
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel("Test Day")
    ax.set_ylabel("Train Day")
    ax.set_title("Cross-Day Transfer (Diagonal Mean AUC)")
    for i in range(len(day_grid)):
        for j in range(len(day_grid)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
    fig.colorbar(im, ax=ax, shrink=0.9, label="AUC")
    fig.tight_layout()
    fig.savefig(fig_cross, dpi=150, bbox_inches="tight")
    plt.close(fig)

    d_diag = d_mat[d_mat["train_day"] == d_mat["test_day"]]
    d_offdiag = d_mat[d_mat["train_day"] != d_mat["test_day"]]
    if d_diag.empty:
        raise ValueError(f"Missing same-day TG matrix rows in {cross_matrix_day_mean_csv}")
    if d_offdiag.empty:
        raise ValueError(f"Missing cross-day TG matrix rows in {cross_matrix_day_mean_csv}")
    diag_vmin = float(d_diag["auc_mean"].min())
    diag_vmax = float(d_diag["auc_mean"].max())
    offdiag_vmin = float(d_offdiag["auc_mean"].min())
    offdiag_vmax = float(d_offdiag["auc_mean"].max())

    fig, axes = plt.subplots(5, 5, figsize=(19.5, 16), squeeze=False)
    im_diag = None
    im_offdiag = None
    for i, train_day in enumerate(day_grid):
        for j, test_day in enumerate(day_grid):
            ax = axes[i, j]
            g = d_mat[
                (d_mat["train_day"] == train_day)
                & (d_mat["test_day"] == test_day)
            ]
            if g.empty:
                raise ValueError(
                    f"Missing TG matrix day pair in {cross_matrix_day_mean_csv}: "
                    f"train_day={train_day}, test_day={test_day}"
                )
            pivot = g.pivot(
                index="train_time_sec", columns="test_time_sec", values="auc_mean"
            )
            if train_day == test_day:
                vmin = diag_vmin
                vmax = diag_vmax
            else:
                vmin = offdiag_vmin
                vmax = offdiag_vmax
            im = ax.imshow(
                pivot.to_numpy(),
                origin="lower",
                aspect="auto",
                extent=[
                    float(pivot.columns.min()),
                    float(pivot.columns.max()),
                    float(pivot.index.min()),
                    float(pivot.index.max()),
                ],
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            if train_day == test_day:
                im_diag = im
            else:
                im_offdiag = im
            ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
            ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
            ax.set_title(f"Train D{train_day} -> Test D{test_day}", fontsize=9)
            if i == len(day_grid) - 1:
                ax.set_xlabel("Test Time (s)")
            if j == 0:
                ax.set_ylabel("Train Time (s)")
    fig.suptitle("Cross-Day Temporal Generalization by Day Pair (AUC)")
    fig.subplots_adjust(
        top=0.94,
        bottom=0.05,
        left=0.05,
        right=0.86,
        wspace=0.30,
        hspace=0.35,
    )
    cax_diag = fig.add_axes([0.88, 0.12, 0.015, 0.74])
    cax_offdiag = fig.add_axes([0.93, 0.12, 0.015, 0.74])
    fig.colorbar(im_diag, cax=cax_diag, label="AUC same-day")
    fig.colorbar(im_offdiag, cax=cax_offdiag, label="AUC cross-day")
    fig.savefig(fig_cross_timegen, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"figure_path": fig_cross, "timegen_figure_path": fig_cross_timegen}


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_tg()
