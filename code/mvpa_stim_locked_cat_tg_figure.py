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
    if not cross_day_mean_csv.exists():
        raise FileNotFoundError(
            f"Missing TG cross-day output in {output_dir}. "
            "Run mvpa_stim_locked_cat_tg_analysis.py first."
        )
    cross_day_mean_df = pd.read_csv(cross_day_mean_csv)
    fig_cross = figures_dir / "mvpa_stim_locked_cat_tg_transfer_5x4.png"
    fig_cross_timegen = figures_dir / "mvpa_stim_locked_cat_tg_timegen_matrices_5x5.png"

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    day_grid = sorted({1, 2, 3, 4, 5})
    mat = np.full((len(day_grid), len(day_grid)), np.nan)
    if not cross_day_mean_df.empty:
        for _, r in cross_day_mean_df.iterrows():
            i = day_grid.index(int(r["train_day"]))
            j = day_grid.index(int(r["test_day"]))
            mat[i, j] = float(r["auc_mean"])
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="magma", aspect="equal")
    ax.set_xticks(range(len(day_grid)))
    ax.set_yticks(range(len(day_grid)))
    ax.set_xticklabels([f"D{d}" for d in day_grid])
    ax.set_yticklabels([f"D{d}" for d in day_grid])
    ax.set_xlabel("Test Day")
    ax.set_ylabel("Train Day")
    ax.set_title("Cross-Day Transfer (Diagonal Mean AUC)")
    for i in range(len(day_grid)):
        for j in range(len(day_grid)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
            elif i == j:
                ax.text(j, i, "-", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.9, label="AUC")
    fig.tight_layout()
    fig.savefig(fig_cross, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if cross_matrix_day_mean_csv.exists():
        d_mat = pd.read_csv(cross_matrix_day_mean_csv)
        if not d_mat.empty:
            fig, axes = plt.subplots(5, 5, figsize=(18, 16), squeeze=False)
            vmin = float(d_mat["auc_mean"].min())
            vmax = float(d_mat["auc_mean"].max())
            im = None
            for i, train_day in enumerate(day_grid):
                for j, test_day in enumerate(day_grid):
                    ax = axes[i, j]
                    g = d_mat[
                        (d_mat["train_day"] == train_day)
                        & (d_mat["test_day"] == test_day)
                    ]
                    if g.empty:
                        ax.axis("off")
                        continue
                    pivot = g.pivot(
                        index="train_time_sec", columns="test_time_sec", values="auc_mean"
                    )
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
                right=0.90,
                wspace=0.30,
                hspace=0.35,
            )
            cax = fig.add_axes([0.92, 0.12, 0.015, 0.74])
            if im is not None:
                fig.colorbar(im, cax=cax, label="AUC")
            fig.savefig(fig_cross_timegen, dpi=150, bbox_inches="tight")
            plt.close(fig)

    return {"figure_path": fig_cross, "timegen_figure_path": fig_cross_timegen}


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_tg()
