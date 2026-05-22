#!/usr/bin/env python3
"""Plot stimulus-feedback cross-epoch temporal generalization figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mvpa_stim_feedback_cat_tg_analysis import FIGURES_DIR, OUTPUT_DIR


def _direction_label(direction: str) -> str:
    return "Stim -> Feedback" if direction == "stim_to_feedback" else "Feedback -> Stim"


def save_fig_mvpa_stim_feedback_cat_tg(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    subject_csv = output_dir / "mvpa_stim_feedback_cat_tg_subject_level.csv"
    matrix_csv = output_dir / "mvpa_stim_feedback_cat_tg_timegen_day_mean.csv"
    matrix_day_pair_csv = output_dir / "mvpa_stim_feedback_cat_tg_timegen_day_pair_mean.csv"
    if (
        (not subject_csv.exists())
        or (not matrix_csv.exists())
        or (not matrix_day_pair_csv.exists())
    ):
        raise FileNotFoundError(
            "Missing TG cross-epoch output in "
            f"{output_dir}. Run mvpa_stim_feedback_cat_tg_analysis.py first."
        )
    subject_df = pd.read_csv(subject_csv)
    matrix_df = pd.read_csv(matrix_csv)
    matrix_day_pair_df = pd.read_csv(matrix_day_pair_csv)
    if subject_df.empty or matrix_df.empty or matrix_day_pair_df.empty:
        raise ValueError(
            "Empty TG cross-epoch output table in "
            f"{subject_csv}, {matrix_csv}, or {matrix_day_pair_csv}"
        )

    fig_5x5_paths: dict[str, Path] = {}
    day_grid = [1, 2, 3, 4, 5]
    direction_order = ["stim_to_feedback", "feedback_to_stim"]
    vmin = float(matrix_day_pair_df["auc_mean"].min())
    vmax = float(matrix_day_pair_df["auc_mean"].max())
    for direction in direction_order:
        d_dir = matrix_day_pair_df[matrix_day_pair_df["direction"] == direction].copy()
        if d_dir.empty:
            raise ValueError(f"Missing cross-epoch direction in {matrix_day_pair_csv}: {direction}")
        fig, axes = plt.subplots(5, 5, figsize=(18, 16), squeeze=False)
        im = None
        for i, train_day in enumerate(day_grid):
            for j, test_day in enumerate(day_grid):
                ax = axes[i, j]
                g = d_dir[
                    (d_dir["train_day"] == train_day)
                    & (d_dir["test_day"] == test_day)
                ].copy()
                if g.empty:
                    raise ValueError(
                        f"Missing cross-epoch day pair in {matrix_day_pair_csv}: "
                        f"direction={direction}, train_day={train_day}, "
                        f"test_day={test_day}"
                    )
                train_times = np.sort(g["train_time_sec"].unique().astype(float))
                test_times = np.sort(g["test_time_sec"].unique().astype(float))
                mat = np.full((len(train_times), len(test_times)), np.nan)
                for _, row in g.iterrows():
                    ii = int(
                        np.where(train_times == float(row["train_time_sec"]))[0][0]
                    )
                    jj = int(
                        np.where(test_times == float(row["test_time_sec"]))[0][0]
                    )
                    mat[ii, jj] = float(row["auc_mean"])
                im = ax.imshow(
                    np.ma.masked_invalid(mat),
                    origin="lower",
                    aspect="auto",
                    extent=[
                        float(test_times.min()),
                        float(test_times.max()),
                        float(train_times.min()),
                        float(train_times.max()),
                    ],
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                )
                ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
                ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
                if i == 0:
                    ax.set_title(f"Test D{test_day}", fontsize=9)
                if j == 0:
                    ax.set_ylabel(f"Train D{train_day}", fontsize=8)
                if i == 4:
                    ax.set_xlabel("Test Time (s)")
                else:
                    ax.set_xticklabels([])
                if j != 0:
                    ax.set_yticklabels([])
        fig.suptitle(
            f"Cross-Epoch Temporal Generalization by Day Pair (A/B) - "
            f"{_direction_label(direction)}"
        )
        fig.subplots_adjust(
            top=0.94, bottom=0.05, left=0.05, right=0.90, wspace=0.30, hspace=0.35
        )
        cax = fig.add_axes([0.92, 0.12, 0.015, 0.74])
        fig.colorbar(im, cax=cax, label="AUC")
        fig_5x5_path = figures_dir / f"mvpa_stim_feedback_cat_tg_timegen_matrices_{direction}_5x5.png"
        fig.savefig(fig_5x5_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        fig_5x5_paths[direction] = fig_5x5_path
    return {
        "figure_path": fig_5x5_paths.get("stim_to_feedback"),
        "timegen_figure_path": fig_5x5_paths.get("stim_to_feedback"),
        "timegen_figure_paths": fig_5x5_paths,
    }




if __name__ == "__main__":
    save_fig_mvpa_stim_feedback_cat_tg()
