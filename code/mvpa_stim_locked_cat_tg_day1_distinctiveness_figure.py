#!/usr/bin/env python3
"""Plot Day-1 distinctiveness TG summary figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mvpa_stim_locked_cat_tg_day1_distinctiveness_analysis import FIGURES_DIR, OUTPUT_DIR


def plot_day_pair_window_matrices_by_summary(matrix_df, fig_path):
    if matrix_df.empty:
        raise ValueError("Empty Day-1 distinctiveness matrix table")
    summaries = ["square_mean", "diagonal_mean", "top10_mean"]
    retained_summaries = []
    summary_set = set(matrix_df["summary"])
    for s in summaries:
        if s in summary_set:
            retained_summaries.append(s)
        else:
            raise ValueError(f"Missing Day-1 distinctiveness summary: {s}")
    summaries = retained_summaries
    days = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(
        len(summaries), 2, figsize=(9.4, 4.1 * len(summaries)), squeeze=False
    )
    for r, summary in enumerate(summaries):
        d_summary = matrix_df[matrix_df["summary"] == summary]
        if d_summary.empty:
            raise ValueError(f"Missing Day-1 distinctiveness summary rows: {summary}")
        vmin = float(d_summary["auc_mean"].min())
        vmax = float(d_summary["auc_mean"].max())
        for c, window_name in enumerate(["early", "late"]):
            ax = axes[r, c]
            mat = np.full((len(days), len(days)), np.nan)
            g = d_summary[d_summary["window"] == window_name]
            if g.empty:
                raise ValueError(
                    "Missing Day-1 distinctiveness window rows: "
                    f"summary={summary}, window={window_name}"
                )
            for _, row in g.iterrows():
                i = days.index(int(row["train_day"]))
                j = days.index(int(row["test_day"]))
                mat[i, j] = float(row["auc_mean"])
            missing_pairs = []
            for train_day in days:
                for test_day in days:
                    i = days.index(train_day)
                    j = days.index(test_day)
                    if not np.isfinite(mat[i, j]):
                        missing_pairs.append(
                            f"train_day={train_day}, test_day={test_day}"
                        )
            if len(missing_pairs) > 0:
                raise ValueError(
                    "Missing Day-1 distinctiveness matrix cells: "
                    f"summary={summary}, window={window_name}\n"
                    + "\n".join(missing_pairs)
                )
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                origin="upper",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xticks(range(len(days)))
            ax.set_yticks(range(len(days)))
            x_labels = []
            for day in days:
                x_labels.append(f"D{day}")
            y_labels = []
            for day in days:
                y_labels.append(f"D{day}")
            ax.set_xticklabels(x_labels)
            ax.set_yticklabels(y_labels)
            ax.set_xlabel("Test day")
            ax.set_ylabel("Train day")
            ax.set_title(f"{summary} | {window_name}")
            for i in range(len(days)):
                for j in range(len(days)):
                    if np.isfinite(mat[i, j]):
                        color = (
                            "black"
                            if mat[i, j] > (vmin + 0.65 * (vmax - vmin))
                            else "white"
                        )
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.3f}",
                            ha="center",
                            va="center",
                            color=color,
                            fontsize=8,
                        )
            fig.colorbar(im, ax=ax, shrink=0.75, label="AUC")
    fig.suptitle("TG Window AUC by Day Pair and Summary (Diagonal = Within-Day)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fig_mvpa_stim_locked_cat_tg_day1_distinctiveness(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_matrix_csv = output_dir / "mvpa_stim_locked_cat_tg_day_pair_window_auc_matrix_by_summary.csv"
    if not summary_matrix_csv.exists():
        raise FileNotFoundError(
            f"Missing Day-1 distinctiveness output in {output_dir}. "
            "Run mvpa_stim_locked_cat_tg_day1_distinctiveness_analysis.py first."
        )
    fig_matrices_by_summary = figures_dir / "mvpa_stim_locked_cat_tg_day_pair_window_matrices_by_summary.png"
    plot_day_pair_window_matrices_by_summary(
        pd.read_csv(summary_matrix_csv), fig_matrices_by_summary
    )
    return {"matrices_by_summary": fig_matrices_by_summary}


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_tg_day1_distinctiveness()
