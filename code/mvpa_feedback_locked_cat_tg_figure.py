#!/usr/bin/env python3
"""Plot feedback-locked temporal generalization figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mvpa_feedback_locked_cat_tg_analysis import FIGURES_DIR, OUTPUT_DIR


def vector_corr(x_vec, y_vec):
    valid = np.isfinite(x_vec) & np.isfinite(y_vec)
    if int(np.sum(valid)) < 3:
        return np.nan
    x_use = x_vec[valid] - np.nanmean(x_vec[valid])
    y_use = y_vec[valid] - np.nanmean(y_vec[valid])
    denom = np.sqrt(np.sum(x_use**2) * np.sum(y_use**2))
    if (not np.isfinite(denom)) or denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x_use * y_use) / denom)


def save_fig_mvpa_feedback_locked_cat_tg(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cross_day_mean_csv = output_dir / "mvpa_feedback_locked_cat_tg_day_mean.csv"
    cross_matrix_day_mean_csv = output_dir / "mvpa_feedback_locked_cat_tg_timegen_day_mean.csv"
    if not cross_day_mean_csv.exists():
        raise FileNotFoundError(
            f"Missing TG feedback-cat output in {output_dir}. "
            "Run mvpa_feedback_locked_cat_tg_analysis.py first."
        )
    cross_day_mean_df = pd.read_csv(cross_day_mean_csv)
    fig_cross = figures_dir / "mvpa_feedback_locked_cat_tg_transfer_5x4.png"
    fig_cross_timegen = figures_dir / "mvpa_feedback_locked_cat_tg_timegen_matrices_5x5.png"

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
    ax.set_title("Feedback-Locked Transfer (Diagonal Mean AUC)")
    for i in range(len(day_grid)):
        for j in range(len(day_grid)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
            elif i == j:
                ax.text(j, i, "—", ha="center", va="center", color="black")
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
            fig.suptitle("Feedback-Locked Temporal Generalization by Day Pair (AUC)")
            fig.subplots_adjust(
                top=0.94,
                bottom=0.05,
                left=0.05,
                right=0.90,
                wspace=0.30,
                hspace=0.35,
            )
            cax = fig.add_axes([0.92, 0.12, 0.015, 0.74])
            fig.colorbar(im, cax=cax, label="AUC")
            fig.savefig(fig_cross_timegen, dpi=150, bbox_inches="tight")
            plt.close(fig)

    return {
        "figure_path": fig_cross,
        "timegen_figure_path": fig_cross_timegen,
    }


def save_feedback_haufe_figure(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    haufe_day_mean_csv = output_dir / "mvpa_feedback_locked_cat_tg_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_feedback_locked_cat_tg_haufe_channel_positions.csv"
    similarity_fig = figures_dir / "mvpa_feedback_locked_cat_tg_haufe_similarity_timegen_matrices_5x5.png"

    if (not haufe_day_mean_csv.exists()) or (not haufe_channel_pos_csv.exists()):
        return {"similarity_figure_path": None}

    haufe_df = pd.read_csv(haufe_day_mean_csv)
    pos_df = pd.read_csv(haufe_channel_pos_csv)
    if haufe_df.empty or pos_df.empty:
        return {"similarity_figure_path": None}

    ch_names = pos_df["channel"].tolist()
    by_day = {}
    for day, d_day in haufe_df.groupby("day"):
        times = np.sort(d_day["time_sec"].dropna().unique().astype(float))
        vecs = {}
        for t in times:
            d_t = d_day[np.isclose(d_day["time_sec"], t)]
            vecs[float(t)] = (
                d_t.set_index("channel").reindex(ch_names)["pattern_mean"]
                .to_numpy(dtype=float)
            )
        by_day[int(day)] = (times, vecs)

    fig, axes = plt.subplots(5, 5, figsize=(18.0, 16.0), squeeze=False)
    im = None
    day_grid = [1, 2, 3, 4, 5]
    for i, train_day in enumerate(day_grid):
        for j, test_day in enumerate(day_grid):
            ax = axes[i, j]
            if train_day not in by_day or test_day not in by_day:
                ax.axis("off")
                continue
            train_times, train_vecs = by_day[train_day]
            test_times, test_vecs = by_day[test_day]
            mat = np.full((len(train_times), len(test_times)), np.nan)
            for ti, t_train in enumerate(train_times):
                x_vec = train_vecs[float(t_train)]
                for tj, t_test in enumerate(test_times):
                    y_vec = test_vecs[float(t_test)]
                    mat[ti, tj] = vector_corr(x_vec, y_vec)
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
                vmin=-1.0,
                vmax=1.0,
                cmap="RdBu_r",
            )
            ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
            ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
            if i == 0:
                ax.set_title(f"Test D{test_day}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Train-day time on D{train_day} (s)", fontsize=9)
            if i == 4:
                ax.set_xlabel("Test-day time (s)")
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

    fig.suptitle("Feedback Haufe Pattern Similarity by Day Pair (A/B)", y=0.98)
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.90, wspace=0.26, hspace=0.36)
    cax = fig.add_axes([0.92, 0.14, 0.015, 0.72])
    if im is not None:
        fig.colorbar(im, cax=cax, label="Pattern correlation")
    else:
        cax.axis("off")
    fig.savefig(similarity_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"similarity_figure_path": similarity_fig}




if __name__ == "__main__":
    save_fig_mvpa_feedback_locked_cat_tg()
    save_feedback_haufe_figure()
