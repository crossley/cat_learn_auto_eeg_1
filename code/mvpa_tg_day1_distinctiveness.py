#!/usr/bin/env python3
"""Day 1 distinctiveness analysis: how day-1-anchored cross-day TG compares to later-only pairs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis_utils import model_term_summary
from mvpa_tg_window_structure import (
    TG_WINDOWS,
    extract_tg_window_auc,
    summarize_tg_window_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "mvpa_tg_day1_distinctiveness"
FIGURES_DIR = PROJECT_DIR / "figures" / "mvpa_tg_day1_distinctiveness"


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_ols(formula, df, cluster_subject=True):
    model = smf.ols(formula, data=df).fit()
    if cluster_subject and "subject" in df.columns and df["subject"].nunique() > 1:
        return model.get_robustcov_results(cov_type="cluster", groups=df["subject"])
    return model


def add_day1_pair_labels(window_df):
    d = window_df.copy()
    d["includes_day1"] = (d["train_day"] == 1) | (d["test_day"] == 1)
    d["pair_group"] = np.where(d["includes_day1"], "day1_pair", "later_only")
    d["day1_pair_type"] = "later_only"
    d.loc[(d["train_day"] == 1) & (d["test_day"] != 1), "day1_pair_type"] = "day1_forward"
    d.loc[(d["train_day"] != 1) & (d["test_day"] == 1), "day1_pair_type"] = "day1_backward"
    d["other_day"] = np.nan
    d.loc[d["train_day"] == 1, "other_day"] = d.loc[d["train_day"] == 1, "test_day"]
    d.loc[d["test_day"] == 1, "other_day"] = d.loc[d["test_day"] == 1, "train_day"]
    return d


def fit_day1_distinctiveness(window_df):
    d = add_day1_pair_labels(window_df)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_auc", "window"]).copy()
    d["window"] = pd.Categorical(d["window"], categories=["early", "late"])
    d["pair_group"] = pd.Categorical(d["pair_group"], categories=["later_only", "day1_pair"])
    d["day1_pair_type"] = pd.Categorical(
        d["day1_pair_type"],
        categories=["later_only", "day1_forward", "day1_backward"],
    )

    model_group = _fit_ols("mean_auc ~ window * pair_group", d)
    group_terms = []
    for term in model_group.model.exog_names:
        if term == "Intercept":
            continue
        item = model_term_summary(model_group, term)
        item["model"] = "day1_pair_vs_later_only"
        group_terms.append(item)

    model_direction = _fit_ols("mean_auc ~ window * day1_pair_type", d)
    direction_terms = []
    for term in model_direction.model.exog_names:
        if term == "Intercept":
            continue
        item = model_term_summary(model_direction, term)
        item["model"] = "day1_forward_backward_vs_later_only"
        direction_terms.append(item)

    return pd.DataFrame(group_terms + direction_terms), model_group, model_direction


def plot_day1_pair_group_bars(window_df, fig_path):
    d = add_day1_pair_labels(window_df).dropna(subset=["mean_auc"]).copy()
    subject_group = (
        d.groupby(["subject", "window", "day1_pair_type"], as_index=False)["mean_auc"]
        .mean()
        .sort_values(["window", "day1_pair_type"])
    )
    plot_df = subject_group.groupby(["window", "day1_pair_type"], as_index=False).agg(
        auc_mean=("mean_auc", "mean"),
        auc_sem=("mean_auc", _sem),
        n_subjects=("subject", "nunique"),
    )
    pair_order = ["later_only", "day1_forward", "day1_backward"]
    labels = ["Later only", "D1 -> later", "Later -> D1"]
    colors = {"early": "tab:blue", "late": "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4), sharey=True, squeeze=False)
    for ax, window_name in zip(axes.ravel(), ["early", "late"]):
        g = plot_df[plot_df["window"] == window_name].set_index("day1_pair_type").reindex(pair_order)
        x = np.arange(len(pair_order))
        ax.bar(x, g["auc_mean"], yerr=g["auc_sem"], color=colors[window_name], alpha=0.75, capsize=3)
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(window_name.title())
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("Mean window AUC")
    fig.suptitle("Day 1 Distinctiveness: Pair-Type Contrast")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_day1_anchored_trajectories(window_df, fig_path):
    d = add_day1_pair_labels(window_df).dropna(subset=["mean_auc", "other_day"]).copy()
    d = d[d["day1_pair_type"].isin(["day1_forward", "day1_backward"])]
    subject_day = (
        d.groupby(["subject", "window", "day1_pair_type", "other_day"], as_index=False)["mean_auc"]
        .mean()
        .sort_values(["window", "day1_pair_type", "other_day"])
    )
    plot_df = subject_day.groupby(["window", "day1_pair_type", "other_day"], as_index=False).agg(
        auc_mean=("mean_auc", "mean"),
        auc_sem=("mean_auc", _sem),
        n_subjects=("subject", "nunique"),
    )
    colors = {"day1_forward": "tab:purple", "day1_backward": "tab:green"}
    labels = {"day1_forward": "D1 -> later", "day1_backward": "Later -> D1"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4), sharey=True, squeeze=False)
    for ax, window_name in zip(axes.ravel(), ["early", "late"]):
        for pair_type in ["day1_forward", "day1_backward"]:
            g = plot_df[
                (plot_df["window"] == window_name) & (plot_df["day1_pair_type"] == pair_type)
            ]
            if g.empty:
                continue
            ax.errorbar(
                g["other_day"],
                g["auc_mean"],
                yerr=g["auc_sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=colors[pair_type],
                label=labels[pair_type],
            )
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xlabel("Other day")
        ax.set_title(window_name.title())
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("Mean window AUC")
    axes.ravel()[0].legend(loc="best")
    fig.suptitle("Day 1 Anchored Cross-Day TG")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def extract_within_day_window_auc(
    within_csv: Path | str = PROJECT_DIR / "output" / "mvpa_tg_within_day" / "tg_within_day_subject_level.csv",
    windows: dict[str, tuple[float, float]] = TG_WINDOWS,
    summary: str = "square_mean",
):
    within_csv = Path(within_csv)
    if not within_csv.exists():
        return pd.DataFrame(
            columns=["subject", "train_day", "test_day", "day_distance", "window", "mean_auc", "n_cells"]
        )
    d = pd.read_csv(within_csv, low_memory=False)
    for col in ["subject", "day", "train_time_sec", "test_time_sec", "auc"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["subject", "day", "train_time_sec", "test_time_sec", "auc"]).copy()
    rows = []
    for (subject, day), g_day in d.groupby(["subject", "day"]):
        pivot = g_day.pivot_table(
            index="train_time_sec", columns="test_time_sec", values="auc", aggfunc="mean"
        )
        train_axis = pivot.index.to_numpy(dtype=float)
        test_axis = pivot.columns.to_numpy(dtype=float)
        auc_mat = pivot.to_numpy(dtype=float)
        for window_name, (tmin, tmax) in windows.items():
            train_mask = (train_axis >= tmin) & (train_axis <= tmax)
            test_mask = (test_axis >= tmin) & (test_axis <= tmax)
            win = auc_mat[np.ix_(train_mask, test_mask)]
            mean_auc, n_cells = summarize_tg_window_matrix(win, summary=summary)
            rows.append(
                {
                    "subject": int(subject),
                    "train_day": int(day),
                    "test_day": int(day),
                    "day_distance": 0,
                    "window": window_name,
                    "window_tmin": tmin,
                    "window_tmax": tmax,
                    "mean_auc": mean_auc,
                    "n_cells": n_cells,
                    "summary": summary,
                    "matrix_file": "",
                    "day1_pair_type": "within_day",
                    "pair_group": "within_day",
                    "includes_day1": int(day) == 1,
                }
            )
    return pd.DataFrame(rows)


def plot_day_pair_window_matrices(window_df, fig_path, within_df=None):
    d = window_df.dropna(subset=["mean_auc"]).copy()
    if within_df is not None and not within_df.empty:
        d = pd.concat([d, within_df.dropna(subset=["mean_auc"]).copy()], ignore_index=True)
    pair_mean = (
        d.groupby(["window", "train_day", "test_day"], as_index=False)["mean_auc"]
        .mean()
        .sort_values(["window", "train_day", "test_day"])
    )
    days = [1, 2, 3, 4, 5]
    vmin = float(pair_mean["mean_auc"].min()) if not pair_mean.empty else 0.45
    vmax = float(pair_mean["mean_auc"].max()) if not pair_mean.empty else 0.55
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), squeeze=False)
    for ax, window_name in zip(axes.ravel(), ["early", "late"]):
        mat = np.full((len(days), len(days)), np.nan)
        g = pair_mean[pair_mean["window"] == window_name]
        for _, row in g.iterrows():
            i = days.index(int(row["train_day"]))
            j = days.index(int(row["test_day"]))
            mat[i, j] = float(row["mean_auc"])
        im = ax.imshow(np.ma.masked_invalid(mat), origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(days)))
        ax.set_yticks(range(len(days)))
        ax.set_xticklabels([f"D{day}" for day in days])
        ax.set_yticklabels([f"D{day}" for day in days])
        ax.set_xlabel("Test day")
        ax.set_ylabel("Train day")
        ax.set_title(window_name.title())
        for i in range(len(days)):
            for j in range(len(days)):
                if np.isfinite(mat[i, j]):
                    color = "black" if mat[i, j] > (vmin + 0.65 * (vmax - vmin)) else "white"
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
    fig.suptitle("TG Window AUC by Day Pair (Diagonal = Within-Day)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="AUC")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_day_pair_window_matrices_by_summary(matrix_df, fig_path):
    summaries = ["square_mean", "diagonal_mean", "top10_mean"]
    summaries = [s for s in summaries if s in set(matrix_df["summary"])]
    days = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(
        len(summaries), 2, figsize=(9.4, 4.1 * len(summaries)), squeeze=False
    )
    for r, summary in enumerate(summaries):
        d_summary = matrix_df[matrix_df["summary"] == summary]
        vmin = float(d_summary["auc_mean"].min()) if not d_summary.empty else 0.45
        vmax = float(d_summary["auc_mean"].max()) if not d_summary.empty else 0.55
        for c, window_name in enumerate(["early", "late"]):
            ax = axes[r, c]
            mat = np.full((len(days), len(days)), np.nan)
            g = d_summary[d_summary["window"] == window_name]
            for _, row in g.iterrows():
                i = days.index(int(row["train_day"]))
                j = days.index(int(row["test_day"]))
                mat[i, j] = float(row["auc_mean"])
            im = ax.imshow(np.ma.masked_invalid(mat), origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(days)))
            ax.set_yticks(range(len(days)))
            ax.set_xticklabels([f"D{day}" for day in days])
            ax.set_yticklabels([f"D{day}" for day in days])
            ax.set_xlabel("Test day")
            ax.set_ylabel("Train day")
            ax.set_title(f"{summary} | {window_name}")
            for i in range(len(days)):
                for j in range(len(days)):
                    if np.isfinite(mat[i, j]):
                        color = "black" if mat[i, j] > (vmin + 0.65 * (vmax - vmin)) else "white"
                        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.75, label="AUC")
    fig.suptitle("TG Window AUC by Day Pair and Summary (Diagonal = Within-Day)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_day1_distinctiveness_analysis(
    matrix_dir: Path | str = PROJECT_DIR / "output" / "mvpa_tg_cross_day" / "tg_cross_day_subject_matrices",
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    window_df = add_day1_pair_labels(extract_tg_window_auc(matrix_dir=matrix_dir, summary="square_mean"))
    within_df = extract_within_day_window_auc()
    summary_matrix_rows = []
    summary_window_rows = []
    for summary_name in ["square_mean", "diagonal_mean", "top10_mean"]:
        d_cross = add_day1_pair_labels(extract_tg_window_auc(matrix_dir=matrix_dir, summary=summary_name))
        d_within = extract_within_day_window_auc(summary=summary_name)
        d_all = pd.concat(
            [d_cross.dropna(subset=["mean_auc"]), d_within.dropna(subset=["mean_auc"])],
            ignore_index=True,
        )
        summary_window_rows.append(d_all)
        summary_matrix_rows.append(
            d_all.groupby(["summary", "window", "train_day", "test_day"], as_index=False)
            .agg(
                auc_mean=("mean_auc", "mean"),
                auc_sem=("mean_auc", _sem),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["summary", "window", "train_day", "test_day"])
        )
    stats_df, _, _ = fit_day1_distinctiveness(window_df)
    group_summary = (
        window_df.dropna(subset=["mean_auc"])
        .groupby(["window", "day1_pair_type"], as_index=False)
        .agg(
            auc_mean=("mean_auc", "mean"),
            auc_sem=("mean_auc", _sem),
            n_subjects=("subject", "nunique"),
            n_rows=("mean_auc", "size"),
        )
        .sort_values(["window", "day1_pair_type"])
    )
    pair_matrix_source = pd.concat(
        [window_df.dropna(subset=["mean_auc"]), within_df.dropna(subset=["mean_auc"])],
        ignore_index=True,
    )
    pair_matrix = (
        pair_matrix_source.groupby(["window", "train_day", "test_day"], as_index=False)
        .agg(
            auc_mean=("mean_auc", "mean"),
            auc_sem=("mean_auc", _sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["window", "train_day", "test_day"])
    )

    window_csv = output_dir / "tg_day1_window_auc_subject_pairs.csv"
    stats_csv = output_dir / "tg_day1_distinctiveness_model_terms.csv"
    summary_csv = output_dir / "tg_day1_pair_type_summary.csv"
    matrix_csv = output_dir / "tg_day_pair_window_auc_matrix.csv"
    summary_window_csv = output_dir / "tg_day_pair_window_auc_subject_pairs_by_summary.csv"
    summary_matrix_csv = output_dir / "tg_day_pair_window_auc_matrix_by_summary.csv"
    fig_pair_types = figures_dir / "tg_day1_pair_type_contrast.png"
    fig_anchored = figures_dir / "tg_day1_anchored_trajectories.png"
    fig_matrices = figures_dir / "tg_day_pair_window_matrices.png"
    fig_matrices_by_summary = figures_dir / "tg_day_pair_window_matrices_by_summary.png"

    window_df.to_csv(window_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    group_summary.to_csv(summary_csv, index=False)
    pair_matrix.to_csv(matrix_csv, index=False)
    pd.concat(summary_window_rows, ignore_index=True).to_csv(summary_window_csv, index=False)
    pd.concat(summary_matrix_rows, ignore_index=True).to_csv(summary_matrix_csv, index=False)
    plot_day1_pair_group_bars(window_df, fig_pair_types)
    plot_day1_anchored_trajectories(window_df, fig_anchored)
    plot_day_pair_window_matrices(window_df, fig_matrices, within_df=within_df)
    plot_day_pair_window_matrices_by_summary(
        pd.concat(summary_matrix_rows, ignore_index=True), fig_matrices_by_summary
    )

    return {
        "window_df": window_df,
        "stats_df": stats_df,
        "group_summary": group_summary,
        "pair_matrix": pair_matrix,
        "window_csv": window_csv,
        "stats_csv": stats_csv,
        "summary_csv": summary_csv,
        "matrix_csv": matrix_csv,
        "summary_window_csv": summary_window_csv,
        "summary_matrix_csv": summary_matrix_csv,
        "figures": {
            "pair_types": fig_pair_types,
            "anchored": fig_anchored,
            "matrices": fig_matrices,
            "matrices_by_summary": fig_matrices_by_summary,
        },
    }


if __name__ == "__main__":
    run_day1_distinctiveness_analysis()
