#!/usr/bin/env python3
"""Plot sequence RNN decoding summaries from saved CSV outputs."""

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

from sequence_rnn_analysis import FIGURES_DIR, OUTPUT_DIR


def _plot_no_data(ax, title):
    ax.text(0.5, 0.5, "No finite rows", ha="center", va="center", transform=ax.transAxes)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def plot_model_feature_summary(group_df, figures_dir):
    fig_path = Path(figures_dir) / "sequence_rnn_auc_summary.png"
    d = group_df[
        (group_df["control"] == "intact")
        & (group_df["evaluation"] == "within_session_cv")
        & np.isclose(group_df["prefix_fraction"].astype(float), 1.0)
    ].copy()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    if d.empty:
        _plot_no_data(ax, "Sequence Decoding Summary")
    else:
        d = d.sort_values(["feature_kind", "model"])
        labels = [f"{r.feature_kind}\n{r.model}" for r in d.itertuples()]
        x = np.arange(len(d))
        y = d["auc_mean"].to_numpy(dtype=float)
        s = d["auc_sem"].fillna(0.0).to_numpy(dtype=float)
        ax.bar(x, y, yerr=s, color="#4c78a8", alpha=0.88, edgecolor="0.2", linewidth=0.6)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("ROC-AUC")
        ax.set_title("Within-session Sequence Decoding (A vs B)")
        ax.set_ylim(0.35, max(0.75, float(np.nanmax(y + s)) + 0.05))
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_early_exit(group_df, figures_dir):
    fig_path = Path(figures_dir) / "sequence_rnn_early_exit.png"
    d = group_df[
        (group_df["control"] == "intact")
        & (group_df["evaluation"] == "within_session_cv")
    ].copy()
    combos = d[["feature_kind", "model"]].drop_duplicates().to_records(index=False)
    n = max(1, len(combos))
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.4 * n_rows), squeeze=False, sharey=True)
    for ax in axes.ravel():
        ax.set_visible(False)
    if d.empty:
        ax = axes.ravel()[0]
        ax.set_visible(True)
        _plot_no_data(ax, "Early-exit Decoding")
    else:
        for ax, (feature_kind, model) in zip(axes.ravel(), combos):
            ax.set_visible(True)
            g = d[(d["feature_kind"] == feature_kind) & (d["model"] == model)]
            g = g.sort_values("prefix_fraction")
            x = g["prefix_fraction"].to_numpy(dtype=float)
            y = g["auc_mean"].to_numpy(dtype=float)
            s = g["auc_sem"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, color="#4c78a8", marker="o", linewidth=1.8)
            ax.fill_between(x, y - s, y + s, color="#4c78a8", alpha=0.18, linewidth=0)
            ax.axhline(0.5, color="k", linestyle="--", linewidth=0.9)
            ax.set_title(f"{feature_kind} / {model}", fontsize=10)
            ax.set_xlabel("Sequence prefix")
            ax.set_ylabel("ROC-AUC")
            ax.set_xlim(0.0, 1.02)
            ax.grid(alpha=0.25)
    fig.suptitle("Truncated-sequence Decoding")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_transfer_matrices(group_df, figures_dir):
    paths = []
    d = group_df[
        (group_df["control"] == "intact")
        & (group_df["evaluation"] == "cross_session_transfer")
        & np.isclose(group_df["prefix_fraction"].astype(float), 1.0)
    ].copy()
    if d.empty:
        fig_path = Path(figures_dir) / "sequence_rnn_transfer_matrices.png"
        fig, ax = plt.subplots(figsize=(5, 4))
        _plot_no_data(ax, "Cross-session Transfer")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return [fig_path]
    combos = d[["feature_kind", "model"]].drop_duplicates().to_records(index=False)
    days = sorted(set(d["train_day"].astype(int)).union(set(d["test_day"].astype(int))))
    n = len(combos)
    n_cols = min(3, n)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.0 * n_cols, 3.8 * n_rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_visible(False)
    im = None
    for ax, (feature_kind, model) in zip(axes.ravel(), combos):
        ax.set_visible(True)
        g = d[(d["feature_kind"] == feature_kind) & (d["model"] == model)]
        mat = np.full((len(days), len(days)), np.nan)
        for _, row in g.iterrows():
            i = days.index(int(row["train_day"]))
            j = days.index(int(row["test_day"]))
            mat[i, j] = float(row["auc_mean"])
        im = ax.imshow(np.ma.masked_invalid(mat), origin="lower", vmin=0.35, vmax=0.75, cmap="viridis")
        ax.set_xticks(np.arange(len(days)))
        ax.set_yticks(np.arange(len(days)))
        ax.set_xticklabels([f"D{d0}" for d0 in days])
        ax.set_yticklabels([f"D{d0}" for d0 in days])
        ax.set_xlabel("Test day")
        ax.set_ylabel("Train day")
        ax.set_title(f"{feature_kind} / {model}", fontsize=10)
    if im is not None:
        fig.colorbar(im, ax=[ax for ax in axes.ravel() if ax.get_visible()], shrink=0.75, label="ROC-AUC")
    fig.suptitle("Cross-session Sequence Transfer")
    fig.subplots_adjust(top=0.88, bottom=0.12, left=0.08, right=0.90, wspace=0.35, hspace=0.38)
    fig_path = Path(figures_dir) / "sequence_rnn_transfer_matrices.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(fig_path)
    return paths


def plot_control_delta(group_df, figures_dir):
    fig_path = Path(figures_dir) / "sequence_rnn_control_delta.png"
    d = group_df[
        (group_df["evaluation"] == "within_session_cv")
        & np.isclose(group_df["prefix_fraction"].astype(float), 1.0)
    ].copy()
    if d.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        _plot_no_data(ax, "Control Delta")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path
    pivot = d.pivot_table(
        index=["feature_kind", "model"],
        columns="control",
        values="auc_mean",
        aggfunc="mean",
    ).reset_index()
    if "intact" not in pivot or "time_shuffled" not in pivot:
        pivot["delta"] = np.nan
    else:
        pivot["delta"] = pivot["intact"] - pivot["time_shuffled"]
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    labels = [f"{r.feature_kind}\n{r.model}" for r in pivot.itertuples()]
    x = np.arange(len(pivot))
    ax.bar(x, pivot["delta"].to_numpy(dtype=float), color="#f58518", alpha=0.88, edgecolor="0.2", linewidth=0.6)
    ax.axhline(0.0, color="k", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("AUC intact - time shuffled")
    ax.set_title("Sequence-order Control")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_sequence_rnn(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    group_csv = output_dir / "sequence_rnn_group_summary.csv"
    if not group_csv.exists():
        raise FileNotFoundError(
            f"Missing sequence RNN output: {group_csv}. "
            "Run sequence_rnn_analysis.py first."
        )
    group_df = pd.read_csv(group_csv)
    if group_df.empty:
        raise ValueError(f"Empty sequence RNN group table: {group_csv}")
    paths = {
        "summary": plot_model_feature_summary(group_df, figures_dir),
        "early_exit": plot_early_exit(group_df, figures_dir),
        "control_delta": plot_control_delta(group_df, figures_dir),
        "transfer": plot_transfer_matrices(group_df, figures_dir),
    }
    print(f"[sequence RNN figure] wrote {paths}", flush=True)
    return {"figure_paths": paths}


if __name__ == "__main__":
    save_fig_sequence_rnn()
