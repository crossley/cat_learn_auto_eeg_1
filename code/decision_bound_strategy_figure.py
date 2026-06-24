#!/usr/bin/env python3
"""Figures for GRT decision-bound strategy and MVPA split links."""

from __future__ import annotations

import os
import argparse
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

from decision_bound_strategy_analysis import OUTPUT_DIR, PROJECT_DIR

FIGURES_DIR = PROJECT_DIR / "figures"


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing decision-bound output: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty decision-bound output: {path}")
    return d


def _prefix(output_label=None):
    return "decision_bound" if not output_label else f"decision_bound_{output_label}"


def save_model_evidence_figure(summary_df, figures_dir, output_label=None):
    fig_path = figures_dir / f"{_prefix(output_label)}_block_model_evidence.png"
    d = summary_df.copy()
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    colors = {"unix": "#4c78a8", "uniy": "#72b7b2", "glc": "#f58518"}
    labels = {"unix": "1D x rule", "uniy": "1D y rule", "glc": "2D GLC"}
    for model in ["unix", "uniy", "glc"]:
        g = d[d["model"] == model].sort_values("block")
        if g.empty:
            continue
        x = g["block"].to_numpy(dtype=float)
        y = g["bic_weight_mean"].to_numpy(dtype=float)
        err = g["bic_weight_sem"].fillna(0.0).to_numpy(dtype=float)
        ax.plot(x, y, marker="o", linewidth=1.8, color=colors[model], label=labels[model])
        ax.fill_between(x, y - err, y + err, color=colors[model], alpha=0.15, linewidth=0)
    for boundary in [5.5, 10.5, 15.5, 20.5]:
        ax.axvline(boundary, color="0.8", linewidth=0.8)
    ax.set_xlabel("Accumulated block")
    ax.set_ylabel("BIC model weight")
    title = "Decision-Bound Model Evidence Over Blocks"
    if "bound_mode" in d.columns and d["bound_mode"].notna().any():
        mode = str(d["bound_mode"].dropna().iloc[0]).replace("_", " ")
        title = f"{title}: {mode}"
    ax.set_title(title)
    ax.set_xticks(np.arange(1, 26, 2))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def subject_order_by_switch(switch_df):
    d = switch_df.copy()
    d["sort_switch"] = d["strategy_switch_block"].fillna(999).astype(float)
    return d.sort_values(["sort_switch", "subject"])["subject"].astype(int).tolist()


def save_glc_weight_subject_trajectories(weights_df, switch_df, figures_dir, output_label=None):
    fig_path = figures_dir / f"{_prefix(output_label)}_glc_weight_subject_trajectories.png"
    glc = weights_df[weights_df["model"] == "glc"].copy()
    subjects = subject_order_by_switch(switch_df)
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for subject in subjects:
        g = glc[glc["subject"] == subject].sort_values("block")
        if g.empty:
            continue
        ax.plot(
            g["block"],
            g["bic_weight"],
            color="0.62",
            linewidth=0.9,
            alpha=0.55,
        )
    summary = (
        glc.groupby("block", as_index=False)
        .agg(
            bic_weight_mean=("bic_weight", "mean"),
            bic_weight_sem=(
                "bic_weight",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            ),
        )
        .sort_values("block")
    )
    x = summary["block"].to_numpy(dtype=float)
    y = summary["bic_weight_mean"].to_numpy(dtype=float)
    err = summary["bic_weight_sem"].fillna(0.0).to_numpy(dtype=float)
    ax.plot(x, y, color="#f58518", linewidth=2.3, marker="o", label="Group mean")
    ax.fill_between(x, y - err, y + err, color="#f58518", alpha=0.18, linewidth=0)
    for boundary in [5.5, 10.5, 15.5, 20.5]:
        ax.axvline(boundary, color="0.82", linewidth=0.8)
    ax.set_xlabel("Accumulated block")
    ax.set_ylabel("GLC BIC model weight")
    ax.set_title("Subject-Level GLC Evidence Trajectories")
    ax.set_xticks(np.arange(1, 26, 2))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_best_model_heatmap(weights_df, switch_df, figures_dir, output_label=None):
    fig_path = figures_dir / f"{_prefix(output_label)}_best_model_heatmap.png"
    subjects = subject_order_by_switch(switch_df)
    models = ["unix", "uniy", "glc"]
    model_to_val = {model: idx for idx, model in enumerate(models)}
    mat = np.full((len(subjects), 25), np.nan)
    best = weights_df[weights_df["is_best_model"].astype(bool)].copy()
    for row_i, subject in enumerate(subjects):
        g = best[best["subject"] == subject]
        for _, row in g.iterrows():
            block = int(row["block"])
            if 1 <= block <= 25:
                mat[row_i, block - 1] = model_to_val[str(row["model"])]
    cmap = plt.matplotlib.colors.ListedColormap(["#4c78a8", "#72b7b2", "#f58518"])
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.28 * len(subjects) + 1.3)))
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=2.5)
    for _, row in switch_df.iterrows():
        subject = int(row["subject"])
        if subject not in subjects or not np.isfinite(row["strategy_switch_block"]):
            continue
        y = subjects.index(subject)
        ax.scatter(row["strategy_switch_block"] - 1, y, marker="|", s=260, c="black", linewidth=2.0)
    for boundary in [4.5, 9.5, 14.5, 19.5]:
        ax.axvline(boundary, color="white", linewidth=1.0)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels([str(s) for s in subjects])
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([str(x) for x in range(1, 26, 2)])
    ax.set_xlabel("Accumulated block")
    ax.set_ylabel("Subject")
    ax.set_title("Best-Fitting Decision-Bound Model by Subject and Block")
    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2], pad=0.02)
    cbar.ax.set_yticklabels(["1D x", "1D y", "GLC"])
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_glc_advantage_heatmap(weights_df, switch_df, figures_dir, output_label=None):
    fig_path = figures_dir / f"{_prefix(output_label)}_glc_advantage_heatmap.png"
    subjects = subject_order_by_switch(switch_df)
    glc = weights_df[weights_df["model"] == "glc"].copy()
    mat = np.full((len(subjects), 25), np.nan)
    for row_i, subject in enumerate(subjects):
        g = glc[glc["subject"] == subject]
        for _, row in g.iterrows():
            block = int(row["block"])
            if 1 <= block <= 25:
                mat[row_i, block - 1] = float(row["glc_bic_advantage_vs_best_1d"])
    finite = mat[np.isfinite(mat)]
    vmax = 20.0 if len(finite) == 0 else max(5.0, min(60.0, float(np.nanpercentile(np.abs(finite), 90))))
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.28 * len(subjects) + 1.3)))
    im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", interpolation="nearest", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for _, row in switch_df.iterrows():
        subject = int(row["subject"])
        if subject not in subjects or not np.isfinite(row["strategy_switch_block"]):
            continue
        y = subjects.index(subject)
        ax.scatter(row["strategy_switch_block"] - 1, y, marker="|", s=260, c="black", linewidth=2.0)
    for boundary in [4.5, 9.5, 14.5, 19.5]:
        ax.axvline(boundary, color="0.75", linewidth=1.0)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels([str(s) for s in subjects])
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([str(x) for x in range(1, 26, 2)])
    ax.set_xlabel("Accumulated block")
    ax.set_ylabel("Subject")
    ax.set_title("GLC BIC Advantage Over Best 1D Model")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("BIC advantage: GLC - best 1D")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_mvpa_link_figure(link_df, figures_dir, output_label=None):
    fig_path = figures_dir / f"{_prefix(output_label)}_mvpa_switch_link.png"
    d = link_df.copy()
    window_label = ""
    if {"mvpa_tmin", "mvpa_tmax"}.issubset(d.columns) and d["mvpa_tmin"].notna().any():
        window_label = f" ({float(d['mvpa_tmin'].dropna().iloc[0]):.1f}-{float(d['mvpa_tmax'].dropna().iloc[0]):.1f} s)"
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    finite = d[np.isfinite(d["strategy_switch_block"]) & np.isfinite(d["mvpa_best_split_block"])].copy()
    if finite.empty:
        axes[0].text(0.5, 0.5, "No finite switch rows", ha="center", va="center", transform=axes[0].transAxes)
    else:
        axes[0].scatter(
            finite["strategy_switch_block"],
            finite["mvpa_best_split_block"],
            s=58,
            color="#4c78a8",
            alpha=0.88,
            edgecolor="0.2",
            linewidth=0.5,
        )
        for _, row in finite.iterrows():
            axes[0].text(row["strategy_switch_block"] + 0.15, row["mvpa_best_split_block"], str(int(row["subject"])), fontsize=7)
        lo = 1
        hi = 25
        axes[0].plot([lo, hi], [lo, hi], color="0.4", linestyle="--", linewidth=1.0)
        if len(finite) > 2:
            corr = np.corrcoef(finite["strategy_switch_block"], finite["mvpa_best_split_block"])[0, 1]
            axes[0].set_title(f"Subject Switch Match{window_label} (r={corr:.2f})")
        else:
            axes[0].set_title(f"Subject Switch Match{window_label}")
    axes[0].set_xlabel("Behavioral GLC switch block")
    axes[0].set_ylabel("Best MVPA discrete split block")
    axes[0].set_xlim(0.5, 25.5)
    axes[0].set_ylim(0.5, 25.5)
    axes[0].grid(alpha=0.25)

    groups = ["day1_or_early", "later", "no_switch"]
    vals = [d[d["switch_group"] == group]["mvpa_best_split_block"].dropna().to_numpy(dtype=float) for group in groups]
    axes[1].boxplot(vals, tick_labels=["early", "later", "none"], showfliers=False)
    for i, arr in enumerate(vals, start=1):
        if len(arr):
            jitter = np.linspace(-0.06, 0.06, len(arr)) if len(arr) > 1 else np.array([0.0])
            axes[1].scatter(np.full(len(arr), i) + jitter, arr, color="#f58518", alpha=0.85, edgecolor="0.2", linewidth=0.4)
    axes[1].set_ylabel("Best MVPA discrete split block")
    axes[1].set_xlabel("Behavioral switch group")
    axes[1].set_title(f"MVPA Split by Behavioral Switch Group{window_label}")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_decision_bound_strategy_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR, output_label=None):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = _prefix(output_label)
    weights = require_csv(output_dir / f"{prefix}_block_model_weights.csv")
    summary = require_csv(output_dir / f"{prefix}_block_model_summary.csv")
    switch = require_csv(output_dir / f"{prefix}_strategy_switch_subject.csv")
    link = require_csv(output_dir / f"{prefix}_mvpa_switch_link.csv")
    paths = {
        "model_evidence": save_model_evidence_figure(summary, figures_dir, output_label=output_label),
        "glc_weight_subject_trajectories": save_glc_weight_subject_trajectories(weights, switch, figures_dir, output_label=output_label),
        "best_model_heatmap": save_best_model_heatmap(weights, switch, figures_dir, output_label=output_label),
        "glc_advantage_heatmap": save_glc_advantage_heatmap(weights, switch, figures_dir, output_label=output_label),
        "mvpa_link": save_mvpa_link_figure(link, figures_dir, output_label=output_label),
    }
    for path in paths.values():
        print(f"[decision bound figure] wrote {path}", flush=True)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--output-label", default=None)
    args = parser.parse_args()
    save_decision_bound_strategy_figures(
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        output_label=args.output_label,
    )
