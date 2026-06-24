#!/usr/bin/env python3
"""Figures for GRT decision-bound strategy and MVPA split links."""

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


def save_model_evidence_figure(summary_df, figures_dir):
    fig_path = figures_dir / "decision_bound_block_model_evidence.png"
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
    ax.set_title("Decision-Bound Model Evidence Over Blocks")
    ax.set_xticks(np.arange(1, 26, 2))
    ax.set_ylim(-0.03, 1.03)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_switch_raster(switch_df, weights_df, figures_dir):
    fig_path = figures_dir / "decision_bound_strategy_switch_raster.png"
    subjects = sorted(switch_df["subject"].dropna().astype(int).unique().tolist())
    subj_to_y = {subject: i for i, subject in enumerate(subjects)}
    glc = weights_df[weights_df["model"] == "glc"].copy()
    fig, ax = plt.subplots(figsize=(10.5, max(4.8, 0.28 * len(subjects) + 1.3)))
    for subject, g in glc.groupby("subject"):
        if int(subject) not in subj_to_y:
            continue
        y = subj_to_y[int(subject)]
        g = g.sort_values("block")
        sizes = 25.0 + 125.0 * np.clip(g["bic_weight"].to_numpy(dtype=float), 0.0, 1.0)
        ax.scatter(
            g["block"],
            np.full(len(g), y),
            s=sizes,
            c=g["glc_bic_advantage_vs_best_1d"],
            cmap="RdBu_r",
            vmin=-20,
            vmax=20,
            alpha=0.85,
            edgecolor="0.25",
            linewidth=0.25,
        )
    for _, row in switch_df.iterrows():
        subject = int(row["subject"])
        if subject not in subj_to_y or not np.isfinite(row["strategy_switch_block"]):
            continue
        ax.scatter(
            [row["strategy_switch_block"]],
            [subj_to_y[subject]],
            marker="|",
            s=260,
            c="black",
            linewidth=2.0,
        )
    for boundary in [5.5, 10.5, 15.5, 20.5]:
        ax.axvline(boundary, color="0.8", linewidth=0.8)
    ax.set_yticks(np.arange(len(subjects)))
    ax.set_yticklabels([str(s) for s in subjects])
    ax.set_xticks(np.arange(1, 26, 2))
    ax.set_xlabel("Accumulated block")
    ax.set_ylabel("Subject")
    ax.set_title("GRT Strategy Switches and GLC Evidence")
    cbar = fig.colorbar(ax.collections[0], ax=ax, pad=0.02)
    cbar.set_label("GLC BIC advantage vs best 1D")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_mvpa_link_figure(link_df, figures_dir):
    fig_path = figures_dir / "decision_bound_mvpa_switch_link.png"
    d = link_df.copy()
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
            axes[0].set_title(f"Subject Switch Match (r={corr:.2f})")
        else:
            axes[0].set_title("Subject Switch Match")
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
    axes[1].set_title("MVPA Split by Behavioral Switch Group")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_decision_bound_strategy_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weights = require_csv(output_dir / "decision_bound_block_model_weights.csv")
    summary = require_csv(output_dir / "decision_bound_block_model_summary.csv")
    switch = require_csv(output_dir / "decision_bound_strategy_switch_subject.csv")
    link = require_csv(output_dir / "decision_bound_mvpa_switch_link.csv")
    paths = {
        "model_evidence": save_model_evidence_figure(summary, figures_dir),
        "switch_raster": save_switch_raster(switch, weights, figures_dir),
        "mvpa_link": save_mvpa_link_figure(link, figures_dir),
    }
    for path in paths.values():
        print(f"[decision bound figure] wrote {path}", flush=True)
    return paths


if __name__ == "__main__":
    save_decision_bound_strategy_figures()
