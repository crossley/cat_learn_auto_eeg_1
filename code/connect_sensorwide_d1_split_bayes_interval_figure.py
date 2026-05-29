#!/usr/bin/env python3
"""Plot Bayesian interval results for D1-split connectivity evidence."""

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

from connect_sensorwide_analysis import FIGURES_DIR, OUTPUT_DIR


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing D1-split Bayesian interval output: {path}. "
            "Run connect_sensorwide_d1_split_bayes_interval_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty D1-split Bayesian interval output: {path}")
    return d


def max_posterior_interval(interval_df, model):
    d = interval_df[interval_df["model"] == model].copy()
    if d.empty:
        raise ValueError(f"Missing interval posterior rows for model={model}")
    row = d.loc[d["posterior_interval_prob"].idxmax()]
    return row


def add_interval_span(ax, lb, ub, color, label):
    ax.axvspan(float(lb), float(ub), color=color, alpha=0.18, linewidth=0)
    ax.text(
        float((lb + ub) / 2.0),
        0.95,
        label,
        transform=ax.get_xaxis_transform(),
        ha="center",
        va="top",
        fontsize=9,
        color=color,
    )


def plot_contrast_with_intervals(contrast_df, interval_df, ax):
    x = contrast_df["time_center_sec"].to_numpy(dtype=float)
    y = contrast_df["contrast_mean"].to_numpy(dtype=float)
    sem = contrast_df["contrast_sem"].to_numpy(dtype=float)
    ax.plot(x, y, color="#303030", lw=2.0)
    ax.fill_between(x, y - sem, y + sem, color="#303030", alpha=0.18, linewidth=0)
    ax.axhline(0, color="#777777", lw=0.8)
    row = max_posterior_interval(interval_df, "early_late")
    add_interval_span(
        ax,
        row["lb_early"],
        row["ub_early"],
        "#3b6fb6",
        "early",
    )
    add_interval_span(
        ax,
        row["lb_late"],
        row["ub_late"],
        "#b33c2e",
        "late",
    )
    ax.set_ylabel("D1 advantage")
    ax.set_title("D1-Split Advantage Over Competing Models")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_model_probs(model_df, ax):
    labels = []
    vals = []
    for model in ["none", "early", "late", "early_late"]:
        g = model_df[model_df["model"] == model]
        if g.empty:
            raise ValueError(f"Missing posterior model row: {model}")
        labels.append(model.replace("_", "+"))
        vals.append(float(g["posterior_model_prob"].iloc[0]))
    ax.bar(np.arange(len(vals)), vals, color="#6b6b6b")
    ax.set_xticks(np.arange(len(vals)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("posterior prob.")
    ax.set_ylim(0, 1)
    ax.set_title("Interval Model Posterior")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def interval_midpoint_rows(interval_df, model, interval_name):
    d = interval_df[interval_df["model"] == model].copy()
    rows = []
    if interval_name == "early":
        lb_col = "lb_early"
        ub_col = "ub_early"
    elif interval_name == "late":
        lb_col = "lb_late"
        ub_col = "ub_late"
    else:
        raise ValueError(f"Unknown interval name: {interval_name}")
    for row in d.itertuples(index=False):
        lb = getattr(row, lb_col)
        ub = getattr(row, ub_col)
        weight = row.posterior_interval_prob
        if np.isfinite(lb) and np.isfinite(ub) and np.isfinite(weight):
            rows.append(
                {
                    "midpoint": float((lb + ub) / 2.0),
                    "weight": float(weight),
                }
            )
    return pd.DataFrame(rows)


def plot_midpoint_density(interval_df, ax):
    colors = {"early": "#3b6fb6", "late": "#b33c2e"}
    for interval_name in ["early", "late"]:
        d = interval_midpoint_rows(interval_df, "early_late", interval_name)
        if d.empty:
            raise ValueError(f"Missing midpoint rows for interval={interval_name}")
        grouped_rows = []
        for midpoint, g in d.groupby("midpoint"):
            grouped_rows.append(
                {
                    "midpoint": float(midpoint),
                    "weight": float(np.sum(g["weight"])),
                }
            )
        grouped = pd.DataFrame(grouped_rows).sort_values("midpoint")
        ax.plot(
            grouped["midpoint"].to_numpy(dtype=float),
            grouped["weight"].to_numpy(dtype=float),
            lw=2.0,
            color=colors[interval_name],
            label=interval_name,
        )
    ax.set_xlabel("stim-locked time (s)")
    ax.set_ylabel("posterior mass")
    ax.set_title("Posterior Interval Midpoints")
    ax.legend(frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_fig_connect_sensorwide_d1_split_bayes_interval(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    contrast_df = require_csv(
        output_dir / "connect_sensorwide_d1_split_contrast_summary.csv"
    )
    model_df = require_csv(output_dir / "connect_sensorwide_d1_split_bayes_models.csv")
    interval_df = require_csv(
        output_dir / "connect_sensorwide_d1_split_bayes_intervals.csv"
    )
    fig, axes = plt.subplots(3, 1, figsize=(8.5, 9.2))
    plot_contrast_with_intervals(contrast_df, interval_df, axes[0])
    plot_model_probs(model_df, axes[1])
    plot_midpoint_density(interval_df, axes[2])
    fig.tight_layout()
    fig_path = figures_dir / "connect_sensorwide_d1_split_bayes_interval.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[connect D1 Bayes] wrote {fig_path}", flush=True)
    return {"bayes_interval_figure": fig_path}


if __name__ == "__main__":
    save_fig_connect_sensorwide_d1_split_bayes_interval()
