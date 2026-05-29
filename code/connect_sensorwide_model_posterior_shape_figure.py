#!/usr/bin/env python3
"""Plot Bayesian temporal-shape comparisons for connectivity model preferences."""

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

MODEL_ORDER = ["none", "global", "one_window", "two_window"]
MODEL_LABELS = {
    "none": "none",
    "global": "global",
    "one_window": "one window",
    "two_window": "two windows",
}


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing posterior shape output: {path}. "
            "Run connect_sensorwide_model_posterior_shape_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty posterior shape output: {path}")
    return d


def compact_contrast_label(contrast):
    label = str(contrast)
    label = label.replace("two_stage_hybrid", "mixed")
    label = label.replace("two_stage_binary", "binary")
    label = label.replace("_minus_", " - ")
    label = label.replace("_", " ")
    return label


def interval_text(row):
    model = str(row["shape_model"])
    if model == "one_window":
        return f"{row['lb_one']:.3f}-{row['ub_one']:.3f}s"
    if model == "two_window":
        return (
            f"E {row['lb_early']:.3f}-{row['ub_early']:.3f}s\n"
            f"L {row['lb_late']:.3f}-{row['ub_late']:.3f}s"
        )
    return ""


def plot_shape_summary(summary_df, figures_dir):
    contrasts = []
    for contrast in summary_df["contrast"].drop_duplicates():
        contrasts.append(str(contrast))
    n_rows = len(contrasts)
    fig, axes = plt.subplots(
        n_rows,
        1,
        figsize=(8.8, max(3.0, 1.7 * n_rows)),
        squeeze=False,
    )
    for row_i, contrast in enumerate(contrasts):
        ax = axes[row_i, 0]
        d = summary_df[summary_df["contrast"] == contrast]
        vals = []
        labels = []
        text_by_model = {}
        for model in MODEL_ORDER:
            g = d[d["shape_model"] == model]
            if g.empty:
                raise ValueError(f"Missing shape model={model}, contrast={contrast}")
            row = g.iloc[0]
            vals.append(float(row["posterior_model_prob"]))
            labels.append(MODEL_LABELS[model])
            text_by_model[model] = interval_text(row)
        x = np.arange(len(vals))
        ax.bar(x, vals, color="#6b6b6b")
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("posterior")
        ax.set_title(compact_contrast_label(contrast), fontsize=10)
        for idx, model in enumerate(MODEL_ORDER):
            text = text_by_model[model]
            if text:
                ax.text(
                    idx,
                    min(vals[idx] + 0.05, 0.92),
                    text,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#303030",
                )
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle("Bayesian Temporal-Shape Model Comparison", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig_path = figures_dir / "connect_sensorwide_model_posterior_shape.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_connect_sensorwide_model_posterior_shape(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_df = require_csv(
        output_dir / "connect_sensorwide_model_posterior_shape_summary.csv"
    )
    fig_path = plot_shape_summary(summary_df, figures_dir)
    print(f"[connect posterior shape] wrote {fig_path}", flush=True)
    return {"posterior_shape": fig_path}


if __name__ == "__main__":
    save_fig_connect_sensorwide_model_posterior_shape()
