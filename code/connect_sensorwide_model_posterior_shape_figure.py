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

ACTIVE_PCT = float(os.environ.get("ACTIVE_PCT", "0.20"))
MODEL_ORDER = ["none", "global", "one_window", "two_window", "three_window"]
MODEL_LABELS = {
    "none": "none",
    "global": "global",
    "one_window": "one window",
    "two_window": "two windows",
    "three_window": "three windows",
}
DIST_CMAP = "magma"


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


def active_pct_suffix():
    if np.isclose(ACTIVE_PCT, 0.20):
        return ""
    pct_int = int(round(ACTIVE_PCT * 100.0))
    return f"_top{pct_int}"


def shape_output_path(output_dir, stem):
    suffix = active_pct_suffix()
    return output_dir / f"{stem}{suffix}.csv"


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
    if model == "three_window":
        return (
            f"E {row['lb_early']:.3f}-{row['ub_early']:.3f}s\n"
            f"M {row['lb_middle']:.3f}-{row['ub_middle']:.3f}s\n"
            f"L {row['lb_late']:.3f}-{row['ub_late']:.3f}s"
        )
    return ""


def unique_sorted(vals):
    out = []
    for val in vals:
        if not np.isfinite(float(val)):
            continue
        if float(val) not in out:
            out.append(float(val))
    out.sort()
    return out


def interval_matrix(d, lb_col, ub_col):
    lbs = unique_sorted(d[lb_col].to_numpy(dtype=float))
    ubs = unique_sorted(d[ub_col].to_numpy(dtype=float))
    mat = np.full((len(lbs), len(ubs)), np.nan)
    for _idx, row in d.iterrows():
        lb = float(row[lb_col])
        ub = float(row[ub_col])
        lb_i = lbs.index(lb)
        ub_i = ubs.index(ub)
        mat[lb_i, ub_i] = float(row["posterior_candidate_prob"])
    return lbs, ubs, mat


def plot_interval_distribution(ax, d, lb_col, ub_col, title):
    if d.empty:
        ax.axis("off")
        return None
    lbs, ubs, mat = interval_matrix(d, lb_col, ub_col)
    image = ax.imshow(
        mat,
        origin="lower",
        aspect="auto",
        cmap=DIST_CMAP,
        vmin=0,
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("upper bound (s)")
    ax.set_ylabel("lower bound (s)")
    x_ticks = np.linspace(0, len(ubs) - 1, min(4, len(ubs)))
    y_ticks = np.linspace(0, len(lbs) - 1, min(4, len(lbs)))
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)
    x_labels = []
    for tick in x_ticks:
        x_labels.append(f"{ubs[int(round(tick))]:.2f}")
    y_labels = []
    for tick in y_ticks:
        y_labels.append(f"{lbs[int(round(tick))]:.2f}")
    ax.set_xticklabels(x_labels)
    ax.set_yticklabels(y_labels)
    return image


def inclusion_probability(d, lb_col, ub_col):
    lbs = unique_sorted(d[lb_col].to_numpy(dtype=float))
    ubs = unique_sorted(d[ub_col].to_numpy(dtype=float))
    times = []
    for val in lbs:
        if val not in times:
            times.append(val)
    for val in ubs:
        if val not in times:
            times.append(val)
    times.sort()
    probs = []
    for time_val in times:
        prob = 0.0
        for _idx, row in d.iterrows():
            lb = float(row[lb_col])
            ub = float(row[ub_col])
            if time_val >= lb and time_val <= ub:
                prob += float(row["posterior_candidate_prob"])
        probs.append(prob)
    return np.asarray(times), np.asarray(probs)


def plot_inclusion_probability(ax, d, lb_col, ub_col, title, color):
    if d.empty:
        ax.axis("off")
        return
    times, probs = inclusion_probability(d, lb_col, ub_col)
    ax.plot(times, probs, color=color, linewidth=1.8)
    ax.set_ylim(0, 1)
    ax.set_xlim(float(np.min(times)), float(np.max(times)))
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("P(included)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


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
                raise ValueError(
                    f"Missing shape model={model}, contrast={contrast}. "
                    "The posterior-shape summary is stale. Rerun "
                    "connect_sensorwide_model_posterior_shape_analysis.py "
                    "with the same ACTIVE_PCT before plotting."
                )
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


def plot_interval_distributions(candidate_df, figures_dir):
    contrasts = []
    for contrast in candidate_df["contrast"].drop_duplicates():
        contrasts.append(str(contrast))
    n_rows = len(contrasts)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(11.5, max(3.0, 2.4 * n_rows)),
        squeeze=False,
    )
    last_image = None
    for row_i, contrast in enumerate(contrasts):
        d = candidate_df[candidate_df["contrast"] == contrast]
        one = d[d["shape_model"] == "one_window"]
        two = d[d["shape_model"] == "two_window"]
        axes[row_i, 0].set_ylabel(compact_contrast_label(contrast))
        image = plot_interval_distribution(
            axes[row_i, 0],
            one,
            "lb_one",
            "ub_one",
            "one-window",
        )
        if image is not None:
            last_image = image
        image = plot_interval_distribution(
            axes[row_i, 1],
            two,
            "lb_early",
            "ub_early",
            "two-window early",
        )
        if image is not None:
            last_image = image
        image = plot_interval_distribution(
            axes[row_i, 2],
            two,
            "lb_late",
            "ub_late",
            "two-window late",
        )
        if image is not None:
            last_image = image
    if last_image is not None:
        fig.colorbar(last_image, ax=axes, shrink=0.55, label="posterior mass")
    fig.suptitle("Posterior Interval Distributions", y=0.995)
    fig.tight_layout(rect=[0, 0, 0.97, 0.985])
    fig_path = (
        figures_dir / "connect_sensorwide_model_posterior_shape_intervals.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_time_inclusion(candidate_df, figures_dir):
    contrasts = []
    for contrast in candidate_df["contrast"].drop_duplicates():
        contrasts.append(str(contrast))
    n_rows = len(contrasts)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(11.5, max(3.0, 2.0 * n_rows)),
        squeeze=False,
        sharey=True,
    )
    for row_i, contrast in enumerate(contrasts):
        d = candidate_df[candidate_df["contrast"] == contrast]
        one = d[d["shape_model"] == "one_window"]
        two = d[d["shape_model"] == "two_window"]
        axes[row_i, 0].set_ylabel(compact_contrast_label(contrast))
        plot_inclusion_probability(
            axes[row_i, 0],
            one,
            "lb_one",
            "ub_one",
            "one-window",
            "#4c78a8",
        )
        plot_inclusion_probability(
            axes[row_i, 1],
            two,
            "lb_early",
            "ub_early",
            "two-window early",
            "#f58518",
        )
        plot_inclusion_probability(
            axes[row_i, 2],
            two,
            "lb_late",
            "ub_late",
            "two-window late",
            "#54a24b",
        )
    fig.suptitle("Posterior Time-Inclusion Probability", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fig_path = (
        figures_dir / "connect_sensorwide_model_posterior_shape_inclusion.png"
    )
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
        shape_output_path(
            output_dir,
            "connect_sensorwide_model_posterior_shape_summary",
        )
    )
    candidate_df = require_csv(
        shape_output_path(
            output_dir,
            "connect_sensorwide_model_posterior_shape_candidates",
        )
    )
    fig_path = plot_shape_summary(summary_df, figures_dir)
    interval_path = plot_interval_distributions(candidate_df, figures_dir)
    inclusion_path = plot_time_inclusion(candidate_df, figures_dir)
    print(f"[connect posterior shape] wrote {fig_path}", flush=True)
    print(f"[connect posterior shape] wrote {interval_path}", flush=True)
    print(f"[connect posterior shape] wrote {inclusion_path}", flush=True)
    return {
        "posterior_shape": fig_path,
        "posterior_shape_intervals": interval_path,
        "posterior_shape_inclusion": inclusion_path,
    }


if __name__ == "__main__":
    save_fig_connect_sensorwide_model_posterior_shape()
