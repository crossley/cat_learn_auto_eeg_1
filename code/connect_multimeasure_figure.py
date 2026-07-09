#!/usr/bin/env python3
"""Figures for multi-measure sensor-ROI connectivity robustness analyses."""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from block_model_timecourse_figure import split_block_color
from connect_multimeasure_edge_timecourse_analysis import OUTPUT_PREFIX
from connect_multimeasure_utils import (
    CONTRAST_ONLY_MEASURES,
    FIGURES_DIR,
    MEASURE_LABELS,
    OUTPUT_DIR,
    PRIMARY_BAND,
    PRIMARY_DECISION_WINDOW,
    PRIMARY_LOCK_TYPE,
    PRIMARY_MEASURE,
    ZERO_LAG_REJECTING_MEASURES,
    sem,
)
from figure_style import DAYS, DAY_COLORS, setup_axis
from sensor_rois import ROI_LABELS


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing connectivity figure input: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty connectivity figure input: {path}")
    return d


def primary_roi_day_mean(output_dir, output_prefix):
    d = require_csv(output_dir / f"{output_prefix}_roi_timecourse_day_mean.csv")
    return d[
        (d["lock_type"] == PRIMARY_LOCK_TYPE)
        & (d["band"] == PRIMARY_BAND)
        & (d["measure"] == PRIMARY_MEASURE)
    ].copy()


def primary_block_summary(output_dir, output_prefix):
    d = require_csv(output_dir / f"{output_prefix}_block_model_timecourse_summary.csv")
    return d[
        (d["band"] == PRIMARY_BAND) & (d["measure"] == PRIMARY_MEASURE)
    ].copy()


def save_dwpli_roi_timecourse(output_dir, figures_dir, output_prefix, figure_prefix):
    d = primary_roi_day_mean(output_dir, output_prefix)
    if d.empty:
        raise ValueError("No primary dwPLI ROI day-mean rows found")
    roi_pairs = ["visual_frontal", "visual_central"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8), sharey=True)
    for ax, roi_pair in zip(axes, roi_pairs):
        for day in DAYS:
            g = d[(d["roi_pair"] == roi_pair) & (d["day"] == day)].sort_values(
                "lock_time"
            )
            if g.empty:
                continue
            x = g["lock_time"].to_numpy(float)
            y = g["conn_mean"].to_numpy(float)
            err = g["conn_sem"].to_numpy(float)
            ax.plot(x, y, color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
            good = np.isfinite(err)
            if np.any(good):
                ax.fill_between(
                    x[good],
                    y[good] - err[good],
                    y[good] + err[good],
                    color=DAY_COLORS[day],
                    alpha=0.13,
                    linewidth=0,
                )
        ax.axvspan(
            PRIMARY_DECISION_WINDOW[0],
            PRIMARY_DECISION_WINDOW[1],
            color="0.5",
            alpha=0.10,
            linewidth=0,
        )
        ax.axvline(0.0, color="0.25", linewidth=0.8)
        ax.set_xlabel("time from stimulus (s)")
        ax.set_title(ROI_LABELS[roi_pair].title())
        setup_axis(ax)
    axes[0].set_ylabel("mean dwPLI")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Primary Broadband dwPLI Connectivity by Day")
    fig.tight_layout()
    path = figures_dir / f"{figure_prefix}_dwpli_roi_timecourse.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[connect multimeasure figure] wrote {path}", flush=True)
    return path


def plot_block_model_lines(ax, d):
    continuous = d[d["model_label"] == "Continuous Restructuring"].sort_values(
        "time_sec"
    )
    if not continuous.empty:
        x = continuous["time_sec"].to_numpy(float)
        y = -continuous["delta_bic_baseline_mean"].to_numpy(float)
        err = continuous["delta_bic_baseline_sem"].to_numpy(float)
        ax.plot(x, y, color="#1f1f1f", linewidth=2.1, label="Continuous")
        good = np.isfinite(err)
        if np.any(good):
            ax.fill_between(
                x[good],
                y[good] - err[good],
                y[good] + err[good],
                color="#1f1f1f",
                alpha=0.10,
                linewidth=0,
            )
    split_rows = d[d["model"] == "discrete"].copy()
    split_rows = split_rows[np.isfinite(split_rows["split_block"].astype(float))]
    for split_block in sorted(split_rows["split_block"].astype(float).unique()):
        g = split_rows[
            np.isclose(split_rows["split_block"].astype(float), float(split_block))
        ].sort_values("time_sec")
        ax.plot(
            g["time_sec"].to_numpy(float),
            -g["delta_bic_baseline_mean"].to_numpy(float),
            color=split_block_color(split_block),
            linewidth=1.0,
            alpha=0.72,
            label=f"B{int(split_block)}",
        )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    ax.axvspan(
        PRIMARY_DECISION_WINDOW[0],
        PRIMARY_DECISION_WINDOW[1],
        color="0.5",
        alpha=0.10,
        linewidth=0,
    )
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("evidence above baseline")
    setup_axis(ax)


def save_dwpli_block_model(output_dir, figures_dir, output_prefix, figure_prefix):
    d = primary_block_summary(output_dir, output_prefix)
    if d.empty:
        raise ValueError("No primary dwPLI block-model summary rows found")
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    plot_block_model_lines(ax, d)
    ax.set_title("Primary Broadband dwPLI Block-Model Evidence")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Continuous and transition after block",
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        ncol=9,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=[0.0, 0.18, 1.0, 1.0])
    path = figures_dir / f"{figure_prefix}_dwpli_block_model_timecourse.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[connect multimeasure figure] wrote {path}", flush=True)
    return path


def decision_window_evidence(subject_df, measures):
    lo, hi = PRIMARY_DECISION_WINDOW
    d = subject_df[
        (subject_df["band"] == PRIMARY_BAND)
        & (subject_df["measure"].isin(measures))
        & (subject_df["model_label"] == "Continuous Restructuring")
        & (subject_df["time_sec"] >= lo)
        & (subject_df["time_sec"] <= hi)
    ].copy()
    if d.empty:
        raise ValueError("No decision-window block-model subject rows found")
    d["evidence"] = -d["delta_bic_baseline"].astype(float)
    rows = []
    for key, group in d.groupby(["measure", "subject"]):
        measure, subject = key
        rows.append(
            {
                "measure": measure,
                "subject": int(subject),
                "evidence": float(np.nanmean(group["evidence"])),
            }
        )
    subject_summary = pd.DataFrame(rows)
    rows = []
    for measure, group in subject_summary.groupby("measure"):
        vals = group["evidence"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "measure": measure,
                "mean": float(np.mean(vals)) if len(vals) else np.nan,
                "sem": sem(vals),
                "n_subjects": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def save_bar_figure(summary, measure_order, title, ylabel, path):
    colors = {
        "imcoh": "#4c78a8",
        "wpli": "#59a14f",
        "wpli2_debiased": "#111111",
        "pli": "#f28e2b",
        "coh": "#b07aa1",
        "plv": "#e15759",
    }
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    xs = np.arange(len(measure_order))
    means = []
    errs = []
    for measure in measure_order:
        g = summary[summary["measure"] == measure]
        means.append(float(g["mean"].iloc[0]) if not g.empty else np.nan)
        errs.append(float(g["sem"].iloc[0]) if not g.empty else np.nan)
    ax.bar(
        xs,
        means,
        yerr=errs,
        color=[colors[measure] for measure in measure_order],
        edgecolor="0.2",
        linewidth=0.6,
        capsize=3,
    )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([MEASURE_LABELS[m] for m in measure_order], rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    setup_axis(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[connect multimeasure figure] wrote {path}", flush=True)
    return path


def save_convergence_panel(output_dir, figures_dir, output_prefix, figure_prefix):
    d = require_csv(output_dir / f"{output_prefix}_block_model_timecourse_subject.csv")
    summary = decision_window_evidence(d, ZERO_LAG_REJECTING_MEASURES)
    path = figures_dir / f"{figure_prefix}_zero_lag_rejecting_convergence.png"
    return save_bar_figure(
        summary,
        ZERO_LAG_REJECTING_MEASURES,
        "Decision-Window Continuous Evidence Across Lagged Measures",
        "mean evidence above baseline",
        path,
    )


def save_lagged_vs_instantaneous(output_dir, figures_dir, output_prefix, figure_prefix):
    d = require_csv(output_dir / f"{output_prefix}_block_model_timecourse_subject.csv")
    measure_order = ["imcoh", "wpli2_debiased"] + CONTRAST_ONLY_MEASURES
    summary = decision_window_evidence(d, measure_order)
    path = figures_dir / f"{figure_prefix}_lagged_vs_instantaneous_contrast.png"
    return save_bar_figure(
        summary,
        measure_order,
        "Lagged Measures vs Zero-Lag-Inclusive Contrast",
        "mean evidence above baseline",
        path,
    )


def save_connect_multimeasure_figures(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
    output_prefix=OUTPUT_PREFIX,
    figure_prefix=OUTPUT_PREFIX,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dwpli_roi": save_dwpli_roi_timecourse(
            output_dir, figures_dir, output_prefix, figure_prefix
        ),
        "dwpli_block": save_dwpli_block_model(
            output_dir, figures_dir, output_prefix, figure_prefix
        ),
        "convergence": save_convergence_panel(
            output_dir, figures_dir, output_prefix, figure_prefix
        ),
        "lagged_contrast": save_lagged_vs_instantaneous(
            output_dir, figures_dir, output_prefix, figure_prefix
        ),
    }


if __name__ == "__main__":
    save_connect_multimeasure_figures()
