#!/usr/bin/env python3
"""Figures for resting-state same-day predictor summaries."""

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

from figure_style import FIGURES_DIR, OUTPUT_DIR, setup_axis
from rest_session_predictors_analysis import OUTCOMES, PREDICTORS


PREDICTOR_LABELS = {
    "posterior_alpha_power": "posterior alpha",
    "frontal_theta_power": "frontal theta",
    "global_rest_connectivity": "global rest conn",
    "visual_central_rest_connectivity": "vis-central rest conn",
    "visual_frontal_rest_connectivity": "vis-frontal rest conn",
}

OUTCOME_LABELS = {
    "accuracy": "accuracy",
    "median_rt_correct": "median RT",
    "learning_slope": "learning slope",
    "early_accuracy": "early accuracy",
    "late_accuracy": "late accuracy",
    "mvpa_late_auc": "MVPA late AUC",
    "mvpa_peak_auc": "MVPA peak AUC",
    "task_connectivity_late_contrast": "task conn contrast",
    "erp_gfp_late": "ERP GFP late",
}


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing rest predictor output: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty rest predictor output: {path}")
    return d


def matrix_for(d, correlation_type):
    g = d[d["correlation_type"] == correlation_type].copy()
    mat = np.full((len(PREDICTORS), len(OUTCOMES)), np.nan, dtype=float)
    for row in g.itertuples(index=False):
        if row.predictor in PREDICTORS and row.outcome in OUTCOMES:
            i = PREDICTORS.index(row.predictor)
            j = OUTCOMES.index(row.outcome)
            mat[i, j] = float(row.r)
    return mat


def save_heatmap_figure(output_dir, figures_dir):
    d = require_csv(output_dir / "rest_session_predictor_correlations.csv")
    panels = [
        ("raw", "Raw Session Correlations"),
        ("subject_centered", "Subject-Centered Correlations"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.0, 4.8), constrained_layout=True)
    for ax, (corr_type, title) in zip(axes, panels):
        mat = matrix_for(d, corr_type)
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(title)
        ax.set_yticks(range(len(PREDICTORS)))
        ax.set_yticklabels([PREDICTOR_LABELS[x] for x in PREDICTORS])
        ax.set_xticks(range(len(OUTCOMES)))
        ax.set_xticklabels([OUTCOME_LABELS[x] for x in OUTCOMES], rotation=40, ha="right")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if abs(mat[i, j]) > 0.55 else "black",
                    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.85)
    cbar.set_label("Pearson r")
    path = figures_dir / "rest_session_predictor_correlations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rest predictors figure] wrote {path}", flush=True)


def subject_center(d, cols):
    out = d.copy()
    for col in cols:
        out[col] = out[col] - out.groupby("subject")[col].transform("mean")
    return out


def save_scatter_figure(output_dir, figures_dir):
    session = require_csv(output_dir / "rest_session_predictors_table.csv")
    corr = require_csv(output_dir / "rest_session_predictor_correlations.csv")
    corr = corr[corr["correlation_type"] == "subject_centered"].copy()
    corr = corr[np.isfinite(corr["r"].to_numpy(float))].copy()
    corr["abs_r"] = np.abs(corr["r"].astype(float))
    corr = corr.sort_values("abs_r", ascending=False).head(6)
    centered = subject_center(session, PREDICTORS + OUTCOMES)

    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4))
    axes = axes.ravel()
    for ax, row in zip(axes, corr.itertuples(index=False)):
        x = centered[row.predictor].to_numpy(float)
        y = centered[row.outcome].to_numpy(float)
        good = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[good], y[good], s=22, color="#4c78a8", alpha=0.75)
        if int(np.sum(good)) >= 3:
            coef = np.polyfit(x[good], y[good], 1)
            xx = np.linspace(float(np.min(x[good])), float(np.max(x[good])), 50)
            ax.plot(xx, coef[0] * xx + coef[1], color="black", linewidth=1.0)
        ax.axhline(0, color="0.75", linewidth=0.8)
        ax.axvline(0, color="0.75", linewidth=0.8)
        ax.set_title(f"r = {row.r:.2f}")
        ax.set_xlabel(PREDICTOR_LABELS[row.predictor])
        ax.set_ylabel(OUTCOME_LABELS[row.outcome])
        setup_axis(ax)
    for ax in axes[len(corr) :]:
        ax.axis("off")
    fig.suptitle("Strongest Subject-Centered Rest-Task Associations")
    fig.tight_layout()
    path = figures_dir / "rest_session_predictor_top_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[rest predictors figure] wrote {path}", flush=True)


def save_rest_session_predictor_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_heatmap_figure(output_dir, figures_dir)
    save_scatter_figure(output_dir, figures_dir)


if __name__ == "__main__":
    save_rest_session_predictor_figures()
