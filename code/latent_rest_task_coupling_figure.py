#!/usr/bin/env python3
"""Figures for rest-to-task latent trajectory coupling."""

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
from latent_dynamics_utils import require_csv
from latent_rest_task_coupling_analysis import OUTCOMES, PREDICTORS

PREDICTOR_LABELS = {
    "posterior_alpha_power": "posterior alpha",
    "frontal_theta_power": "frontal theta",
    "global_rest_connectivity": "global rest conn",
    "visual_central_rest_connectivity": "vis-central rest conn",
    "visual_frontal_rest_connectivity": "vis-frontal rest conn",
}

OUTCOME_LABELS = {
    "erp_path_length": "ERP path length",
    "erp_mean_speed": "ERP speed",
    "erp_end_norm": "ERP end norm",
    "erp_latent_1_late_mean": "ERP late latent 1",
    "erp_latent_2_late_mean": "ERP late latent 2",
    "connect_path_length": "conn path length",
    "connect_mean_speed": "conn speed",
    "connect_end_norm": "conn end norm",
    "connect_latent_1_late_mean": "conn late latent 1",
    "connect_latent_2_late_mean": "conn late latent 2",
}


def matrix_for(d, correlation_type):
    g = d[d["correlation_type"] == correlation_type].copy()
    mat = np.full((len(PREDICTORS), len(OUTCOMES)), np.nan, dtype=float)
    for row in g.itertuples(index=False):
        if row.predictor in PREDICTORS and row.outcome in OUTCOMES:
            i = PREDICTORS.index(row.predictor)
            j = OUTCOMES.index(row.outcome)
            mat[i, j] = float(row.r)
    return mat


def save_coupling_heatmap(output_dir, figures_dir):
    d = require_csv(output_dir / "latent_rest_task_coupling_correlations.csv")
    panels = [
        ("raw", "Raw Session Correlations"),
        ("subject_centered", "Subject-Centered Correlations"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 4.8), constrained_layout=True)
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
    path = figures_dir / "latent_rest_task_coupling_correlations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent rest-task figure] wrote {path}", flush=True)


def subject_center(d, cols):
    out = d.copy()
    for col in cols:
        out[col] = out[col] - out.groupby("subject")[col].transform("mean")
    return out


def save_coupling_scatter(output_dir, figures_dir):
    session = require_csv(output_dir / "latent_rest_task_coupling_table.csv")
    corr = require_csv(output_dir / "latent_rest_task_coupling_correlations.csv")
    corr = corr[corr["correlation_type"] == "subject_centered"].copy()
    corr = corr[np.isfinite(corr["r"].to_numpy(float))].copy()
    corr["abs_r"] = np.abs(corr["r"].astype(float))
    corr = corr.sort_values("abs_r", ascending=False).head(6)
    centered = subject_center(session, PREDICTORS + OUTCOMES)

    fig, axes = plt.subplots(2, 3, figsize=(11.8, 6.5))
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
    fig.suptitle("Strongest Subject-Centered Rest-Latent Associations")
    fig.tight_layout()
    path = figures_dir / "latent_rest_task_coupling_top_scatter.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[latent rest-task figure] wrote {path}", flush=True)


def save_latent_rest_task_coupling_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    save_coupling_heatmap(output_dir, figures_dir)
    save_coupling_scatter(output_dir, figures_dir)


if __name__ == "__main__":
    save_latent_rest_task_coupling_figures()
