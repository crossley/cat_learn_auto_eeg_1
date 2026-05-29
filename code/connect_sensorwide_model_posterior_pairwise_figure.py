#!/usr/bin/env python3
"""Plot Bayesian pairwise model comparisons for connectivity timecourses."""

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
from mvpa_stim_locked_cat_tg_window_structure_analysis import MVPA_CAT_TG_WINDOWS

CONTRAST_LABELS = {
    "mixed_D1_minus_gradual": "mixed D1 - gradual",
    "mixed_D1_minus_mixed_D2": "mixed D1 - mixed D2",
    "binary_D1_minus_binary_D2": "binary D1 - binary D2",
    "mixed_D1_minus_binary_D1": "mixed D1 - binary D1",
}


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing posterior pairwise output: {path}. "
            "Run connect_sensorwide_model_posterior_pairwise_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty posterior pairwise output: {path}")
    return d


def add_window_spans(ax):
    for window, bounds in MVPA_CAT_TG_WINDOWS.items():
        color = "#c7c7c7"
        if window == "late":
            color = "#9e9e9e"
        ax.axvspan(bounds[0], bounds[1], color=color, alpha=0.16, linewidth=0)


def plot_pairwise_timecourse(time_df, interval_df, figures_dir):
    contrasts = []
    for contrast in CONTRAST_LABELS.keys():
        if contrast in time_df["contrast"].unique():
            contrasts.append(contrast)
    if len(contrasts) == 0:
        raise ValueError("No requested posterior pairwise contrasts are available")

    fig, axes = plt.subplots(
        len(contrasts),
        2,
        figsize=(11.0, 2.6 * len(contrasts)),
        sharex=True,
    )
    if len(contrasts) == 1:
        axes = np.asarray([axes])
    for row_i, contrast in enumerate(contrasts):
        d = time_df[time_df["contrast"] == contrast].sort_values("time_center_sec")
        if d.empty:
            raise ValueError(f"Missing time rows for contrast={contrast}")
        x = d["time_center_sec"].to_numpy(dtype=float)
        y = d["mean_diff"].to_numpy(dtype=float)
        lo = d["ci_lower"].to_numpy(dtype=float)
        hi = d["ci_upper"].to_numpy(dtype=float)
        p = d["p_gt0"].to_numpy(dtype=float)

        ax = axes[row_i, 0]
        add_window_spans(ax)
        ax.plot(x, y, color="#303030", lw=1.8)
        ax.fill_between(x, lo, hi, color="#303030", alpha=0.18, linewidth=0)
        ax.axhline(0, color="#777777", lw=0.8)
        ax.set_ylabel("rho diff")
        ax.set_title(CONTRAST_LABELS[contrast])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax = axes[row_i, 1]
        add_window_spans(ax)
        ax.plot(x, p, color="#3b6fb6", lw=1.8)
        ax.axhline(0.95, color="#777777", lw=0.8, ls="--")
        ax.axhline(0.50, color="#bdbdbd", lw=0.8)
        ax.set_ylim(0, 1)
        ax.set_ylabel("P(diff > 0)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        text_lines = []
        g_int = interval_df[interval_df["contrast"] == contrast]
        for window in ["early", "late"]:
            row = g_int[g_int["window"] == window]
            if row.empty:
                continue
            val = float(row["p_gt0"].iloc[0])
            diff = float(row["mean_diff"].iloc[0])
            text_lines.append(f"{window}: p={val:.2f}, d={diff:.3f}")
        ax.text(
            0.98,
            0.04,
            "\n".join(text_lines),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#303030",
        )

    axes[-1, 0].set_xlabel("stim-locked time (s)")
    axes[-1, 1].set_xlabel("stim-locked time (s)")
    fig.suptitle("Bayesian Pairwise Connectivity Model Comparisons", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig_path = figures_dir / "connect_sensorwide_model_posterior_pairwise.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_connect_sensorwide_model_posterior_pairwise(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    time_df = require_csv(
        output_dir / "connect_sensorwide_model_posterior_pairwise_time.csv"
    )
    interval_df = require_csv(
        output_dir / "connect_sensorwide_model_posterior_pairwise_intervals.csv"
    )
    fig_path = plot_pairwise_timecourse(time_df, interval_df, figures_dir)
    print(f"[connect posterior pairwise] wrote {fig_path}", flush=True)
    return {"posterior_pairwise": fig_path}


if __name__ == "__main__":
    save_fig_connect_sensorwide_model_posterior_pairwise()
