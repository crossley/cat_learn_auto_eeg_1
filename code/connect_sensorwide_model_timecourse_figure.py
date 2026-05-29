#!/usr/bin/env python3
"""Plot connectivity day-geometry model correlations across time."""

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

PLOT_MODELS = [
    ("gradual", "gradual", "#3b6fb6"),
    ("two_stage_binary_best", "binary reconfig", "#b33c2e"),
    ("two_stage_hybrid_best", "mixed reconfig", "#6a4c9c"),
]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing connectivity model-timecourse output: {path}. "
            "Run connect_sensorwide_model_timecourse_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty connectivity model-timecourse output: {path}")
    return d


def plot_group_model_timecourse(summary_df, best_df, figures_dir):
    rows = []
    d_gradual = summary_df[summary_df["model"] == "gradual"].copy()
    for row in d_gradual.itertuples(index=False):
        rows.append(
            {
                "time_center_sec": float(row.time_center_sec),
                "plot_model": "gradual",
                "label": "gradual",
                "color": "#3b6fb6",
                "rho_mean": float(row.rho_mean),
                "rho_sem": float(row.rho_sem),
                "n_subjects": int(row.n_subjects),
                "split_day": np.nan,
            }
        )
    for row in best_df.itertuples(index=False):
        label = ""
        color = ""
        plot_model = str(row.model)
        if plot_model == "two_stage_binary_best":
            label = "binary reconfig"
            color = "#b33c2e"
        elif plot_model == "two_stage_hybrid_best":
            label = "mixed reconfig"
            color = "#6a4c9c"
        else:
            continue
        rows.append(
            {
                "time_center_sec": float(row.time_center_sec),
                "plot_model": plot_model,
                "label": label,
                "color": color,
                "rho_mean": float(row.rho_mean),
                "rho_sem": float(row.rho_sem),
                "n_subjects": int(row.n_subjects),
                "split_day": row.split_day,
            }
        )
    plot_df = pd.DataFrame(rows)
    if plot_df.empty:
        raise ValueError("No model-timecourse rows available to plot")

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for model_name, label, color in PLOT_MODELS:
        g = plot_df[plot_df["plot_model"] == model_name].sort_values(
            "time_center_sec"
        )
        if g.empty:
            raise ValueError(f"Missing model-timecourse rows for {model_name}")
        x = g["time_center_sec"].to_numpy(dtype=float)
        y = g["rho_mean"].to_numpy(dtype=float)
        sem = g["rho_sem"].to_numpy(dtype=float)
        ax.plot(x, y, lw=2.0, color=color, label=label)
        ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)

    for window_name, bounds in MVPA_CAT_TG_WINDOWS.items():
        if window_name == "early":
            color = "#bdbdbd"
        else:
            color = "#969696"
        ax.axvspan(bounds[0], bounds[1], color=color, alpha=0.18, linewidth=0)
        ax.text(
            float(np.mean(bounds)),
            0.98,
            window_name,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            color="#404040",
        )

    ax.axhline(0, color="#404040", lw=0.8)
    ax.set_xlabel("stim-locked time (s)")
    ax.set_ylabel("model correlation (Spearman rho)")
    ax.set_title("Connectivity Day-Geometry Model Correlations")
    ax.legend(frameon=False, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_path = (
        figures_dir
        / "connect_sensorwide_model_timecourse_z_euclidean_top20_stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_best_split_timecourse(best_df, figures_dir):
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    colors = {
        "two_stage_binary_best": "#b33c2e",
        "two_stage_hybrid_best": "#6a4c9c",
    }
    labels = {
        "two_stage_binary_best": "binary reconfig",
        "two_stage_hybrid_best": "mixed reconfig",
    }
    for model in ["two_stage_binary_best", "two_stage_hybrid_best"]:
        g = best_df[best_df["model"] == model].sort_values("time_center_sec")
        if g.empty:
            raise ValueError(f"Missing best split rows for {model}")
        ax.step(
            g["time_center_sec"].to_numpy(dtype=float),
            g["split_day"].to_numpy(dtype=float),
            where="mid",
            lw=2.0,
            color=colors[model],
            label=labels[model],
        )
    ax.set_ylim(0.75, 4.25)
    ax.set_yticks([1, 2, 3, 4])
    ax.set_yticklabels(["D1", "D2", "D3", "D4"])
    ax.set_xlabel("stim-locked time (s)")
    ax.set_ylabel("best split after")
    ax.set_title("Best Shared Two-Stage Split Across Time")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig_path = (
        figures_dir
        / "connect_sensorwide_model_timecourse_best_split_top20_stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_connect_sensorwide_model_timecourse(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_df = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_summary.csv"
    )
    best_df = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_best_summary.csv"
    )
    model_fig = plot_group_model_timecourse(summary_df, best_df, figures_dir)
    split_fig = plot_best_split_timecourse(best_df, figures_dir)
    print(f"[connect model-timecourse] wrote {model_fig}", flush=True)
    print(f"[connect model-timecourse] wrote {split_fig}", flush=True)
    return {"model_timecourse": model_fig, "best_split_timecourse": split_fig}


if __name__ == "__main__":
    save_fig_connect_sensorwide_model_timecourse()
