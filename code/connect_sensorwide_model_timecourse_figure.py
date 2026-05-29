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
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

from connect_sensorwide_analysis import FIGURES_DIR, OUTPUT_DIR
from mvpa_stim_locked_cat_tg_window_structure_analysis import MVPA_CAT_TG_WINDOWS

PLOT_MODELS = [
    ("gradual", "gradual", "#3b6fb6"),
    ("two_stage_binary_best", "binary reconfig", "#b33c2e"),
    ("two_stage_hybrid_best", "mixed reconfig", "#6a4c9c"),
]

SPLIT_MODEL_COLORS = {
    "gradual": "#303030",
    "two_stage_binary_D1": "#f4a582",
    "two_stage_binary_D2": "#d6604d",
    "two_stage_binary_D3": "#b2182b",
    "two_stage_binary_D4": "#67001f",
    "two_stage_hybrid_D1": "#c2a5cf",
    "two_stage_hybrid_D2": "#9970ab",
    "two_stage_hybrid_D3": "#762a83",
    "two_stage_hybrid_D4": "#40004b",
}

ALL_MODEL_ROW_PCTS = [0.10, 0.30, 0.50, 0.70, 0.90, 1.00]
LEGACY_ACTIVE_PCT = 0.20


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


def filter_active_pct(d, active_pct):
    if "active_pct" not in d.columns:
        return d.copy()
    g = d[np.isclose(d["active_pct"].astype(float), float(active_pct))].copy()
    if g.empty:
        raise ValueError(f"Missing model-timecourse rows for active_pct={active_pct}")
    return g


def plot_group_model_timecourse(summary_df, best_df, figures_dir):
    summary_df = filter_active_pct(summary_df, LEGACY_ACTIVE_PCT)
    best_df = filter_active_pct(best_df, LEGACY_ACTIVE_PCT)
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
    best_df = filter_active_pct(best_df, LEGACY_ACTIVE_PCT)
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


def split_model_label(model, split_day):
    if model == "gradual":
        return "gradual"
    split_label = f"D{int(split_day)}"
    if model == "two_stage_binary":
        return f"binary {split_label}"
    if model == "two_stage_hybrid":
        return f"mixed {split_label}"
    raise ValueError(f"Unknown split-specific model: {model}")


def split_model_key(model, split_day):
    if model == "gradual":
        return "gradual"
    return f"{model}_D{int(split_day)}"


def model_distance(model, day_i, day_j, split_day=None):
    if model == "gradual":
        return float(abs(day_i - day_j) / 4.0)
    if model == "two_stage_binary":
        if split_day is None:
            raise ValueError("two_stage_binary requires split_day")
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late == j_late:
            return 0.0
        return 1.0
    if model == "two_stage_hybrid":
        if split_day is None:
            raise ValueError("two_stage_hybrid requires split_day")
        gradual = float(abs(day_i - day_j) / 4.0)
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late == j_late:
            return 0.5 * gradual
        return 0.5 + 0.5 * gradual
    raise ValueError(f"Unknown model: {model}")


def split_model_specs():
    rows = [{"model": "gradual", "split_day": np.nan}]
    for model in ["two_stage_binary", "two_stage_hybrid"]:
        for split_day in [1, 2, 3, 4]:
            rows.append({"model": model, "split_day": float(split_day)})
    return rows


def model_matrix(model, split_day):
    mat = np.full((5, 5), np.nan, dtype=float)
    split_arg = None
    if np.isfinite(split_day):
        split_arg = int(split_day)
    days = [1, 2, 3, 4, 5]
    for r, day_i in enumerate(days):
        for c, day_j in enumerate(days):
            mat[r, c] = model_distance(
                model,
                day_i,
                day_j,
                split_day=split_arg,
            )
    return mat


def draw_model_matrices(fig, grid):
    specs = split_model_specs()
    for col, spec in enumerate(specs):
        ax = fig.add_subplot(grid[0, col])
        mat = model_matrix(spec["model"], spec["split_day"])
        ax.imshow(mat, cmap="Greys", vmin=0, vmax=1)
        label = split_model_label(spec["model"], spec["split_day"])
        ax.set_title(label, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for r in range(mat.shape[0]):
            for c in range(mat.shape[1]):
                val = mat[r, c]
                text_color = "white"
                if val < 0.55:
                    text_color = "#303030"
                ax.text(
                    c,
                    r,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=5,
                    color=text_color,
                )


def plot_split_model_timecourse(summary_df, figures_dir):
    fig = plt.figure(figsize=(13.5, 13.0))
    grid = GridSpec(
        len(ALL_MODEL_ROW_PCTS) + 1,
        9,
        figure=fig,
        height_ratios=[1.0, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35],
        hspace=0.55,
        wspace=0.35,
    )
    draw_model_matrices(fig, grid)
    legend_handles = []
    legend_labels = []
    for row_i, active_pct in enumerate(ALL_MODEL_ROW_PCTS, start=1):
        ax = fig.add_subplot(grid[row_i, :])
        d_pct = filter_active_pct(summary_df, active_pct)
        for spec in split_model_specs():
            model = spec["model"]
            split_day = spec["split_day"]
            key = split_model_key(model, split_day)
            if key not in SPLIT_MODEL_COLORS:
                continue
            label = split_model_label(model, split_day)
            color = SPLIT_MODEL_COLORS[key]
            split_val = -1.0
            if np.isfinite(split_day):
                split_val = float(split_day)
            g = d_pct[
                (d_pct["model"] == model)
                & (d_pct["split_day"].fillna(-1.0) == split_val)
            ].sort_values("time_center_sec")
            if g.empty:
                raise ValueError(
                    "Missing split-specific model rows: "
                    f"active_pct={active_pct}, model={key}"
                )
            x = g["time_center_sec"].to_numpy(dtype=float)
            y = g["rho_mean"].to_numpy(dtype=float)
            lw = 2.4
            alpha = 1.0
            if model != "gradual":
                lw = 1.45
                alpha = 0.88
            handle = ax.plot(
                x,
                y,
                lw=lw,
                alpha=alpha,
                color=color,
                label=label,
            )[0]
            if row_i == 1:
                legend_handles.append(handle)
                legend_labels.append(label)

        for window_name, bounds in MVPA_CAT_TG_WINDOWS.items():
            if window_name == "early":
                color = "#bdbdbd"
            else:
                color = "#969696"
            ax.axvspan(bounds[0], bounds[1], color=color, alpha=0.12, linewidth=0)
        ax.axhline(0, color="#404040", lw=0.8)
        ax.set_ylabel(f"{active_pct * 100:.0f}%\nrho")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if row_i < len(ALL_MODEL_ROW_PCTS):
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("stim-locked time (s)")
    fig.legend(
        legend_handles,
        legend_labels,
        frameon=False,
        loc="lower center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.suptitle(
        "Connectivity Day-Geometry Model Correlations by Template and Edge Set",
        y=0.99,
    )
    fig.tight_layout(rect=[0.02, 0.04, 0.98, 0.97])
    fig_path = (
        figures_dir
        / "connect_sensorwide_model_timecourse_all_models_z_euclidean_edge_pcts_"
        "stim_broadband.png"
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
    all_models_fig = plot_split_model_timecourse(summary_df, figures_dir)
    split_fig = plot_best_split_timecourse(best_df, figures_dir)
    print(f"[connect model-timecourse] wrote {model_fig}", flush=True)
    print(f"[connect model-timecourse] wrote {all_models_fig}", flush=True)
    print(f"[connect model-timecourse] wrote {split_fig}", flush=True)
    return {
        "model_timecourse": model_fig,
        "all_models_timecourse": all_models_fig,
        "best_split_timecourse": split_fig,
    }


if __name__ == "__main__":
    save_fig_connect_sensorwide_model_timecourse()
