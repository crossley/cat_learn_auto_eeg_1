#!/usr/bin/env python3
"""Plot day-RDM model-comparison results."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from day_rdm_model_compare_analysis import DAYS, OUTPUT_DIR, model_distance

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing day-RDM model-comparison output: {path}. "
            "Run day_rdm_model_compare_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty day-RDM model-comparison output: {path}")
    return d


def modality_title(modality):
    if modality == "mvpa":
        return "MVPA"
    if modality == "connectivity":
        return "Connectivity"
    if modality == "rsa":
        return "RSA"
    return str(modality)


def compact_label(measure, window):
    measure_label = str(measure).replace("stim_", "")
    return f"{measure_label} {window}"


def ordered_conditions(df, modality):
    rows = []
    seen = set()
    d_mod = df[df["modality"] == modality]
    for _, row in d_mod.iterrows():
        key = (row["measure"], row["window"], row["value_kind"])
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return rows


def save_modality_score_figure(summary_df, pairwise_df, figures_dir, modality):
    d_mod = summary_df[summary_df["modality"] == modality].copy()
    if d_mod.empty:
        raise ValueError(f"No day-RDM summary rows for modality={modality}")
    conditions = ordered_conditions(d_mod, modality)
    fig, axes = plt.subplots(
        len(conditions),
        1,
        figsize=(8.2, max(3.0, 2.3 * len(conditions))),
        squeeze=False,
    )
    colors = {
        "gradual": "#2a7f62",
        "two_stage": "#9370b8",
        "two_stage_best": "#5e3c99",
    }
    for row_i, condition in enumerate(conditions):
        measure, window, value_kind = condition
        ax = axes[row_i, 0]
        labels = []
        vals = []
        errs = []
        bar_colors = []
        d_condition = d_mod[
            (d_mod["measure"] == measure)
            & (d_mod["window"] == window)
            & (d_mod["value_kind"] == value_kind)
        ]
        d_grad = d_condition[d_condition["model"] == "gradual"]
        if not d_grad.empty:
            labels.append("gradual")
            vals.append(float(d_grad["mean_rho"].iloc[0]))
            errs.append(float(d_grad["sem_rho"].iloc[0]))
            bar_colors.append(colors["gradual"])
        d_two = d_condition[d_condition["model"] == "two_stage"]
        for split_day in [1, 2, 3, 4]:
            d_split = d_two[d_two["split_day"] == split_day]
            if d_split.empty:
                continue
            labels.append(f"2-stage D{split_day}")
            vals.append(float(d_split["mean_rho"].iloc[0]))
            errs.append(float(d_split["sem_rho"].iloc[0]))
            bar_colors.append(colors["two_stage"])
        d_best = d_condition[d_condition["model"] == "two_stage_best"]
        if not d_best.empty:
            labels.append("best 2-stage")
            vals.append(float(d_best["mean_rho"].iloc[0]))
            errs.append(float(d_best["sem_rho"].iloc[0]))
            bar_colors.append(colors["two_stage_best"])
        x = np.arange(len(vals), dtype=float)
        ax.bar(
            x,
            vals,
            yerr=errs,
            color=bar_colors,
            error_kw={"linewidth": 0.8, "capsize": 2},
        )
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Spearman rho")
        ax.set_title(compact_label(measure, window))
        ax.grid(axis="y", alpha=0.25)
        d_pair = pairwise_df[
            (pairwise_df["modality"] == modality)
            & (pairwise_df["measure"] == measure)
            & (pairwise_df["window"] == window)
            & (pairwise_df["value_kind"] == value_kind)
        ]
        if not d_pair.empty:
            diff = float(d_pair["mean_diff_gradual_minus_two_best"].iloc[0])
            p_val = float(d_pair["p_perm_two_sided"].iloc[0])
            ax.text(
                0.99,
                0.96,
                f"gradual - best two = {diff:.2f}, p={p_val:.3f}",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
            )
    fig.suptitle(f"{modality_title(modality)} Day-RDM Model Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = figures_dir / f"day_rdm_model_compare_scores_{modality}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def rdm_matrix_from_group(d_group):
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for i in range(len(DAYS)):
        mat[i, i] = 0.0
    for _, row in d_group.iterrows():
        day_i = int(row["day_i"])
        day_j = int(row["day_j"])
        i = DAYS.index(day_i)
        j = DAYS.index(day_j)
        val = float(row["distance_mean"])
        mat[i, j] = val
        mat[j, i] = val
    return mat


def model_matrix(model, split_day=None):
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            if day_i == day_j:
                mat[i, j] = 0.0
            else:
                mat[i, j] = model_distance(
                    model,
                    day_i,
                    day_j,
                    split_day=split_day,
                )
    return mat


def save_modality_matrix_figure(group_df, summary_df, figures_dir, modality):
    d_mod = summary_df[summary_df["modality"] == modality].copy()
    if d_mod.empty:
        raise ValueError(f"No day-RDM matrix rows for modality={modality}")
    conditions = ordered_conditions(d_mod, modality)
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    fig, axes = plt.subplots(
        len(conditions),
        3,
        figsize=(8.3, max(3.0, 2.4 * len(conditions))),
        squeeze=False,
    )
    for row_i, condition in enumerate(conditions):
        measure, window, value_kind = condition
        d_group = group_df[
            (group_df["modality"] == modality)
            & (group_df["measure"] == measure)
            & (group_df["window"] == window)
            & (group_df["value_kind"] == value_kind)
        ]
        if d_group.empty:
            raise ValueError(f"Missing group day RDM for {modality}, {condition}")
        observed = rdm_matrix_from_group(d_group)
        d_best = d_mod[
            (d_mod["measure"] == measure)
            & (d_mod["window"] == window)
            & (d_mod["value_kind"] == value_kind)
            & (d_mod["model"] == "two_stage_best")
        ]
        split_day = 2
        if not d_best.empty and np.isfinite(float(d_best["split_day"].iloc[0])):
            split_day = int(d_best["split_day"].iloc[0])
        mats = [
            observed,
            model_matrix("gradual"),
            model_matrix("two_stage", split_day=split_day),
        ]
        titles = [
            "observed",
            "gradual model",
            f"best two-stage D{split_day}",
        ]
        vals = []
        for mat in mats:
            for val in mat.ravel():
                if np.isfinite(val):
                    vals.append(float(val))
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="0.82")
        im = None
        for col_i, mat in enumerate(mats):
            ax = axes[row_i, col_i]
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                origin="upper",
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
            )
            if row_i == 0:
                ax.set_title(titles[col_i])
            if col_i == 0:
                ax.set_ylabel(compact_label(measure, window))
            ax.set_xticks(range(len(DAYS)))
            ax.set_yticks(range(len(DAYS)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            for r in range(len(DAYS)):
                for c in range(len(DAYS)):
                    val = mat[r, c]
                    if np.isfinite(val):
                        color = "white"
                        if val > vmin + 0.65 * (vmax - vmin):
                            color = "black"
                        ax.text(
                            c,
                            r,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=color,
                        )
        cax = fig.add_axes(
            [
                0.91,
                0.12 + (len(conditions) - row_i - 1) * 0.78 / len(conditions),
                0.012,
                0.56 / len(conditions),
            ]
        )
        fig.colorbar(im, cax=cax)
    fig.suptitle(f"{modality_title(modality)} Day RDMs")
    fig.subplots_adjust(
        top=0.92,
        bottom=0.06,
        left=0.10,
        right=0.89,
        wspace=0.32,
        hspace=0.45,
    )
    fig_path = figures_dir / f"day_rdm_model_compare_matrices_{modality}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_day_rdm_model_compare(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = require_csv(output_dir / "day_rdm_model_compare_summary.csv")
    pairwise = require_csv(output_dir / "day_rdm_model_compare_pairwise.csv")
    group = require_csv(output_dir / "day_rdm_model_compare_group_rdms.csv")
    paths = {}
    for modality in ["mvpa", "connectivity", "rsa"]:
        score_key = f"scores_{modality}"
        matrix_key = f"matrices_{modality}"
        paths[score_key] = save_modality_score_figure(
            summary,
            pairwise,
            figures_dir,
            modality,
        )
        paths[matrix_key] = save_modality_matrix_figure(
            group,
            summary,
            figures_dir,
            modality,
        )
        print(f"[day RDM] wrote {paths[score_key]}", flush=True)
        print(f"[day RDM] wrote {paths[matrix_key]}", flush=True)
    return paths


if __name__ == "__main__":
    save_fig_day_rdm_model_compare()
