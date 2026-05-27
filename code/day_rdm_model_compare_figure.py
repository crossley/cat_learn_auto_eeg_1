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
        "two_stage_binary": "#9370b8",
        "two_stage_binary_shared_best": "#5e3c99",
        "two_stage_hybrid": "#d18f3f",
        "two_stage_hybrid_shared_best": "#a45a00",
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
        for stage_model, prefix in [
            ("two_stage_binary", "binary"),
            ("two_stage_hybrid", "hybrid"),
        ]:
            d_two = d_condition[d_condition["model"] == stage_model]
            for split_day in [1, 2, 3, 4]:
                d_split = d_two[d_two["split_day"] == split_day]
                if d_split.empty:
                    continue
                labels.append(f"{prefix} D{split_day}")
                vals.append(float(d_split["mean_rho"].iloc[0]))
                errs.append(float(d_split["sem_rho"].iloc[0]))
                bar_colors.append(colors[stage_model])
            d_best = d_condition[d_condition["model"] == f"{stage_model}_shared_best"]
            if not d_best.empty:
                split_day = int(d_best["split_day"].iloc[0])
                labels.append(f"{prefix} best D{split_day}")
                vals.append(float(d_best["mean_rho"].iloc[0]))
                errs.append(float(d_best["sem_rho"].iloc[0]))
                bar_colors.append(colors[f"{stage_model}_shared_best"])
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
            text_lines = []
            for _, pair_row in d_pair.iterrows():
                family = str(pair_row["stage_family"]).replace("two_stage_", "")
                diff = float(pair_row["mean_diff_shared_stage_minus_gradual"])
                p_val = float(pair_row["p_perm_shared_stage_greater_gradual"])
                text_lines.append(f"{family} - gradual = {diff:.2f}, p={p_val:.3f}")
            ax.text(
                0.99,
                0.96,
                "\n".join(text_lines),
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
        10,
        figsize=(22.0, max(3.0, 2.35 * len(conditions))),
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
        mats = [
            observed,
            model_matrix("gradual"),
            model_matrix("two_stage_binary", split_day=1),
            model_matrix("two_stage_binary", split_day=2),
            model_matrix("two_stage_binary", split_day=3),
            model_matrix("two_stage_binary", split_day=4),
            model_matrix("two_stage_hybrid", split_day=1),
            model_matrix("two_stage_hybrid", split_day=2),
            model_matrix("two_stage_hybrid", split_day=3),
            model_matrix("two_stage_hybrid", split_day=4),
        ]
        titles = [
            "observed",
            "gradual model",
            "binary D1",
            "binary D2",
            "binary D3",
            "binary D4",
            "hybrid D1",
            "hybrid D2",
            "hybrid D3",
            "hybrid D4",
        ]
        for col_i, mat in enumerate(mats):
            ax = axes[row_i, col_i]
            vals = []
            for val in mat.ravel():
                if np.isfinite(val):
                    vals.append(float(val))
            vmin = float(np.min(vals))
            vmax = float(np.max(vals))
            if vmax <= vmin:
                vmax = vmin + 1e-6
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_bad(color="0.82")
            ax.imshow(
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
            ax.text(
                0.98,
                0.02,
                f"{vmin:.2f}-{vmax:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=6,
                color="white",
                bbox={
                    "boxstyle": "round,pad=0.12",
                    "facecolor": "black",
                    "alpha": 0.35,
                    "linewidth": 0,
                },
            )
    fig.suptitle(f"{modality_title(modality)} Day RDMs: Independent Panel Scales")
    fig.subplots_adjust(
        top=0.90,
        bottom=0.06,
        left=0.08,
        right=0.98,
        wspace=0.28,
        hspace=0.45,
    )
    fig_path = figures_dir / f"day_rdm_model_compare_matrices_{modality}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_model_correlation_figure(model_corr_df, figures_dir):
    labels = []
    for label in model_corr_df["model_i"]:
        if label not in labels:
            labels.append(label)
    mat = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for _, row in model_corr_df.iterrows():
        i = labels.index(row["model_i"])
        j = labels.index(row["model_j"])
        mat[i, j] = float(row["spearman_rho"])
    fig, ax = plt.subplots(figsize=(5.7, 5.0))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.82")
    im = ax.imshow(
        np.ma.masked_invalid(mat),
        origin="upper",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
    )
    display_labels = []
    for label in labels:
        display_labels.append(label.replace("_", " "))
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=35, ha="right")
    ax.set_yticklabels(display_labels)
    for r in range(len(labels)):
        for c in range(len(labels)):
            if np.isfinite(mat[r, c]):
                color = "white"
                if abs(mat[r, c]) < 0.55:
                    color = "black"
                ax.text(
                    c,
                    r,
                    f"{mat[r, c]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                )
    ax.set_title("Model RDM Correlations")
    fig.subplots_adjust(
        top=0.90,
        bottom=0.22,
        left=0.24,
        right=0.86,
    )
    cax = fig.add_axes([0.88, 0.25, 0.020, 0.56])
    fig.colorbar(im, cax=cax, label="Spearman rho")
    fig_path = figures_dir / "day_rdm_model_compare_model_correlations.png"
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
    model_corr = require_csv(
        output_dir / "day_rdm_model_compare_model_correlations.csv"
    )
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
    paths["model_correlations"] = save_model_correlation_figure(
        model_corr,
        figures_dir,
    )
    print(f"[day RDM] wrote {paths['model_correlations']}", flush=True)
    return paths


if __name__ == "__main__":
    save_fig_day_rdm_model_compare()
