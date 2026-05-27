#!/usr/bin/env python3
"""Plot MVPA transfer-template model comparisons."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mvpa_transfer_model_compare_analysis import (
    DAYS,
    OUTPUT_DIR,
    template_value,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MVPA transfer model output: {path}. "
            "Run mvpa_transfer_model_compare_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty MVPA transfer model output: {path}")
    return d


def group_matrix(group_df, classifier, window):
    d = group_df[
        (group_df["classifier"] == classifier) & (group_df["window"] == window)
    ]
    if d.empty:
        raise ValueError(f"Missing MVPA group matrix for {classifier}, {window}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in d.iterrows():
        i = DAYS.index(int(row["train_day"]))
        j = DAYS.index(int(row["test_day"]))
        mat[i, j] = float(row["auc_mean"])
    return mat


def template_matrix(template, split_day=None):
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for i, train_day in enumerate(DAYS):
        for j, test_day in enumerate(DAYS):
            mat[i, j] = template_value(
                template,
                train_day,
                test_day,
                split_day=split_day,
            )
    return mat


def ordered_conditions(summary_df):
    rows = []
    seen = set()
    for _, row in summary_df.iterrows():
        key = (row["classifier"], row["window"], bool(row["include_diagonal"]))
        if key in seen:
            continue
        seen.add(key)
        rows.append(key)
    return rows


def save_score_figure(summary_df, pairwise_df, figures_dir):
    conditions = ordered_conditions(summary_df)
    colors = {
        "one_stage_bottleneck": "#2a7f62",
        "one_stage_closeness": "#4f9bdf",
        "two_stage_binary": "#9370b8",
        "two_stage_binary_shared_best": "#5e3c99",
        "two_stage_bottleneck": "#d18f3f",
        "two_stage_bottleneck_shared_best": "#a45a00",
    }
    fig, axes = plt.subplots(
        len(conditions),
        1,
        figsize=(9.8, max(3.0, 2.45 * len(conditions))),
        squeeze=False,
    )
    for row_i, condition in enumerate(conditions):
        classifier, window, include_diagonal = condition
        ax = axes[row_i, 0]
        d = summary_df[
            (summary_df["classifier"] == classifier)
            & (summary_df["window"] == window)
            & (summary_df["include_diagonal"] == include_diagonal)
        ]
        labels = []
        vals = []
        errs = []
        bar_colors = []
        for template in ["one_stage_bottleneck", "one_stage_closeness"]:
            d_template = d[d["template"] == template]
            if d_template.empty:
                continue
            labels.append(template.replace("one_stage_", "1-stage "))
            vals.append(float(d_template["mean_rho"].iloc[0]))
            errs.append(float(d_template["sem_rho"].iloc[0]))
            bar_colors.append(colors[template])
        for family, prefix in [
            ("two_stage_binary", "binary"),
            ("two_stage_bottleneck", "stage+bottle"),
        ]:
            d_family = d[d["template"] == family]
            for split_day in [1, 2, 3, 4]:
                d_split = d_family[d_family["split_day"] == split_day]
                if d_split.empty:
                    continue
                labels.append(f"{prefix} D{split_day}")
                vals.append(float(d_split["mean_rho"].iloc[0]))
                errs.append(float(d_split["sem_rho"].iloc[0]))
                bar_colors.append(colors[family])
            d_best = d[d["template"] == f"{family}_shared_best"]
            if not d_best.empty:
                split_day = int(d_best["split_day"].iloc[0])
                labels.append(f"{prefix} best D{split_day}")
                vals.append(float(d_best["mean_rho"].iloc[0]))
                errs.append(float(d_best["sem_rho"].iloc[0]))
                bar_colors.append(colors[f"{family}_shared_best"])
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
        diag_label = "with diagonal"
        if not include_diagonal:
            diag_label = "off diagonal"
        ax.set_title(f"{classifier} {window}, {diag_label}")
        ax.grid(axis="y", alpha=0.25)
        d_pair = pairwise_df[
            (pairwise_df["classifier"] == classifier)
            & (pairwise_df["window"] == window)
            & (pairwise_df["include_diagonal"] == include_diagonal)
        ]
        text_lines = []
        for _, row in d_pair.iterrows():
            family = str(row["stage_family"]).replace("two_stage_", "")
            one = str(row["one_stage_template"]).replace("one_stage_", "")
            diff = float(row["mean_diff_stage_minus_one_stage"])
            p_val = float(row["p_perm_stage_greater_one_stage"])
            text_lines.append(f"{family} - {one} = {diff:.2f}, p={p_val:.3f}")
        if len(text_lines) > 0:
            ax.text(
                0.99,
                0.96,
                "\n".join(text_lines),
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=7,
            )
    fig.suptitle("MVPA Transfer Matrix Template Comparison")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = figures_dir / "mvpa_transfer_model_compare_scores.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_matrix_figure(group_df, figures_dir):
    conditions = []
    seen = set()
    for _, row in group_df.iterrows():
        key = (row["classifier"], row["window"])
        if key in seen:
            continue
        seen.add(key)
        conditions.append(key)
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    titles = [
        "observed AUC",
        "1-stage bottleneck",
        "1-stage closeness",
        "binary D1",
        "binary D2",
        "binary D3",
        "binary D4",
        "stage+bottle D1",
        "stage+bottle D2",
        "stage+bottle D3",
        "stage+bottle D4",
    ]
    fig, axes = plt.subplots(
        len(conditions),
        len(titles),
        figsize=(24.0, max(3.0, 2.2 * len(conditions))),
        squeeze=False,
    )
    for row_i, condition in enumerate(conditions):
        classifier, window = condition
        mats = [
            group_matrix(group_df, classifier, window),
            template_matrix("one_stage_bottleneck"),
            template_matrix("one_stage_closeness"),
            template_matrix("two_stage_binary", split_day=1),
            template_matrix("two_stage_binary", split_day=2),
            template_matrix("two_stage_binary", split_day=3),
            template_matrix("two_stage_binary", split_day=4),
            template_matrix("two_stage_bottleneck", split_day=1),
            template_matrix("two_stage_bottleneck", split_day=2),
            template_matrix("two_stage_bottleneck", split_day=3),
            template_matrix("two_stage_bottleneck", split_day=4),
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
                ax.set_title(titles[col_i], fontsize=9)
            if col_i == 0:
                ax.set_ylabel(f"{classifier} {window}")
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
                            fontsize=6,
                            color=color,
                        )
    fig.suptitle("MVPA Transfer Matrices and Native Prediction Templates")
    fig.subplots_adjust(
        top=0.90,
        bottom=0.06,
        left=0.05,
        right=0.99,
        wspace=0.22,
        hspace=0.36,
    )
    fig_path = figures_dir / "mvpa_transfer_model_compare_matrices.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_model_correlation_figure(corr_df, figures_dir):
    labels = []
    for label in corr_df["model_i"]:
        if label not in labels:
            labels.append(label)
    mat = np.full((len(labels), len(labels)), np.nan, dtype=float)
    for _, row in corr_df.iterrows():
        i = labels.index(row["model_i"])
        j = labels.index(row["model_j"])
        mat[i, j] = float(row["spearman_rho"])
    display_labels = []
    for label in labels:
        display_labels.append(
            label.replace("one_stage_", "1-stage ")
            .replace("two_stage_", "2-stage ")
            .replace("_", " ")
        )
    fig, ax = plt.subplots(figsize=(8.0, 7.0))
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(color="0.82")
    im = ax.imshow(
        np.ma.masked_invalid(mat),
        origin="upper",
        cmap=cmap,
        vmin=-1.0,
        vmax=1.0,
    )
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(display_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(display_labels, fontsize=7)
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
                    fontsize=6,
                    color=color,
                )
    ax.set_title("MVPA Transfer Template Correlations")
    fig.subplots_adjust(
        top=0.92,
        bottom=0.28,
        left=0.28,
        right=0.86,
    )
    cax = fig.add_axes([0.88, 0.30, 0.018, 0.52])
    fig.colorbar(im, cax=cax, label="Spearman rho")
    fig_path = figures_dir / "mvpa_transfer_model_compare_model_correlations.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_mvpa_transfer_model_compare(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = require_csv(output_dir / "mvpa_transfer_model_compare_summary.csv")
    pairwise = require_csv(output_dir / "mvpa_transfer_model_compare_pairwise.csv")
    group = require_csv(output_dir / "mvpa_transfer_model_compare_group_matrices.csv")
    corr = require_csv(
        output_dir / "mvpa_transfer_model_compare_model_correlations.csv"
    )
    paths = {}
    paths["scores"] = save_score_figure(summary, pairwise, figures_dir)
    paths["matrices"] = save_matrix_figure(group, figures_dir)
    paths["model_correlations"] = save_model_correlation_figure(corr, figures_dir)
    for key, path in paths.items():
        print(f"[MVPA transfer models] wrote {key}: {path}", flush=True)
    return paths


if __name__ == "__main__":
    save_fig_mvpa_transfer_model_compare()
