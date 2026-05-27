#!/usr/bin/env python3
"""Plot fixed-template similarity results for 5x5 day matrices."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_compare_5x5_analysis import DAYS
from model_compare_5x5_template_analysis import OUTPUT_DIR, template_vector

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing template-similarity output: {path}. "
            "Run model_compare_5x5_template_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty template-similarity output: {path}")
    return d


def compact_label(measure, window):
    label = str(measure).replace("stim_", "")
    return f"{label} {window}"


def modality_title(modality):
    if modality == "mvpa":
        return "MVPA"
    if modality == "connectivity":
        return "Connectivity"
    if modality == "rsa":
        return "RSA"
    return str(modality)


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
        raise ValueError(f"No template summary rows for modality={modality}")
    conditions = ordered_conditions(d_mod, modality)
    templates = [
        "one_stage",
        "two_stage",
        "two_stage_best",
    ]
    template_labels = {
        "one_stage": "one stage",
        "two_stage": "two stage splits",
        "two_stage_best": "best two stage",
    }
    colors = {
        "one_stage": "#2a7f62",
        "two_stage": "#9370b8",
        "two_stage_best": "#5e3c99",
    }
    fig, axes = plt.subplots(
        len(conditions),
        1,
        figsize=(8.2, max(3.0, 2.3 * len(conditions))),
        squeeze=False,
    )
    for row_i, condition in enumerate(conditions):
        measure, window, value_kind = condition
        ax = axes[row_i, 0]
        x_vals = []
        x_errs = []
        labels = []
        bar_colors = []
        for template in templates:
            d_template = d_mod[
                (d_mod["measure"] == measure)
                & (d_mod["window"] == window)
                & (d_mod["value_kind"] == value_kind)
                & (d_mod["template"] == template)
            ]
            if template == "two_stage":
                for split_day in [1, 2, 3, 4]:
                    d_split = d_template[d_template["split_day"] == split_day]
                    if d_split.empty:
                        continue
                    x_vals.append(float(d_split["mean_similarity"].iloc[0]))
                    x_errs.append(float(d_split["sem_similarity"].iloc[0]))
                    labels.append(f"2-stage D{split_day}")
                    bar_colors.append(colors[template])
            else:
                if d_template.empty:
                    continue
                x_vals.append(float(d_template["mean_similarity"].iloc[0]))
                x_errs.append(float(d_template["sem_similarity"].iloc[0]))
                labels.append(template_labels[template])
                bar_colors.append(colors[template])
        x = np.arange(len(x_vals), dtype=float)
        ax.bar(
            x,
            x_vals,
            yerr=x_errs,
            color=bar_colors,
            error_kw={"linewidth": 0.8, "capsize": 2},
        )
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_ylabel("Template similarity")
        ax.set_title(compact_label(measure, window))
        ax.grid(axis="y", alpha=0.25)
        d_pair = pairwise_df[
            (pairwise_df["modality"] == modality)
            & (pairwise_df["measure"] == measure)
            & (pairwise_df["window"] == window)
            & (pairwise_df["value_kind"] == value_kind)
        ]
        if not d_pair.empty:
            diff = float(d_pair["mean_diff_one_minus_two_best"].iloc[0])
            p_val = float(d_pair["p_perm_two_sided"].iloc[0])
            ax.text(
                0.99,
                0.96,
                f"one - best two = {diff:.2f}, p={p_val:.3f}",
                ha="right",
                va="top",
                transform=ax.transAxes,
                fontsize=8,
            )
    fig.suptitle(f"{modality_title(modality)} Fixed-Template Similarity")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = figures_dir / f"model_compare_5x5_template_scores_{modality}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def pair_rows_all():
    rows = []
    for train_day in DAYS:
        for test_day in DAYS:
            if train_day == test_day:
                continue
            rows.append({"train_day": train_day, "test_day": test_day})
    return rows


def template_matrix(value_kind, template, split_day=None):
    pair_rows = pair_rows_all()
    vals = template_vector(
        pair_rows,
        value_kind,
        template,
        split_day=split_day,
    )
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for idx, row in enumerate(pair_rows):
        i = DAYS.index(int(row["train_day"]))
        j = DAYS.index(int(row["test_day"]))
        mat[i, j] = float(vals[idx])
    return mat


def save_template_prediction_figure(figures_dir):
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    rows = [
        ("similarity", "one_stage", None, "similarity one stage"),
        ("similarity", "two_stage", 2, "similarity two stage D2"),
        ("distance", "one_stage", None, "distance one stage"),
        ("distance", "two_stage", 2, "distance two stage D2"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.0), squeeze=False)
    cmap = plt.get_cmap("Greys").copy()
    cmap.set_bad(color="0.82")
    for row_i, row in enumerate(rows):
        value_kind, template, split_day, title = row
        mat = template_matrix(value_kind, template, split_day=split_day)
        ax = axes[int(row_i / 2), row_i % 2]
        im = ax.imshow(np.ma.masked_invalid(mat), origin="upper", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(DAYS)))
        ax.set_yticks(range(len(DAYS)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for r in range(len(DAYS)):
            for c in range(len(DAYS)):
                if np.isfinite(mat[r, c]):
                    ax.text(
                        c,
                        r,
                        f"{mat[r, c]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
    fig.suptitle("Fixed 5x5 Template Predictions")
    fig.subplots_adjust(
        top=0.90,
        bottom=0.08,
        left=0.08,
        right=0.88,
        wspace=0.28,
        hspace=0.36,
    )
    cax = fig.add_axes([0.90, 0.20, 0.015, 0.60])
    fig.colorbar(im, cax=cax, label="Template value")
    fig_path = figures_dir / "model_compare_5x5_template_predictions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_model_compare_5x5_template(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = require_csv(output_dir / "model_compare_5x5_template_summary.csv")
    pairwise = require_csv(output_dir / "model_compare_5x5_template_pairwise.csv")
    paths = {}
    for modality in ["mvpa", "connectivity", "rsa"]:
        key = f"scores_{modality}"
        paths[key] = save_modality_score_figure(
            summary,
            pairwise,
            figures_dir,
            modality,
        )
        print(f"[5x5 template] wrote {paths[key]}", flush=True)
    paths["predictions"] = save_template_prediction_figure(figures_dir)
    print(f"[5x5 template] wrote {paths['predictions']}", flush=True)
    return paths


if __name__ == "__main__":
    save_fig_model_compare_5x5_template()
