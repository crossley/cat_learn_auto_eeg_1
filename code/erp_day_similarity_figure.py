#!/usr/bin/env python3
"""Plot ERP/GFP day-similarity 5x5 model comparisons."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_compare_5x5_analysis import DAYS, design_rows_for_model, fit_ols_model
from erp_day_similarity_analysis import OUTPUT_DIR

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ERP day-similarity output: {path}. "
            "Run erp_day_similarity_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty ERP day-similarity output: {path}")
    return d


def condition_label(modality, window):
    prefix = "ERP" if modality == "erp" else "GFP"
    labels = {
        "full_000_800": "0-800 ms",
        "early_060_180": "60-180 ms",
        "late_300_600": "300-600 ms",
    }
    return f"{prefix} {labels.get(window, window)}"


def score_summary(scores):
    best_rows = []
    group_cols = ["modality", "measure", "window", "value_kind", "subject"]
    for key, g in scores.groupby(group_cols):
        modality, measure, window, value_kind, subject = key
        for family in ["sensory_stable", "one_stage", "two_stage"]:
            d_family = g[g["model_family"] == family]
            if d_family.empty:
                continue
            row = d_family.loc[d_family["bic"].idxmin()]
            best_rows.append(
                {
                    "modality": modality,
                    "measure": measure,
                    "window": window,
                    "value_kind": value_kind,
                    "subject": int(subject),
                    "model_family": family,
                    "bic_support": float(row["bic_support"]),
                }
            )
    best = pd.DataFrame(best_rows)
    rows = []
    for key, g in best.groupby(["modality", "window", "model_family"]):
        modality, window, family = key
        vals = g["bic_support"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        sem = np.nan
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        rows.append(
            {
                "modality": modality,
                "window": window,
                "model_family": family,
                "mean": float(np.mean(vals)),
                "sem": sem,
            }
        )
    return pd.DataFrame(rows)


def ordered_conditions(group):
    rows = []
    for modality in ["erp", "erp_gfp"]:
        for window in ["full_000_800", "early_060_180", "late_300_600"]:
            d = group[(group["modality"] == modality) & (group["window"] == window)]
            if not d.empty:
                rows.append((modality, window))
    return rows


def save_score_figure(scores, group, figures_dir):
    summary = score_summary(scores)
    conditions = ordered_conditions(group)
    families = ["sensory_stable", "one_stage", "two_stage"]
    colors = {
        "sensory_stable": "0.55",
        "one_stage": "#2a7f62",
        "two_stage": "#7b4aa0",
    }
    fig, ax = plt.subplots(figsize=(8.8, max(3.0, 0.55 * len(conditions) + 1.2)))
    y_pos = np.arange(len(conditions), dtype=float)
    width = 0.23
    for family_i, family in enumerate(families):
        vals = []
        errs = []
        for modality, window in conditions:
            d = summary[
                (summary["modality"] == modality)
                & (summary["window"] == window)
                & (summary["model_family"] == family)
            ]
            vals.append(float(d["mean"].iloc[0]) if not d.empty else np.nan)
            errs.append(float(d["sem"].iloc[0]) if not d.empty else np.nan)
        ax.barh(
            y_pos + (family_i - 1) * width,
            vals,
            height=width,
            color=colors[family],
            label=family.replace("_", " "),
            xerr=errs,
            error_kw={"linewidth": 0.8, "capsize": 2},
        )
    ax.axvline(0.0, color="0.25", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([condition_label(*condition) for condition in conditions])
    ax.invert_yaxis()
    ax.set_xlabel("BIC support relative to subject's best model")
    ax.set_title("ERP/GFP 5x5 Model Comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = figures_dir / "erp_day_similarity_model_scores.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def group_matrix(group, modality, window):
    d = group[(group["modality"] == modality) & (group["window"] == window)]
    if d.empty:
        raise ValueError(f"Missing ERP group matrix rows for {modality}, {window}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in d.iterrows():
        i = DAYS.index(int(row["train_day"]))
        j = DAYS.index(int(row["test_day"]))
        mat[i, j] = float(row["value_mean"])
    return mat


def fit_template(mat, family, split_day=None):
    y = []
    for train_day in DAYS:
        for test_day in DAYS:
            if train_day == test_day:
                continue
            y.append(float(mat[DAYS.index(train_day), DAYS.index(test_day)]))
    rows = design_rows_for_model(family, split_day)
    n_cols = len(rows[0]["x_vals"]) if rows and rows[0]["x_vals"] else 0
    x = np.zeros((len(rows), n_cols), dtype=float)
    for row_i, row in enumerate(rows):
        for col_i, val in enumerate(row["x_vals"]):
            x[row_i, col_i] = float(val)
    fit = fit_ols_model(y, x)
    pred = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for row in rows:
        val = float(fit["beta"][0])
        for col_i, x_val in enumerate(row["x_vals"]):
            val += float(fit["beta"][col_i + 1]) * float(x_val)
        pred[DAYS.index(int(row["train_day"])), DAYS.index(int(row["test_day"]))] = val
    for day in DAYS:
        pred[DAYS.index(day), DAYS.index(day)] = 1.0
    return pred, fit


def best_two_stage(mat):
    best = None
    for split_day in [1, 2, 3, 4]:
        pred, fit = fit_template(mat, "two_stage", split_day)
        if best is None or float(fit["bic"]) < float(best[1]["bic"]):
            best = (pred, fit, split_day)
    return best


def save_matrix_figure(group, figures_dir):
    conditions = ordered_conditions(group)
    fig, axes = plt.subplots(
        len(conditions),
        4,
        figsize=(10.4, max(3.2, 2.05 * len(conditions))),
        squeeze=False,
    )
    day_labels = [f"D{day}" for day in DAYS]
    for row_i, (modality, window) in enumerate(conditions):
        observed = group_matrix(group, modality, window)
        sensory, _ = fit_template(observed, "sensory_stable")
        one_stage, _ = fit_template(observed, "one_stage")
        two_stage, _fit, split_day = best_two_stage(observed)
        mats = [observed, sensory, one_stage, two_stage]
        titles = ["observed", "stable", "one stage", f"two stage D{split_day}"]
        vals = np.asarray([val for mat in mats for val in mat.ravel() if np.isfinite(val)])
        vmin = float(np.min(vals))
        vmax = float(np.max(vals))
        if vmax <= vmin:
            vmax = vmin + 1e-6
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad(color="0.82")
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
                ax.set_ylabel(condition_label(modality, window))
            ax.set_xticks(range(len(DAYS)))
            ax.set_yticks(range(len(DAYS)))
            ax.set_xticklabels(day_labels)
            ax.set_yticklabels(day_labels)
            for r in range(len(DAYS)):
                for c in range(len(DAYS)):
                    val = mat[r, c]
                    if np.isfinite(val):
                        color = "white" if val < vmin + 0.65 * (vmax - vmin) else "black"
                        ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
        cax = fig.add_axes([0.92, 0.10 + (len(conditions) - row_i - 1) * 0.80 / len(conditions), 0.010, 0.62 / len(conditions)])
        fig.colorbar(im, cax=cax)
    fig.suptitle("ERP/GFP Observed Day Similarity and Template Models")
    fig.subplots_adjust(top=0.96, bottom=0.04, left=0.10, right=0.90, wspace=0.30, hspace=0.42)
    path = figures_dir / "erp_day_similarity_matrices.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_fig_erp_day_similarity(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    scores = require_csv(output_dir / "erp_day_similarity_model_scores.csv")
    group = require_csv(output_dir / "erp_day_similarity_group_matrices.csv")
    paths = {
        "scores": save_score_figure(scores, group, figures_dir),
        "matrices": save_matrix_figure(group, figures_dir),
    }
    for path in paths.values():
        print(f"[ERP day similarity] wrote {path}", flush=True)
    return paths


if __name__ == "__main__":
    save_fig_erp_day_similarity()
