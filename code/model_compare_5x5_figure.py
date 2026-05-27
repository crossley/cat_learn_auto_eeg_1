#!/usr/bin/env python3
"""Plot one-stage, two-stage, and sensory-stable 5x5 model comparisons."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from model_compare_5x5_analysis import (
    DAYS,
    OUTPUT_DIR,
    design_rows_for_model,
    fit_ols_model,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing 5x5 model-comparison output: {path}. "
            "Run model_compare_5x5_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty 5x5 model-comparison output: {path}")
    return d


def compact_label(modality, measure, window):
    if modality == "mvpa":
        return f"MVPA {measure} {window}"
    if modality == "rsa":
        label = measure.replace("stim_", "")
        return f"RSA {label} {window}"
    if modality == "connectivity":
        return f"Connect {window}"
    return f"{modality} {measure} {window}"


def family_score_summary(scores_df):
    best_rows = []
    group_cols = ["modality", "measure", "window", "value_kind", "subject"]
    for key, g in scores_df.groupby(group_cols):
        modality, measure, window, value_kind, subject = key
        families = ["sensory_stable", "one_stage", "two_stage"]
        for family in families:
            d_family = g[g["model_family"] == family]
            if d_family.empty:
                continue
            best_idx = d_family["bic"].idxmin()
            row = d_family.loc[best_idx]
            best_rows.append(
                {
                    "modality": modality,
                    "measure": measure,
                    "window": window,
                    "value_kind": value_kind,
                    "subject": int(subject),
                    "model_family": family,
                    "split_day": row["split_day"],
                    "bic_support": float(row["bic_support"]),
                    "delta_bic": float(row["delta_bic"]),
                    "adj_r2": float(row["adj_r2"]),
                }
            )
    best_df = pd.DataFrame(best_rows)
    rows = []
    group_cols = ["modality", "measure", "window", "value_kind", "model_family"]
    for key, g in best_df.groupby(group_cols):
        modality, measure, window, value_kind, model_family = key
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
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "model_family": model_family,
                "mean": float(np.mean(vals)),
                "sem": sem,
                "n": int(len(vals)),
            }
        )
    return pd.DataFrame(rows), best_df


def ordered_conditions(summary_df):
    rows = []
    seen = set()
    modality_order = ["mvpa", "connectivity", "rsa"]
    for modality in modality_order:
        d_mod = summary_df[summary_df["modality"] == modality]
        for _, row in d_mod.iterrows():
            key = (row["modality"], row["measure"], row["window"], row["value_kind"])
            if key in seen:
                continue
            seen.add(key)
            rows.append(key)
    return rows


def modality_title(modality):
    if modality == "mvpa":
        return "MVPA"
    if modality == "connectivity":
        return "Connectivity"
    if modality == "rsa":
        return "RSA"
    return str(modality)


def save_score_figure(scores_df, figures_dir, modality):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_df, _best_df = family_score_summary(scores_df)
    summary_df = summary_df[summary_df["modality"] == modality].copy()
    if summary_df.empty:
        raise ValueError(f"No model-comparison score rows for modality={modality}")
    conditions = ordered_conditions(summary_df)
    if len(conditions) == 0:
        raise ValueError("No model-comparison score conditions to plot")
    families = ["sensory_stable", "one_stage", "two_stage"]
    colors = {
        "sensory_stable": "0.55",
        "one_stage": "#2a7f62",
        "two_stage": "#7b4aa0",
    }
    n_rows = len(conditions)
    fig_height = max(3.0, 0.45 * n_rows + 1.2)
    fig, ax = plt.subplots(figsize=(9.2, fig_height))
    y_positions = np.arange(n_rows, dtype=float)
    width = 0.23
    for family_i, family in enumerate(families):
        x_vals = []
        err_vals = []
        for condition in conditions:
            modality, measure, window, value_kind = condition
            d = summary_df[
                (summary_df["modality"] == modality)
                & (summary_df["measure"] == measure)
                & (summary_df["window"] == window)
                & (summary_df["value_kind"] == value_kind)
                & (summary_df["model_family"] == family)
            ]
            val = np.nan
            err = np.nan
            if not d.empty:
                val = float(d["mean"].iloc[0])
                err = float(d["sem"].iloc[0])
            x_vals.append(val)
            err_vals.append(err)
        offset = (family_i - 1) * width
        ax.barh(
            y_positions + offset,
            x_vals,
            height=width,
            color=colors[family],
            label=family.replace("_", " "),
            xerr=err_vals,
            error_kw={"linewidth": 0.8, "capsize": 2},
        )
    labels = []
    for condition in conditions:
        modality, measure, window, _value_kind = condition
        labels.append(compact_label(modality, measure, window))
    ax.axvline(0.0, color="0.25", linewidth=0.8)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("BIC support relative to subject's best model")
    ax.set_title(f"{modality_title(modality)} 5x5 Model Comparison")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig_path = figures_dir / f"model_compare_5x5_scores_{modality}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def group_matrix(group_df, condition):
    modality, measure, window, value_kind = condition
    d = group_df[
        (group_df["modality"] == modality)
        & (group_df["measure"] == measure)
        & (group_df["window"] == window)
        & (group_df["value_kind"] == value_kind)
    ]
    if d.empty:
        raise ValueError(f"Missing group matrix rows for {condition}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in d.iterrows():
        i = DAYS.index(int(row["train_day"]))
        j = DAYS.index(int(row["test_day"]))
        mat[i, j] = float(row["value_mean"])
    if value_kind == "distance":
        for i in range(len(DAYS)):
            mat[i, i] = 0.0
    return mat


def fit_group_template(mat, model_family, split_day=None):
    y_vals = []
    for train_day in DAYS:
        for test_day in DAYS:
            if train_day == test_day:
                continue
            i = DAYS.index(train_day)
            j = DAYS.index(test_day)
            y_vals.append(float(mat[i, j]))
    rows = design_rows_for_model(model_family, split_day)
    n_cols = 0
    if len(rows) > 0 and len(rows[0]["x_vals"]) > 0:
        n_cols = len(rows[0]["x_vals"])
    x = np.zeros((len(rows), n_cols), dtype=float)
    for row_i, row in enumerate(rows):
        for col_i, val in enumerate(row["x_vals"]):
            x[row_i, col_i] = float(val)
    fit = fit_ols_model(y_vals, x)
    beta = fit["beta"]
    pred_mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for row_i, row in enumerate(rows):
        pred = float(beta[0])
        for col_i, val in enumerate(row["x_vals"]):
            pred += float(beta[col_i + 1]) * float(val)
        train_day = int(row["train_day"])
        test_day = int(row["test_day"])
        i = DAYS.index(train_day)
        j = DAYS.index(test_day)
        pred_mat[i, j] = pred
    return pred_mat, fit


def best_two_stage_group_template(mat):
    best_mat = None
    best_fit = None
    best_split = None
    for split_day in [1, 2, 3, 4]:
        pred_mat, fit = fit_group_template(mat, "two_stage", split_day)
        if best_fit is None or float(fit["bic"]) < float(best_fit["bic"]):
            best_mat = pred_mat
            best_fit = fit
            best_split = split_day
    return best_mat, best_fit, best_split


def save_matrix_figure(group_df, scores_df, figures_dir, modality):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_df, _best_df = family_score_summary(scores_df)
    summary_df = summary_df[summary_df["modality"] == modality].copy()
    if summary_df.empty:
        raise ValueError(f"No model-comparison matrix rows for modality={modality}")
    conditions = ordered_conditions(summary_df)
    if len(conditions) == 0:
        raise ValueError("No model-comparison matrices to plot")
    n_rows = len(conditions)
    fig, axes = plt.subplots(
        n_rows,
        4,
        figsize=(10.4, max(3.0, 2.15 * n_rows)),
        squeeze=False,
    )
    labels = []
    for day in DAYS:
        labels.append(f"D{day}")
    for row_i, condition in enumerate(conditions):
        modality, measure, window, value_kind = condition
        observed = group_matrix(group_df, condition)
        sensory, _sens_fit = fit_group_template(observed, "sensory_stable")
        one_stage, _one_fit = fit_group_template(observed, "one_stage")
        two_stage, _two_fit, split_day = best_two_stage_group_template(observed)
        mats = [observed, sensory, one_stage, two_stage]
        titles = [
            "observed",
            "sensory stable",
            "one stage",
            f"two stage split D{split_day}",
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
                ax.set_ylabel(compact_label(modality, measure, window))
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
        cax = fig.add_axes(
            [
                0.92,
                0.10 + (n_rows - row_i - 1) * 0.80 / n_rows,
                0.010,
                0.62 / n_rows,
            ]
        )
        fig.colorbar(im, cax=cax)
    fig.suptitle(
        f"{modality_title(modality)} Observed Day Structure and Template Models"
    )
    fig.subplots_adjust(
        top=0.96,
        bottom=0.04,
        left=0.10,
        right=0.90,
        wspace=0.30,
        hspace=0.42,
    )
    fig_path = figures_dir / f"model_compare_5x5_matrices_{modality}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_model_compare_5x5(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    scores = require_csv(output_dir / "model_compare_5x5_scores.csv")
    group = require_csv(output_dir / "model_compare_5x5_group_matrices.csv")
    paths = {}
    for modality in ["mvpa", "connectivity", "rsa"]:
        score_key = f"scores_{modality}"
        matrix_key = f"matrices_{modality}"
        paths[score_key] = save_score_figure(scores, figures_dir, modality)
        paths[matrix_key] = save_matrix_figure(group, scores, figures_dir, modality)
        print(f"[5x5 model comparison] wrote {paths[score_key]}", flush=True)
        print(f"[5x5 model comparison] wrote {paths[matrix_key]}", flush=True)
    return paths


if __name__ == "__main__":
    save_fig_model_compare_5x5()
