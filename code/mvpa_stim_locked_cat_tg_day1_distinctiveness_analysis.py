#!/usr/bin/env python3
"""Day 1 distinctiveness analysis: how day-1-anchored cross-day TG compares to later-only pairs."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis_utils import model_term_summary
from mvpa_stim_locked_cat_tg_window_structure_analysis import (
    MVPA_STIM_LOCKED_CAT_TG_MATRIX_GLOB,
    MVPA_CAT_TG_WINDOWS,
    extract_tg_window_auc,
    summarize_tg_window_matrix,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_ols(formula, df, cluster_subject=True):
    model = smf.ols(formula, data=df).fit()
    if cluster_subject and "subject" in df.columns and df["subject"].nunique() > 1:
        return model.get_robustcov_results(cov_type="cluster", groups=df["subject"])
    return model


def add_day1_pair_labels(window_df):
    d = window_df.copy()
    d["includes_day1"] = (d["train_day"] == 1) | (d["test_day"] == 1)
    d["pair_group"] = np.where(d["includes_day1"], "day1_pair", "later_only")
    d["day1_pair_type"] = "later_only"
    d.loc[(d["train_day"] == 1) & (d["test_day"] != 1), "day1_pair_type"] = "day1_forward"
    d.loc[(d["train_day"] != 1) & (d["test_day"] == 1), "day1_pair_type"] = "day1_backward"
    d["other_day"] = np.nan
    d.loc[d["train_day"] == 1, "other_day"] = d.loc[d["train_day"] == 1, "test_day"]
    d.loc[d["test_day"] == 1, "other_day"] = d.loc[d["test_day"] == 1, "train_day"]
    return d


def fit_day1_distinctiveness(window_df):
    d = add_day1_pair_labels(window_df)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_auc", "window"]).copy()
    d["window"] = pd.Categorical(d["window"], categories=["early", "late"])
    d["pair_group"] = pd.Categorical(d["pair_group"], categories=["later_only", "day1_pair"])
    d["day1_pair_type"] = pd.Categorical(
        d["day1_pair_type"],
        categories=["later_only", "day1_forward", "day1_backward"],
    )

    model_group = _fit_ols("mean_auc ~ window * pair_group", d)
    group_terms = []
    for term in model_group.model.exog_names:
        if term == "Intercept":
            continue
        item = model_term_summary(model_group, term)
        item["model"] = "day1_pair_vs_later_only"
        group_terms.append(item)

    model_direction = _fit_ols("mean_auc ~ window * day1_pair_type", d)
    direction_terms = []
    for term in model_direction.model.exog_names:
        if term == "Intercept":
            continue
        item = model_term_summary(model_direction, term)
        item["model"] = "day1_forward_backward_vs_later_only"
        direction_terms.append(item)

    return pd.DataFrame(group_terms + direction_terms), model_group, model_direction


def extract_within_day_window_auc(
    within_csv: Path | str = PROJECT_DIR / "output" / "mvpa_stim_locked_cat_tg_within_day_subject_level.csv",
    windows: dict[str, tuple[float, float]] = MVPA_CAT_TG_WINDOWS,
    summary: str = "square_mean",
):
    within_csv = Path(within_csv)
    if not within_csv.exists():
        return pd.DataFrame(
            columns=["subject", "train_day", "test_day", "day_distance", "window", "mean_auc", "n_cells"]
        )
    d = pd.read_csv(within_csv, low_memory=False)
    for col in ["subject", "day", "train_time_sec", "test_time_sec", "auc"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["subject", "day", "train_time_sec", "test_time_sec", "auc"]).copy()
    rows = []
    for (subject, day), g_day in d.groupby(["subject", "day"]):
        pivot = g_day.pivot_table(
            index="train_time_sec", columns="test_time_sec", values="auc", aggfunc="mean"
        )
        train_axis = pivot.index.to_numpy(dtype=float)
        test_axis = pivot.columns.to_numpy(dtype=float)
        auc_mat = pivot.to_numpy(dtype=float)
        for window_name, (tmin, tmax) in windows.items():
            train_mask = (train_axis >= tmin) & (train_axis <= tmax)
            test_mask = (test_axis >= tmin) & (test_axis <= tmax)
            win = auc_mat[np.ix_(train_mask, test_mask)]
            mean_auc, n_cells = summarize_tg_window_matrix(win, summary=summary)
            rows.append(
                {
                    "subject": int(subject),
                    "train_day": int(day),
                    "test_day": int(day),
                    "day_distance": 0,
                    "window": window_name,
                    "window_tmin": tmin,
                    "window_tmax": tmax,
                    "mean_auc": mean_auc,
                    "n_cells": n_cells,
                    "summary": summary,
                    "matrix_file": "",
                    "day1_pair_type": "within_day",
                    "pair_group": "within_day",
                    "includes_day1": int(day) == 1,
                }
            )
    return pd.DataFrame(rows)


def run_mvpa_stim_locked_cat_tg_day1_distinctiveness(
    matrix_dir: Path | str = OUTPUT_DIR,
    matrix_glob: str = MVPA_STIM_LOCKED_CAT_TG_MATRIX_GLOB,
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_df = add_day1_pair_labels(
        extract_tg_window_auc(
            matrix_dir=matrix_dir,
            matrix_glob=matrix_glob,
            summary="square_mean",
        )
    )
    within_df = extract_within_day_window_auc()
    summary_matrix_rows = []
    summary_window_rows = []
    for summary_name in ["square_mean", "diagonal_mean", "top10_mean"]:
        d_cross = add_day1_pair_labels(
            extract_tg_window_auc(
                matrix_dir=matrix_dir,
                matrix_glob=matrix_glob,
                summary=summary_name,
            )
        )
        d_within = extract_within_day_window_auc(summary=summary_name)
        d_all = pd.concat(
            [d_cross.dropna(subset=["mean_auc"]), d_within.dropna(subset=["mean_auc"])],
            ignore_index=True,
        )
        summary_window_rows.append(d_all)
        summary_matrix_rows.append(
            d_all.groupby(["summary", "window", "train_day", "test_day"], as_index=False)
            .agg(
                auc_mean=("mean_auc", "mean"),
                auc_sem=("mean_auc", _sem),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["summary", "window", "train_day", "test_day"])
        )
    stats_df, _, _ = fit_day1_distinctiveness(window_df)
    group_summary = (
        window_df.dropna(subset=["mean_auc"])
        .groupby(["window", "day1_pair_type"], as_index=False)
        .agg(
            auc_mean=("mean_auc", "mean"),
            auc_sem=("mean_auc", _sem),
            n_subjects=("subject", "nunique"),
            n_rows=("mean_auc", "size"),
        )
        .sort_values(["window", "day1_pair_type"])
    )
    pair_matrix_source = pd.concat(
        [window_df.dropna(subset=["mean_auc"]), within_df.dropna(subset=["mean_auc"])],
        ignore_index=True,
    )
    pair_matrix = (
        pair_matrix_source.groupby(["window", "train_day", "test_day"], as_index=False)
        .agg(
            auc_mean=("mean_auc", "mean"),
            auc_sem=("mean_auc", _sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["window", "train_day", "test_day"])
    )

    window_csv = output_dir / "mvpa_stim_locked_cat_tg_day1_window_auc_subject_pairs.csv"
    stats_csv = output_dir / "mvpa_stim_locked_cat_tg_day1_distinctiveness_model_terms.csv"
    summary_csv = output_dir / "mvpa_stim_locked_cat_tg_day1_pair_type_summary.csv"
    matrix_csv = output_dir / "mvpa_stim_locked_cat_tg_day_pair_window_auc_matrix.csv"
    summary_window_csv = output_dir / "mvpa_stim_locked_cat_tg_day_pair_window_auc_subject_pairs_by_summary.csv"
    summary_matrix_csv = output_dir / "mvpa_stim_locked_cat_tg_day_pair_window_auc_matrix_by_summary.csv"

    window_df.to_csv(window_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    group_summary.to_csv(summary_csv, index=False)
    pair_matrix.to_csv(matrix_csv, index=False)
    pd.concat(summary_window_rows, ignore_index=True).to_csv(summary_window_csv, index=False)
    pd.concat(summary_matrix_rows, ignore_index=True).to_csv(summary_matrix_csv, index=False)
    return {
        "window_df": window_df,
        "stats_df": stats_df,
        "group_summary": group_summary,
        "pair_matrix": pair_matrix,
        "window_csv": window_csv,
        "stats_csv": stats_csv,
        "summary_csv": summary_csv,
        "matrix_csv": matrix_csv,
        "summary_window_csv": summary_window_csv,
        "summary_matrix_csv": summary_matrix_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_tg_day1_distinctiveness()
