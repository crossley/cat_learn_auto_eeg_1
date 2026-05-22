#!/usr/bin/env python3
"""Day-structure analyses for stimulus-locked TG window summaries."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from scipy.stats import ttest_1samp

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

INPUT_CSV = OUTPUT_DIR / "mvpa_stim_locked_cat_tg_day_pair_window_auc_subject_pairs_by_summary.csv"
DAYS = [1, 2, 3, 4, 5]
SUMMARIES = ["square_mean", "diagonal_mean", "top10_mean"]
WINDOWS = ["early", "late"]


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def pearson_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x * y) / denom)


def candidate_value(model, day_low, day_high):
    day_distance = abs(day_high - day_low)
    if model == "day_distance_gradient":
        return float(-day_distance)
    if model == "day1_isolated_later_only":
        return float((day_low > 1) and (day_high > 1))
    if model == "adjacent_clusters_23_45":
        return float((day_low == 2 and day_high == 3) or (day_low == 4 and day_high == 5))
    if model == "stage_blocks_1_23_45":
        stage_map = {1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
        return float(-abs(stage_map[day_low] - stage_map[day_high]))
    if model == "late_cluster_45":
        return float(day_low == 4 and day_high == 5)
    raise ValueError(f"Unknown candidate model: {model}")


def candidate_models():
    models = []
    models.append("day_distance_gradient")
    models.append("day1_isolated_later_only")
    models.append("adjacent_clusters_23_45")
    models.append("stage_blocks_1_23_45")
    models.append("late_cluster_45")
    return models


def load_input(input_csv):
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing TG window summary input: {input_csv}. "
            "Run mvpa_stim_locked_cat_tg_day1_distinctiveness_analysis.py first."
        )
    d = pd.read_csv(input_csv)
    if d.empty:
        raise ValueError(f"Empty TG window summary input: {input_csv}")
    required = [
        "subject",
        "summary",
        "window",
        "train_day",
        "test_day",
        "mean_auc",
    ]
    missing = []
    for col in required:
        if col not in d.columns:
            missing.append(col)
    if len(missing) > 0:
        raise ValueError(f"Missing columns in {input_csv}: {missing}")
    for col in ["subject", "train_day", "test_day", "mean_auc"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=required).copy()
    d = d[d["train_day"] != d["test_day"]].copy()
    if d.empty:
        raise ValueError(f"No off-diagonal TG rows in {input_csv}")
    d["subject"] = d["subject"].astype(int)
    d["train_day"] = d["train_day"].astype(int)
    d["test_day"] = d["test_day"].astype(int)
    return d


def make_symmetrised_similarity(d):
    d = d.copy()
    d["day_low"] = np.minimum(d["train_day"], d["test_day"])
    d["day_high"] = np.maximum(d["train_day"], d["test_day"])
    subject_rows = (
        d.groupby(["summary", "window", "subject", "day_low", "day_high"], as_index=False)
        .agg(
            similarity=("mean_auc", "mean"),
            n_directions=("mean_auc", "size"),
        )
        .sort_values(["summary", "window", "subject", "day_low", "day_high"])
    )
    group_rows = (
        subject_rows.groupby(["summary", "window", "day_low", "day_high"], as_index=False)
        .agg(
            similarity_mean=("similarity", "mean"),
            similarity_sem=("similarity", sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["summary", "window", "day_low", "day_high"])
    )
    out = []
    for _, row in subject_rows.iterrows():
        out.append(
            {
                "row_type": "subject",
                "summary": row["summary"],
                "window": row["window"],
                "subject": int(row["subject"]),
                "day_low": int(row["day_low"]),
                "day_high": int(row["day_high"]),
                "similarity": float(row["similarity"]),
                "similarity_mean": np.nan,
                "similarity_sem": np.nan,
                "n_subjects": np.nan,
                "n_directions": int(row["n_directions"]),
            }
        )
    for _, row in group_rows.iterrows():
        out.append(
            {
                "row_type": "group",
                "summary": row["summary"],
                "window": row["window"],
                "subject": np.nan,
                "day_low": int(row["day_low"]),
                "day_high": int(row["day_high"]),
                "similarity": np.nan,
                "similarity_mean": float(row["similarity_mean"]),
                "similarity_sem": float(row["similarity_sem"]),
                "n_subjects": int(row["n_subjects"]),
                "n_directions": np.nan,
            }
        )
    return pd.DataFrame(out)


def group_similarity_matrix(sym_df, summary, window):
    g = sym_df[
        (sym_df["row_type"] == "group")
        & (sym_df["summary"] == summary)
        & (sym_df["window"] == window)
    ]
    if g.empty:
        raise ValueError(f"Missing symmetrised group rows: summary={summary}, window={window}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row["similarity_mean"])
        mat[j, i] = float(row["similarity_mean"])
    missing = []
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            if i == j:
                continue
            if not np.isfinite(mat[i, j]):
                missing.append(f"D{day_i}-D{day_j}")
    if len(missing) > 0:
        raise ValueError(
            f"Missing symmetrised day pairs for summary={summary}, window={window}: "
            + ", ".join(missing)
        )
    return mat


def distance_matrix_from_similarity(sim_mat):
    finite = sim_mat[np.isfinite(sim_mat)]
    if len(finite) == 0:
        raise ValueError("Cannot build distance matrix from empty similarity matrix")
    max_sim = float(np.max(finite))
    dist = np.full_like(sim_mat, np.nan, dtype=float)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if i == j:
                dist[i, j] = 0.0
            elif np.isfinite(sim_mat[i, j]):
                dist[i, j] = max_sim - float(sim_mat[i, j])
    if not np.all(np.isfinite(dist)):
        raise ValueError("Distance matrix contains missing values")
    return dist


def make_clusters(sym_df):
    rows = []
    for summary in SUMMARIES:
        for window in WINDOWS:
            sim_mat = group_similarity_matrix(sym_df, summary, window)
            dist = distance_matrix_from_similarity(sim_mat)
            condensed = squareform(dist, checks=False)
            z = linkage(condensed, method="average")
            order_idx = leaves_list(z)
            order_days = []
            for idx in order_idx:
                order_days.append(DAYS[int(idx)])
            order_labels = []
            for day in order_days:
                order_labels.append(str(day))
            day_order = ",".join(order_labels)
            for merge_idx in range(z.shape[0]):
                rows.append(
                    {
                        "row_type": "linkage",
                        "summary": summary,
                        "window": window,
                        "merge_index": int(merge_idx),
                        "child_1": float(z[merge_idx, 0]),
                        "child_2": float(z[merge_idx, 1]),
                        "distance": float(z[merge_idx, 2]),
                        "n_members": int(z[merge_idx, 3]),
                        "day": np.nan,
                        "order_position": np.nan,
                        "day_order": day_order,
                    }
                )
            for pos, day in enumerate(order_days):
                rows.append(
                    {
                        "row_type": "order",
                        "summary": summary,
                        "window": window,
                        "merge_index": np.nan,
                        "child_1": np.nan,
                        "child_2": np.nan,
                        "distance": np.nan,
                        "n_members": np.nan,
                        "day": int(day),
                        "order_position": int(pos),
                        "day_order": day_order,
                    }
                )
    return pd.DataFrame(rows)


def classical_mds(dist):
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    h = np.eye(n) - np.ones((n, n)) / float(n)
    b = -0.5 * h @ (dist**2) @ h
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    coords = np.zeros((n, 2), dtype=float)
    for dim in range(2):
        if dim < len(eigvals) and eigvals[dim] > 0:
            coords[:, dim] = eigvecs[:, dim] * np.sqrt(eigvals[dim])
    return coords, eigvals


def make_embedding(sym_df):
    rows = []
    for summary in SUMMARIES:
        for window in WINDOWS:
            sim_mat = group_similarity_matrix(sym_df, summary, window)
            dist = distance_matrix_from_similarity(sim_mat)
            coords, eigvals = classical_mds(dist)
            positive_total = 0.0
            for val in eigvals:
                if val > 0:
                    positive_total += float(val)
            explained = np.nan
            if positive_total > 0:
                numerator = 0.0
                for dim in range(min(2, len(eigvals))):
                    if eigvals[dim] > 0:
                        numerator += float(eigvals[dim])
                explained = numerator / positive_total
            for i, day in enumerate(DAYS):
                rows.append(
                    {
                        "summary": summary,
                        "window": window,
                        "day": int(day),
                        "x": float(coords[i, 0]),
                        "y": float(coords[i, 1]),
                        "variance_explained_2d": float(explained),
                    }
                )
    return pd.DataFrame(rows)


def make_model_comparison(sym_df):
    d = sym_df[sym_df["row_type"] == "subject"].copy()
    if d.empty:
        raise ValueError("No subject-level symmetrised rows for candidate model comparison")
    model_names = candidate_models()
    subject_rows = []
    for (summary, window, subject), g in d.groupby(["summary", "window", "subject"]):
        obs = g["similarity"].to_numpy(dtype=float)
        for model in model_names:
            pred_vals = []
            for _, row in g.iterrows():
                pred_vals.append(
                    candidate_value(model, int(row["day_low"]), int(row["day_high"]))
                )
            r = pearson_corr(obs, np.asarray(pred_vals, dtype=float))
            subject_rows.append(
                {
                    "row_type": "subject",
                    "summary": summary,
                    "window": window,
                    "subject": int(subject),
                    "model": model,
                    "r": r,
                    "r_mean": np.nan,
                    "r_sem": np.nan,
                    "t": np.nan,
                    "p": np.nan,
                    "n_subjects": np.nan,
                }
            )
    subject_df = pd.DataFrame(subject_rows)
    summary_rows = []
    for (summary, window, model), g in subject_df.groupby(["summary", "window", "model"]):
        vals = g["r"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            raise ValueError(
                f"No finite candidate-model correlations: "
                f"summary={summary}, window={window}, model={model}"
            )
        t_res = ttest_1samp(vals, 0.0, nan_policy="omit")
        summary_rows.append(
            {
                "row_type": "summary",
                "summary": summary,
                "window": window,
                "subject": np.nan,
                "model": model,
                "r": np.nan,
                "r_mean": float(np.mean(vals)),
                "r_sem": sem(vals),
                "t": float(t_res.statistic),
                "p": float(t_res.pvalue),
                "n_subjects": int(len(vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    return pd.concat([subject_df, summary_df], ignore_index=True)


def run_mvpa_stim_locked_cat_tg_day_structure(
    input_csv: Path | str = INPUT_CSV,
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    d = load_input(input_csv)
    sym_df = make_symmetrised_similarity(d)
    clusters_df = make_clusters(sym_df)
    embedding_df = make_embedding(sym_df)
    model_df = make_model_comparison(sym_df)

    sym_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_symmetrised.csv"
    clusters_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_clusters.csv"
    embedding_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_embedding.csv"
    model_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_model_comparison.csv"

    sym_df.to_csv(sym_csv, index=False)
    clusters_df.to_csv(clusters_csv, index=False)
    embedding_df.to_csv(embedding_csv, index=False)
    model_df.to_csv(model_csv, index=False)
    print(f"[TG day-structure] Wrote {sym_csv}")
    print(f"[TG day-structure] Wrote {clusters_csv}")
    print(f"[TG day-structure] Wrote {embedding_csv}")
    print(f"[TG day-structure] Wrote {model_csv}")
    return {
        "symmetrised_df": sym_df,
        "clusters_df": clusters_df,
        "embedding_df": embedding_df,
        "model_df": model_df,
        "symmetrised_csv": sym_csv,
        "clusters_csv": clusters_csv,
        "embedding_csv": embedding_csv,
        "model_csv": model_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_tg_day_structure()
