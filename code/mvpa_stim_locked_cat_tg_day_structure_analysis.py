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

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

INPUT_CSV = OUTPUT_DIR / "mvpa_stim_locked_cat_tg_day_pair_window_auc_subject_pairs_by_summary.csv"
DAYS = [1, 2, 3, 4, 5]
SUMMARIES = ["square_mean", "diagonal_mean", "top10_mean"]
WINDOWS = ["early", "late"]
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


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


def finite_corr(x, y):
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


def matrix_to_pair_vector(mat):
    vals = []
    for i in range(len(DAYS)):
        for j in range(i + 1, len(DAYS)):
            vals.append(float(mat[i, j]))
    return np.asarray(vals, dtype=float)


def cluster_members(node_id, z):
    node_id = int(node_id)
    n_days = len(DAYS)
    if node_id < n_days:
        return [DAYS[node_id]]
    merge_idx = node_id - n_days
    members = []
    for child_col in [0, 1]:
        child_members = cluster_members(int(z[merge_idx, child_col]), z)
        for day in child_members:
            members.append(day)
    return sorted(members)


def cluster_description_from_distance(dist):
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

    first_members = cluster_members(int(z[0, 0]), z)
    for day in cluster_members(int(z[0, 1]), z):
        first_members.append(day)
    first_members = sorted(first_members)
    first_pair = ""
    if len(first_members) == 2:
        first_pair = f"D{first_members[0]}-D{first_members[1]}"

    last_row = z[z.shape[0] - 1]
    left_members = cluster_members(int(last_row[0]), z)
    right_members = cluster_members(int(last_row[1]), z)
    last_singleton_day = np.nan
    if len(left_members) == 1 and len(right_members) > 1:
        last_singleton_day = int(left_members[0])
    if len(right_members) == 1 and len(left_members) > 1:
        last_singleton_day = int(right_members[0])
    return z, day_order, first_pair, last_singleton_day


def make_clusters(sym_df):
    rows = []
    for summary in SUMMARIES:
        for window in WINDOWS:
            sim_mat = group_similarity_matrix(sym_df, summary, window)
            dist = distance_matrix_from_similarity(sim_mat)
            z, day_order, _first_pair, _last_singleton_day = cluster_description_from_distance(dist)
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
            order_parts = day_order.split(",")
            for pos, day_text in enumerate(order_parts):
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
                        "day": int(day_text),
                        "order_position": int(pos),
                        "day_order": day_order,
                    }
                )
    return pd.DataFrame(rows)


def subject_similarity_matrix(sym_df, summary, window, subject):
    g = sym_df[
        (sym_df["row_type"] == "subject")
        & (sym_df["summary"] == summary)
        & (sym_df["window"] == window)
        & (sym_df["subject"] == subject)
    ]
    if g.empty:
        raise ValueError(
            f"Missing subject symmetrised rows: summary={summary}, "
            f"window={window}, subject={subject}"
        )
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row["similarity"])
        mat[j, i] = float(row["similarity"])
    missing = []
    for i, day_i in enumerate(DAYS):
        for j in range(i + 1, len(DAYS)):
            day_j = DAYS[j]
            if not np.isfinite(mat[i, j]):
                missing.append(f"D{day_i}-D{day_j}")
    if len(missing) > 0:
        raise ValueError(
            f"Missing subject day pairs for summary={summary}, window={window}, "
            f"subject={subject}: " + ", ".join(missing)
        )
    return mat


def complete_subjects(sym_df, summary, window):
    d_key = sym_df[
        (sym_df["row_type"] == "subject")
        & (sym_df["summary"] == summary)
        & (sym_df["window"] == window)
    ]
    subjects = sorted(d_key["subject"].dropna().unique().astype(int))
    retained = []
    for subject in subjects:
        try:
            subject_similarity_matrix(sym_df, summary, window, subject)
            retained.append(int(subject))
        except ValueError:
            pass
    if len(retained) == 0:
        raise ValueError(f"No complete subjects: summary={summary}, window={window}")
    return retained


def subject_matrix_cache(sym_df, summary, window):
    subjects = complete_subjects(sym_df, summary, window)
    matrices = {}
    for subject in subjects:
        matrices[int(subject)] = subject_similarity_matrix(
            sym_df,
            summary,
            window,
            int(subject),
        )
    return subjects, matrices


def make_subject_clusters(sym_df):
    rows = []
    for summary in SUMMARIES:
        for window in WINDOWS:
            subjects, matrices = subject_matrix_cache(sym_df, summary, window)
            for subject in subjects:
                sim_mat = matrices[int(subject)]
                dist = distance_matrix_from_similarity(sim_mat)
                z, day_order, first_pair, last_singleton_day = cluster_description_from_distance(dist)
                for merge_idx in range(z.shape[0]):
                    rows.append(
                        {
                            "row_type": "linkage",
                            "summary": summary,
                            "window": window,
                            "subject": int(subject),
                            "merge_index": int(merge_idx),
                            "child_1": float(z[merge_idx, 0]),
                            "child_2": float(z[merge_idx, 1]),
                            "distance": float(z[merge_idx, 2]),
                            "n_members": int(z[merge_idx, 3]),
                            "day": np.nan,
                            "order_position": np.nan,
                            "day_order": day_order,
                            "first_pair": first_pair,
                            "last_singleton_day": last_singleton_day,
                        }
                    )
                order_parts = day_order.split(",")
                for pos, day_text in enumerate(order_parts):
                    rows.append(
                        {
                            "row_type": "order",
                            "summary": summary,
                            "window": window,
                            "subject": int(subject),
                            "merge_index": np.nan,
                            "child_1": np.nan,
                            "child_2": np.nan,
                            "distance": np.nan,
                            "n_members": np.nan,
                            "day": int(day_text),
                            "order_position": int(pos),
                            "day_order": day_order,
                            "first_pair": first_pair,
                            "last_singleton_day": last_singleton_day,
                        }
                    )
    return pd.DataFrame(rows)


def bootstrap_mean_similarity_matrix(matrices, sampled_subjects, summary, window):
    mats = []
    for subject in sampled_subjects:
        mats.append(matrices[int(subject)])
    if len(mats) == 0:
        raise ValueError("Cannot bootstrap empty subject sample")
    arr = np.stack(mats, axis=0)
    out = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for i in range(len(DAYS)):
        for j in range(len(DAYS)):
            if i == j:
                continue
            vals = arr[:, i, j]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                raise ValueError(
                    "Bootstrap sample has no finite values: "
                    f"summary={summary}, window={window}, "
                    f"day_pair=D{DAYS[i]}-D{DAYS[j]}"
                )
            out[i, j] = float(np.mean(vals))
    return out


def make_bootstrap_clusters(sym_df, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    rows = []
    for summary in SUMMARIES:
        for window in WINDOWS:
            print(
                f"[TG day-structure] Bootstrapping {summary}, {window} "
                f"({n_bootstrap} resamples)..."
            )
            subjects, matrices = subject_matrix_cache(sym_df, summary, window)
            if len(subjects) == 0:
                raise ValueError(
                    f"No subjects for bootstrap: summary={summary}, window={window}"
                )
            first_pair_counts = {}
            last_day_counts = {}
            for boot_idx in range(n_bootstrap):
                sampled = rng.choice(subjects, size=len(subjects), replace=True)
                sim_mat = bootstrap_mean_similarity_matrix(
                    matrices,
                    sampled,
                    summary,
                    window,
                )
                dist = distance_matrix_from_similarity(sim_mat)
                _z, day_order, first_pair, last_singleton_day = cluster_description_from_distance(dist)
                sampled_labels = []
                for subject in sampled:
                    sampled_labels.append(str(int(subject)))
                sampled_subjects = ",".join(sampled_labels)
                rows.append(
                    {
                        "row_type": "bootstrap",
                        "summary": summary,
                        "window": window,
                        "bootstrap": int(boot_idx),
                        "n_subjects": int(len(subjects)),
                        "sampled_subjects": sampled_subjects,
                        "first_pair": first_pair,
                        "last_singleton_day": last_singleton_day,
                        "day_order": day_order,
                        "support_type": "",
                        "event": "",
                        "support": np.nan,
                    }
                )
                if first_pair not in first_pair_counts:
                    first_pair_counts[first_pair] = 0
                first_pair_counts[first_pair] += 1
                if np.isfinite(last_singleton_day):
                    last_key = f"D{int(last_singleton_day)}"
                else:
                    last_key = "none"
                if last_key not in last_day_counts:
                    last_day_counts[last_key] = 0
                last_day_counts[last_key] += 1
            for event, count in sorted(first_pair_counts.items()):
                rows.append(
                    {
                        "row_type": "support",
                        "summary": summary,
                        "window": window,
                        "bootstrap": np.nan,
                        "n_subjects": int(len(subjects)),
                        "sampled_subjects": "",
                        "first_pair": "",
                        "last_singleton_day": np.nan,
                        "day_order": "",
                        "support_type": "first_pair",
                        "event": event,
                        "support": float(count) / float(n_bootstrap),
                    }
                )
            for event, count in sorted(last_day_counts.items()):
                rows.append(
                    {
                        "row_type": "support",
                        "summary": summary,
                        "window": window,
                        "bootstrap": np.nan,
                        "n_subjects": int(len(subjects)),
                        "sampled_subjects": "",
                        "first_pair": "",
                        "last_singleton_day": np.nan,
                        "day_order": "",
                        "support_type": "last_singleton_day",
                        "event": event,
                        "support": float(count) / float(n_bootstrap),
                    }
                )
    return pd.DataFrame(rows)


def make_distance_stability(sym_df):
    rows = []
    for summary in SUMMARIES:
        for window in WINDOWS:
            group_sim = group_similarity_matrix(sym_df, summary, window)
            group_dist = distance_matrix_from_similarity(group_sim)
            group_vec = matrix_to_pair_vector(group_dist)
            subjects, matrices = subject_matrix_cache(sym_df, summary, window)
            subject_corrs = []
            for subject in subjects:
                subject_sim = matrices[int(subject)]
                subject_dist = distance_matrix_from_similarity(subject_sim)
                subject_vec = matrix_to_pair_vector(subject_dist)
                r = finite_corr(subject_vec, group_vec)
                subject_corrs.append(r)
                rows.append(
                    {
                        "row_type": "subject",
                        "summary": summary,
                        "window": window,
                        "subject": int(subject),
                        "distance_correlation": r,
                        "mean_distance_correlation": np.nan,
                        "sem_distance_correlation": np.nan,
                        "n_subjects": np.nan,
                    }
                )
            vals = np.asarray(subject_corrs, dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                raise ValueError(
                    f"No finite distance-stability values: summary={summary}, window={window}"
                )
            rows.append(
                {
                    "row_type": "summary",
                    "summary": summary,
                    "window": window,
                    "subject": np.nan,
                    "distance_correlation": np.nan,
                    "mean_distance_correlation": float(np.mean(vals)),
                    "sem_distance_correlation": sem(vals),
                    "n_subjects": int(len(vals)),
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
    subject_clusters_df = make_subject_clusters(sym_df)
    bootstrap_clusters_df = make_bootstrap_clusters(sym_df)
    distance_stability_df = make_distance_stability(sym_df)

    sym_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_symmetrised.csv"
    clusters_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_clusters.csv"
    embedding_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_embedding.csv"
    subject_clusters_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_subject_clusters.csv"
    bootstrap_clusters_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_bootstrap_clusters.csv"
    distance_stability_csv = output_dir / "mvpa_stim_locked_cat_tg_day_structure_distance_stability.csv"

    sym_df.to_csv(sym_csv, index=False)
    clusters_df.to_csv(clusters_csv, index=False)
    embedding_df.to_csv(embedding_csv, index=False)
    subject_clusters_df.to_csv(subject_clusters_csv, index=False)
    bootstrap_clusters_df.to_csv(bootstrap_clusters_csv, index=False)
    distance_stability_df.to_csv(distance_stability_csv, index=False)
    print(f"[TG day-structure] Wrote {sym_csv}")
    print(f"[TG day-structure] Wrote {clusters_csv}")
    print(f"[TG day-structure] Wrote {embedding_csv}")
    print(f"[TG day-structure] Wrote {subject_clusters_csv}")
    print(f"[TG day-structure] Wrote {bootstrap_clusters_csv}")
    print(f"[TG day-structure] Wrote {distance_stability_csv}")
    return {
        "symmetrised_df": sym_df,
        "clusters_df": clusters_df,
        "embedding_df": embedding_df,
        "subject_clusters_df": subject_clusters_df,
        "bootstrap_clusters_df": bootstrap_clusters_df,
        "distance_stability_df": distance_stability_df,
        "symmetrised_csv": sym_csv,
        "clusters_csv": clusters_csv,
        "embedding_csv": embedding_csv,
        "subject_clusters_csv": subject_clusters_csv,
        "bootstrap_clusters_csv": bootstrap_clusters_csv,
        "distance_stability_csv": distance_stability_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_tg_day_structure()
