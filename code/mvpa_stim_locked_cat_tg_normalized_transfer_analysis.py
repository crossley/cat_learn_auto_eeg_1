#!/usr/bin/env python3
"""TG day structure after normalising cross-day transfer by within-day decoding."""

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
CHANCE_AUC = 0.5
MIN_WITHIN_SIGNAL = 1.0e-6


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
            f"Missing TG window input: {input_csv}. "
            "Run mvpa_stim_locked_cat_tg_day1_distinctiveness_analysis.py first."
        )
    d = pd.read_csv(input_csv)
    if d.empty:
        raise ValueError(f"Empty TG window input: {input_csv}")
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
    if d.empty:
        raise ValueError(f"No usable rows in TG window input: {input_csv}")
    d["subject"] = d["subject"].astype(int)
    d["train_day"] = d["train_day"].astype(int)
    d["test_day"] = d["test_day"].astype(int)
    return d


def within_lookup(d):
    d_within = d[d["train_day"] == d["test_day"]].copy()
    if d_within.empty:
        raise ValueError("TG window input has no within-day rows")
    lookup = {}
    for _, row in d_within.iterrows():
        key = (
            int(row["subject"]),
            str(row["summary"]),
            str(row["window"]),
            int(row["train_day"]),
        )
        if key in lookup:
            raise ValueError(f"Duplicate within-day TG row: {key}")
        lookup[key] = float(row["mean_auc"])
    return lookup


def make_subject_normalized_pairs(d):
    lookup = within_lookup(d)
    d_cross = d[d["train_day"] != d["test_day"]].copy()
    if d_cross.empty:
        raise ValueError("TG window input has no cross-day rows")
    direction_values = {}
    for _, row in d_cross.iterrows():
        day_low = min(int(row["train_day"]), int(row["test_day"]))
        day_high = max(int(row["train_day"]), int(row["test_day"]))
        key = (
            int(row["subject"]),
            str(row["summary"]),
            str(row["window"]),
            day_low,
            day_high,
        )
        if key not in direction_values:
            direction_values[key] = []
        direction_values[key].append(float(row["mean_auc"]))

    rows = []
    for key, vals in sorted(direction_values.items()):
        subject, summary, window, day_low, day_high = key
        within_low_key = (subject, summary, window, day_low)
        within_high_key = (subject, summary, window, day_high)
        if within_low_key not in lookup or within_high_key not in lookup:
            raise ValueError(
                f"Missing within-day denominator for subject={subject}, "
                f"summary={summary}, window={window}, D{day_low}-D{day_high}"
            )
        cross_auc = float(np.mean(vals))
        within_low = float(lookup[within_low_key])
        within_high = float(lookup[within_high_key])
        cross_signal = cross_auc - CHANCE_AUC
        low_signal = within_low - CHANCE_AUC
        high_signal = within_high - CHANCE_AUC
        denom = np.nan
        norm = np.nan
        denominator_status = "ok"
        if low_signal <= MIN_WITHIN_SIGNAL or high_signal <= MIN_WITHIN_SIGNAL:
            denominator_status = "nonpositive_within_signal"
        else:
            denom = float(np.sqrt(low_signal * high_signal))
            norm = float(cross_signal / denom)
        rows.append(
            {
                "subject": int(subject),
                "summary": summary,
                "window": window,
                "day_low": int(day_low),
                "day_high": int(day_high),
                "day_distance": int(day_high - day_low),
                "cross_auc": cross_auc,
                "within_low_auc": within_low,
                "within_high_auc": within_high,
                "cross_signal": cross_signal,
                "within_low_signal": low_signal,
                "within_high_signal": high_signal,
                "denominator": denom,
                "normalized_transfer": norm,
                "n_directions": int(len(vals)),
                "denominator_status": denominator_status,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No normalised transfer rows were produced")
    return out


def make_group_summary(subject_df):
    rows = []
    for (summary, window, day_low, day_high), g in subject_df.groupby(
        ["summary", "window", "day_low", "day_high"]
    ):
        valid_norm = g["normalized_transfer"].to_numpy(dtype=float)
        valid_norm = valid_norm[np.isfinite(valid_norm)]
        rows.append(
            {
                "summary": summary,
                "window": window,
                "day_low": int(day_low),
                "day_high": int(day_high),
                "day_distance": int(day_high - day_low),
                "cross_auc_mean": float(np.mean(g["cross_auc"])),
                "cross_auc_sem": sem(g["cross_auc"]),
                "normalized_transfer_mean": (
                    float(np.mean(valid_norm)) if len(valid_norm) > 0 else np.nan
                ),
                "normalized_transfer_sem": sem(valid_norm),
                "n_subjects_total": int(g["subject"].nunique()),
                "n_subjects_valid_normalized": int(len(valid_norm)),
                "n_denominator_failed": int(
                    np.sum(g["denominator_status"] != "ok")
                ),
            }
        )
    out = pd.DataFrame(rows).sort_values(["summary", "window", "day_low", "day_high"])
    if out.empty:
        raise ValueError("No group normalised transfer rows were produced")
    return out


def group_similarity_matrix(group_df, summary, window, value_col):
    g = group_df[
        (group_df["summary"] == summary)
        & (group_df["window"] == window)
    ]
    if g.empty:
        raise ValueError(f"Missing group rows: summary={summary}, window={window}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row[value_col])
        mat[j, i] = float(row[value_col])
    missing = []
    for i, day_i in enumerate(DAYS):
        for j in range(i + 1, len(DAYS)):
            day_j = DAYS[j]
            if not np.isfinite(mat[i, j]):
                missing.append(f"D{day_i}-D{day_j}")
    if len(missing) > 0:
        raise ValueError(
            f"Missing finite {value_col} values for summary={summary}, "
            f"window={window}: " + ", ".join(missing)
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
        order_days.append(str(DAYS[int(idx)]))
    first_members = cluster_members(int(z[0, 0]), z)
    for day in cluster_members(int(z[0, 1]), z):
        first_members.append(day)
    first_members = sorted(first_members)
    first_labels = []
    for day in first_members:
        first_labels.append(f"D{day}")
    first_pair = "-".join(first_labels)
    final_left = cluster_members(int(z[-1, 0]), z)
    final_right = cluster_members(int(z[-1, 1]), z)
    last_singleton_day = np.nan
    if len(final_left) == 1 and len(final_right) > 1:
        last_singleton_day = int(final_left[0])
    elif len(final_right) == 1 and len(final_left) > 1:
        last_singleton_day = int(final_right[0])
    return z, ",".join(order_days), first_pair, last_singleton_day


def make_clusters(group_df, value_col="normalized_transfer_mean"):
    rows = []
    for summary in SUMMARIES:
        for window in WINDOWS:
            sim_mat = group_similarity_matrix(group_df, summary, window, value_col)
            dist = distance_matrix_from_similarity(sim_mat)
            z, day_order, first_pair, last_singleton_day = cluster_description_from_distance(dist)
            for merge_idx in range(z.shape[0]):
                rows.append(
                    {
                        "row_type": "linkage",
                        "summary": summary,
                        "window": window,
                        "value": value_col,
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
                        "value": value_col,
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


def run_mvpa_stim_locked_cat_tg_normalized_transfer(
    input_csv: Path | str = INPUT_CSV,
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    d = load_input(input_csv)
    subject_df = make_subject_normalized_pairs(d)
    group_df = make_group_summary(subject_df)
    clusters_df = make_clusters(group_df)

    subject_csv = output_dir / "mvpa_stim_locked_cat_tg_normalized_transfer_subject_pairs.csv"
    group_csv = output_dir / "mvpa_stim_locked_cat_tg_normalized_transfer_group_pairs.csv"
    clusters_csv = output_dir / "mvpa_stim_locked_cat_tg_normalized_transfer_clusters.csv"

    subject_df.to_csv(subject_csv, index=False)
    group_df.to_csv(group_csv, index=False)
    clusters_df.to_csv(clusters_csv, index=False)

    print(f"[TG normalized transfer] Wrote {subject_csv}")
    print(f"[TG normalized transfer] Wrote {group_csv}")
    print(f"[TG normalized transfer] Wrote {clusters_csv}")
    return {
        "subject_df": subject_df,
        "group_df": group_df,
        "clusters_df": clusters_df,
        "subject_csv": subject_csv,
        "group_csv": group_csv,
        "clusters_csv": clusters_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_tg_normalized_transfer()
