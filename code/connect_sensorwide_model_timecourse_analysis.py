#!/usr/bin/env python3
"""Connectivity day-geometry model correlations across time."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from connect_sensorwide_analysis import OUTPUT_DIR

DAYS = [1, 2, 3, 4, 5]
LOCK_TYPE = "stim"
BAND = "broadband"
ACTIVE_PAIR_PCT = 0.20
ACTIVE_PAIR_PCTS = [0.10, 0.20, 0.30, 0.50, 0.70, 0.90, 1.00]
WINDOW_SEC = 0.05
METRIC = "z_euclidean"
MIN_DAY_PAIRS = 6


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing sensorwide connectivity input: {path}. "
            "Run connect_sensorwide_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty sensorwide connectivity input: {path}")
    return d


def pair_label(ch_i, ch_j):
    return f"{ch_i}--{ch_j}"


def add_pair_labels(d):
    labels = []
    for row in d.itertuples(index=False):
        labels.append(pair_label(row.ch_i, row.ch_j))
    d = d.copy()
    d["pair_label"] = labels
    return d


def rank_active_pairs(carpet_df):
    d = carpet_df[
        (carpet_df["lock_type"] == LOCK_TYPE) & (carpet_df["band"] == BAND)
    ].copy()
    if d.empty:
        raise ValueError("Missing stim broadband rows in sensorwide carpet table")
    d = add_pair_labels(d)
    rows = []
    for pair, g in d.groupby("pair_label"):
        vals = g["conn_val"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "pair_label": pair,
                "ch_i": str(g["ch_i"].iloc[0]),
                "ch_j": str(g["ch_j"].iloc[0]),
                "modulation": float(np.max(vals) - np.min(vals)),
            }
        )
    pair_df = pd.DataFrame(rows)
    if pair_df.empty:
        raise ValueError("No finite stim broadband sensor pairs found")
    pair_df = pair_df.sort_values("modulation", ascending=False).reset_index(drop=True)
    pair_df["active_rank"] = np.arange(1, len(pair_df) + 1)
    return pair_df


def select_active_pairs(ranked_pair_df, active_pct):
    n_pairs = int(len(ranked_pair_df))
    top_k = max(1, int(np.ceil(float(active_pct) * n_pairs)))
    active_df = ranked_pair_df.iloc[:top_k].copy()
    active_df["active_pct"] = float(active_pct)
    active_df["n_edges"] = int(top_k)
    return active_df


def rank_vector(vals):
    vals = np.asarray(vals, dtype=float)
    good = np.isfinite(vals)
    out = np.full(vals.shape, np.nan, dtype=float)
    if int(np.sum(good)) == 0:
        return out
    finite = vals[good]
    order = np.argsort(finite, kind="mergesort")
    ranks = np.empty(len(finite), dtype=float)
    sorted_vals = finite[order]
    start = 0
    while start < len(sorted_vals):
        stop = start + 1
        while stop < len(sorted_vals) and sorted_vals[stop] == sorted_vals[start]:
            stop += 1
        rank_val = (start + stop - 1) / 2.0
        for idx in range(start, stop):
            ranks[order[idx]] = rank_val
        start = stop
    out[good] = ranks
    return out


def finite_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 3:
        return np.nan
    x = x[good]
    y = y[good]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x * y) / denom)


def spearman_corr(x, y):
    return finite_corr(rank_vector(x), rank_vector(y))


def edge_vector_distance(vec_a, vec_b):
    vals_a = []
    vals_b = []
    for idx, val_a in enumerate(vec_a):
        val_b = vec_b[idx]
        if np.isfinite(val_a) and np.isfinite(val_b):
            vals_a.append(float(val_a))
            vals_b.append(float(val_b))
    if len(vals_a) < 2:
        return np.nan
    arr_a = np.asarray(vals_a, dtype=float)
    arr_b = np.asarray(vals_b, dtype=float)
    std_a = float(np.std(arr_a))
    std_b = float(np.std(arr_b))
    if std_a <= np.finfo(float).eps or std_b <= np.finfo(float).eps:
        return np.nan
    z_a = (arr_a - float(np.mean(arr_a))) / std_a
    z_b = (arr_b - float(np.mean(arr_b))) / std_b
    return float(np.sqrt(np.sum((z_a - z_b) ** 2)))


def model_distance(model, day_i, day_j, split_day=None):
    if model == "gradual":
        return float(abs(day_i - day_j) / 4.0)
    if model == "two_stage_binary":
        if split_day is None:
            raise ValueError("two_stage_binary requires split_day")
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late == j_late:
            return 0.0
        return 1.0
    if model == "two_stage_hybrid":
        if split_day is None:
            raise ValueError("two_stage_hybrid requires split_day")
        gradual = float(abs(day_i - day_j) / 4.0)
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late == j_late:
            return 0.5 * gradual
        return 0.5 + 0.5 * gradual
    raise ValueError(f"Unknown model: {model}")


def model_specs():
    rows = [{"model": "gradual", "split_day": np.nan, "model_label": "gradual"}]
    for model in ["two_stage_binary", "two_stage_hybrid"]:
        for split_day in [1, 2, 3, 4]:
            rows.append(
                {
                    "model": model,
                    "split_day": float(split_day),
                    "model_label": f"{model}_D{split_day}",
                }
            )
    return rows


def active_subject_rows(subject_df, active_df):
    d = subject_df[
        (subject_df["lock_type"] == LOCK_TYPE) & (subject_df["band"] == BAND)
    ].copy()
    if d.empty:
        raise ValueError("Missing subject-level stim broadband connectivity rows")
    d = add_pair_labels(d)
    keep_pairs = set(active_df["pair_label"].tolist())
    keep = []
    for pair in d["pair_label"]:
        keep.append(pair in keep_pairs)
    d = d[np.asarray(keep, dtype=bool)].copy()
    if d.empty:
        raise ValueError("No subject-level rows remain after active-pair filtering")
    d["time_center_sec"] = d["lock_time"].astype(float) + WINDOW_SEC / 2.0
    return d


def build_vectors(d, active_df):
    active_pairs = active_df["pair_label"].tolist()
    vectors = {}
    group_cols = ["subject", "day", "lock_time", "time_center_sec"]
    for key, g in d.groupby(group_cols):
        subject, day, lock_time, time_center = key
        pair_vals = {}
        for row in g.itertuples(index=False):
            pair_vals[row.pair_label] = float(row.conn_val)
        vals = []
        for pair in active_pairs:
            vals.append(pair_vals.get(pair, np.nan))
        vectors[
            (
                int(subject),
                int(day),
                float(lock_time),
                float(time_center),
            )
        ] = np.asarray(vals, dtype=float)
    if not vectors:
        raise ValueError("No subject/time connectivity vectors were built")
    return vectors


def subject_time_day_pairs(vectors, subject, lock_time, time_center):
    rows = []
    emp_vals = []
    for day_i in DAYS:
        key_i = (subject, day_i, lock_time, time_center)
        if key_i not in vectors:
            continue
        for day_j in DAYS:
            if day_j <= day_i:
                continue
            key_j = (subject, day_j, lock_time, time_center)
            if key_j not in vectors:
                continue
            dist = edge_vector_distance(vectors[key_i], vectors[key_j])
            if not np.isfinite(dist):
                continue
            rows.append({"day_i": day_i, "day_j": day_j})
            emp_vals.append(float(dist))
    return rows, np.asarray(emp_vals, dtype=float)


def score_subject_time(day_rows, emp_vals):
    rows = []
    for spec in model_specs():
        pred_vals = []
        split_arg = None
        if np.isfinite(spec["split_day"]):
            split_arg = int(spec["split_day"])
        for row in day_rows:
            pred_vals.append(
                model_distance(
                    spec["model"],
                    int(row["day_i"]),
                    int(row["day_j"]),
                    split_day=split_arg,
                )
            )
        pred = np.asarray(pred_vals, dtype=float)
        rows.append(
            {
                "model": spec["model"],
                "split_day": spec["split_day"],
                "model_label": spec["model_label"],
                "rho": spearman_corr(emp_vals, pred),
            }
        )
    return rows


def subject_scores(vectors, active_pct, n_edges):
    subject_times = []
    for key in vectors.keys():
        subject, _day, lock_time, time_center = key
        value = (subject, lock_time, time_center)
        if value not in subject_times:
            subject_times.append(value)
    subject_times = sorted(subject_times)

    rows = []
    for subject, lock_time, time_center in subject_times:
        day_rows, emp_vals = subject_time_day_pairs(
            vectors,
            int(subject),
            float(lock_time),
            float(time_center),
        )
        if len(emp_vals) < MIN_DAY_PAIRS:
            continue
        score_rows = score_subject_time(day_rows, emp_vals)
        for row in score_rows:
            rows.append(
                {
                    "subject": int(subject),
                    "lock_time": float(lock_time),
                    "time_center_sec": float(time_center),
                    "metric": METRIC,
                    "active_pct": float(active_pct),
                    "n_edges": int(n_edges),
                    "n_day_pairs": int(len(emp_vals)),
                    "model": row["model"],
                    "split_day": row["split_day"],
                    "model_label": row["model_label"],
                    "rho": row["rho"],
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No model timecourse subject scores were produced")
    return out


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def group_summary(score_df):
    rows = []
    group_cols = [
        "active_pct",
        "lock_time",
        "time_center_sec",
        "metric",
        "model",
        "split_day",
    ]
    for key, g in score_df.groupby(group_cols, dropna=False):
        active_pct, lock_time, time_center, metric, model, split_day = key
        vals = g["rho"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        model_label = str(g["model_label"].iloc[0])
        rows.append(
            {
                "active_pct": float(active_pct),
                "lock_time": float(lock_time),
                "time_center_sec": float(time_center),
                "metric": metric,
                "model": model,
                "split_day": split_day,
                "model_label": model_label,
                "rho_mean": float(np.mean(vals)),
                "rho_sem": sem(vals),
                "n_subjects": int(len(vals)),
                "n_edges": int(g["n_edges"].iloc[0]),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No model timecourse group summary rows were produced")
    return out.sort_values(["time_center_sec", "model_label"])


def best_family_summary(score_df):
    rows = []
    for key, g_time in score_df.groupby(
        ["active_pct", "lock_time", "time_center_sec"]
    ):
        active_pct, lock_time, time_center = key
        for family in ["two_stage_binary", "two_stage_hybrid"]:
            best_split = np.nan
            best_mean = np.nan
            best_sem = np.nan
            best_n = 0
            for split_day in [1, 2, 3, 4]:
                g = g_time[
                    (g_time["model"] == family)
                    & (g_time["split_day"] == float(split_day))
                ]
                vals = g["rho"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if len(vals) == 0:
                    continue
                mean_val = float(np.mean(vals))
                if not np.isfinite(best_mean) or mean_val > best_mean:
                    best_split = float(split_day)
                    best_mean = mean_val
                    best_sem = sem(vals)
                    best_n = int(len(vals))
            if np.isfinite(best_mean):
                rows.append(
                    {
                        "active_pct": float(active_pct),
                        "lock_time": float(lock_time),
                        "time_center_sec": float(time_center),
                        "metric": METRIC,
                        "model": f"{family}_best",
                        "split_day": best_split,
                        "rho_mean": best_mean,
                        "rho_sem": best_sem,
                        "n_subjects": best_n,
                        "n_edges": int(g_time["n_edges"].iloc[0]),
                    }
                )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No best-family summary rows were produced")
    return out.sort_values(["time_center_sec", "model"])


def run_connect_sensorwide_model_timecourse(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    carpet_df = require_csv(output_dir / "sensorwide_carpet_timeseries.csv")
    subject_df = require_csv(output_dir / "sensorwide_carpet_subject_timeseries.csv")
    ranked_pair_df = rank_active_pairs(carpet_df)
    active_frames = []
    score_frames = []
    for active_pct in ACTIVE_PAIR_PCTS:
        active_df = select_active_pairs(ranked_pair_df, active_pct)
        print(
            "[connect model-timecourse] "
            f"active_pct={active_pct:.2f}, n_edges={len(active_df)}",
            flush=True,
        )
        active_frames.append(active_df)
        active_rows = active_subject_rows(subject_df, active_df)
        vectors = build_vectors(active_rows, active_df)
        score_df_this = subject_scores(vectors, active_pct, len(active_df))
        score_frames.append(score_df_this)
    active_df = pd.concat(active_frames, ignore_index=True)
    score_df = pd.concat(score_frames, ignore_index=True)
    summary_df = group_summary(score_df)
    best_df = best_family_summary(score_df)

    active_path = output_dir / "connect_sensorwide_model_timecourse_active_pairs.csv"
    score_path = output_dir / "connect_sensorwide_model_timecourse_subject_scores.csv"
    summary_path = output_dir / "connect_sensorwide_model_timecourse_summary.csv"
    best_path = output_dir / "connect_sensorwide_model_timecourse_best_summary.csv"

    active_df.to_csv(active_path, index=False)
    score_df.to_csv(score_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    best_df.to_csv(best_path, index=False)

    print(f"[connect model-timecourse] wrote {active_path}", flush=True)
    print(f"[connect model-timecourse] wrote {score_path}", flush=True)
    print(f"[connect model-timecourse] wrote {summary_path}", flush=True)
    print(f"[connect model-timecourse] wrote {best_path}", flush=True)
    return {
        "active_pairs": active_path,
        "subject_scores": score_path,
        "summary": summary_path,
        "best_summary": best_path,
    }


if __name__ == "__main__":
    run_connect_sensorwide_model_timecourse()
