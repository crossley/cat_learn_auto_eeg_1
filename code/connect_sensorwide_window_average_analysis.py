#!/usr/bin/env python3
"""Window-averaged sensorwide connectivity day-distance matrices."""

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
from mvpa_stim_locked_cat_tg_window_structure_analysis import MVPA_CAT_TG_WINDOWS

DAYS = [1, 2, 3, 4, 5]
LOCK_TYPE = "stim"
BAND = "broadband"
ACTIVE_PAIR_PCT = 0.20
WINDOW_SEC = 0.05
METRICS = ["euclidean", "z_euclidean", "correlation"]


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


def select_active_pairs(carpet_df):
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
    n_pairs = int(len(pair_df))
    top_k = max(1, int(np.ceil(ACTIVE_PAIR_PCT * n_pairs)))
    pair_df = pair_df.sort_values("modulation", ascending=False).reset_index(drop=True)
    pair_df["active_rank"] = np.arange(1, len(pair_df) + 1)
    pair_df["active_pct"] = ACTIVE_PAIR_PCT
    active_df = pair_df.iloc[:top_k].copy()
    return active_df


def window_label_rows():
    rows = []
    for window, bounds in MVPA_CAT_TG_WINDOWS.items():
        rows.append(
            {
                "window": window,
                "window_start_sec": float(bounds[0]),
                "window_end_sec": float(bounds[1]),
            }
        )
    return rows


def make_window_subject_edges(subject_df, active_df):
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
    d["window_center_sec"] = d["lock_time"].astype(float) + WINDOW_SEC / 2.0

    rows = []
    for win_row in window_label_rows():
        lo = float(win_row["window_start_sec"])
        hi = float(win_row["window_end_sec"])
        d_win = d[(d["window_center_sec"] >= lo) & (d["window_center_sec"] <= hi)]
        if d_win.empty:
            raise ValueError(
                "No subject-level connectivity rows in window: "
                f"{win_row['window']} {lo}-{hi}s"
            )
        group_cols = ["subject", "day", "pair_label", "ch_i", "ch_j"]
        for key, g in d_win.groupby(group_cols):
            subject, day, pair, ch_i, ch_j = key
            vals = g["conn_val"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "window": str(win_row["window"]),
                    "window_start_sec": lo,
                    "window_end_sec": hi,
                    "pair_label": str(pair),
                    "ch_i": str(ch_i),
                    "ch_j": str(ch_j),
                    "conn_mean": float(np.mean(vals)),
                    "n_time_bins": int(len(vals)),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No window-averaged subject edge rows were produced")
    return out


def edge_vector_distance(vec_a, vec_b, metric):
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
    if metric == "euclidean":
        return float(np.sqrt(np.sum((arr_a - arr_b) ** 2)))
    if metric == "z_euclidean":
        std_a = float(np.std(arr_a))
        std_b = float(np.std(arr_b))
        if std_a <= np.finfo(float).eps or std_b <= np.finfo(float).eps:
            return np.nan
        z_a = (arr_a - float(np.mean(arr_a))) / std_a
        z_b = (arr_b - float(np.mean(arr_b))) / std_b
        return float(np.sqrt(np.sum((z_a - z_b) ** 2)))
    if metric == "correlation":
        std_a = float(np.std(arr_a))
        std_b = float(np.std(arr_b))
        if std_a <= np.finfo(float).eps or std_b <= np.finfo(float).eps:
            return np.nan
        corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
        return float(1.0 - corr)
    raise ValueError(f"Unknown distance metric: {metric}")


def subject_vectors(edge_df, active_df):
    active_pairs = active_df["pair_label"].tolist()
    vectors = {}
    for key, g in edge_df.groupby(["subject", "day", "window"]):
        subject, day, window = key
        pair_vals = {}
        for row in g.itertuples(index=False):
            pair_vals[row.pair_label] = float(row.conn_mean)
        vals = []
        for pair in active_pairs:
            vals.append(pair_vals.get(pair, np.nan))
        vectors[(int(subject), int(day), str(window))] = np.asarray(vals, dtype=float)
    if not vectors:
        raise ValueError("No subject window-average edge vectors were built")
    return vectors


def subject_distance_rows(edge_df, active_df):
    vectors = subject_vectors(edge_df, active_df)
    subjects = []
    for subject, _day, _window in vectors.keys():
        if subject not in subjects:
            subjects.append(subject)
    subjects = sorted(subjects)
    rows = []
    for subject in subjects:
        for window in MVPA_CAT_TG_WINDOWS.keys():
            for metric in METRICS:
                for train_day in DAYS:
                    for test_day in DAYS:
                        key_a = (subject, train_day, window)
                        key_b = (subject, test_day, window)
                        if key_a not in vectors or key_b not in vectors:
                            continue
                        dist = edge_vector_distance(
                            vectors[key_a],
                            vectors[key_b],
                            metric,
                        )
                        rows.append(
                            {
                                "subject": int(subject),
                                "window": window,
                                "metric": metric,
                                "train_day": int(train_day),
                                "test_day": int(test_day),
                                "day_distance": int(abs(train_day - test_day)),
                                "distance": dist,
                                "n_edges": int(len(active_df)),
                            }
                        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No subject window-average distance rows were produced")
    return out


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def group_distance_rows(subject_dist_df):
    rows = []
    group_cols = ["window", "metric", "train_day", "test_day"]
    for key, g in subject_dist_df.groupby(group_cols):
        window, metric, train_day, test_day = key
        vals = g["distance"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "window": window,
                "metric": metric,
                "train_day": int(train_day),
                "test_day": int(test_day),
                "day_distance": int(abs(train_day - test_day)),
                "distance_mean": float(np.mean(vals)),
                "distance_sem": sem(vals),
                "n_subjects": int(len(vals)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No group window-average distance rows were produced")
    return out


def run_connect_sensorwide_window_average(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    carpet_df = require_csv(output_dir / "sensorwide_carpet_timeseries.csv")
    subject_df = require_csv(output_dir / "sensorwide_carpet_subject_timeseries.csv")

    active_df = select_active_pairs(carpet_df)
    edge_df = make_window_subject_edges(subject_df, active_df)
    subject_dist_df = subject_distance_rows(edge_df, active_df)
    group_dist_df = group_distance_rows(subject_dist_df)

    active_path = output_dir / "connect_sensorwide_window_average_active_pairs.csv"
    edge_path = output_dir / "connect_sensorwide_window_average_subject_edges.csv"
    subject_path = (
        output_dir / "connect_sensorwide_window_average_subject_distances.csv"
    )
    group_path = output_dir / "connect_sensorwide_window_average_group_distances.csv"

    active_df.to_csv(active_path, index=False)
    edge_df.to_csv(edge_path, index=False)
    subject_dist_df.to_csv(subject_path, index=False)
    group_dist_df.to_csv(group_path, index=False)

    print(f"[connect window-average] wrote {active_path}", flush=True)
    print(f"[connect window-average] wrote {edge_path}", flush=True)
    print(f"[connect window-average] wrote {subject_path}", flush=True)
    print(f"[connect window-average] wrote {group_path}", flush=True)
    return {
        "active_pairs": active_path,
        "subject_edges": edge_path,
        "subject_distances": subject_path,
        "group_distances": group_path,
    }


if __name__ == "__main__":
    run_connect_sensorwide_window_average()
