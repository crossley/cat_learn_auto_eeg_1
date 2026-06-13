#!/usr/bin/env python3
"""Stimulus-locked connectivity time courses for strict cross-region sensor pairs."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from connect_sensorwide_analysis import OUTPUT_DIR
from sensor_rois import cross_roi_pairs

ROI_PAIR_SPECS = {
    "visual_frontal": ("visual", "frontal"),
    "visual_central": ("visual", "central"),
}


def require_path(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing connectivity ROI input: {path}. "
            "Run connect_sensorwide_analysis.py first."
        )
    return path


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def pair_lookup():
    lookup = {}
    for roi_pair, (source, target) in ROI_PAIR_SPECS.items():
        for ch_i, ch_j in cross_roi_pairs(source, target):
            lookup[f"{ch_i}\t{ch_j}"] = roi_pair
            lookup[f"{ch_j}\t{ch_i}"] = roi_pair
    return lookup


def summarize_subject_chunks(subject_path, lookup):
    aggregate = {}
    usecols = ["subject", "day", "lock_type", "band", "lock_time", "ch_i", "ch_j", "conn_val"]
    for chunk_i, chunk in enumerate(pd.read_csv(subject_path, usecols=usecols, chunksize=250000)):
        chunk = chunk[(chunk["lock_type"] == "stim") & (chunk["band"] == "broadband")].copy()
        if chunk.empty:
            continue
        chunk["pair_key"] = chunk["ch_i"].astype(str) + "\t" + chunk["ch_j"].astype(str)
        chunk["roi_pair"] = chunk["pair_key"].map(lookup)
        chunk = chunk[chunk["roi_pair"].notna()].copy()
        if chunk.empty:
            continue
        summary = (
            chunk.groupby(["roi_pair", "subject", "day", "lock_time"], as_index=False)
            .agg(
                conn_sum=("conn_val", "sum"),
                conn_count=("conn_val", "count"),
                n_edges=("pair_key", "nunique"),
            )
        )
        for row in summary.itertuples(index=False):
            key = (row.roi_pair, int(row.subject), int(row.day), float(row.lock_time))
            if key not in aggregate:
                aggregate[key] = [0.0, 0, 0]
            aggregate[key][0] += float(row.conn_sum)
            aggregate[key][1] += int(row.conn_count)
            aggregate[key][2] = max(aggregate[key][2], int(row.n_edges))
        if (chunk_i + 1) % 10 == 0:
            print(f"[connect ROI] subject chunks {chunk_i + 1}", flush=True)
    if not aggregate:
        raise ValueError("No connectivity ROI subject rows found")
    rows = []
    for key, vals in aggregate.items():
        roi_pair, subject, day, lock_time = key
        conn_sum, conn_count, n_edges = vals
        rows.append(
            {
                "roi_pair": roi_pair,
                "subject": int(subject),
                "day": int(day),
                "lock_time": float(lock_time),
                "conn_val": float(conn_sum) / float(conn_count),
                "n_edges": int(n_edges),
            }
        )
    return pd.DataFrame(rows).sort_values(["roi_pair", "subject", "day", "lock_time"])


def summarize_day_means(subject_df):
    rows = []
    for key, g in subject_df.groupby(["roi_pair", "day", "lock_time"]):
        roi_pair, day, lock_time = key
        vals = g["conn_val"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "roi_pair": roi_pair,
                "day": int(day),
                "lock_time": float(lock_time),
                "conn_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "conn_sem": sem(vals),
                "n_subjects": int(len(vals)),
                "n_edges": int(g["n_edges"].max()),
            }
        )
    return pd.DataFrame(rows).sort_values(["roi_pair", "day", "lock_time"])


def run_connect_roi_timecourse(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_path = require_path(output_dir / "sensorwide_carpet_subject_timeseries.csv")
    lookup = pair_lookup()
    subject_df = summarize_subject_chunks(subject_path, lookup)
    day_df = summarize_day_means(subject_df)
    subject_csv = output_dir / "connect_roi_timecourse_subject.csv"
    day_csv = output_dir / "connect_roi_timecourse_day_mean.csv"
    subject_df.to_csv(subject_csv, index=False)
    day_df.to_csv(day_csv, index=False)
    print(f"[connect ROI] wrote {subject_csv}", flush=True)
    print(f"[connect ROI] wrote {day_csv}", flush=True)
    return {"subject": subject_csv, "day_mean": day_csv}


if __name__ == "__main__":
    run_connect_roi_timecourse()
