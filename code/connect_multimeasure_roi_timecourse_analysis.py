#!/usr/bin/env python3
"""Collapse multi-measure strict sensor edges into ROI time courses."""

from __future__ import annotations

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from connect_multimeasure_edge_timecourse_analysis import OUTPUT_PREFIX
from connect_multimeasure_utils import OUTPUT_DIR, sem

DEFAULT_EXTRA_GROUP_COLS = ["lock_type", "band", "measure"]


def require_path(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing multi-measure edge input: {path}. "
            "Run connect_multimeasure_edge_timecourse_analysis.py first."
        )
    return path


def summarize_edge_chunks(edge_path, extra_group_cols):
    usecols = (
        ["subject", "day", "roi_pair", "lock_time", "ch_i", "ch_j", "conn_val"]
        + list(extra_group_cols)
    )
    aggregate = {}
    for chunk_i, chunk in enumerate(
        pd.read_csv(edge_path, usecols=usecols, chunksize=250000)
    ):
        if chunk.empty:
            continue
        chunk["pair_key"] = chunk["ch_i"].astype(str) + "\t" + chunk["ch_j"].astype(str)
        group_cols = (
            list(extra_group_cols)
            + ["roi_pair", "subject", "day", "lock_time"]
        )
        summary = (
            chunk.groupby(group_cols, as_index=False)
            .agg(
                conn_sum=("conn_val", "sum"),
                conn_count=("conn_val", "count"),
                n_edges=("pair_key", "nunique"),
            )
        )
        for row in summary.itertuples(index=False):
            row_dict = row._asdict()
            key = tuple(row_dict[col] for col in group_cols)
            if key not in aggregate:
                aggregate[key] = [0.0, 0, 0]
            aggregate[key][0] += float(row_dict["conn_sum"])
            aggregate[key][1] += int(row_dict["conn_count"])
            aggregate[key][2] = max(
                aggregate[key][2], int(row_dict["n_edges"])
            )
        if (chunk_i + 1) % 10 == 0:
            print(f"[connect multimeasure ROI] chunks {chunk_i + 1}", flush=True)

    if not aggregate:
        raise ValueError("No multi-measure ROI rows found")

    rows = []
    group_cols = (
        list(extra_group_cols)
        + ["roi_pair", "subject", "day", "lock_time"]
    )
    for key, vals in aggregate.items():
        row = dict(zip(group_cols, key))
        conn_sum, conn_count, n_edges = vals
        row["subject"] = int(row["subject"])
        row["day"] = int(row["day"])
        row["lock_time"] = float(row["lock_time"])
        row["conn_val"] = float(conn_sum) / float(conn_count)
        row["n_edges"] = int(n_edges)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        list(extra_group_cols) + ["roi_pair", "subject", "day", "lock_time"]
    )


def summarize_day_means(subject_df, extra_group_cols):
    rows = []
    group_cols = list(extra_group_cols) + ["roi_pair", "day", "lock_time"]
    for key, group in subject_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        vals = group["conn_val"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        row["day"] = int(row["day"])
        row["lock_time"] = float(row["lock_time"])
        row["conn_mean"] = float(np.mean(vals)) if len(vals) else np.nan
        row["conn_sem"] = sem(vals)
        row["n_subjects"] = int(group["subject"].nunique())
        row["n_edges"] = int(group["n_edges"].max())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        list(extra_group_cols) + ["roi_pair", "day", "lock_time"]
    )


def run_connect_multimeasure_roi_timecourse(
    output_dir=OUTPUT_DIR,
    input_prefix=OUTPUT_PREFIX,
    output_prefix=OUTPUT_PREFIX,
    extra_group_cols=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if extra_group_cols is None:
        extra_group_cols = list(DEFAULT_EXTRA_GROUP_COLS)
    edge_path = require_path(output_dir / f"{input_prefix}_edge_subject_timeseries.csv")
    subject_df = summarize_edge_chunks(edge_path, extra_group_cols)
    day_df = summarize_day_means(subject_df, extra_group_cols)
    subject_path = output_dir / f"{output_prefix}_roi_timecourse_subject.csv"
    day_path = output_dir / f"{output_prefix}_roi_timecourse_day_mean.csv"
    subject_df.to_csv(subject_path, index=False)
    day_df.to_csv(day_path, index=False)
    print(f"[connect multimeasure ROI] wrote {subject_path}", flush=True)
    print(f"[connect multimeasure ROI] wrote {day_path}", flush=True)
    return {"subject": subject_path, "day_mean": day_path}


if __name__ == "__main__":
    run_connect_multimeasure_roi_timecourse()
