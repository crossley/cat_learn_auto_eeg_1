#!/usr/bin/env python3
"""BIC comparisons for connectivity day-distance model timecourses."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from connect_sensorwide_analysis import OUTPUT_DIR
from connect_sensorwide_model_timecourse_analysis import model_distance

ACTIVE_PCT = float(os.environ.get("ACTIVE_PCT", "0.10"))


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing connectivity model-timecourse input: {path}. "
            "Run connect_sensorwide_model_timecourse_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty connectivity model-timecourse input: {path}")
    return d


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def fit_bic(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    good = np.isfinite(y)
    for col_i in range(x.shape[1]):
        good &= np.isfinite(x[:, col_i])
    y = y[good]
    x = x[good]
    n_obs = int(len(y))
    if n_obs < 4:
        return {"bic": np.nan, "r2": np.nan, "n_obs": n_obs, "n_params": np.nan}
    keep_cols = [0]
    for col_i in range(1, x.shape[1]):
        col = x[:, col_i]
        if float(np.nanmax(col) - np.nanmin(col)) > np.finfo(float).eps:
            keep_cols.append(col_i)
    x = x[:, keep_cols]
    n_params = int(x.shape[1])
    beta, _resid, rank, _singular = np.linalg.lstsq(x, y, rcond=None)
    if int(rank) < n_params:
        return {"bic": np.nan, "r2": np.nan, "n_obs": n_obs, "n_params": n_params}
    pred = x @ beta
    resid = y - pred
    rss = max(float(np.sum(resid**2)), np.finfo(float).eps)
    tss = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = np.nan
    if tss > np.finfo(float).eps:
        r2 = 1.0 - rss / tss
    bic = float(n_obs * np.log(rss / float(n_obs)) + n_params * np.log(float(n_obs)))
    return {"bic": bic, "r2": float(r2), "n_obs": n_obs, "n_params": n_params}


def model_specs():
    rows = [{"model": "baseline", "split_day": np.nan, "model_label": "baseline"}]
    rows.append({"model": "gradual", "split_day": np.nan, "model_label": "gradual"})
    for split_day in [1, 2, 3, 4]:
        rows.append(
            {
                "model": "two_stage_hybrid",
                "split_day": float(split_day),
                "model_label": f"two_stage_hybrid_D{split_day}",
            }
        )
    return rows


def design_matrix(day_rows, spec):
    intercept = np.ones(len(day_rows), dtype=float)
    if spec["model"] == "baseline":
        return intercept.reshape(-1, 1)
    split_day = None
    if np.isfinite(spec["split_day"]):
        split_day = int(spec["split_day"])
    pred = []
    for row in day_rows:
        pred.append(
            model_distance(
                spec["model"],
                int(row["day_i"]),
                int(row["day_j"]),
                split_day=split_day,
            )
        )
    return np.column_stack([intercept, np.asarray(pred, dtype=float)])


def score_subject_time_rows(d_subject_time):
    day_rows = []
    emp_vals = []
    for row in d_subject_time.itertuples(index=False):
        day_rows.append({"day_i": int(row.day_i), "day_j": int(row.day_j)})
        emp_vals.append(float(row.distance))
    y = np.asarray(emp_vals, dtype=float)
    rows = []
    for spec in model_specs():
        fit = fit_bic(y, design_matrix(day_rows, spec))
        rows.append(
            {
                "model": spec["model"],
                "split_day": spec["split_day"],
                "model_label": spec["model_label"],
                "bic": fit["bic"],
                "r2": fit["r2"],
                "n_obs": fit["n_obs"],
                "n_params": fit["n_params"],
            }
        )
    return rows


def add_delta_bic(score_df):
    frames = []
    group_cols = ["active_pct", "subject", "lock_time", "time_center_sec", "metric"]
    for _key, g in score_df.groupby(group_cols, dropna=False):
        g = g.copy()
        finite = g["bic"].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        if len(finite) == 0:
            g["delta_bic_best"] = np.nan
        else:
            g["delta_bic_best"] = g["bic"].astype(float) - float(np.min(finite))
        baseline = g[g["model_label"] == "baseline"]
        if baseline.empty or not np.isfinite(float(baseline["bic"].iloc[0])):
            g["delta_bic_baseline"] = np.nan
        else:
            g["delta_bic_baseline"] = (
                g["bic"].astype(float) - float(baseline["bic"].iloc[0])
            )
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def summarize(score_df):
    rows = []
    group_cols = [
        "active_pct",
        "lock_time",
        "time_center_sec",
        "metric",
        "model",
        "split_day",
        "model_label",
    ]
    for key, g in score_df.groupby(group_cols, dropna=False):
        active_pct, lock_time, time_center, metric, model, split_day, model_label = key
        bic_vals = g["bic"].to_numpy(float)
        delta_best = g["delta_bic_best"].to_numpy(float)
        delta_base = g["delta_bic_baseline"].to_numpy(float)
        r2_vals = g["r2"].to_numpy(float)
        bic_vals = bic_vals[np.isfinite(bic_vals)]
        delta_best = delta_best[np.isfinite(delta_best)]
        delta_base = delta_base[np.isfinite(delta_base)]
        r2_vals = r2_vals[np.isfinite(r2_vals)]
        if len(bic_vals) == 0:
            continue
        rows.append(
            {
                "active_pct": float(active_pct),
                "lock_time": float(lock_time),
                "time_center_sec": float(time_center),
                "metric": metric,
                "model": model,
                "split_day": split_day,
                "model_label": model_label,
                "bic_mean": float(np.mean(bic_vals)),
                "bic_sem": sem(bic_vals),
                "delta_bic_best_mean": float(np.mean(delta_best))
                if len(delta_best)
                else np.nan,
                "delta_bic_baseline_mean": float(np.mean(delta_base))
                if len(delta_base)
                else np.nan,
                "r2_mean": float(np.mean(r2_vals)) if len(r2_vals) else np.nan,
                "r2_sem": sem(r2_vals),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No connectivity BIC summary rows produced")
    return out.sort_values(["active_pct", "time_center_sec", "delta_bic_best_mean"])


def run_connect_sensorwide_model_bic_timecourse(
    output_dir: Path | str = OUTPUT_DIR,
    active_pct: float = ACTIVE_PCT,
):
    output_dir = Path(output_dir)
    subject_df = require_csv(output_dir / "sensorwide_carpet_subject_timeseries.csv")

    # Rebuild the empirical day-pair distances from the active edge sets so BIC
    # is fit to distances directly instead of to correlation summaries.
    from connect_sensorwide_model_timecourse_analysis import (
        active_subject_rows,
        build_vectors,
        select_active_pairs,
        subject_time_day_pairs,
        rank_active_pairs,
    )

    carpet_df = require_csv(output_dir / "sensorwide_carpet_timeseries.csv")
    ranked_pair_df = rank_active_pairs(carpet_df)

    rows = []
    active_pct = float(active_pct)
    active_this = select_active_pairs(ranked_pair_df, active_pct)
    active_rows = active_subject_rows(subject_df, active_this)
    vectors = build_vectors(active_rows, active_this)
    subject_times = []
    seen_subject_times = set()
    for key in vectors.keys():
        subject, _day, lock_time, time_center = key
        value = (subject, lock_time, time_center)
        if value not in seen_subject_times:
            seen_subject_times.add(value)
            subject_times.append(value)
    for subject, lock_time, time_center in sorted(subject_times):
        day_rows, emp_vals = subject_time_day_pairs(
            vectors,
            int(subject),
            float(lock_time),
            float(time_center),
        )
        if len(emp_vals) < 6:
            continue
        d_subject_time = pd.DataFrame(day_rows)
        d_subject_time["distance"] = emp_vals
        for score in score_subject_time_rows(d_subject_time):
            rows.append(
                {
                    "active_pct": active_pct,
                    "subject": int(subject),
                    "lock_time": float(lock_time),
                    "time_center_sec": float(time_center),
                    "metric": "z_euclidean",
                    "n_edges": int(len(active_this)),
                    "n_day_pairs": int(len(emp_vals)),
                    **score,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No connectivity BIC subject rows produced")
    out = add_delta_bic(out)
    summary = summarize(out)

    pct_label = f"top{int(round(active_pct * 100)):02d}"
    subject_path = output_dir / f"connect_sensorwide_model_bic_timecourse_subject_{pct_label}.csv"
    summary_path = output_dir / f"connect_sensorwide_model_bic_timecourse_summary_{pct_label}.csv"
    out.to_csv(subject_path, index=False)
    summary.to_csv(summary_path, index=False)
    print(f"[connect model BIC] wrote {subject_path}", flush=True)
    print(f"[connect model BIC] wrote {summary_path}", flush=True)
    return {"subject": subject_path, "summary": summary_path}


if __name__ == "__main__":
    run_connect_sensorwide_model_bic_timecourse()
