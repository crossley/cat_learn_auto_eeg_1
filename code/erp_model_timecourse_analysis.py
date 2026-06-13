#!/usr/bin/env python3
"""Time-resolved model evidence for cross-day stimulus ERP similarity."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DAYS = [1, 2, 3, 4, 5]
WINDOW_WIDTH_SEC = 0.050
WINDOW_STEP_SEC = 0.025


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ERP model-timecourse input: {path}. "
            "Run erp_grand_average_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty ERP model-timecourse input: {path}")
    return d


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def vector_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 3:
        return np.nan
    x = x[good] - float(np.mean(x[good]))
    y = y[good] - float(np.mean(y[good]))
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x * y) / denom)


def template_value(model, day_i, day_j, split_day=None):
    if model == "baseline":
        return 0.0
    if model == "continuous":
        return float(0.65 * min(day_i, day_j) / float(max(DAYS)))
    if model == "discrete":
        if split_day is None:
            raise ValueError("discrete model requires split_day")
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late != j_late:
            return 0.0
        return float(0.65 * min(day_i, day_j) / float(max(DAYS)))
    raise ValueError(f"Unknown model: {model}")


def model_specs():
    rows = [
        {"model_label": "Baseline", "model": "baseline", "split_day": np.nan},
        {
            "model_label": "Continuous Restructuring",
            "model": "continuous",
            "split_day": np.nan,
        },
    ]
    for split_day in [1, 2, 3, 4]:
        rows.append(
            {
                "model_label": f"Discrete Restructuring D{split_day}",
                "model": "discrete",
                "split_day": float(split_day),
            }
        )
    return rows


def design_matrix(pair_rows, spec):
    intercept = np.ones(len(pair_rows), dtype=float)
    if spec["model"] == "baseline":
        return intercept.reshape(-1, 1)
    split_day = int(spec["split_day"]) if np.isfinite(spec["split_day"]) else None
    pred = [
        template_value(spec["model"], int(row["day_i"]), int(row["day_j"]), split_day)
        for row in pair_rows
    ]
    return np.column_stack([intercept, np.asarray(pred, dtype=float)])


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


def subject_day_arrays(d):
    channels = sorted(d["channel"].dropna().unique().tolist())
    times = sorted(d["time_s"].dropna().unique().tolist())
    arrays = {}
    for key, g in d.groupby(["subject", "day"]):
        subject, day = key
        pivot = g.pivot_table(
            index="channel",
            columns="time_s",
            values="amplitude_v",
            aggfunc="mean",
        ).reindex(index=channels, columns=times)
        if pivot.isna().any().any():
            continue
        arrays[(int(subject), int(day))] = pivot.to_numpy(dtype=float)
    return arrays, np.asarray(times, dtype=float)


def score_subject_center(subject, center, keep_time, arrays):
    pair_rows = []
    y_vals = []
    for day_i in DAYS:
        for day_j in DAYS:
            if day_i >= day_j:
                continue
            key_i = (subject, day_i)
            key_j = (subject, day_j)
            if key_i not in arrays or key_j not in arrays:
                continue
            vec_i = arrays[key_i][:, keep_time].ravel()
            vec_j = arrays[key_j][:, keep_time].ravel()
            val = vector_corr(vec_i, vec_j)
            if not np.isfinite(val):
                continue
            pair_rows.append({"day_i": day_i, "day_j": day_j})
            y_vals.append(float(val))
    if len(y_vals) < 6:
        return []
    rows = []
    for spec in model_specs():
        fit = fit_bic(y_vals, design_matrix(pair_rows, spec))
        rows.append(
            {
                "subject": int(subject),
                "time_center_sec": float(center),
                "window_start_sec": float(center - WINDOW_WIDTH_SEC / 2.0),
                "window_end_sec": float(center + WINDOW_WIDTH_SEC / 2.0),
                "model_label": spec["model_label"],
                "model": spec["model"],
                "split_day": spec["split_day"],
                "bic": fit["bic"],
                "r2": fit["r2"],
                "n_obs": fit["n_obs"],
                "n_params": fit["n_params"],
            }
        )
    return rows


def add_delta_bic(score_df):
    frames = []
    for _key, g in score_df.groupby(["subject", "time_center_sec"], dropna=False):
        g = g.copy()
        finite = g["bic"].to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        g["delta_bic_best"] = np.nan if len(finite) == 0 else g["bic"] - float(np.min(finite))
        baseline = g[g["model_label"] == "Baseline"]
        if baseline.empty or not np.isfinite(float(baseline["bic"].iloc[0])):
            g["delta_bic_baseline"] = np.nan
        else:
            g["delta_bic_baseline"] = g["bic"] - float(baseline["bic"].iloc[0])
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def summarize(score_df):
    rows = []
    group_cols = ["time_center_sec", "window_start_sec", "window_end_sec", "model_label", "model", "split_day"]
    for key, g in score_df.groupby(group_cols, dropna=False):
        key_vals = dict(zip(group_cols, key))
        delta_base = g["delta_bic_baseline"].to_numpy(dtype=float)
        delta_best = g["delta_bic_best"].to_numpy(dtype=float)
        r2_vals = g["r2"].to_numpy(dtype=float)
        delta_base = delta_base[np.isfinite(delta_base)]
        delta_best = delta_best[np.isfinite(delta_best)]
        r2_vals = r2_vals[np.isfinite(r2_vals)]
        rows.append(
            {
                **key_vals,
                "delta_bic_baseline_mean": float(np.mean(delta_base)) if len(delta_base) else np.nan,
                "delta_bic_baseline_sem": sem(delta_base),
                "delta_bic_best_mean": float(np.mean(delta_best)) if len(delta_best) else np.nan,
                "r2_mean": float(np.mean(r2_vals)) if len(r2_vals) else np.nan,
                "r2_sem": sem(r2_vals),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["time_center_sec", "delta_bic_best_mean"])


def run_erp_model_timecourse(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    d = require_csv(output_dir / "erp_grand_average_subject_day_all.csv")
    d = d[(d["lock_type"] == "stim") & (d["condition"] == "all")].copy()
    if d.empty:
        raise ValueError("No stim/all ERP rows available")
    arrays, times = subject_day_arrays(d)
    subjects = sorted({subject for subject, _day in arrays})
    centers = np.arange(
        float(times.min()) + WINDOW_WIDTH_SEC / 2.0,
        float(times.max()) - WINDOW_WIDTH_SEC / 2.0 + WINDOW_STEP_SEC / 2.0,
        WINDOW_STEP_SEC,
    )
    rows = []
    for center_i, center in enumerate(centers, start=1):
        keep_time = (times >= center - WINDOW_WIDTH_SEC / 2.0) & (
            times <= center + WINDOW_WIDTH_SEC / 2.0
        )
        if int(np.sum(keep_time)) < 2:
            continue
        for subject in subjects:
            rows.extend(score_subject_center(subject, float(center), keep_time, arrays))
        if (center_i % 10) == 0:
            print(f"[ERP model timecourse] centers {center_i}/{len(centers)}", flush=True)
    score_df = add_delta_bic(pd.DataFrame(rows))
    summary_df = summarize(score_df)
    subject_path = output_dir / "erp_model_timecourse_subject.csv"
    summary_path = output_dir / "erp_model_timecourse_summary.csv"
    score_df.to_csv(subject_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[ERP model timecourse] wrote {subject_path}", flush=True)
    print(f"[ERP model timecourse] wrote {summary_path}", flush=True)
    return {"subject": subject_path, "summary": summary_path}


if __name__ == "__main__":
    run_erp_model_timecourse()
