#!/usr/bin/env python3
"""Shared helpers for latent trajectory day-geometry analyses."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAYS = [1, 2, 3, 4, 5]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty input table: {path}")
    return d


def z_euclidean(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(good)) < 2:
        return np.nan
    a = a[good]
    b = b[good]
    if np.std(a) <= np.finfo(float).eps or np.std(b) <= np.finfo(float).eps:
        return np.nan
    a = (a - float(np.mean(a))) / float(np.std(a))
    b = (b - float(np.mean(b))) / float(np.std(b))
    return float(np.sqrt(np.sum((a - b) ** 2)))


def model_distance(model, day_i, day_j, split_day=None):
    if model == "baseline":
        return 0.0
    if model == "gradual":
        return float(abs(day_i - day_j) / 4.0)
    if model == "discrete":
        if split_day is None:
            raise ValueError("discrete model requires split_day")
        gradual = float(abs(day_i - day_j) / 4.0)
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late == j_late:
            return 0.5 * gradual
        return 0.5 + 0.5 * gradual
    raise ValueError(f"Unknown model: {model}")


def model_specs():
    rows = [
        {"model_label": "Baseline", "model": "baseline", "split_day": np.nan},
        {"model_label": "Gradual", "model": "gradual", "split_day": np.nan},
    ]
    for split_day in [1, 2, 3, 4]:
        rows.append(
            {
                "model_label": f"Discrete D{split_day}",
                "model": "discrete",
                "split_day": float(split_day),
            }
        )
    return rows


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
    return {"bic": bic, "r2": r2, "n_obs": n_obs, "n_params": n_params}


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
                split_day,
            )
        )
    return np.column_stack([intercept, np.asarray(pred, dtype=float)])


def score_subject_day_distances(subject, distance_rows, analysis):
    day_rows = []
    y = []
    for row in distance_rows:
        day_rows.append({"day_i": int(row["day_i"]), "day_j": int(row["day_j"])})
        y.append(float(row["distance"]))
    rows = []
    for spec in model_specs():
        fit = fit_bic(y, design_matrix(day_rows, spec))
        rows.append(
            {
                "analysis": analysis,
                "subject": int(subject),
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
    for _key, g in score_df.groupby(["analysis", "subject"], dropna=False):
        g = g.copy()
        finite = g["bic"].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        g["delta_bic_best"] = np.nan if len(finite) == 0 else g["bic"] - np.min(finite)
        baseline = g[g["model_label"] == "Baseline"]
        if baseline.empty or not np.isfinite(float(baseline["bic"].iloc[0])):
            g["delta_bic_baseline"] = np.nan
        else:
            g["delta_bic_baseline"] = g["bic"] - float(baseline["bic"].iloc[0])
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def summarize_scores(score_df):
    rows = []
    group_cols = ["analysis", "model_label", "model", "split_day"]
    for key, g in score_df.groupby(group_cols, dropna=False):
        analysis, model_label, model, split_day = key
        delta_base = g["delta_bic_baseline"].to_numpy(float)
        delta_best = g["delta_bic_best"].to_numpy(float)
        r2_vals = g["r2"].to_numpy(float)
        delta_base = delta_base[np.isfinite(delta_base)]
        delta_best = delta_best[np.isfinite(delta_best)]
        r2_vals = r2_vals[np.isfinite(r2_vals)]
        rows.append(
            {
                "analysis": analysis,
                "model_label": model_label,
                "model": model,
                "split_day": split_day,
                "delta_bic_baseline_mean": float(np.mean(delta_base))
                if len(delta_base)
                else np.nan,
                "delta_bic_baseline_sem": sem(delta_base),
                "delta_bic_best_mean": float(np.mean(delta_best))
                if len(delta_best)
                else np.nan,
                "r2_mean": float(np.mean(r2_vals)) if len(r2_vals) else np.nan,
                "r2_sem": sem(r2_vals),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis", "delta_bic_best_mean"])


def trajectory_vectors(points_df, component_cols, time_col):
    vectors = {}
    for (subject, day), g in points_df.groupby(["subject", "day"]):
        g = g.sort_values(time_col)
        vectors[(int(subject), int(day))] = g[component_cols].to_numpy(float).ravel()
    return vectors


def score_trajectory_geometry(points_df, component_cols, time_col, analysis):
    vectors = trajectory_vectors(points_df, component_cols, time_col)
    subjects = sorted({subject for subject, _day in vectors})
    distance_rows = []
    score_rows = []
    for subject in subjects:
        subject_distances = []
        for day_i in DAYS:
            key_i = (subject, day_i)
            if key_i not in vectors:
                continue
            for day_j in DAYS:
                key_j = (subject, day_j)
                if day_j <= day_i or key_j not in vectors:
                    continue
                dist = z_euclidean(vectors[key_i], vectors[key_j])
                row = {
                    "analysis": analysis,
                    "subject": int(subject),
                    "day_i": int(day_i),
                    "day_j": int(day_j),
                    "distance": float(dist),
                }
                distance_rows.append(row)
                subject_distances.append(row)
        if len(subject_distances) >= 6:
            score_rows.extend(
                score_subject_day_distances(subject, subject_distances, analysis)
            )
    score_df = add_delta_bic(pd.DataFrame(score_rows))
    summary_df = summarize_scores(score_df)
    distance_df = pd.DataFrame(distance_rows)
    return distance_df, score_df, summary_df


def trajectory_metrics(points_df, component_cols, time_col, analysis):
    rows = []
    for (subject, day), g in points_df.groupby(["subject", "day"]):
        g = g.sort_values(time_col)
        x = g[component_cols].to_numpy(float)
        time = g[time_col].to_numpy(float)
        diffs = np.diff(x, axis=0)
        dt = np.diff(time)
        step = np.sqrt(np.sum(diffs**2, axis=1))
        speed = step / np.maximum(dt, np.finfo(float).eps)
        late = g[(g[time_col] >= 0.30) & (g[time_col] <= 0.60)]
        row = {
            "analysis": analysis,
            "subject": int(subject),
            "day": int(day),
            "path_length": float(np.nansum(step)),
            "mean_speed": float(np.nanmean(speed)),
            "start_norm": float(np.sqrt(np.nansum(x[0] ** 2))),
            "end_norm": float(np.sqrt(np.nansum(x[-1] ** 2))),
        }
        for comp in component_cols:
            row[f"{comp}_late_mean"] = float(np.nanmean(late[comp].to_numpy(float)))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["subject", "day"])


def group_distance_summary(distance_df):
    rows = []
    for key, g in distance_df.groupby(["analysis", "day_i", "day_j"]):
        analysis, day_i, day_j = key
        vals = g["distance"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "analysis": analysis,
                "day_i": int(day_i),
                "day_j": int(day_j),
                "distance_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "distance_sem": sem(vals),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["analysis", "day_i", "day_j"])
