#!/usr/bin/env python3
"""Compare one-stage, two-stage, and sensory-stable 5x5 day models."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from connect_sensorwide_figure import (
    ACTIVE_PAIR_PEAK_WINDOWS,
    edge_vector_distance,
    matrix_offdiag_minmax_scaled,
)
from mvpa_stim_locked_cat_late_window_analysis import CLASSIFIERS
from util_rsa_time_resolved import GEOMETRY_WINDOWS

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DAYS = [1, 2, 3, 4, 5]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing model-comparison input: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty model-comparison input: {path}")
    return d


def pair_key(ch_i, ch_j):
    return f"{ch_i}\t{ch_j}"


def safe_adj_r2(r2, n_obs, n_params):
    denom = n_obs - n_params
    if denom <= 0:
        return np.nan
    return 1.0 - (1.0 - r2) * (n_obs - 1.0) / denom


def fit_ols_model(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    good = np.isfinite(y)
    for col_i in range(x.shape[1]):
        good = good & np.isfinite(x[:, col_i])
    y = y[good]
    x = x[good]
    n_obs = int(len(y))
    if n_obs == 0:
        raise ValueError("Cannot fit model with no finite observations")
    keep_cols = []
    for col_i in range(x.shape[1]):
        col = x[:, col_i]
        if float(np.nanmax(col) - np.nanmin(col)) > np.finfo(float).eps:
            keep_cols.append(col_i)
    if len(keep_cols) == 0:
        x = np.zeros((n_obs, 0), dtype=float)
    else:
        x_keep = np.zeros((n_obs, len(keep_cols)), dtype=float)
        for new_i, old_i in enumerate(keep_cols):
            x_keep[:, new_i] = x[:, old_i]
        x = x_keep
    intercept = np.ones((n_obs, 1), dtype=float)
    design = np.concatenate([intercept, x], axis=1)
    n_params = int(design.shape[1])
    beta, _resid, rank, _singular = np.linalg.lstsq(design, y, rcond=None)
    if int(rank) < n_params:
        raise ValueError("Singular model design in 5x5 model comparison")
    pred = design @ beta
    resid = y - pred
    rss = float(np.sum(resid**2))
    tss = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = np.nan
    if tss > np.finfo(float).eps:
        r2 = 1.0 - rss / tss
    sigma2 = max(rss / float(n_obs), np.finfo(float).eps)
    aic = float(n_obs * np.log(sigma2) + 2.0 * n_params)
    bic = float(n_obs * np.log(sigma2) + np.log(float(n_obs)) * n_params)
    return {
        "n_obs": n_obs,
        "n_params": n_params,
        "rss": rss,
        "r2": float(r2),
        "adj_r2": float(safe_adj_r2(r2, n_obs, n_params)),
        "aic": aic,
        "bic": bic,
        "beta": beta,
    }


def design_rows_for_model(model_family, split_day=None):
    rows = []
    for train_day in DAYS:
        for test_day in DAYS:
            if train_day == test_day:
                continue
            min_day = (min(train_day, test_day) - 1.0) / 4.0
            day_distance = abs(train_day - test_day) / 4.0
            cross_stage = np.nan
            if split_day is not None:
                train_late = train_day > split_day
                test_late = test_day > split_day
                cross_stage = 0.0
                if train_late != test_late:
                    cross_stage = 1.0
            x_vals = []
            if model_family == "sensory_stable":
                pass
            elif model_family == "one_stage":
                x_vals.append(float(min_day))
                x_vals.append(float(day_distance))
            elif model_family == "two_stage":
                x_vals.append(float(cross_stage))
                x_vals.append(float(min_day))
            else:
                raise ValueError(f"Unknown model family: {model_family}")
            rows.append(
                {
                    "train_day": train_day,
                    "test_day": test_day,
                    "x_vals": x_vals,
                }
            )
    return rows


def design_vals_for_pair(model_family, train_day, test_day, split_day=None):
    min_day = (min(train_day, test_day) - 1.0) / 4.0
    day_distance = abs(train_day - test_day) / 4.0
    cross_stage = np.nan
    if split_day is not None:
        train_late = train_day > split_day
        test_late = test_day > split_day
        cross_stage = 0.0
        if train_late != test_late:
            cross_stage = 1.0
    x_vals = []
    if model_family == "sensory_stable":
        pass
    elif model_family == "one_stage":
        x_vals.append(float(min_day))
        x_vals.append(float(day_distance))
    elif model_family == "two_stage":
        x_vals.append(float(cross_stage))
        x_vals.append(float(min_day))
    else:
        raise ValueError(f"Unknown model family: {model_family}")
    return x_vals


def model_specs():
    specs = []
    specs.append({"model_family": "sensory_stable", "split_day": np.nan})
    specs.append({"model_family": "one_stage", "split_day": np.nan})
    for split_day in [1, 2, 3, 4]:
        specs.append({"model_family": "two_stage", "split_day": split_day})
    return specs


def compare_models_for_subject(d_subject):
    y_vals = []
    pair_rows = []
    for train_day in DAYS:
        for test_day in DAYS:
            if train_day == test_day:
                continue
            g = d_subject[
                (d_subject["train_day"] == train_day)
                & (d_subject["test_day"] == test_day)
            ]
            if g.empty:
                continue
            val = float(g["value"].mean())
            y_vals.append(val)
            pair_rows.append({"train_day": train_day, "test_day": test_day})
    if len(y_vals) < 6:
        raise ValueError(
            "Too few empirical 5x5 off-diagonal values for subject-level "
            f"model comparison: n={len(y_vals)}"
        )

    scores = []
    for spec in model_specs():
        model_family = spec["model_family"]
        split_day = spec["split_day"]
        split_arg = None
        if np.isfinite(split_day):
            split_arg = int(split_day)
        x_rows = []
        for row in pair_rows:
            x_rows.append(
                design_vals_for_pair(
                    model_family,
                    int(row["train_day"]),
                    int(row["test_day"]),
                    split_arg,
                )
            )
        if len(x_rows) == 0:
            raise ValueError("No model-design rows")
        n_cols = 0
        if len(x_rows[0]) > 0:
            n_cols = len(x_rows[0])
        x = np.zeros((len(x_rows), n_cols), dtype=float)
        for row_i, x_vals in enumerate(x_rows):
            for col_i, val in enumerate(x_vals):
                x[row_i, col_i] = float(val)
        fit = fit_ols_model(y_vals, x)
        scores.append(
            {
                "model_family": model_family,
                "split_day": split_day,
                "n_obs": fit["n_obs"],
                "n_params": fit["n_params"],
                "rss": fit["rss"],
                "r2": fit["r2"],
                "adj_r2": fit["adj_r2"],
                "aic": fit["aic"],
                "bic": fit["bic"],
            }
        )
    return scores


def append_model_scores(rows, empirical_df):
    group_cols = [
        "modality",
        "measure",
        "window",
        "value_kind",
        "subject",
    ]
    for key, g in empirical_df.groupby(group_cols):
        modality, measure, window, value_kind, subject = key
        scores = compare_models_for_subject(g)
        for score in scores:
            row = {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "subject": int(subject),
            }
            for col, val in score.items():
                row[col] = val
            rows.append(row)


def mvpa_empirical_rows(output_dir):
    path = output_dir / "mvpa_stim_locked_cat_late_window_transfer_subject_pairs.csv"
    d = require_csv(path)
    rows = []
    for _, row in d.iterrows():
        if str(row["fit_status"]) not in ["transfer", "cv"]:
            continue
        if int(row["train_day"]) == int(row["test_day"]):
            continue
        classifier = str(row["classifier"])
        if classifier not in CLASSIFIERS:
            continue
        rows.append(
            {
                "modality": "mvpa",
                "measure": classifier,
                "window": str(row["window"]),
                "value_kind": "similarity",
                "subject": int(row["subject"]),
                "train_day": int(row["train_day"]),
                "test_day": int(row["test_day"]),
                "value": float(row["auc"]),
            }
        )
    return rows


def rsa_empirical_rows(output_dir, filename, measure):
    path = output_dir / filename
    d = require_csv(path)
    rows = []
    for window, bounds in GEOMETRY_WINDOWS.items():
        tmin, tmax = bounds
        g = d[(d["time_sec"] >= tmin) & (d["time_sec"] <= tmax)]
        if g.empty:
            raise ValueError(f"Missing RSA rows for {filename}, window={window}")
        group_cols = ["subject", "train_day", "test_day"]
        summary = g.groupby(group_cols, as_index=False).agg(value=("rho", "mean"))
        for _, row in summary.iterrows():
            train_day = int(row["train_day"])
            test_day = int(row["test_day"])
            if train_day == test_day:
                continue
            rows.append(
                {
                    "modality": "rsa",
                    "measure": measure,
                    "window": window,
                    "value_kind": "similarity",
                    "subject": int(row["subject"]),
                    "train_day": train_day,
                    "test_day": test_day,
                    "value": float(row["value"]),
                }
            )
    return rows


def sensorwide_group_context(output_dir):
    carpet_path = output_dir / "sensorwide_carpet_timeseries.csv"
    d = require_csv(carpet_path)
    d = d[(d["lock_type"] == "stim") & (d["band"] == "broadband")].copy()
    if d.empty:
        raise ValueError("Missing stim/broadband sensorwide group rows")
    d["pair_key"] = d["ch_i"].astype(str) + "\t" + d["ch_j"].astype(str)
    pivot = d.pivot_table(
        index=["day", "lock_time"],
        columns="pair_key",
        values="conn_val",
        aggfunc="mean",
    )
    pair_cols = []
    for col in pivot.columns:
        pair_cols.append(col)
    active_scores = []
    for col in pair_cols:
        vals = pivot[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        score = np.nan
        if len(vals) > 0:
            score = float(np.max(vals) - np.min(vals))
        active_scores.append(score)
    finite_scores = np.asarray(active_scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if len(finite_scores) == 0:
        raise ValueError("No finite sensorwide active-pair scores")
    top_k = max(1, int(np.ceil(0.20 * len(pair_cols))))
    threshold = float(np.sort(finite_scores)[-top_k])
    active_keys = []
    for idx, score in enumerate(active_scores):
        if np.isfinite(score) and score >= threshold:
            active_keys.append(pair_cols[idx])
    if len(active_keys) == 0:
        raise ValueError("No top-20% sensorwide active pairs selected")

    active_mean = pivot[active_keys].mean(axis=1)
    peak_rows = []
    for day in DAYS:
        d_day = active_mean.loc[day].sort_index()
        times = d_day.index.to_numpy(dtype=float)
        signal = d_day.to_numpy(dtype=float)
        for peak_i, window in enumerate(ACTIVE_PAIR_PEAK_WINDOWS, start=1):
            lo, hi = window
            idx_vals = []
            for idx, time_val in enumerate(times):
                if time_val >= lo and time_val <= hi and np.isfinite(signal[idx]):
                    idx_vals.append(idx)
            if len(idx_vals) == 0:
                raise ValueError(f"No sensorwide peak candidates for D{day}")
            best_idx = idx_vals[0]
            best_val = float(signal[best_idx])
            for idx in idx_vals:
                val = float(signal[idx])
                if val > best_val:
                    best_idx = idx
                    best_val = val
            peak_rows.append(
                {
                    "day": day,
                    "peak": peak_i,
                    "peak_time": float(times[best_idx]),
                }
            )
    active_edge_pos = {}
    for edge_pos, key in enumerate(active_keys):
        active_edge_pos[key] = int(edge_pos)
    return {
        "active_keys": active_keys,
        "active_key_set": set(active_keys),
        "active_edge_pos": active_edge_pos,
        "peak_rows": peak_rows,
    }


def sensorwide_peak_lookup(context):
    peak_lookup = {}
    time_lookup = {}
    for row in context["peak_rows"]:
        day = int(row["day"])
        peak = int(row["peak"])
        time_key = round(float(row["peak_time"]), 6)
        peak_lookup[(day, time_key)] = peak
        if day not in time_lookup:
            time_lookup[day] = set()
        time_lookup[day].add(time_key)
    return peak_lookup, time_lookup


def read_sensorwide_subject_chunks(path):
    usecols = [
        "subject",
        "day",
        "lock_type",
        "band",
        "lock_time",
        "ch_i",
        "ch_j",
        "conn_val",
    ]
    return pd.read_csv(path, usecols=usecols, chunksize=500000)


def connectivity_vectors(output_dir, context):
    subject_path = output_dir / "sensorwide_carpet_subject_timeseries.csv"
    if not subject_path.exists():
        raise FileNotFoundError(f"Missing sensorwide subject table: {subject_path}")
    peak_lookup, time_lookup = sensorwide_peak_lookup(context)
    n_edges = len(context["active_keys"])
    vector_map = {}
    for chunk_i, chunk in enumerate(read_sensorwide_subject_chunks(subject_path)):
        chunk = chunk[
            (chunk["lock_type"] == "stim") & (chunk["band"] == "broadband")
        ].copy()
        if chunk.empty:
            continue
        chunk["pair_key"] = (
            chunk["ch_i"].astype(str) + "\t" + chunk["ch_j"].astype(str)
        )
        chunk = chunk[chunk["pair_key"].isin(context["active_key_set"])].copy()
        if chunk.empty:
            continue
        keep = []
        peaks = []
        for _, row in chunk.iterrows():
            day = int(row["day"])
            time_key = round(float(row["lock_time"]), 6)
            peak = peak_lookup.get((day, time_key))
            keep_row = False
            if day in time_lookup and time_key in time_lookup[day]:
                keep_row = peak is not None
            keep.append(keep_row)
            peaks.append(peak)
        chunk["peak"] = peaks
        chunk = chunk[np.asarray(keep, dtype=bool)].copy()
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            subject = int(row["subject"])
            day = int(row["day"])
            peak = int(row["peak"])
            key = (subject, day, peak)
            if key not in vector_map:
                vector_map[key] = np.full(n_edges, np.nan, dtype=float)
            edge_pos = context["active_edge_pos"][row["pair_key"]]
            vector_map[key][edge_pos] = float(row["conn_val"])
        if (chunk_i + 1) % 10 == 0:
            print(
                f"[5x5 model comparison] sensorwide chunks {chunk_i + 1}",
                flush=True,
            )
    if len(vector_map) == 0:
        raise ValueError("No sensorwide subject peak vectors built")
    return vector_map


def connectivity_empirical_rows(output_dir):
    context = sensorwide_group_context(output_dir)
    vector_map = connectivity_vectors(output_dir, context)
    subjects = set()
    for subject, _day, _peak in vector_map.keys():
        subjects.add(subject)
    rows = []
    for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
        for subject in sorted(subjects):
            for train_day in DAYS:
                for test_day in DAYS:
                    if train_day == test_day:
                        continue
                    key_i = (subject, train_day, peak_i)
                    key_j = (subject, test_day, peak_i)
                    if key_i not in vector_map or key_j not in vector_map:
                        continue
                    dist = edge_vector_distance(
                        vector_map[key_i], vector_map[key_j], "z_euclidean"
                    )
                    if not np.isfinite(dist):
                        continue
                    rows.append(
                        {
                            "modality": "connectivity",
                            "measure": "z_euclidean_top20",
                            "window": f"peak{peak_i}",
                            "value_kind": "distance",
                            "subject": int(subject),
                            "train_day": train_day,
                            "test_day": test_day,
                            "value": float(dist),
                        }
                    )
    return rows


def write_group_matrices(empirical_df, output_dir):
    rows = []
    group_cols = ["modality", "measure", "window", "value_kind"]
    for key, g in empirical_df.groupby(group_cols):
        modality, measure, window, value_kind = key
        for train_day in DAYS:
            for test_day in DAYS:
                d_pair = g[
                    (g["train_day"] == train_day) & (g["test_day"] == test_day)
                ]
                val = np.nan
                n_subjects = 0
                if not d_pair.empty:
                    vals = d_pair["value"].to_numpy(dtype=float)
                    vals = vals[np.isfinite(vals)]
                    n_subjects = int(len(vals))
                    if len(vals) > 0:
                        val = float(np.mean(vals))
                rows.append(
                    {
                        "modality": modality,
                        "measure": measure,
                        "window": window,
                        "value_kind": value_kind,
                        "train_day": train_day,
                        "test_day": test_day,
                        "value_mean": val,
                        "n_subjects": n_subjects,
                    }
                )
    path = output_dir / "model_compare_5x5_group_matrices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def add_delta_scores(scores_df):
    rows = []
    group_cols = ["modality", "measure", "window", "value_kind", "subject"]
    for key, g in scores_df.groupby(group_cols):
        min_bic = float(g["bic"].min())
        min_aic = float(g["aic"].min())
        for _, row in g.iterrows():
            out = {}
            for col in scores_df.columns:
                out[col] = row[col]
            out["delta_bic"] = float(row["bic"]) - min_bic
            out["bic_support"] = min_bic - float(row["bic"])
            out["delta_aic"] = float(row["aic"]) - min_aic
            rows.append(out)
    return pd.DataFrame(rows)


def run_model_compare_5x5(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    empirical_rows = []

    print("[5x5 model comparison] loading MVPA transfer rows", flush=True)
    for row in mvpa_empirical_rows(output_dir):
        empirical_rows.append(row)

    print("[5x5 model comparison] loading RSA rows", flush=True)
    rsa_configs = [
        (
            "rsa_stim_cross_day_geometry_similarity.csv",
            "stim_time_resolved",
        ),
        (
            "rsa_stim_windowed_cross_day_geometry_similarity.csv",
            "stim_windowed",
        ),
    ]
    for filename, measure in rsa_configs:
        for row in rsa_empirical_rows(output_dir, filename, measure):
            empirical_rows.append(row)

    print("[5x5 model comparison] loading connectivity rows", flush=True)
    for row in connectivity_empirical_rows(output_dir):
        empirical_rows.append(row)

    empirical_df = pd.DataFrame(empirical_rows)
    if empirical_df.empty:
        raise ValueError("No empirical 5x5 rows collected")
    empirical_path = output_dir / "model_compare_5x5_empirical_values.csv"
    empirical_df.to_csv(empirical_path, index=False)

    print("[5x5 model comparison] fitting template models", flush=True)
    score_rows = []
    append_model_scores(score_rows, empirical_df)
    scores_df = add_delta_scores(pd.DataFrame(score_rows))
    scores_path = output_dir / "model_compare_5x5_scores.csv"
    scores_df.to_csv(scores_path, index=False)

    group_path = write_group_matrices(empirical_df, output_dir)
    print(f"[5x5 model comparison] wrote {empirical_path}", flush=True)
    print(f"[5x5 model comparison] wrote {scores_path}", flush=True)
    print(f"[5x5 model comparison] wrote {group_path}", flush=True)
    return {
        "empirical_values": empirical_path,
        "scores": scores_path,
        "group_matrices": group_path,
    }


if __name__ == "__main__":
    run_model_compare_5x5()
