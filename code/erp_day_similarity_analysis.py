#!/usr/bin/env python3
"""Day-by-day stimulus ERP and GFP similarity with 5x5 template model scores."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from model_compare_5x5_analysis import DAYS, fit_ols_model, model_specs, design_vals_for_pair

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

WINDOWS = {
    "full_000_800": (0.00, 0.80),
    "early_060_180": (0.06, 0.18),
    "late_300_600": (0.30, 0.60),
}


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ERP day-similarity input: {path}. "
            "Run erp_grand_average_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty ERP day-similarity input: {path}")
    return d


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


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def subject_day_erp_vectors(d, lo, hi):
    d_win = d[(d["time_s"] >= lo) & (d["time_s"] <= hi)].copy()
    if d_win.empty:
        raise ValueError(f"No ERP rows in window {lo}-{hi} s")
    channels = sorted(d_win["channel"].dropna().unique().tolist())
    times = sorted(d_win["time_s"].dropna().unique().tolist())
    vectors = {}
    for key, g in d_win.groupby(["subject", "day"]):
        subject, day = key
        pivot = g.pivot_table(
            index="channel",
            columns="time_s",
            values="amplitude_v",
            aggfunc="mean",
        )
        pivot = pivot.reindex(index=channels, columns=times)
        if pivot.isna().any().any():
            continue
        vectors[(int(subject), int(day))] = pivot.to_numpy(dtype=float).ravel()
    return vectors


def subject_day_gfp_vectors(d, lo, hi):
    d_win = d[(d["time_s"] >= lo) & (d["time_s"] <= hi)].copy()
    if d_win.empty:
        raise ValueError(f"No GFP rows in window {lo}-{hi} s")
    rows = []
    for key, g in d_win.groupby(["subject", "day", "time_s"]):
        subject, day, time_s = key
        vals = g["amplitude_v"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "time_s": float(time_s),
                "gfp": float(np.std(vals, ddof=1)),
            }
        )
    gfp = pd.DataFrame(rows)
    if gfp.empty:
        raise ValueError("Could not compute subject-day GFP vectors")
    times = sorted(gfp["time_s"].dropna().unique().tolist())
    vectors = {}
    for key, g in gfp.groupby(["subject", "day"]):
        subject, day = key
        s = g.set_index("time_s").reindex(times)["gfp"]
        if s.isna().any():
            continue
        vectors[(int(subject), int(day))] = s.to_numpy(dtype=float)
    return vectors


def complete_subjects(vectors):
    subjects = sorted({subject for subject, _day in vectors})
    out = []
    for subject in subjects:
        if all((subject, day) in vectors for day in DAYS):
            out.append(subject)
    return out


def empirical_rows_for_vectors(vectors, modality, measure, window):
    rows = []
    for subject in complete_subjects(vectors):
        for train_day in DAYS:
            for test_day in DAYS:
                val = vector_corr(vectors[(subject, train_day)], vectors[(subject, test_day)])
                rows.append(
                    {
                        "modality": modality,
                        "measure": measure,
                        "window": window,
                        "value_kind": "similarity",
                        "subject": int(subject),
                        "train_day": int(train_day),
                        "test_day": int(test_day),
                        "value": float(val),
                    }
                )
    return rows


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
            if np.isfinite(val):
                y_vals.append(val)
                pair_rows.append({"train_day": train_day, "test_day": test_day})
    if len(y_vals) < 6:
        raise ValueError(f"Too few ERP day-pair values: n={len(y_vals)}")
    rows = []
    for spec in model_specs():
        split_day = spec["split_day"]
        split_arg = int(split_day) if np.isfinite(split_day) else None
        x_rows = [
            design_vals_for_pair(
                spec["model_family"],
                int(row["train_day"]),
                int(row["test_day"]),
                split_arg,
            )
            for row in pair_rows
        ]
        n_cols = len(x_rows[0]) if x_rows and x_rows[0] else 0
        x = np.zeros((len(x_rows), n_cols), dtype=float)
        for row_i, x_vals in enumerate(x_rows):
            for col_i, val in enumerate(x_vals):
                x[row_i, col_i] = float(val)
        fit = fit_ols_model(y_vals, x)
        rows.append(
            {
                "model_family": spec["model_family"],
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
    return rows


def score_models(empirical_df):
    rows = []
    group_cols = ["modality", "measure", "window", "value_kind", "subject"]
    for key, g in empirical_df.groupby(group_cols):
        modality, measure, window, value_kind, subject = key
        scores = compare_models_for_subject(g)
        min_bic = min(float(score["bic"]) for score in scores)
        min_aic = min(float(score["aic"]) for score in scores)
        for score in scores:
            row = {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "subject": int(subject),
            }
            row.update(score)
            row["delta_bic"] = float(score["bic"]) - min_bic
            row["bic_support"] = min_bic - float(score["bic"])
            row["delta_aic"] = float(score["aic"]) - min_aic
            rows.append(row)
    return pd.DataFrame(rows)


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
                vals = d_pair["value"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                rows.append(
                    {
                        "modality": modality,
                        "measure": measure,
                        "window": window,
                        "value_kind": value_kind,
                        "train_day": int(train_day),
                        "test_day": int(test_day),
                        "value_mean": float(np.mean(vals)) if len(vals) else np.nan,
                        "value_sem": sem(vals),
                        "n_subjects": int(len(vals)),
                    }
                )
    path = output_dir / "erp_day_similarity_group_matrices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_erp_day_similarity(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "erp_grand_average_subject_day_all.csv")
    d = d[(d["lock_type"] == "stim") & (d["condition"] == "all")].copy()
    if d.empty:
        raise ValueError("No stim/all rows in ERP subject-day output")

    rows = []
    for window, (lo, hi) in WINDOWS.items():
        erp_vectors = subject_day_erp_vectors(d, lo, hi)
        rows.extend(
            empirical_rows_for_vectors(
                erp_vectors, "erp", "channel_time_correlation", window
            )
        )
        gfp_vectors = subject_day_gfp_vectors(d, lo, hi)
        rows.extend(
            empirical_rows_for_vectors(
                gfp_vectors, "erp_gfp", "gfp_time_correlation", window
            )
        )

    empirical_df = pd.DataFrame(rows)
    if empirical_df.empty:
        raise ValueError("No ERP day-similarity rows computed")
    empirical_path = output_dir / "erp_day_similarity_empirical_values.csv"
    scores_path = output_dir / "erp_day_similarity_model_scores.csv"
    empirical_df.to_csv(empirical_path, index=False)
    scores_df = score_models(empirical_df[empirical_df["train_day"] != empirical_df["test_day"]])
    scores_df.to_csv(scores_path, index=False)
    group_path = write_group_matrices(empirical_df, output_dir)
    print(f"[ERP day similarity] wrote {empirical_path}", flush=True)
    print(f"[ERP day similarity] wrote {scores_path}", flush=True)
    print(f"[ERP day similarity] wrote {group_path}", flush=True)
    return {
        "empirical_values": empirical_path,
        "scores": scores_path,
        "group_matrices": group_path,
    }


if __name__ == "__main__":
    run_erp_day_similarity()
