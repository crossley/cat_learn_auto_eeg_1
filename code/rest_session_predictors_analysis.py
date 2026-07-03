#!/usr/bin/env python3
"""Same-day task outcomes predicted by pre-task resting-state summaries."""

from __future__ import annotations

from pathlib import Path
import json
import os
import time

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from figure_style import OUTPUT_DIR
from load_project_data import load_sessions
from sensor_rois import STRICT_SENSOR_ROIS, cross_roi_pairs

PREDICTORS = [
    "posterior_alpha_power",
    "frontal_theta_power",
    "global_rest_connectivity",
    "visual_central_rest_connectivity",
    "visual_frontal_rest_connectivity",
]

OUTCOMES = [
    "accuracy",
    "median_rt_correct",
    "learning_slope",
    "early_accuracy",
    "late_accuracy",
    "mvpa_late_auc",
    "mvpa_peak_auc",
    "task_connectivity_late_contrast",
    "erp_gfp_late",
]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input table: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty input table: {path}")
    return d


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def pair_label(pair):
    a, b = tuple(pair)
    return f"{a}--{b}"


def mean_feature(d, feature_kind, band, features=None):
    g = d[(d["feature_kind"] == feature_kind) & (d["band"] == band)].copy()
    if features is not None:
        g = g[g["feature"].isin(features)].copy()
    vals = g["value"].to_numpy(float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    return float(np.mean(vals))


def rest_predictor_rows(rest_df):
    visual_central = {pair_label(p) for p in cross_roi_pairs("visual", "central")}
    visual_frontal = {pair_label(p) for p in cross_roi_pairs("visual", "frontal")}
    rows = []
    for (subject, day), g in rest_df.groupby(["subject", "day"]):
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "posterior_alpha_power": mean_feature(
                    g,
                    "spectral",
                    "alpha",
                    STRICT_SENSOR_ROIS["visual"],
                ),
                "frontal_theta_power": mean_feature(
                    g,
                    "spectral",
                    "theta",
                    STRICT_SENSOR_ROIS["frontal"],
                ),
                "global_rest_connectivity": mean_feature(
                    g,
                    "connectivity",
                    "alpha",
                    None,
                ),
                "visual_central_rest_connectivity": mean_feature(
                    g,
                    "connectivity",
                    "alpha",
                    visual_central,
                ),
                "visual_frontal_rest_connectivity": mean_feature(
                    g,
                    "connectivity",
                    "alpha",
                    visual_frontal,
                ),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No resting-state predictor rows were produced")
    return out


def behaviour_rows():
    rows = []
    for item in load_sessions(load_epochs=False):
        subject = int(item["subject"])
        day = int(item["day"])
        beh = item["beh"].sort_values("trial").reset_index(drop=True).copy()
        correct = beh["fb"].astype(str).str.lower().to_numpy() == "correct"
        rt = pd.to_numeric(beh["rt"], errors="coerce").to_numpy(float)
        early_idx = np.arange(len(beh)) < int(np.ceil(0.2 * len(beh)))
        late_idx = np.arange(len(beh)) >= int(np.floor(0.8 * len(beh)))
        trial = np.arange(len(beh), dtype=float)
        learning_slope = np.nan
        if np.std(trial) > 0 and np.std(correct.astype(float)) > 0:
            learning_slope = float(np.corrcoef(trial, correct.astype(float))[0, 1])
        rows.append(
            {
                "subject": subject,
                "day": day,
                "accuracy": float(np.mean(correct)),
                "median_rt_correct": float(np.nanmedian(rt[correct])),
                "learning_slope": learning_slope,
                "early_accuracy": float(np.mean(correct[early_idx])),
                "late_accuracy": float(np.mean(correct[late_idx])),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No behavioural outcome rows were produced")
    return out


def mvpa_rows(output_dir):
    path = Path(output_dir) / "mvpa_stim_locked_cat_subject_day_timecourse.csv"
    if not path.exists():
        return pd.DataFrame(columns=["subject", "day", "mvpa_late_auc", "mvpa_peak_auc"])
    d = require_csv(path)
    rows = []
    for (subject, day), g in d.groupby(["subject", "day"]):
        late = g[(g["time_sec"] >= 0.30) & (g["time_sec"] <= 0.60)]
        task = g[(g["time_sec"] >= 0.00) & (g["time_sec"] <= 0.80)]
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "mvpa_late_auc": float(np.nanmean(late["auc"].to_numpy(float))),
                "mvpa_peak_auc": float(np.nanmax(task["auc"].to_numpy(float))),
            }
        )
    return pd.DataFrame(rows)


def task_connectivity_rows(output_dir):
    path = Path(output_dir) / "connect_roi_timecourse_25ms_contrast_subject.csv"
    if not path.exists():
        path = Path(output_dir) / "connect_roi_timecourse_contrast_subject.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=["subject", "day", "task_connectivity_late_contrast"]
        )
    d = require_csv(path)
    rows = []
    for (subject, day), g in d.groupby(["subject", "day"]):
        late = g[(g["lock_time"] >= 0.30) & (g["lock_time"] <= 0.60)]
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "task_connectivity_late_contrast": float(
                    np.nanmean(late["contrast"].to_numpy(float))
                ),
            }
        )
    return pd.DataFrame(rows)


def erp_gfp_rows(output_dir):
    path = Path(output_dir) / "erp_grand_average_subject_day_all.csv"
    if not path.exists():
        return pd.DataFrame(columns=["subject", "day", "erp_gfp_late"])
    d = require_csv(path)
    d = d[
        (d["lock_type"] == "stim")
        & (d["condition"] == "all")
        & (d["time_s"] >= 0.30)
        & (d["time_s"] <= 0.60)
    ].copy()
    rows = []
    for (subject, day, time_s), g in d.groupby(["subject", "day", "time_s"]):
        amp = g["amplitude_v"].to_numpy(float)
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "time_s": float(time_s),
                "gfp": float(np.sqrt(np.nanmean(amp**2))),
            }
        )
    gfp = pd.DataFrame(rows)
    if gfp.empty:
        return pd.DataFrame(columns=["subject", "day", "erp_gfp_late"])
    out = (
        gfp.groupby(["subject", "day"], as_index=False)
        .agg(erp_gfp_late=("gfp", "mean"))
        .sort_values(["subject", "day"])
    )
    return out


def pearson_summary(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    x = x[good]
    y = y[good]
    n = int(len(x))
    if n < 4 or np.std(x) <= np.finfo(float).eps or np.std(y) <= np.finfo(float).eps:
        return {"r": np.nan, "n": n}
    return {"r": float(np.corrcoef(x, y)[0, 1]), "n": n}


def subject_center(d, cols):
    out = d.copy()
    for col in cols:
        out[col] = out[col] - out.groupby("subject")[col].transform("mean")
    return out


def correlation_rows(session_df):
    centered = subject_center(session_df, PREDICTORS + OUTCOMES)
    rows = []
    for predictor in PREDICTORS:
        for outcome in OUTCOMES:
            raw = pearson_summary(session_df[predictor], session_df[outcome])
            within = pearson_summary(centered[predictor], centered[outcome])
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "correlation_type": "raw",
                    **raw,
                }
            )
            rows.append(
                {
                    "predictor": predictor,
                    "outcome": outcome,
                    "correlation_type": "subject_centered",
                    **within,
                }
            )
    return pd.DataFrame(rows)


def run_rest_session_predictors(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "rest_session_predictors_progress.json"
    t0 = time.time()
    progress_path.write_text(json.dumps({"stage": "loading"}, indent=2))

    rest = require_csv(output_dir / "rest_day_features.csv")
    predictors = rest_predictor_rows(rest)
    outcomes = behaviour_rows()
    for extra in [
        mvpa_rows(output_dir),
        task_connectivity_rows(output_dir),
        erp_gfp_rows(output_dir),
    ]:
        outcomes = outcomes.merge(extra, on=["subject", "day"], how="left")

    session_df = predictors.merge(outcomes, on=["subject", "day"], how="inner")
    if session_df.empty:
        raise ValueError("No matched resting predictor and task outcome rows")
    corr_df = correlation_rows(session_df)

    session_path = output_dir / "rest_session_predictors_table.csv"
    corr_path = output_dir / "rest_session_predictor_correlations.csv"
    session_df.to_csv(session_path, index=False)
    corr_df.to_csv(corr_path, index=False)
    progress_path.write_text(
        json.dumps(
            {
                "stage": "complete",
                "elapsed_sec": time.time() - t0,
                "n_sessions": int(len(session_df)),
                "n_subjects": int(session_df["subject"].nunique()),
            },
            indent=2,
        )
    )
    print(f"[rest predictors] wrote {session_path}", flush=True)
    print(f"[rest predictors] wrote {corr_path}", flush=True)
    return session_path, corr_path


if __name__ == "__main__":
    run_rest_session_predictors()
