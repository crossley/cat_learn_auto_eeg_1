#!/usr/bin/env python3
"""Bayesian pairwise model comparisons for connectivity timecourses."""

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

ACTIVE_PCT = float(os.environ.get("ACTIVE_PCT", "0.20"))
N_SAMPLES = int(os.environ.get("N_POSTERIOR_SAMPLES", "20000"))
RANDOM_STATE = int(os.environ.get("RANDOM_STATE", "42"))

CONTRASTS = [
    {
        "contrast": "mixed_D1_minus_gradual",
        "model_a": "two_stage_hybrid_D1",
        "model_b": "gradual",
    },
    {
        "contrast": "mixed_D1_minus_mixed_D2",
        "model_a": "two_stage_hybrid_D1",
        "model_b": "two_stage_hybrid_D2",
    },
    {
        "contrast": "binary_D1_minus_binary_D2",
        "model_a": "two_stage_binary_D1",
        "model_b": "two_stage_binary_D2",
    },
    {
        "contrast": "mixed_D1_minus_binary_D1",
        "model_a": "two_stage_hybrid_D1",
        "model_b": "two_stage_binary_D1",
    },
]


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing posterior pairwise input: {path}. "
            "Run connect_sensorwide_model_timecourse_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty posterior pairwise input: {path}")
    return d


def filter_active_pct(d):
    if "active_pct" not in d.columns:
        return d.copy()
    g = d[np.isclose(d["active_pct"].astype(float), ACTIVE_PCT)].copy()
    if g.empty:
        raise ValueError(f"Missing rows for active_pct={ACTIVE_PCT}")
    return g


def posterior_mean_samples(vals, rng):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        raise ValueError("Need at least two finite paired differences")
    mean = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1))
    sem = sd / np.sqrt(float(len(vals)))
    if sem <= np.finfo(float).eps:
        return np.full(N_SAMPLES, mean, dtype=float)
    draws = rng.standard_t(df=len(vals) - 1, size=N_SAMPLES)
    return mean + sem * draws


def model_wide(score_df):
    d = filter_active_pct(score_df)
    cols = [
        "subject",
        "time_center_sec",
        "model_label",
        "rho",
    ]
    d = d[cols].copy()
    wide = d.pivot_table(
        index=["subject", "time_center_sec"],
        columns="model_label",
        values="rho",
        aggfunc="mean",
    ).reset_index()
    if wide.empty:
        raise ValueError("No posterior pairwise model table could be built")
    return wide


def paired_differences(wide, model_a, model_b):
    if model_a not in wide.columns:
        raise ValueError(f"Missing model column: {model_a}")
    if model_b not in wide.columns:
        raise ValueError(f"Missing model column: {model_b}")
    rows = []
    for row in wide.itertuples(index=False):
        val_a = getattr(row, model_a)
        val_b = getattr(row, model_b)
        if np.isfinite(val_a) and np.isfinite(val_b):
            rows.append(
                {
                    "subject": int(row.subject),
                    "time_center_sec": float(row.time_center_sec),
                    "diff": float(val_a) - float(val_b),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No paired differences for {model_a} - {model_b}")
    return out


def summarize_differences(diff_df, contrast, model_a, model_b, rng):
    rows = []
    for time_center, g in diff_df.groupby("time_center_sec"):
        vals = g["diff"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        samples = posterior_mean_samples(vals, rng)
        rows.append(
            {
                "contrast": contrast,
                "model_a": model_a,
                "model_b": model_b,
                "active_pct": ACTIVE_PCT,
                "time_center_sec": float(time_center),
                "mean_diff": float(np.mean(vals)),
                "sem_diff": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                "ci_lower": float(np.quantile(samples, 0.025)),
                "ci_upper": float(np.quantile(samples, 0.975)),
                "p_gt0": float(np.mean(samples > 0.0)),
                "n_subjects": int(len(vals)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No posterior rows for contrast={contrast}")
    return out


def interval_summary(diff_df, contrast, model_a, model_b, rng):
    rows = []
    for window, bounds in MVPA_CAT_TG_WINDOWS.items():
        lo = float(bounds[0])
        hi = float(bounds[1])
        d_win = diff_df[
            (diff_df["time_center_sec"] >= lo) & (diff_df["time_center_sec"] <= hi)
        ]
        if d_win.empty:
            raise ValueError(f"No difference rows in interval: {window}")
        subject_vals = []
        for subject, g in d_win.groupby("subject"):
            vals = g["diff"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            subject_vals.append({"subject": int(subject), "diff": float(np.mean(vals))})
        vals = pd.DataFrame(subject_vals)["diff"].to_numpy(dtype=float)
        if len(vals) < 2:
            continue
        samples = posterior_mean_samples(vals, rng)
        rows.append(
            {
                "contrast": contrast,
                "model_a": model_a,
                "model_b": model_b,
                "active_pct": ACTIVE_PCT,
                "window": window,
                "window_start_sec": lo,
                "window_end_sec": hi,
                "mean_diff": float(np.mean(vals)),
                "sem_diff": float(np.std(vals, ddof=1) / np.sqrt(len(vals))),
                "ci_lower": float(np.quantile(samples, 0.025)),
                "ci_upper": float(np.quantile(samples, 0.975)),
                "p_gt0": float(np.mean(samples > 0.0)),
                "n_subjects": int(len(vals)),
            }
        )
    return rows


def run_connect_sensorwide_model_posterior_pairwise(
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    rng = np.random.default_rng(RANDOM_STATE)
    score_df = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_subject_scores.csv"
    )
    wide = model_wide(score_df)
    subject_rows = []
    time_rows = []
    interval_rows = []
    for spec in CONTRASTS:
        contrast = spec["contrast"]
        model_a = spec["model_a"]
        model_b = spec["model_b"]
        diff_df = paired_differences(wide, model_a, model_b)
        diff_df["contrast"] = contrast
        diff_df["model_a"] = model_a
        diff_df["model_b"] = model_b
        diff_df["active_pct"] = ACTIVE_PCT
        for row in diff_df.itertuples(index=False):
            subject_rows.append(row._asdict())
        summary_df = summarize_differences(
            diff_df,
            contrast,
            model_a,
            model_b,
            rng,
        )
        for row in summary_df.itertuples(index=False):
            time_rows.append(row._asdict())
        for row in interval_summary(diff_df, contrast, model_a, model_b, rng):
            interval_rows.append(row)

    subject_path = (
        output_dir / "connect_sensorwide_model_posterior_pairwise_subject.csv"
    )
    time_path = output_dir / "connect_sensorwide_model_posterior_pairwise_time.csv"
    interval_path = (
        output_dir / "connect_sensorwide_model_posterior_pairwise_intervals.csv"
    )
    pd.DataFrame(subject_rows).to_csv(subject_path, index=False)
    pd.DataFrame(time_rows).to_csv(time_path, index=False)
    pd.DataFrame(interval_rows).to_csv(interval_path, index=False)
    print(f"[connect posterior pairwise] wrote {subject_path}", flush=True)
    print(f"[connect posterior pairwise] wrote {time_path}", flush=True)
    print(f"[connect posterior pairwise] wrote {interval_path}", flush=True)
    return {
        "subject": subject_path,
        "time": time_path,
        "intervals": interval_path,
    }


if __name__ == "__main__":
    run_connect_sensorwide_model_posterior_pairwise()
