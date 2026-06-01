#!/usr/bin/env python3
"""Compare late-subwindow MVPA transfer to the presentation model templates."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from mvpa_stim_locked_cat_late_window_analysis import OUTPUT_DIR
from mvpa_stim_locked_cat_late_subwindow_transfer_analysis import LATE_SUBWINDOWS

DAYS = [1, 2, 3, 4, 5]


def finite_corr(x, y):
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


def template_value(model, train_day, test_day, split_day=None):
    if model == "continuous":
        val = 0.65 * min(train_day, test_day) / float(max(DAYS))
        if train_day == test_day:
            val = train_day / float(max(DAYS))
        return float(val)
    if model == "discrete":
        if split_day is None:
            raise ValueError("discrete model requires split_day")
        train_late = train_day > split_day
        test_late = test_day > split_day
        if train_late != test_late:
            return 0.0
        val = 0.65 * min(train_day, test_day) / float(max(DAYS))
        if train_day == test_day:
            val = train_day / float(max(DAYS))
        return float(val)
    raise ValueError(f"Unknown presentation model: {model}")


def model_specs():
    rows = [
        {
            "model_label": "Continuous Restructuring",
            "model": "continuous",
            "split_day": np.nan,
        }
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


def model_vector(spec):
    split_day = None
    if np.isfinite(spec["split_day"]):
        split_day = int(spec["split_day"])
    vals = []
    for train_day in DAYS:
        for test_day in DAYS:
            vals.append(
                template_value(
                    spec["model"],
                    int(train_day),
                    int(test_day),
                    split_day=split_day,
                )
            )
    return np.asarray(vals, dtype=float)


def subject_matrix_vector(d_subject):
    vals = []
    for train_day in DAYS:
        for test_day in DAYS:
            g = d_subject[
                (d_subject["train_day"] == train_day)
                & (d_subject["test_day"] == test_day)
            ]
            if g.empty:
                vals.append(np.nan)
            else:
                vals.append(float(np.nanmean(g["auc"].to_numpy(float))))
    return np.asarray(vals, dtype=float)


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def run_mvpa_late_subwindow_presentation_model_compare(
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    score_rows = []
    for window, _start_sec, _end_sec in LATE_SUBWINDOWS:
        path = (
            output_dir
            / f"mvpa_stim_locked_cat_{window}_window_transfer_subject_pairs.csv"
        )
        if not path.exists():
            raise FileNotFoundError(
                f"Missing late-subwindow MVPA transfer output: {path}"
            )
        d = pd.read_csv(path)
        if d.empty:
            raise ValueError(f"Empty late-subwindow MVPA transfer output: {path}")
        for (classifier, subject), g in d.groupby(["classifier", "subject"]):
            empirical = subject_matrix_vector(g)
            for spec in model_specs():
                score_rows.append(
                    {
                        "classifier": classifier,
                        "subject": int(subject),
                        "window": window,
                        "model_label": spec["model_label"],
                        "model": spec["model"],
                        "split_day": spec["split_day"],
                        "rho": finite_corr(empirical, model_vector(spec)),
                    }
                )
    score_df = pd.DataFrame(score_rows)
    summary_rows = []
    for key, g in score_df.groupby(["classifier", "window", "model_label"]):
        classifier, window, model_label = key
        vals = g["rho"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        summary_rows.append(
            {
                "classifier": classifier,
                "window": window,
                "model_label": model_label,
                "rho_mean": float(np.mean(vals)),
                "rho_sem": sem(vals),
                "n_subjects": int(len(vals)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["classifier", "window", "rho_mean"],
        ascending=[True, True, False],
    )

    score_path = (
        output_dir / "mvpa_late_subwindow_presentation_model_subject_scores.csv"
    )
    summary_path = (
        output_dir / "mvpa_late_subwindow_presentation_model_summary.csv"
    )
    score_df.to_csv(score_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f"[MVPA late-subwindow presentation models] wrote {score_path}")
    print(f"[MVPA late-subwindow presentation models] wrote {summary_path}")
    return {"subject_scores": score_path, "summary": summary_path}


if __name__ == "__main__":
    run_mvpa_late_subwindow_presentation_model_compare()
