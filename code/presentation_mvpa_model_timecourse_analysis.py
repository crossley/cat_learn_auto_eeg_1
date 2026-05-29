#!/usr/bin/env python3
"""Compute presentation MVPA transfer-template correlations across time."""

from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from mvpa_transfer_model_compare_analysis import DAYS, finite_corr

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
MATRIX_RE = re.compile(
    r"^mvpa_stim_locked_cat_tg_matrix_sub_(\d+)_trainD(\d+)_testD(\d+)\.npz$"
)


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing presentation MVPA input: {path}. "
            "Run mvpa_stim_locked_cat_tg_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty presentation MVPA input: {path}")
    return d


def model_specs():
    rows = []
    rows.append(
        {
            "model": "gradual",
            "split_day": np.nan,
            "label": "gradual",
        }
    )
    for model in ["split_gradual", "split_binary"]:
        for split_day in [1, 2, 3, 4]:
            label = f"{model}_D{split_day}"
            rows.append(
                {
                    "model": model,
                    "split_day": split_day,
                    "label": label,
                }
            )
    return rows


def template_value(model, train_day, test_day, split_day=None):
    if model == "gradual":
        val = 0.65 * min(train_day, test_day) / float(max(DAYS))
        if train_day == test_day:
            val = train_day / float(max(DAYS))
        return float(val)
    if model == "split_gradual":
        if split_day is None:
            raise ValueError("split_gradual requires split_day")
        train_late = train_day > split_day
        test_late = test_day > split_day
        if train_late != test_late:
            return 0.0
        val = 0.65 * min(train_day, test_day) / float(max(DAYS))
        if train_day == test_day:
            val = train_day / float(max(DAYS))
        return float(val)
    if model == "split_binary":
        if split_day is None:
            raise ValueError("split_binary requires split_day")
        train_late = train_day > split_day
        test_late = test_day > split_day
        if train_late == test_late:
            return 1.0
        return 0.0
    raise ValueError(f"Unknown presentation MVPA model: {model}")


def template_vector(spec, pairs):
    vals = []
    split_day = None
    if np.isfinite(float(spec["split_day"])):
        split_day = int(spec["split_day"])
    for pair in pairs:
        vals.append(
            template_value(
                str(spec["model"]),
                int(pair["train_day"]),
                int(pair["test_day"]),
                split_day=split_day,
            )
        )
    return np.asarray(vals, dtype=float)


def zscore(vals):
    vals = np.asarray(vals, dtype=float)
    mean_val = float(np.nanmean(vals))
    std_val = float(np.nanstd(vals))
    if std_val <= np.finfo(float).eps:
        return np.full(vals.shape, np.nan, dtype=float)
    return (vals - mean_val) / std_val


def model_corr(auc_vals, pred):
    return finite_corr(zscore(auc_vals), zscore(pred))


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def score_time_rows(pair_rows, auc_vals):
    auc_vals = np.asarray(auc_vals, dtype=float)
    rows = []
    for spec in model_specs():
        pred = template_vector(spec, pair_rows)
        rows.append(
            {
                "model": str(spec["model"]),
                "split_day": spec["split_day"],
                "model_label": str(spec["label"]),
                "rho": model_corr(auc_vals, pred),
            }
        )
    return rows


def score_group_time(d_time):
    pairs = []
    auc_vals = []
    for row in d_time.itertuples(index=False):
        pairs.append(
            {
                "train_day": int(row.train_day),
                "test_day": int(row.test_day),
            }
        )
        auc_vals.append(float(row.auc_mean))
    return score_time_rows(pairs, auc_vals)


def matrix_file_rows(output_dir):
    rows = []
    for path in sorted(output_dir.glob("mvpa_stim_locked_cat_tg_matrix_sub_*.npz")):
        match = MATRIX_RE.match(path.name)
        if match is None:
            continue
        rows.append(
            {
                "subject": int(match.group(1)),
                "train_day": int(match.group(2)),
                "test_day": int(match.group(3)),
                "path": path,
            }
        )
    if len(rows) == 0:
        raise FileNotFoundError(
            f"No MVPA TG matrix files found in {output_dir}. "
            "Run mvpa_stim_locked_cat_tg_analysis.py first."
        )
    return rows


def subject_timecourse_rows(output_dir):
    file_rows = matrix_file_rows(output_dir)
    by_subject = {}
    for row in file_rows:
        subject = int(row["subject"])
        if subject not in by_subject:
            by_subject[subject] = []
        by_subject[subject].append(row)

    rows = []
    for subject in sorted(by_subject.keys()):
        pair_data = []
        times_ref = None
        for row in by_subject[subject]:
            data = np.load(row["path"])
            auc = data["auc"]
            times = data["time_sec"]
            if times_ref is None:
                times_ref = np.asarray(times, dtype=float)
            if len(times) != len(times_ref):
                raise ValueError(f"Time length mismatch in {row['path']}")
            diag_vals = np.diag(auc)
            pair_data.append(
                {
                    "train_day": int(row["train_day"]),
                    "test_day": int(row["test_day"]),
                    "diag_vals": np.asarray(diag_vals, dtype=float),
                }
            )
        if times_ref is None:
            continue
        for time_idx, time_sec in enumerate(times_ref):
            pair_rows = []
            auc_vals = []
            for pair in pair_data:
                val = float(pair["diag_vals"][time_idx])
                if not np.isfinite(val):
                    continue
                pair_rows.append(
                    {
                        "train_day": int(pair["train_day"]),
                        "test_day": int(pair["test_day"]),
                    }
                )
                auc_vals.append(val)
            if len(auc_vals) < 10:
                continue
            for score in score_time_rows(pair_rows, auc_vals):
                score["subject"] = int(subject)
                score["time_sec"] = float(time_sec)
                score["n_pairs"] = int(len(auc_vals))
                rows.append(score)
    if len(rows) == 0:
        raise ValueError("No subject-level MVPA presentation rows computed")
    return pd.DataFrame(rows)


def group_summary(subject_df):
    rows = []
    group_cols = ["model", "split_day", "model_label", "time_sec"]
    for key, g in subject_df.groupby(group_cols, dropna=False):
        model, split_day, model_label, time_sec = key
        vals = g["rho"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "model": str(model),
                "split_day": split_day,
                "model_label": str(model_label),
                "time_sec": float(time_sec),
                "rho_mean": float(np.mean(vals)),
                "rho_sem": sem(vals),
                "n_subjects": int(len(vals)),
            }
        )
    if len(rows) == 0:
        raise ValueError("No group MVPA presentation summary rows computed")
    return pd.DataFrame(rows)


def run_presentation_mvpa_model_timecourse(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    subject_df = subject_timecourse_rows(output_dir)
    out = group_summary(subject_df)
    subject_path = output_dir / "presentation_mvpa_model_timecourse_subject.csv"
    out_path = output_dir / "presentation_mvpa_model_timecourse.csv"
    subject_df.to_csv(subject_path, index=False)
    out.to_csv(out_path, index=False)
    print(f"[presentation MVPA] wrote {subject_path}", flush=True)
    print(f"[presentation MVPA] wrote {out_path}", flush=True)
    return {"subject": subject_path, "summary": out_path}


if __name__ == "__main__":
    run_presentation_mvpa_model_timecourse()
