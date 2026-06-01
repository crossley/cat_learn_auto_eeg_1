#!/usr/bin/env python3
"""BIC comparisons for presentation MVPA day-structure models.

The fitted model is:

    AUC ~ intercept + same_day + model_structure

where same_day captures the generic within-day advantage and model_structure is
set to zero on same-day cells so model evidence is driven by cross-day
structure after controlling the bright day diagonal.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DAYS = [1, 2, 3, 4, 5]
TG_MATRIX_RE = re.compile(
    r"^mvpa_stim_locked_cat_tg_matrix_sub_(\d+)_trainD(\d+)_testD(\d+)\.npz$"
)
LATE_SUBWINDOWS = [
    ("late_030_040", 0.30, 0.40),
    ("late_040_050", 0.40, 0.50),
    ("late_050_060", 0.50, 0.60),
]


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def presentation_template_value(model, train_day, test_day, split_day=None):
    if model == "null":
        return 0.0
    if train_day == test_day:
        return 0.0
    if model == "continuous":
        return float(0.65 * min(train_day, test_day) / float(max(DAYS)))
    if model == "discrete":
        if split_day is None:
            raise ValueError("discrete model requires split_day")
        train_late = train_day > split_day
        test_late = test_day > split_day
        if train_late != test_late:
            return 0.0
        return float(0.65 * min(train_day, test_day) / float(max(DAYS)))
    raise ValueError(f"Unknown model: {model}")


def model_specs():
    rows = [
        {"model_label": "Diagonal baseline", "model": "null", "split_day": np.nan},
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
    same_day = []
    structure = []
    split_day = None
    if np.isfinite(spec["split_day"]):
        split_day = int(spec["split_day"])
    for row in pair_rows:
        train_day = int(row["train_day"])
        test_day = int(row["test_day"])
        same_day.append(1.0 if train_day == test_day else 0.0)
        structure.append(
            presentation_template_value(
                str(spec["model"]),
                train_day,
                test_day,
                split_day=split_day,
            )
        )
    same_day = np.asarray(same_day, dtype=float)
    structure = np.asarray(structure, dtype=float)
    cols = [np.ones(len(pair_rows), dtype=float), same_day]
    if str(spec["model"]) != "null":
        cols.append(structure)
    return np.column_stack(cols)


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


def score_payload(payload, extra_cols):
    rows = []
    y = np.asarray(payload["auc"], dtype=float)
    pair_rows = payload["pair_rows"]
    for spec in model_specs():
        fit = fit_bic(y, design_matrix(pair_rows, spec))
        row = {
            **extra_cols,
            "model_label": spec["model_label"],
            "model": spec["model"],
            "split_day": spec["split_day"],
            "bic": fit["bic"],
            "r2": fit["r2"],
            "n_obs": fit["n_obs"],
            "n_params": fit["n_params"],
        }
        rows.append(row)
    return rows


def add_delta_bic(score_df, group_cols):
    frames = []
    for _key, g in score_df.groupby(group_cols, dropna=False):
        g = g.copy()
        finite_bic = g["bic"].to_numpy(float)
        finite_bic = finite_bic[np.isfinite(finite_bic)]
        if len(finite_bic) == 0:
            g["delta_bic_best"] = np.nan
        else:
            g["delta_bic_best"] = g["bic"].astype(float) - float(np.min(finite_bic))
        null = g[g["model_label"] == "Diagonal baseline"]
        if null.empty or not np.isfinite(float(null["bic"].iloc[0])):
            g["delta_bic_null"] = np.nan
        else:
            g["delta_bic_null"] = g["bic"].astype(float) - float(null["bic"].iloc[0])
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def summarize(score_df, group_cols):
    rows = []
    for key, g in score_df.groupby(group_cols + ["model_label"], dropna=False):
        key_vals = dict(zip(group_cols + ["model_label"], key))
        bic_vals = g["bic"].to_numpy(float)
        delta_best = g["delta_bic_best"].to_numpy(float)
        delta_null = g["delta_bic_null"].to_numpy(float)
        r2_vals = g["r2"].to_numpy(float)
        bic_vals = bic_vals[np.isfinite(bic_vals)]
        delta_best = delta_best[np.isfinite(delta_best)]
        delta_null = delta_null[np.isfinite(delta_null)]
        r2_vals = r2_vals[np.isfinite(r2_vals)]
        if len(bic_vals) == 0:
            continue
        rows.append(
            {
                **key_vals,
                "bic_mean": float(np.mean(bic_vals)),
                "bic_sem": sem(bic_vals),
                "delta_bic_best_mean": float(np.mean(delta_best))
                if len(delta_best)
                else np.nan,
                "delta_bic_null_mean": float(np.mean(delta_null))
                if len(delta_null)
                else np.nan,
                "r2_mean": float(np.mean(r2_vals)) if len(r2_vals) else np.nan,
                "r2_sem": sem(r2_vals),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(group_cols + ["delta_bic_best_mean", "bic_mean"])


def tg_subject_payloads(output_dir):
    files = []
    for path in sorted(output_dir.glob("mvpa_stim_locked_cat_tg_matrix_sub_*.npz")):
        match = TG_MATRIX_RE.match(path.name)
        if match is None:
            continue
        files.append(
            {
                "subject": int(match.group(1)),
                "train_day": int(match.group(2)),
                "test_day": int(match.group(3)),
                "path": path,
            }
        )
    if not files:
        raise FileNotFoundError(f"No TG matrix files found in {output_dir}")

    by_subject = {}
    for row in files:
        by_subject.setdefault(row["subject"], []).append(row)

    payloads = []
    for subject, rows in sorted(by_subject.items()):
        pair_data = []
        times_ref = None
        for row in rows:
            data = np.load(row["path"])
            auc = data["auc"]
            times = np.asarray(data["time_sec"], dtype=float)
            if times_ref is None:
                times_ref = times
            if len(times_ref) != len(times):
                raise ValueError(f"Time length mismatch: {row['path']}")
            pair_data.append(
                {
                    "train_day": int(row["train_day"]),
                    "test_day": int(row["test_day"]),
                    "diag": np.asarray(np.diag(auc), dtype=float),
                }
            )
        for time_i, time_sec in enumerate(times_ref):
            pair_rows = []
            auc_vals = []
            for row in pair_data:
                val = float(row["diag"][time_i])
                if not np.isfinite(val):
                    continue
                pair_rows.append(
                    {
                        "train_day": int(row["train_day"]),
                        "test_day": int(row["test_day"]),
                    }
                )
                auc_vals.append(val)
            if len(auc_vals) >= 10:
                payloads.append(
                    {
                        "subject": int(subject),
                        "time_sec": float(time_sec),
                        "pair_rows": pair_rows,
                        "auc": auc_vals,
                    }
                )
    return payloads


def subwindow_subject_payloads(output_dir):
    payloads = []
    for window, _lo, _hi in LATE_SUBWINDOWS:
        path = (
            output_dir
            / f"mvpa_stim_locked_cat_{window}_window_transfer_subject_pairs.csv"
        )
        if not path.exists():
            raise FileNotFoundError(f"Missing late-subwindow transfer file: {path}")
        d = pd.read_csv(path)
        for (classifier, subject), g in d.groupby(["classifier", "subject"]):
            pair_rows = []
            auc_vals = []
            for row in g.itertuples(index=False):
                if not np.isfinite(float(row.auc)):
                    continue
                pair_rows.append(
                    {
                        "train_day": int(row.train_day),
                        "test_day": int(row.test_day),
                    }
                )
                auc_vals.append(float(row.auc))
            if len(auc_vals) >= 10:
                payloads.append(
                    {
                        "classifier": str(classifier),
                        "subject": int(subject),
                        "window": window,
                        "pair_rows": pair_rows,
                        "auc": auc_vals,
                    }
                )
    return payloads


def run_mvpa_presentation_model_bic(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)

    tg_rows = []
    for payload in tg_subject_payloads(output_dir):
        tg_rows.extend(
            score_payload(
                payload,
                {
                    "analysis": "tg_diagonal",
                    "classifier": "logreg",
                    "subject": payload["subject"],
                    "time_sec": payload["time_sec"],
                },
            )
        )
    tg_score = pd.DataFrame(tg_rows)
    tg_score = add_delta_bic(tg_score, ["subject", "time_sec"])
    tg_summary = summarize(tg_score, ["analysis", "classifier", "time_sec"])

    sub_rows = []
    for payload in subwindow_subject_payloads(output_dir):
        sub_rows.extend(
            score_payload(
                payload,
                {
                    "analysis": "spatiotemporal_window",
                    "classifier": payload["classifier"],
                    "subject": payload["subject"],
                    "window": payload["window"],
                },
            )
        )
    sub_score = pd.DataFrame(sub_rows)
    sub_score = add_delta_bic(sub_score, ["classifier", "subject", "window"])
    sub_summary = summarize(sub_score, ["analysis", "classifier", "window"])

    tg_score_path = output_dir / "mvpa_tg_diagonal_presentation_model_bic_subject.csv"
    tg_summary_path = output_dir / "mvpa_tg_diagonal_presentation_model_bic_summary.csv"
    sub_score_path = (
        output_dir / "mvpa_late_subwindow_presentation_model_bic_subject.csv"
    )
    sub_summary_path = (
        output_dir / "mvpa_late_subwindow_presentation_model_bic_summary.csv"
    )

    tg_score.to_csv(tg_score_path, index=False)
    tg_summary.to_csv(tg_summary_path, index=False)
    sub_score.to_csv(sub_score_path, index=False)
    sub_summary.to_csv(sub_summary_path, index=False)
    print(f"[MVPA presentation BIC] wrote {tg_score_path}", flush=True)
    print(f"[MVPA presentation BIC] wrote {tg_summary_path}", flush=True)
    print(f"[MVPA presentation BIC] wrote {sub_score_path}", flush=True)
    print(f"[MVPA presentation BIC] wrote {sub_summary_path}", flush=True)
    return {
        "tg_subject": tg_score_path,
        "tg_summary": tg_summary_path,
        "subwindow_subject": sub_score_path,
        "subwindow_summary": sub_summary_path,
    }


if __name__ == "__main__":
    run_mvpa_presentation_model_bic()
