#!/usr/bin/env python3
"""Compute presentation MVPA transfer-template correlations across time."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from mvpa_transfer_model_compare_analysis import DAYS, spearman_corr, template_value

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


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
            "model": "one_stage_bottleneck",
            "split_day": np.nan,
            "label": "gradual",
        }
    )
    rows.append(
        {
            "model": "one_stage_closeness",
            "split_day": np.nan,
            "label": "day closeness",
        }
    )
    for model in ["two_stage_binary", "two_stage_bottleneck"]:
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


def score_time(d_time):
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
    auc_vals = np.asarray(auc_vals, dtype=float)
    rows = []
    for spec in model_specs():
        pred = template_vector(spec, pairs)
        rows.append(
            {
                "model": str(spec["model"]),
                "split_day": spec["split_day"],
                "model_label": str(spec["label"]),
                "rho": spearman_corr(auc_vals, pred),
            }
        )
    return rows


def run_presentation_mvpa_model_timecourse(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    d = require_csv(output_dir / "mvpa_stim_locked_cat_tg_timegen_day_mean.csv")
    d = d[np.isclose(d["train_time_sec"], d["test_time_sec"])].copy()
    if d.empty:
        raise ValueError("No diagonal time rows in MVPA TG day mean output")
    rows = []
    for time_sec in sorted(d["train_time_sec"].dropna().unique().tolist()):
        d_time = d[np.isclose(d["train_time_sec"], float(time_sec))]
        if len(d_time) < 25:
            raise ValueError(f"Missing day-pair cells for time={time_sec}")
        for row in score_time(d_time):
            row["time_sec"] = float(time_sec)
            rows.append(row)
    out = pd.DataFrame(rows)
    out_path = output_dir / "presentation_mvpa_model_timecourse.csv"
    out.to_csv(out_path, index=False)
    print(f"[presentation MVPA] wrote {out_path}", flush=True)
    return {"mvpa_model_timecourse": out_path}


if __name__ == "__main__":
    run_presentation_mvpa_model_timecourse()
