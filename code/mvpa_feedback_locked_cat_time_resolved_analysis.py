#!/usr/bin/env python3
"""Extract feedback-locked time-resolved MVPA outputs from TG diagonals."""

from pathlib import Path

import numpy as np
import pandas as pd

from mvpa_feedback_locked_cat_tg_analysis import OUTPUT_DIR

PROJECT_DIR = Path(__file__).resolve().parent.parent


def run_mvpa_feedback_locked_cat_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tg_csv = output_dir / "mvpa_feedback_locked_cat_tg_timegen_day_mean.csv"
    day_means_csv = output_dir / "mvpa_feedback_locked_cat_time_resolved_day_means_timecourse.csv"
    if not tg_csv.exists():
        raise FileNotFoundError(
            f"Missing feedback TG output in {output_dir}. "
            "Run mvpa_feedback_locked_cat_tg_analysis.py first."
        )

    tg_df = pd.read_csv(tg_csv)
    if tg_df.empty:
        day_means_df = pd.DataFrame(
            columns=["day", "time_sec", "auc_mean", "auc_sem", "n_subjects"]
        )
        day_means_df.to_csv(day_means_csv, index=False)
        return {"day_means_df": day_means_df, "day_means_csv": day_means_csv}

    d = tg_df[
        (tg_df["train_day"] == tg_df["test_day"])
        & np.isclose(tg_df["train_time_sec"], tg_df["test_time_sec"])
    ].copy()
    if "auc_sem" not in d.columns:
        d["auc_sem"] = np.nan
    if "n_subjects" not in d.columns:
        d["n_subjects"] = np.nan
    day_means_df = (
        d.rename(columns={"train_day": "day", "train_time_sec": "time_sec"})
        [["day", "time_sec", "auc_mean", "auc_sem", "n_subjects"]]
        .sort_values(["day", "time_sec"])
    )
    day_means_df.to_csv(day_means_csv, index=False)
    return {"day_means_df": day_means_df, "day_means_csv": day_means_csv}


if __name__ == "__main__":
    run_mvpa_feedback_locked_cat_time_resolved()
