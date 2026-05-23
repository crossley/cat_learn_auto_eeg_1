#!/usr/bin/env python3
"""Cross-day TG window structure: early/late AUC windows and day-distance gradients."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from analysis_utils import model_term_summary

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
MVPA_STIM_LOCKED_CAT_TG_MATRIX_GLOB = "mvpa_stim_locked_cat_tg_matrix_sub_*_trainD*_testD*.npz"

MVPA_CAT_TG_WINDOWS: dict[str, tuple[float, float]] = {
    "early": (0.060, 0.180),
    "late": (0.400, 0.600),
}


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_ols(formula, df, cluster_subject=True):
    model = smf.ols(formula, data=df).fit()
    if cluster_subject and "subject" in df.columns and df["subject"].nunique() > 1:
        return model.get_robustcov_results(cov_type="cluster", groups=df["subject"])
    return model


def parse_matrix_path(path: Path):
    m = re.match(r"(?:.*_)?sub_(\d+)_trainD(\d+)_testD(\d+)\.npz$", path.name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def summarize_tg_window_matrix(
    win,
    train_times=None,
    test_times=None,
    summary: str = "square_mean",
    top_prop: float = 0.10,
):
    win = np.asarray(win, dtype=float)
    if summary == "square_mean":
        vals = win[np.isfinite(win)]
    elif summary == "diagonal_mean":
        vals = np.diag(win)
        vals = vals[np.isfinite(vals)]
    elif summary == "top10_mean":
        vals = win[np.isfinite(win)]
        if len(vals) > 0:
            n_top = max(1, int(np.ceil(len(vals) * top_prop)))
            vals = np.sort(vals)[-n_top:]
    else:
        raise ValueError(f"Unknown TG window summary: {summary}")
    if len(vals) == 0:
        return np.nan, 0
    return float(np.nanmean(vals)), int(len(vals))


def extract_tg_window_auc(
    matrix_dir: Path | str = OUTPUT_DIR,
    matrix_glob: str = MVPA_STIM_LOCKED_CAT_TG_MATRIX_GLOB,
    windows: dict[str, tuple[float, float]] = MVPA_CAT_TG_WINDOWS,
    summary: str = "square_mean",
):
    matrix_dir = Path(matrix_dir)
    rows = []
    for path in sorted(matrix_dir.glob(matrix_glob)):
        parsed = parse_matrix_path(path)
        if parsed is None:
            continue
        subject, train_day, test_day = parsed
        if train_day == test_day:
            continue
        with np.load(path, allow_pickle=False) as z:
            auc = np.asarray(z["auc"], dtype=float)
            time_sec = np.asarray(z["time_sec"], dtype=float)
        for window_name, (tmin, tmax) in windows.items():
            mask = (time_sec >= tmin) & (time_sec <= tmax)
            if not np.any(mask):
                mean_auc = np.nan
                n_cells = 0
            else:
                win = auc[np.ix_(mask, mask)]
                mean_auc, n_cells = summarize_tg_window_matrix(win, summary=summary)
            rows.append(
                {
                    "subject": subject,
                    "train_day": train_day,
                    "test_day": test_day,
                    "day_distance": abs(train_day - test_day),
                    "window": window_name,
                    "window_tmin": tmin,
                    "window_tmax": tmax,
                    "mean_auc": mean_auc,
                    "n_cells": n_cells,
                    "summary": summary,
                    "matrix_file": path.name,
                }
            )
    return pd.DataFrame(rows)


def fit_tg_window_gradients(window_df):
    d = (
        window_df.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["mean_auc", "day_distance", "window"])
        .copy()
    )
    d = d[d["day_distance"] > 0]
    summaries = []
    models = {}
    for window_name, g in d.groupby("window"):
        model = _fit_ols("mean_auc ~ day_distance", g)
        item = model_term_summary(model, "day_distance")
        item.update(
            {
                "window": window_name,
                "n_rows": int(len(g)),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
        summaries.append(item)
        models[window_name] = model

    d["window"] = pd.Categorical(d["window"], categories=["early", "late"], ordered=False)
    interaction = _fit_ols("mean_auc ~ day_distance * window", d)
    interaction_term = "day_distance:window[T.late]"
    interaction_summary = model_term_summary(interaction, interaction_term)
    interaction_summary.update({"comparison": "late_minus_early_slope"})
    return pd.DataFrame(summaries).sort_values("window"), interaction_summary, models, interaction


def run_mvpa_stim_locked_cat_tg_window_structure(
    matrix_dir: Path | str = OUTPUT_DIR,
    matrix_glob: str = MVPA_STIM_LOCKED_CAT_TG_MATRIX_GLOB,
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    window_df = extract_tg_window_auc(matrix_dir=matrix_dir, matrix_glob=matrix_glob)
    slope_df, interaction_summary, _, _ = fit_tg_window_gradients(window_df)
    window_csv = output_dir / "mvpa_stim_locked_cat_tg_window_auc_subject_pairs.csv"
    slope_csv = output_dir / "mvpa_stim_locked_cat_tg_window_gradient_slopes.csv"
    interaction_json = output_dir / "mvpa_stim_locked_cat_tg_window_slope_difference.json"
    window_df.to_csv(window_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    interaction_json.write_text(json.dumps(interaction_summary, indent=2))
    return {
        "window_df": window_df,
        "slope_df": slope_df,
        "interaction_summary": interaction_summary,
        "window_csv": window_csv,
        "slope_csv": slope_csv,
        "interaction_json": interaction_json,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_tg_window_structure()
