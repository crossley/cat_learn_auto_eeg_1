#!/usr/bin/env python3
"""Couple pre-task resting state to same-day task latent trajectory metrics."""

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
from latent_dynamics_utils import require_csv

PREDICTORS = [
    "posterior_alpha_power",
    "frontal_theta_power",
    "global_rest_connectivity",
    "visual_central_rest_connectivity",
    "visual_frontal_rest_connectivity",
]

OUTCOMES = [
    "erp_path_length",
    "erp_mean_speed",
    "erp_end_norm",
    "erp_latent_1_late_mean",
    "erp_latent_2_late_mean",
    "connect_path_length",
    "connect_mean_speed",
    "connect_end_norm",
    "connect_latent_1_late_mean",
    "connect_latent_2_late_mean",
]


def prefix_metrics(d, prefix):
    keep = ["subject", "day", "path_length", "mean_speed", "end_norm"]
    for col in ["latent_1_late_mean", "latent_2_late_mean"]:
        if col in d.columns:
            keep.append(col)
    out = d[keep].copy()
    rename = {}
    for col in keep:
        if col not in ["subject", "day"]:
            rename[col] = f"{prefix}_{col}"
    return out.rename(columns=rename)


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


def run_latent_rest_task_coupling(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "latent_rest_task_coupling_progress.json"
    t0 = time.time()
    progress_path.write_text(json.dumps({"stage": "loading"}, indent=2))

    rest = require_csv(output_dir / "rest_session_predictors_table.csv")
    missing_predictors = [col for col in PREDICTORS if col not in rest.columns]
    if missing_predictors:
        raise ValueError(f"Missing rest predictor columns: {missing_predictors}")

    erp = prefix_metrics(
        require_csv(output_dir / "latent_erp_trajectory_metrics.csv"),
        "erp",
    )
    connect = prefix_metrics(
        require_csv(output_dir / "latent_connectivity_trajectory_metrics.csv"),
        "connect",
    )

    session = rest[["subject", "day"] + PREDICTORS].copy()
    session = session.merge(erp, on=["subject", "day"], how="inner")
    session = session.merge(connect, on=["subject", "day"], how="inner")
    if session.empty:
        raise ValueError("No matched rest and latent task trajectory rows")

    corr = correlation_rows(session)
    table_path = output_dir / "latent_rest_task_coupling_table.csv"
    corr_path = output_dir / "latent_rest_task_coupling_correlations.csv"
    session.to_csv(table_path, index=False)
    corr.to_csv(corr_path, index=False)

    progress_path.write_text(
        json.dumps(
            {
                "stage": "complete",
                "elapsed_sec": time.time() - t0,
                "n_sessions": int(len(session)),
                "n_subjects": int(session["subject"].nunique()),
            },
            indent=2,
        )
    )
    print(f"[latent rest-task] wrote {table_path}", flush=True)
    print(f"[latent rest-task] wrote {corr_path}", flush=True)
    return table_path, corr_path


if __name__ == "__main__":
    run_latent_rest_task_coupling()
