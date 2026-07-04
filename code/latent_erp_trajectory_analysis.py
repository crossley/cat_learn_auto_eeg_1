#!/usr/bin/env python3
"""Latent trajectories from stimulus-locked ERP topographies."""

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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from figure_style import OUTPUT_DIR
from latent_dynamics_utils import (
    group_distance_summary,
    require_csv,
    score_trajectory_geometry,
    trajectory_metrics,
)

N_COMPONENTS = 6
TMIN = 0.0
TMAX = 0.8
ANALYSIS = "erp_voltage"


def build_erp_matrix(output_dir):
    d = require_csv(Path(output_dir) / "erp_grand_average_subject_day_all.csv")
    d = d[
        (d["lock_type"] == "stim")
        & (d["condition"] == "all")
        & (d["time_s"] >= TMIN)
        & (d["time_s"] <= TMAX)
    ].copy()
    if d.empty:
        raise ValueError("No stimulus-locked ERP rows in requested time window")
    wide = (
        d.pivot_table(
            index=["subject", "day", "time_s"],
            columns="channel",
            values="amplitude_v",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["subject", "day", "time_s"])
    )
    channels = sorted([col for col in wide.columns if col not in ["subject", "day", "time_s"]])
    if len(channels) == 0:
        raise ValueError("No ERP sensor columns after pivot")
    x = wide[channels].to_numpy(float)
    if not np.all(np.isfinite(x)):
        raise ValueError("ERP matrix contains non-finite values")
    return wide, channels


def run_latent_erp_trajectory(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "latent_erp_trajectory_progress.json"
    t0 = time.time()
    progress_path.write_text(json.dumps({"stage": "loading"}, indent=2))

    wide, channels = build_erp_matrix(output_dir)
    component_cols = [f"latent_{idx}" for idx in range(1, N_COMPONENTS + 1)]

    progress_path.write_text(
        json.dumps(
            {"stage": "pca", "n_rows": int(len(wide)), "n_channels": len(channels)},
            indent=2,
        )
    )
    x = wide[channels].to_numpy(float)
    scaler = StandardScaler()
    xz = scaler.fit_transform(x)
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    scores = pca.fit_transform(xz)

    points = wide[["subject", "day", "time_s"]].copy()
    points = points.rename(columns={"time_s": "time_sec"})
    for idx, col in enumerate(component_cols):
        points[col] = scores[:, idx]
    points["analysis"] = ANALYSIS

    components = []
    for comp_i, var in enumerate(pca.explained_variance_ratio_, start=1):
        for channel, weight in zip(channels, pca.components_[comp_i - 1]):
            components.append(
                {
                    "analysis": ANALYSIS,
                    "component": int(comp_i),
                    "channel": channel,
                    "weight": float(weight),
                    "explained_variance_ratio": float(var),
                }
            )
    components = pd.DataFrame(components)

    progress_path.write_text(json.dumps({"stage": "geometry"}, indent=2))
    distance_df, score_df, summary_df = score_trajectory_geometry(
        points,
        component_cols,
        "time_sec",
        ANALYSIS,
    )
    group_distance_df = group_distance_summary(distance_df)
    metric_df = trajectory_metrics(points, component_cols, "time_sec", ANALYSIS)

    points_path = output_dir / "latent_erp_trajectory_points.csv"
    components_path = output_dir / "latent_erp_trajectory_components.csv"
    metrics_path = output_dir / "latent_erp_trajectory_metrics.csv"
    distance_path = output_dir / "latent_erp_trajectory_subject_distances.csv"
    group_distance_path = output_dir / "latent_erp_trajectory_group_distances.csv"
    score_path = output_dir / "latent_erp_trajectory_model_subject.csv"
    summary_path = output_dir / "latent_erp_trajectory_model_summary.csv"

    points.to_csv(points_path, index=False)
    components.to_csv(components_path, index=False)
    metric_df.to_csv(metrics_path, index=False)
    distance_df.to_csv(distance_path, index=False)
    group_distance_df.to_csv(group_distance_path, index=False)
    score_df.to_csv(score_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    progress_path.write_text(
        json.dumps(
            {
                "stage": "complete",
                "elapsed_sec": time.time() - t0,
                "n_rows": int(len(points)),
                "n_subjects": int(points["subject"].nunique()),
                "n_channels": int(len(channels)),
            },
            indent=2,
        )
    )
    print(f"[latent ERP] wrote {points_path}", flush=True)
    print(f"[latent ERP] wrote {summary_path}", flush=True)
    return points_path, summary_path


if __name__ == "__main__":
    run_latent_erp_trajectory()
