#!/usr/bin/env python3
"""Latent trajectories from time-resolved sensorwide connectivity."""

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
ACTIVE_PCT = 0.10
TMIN = 0.0
TMAX = 0.75
CHUNKSIZE = 1_000_000
ANALYSIS = "connectivity_top10"


def active_pair_labels(output_dir):
    active = require_csv(Path(output_dir) / "connect_sensorwide_model_timecourse_active_pairs.csv")
    active = active[np.isclose(active["active_pct"].astype(float), ACTIVE_PCT)].copy()
    if active.empty:
        raise ValueError(f"No active connectivity pairs for active_pct={ACTIVE_PCT}")
    active = active.sort_values("active_rank")
    return active["pair_label"].astype(str).tolist(), active


def load_active_connectivity_rows(output_dir, active_pairs, progress_path):
    path = Path(output_dir) / "sensorwide_carpet_subject_timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing sensorwide connectivity table: {path}. "
            "Run connect_sensorwide_analysis.py first."
        )
    keep_pairs = set(active_pairs)
    frames = []
    usecols = ["subject", "day", "lock_type", "band", "lock_time", "ch_i", "ch_j", "conn_val"]
    for chunk_i, chunk in enumerate(
        pd.read_csv(path, usecols=usecols, chunksize=CHUNKSIZE),
        start=1,
    ):
        d = chunk[
            (chunk["lock_type"] == "stim")
            & (chunk["band"] == "broadband")
            & (chunk["lock_time"] >= TMIN)
            & (chunk["lock_time"] <= TMAX)
        ].copy()
        if not d.empty:
            d["pair_label"] = d["ch_i"].astype(str) + "--" + d["ch_j"].astype(str)
            d = d[d["pair_label"].isin(keep_pairs)].copy()
            if not d.empty:
                frames.append(d[["subject", "day", "lock_time", "pair_label", "conn_val"]])
        if chunk_i % 10 == 0:
            progress_path.write_text(
                json.dumps({"stage": "reading", "chunks_read": chunk_i}, indent=2)
            )
    if not frames:
        raise ValueError("No active-pair connectivity rows found")
    return pd.concat(frames, ignore_index=True)


def build_connectivity_matrix(rows, active_pairs):
    wide = (
        rows.pivot_table(
            index=["subject", "day", "lock_time"],
            columns="pair_label",
            values="conn_val",
            aggfunc="mean",
        )
        .reset_index()
        .sort_values(["subject", "day", "lock_time"])
    )
    missing = [pair for pair in active_pairs if pair not in wide.columns]
    if missing:
        raise ValueError(f"Missing active connectivity pairs after pivot: {missing[:5]}")
    x = wide[active_pairs].to_numpy(float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Connectivity latent matrix contains non-finite values")
    return wide


def run_latent_connectivity_trajectory(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "latent_connectivity_trajectory_progress.json"
    t0 = time.time()
    progress_path.write_text(json.dumps({"stage": "active_pairs"}, indent=2))

    active_pairs, active_df = active_pair_labels(output_dir)
    rows = load_active_connectivity_rows(output_dir, active_pairs, progress_path)
    progress_path.write_text(
        json.dumps(
            {
                "stage": "pca",
                "n_rows": int(len(rows)),
                "n_active_pairs": int(len(active_pairs)),
            },
            indent=2,
        )
    )
    wide = build_connectivity_matrix(rows, active_pairs)
    component_cols = [f"latent_{idx}" for idx in range(1, N_COMPONENTS + 1)]

    x = wide[active_pairs].to_numpy(float)
    scaler = StandardScaler()
    xz = scaler.fit_transform(x)
    pca = PCA(n_components=N_COMPONENTS, random_state=42)
    scores = pca.fit_transform(xz)

    points = wide[["subject", "day", "lock_time"]].copy()
    points = points.rename(columns={"lock_time": "time_sec"})
    for idx, col in enumerate(component_cols):
        points[col] = scores[:, idx]
    points["analysis"] = ANALYSIS

    components = []
    for comp_i, var in enumerate(pca.explained_variance_ratio_, start=1):
        for pair, weight in zip(active_pairs, pca.components_[comp_i - 1]):
            components.append(
                {
                    "analysis": ANALYSIS,
                    "component": int(comp_i),
                    "pair_label": pair,
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

    points_path = output_dir / "latent_connectivity_trajectory_points.csv"
    components_path = output_dir / "latent_connectivity_trajectory_components.csv"
    active_path = output_dir / "latent_connectivity_trajectory_active_pairs.csv"
    metrics_path = output_dir / "latent_connectivity_trajectory_metrics.csv"
    distance_path = output_dir / "latent_connectivity_trajectory_subject_distances.csv"
    group_distance_path = output_dir / "latent_connectivity_trajectory_group_distances.csv"
    score_path = output_dir / "latent_connectivity_trajectory_model_subject.csv"
    summary_path = output_dir / "latent_connectivity_trajectory_model_summary.csv"

    points.to_csv(points_path, index=False)
    components.to_csv(components_path, index=False)
    active_df.to_csv(active_path, index=False)
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
                "n_active_pairs": int(len(active_pairs)),
            },
            indent=2,
        )
    )
    print(f"[latent connectivity] wrote {points_path}", flush=True)
    print(f"[latent connectivity] wrote {summary_path}", flush=True)
    return points_path, summary_path


if __name__ == "__main__":
    run_latent_connectivity_trajectory()
