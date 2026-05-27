#!/usr/bin/env python3
"""Create selected presentation figures for participant exclusion cohorts."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import numpy as np
import pandas as pd

from connect_sensorwide_analysis import OUTPUT_DIR
from connect_sensorwide_figure import (
    plot_active_pair_overlay_single_pct,
    plot_active_pair_subject_z_euclidean_scaled_matrices,
)
from mvpa_stim_locked_cat_late_window_transfer_figure import (
    save_symmetrised_logreg_figure,
)
from util_rsa_figure import save_cross_day_geometry_figure

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"

DOSE_FLAG_SUBJECTS = {2, 8, 15, 19}
TIMING_FLAG_SESSIONS = {
    (1, 3),
    (1, 4),
    (4, 2),
    (11, 2),
    (14, 4),
    (15, 3),
}
STRICT_FLAG_SUBJECTS = {1, 2, 4, 8, 11, 14, 15, 19}


def keep_subject_day(subject, day, cohort):
    subject = int(subject)
    day = int(day)
    if cohort == "ppt_full":
        return True
    if cohort == "ppt_no_dose_flags":
        return subject not in DOSE_FLAG_SUBJECTS
    if cohort == "ppt_no_timing_flags":
        return (subject, day) not in TIMING_FLAG_SESSIONS
    if cohort == "ppt_strict_clean":
        return subject not in STRICT_FLAG_SUBJECTS
    raise ValueError(f"Unknown cohort: {cohort}")


def filter_subject_day_df(df, cohort):
    keep = []
    for _, row in df.iterrows():
        keep.append(keep_subject_day(row["subject"], row["day"], cohort))
    return df[np.asarray(keep, dtype=bool)].copy()


def filter_pair_df(df, cohort):
    keep = []
    for _, row in df.iterrows():
        train_keep = keep_subject_day(row["subject"], row["train_day"], cohort)
        test_keep = keep_subject_day(row["subject"], row["test_day"], cohort)
        keep.append(train_keep and test_keep)
    return df[np.asarray(keep, dtype=bool)].copy()


def group_mvpa_transfer_subject_rows(subject_df):
    rows = []
    group_cols = ["classifier", "train_day", "test_day", "day_distance", "window"]
    for key, g in subject_df.groupby(group_cols):
        classifier, train_day, test_day, day_distance, window = key
        vals = g["auc"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        auc_sem = np.nan
        if len(vals) > 1:
            auc_sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        rows.append(
            {
                "classifier": classifier,
                "train_day": int(train_day),
                "test_day": int(test_day),
                "day_distance": int(day_distance),
                "window": window,
                "auc_mean": float(np.mean(vals)),
                "auc_sem": auc_sem,
                "n_subjects": int(len(vals)),
            }
        )
    if len(rows) == 0:
        raise ValueError("No MVPA transfer rows after cohort filtering")
    return pd.DataFrame(rows)


def save_mvpa_transfer_figure(output_dir, figures_dir, cohort):
    subject_path = (
        output_dir / "mvpa_stim_locked_cat_late_window_transfer_subject_pairs.csv"
    )
    if not subject_path.exists():
        raise FileNotFoundError(f"Missing MVPA transfer subject table: {subject_path}")
    d_subject = pd.read_csv(subject_path)
    d_subject = filter_pair_df(d_subject, cohort)
    d_group = group_mvpa_transfer_subject_rows(d_subject)
    return save_symmetrised_logreg_figure(d_group, figures_dir)


def save_rsa_figures(output_dir, figures_dir, cohort):
    figures = []
    tmp_dir = figures_dir / "_rsa_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    configs = [
        (
            "rsa_stim_cross_day_geometry_similarity.csv",
            "rsa_stim_cross_day_geometry_similarity.png",
        ),
        (
            "rsa_stim_windowed_cross_day_geometry_similarity.csv",
            "rsa_stim_windowed_cross_day_geometry_similarity.png",
        ),
    ]
    for csv_name, fig_name in configs:
        path = output_dir / csv_name
        if not path.exists():
            raise FileNotFoundError(f"Missing RSA table: {path}")
        d = pd.read_csv(path)
        d = filter_pair_df(d, cohort)
        save_cross_day_geometry_figure(d, tmp_dir)
        src = tmp_dir / "rsa_stim_cross_day_geometry_similarity.png"
        dst = figures_dir / fig_name
        if not src.exists():
            raise FileNotFoundError(f"Expected RSA figure was not written: {src}")
        if dst.exists():
            dst.unlink()
        src.replace(dst)
        for extra_name in [
            "rsa_stim_cross_day_geometry_timecourse.png",
            "rsa_stim_cross_day_geometry_timecourse_5x5.png",
        ]:
            extra_path = tmp_dir / extra_name
            if extra_path.exists():
                extra_path.unlink()
        figures.append(dst)
    tmp_dir.rmdir()
    return figures


def sensorwide_day_data_and_pairs(output_dir):
    carpet_path = output_dir / "sensorwide_carpet_timeseries.csv"
    channels_path = output_dir / "sensorwide_channel_layout.csv"
    if not carpet_path.exists() or not channels_path.exists():
        raise FileNotFoundError("Missing sensorwide carpet/channel outputs")
    d_carpet = pd.read_csv(carpet_path)
    d_channels = pd.read_csv(channels_path)
    channel_subset = []
    for ch in d_channels["channel"]:
        channel_subset.append(ch)
    ch_to_idx = {}
    for i, ch in enumerate(channel_subset):
        ch_to_idx[ch] = i
    pair_idx = []
    for i in range(len(channel_subset)):
        for j in range(i + 1, len(channel_subset)):
            pair_idx.append((i, j))

    day_data = {}
    d_lb = d_carpet[
        (d_carpet["lock_type"] == "stim") & (d_carpet["band"] == "broadband")
    ]
    if d_lb.empty:
        raise ValueError("Missing stim/broadband sensorwide rows")
    days = sorted(d_lb["day"].dropna().unique().astype(int).tolist())
    for day in days:
        d_day = d_lb[d_lb["day"] == day]
        times_this = sorted(d_day["lock_time"].dropna().unique().tolist())
        mats = []
        for time_val in times_this:
            mat = np.full((len(channel_subset), len(channel_subset)), np.nan)
            d_t = d_day[d_day["lock_time"] == time_val]
            for _, row in d_t.iterrows():
                i = ch_to_idx.get(row["ch_i"])
                j = ch_to_idx.get(row["ch_j"])
                if i is None or j is None:
                    continue
                mat[i, j] = float(row["conn_val"])
                mat[j, i] = float(row["conn_val"])
            np.fill_diagonal(mat, 0.0)
            mats.append(mat)
        day_data[day] = {
            "times": np.asarray(times_this, dtype=float),
            "mats": mats,
        }
    return day_data, pair_idx, channel_subset


def save_sensorwide_figures(output_dir, figures_dir, cohort):
    subject_path = output_dir / "sensorwide_carpet_subject_timeseries.csv"
    if not subject_path.exists():
        raise FileNotFoundError(f"Missing sensorwide subject table: {subject_path}")
    d_subject = pd.read_csv(subject_path)
    d_subject = filter_subject_day_df(d_subject, cohort)
    day_data, pair_idx, channel_subset = sensorwide_day_data_and_pairs(output_dir)
    paths = []
    paths.append(
        plot_active_pair_overlay_single_pct(
            day_data,
            pair_idx,
            "stim",
            "broadband",
            figures_dir,
            0.20,
            d_subject,
            channel_subset,
        )
    )
    paths.append(
        plot_active_pair_subject_z_euclidean_scaled_matrices(
            day_data,
            pair_idx,
            "stim",
            "broadband",
            figures_dir,
            d_subject,
            channel_subset,
        )
    )
    return paths


def populate_cohort(cohort, output_dir=OUTPUT_DIR, figures_root=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_root) / cohort
    figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ppt exclusions] writing {figures_dir}", flush=True)
    paths = []
    print(f"[ppt exclusions] {cohort}: MVPA late-window transfer", flush=True)
    paths.append(save_mvpa_transfer_figure(output_dir, figures_dir, cohort))
    print(f"[ppt exclusions] {cohort}: RSA geometry figures", flush=True)
    for path in save_rsa_figures(output_dir, figures_dir, cohort):
        paths.append(path)
    print(f"[ppt exclusions] {cohort}: sensorwide figures", flush=True)
    for path in save_sensorwide_figures(output_dir, figures_dir, cohort):
        paths.append(path)
    print(f"[ppt exclusions] done {cohort}: {len(paths)} figures", flush=True)
    return paths


def populate_all():
    cohorts = [
        "ppt_full",
        "ppt_no_dose_flags",
        "ppt_no_timing_flags",
        "ppt_strict_clean",
    ]
    out = {}
    for cohort in cohorts:
        out[cohort] = populate_cohort(cohort)
    return out


if __name__ == "__main__":
    populate_all()
