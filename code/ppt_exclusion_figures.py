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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from connect_sensorwide_analysis import OUTPUT_DIR
from connect_sensorwide_figure import (
    ACTIVE_PAIR_PEAK_WINDOWS,
    edge_vector_distance,
    get_day_colors,
    matrix_offdiag_minmax_scaled,
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


def cohort_subject_filter(chunk, cohort):
    if cohort == "ppt_full":
        return np.ones(len(chunk), dtype=bool)
    if cohort == "ppt_no_dose_flags":
        return ~chunk["subject"].astype(int).isin(DOSE_FLAG_SUBJECTS).to_numpy()
    if cohort == "ppt_strict_clean":
        return ~chunk["subject"].astype(int).isin(STRICT_FLAG_SUBJECTS).to_numpy()
    if cohort == "ppt_no_timing_flags":
        subjects = chunk["subject"].astype(int)
        days = chunk["day"].astype(int)
        excluded = np.zeros(len(chunk), dtype=bool)
        for subject, day in TIMING_FLAG_SESSIONS:
            excluded = excluded | (
                (subjects.to_numpy() == subject) & (days.to_numpy() == day)
            )
        return ~excluded
    raise ValueError(f"Unknown cohort: {cohort}")


def pair_key(ch_i, ch_j):
    return f"{ch_i}\t{ch_j}"


def sensorwide_group_context(output_dir):
    carpet_path = output_dir / "sensorwide_carpet_timeseries.csv"
    channels_path = output_dir / "sensorwide_channel_layout.csv"
    if not carpet_path.exists() or not channels_path.exists():
        raise FileNotFoundError("Missing sensorwide carpet/channel outputs")
    d_channels = pd.read_csv(channels_path)
    d_carpet = pd.read_csv(
        carpet_path,
        usecols=["lock_type", "day", "band", "lock_time", "ch_i", "ch_j", "conn_val"],
    )
    d_carpet = d_carpet[
        (d_carpet["lock_type"] == "stim") & (d_carpet["band"] == "broadband")
    ].copy()
    if d_carpet.empty:
        raise ValueError("Missing stim/broadband sensorwide rows")
    d_carpet["pair_key"] = (
        d_carpet["ch_i"].astype(str) + "\t" + d_carpet["ch_j"].astype(str)
    )
    pivot = d_carpet.pivot_table(
        index=["day", "lock_time"],
        columns="pair_key",
        values="conn_val",
        aggfunc="mean",
    )
    pair_cols = []
    for col in pivot.columns:
        pair_cols.append(col)
    n_pairs = len(pair_cols)
    if n_pairs == 0:
        raise ValueError("No sensor pairs found in sensorwide group table")
    active_scores = []
    for col in pair_cols:
        vals = pivot[col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        score = np.nan
        if len(vals) > 0:
            score = float(np.max(vals) - np.min(vals))
        active_scores.append(score)
    finite_scores = np.asarray(active_scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if len(finite_scores) == 0:
        raise ValueError("No finite sensorwide active-pair scores")
    top_k = max(1, int(np.ceil(0.20 * n_pairs)))
    threshold = float(np.sort(finite_scores)[-top_k])
    active_keys = []
    for idx, score in enumerate(active_scores):
        if np.isfinite(score) and score >= threshold:
            active_keys.append(pair_cols[idx])
    if len(active_keys) == 0:
        raise ValueError("No top-20% active sensor pairs selected")

    days = sorted(d_carpet["day"].dropna().unique().astype(int).tolist())
    active_mean = pivot[active_keys].mean(axis=1)
    active_vals = active_mean.to_numpy(dtype=float)
    active_vals = active_vals[np.isfinite(active_vals)]
    y_lim = None
    if len(active_vals) > 0:
        y_min = float(np.min(active_vals))
        y_max = float(np.max(active_vals))
        y_pad = 0.08 * max(y_max - y_min, 1e-12)
        y_lim = (y_min - y_pad, y_max + y_pad)

    peak_rows = []
    for day in days:
        d_day = active_mean.loc[day].sort_index()
        times = d_day.index.to_numpy(dtype=float)
        signal = d_day.to_numpy(dtype=float)
        for peak_i, window in enumerate(ACTIVE_PAIR_PEAK_WINDOWS, start=1):
            idx_vals = []
            lo, hi = window
            for idx, time_val in enumerate(times):
                if time_val >= lo and time_val <= hi and np.isfinite(signal[idx]):
                    idx_vals.append(idx)
            if len(idx_vals) == 0:
                raise ValueError(f"No sensorwide peak candidates for day={day}")
            best_idx = idx_vals[0]
            best_val = float(signal[best_idx])
            for idx in idx_vals:
                val = float(signal[idx])
                if val > best_val:
                    best_idx = idx
                    best_val = val
            peak_rows.append(
                {
                    "day": int(day),
                    "peak": int(peak_i),
                    "peak_time": float(times[best_idx]),
                }
            )
    active_edge_pos = {}
    active_pair_rows = []
    for edge_pos, key in enumerate(active_keys):
        ch_i, ch_j = key.split("\t")
        active_edge_pos[key] = int(edge_pos)
        active_pair_rows.append(
            {
                "pair_key": key,
                "ch_i": ch_i,
                "ch_j": ch_j,
                "edge_pos": int(edge_pos),
            }
        )
    return {
        "active_keys": active_keys,
        "active_key_set": set(active_keys),
        "active_edge_pos": active_edge_pos,
        "active_pair_df": pd.DataFrame(active_pair_rows),
        "days": days,
        "peak_rows": peak_rows,
        "y_lim": y_lim,
    }


def read_sensorwide_subject_chunks(subject_path):
    usecols = [
        "subject",
        "day",
        "lock_type",
        "band",
        "lock_time",
        "ch_i",
        "ch_j",
        "conn_val",
    ]
    return pd.read_csv(subject_path, usecols=usecols, chunksize=500000)


def prepare_sensorwide_subject_chunk(chunk, cohort, active_key_set):
    chunk = chunk[
        (chunk["lock_type"] == "stim") & (chunk["band"] == "broadband")
    ].copy()
    if chunk.empty:
        return chunk
    keep = cohort_subject_filter(chunk, cohort)
    chunk = chunk[keep].copy()
    if chunk.empty:
        return chunk
    chunk["pair_key"] = chunk["ch_i"].astype(str) + "\t" + chunk["ch_j"].astype(str)
    chunk = chunk[chunk["pair_key"].isin(active_key_set)].copy()
    return chunk


def save_sensorwide_overlay_figure(subject_path, figures_dir, cohort, context):
    aggregate = {}
    for chunk in read_sensorwide_subject_chunks(subject_path):
        chunk = prepare_sensorwide_subject_chunk(
            chunk, cohort, context["active_key_set"]
        )
        if chunk.empty:
            continue
        grouped = chunk.groupby(["subject", "day", "lock_time"])["conn_val"].agg(
            ["sum", "count"]
        )
        for key, row in grouped.iterrows():
            subject, day, lock_time = key
            out_key = (int(subject), int(day), float(lock_time))
            if out_key not in aggregate:
                aggregate[out_key] = [0.0, 0]
            aggregate[out_key][0] += float(row["sum"])
            aggregate[out_key][1] += int(row["count"])
    if len(aggregate) == 0:
        raise ValueError(f"No sensorwide overlay rows remain for cohort={cohort}")

    session_rows = []
    for key, vals in aggregate.items():
        subject, day, lock_time = key
        total, count = vals
        session_rows.append(
            {
                "subject": subject,
                "day": day,
                "lock_time": lock_time,
                "conn_mean": float(total / count),
            }
        )
    session_df = pd.DataFrame(session_rows)
    stat_rows = []
    for key, g in session_df.groupby(["day", "lock_time"]):
        day, lock_time = key
        vals = g["conn_mean"].to_numpy(dtype=float)
        sem = np.nan
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        stat_rows.append(
            {
                "day": int(day),
                "lock_time": float(lock_time),
                "mean": float(np.mean(vals)),
                "sem": sem,
                "n": int(len(vals)),
            }
        )
    stat_df = pd.DataFrame(stat_rows)
    days = context["days"]
    day_colors = get_day_colors(days)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for day in days:
        d_day = stat_df[stat_df["day"] == day].sort_values("lock_time")
        if d_day.empty:
            raise ValueError(f"Missing sensorwide overlay stats for day {day}")
        times = d_day["lock_time"].to_numpy(dtype=float)
        mean = d_day["mean"].to_numpy(dtype=float)
        sem = d_day["sem"].to_numpy(dtype=float)
        ax.plot(times, mean, color=day_colors[day], linewidth=2.0, label=f"D{day}")
        ax.fill_between(
            times,
            mean - sem,
            mean + sem,
            color=day_colors[day],
            alpha=0.16,
            linewidth=0,
        )
    ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlabel("Stim-locked time (s)")
    ax.set_ylabel("Connectivity")
    ax.set_title(f"Top 20% active pairs (n={len(context['active_keys'])})")
    if context["y_lim"] is not None:
        ax.set_ylim(context["y_lim"])
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle("Active Sensor-Pair Connectivity: stim, broadband")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = (
        figures_dir
        / "sensorwide_active_pair_overlay_top20_stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def sensorwide_peak_lookup(context):
    lookup = {}
    times_by_day = {}
    for row in context["peak_rows"]:
        day = int(row["day"])
        peak = int(row["peak"])
        time_val = float(row["peak_time"])
        time_key = round(time_val, 6)
        lookup[(day, time_key)] = peak
        if day not in times_by_day:
            times_by_day[day] = set()
        times_by_day[day].add(time_key)
    return lookup, times_by_day


def save_sensorwide_distance_figure(subject_path, figures_dir, cohort, context):
    peak_lookup, times_by_day = sensorwide_peak_lookup(context)
    vector_map = {}
    n_edges = len(context["active_keys"])
    for chunk in read_sensorwide_subject_chunks(subject_path):
        chunk = prepare_sensorwide_subject_chunk(
            chunk, cohort, context["active_key_set"]
        )
        if chunk.empty:
            continue
        keep = []
        time_keys = []
        peaks = []
        for _, row in chunk.iterrows():
            day = int(row["day"])
            time_key = round(float(row["lock_time"]), 6)
            peak = peak_lookup.get((day, time_key))
            keep_row = False
            if day in times_by_day and time_key in times_by_day[day]:
                keep_row = peak is not None
            keep.append(keep_row)
            time_keys.append(time_key)
            peaks.append(peak)
        chunk["time_key"] = time_keys
        chunk["peak"] = peaks
        chunk = chunk[np.asarray(keep, dtype=bool)].copy()
        if chunk.empty:
            continue
        for _, row in chunk.iterrows():
            subject = int(row["subject"])
            day = int(row["day"])
            peak = int(row["peak"])
            key = (subject, day, peak)
            if key not in vector_map:
                vector_map[key] = np.full(n_edges, np.nan, dtype=float)
            edge_pos = context["active_edge_pos"][row["pair_key"]]
            vector_map[key][edge_pos] = float(row["conn_val"])
    if len(vector_map) == 0:
        raise ValueError(f"No sensorwide peak vectors remain for cohort={cohort}")

    subjects = set()
    for subject, _day, _peak in vector_map.keys():
        subjects.add(subject)
    days = context["days"]
    display_mats = {}
    for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
        mean_mat = np.full((len(days), len(days)), np.nan, dtype=float)
        for row_i, day_i in enumerate(days):
            for col_j, day_j in enumerate(days):
                if day_i == day_j:
                    mean_mat[row_i, col_j] = 0.0
                    continue
                vals = []
                for subject in subjects:
                    key_i = (subject, day_i, peak_i)
                    key_j = (subject, day_j, peak_i)
                    if key_i not in vector_map or key_j not in vector_map:
                        continue
                    dist = edge_vector_distance(
                        vector_map[key_i], vector_map[key_j], "z_euclidean"
                    )
                    if np.isfinite(dist):
                        vals.append(float(dist))
                if len(vals) > 0:
                    mean_mat[row_i, col_j] = float(np.mean(vals))
        display_mats[peak_i] = matrix_offdiag_minmax_scaled(mean_mat)

    labels = []
    for day in days:
        labels.append(f"D{day}")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.82")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), squeeze=False)
    im = None
    for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
        ax = axes[0, peak_i - 1]
        mat = display_mats[peak_i]
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            origin="upper",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_title(f"Peak {peak_i}")
        ax.set_xticks(range(len(days)))
        ax.set_yticks(range(len(days)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Day")
        ax.set_ylabel("Day")
        for r in range(len(days)):
            for c in range(len(days)):
                if np.isfinite(mat[r, c]):
                    val = float(mat[r, c])
                    color = "white"
                    if val > 0.65:
                        color = "black"
                    ax.text(
                        c,
                        r,
                        f"{val:.2f}",
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=color,
                    )
    fig.suptitle("Subject-Averaged Z-Euclidean Network Distance")
    fig.subplots_adjust(
        top=0.82,
        bottom=0.16,
        left=0.07,
        right=0.90,
        wspace=0.32,
    )
    cax = fig.add_axes([0.92, 0.22, 0.014, 0.52])
    fig.colorbar(im, cax=cax, label="Scaled distance")
    fig_path = (
        figures_dir
        / "sensorwide_active_pair_subject_z_euclidean_scaled_matrices_"
        "top20_stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_sensorwide_figures(output_dir, figures_dir, cohort):
    subject_path = output_dir / "sensorwide_carpet_subject_timeseries.csv"
    if not subject_path.exists():
        raise FileNotFoundError(f"Missing sensorwide subject table: {subject_path}")
    context = sensorwide_group_context(output_dir)
    paths = []
    paths.append(
        save_sensorwide_overlay_figure(subject_path, figures_dir, cohort, context)
    )
    paths.append(
        save_sensorwide_distance_figure(subject_path, figures_dir, cohort, context)
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
