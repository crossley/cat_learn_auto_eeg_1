#!/usr/bin/env python3
"""Day-structure analyses for stimulus-locked category Haufe patterns."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

from mvpa_stim_locked_cat_tg_window_structure_analysis import MVPA_CAT_TG_WINDOWS

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

INPUT_CSV = OUTPUT_DIR / "mvpa_stim_locked_cat_haufe_session_channel_time.csv"
DAYS = [1, 2, 3, 4, 5]
WINDOWS = ["early", "late"]
N_BOOTSTRAP = 1000
RANDOM_STATE = 42


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def finite_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x * y) / denom)


def load_haufe_input(input_csv):
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Missing stimulus Haufe input: {input_csv}. "
            "Run mvpa_stim_locked_cat_time_resolved_analysis.py first."
        )
    d = pd.read_csv(input_csv)
    if d.empty:
        raise ValueError(f"Empty stimulus Haufe input: {input_csv}")
    required = [
        "subject",
        "day",
        "channel",
        "time_sec",
        "pattern",
    ]
    missing = []
    for col in required:
        if col not in d.columns:
            missing.append(col)
    if len(missing) > 0:
        raise ValueError(f"Missing columns in {input_csv}: {missing}")
    for col in ["subject", "day", "time_sec", "pattern"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=required).copy()
    if d.empty:
        raise ValueError(f"No usable stimulus Haufe rows in {input_csv}")
    d["subject"] = d["subject"].astype(int)
    d["day"] = d["day"].astype(int)
    return d


def make_window_patterns(haufe_df, windows=MVPA_CAT_TG_WINDOWS):
    rows = []
    for window, bounds in windows.items():
        lo = float(bounds[0])
        hi = float(bounds[1])
        d_win = haufe_df[
            (haufe_df["time_sec"] >= lo)
            & (haufe_df["time_sec"] <= hi)
        ].copy()
        if d_win.empty:
            raise ValueError(f"No Haufe rows in window {window}: {lo}-{hi}s")
        grouped = (
            d_win.groupby(["subject", "day", "channel"], as_index=False)
            .agg(
                pattern_mean=("pattern", "mean"),
                n_timepoints=("pattern", "size"),
            )
            .sort_values(["subject", "day", "channel"])
        )
        for _, row in grouped.iterrows():
            rows.append(
                {
                    "subject": int(row["subject"]),
                    "day": int(row["day"]),
                    "window": window,
                    "channel": row["channel"],
                    "pattern_mean": float(row["pattern_mean"]),
                    "n_timepoints": int(row["n_timepoints"]),
                    "window_start_sec": lo,
                    "window_end_sec": hi,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No window-averaged Haufe patterns were produced")
    return out


def subject_day_vector(pattern_df, subject, day, window, channels):
    g = pattern_df[
        (pattern_df["subject"] == int(subject))
        & (pattern_df["day"] == int(day))
        & (pattern_df["window"] == window)
    ]
    if g.empty:
        raise ValueError(
            f"Missing Haufe pattern: subject={subject}, day={day}, window={window}"
        )
    s = g.set_index("channel")["pattern_mean"]
    vals = []
    missing = []
    for ch in channels:
        if ch not in s.index:
            missing.append(ch)
            vals.append(np.nan)
        else:
            vals.append(float(s.loc[ch]))
    if len(missing) > 0:
        raise ValueError(
            f"Missing channels for subject={subject}, day={day}, "
            f"window={window}: {missing}"
        )
    return np.asarray(vals, dtype=float)


def complete_subjects(pattern_df, window, channels):
    subjects = sorted(pattern_df["subject"].dropna().unique().astype(int))
    retained = []
    for subject in subjects:
        complete = True
        for day in DAYS:
            try:
                subject_day_vector(pattern_df, int(subject), int(day), window, channels)
            except ValueError:
                complete = False
        if complete:
            retained.append(int(subject))
    if len(retained) == 0:
        raise ValueError(f"No complete subjects for Haufe window={window}")
    return retained


def make_subject_vector_cache(pattern_df, window, channels):
    subjects = complete_subjects(pattern_df, window, channels)
    cache = {}
    for subject in subjects:
        day_cache = {}
        for day in DAYS:
            day_cache[int(day)] = subject_day_vector(
                pattern_df,
                int(subject),
                int(day),
                window,
                channels,
            )
        cache[int(subject)] = day_cache
    return subjects, cache


def make_symmetrised_similarity(pattern_df):
    channels = sorted(pattern_df["channel"].dropna().unique().tolist())
    if len(channels) < 3:
        raise ValueError("Need at least three channels for Haufe pattern similarity")
    rows = []
    for window in WINDOWS:
        subjects, cache = make_subject_vector_cache(pattern_df, window, channels)
        for subject in subjects:
            for i, day_i in enumerate(DAYS):
                for j in range(i + 1, len(DAYS)):
                    day_j = DAYS[j]
                    sim = finite_corr(
                        cache[int(subject)][int(day_i)],
                        cache[int(subject)][int(day_j)],
                    )
                    if not np.isfinite(sim):
                        raise ValueError(
                            f"Non-finite Haufe similarity: subject={subject}, "
                            f"window={window}, D{day_i}-D{day_j}"
                        )
                    rows.append(
                        {
                            "row_type": "subject",
                            "window": window,
                            "subject": int(subject),
                            "day_low": int(day_i),
                            "day_high": int(day_j),
                            "similarity": float(sim),
                            "similarity_mean": np.nan,
                            "similarity_sem": np.nan,
                            "n_subjects": np.nan,
                            "n_channels": int(len(channels)),
                        }
                    )
    subject_rows = pd.DataFrame(rows)
    if subject_rows.empty:
        raise ValueError("No subject-level Haufe day similarities were produced")
    group_rows = (
        subject_rows.groupby(["window", "day_low", "day_high"], as_index=False)
        .agg(
            similarity_mean=("similarity", "mean"),
            similarity_sem=("similarity", sem),
            n_subjects=("subject", "nunique"),
            n_channels=("n_channels", "max"),
        )
        .sort_values(["window", "day_low", "day_high"])
    )
    group_out = []
    for _, row in group_rows.iterrows():
        group_out.append(
            {
                "row_type": "group",
                "window": row["window"],
                "subject": np.nan,
                "day_low": int(row["day_low"]),
                "day_high": int(row["day_high"]),
                "similarity": np.nan,
                "similarity_mean": float(row["similarity_mean"]),
                "similarity_sem": float(row["similarity_sem"]),
                "n_subjects": int(row["n_subjects"]),
                "n_channels": int(row["n_channels"]),
            }
        )
    group_df = pd.DataFrame(group_out)
    out = pd.concat([subject_rows, group_df], ignore_index=True)
    return out


def group_similarity_matrix(sym_df, window):
    g = sym_df[
        (sym_df["row_type"] == "group")
        & (sym_df["window"] == window)
    ]
    if g.empty:
        raise ValueError(f"Missing Haufe group rows: window={window}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row["similarity_mean"])
        mat[j, i] = float(row["similarity_mean"])
    missing = []
    for i, day_i in enumerate(DAYS):
        for j, day_j in enumerate(DAYS):
            if i == j:
                continue
            if not np.isfinite(mat[i, j]):
                missing.append(f"D{day_i}-D{day_j}")
    if len(missing) > 0:
        raise ValueError(
            f"Missing Haufe day pairs for window={window}: " + ", ".join(missing)
        )
    return mat


def subject_similarity_matrix(sym_df, window, subject):
    g = sym_df[
        (sym_df["row_type"] == "subject")
        & (sym_df["window"] == window)
        & (sym_df["subject"] == int(subject))
    ]
    if g.empty:
        raise ValueError(f"Missing subject Haufe rows: window={window}, subject={subject}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row["similarity"])
        mat[j, i] = float(row["similarity"])
    missing = []
    for i, day_i in enumerate(DAYS):
        for j in range(i + 1, len(DAYS)):
            day_j = DAYS[j]
            if not np.isfinite(mat[i, j]):
                missing.append(f"D{day_i}-D{day_j}")
    if len(missing) > 0:
        raise ValueError(
            f"Missing subject Haufe day pairs for window={window}, "
            f"subject={subject}: " + ", ".join(missing)
        )
    return mat


def distance_matrix_from_similarity(sim_mat):
    finite = sim_mat[np.isfinite(sim_mat)]
    if len(finite) == 0:
        raise ValueError("Cannot build distance matrix from empty similarity matrix")
    max_sim = float(np.max(finite))
    dist = np.full_like(sim_mat, np.nan, dtype=float)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if i == j:
                dist[i, j] = 0.0
            elif np.isfinite(sim_mat[i, j]):
                dist[i, j] = max_sim - float(sim_mat[i, j])
    if not np.all(np.isfinite(dist)):
        raise ValueError("Distance matrix contains missing values")
    return dist


def matrix_to_pair_vector(mat):
    vals = []
    for i in range(len(DAYS)):
        for j in range(i + 1, len(DAYS)):
            vals.append(float(mat[i, j]))
    return np.asarray(vals, dtype=float)


def cluster_members(node_id, z):
    node_id = int(node_id)
    n_days = len(DAYS)
    if node_id < n_days:
        return [DAYS[node_id]]
    merge_idx = node_id - n_days
    members = []
    for child_col in [0, 1]:
        child_members = cluster_members(int(z[merge_idx, child_col]), z)
        for day in child_members:
            members.append(day)
    return sorted(members)


def cluster_description_from_distance(dist):
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    order_idx = leaves_list(z)
    order_days = []
    for idx in order_idx:
        order_days.append(str(DAYS[int(idx)]))
    first_members = cluster_members(int(z[0, 0]), z)
    for day in cluster_members(int(z[0, 1]), z):
        first_members.append(day)
    first_members = sorted(first_members)
    first_labels = []
    for day in first_members:
        first_labels.append(f"D{day}")
    first_pair = "-".join(first_labels)
    final_left = cluster_members(int(z[-1, 0]), z)
    final_right = cluster_members(int(z[-1, 1]), z)
    last_singleton_day = np.nan
    if len(final_left) == 1 and len(final_right) > 1:
        last_singleton_day = int(final_left[0])
    elif len(final_right) == 1 and len(final_left) > 1:
        last_singleton_day = int(final_right[0])
    return z, ",".join(order_days), first_pair, last_singleton_day


def make_clusters(sym_df):
    rows = []
    for window in WINDOWS:
        sim_mat = group_similarity_matrix(sym_df, window)
        dist = distance_matrix_from_similarity(sim_mat)
        z, day_order, first_pair, last_singleton_day = cluster_description_from_distance(dist)
        for merge_idx in range(z.shape[0]):
            rows.append(
                {
                    "row_type": "linkage",
                    "window": window,
                    "merge_index": int(merge_idx),
                    "child_1": float(z[merge_idx, 0]),
                    "child_2": float(z[merge_idx, 1]),
                    "distance": float(z[merge_idx, 2]),
                    "n_members": int(z[merge_idx, 3]),
                    "day": np.nan,
                    "order_position": np.nan,
                    "day_order": day_order,
                    "first_pair": first_pair,
                    "last_singleton_day": last_singleton_day,
                }
            )
        order_parts = day_order.split(",")
        for pos, day_text in enumerate(order_parts):
            rows.append(
                {
                    "row_type": "order",
                    "window": window,
                    "merge_index": np.nan,
                    "child_1": np.nan,
                    "child_2": np.nan,
                    "distance": np.nan,
                    "n_members": np.nan,
                    "day": int(day_text),
                    "order_position": int(pos),
                    "day_order": day_order,
                    "first_pair": first_pair,
                    "last_singleton_day": last_singleton_day,
                }
            )
    return pd.DataFrame(rows)


def complete_similarity_subjects(sym_df, window):
    d_key = sym_df[
        (sym_df["row_type"] == "subject")
        & (sym_df["window"] == window)
    ]
    subjects = sorted(d_key["subject"].dropna().unique().astype(int))
    retained = []
    for subject in subjects:
        try:
            subject_similarity_matrix(sym_df, window, int(subject))
            retained.append(int(subject))
        except ValueError:
            pass
    if len(retained) == 0:
        raise ValueError(f"No complete Haufe similarity subjects: window={window}")
    return retained


def subject_matrix_cache(sym_df, window):
    subjects = complete_similarity_subjects(sym_df, window)
    matrices = {}
    for subject in subjects:
        matrices[int(subject)] = subject_similarity_matrix(sym_df, window, int(subject))
    return subjects, matrices


def bootstrap_mean_similarity_matrix(matrices, sampled_subjects, window):
    mats = []
    for subject in sampled_subjects:
        mats.append(matrices[int(subject)])
    if len(mats) == 0:
        raise ValueError("Cannot bootstrap empty Haufe subject sample")
    arr = np.stack(mats, axis=0)
    out = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for i in range(len(DAYS)):
        for j in range(len(DAYS)):
            if i == j:
                continue
            vals = arr[:, i, j]
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                raise ValueError(
                    f"Bootstrap sample has no finite Haufe values: "
                    f"window={window}, day_pair=D{DAYS[i]}-D{DAYS[j]}"
                )
            out[i, j] = float(np.mean(vals))
    return out


def make_bootstrap_clusters(sym_df, n_bootstrap=N_BOOTSTRAP, random_state=RANDOM_STATE):
    rng = np.random.default_rng(random_state)
    rows = []
    for window in WINDOWS:
        print(
            f"[Haufe day-structure] Bootstrapping {window} "
            f"({n_bootstrap} resamples)..."
        )
        subjects, matrices = subject_matrix_cache(sym_df, window)
        first_pair_counts = {}
        last_day_counts = {}
        for boot_idx in range(n_bootstrap):
            sampled = rng.choice(subjects, size=len(subjects), replace=True)
            sim_mat = bootstrap_mean_similarity_matrix(matrices, sampled, window)
            dist = distance_matrix_from_similarity(sim_mat)
            _z, day_order, first_pair, last_singleton_day = cluster_description_from_distance(dist)
            sampled_labels = []
            for subject in sampled:
                sampled_labels.append(str(int(subject)))
            sampled_subjects = ",".join(sampled_labels)
            rows.append(
                {
                    "row_type": "bootstrap",
                    "window": window,
                    "bootstrap": int(boot_idx),
                    "n_subjects": int(len(subjects)),
                    "sampled_subjects": sampled_subjects,
                    "first_pair": first_pair,
                    "last_singleton_day": last_singleton_day,
                    "day_order": day_order,
                    "support_type": "",
                    "event": "",
                    "support": np.nan,
                }
            )
            if first_pair not in first_pair_counts:
                first_pair_counts[first_pair] = 0
            first_pair_counts[first_pair] += 1
            if np.isfinite(last_singleton_day):
                last_key = f"D{int(last_singleton_day)}"
            else:
                last_key = "none"
            if last_key not in last_day_counts:
                last_day_counts[last_key] = 0
            last_day_counts[last_key] += 1
        for event, count in sorted(first_pair_counts.items()):
            rows.append(
                {
                    "row_type": "support",
                    "window": window,
                    "bootstrap": np.nan,
                    "n_subjects": int(len(subjects)),
                    "sampled_subjects": "",
                    "first_pair": "",
                    "last_singleton_day": np.nan,
                    "day_order": "",
                    "support_type": "first_pair",
                    "event": event,
                    "support": float(count) / float(n_bootstrap),
                }
            )
        for event, count in sorted(last_day_counts.items()):
            rows.append(
                {
                    "row_type": "support",
                    "window": window,
                    "bootstrap": np.nan,
                    "n_subjects": int(len(subjects)),
                    "sampled_subjects": "",
                    "first_pair": "",
                    "last_singleton_day": np.nan,
                    "day_order": "",
                    "support_type": "last_singleton_day",
                    "event": event,
                    "support": float(count) / float(n_bootstrap),
                }
            )
    return pd.DataFrame(rows)


def make_distance_stability(sym_df):
    rows = []
    for window in WINDOWS:
        group_sim = group_similarity_matrix(sym_df, window)
        group_dist = distance_matrix_from_similarity(group_sim)
        group_vec = matrix_to_pair_vector(group_dist)
        subjects, matrices = subject_matrix_cache(sym_df, window)
        subject_corrs = []
        for subject in subjects:
            subject_dist = distance_matrix_from_similarity(matrices[int(subject)])
            subject_vec = matrix_to_pair_vector(subject_dist)
            r = finite_corr(subject_vec, group_vec)
            subject_corrs.append(r)
            rows.append(
                {
                    "row_type": "subject",
                    "window": window,
                    "subject": int(subject),
                    "distance_correlation": r,
                    "mean_distance_correlation": np.nan,
                    "sem_distance_correlation": np.nan,
                    "n_subjects": np.nan,
                }
            )
        vals = np.asarray(subject_corrs, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            raise ValueError(f"No finite Haufe distance-stability values: window={window}")
        rows.append(
            {
                "row_type": "summary",
                "window": window,
                "subject": np.nan,
                "distance_correlation": np.nan,
                "mean_distance_correlation": float(np.mean(vals)),
                "sem_distance_correlation": sem(vals),
                "n_subjects": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def classical_mds(dist):
    dist = np.asarray(dist, dtype=float)
    n = dist.shape[0]
    h = np.eye(n) - np.ones((n, n)) / float(n)
    b = -0.5 * h @ (dist**2) @ h
    eigvals, eigvecs = np.linalg.eigh(b)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    coords = np.zeros((n, 2), dtype=float)
    for dim in range(2):
        if dim < len(eigvals) and eigvals[dim] > 0:
            coords[:, dim] = eigvecs[:, dim] * np.sqrt(eigvals[dim])
    return coords, eigvals


def make_embedding(sym_df):
    rows = []
    for window in WINDOWS:
        sim_mat = group_similarity_matrix(sym_df, window)
        dist = distance_matrix_from_similarity(sim_mat)
        coords, eigvals = classical_mds(dist)
        positive_total = 0.0
        for val in eigvals:
            if val > 0:
                positive_total += float(val)
        explained = np.nan
        if positive_total > 0:
            numerator = 0.0
            for dim in range(min(2, len(eigvals))):
                if eigvals[dim] > 0:
                    numerator += float(eigvals[dim])
            explained = numerator / positive_total
        for i, day in enumerate(DAYS):
            rows.append(
                {
                    "window": window,
                    "day": int(day),
                    "x": float(coords[i, 0]),
                    "y": float(coords[i, 1]),
                    "variance_explained_2d": float(explained),
                }
            )
    return pd.DataFrame(rows)


def run_mvpa_stim_locked_cat_haufe_day_structure(
    input_csv: Path | str = INPUT_CSV,
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    haufe_df = load_haufe_input(input_csv)
    pattern_df = make_window_patterns(haufe_df)
    sym_df = make_symmetrised_similarity(pattern_df)
    clusters_df = make_clusters(sym_df)
    embedding_df = make_embedding(sym_df)
    bootstrap_df = make_bootstrap_clusters(sym_df)
    stability_df = make_distance_stability(sym_df)

    pattern_csv = output_dir / "mvpa_stim_locked_cat_haufe_day_structure_window_patterns.csv"
    sym_csv = output_dir / "mvpa_stim_locked_cat_haufe_day_structure_symmetrised.csv"
    clusters_csv = output_dir / "mvpa_stim_locked_cat_haufe_day_structure_clusters.csv"
    embedding_csv = output_dir / "mvpa_stim_locked_cat_haufe_day_structure_embedding.csv"
    bootstrap_csv = output_dir / "mvpa_stim_locked_cat_haufe_day_structure_bootstrap_clusters.csv"
    stability_csv = output_dir / "mvpa_stim_locked_cat_haufe_day_structure_distance_stability.csv"

    pattern_df.to_csv(pattern_csv, index=False)
    sym_df.to_csv(sym_csv, index=False)
    clusters_df.to_csv(clusters_csv, index=False)
    embedding_df.to_csv(embedding_csv, index=False)
    bootstrap_df.to_csv(bootstrap_csv, index=False)
    stability_df.to_csv(stability_csv, index=False)

    print(f"[Haufe day-structure] Wrote {pattern_csv}")
    print(f"[Haufe day-structure] Wrote {sym_csv}")
    print(f"[Haufe day-structure] Wrote {clusters_csv}")
    print(f"[Haufe day-structure] Wrote {embedding_csv}")
    print(f"[Haufe day-structure] Wrote {bootstrap_csv}")
    print(f"[Haufe day-structure] Wrote {stability_csv}")

    return {
        "pattern_df": pattern_df,
        "symmetrised_df": sym_df,
        "clusters_df": clusters_df,
        "embedding_df": embedding_df,
        "bootstrap_df": bootstrap_df,
        "stability_df": stability_df,
        "pattern_csv": pattern_csv,
        "symmetrised_csv": sym_csv,
        "clusters_csv": clusters_csv,
        "embedding_csv": embedding_csv,
        "bootstrap_csv": bootstrap_csv,
        "stability_csv": stability_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_haufe_day_structure()
