#!/usr/bin/env python3
"""Shared time-resolved RSA compute engine."""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from util_boundary_distance import load_behaviour_with_boundary
from load_project_data import align_behaviour_to_epochs, load_sessions
from util_mvpa import pick_eeg_interpolate_bads
from rsa_model_prediction_analysis import (
    MIN_TRIALS_PER_BIN_SESSION,
    assign_grid_bins,
    choose_grid,
    make_bin_table,
    make_model_rdms,
    run_rsa_model_predictions,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8
MIN_EPOCHS_PER_BIN = 5
SNAPSHOT_TIMES = [0.10, 0.20, 0.35, 0.60]
WINDOW_CENTER_STEP_SEC = 0.025
WINDOW_WIDTH_SEC = 0.050
GEOMETRY_WINDOWS = {
    "early": (0.06, 0.18),
    "late": (0.30, 0.60),
}


def _vector_corr(x_vec, y_vec):
    x_vec = np.asarray(x_vec, dtype=float)
    y_vec = np.asarray(y_vec, dtype=float)
    good = np.isfinite(x_vec) & np.isfinite(y_vec)
    if np.sum(good) < 3:
        return np.nan
    x = x_vec[good] - np.nanmean(x_vec[good])
    y = y_vec[good] - np.nanmean(y_vec[good])
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom == 0.0:
        return np.nan
    return float(np.sum(x * y) / denom)


def _append_csv(df: pd.DataFrame, path: Path, wrote_flag: bool):
    if df.empty:
        return wrote_flag
    df.to_csv(path, mode="a", header=not wrote_flag, index=False)
    return True


def _load_or_build_bins(output_dir):
    bins_csv = output_dir / "rsa_model_stimulus_bins.csv"
    diagnostics_csv = output_dir / "rsa_model_grid_diagnostics.csv"
    rdm_csv = output_dir / "rsa_model_rdms.csv"
    if not bins_csv.exists() or not diagnostics_csv.exists() or not rdm_csv.exists():
        run_rsa_model_predictions(output_dir=output_dir)

    beh, boundary = load_behaviour_with_boundary()
    selected_n, _ = choose_grid(beh)
    beh_binned, x_edges, y_edges = assign_grid_bins(beh, selected_n)
    _, retained_bins = make_bin_table(beh, boundary, selected_n)
    model_rdms = make_model_rdms(retained_bins)
    retained_keys = []
    for row in retained_bins.itertuples():
        retained_keys.append((int(row.sf_bin), int(row.ori_bin)))
    return beh_binned, retained_bins, retained_keys, model_rdms, x_edges, y_edges


def _assign_existing_grid_bins(beh, x_edges, y_edges):
    d = beh.copy()
    d["sf_bin"] = pd.cut(
        d["x"], bins=x_edges, labels=False, include_lowest=True
    ).astype("Int64")
    d["ori_bin"] = pd.cut(
        d["y"], bins=y_edges, labels=False, include_lowest=True
    ).astype("Int64")
    return d


def _make_model_vectors(model_rdms):
    rows = []
    for name, mat in model_rdms.items():
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[1]):
                rows.append(
                    {
                        "model": name,
                        "bin_i": int(i),
                        "bin_j": int(j),
                        "model_dissimilarity": float(mat[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def process_rsa_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    session_file = task["epo_file"]
    retained_keys = task["retained_keys"]
    x_edges = task["x_edges"]
    y_edges = task["y_edges"]

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh_aligned = align_behaviour_to_epochs(
            task["beh"],
            epochs,
            event_names=tuple(task.get("event_names", ("Stim/A", "Stim/B"))),
        )
        stim_epochs = stim_epochs.copy()
        stim_epochs.load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        stim_epochs.resample(128, npad="auto")
        beh_aligned = _assign_existing_grid_bins(beh_aligned, x_edges, y_edges)
        X = stim_epochs.get_data()
        times = stim_epochs.times.copy()
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "preprocess",
                "reason": "prep_error",
                "detail": str(exc),
            },
        }

    bin_patterns = []
    bin_counts = []
    for sf_bin, ori_bin in retained_keys:
        keep = (
            (beh_aligned["sf_bin"].astype("Int64") == sf_bin)
            & (beh_aligned["ori_bin"].astype("Int64") == ori_bin)
        ).to_numpy()
        n_epochs = int(np.sum(keep))
        bin_counts.append(n_epochs)
        if n_epochs < MIN_EPOCHS_PER_BIN:
            bin_patterns.append(np.full((X.shape[1], X.shape[2]), np.nan))
        else:
            bin_patterns.append(np.nanmean(X[keep], axis=0))

    patterns = np.stack(bin_patterns, axis=0)
    n_bins = patterns.shape[0]
    rdm_rows = []
    corr_rows = []
    for ti, time_sec in enumerate(times):
        model_vec_rows = []
        for i in range(n_bins):
            x_vec = patterns[i, :, ti]
            for j in range(i + 1, n_bins):
                sim = _vector_corr(x_vec, patterns[j, :, ti])
                if not np.isfinite(sim):
                    continue
                dissim = 1.0 - sim
                row = {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "time_sec": float(time_sec),
                    "bin_i": int(i),
                    "bin_j": int(j),
                    "dissimilarity": float(dissim),
                    "n_i": int(bin_counts[i]),
                    "n_j": int(bin_counts[j]),
                }
                rdm_rows.append(row)
                model_vec_rows.append(row)
        if model_vec_rows:
            corr_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "time_sec": float(time_sec),
                    "n_pairs": int(len(model_vec_rows)),
                }
            )

    count_rows = []
    for i in range(n_bins):
        count_rows.append(
            {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "bin_i": int(i),
                "sf_bin": int(retained_keys[i][0]),
                "ori_bin": int(retained_keys[i][1]),
                "n_epochs": int(bin_counts[i]),
                "usable": bool(bin_counts[i] >= MIN_EPOCHS_PER_BIN),
            }
        )

    return {
        "ok": True,
        "rdm_df": pd.DataFrame(rdm_rows),
        "count_df": pd.DataFrame(count_rows),
        "time_df": pd.DataFrame(corr_rows),
    }


def process_windowed_rsa_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    session_file = task["epo_file"]
    retained_keys = task["retained_keys"]
    x_edges = task["x_edges"]
    y_edges = task["y_edges"]
    window_width_sec = float(task["window_width_sec"])
    center_step_sec = float(task["center_step_sec"])

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh_aligned = align_behaviour_to_epochs(
            task["beh"],
            epochs,
            event_names=tuple(task.get("event_names", ("Stim/A", "Stim/B"))),
        )
        stim_epochs = stim_epochs.copy()
        stim_epochs.load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        stim_epochs.resample(128, npad="auto")
        beh_aligned = _assign_existing_grid_bins(beh_aligned, x_edges, y_edges)
        X = stim_epochs.get_data()
        times = stim_epochs.times.copy()
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "preprocess",
                "reason": "prep_error",
                "detail": str(exc),
            },
        }

    bin_patterns = []
    bin_counts = []
    for sf_bin, ori_bin in retained_keys:
        keep = (
            (beh_aligned["sf_bin"].astype("Int64") == sf_bin)
            & (beh_aligned["ori_bin"].astype("Int64") == ori_bin)
        ).to_numpy()
        n_epochs = int(np.sum(keep))
        bin_counts.append(n_epochs)
        if n_epochs < MIN_EPOCHS_PER_BIN:
            bin_patterns.append(np.full((X.shape[1], X.shape[2]), np.nan))
        else:
            bin_patterns.append(np.nanmean(X[keep], axis=0))

    patterns = np.stack(bin_patterns, axis=0)
    n_bins = patterns.shape[0]
    centers = np.arange(
        float(times.min()) + window_width_sec / 2.0,
        float(times.max()) - window_width_sec / 2.0 + center_step_sec / 2.0,
        center_step_sec,
    )
    rdm_rows = []
    for center in centers:
        keep_time = (times >= center - window_width_sec / 2.0) & (
            times <= center + window_width_sec / 2.0
        )
        if int(np.sum(keep_time)) < 2:
            continue
        for i in range(n_bins):
            x_vec = patterns[i, :, keep_time].reshape(-1)
            for j in range(i + 1, n_bins):
                sim = _vector_corr(x_vec, patterns[j, :, keep_time].reshape(-1))
                if not np.isfinite(sim):
                    continue
                rdm_rows.append(
                    {
                        "session_file": session_file,
                        "subject": subject,
                        "day": day,
                        "time_sec": float(center),
                        "window_start_sec": float(center - window_width_sec / 2.0),
                        "window_end_sec": float(center + window_width_sec / 2.0),
                        "window_width_sec": window_width_sec,
                        "bin_i": int(i),
                        "bin_j": int(j),
                        "dissimilarity": float(1.0 - sim),
                        "n_i": int(bin_counts[i]),
                        "n_j": int(bin_counts[j]),
                    }
                )

    count_rows = []
    for i in range(n_bins):
        count_rows.append(
            {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "bin_i": int(i),
                "sf_bin": int(retained_keys[i][0]),
                "ori_bin": int(retained_keys[i][1]),
                "n_epochs": int(bin_counts[i]),
                "usable": bool(bin_counts[i] >= MIN_EPOCHS_PER_BIN),
            }
        )
    return {
        "ok": True,
        "rdm_df": pd.DataFrame(rdm_rows),
        "count_df": pd.DataFrame(count_rows),
    }


def compute_model_fit_timecourses(rdm_df, model_vec_df):
    if rdm_df.empty:
        return pd.DataFrame()
    merged = rdm_df.merge(model_vec_df, on=["bin_i", "bin_j"], how="inner")
    rows = []
    for (subject, day, time_sec, model), g in merged.groupby(
        ["subject", "day", "time_sec", "model"], sort=False
    ):
        if len(g) < 8:
            rho = np.nan
        else:
            rho = g["dissimilarity"].corr(
                g["model_dissimilarity"], method="spearman"
            )
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "time_sec": float(time_sec),
                "model": model,
                "rho": float(rho) if np.isfinite(rho) else np.nan,
                "n_pairs": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def compute_cross_day_geometry_similarity(rdm_df):
    if rdm_df.empty:
        return pd.DataFrame()
    vec_map = {}
    for key, g in rdm_df.groupby(["subject", "day", "time_sec"], sort=False):
        g = g.sort_values(["bin_i", "bin_j"])
        vec_map[key] = g[["bin_i", "bin_j", "dissimilarity"]].reset_index(drop=True)

    rows = []
    for subject in sorted(rdm_df["subject"].dropna().unique().astype(int)):
        days = sorted(rdm_df.loc[rdm_df["subject"] == subject, "day"].dropna().unique().astype(int))
        times = sorted(
            rdm_df.loc[rdm_df["subject"] == subject, "time_sec"].dropna().unique().astype(float)
        )
        for d_train in days:
            for d_test in days:
                if d_train == d_test:
                    continue
                for time_sec in times:
                    key_train = (subject, d_train, time_sec)
                    key_test = (subject, d_test, time_sec)
                    if key_train not in vec_map or key_test not in vec_map:
                        continue
                    merged = vec_map[key_train].merge(
                        vec_map[key_test],
                        on=["bin_i", "bin_j"],
                        suffixes=("_train", "_test"),
                    )
                    if len(merged) < 8:
                        rho = np.nan
                    else:
                        rho = merged["dissimilarity_train"].corr(
                            merged["dissimilarity_test"], method="spearman"
                        )
                    rows.append(
                        {
                            "subject": int(subject),
                            "train_day": int(d_train),
                            "test_day": int(d_test),
                            "day_distance": int(abs(d_train - d_test)),
                            "day_pair_type": (
                                "day1_involving"
                                if d_train == 1 or d_test == 1
                                else "later_only"
                            ),
                            "time_sec": float(time_sec),
                            "rho": float(rho) if np.isfinite(rho) else np.nan,
                            "n_pairs": int(len(merged)),
                        }
                    )
    return pd.DataFrame(rows)


def _copy_with_backup(src, dst, backups):
    dst = Path(dst)
    if dst.exists():
        backup = dst.with_suffix(dst.suffix + ".rsa_backup")
        dst.replace(backup)
        backups[dst] = backup
    shutil.copy2(src, dst)


def _restore_backups(backups):
    for dst, backup in backups.items():
        if dst.exists():
            dst.unlink()
        backup.replace(dst)


def run_rsa_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    n_workers: int | None = None,
    progress_every: int = 5,
    output_prefix: str = "rsa_stim",
    event_names: tuple[str, str] = ("Stim/A", "Stim/B"),
    log_label: str = "RSA stim",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    rdm_csv = output_dir / f"{output_prefix}_time_resolved_rdms.csv"
    count_csv = output_dir / f"{output_prefix}_bin_epoch_counts.csv"
    model_fit_csv = output_dir / f"{output_prefix}_model_fit_timecourses.csv"
    cross_day_csv = output_dir / f"{output_prefix}_cross_day_geometry_similarity.csv"
    qc_csv = output_dir / f"{output_prefix}_time_resolved_qc_log.csv"
    for path in [rdm_csv, count_csv, model_fit_csv, cross_day_csv, qc_csv]:
        if path.exists():
            path.unlink()

    _, retained_bins, retained_keys, model_rdms, x_edges, y_edges = _load_or_build_bins(
        output_dir
    )
    model_vec_df = _make_model_vectors(model_rdms)
    model_vec_csv = output_dir / f"{output_prefix}_model_vectors.csv"
    model_vec_df.to_csv(model_vec_csv, index=False)

    sessions = load_sessions(load_epochs=False)
    tasks = []
    for session in sessions:
        task = dict(session)
        task["retained_keys"] = retained_keys
        task["x_edges"] = x_edges
        task["y_edges"] = y_edges
        task["event_names"] = event_names
        tasks.append(task)
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    t0 = time.time()
    print(
        f"[{log_label}] Processing {len(tasks)} sessions "
        f"({len(retained_bins)} bins, n_workers={n_workers})...",
        flush=True,
    )

    wrote_rdm = False
    wrote_count = False
    wrote_qc = False
    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    fit_frames = []
    done = 0

    def handle_result(result):
        nonlocal wrote_rdm, wrote_count, wrote_qc, done
        done += 1
        if result["ok"]:
            wrote_rdm = _append_csv(result["rdm_df"], rdm_csv, wrote_rdm)
            wrote_count = _append_csv(result["count_df"], count_csv, wrote_count)
            fit_frames.append(
                compute_model_fit_timecourses(result["rdm_df"], model_vec_df)
            )
        else:
            wrote_qc = _append_csv(
                pd.DataFrame([result["qc"]], columns=qc_columns), qc_csv, wrote_qc
            )
        if done % max(progress_every, 1) == 0:
            elapsed = time.time() - t0
            print(
                f"[{log_label}] complete {done}/{len(tasks)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    def iter_rsa_jobs():
        for task in tasks:
            yield delayed(process_rsa_session)(task)

    if n_workers == 1:
        for task in tasks:
            handle_result(process_rsa_session(task))
    elif threadpool_limits is None:
        result_iter = Parallel(
            n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
        )(iter_rsa_jobs())
        for result in result_iter:
            handle_result(result)
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(
                n_jobs=n_workers,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(iter_rsa_jobs())
            for result in result_iter:
                handle_result(result)

    fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame()
    fit_df.to_csv(model_fit_csv, index=False)
    rdm_df = pd.read_csv(rdm_csv) if rdm_csv.exists() else pd.DataFrame()
    cross_day_df = compute_cross_day_geometry_similarity(rdm_df)
    cross_day_df.to_csv(cross_day_csv, index=False)
    if not qc_csv.exists():
        pd.DataFrame(columns=qc_columns).to_csv(qc_csv, index=False)

    elapsed = time.time() - t0
    print(f"[{log_label}] Done in {elapsed/60:.1f} min.", flush=True)

    return {
        "rdm_csv": rdm_csv,
        "count_csv": count_csv,
        "model_fit_csv": model_fit_csv,
        "cross_day_csv": cross_day_csv,
        "model_vec_csv": model_vec_csv,
        "qc_csv": qc_csv,
    }


def run_rsa_windowed(
    output_dir: Path | str = OUTPUT_DIR,
    n_workers: int | None = None,
    progress_every: int = 5,
    window_width_sec: float = WINDOW_WIDTH_SEC,
    center_step_sec: float = WINDOW_CENTER_STEP_SEC,
    output_prefix: str = "rsa_stim_windowed",
    event_names: tuple[str, str] = ("Stim/A", "Stim/B"),
    log_label: str = "RSA stim windowed",
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    rdm_csv = output_dir / f"{output_prefix}_rdms.csv"
    count_csv = output_dir / f"{output_prefix}_bin_epoch_counts.csv"
    model_fit_csv = output_dir / f"{output_prefix}_model_fit_timecourses.csv"
    cross_day_csv = output_dir / f"{output_prefix}_cross_day_geometry_similarity.csv"
    qc_csv = output_dir / f"{output_prefix}_qc_log.csv"
    for path in [rdm_csv, count_csv, model_fit_csv, cross_day_csv, qc_csv]:
        if path.exists():
            path.unlink()

    _, retained_bins, retained_keys, model_rdms, x_edges, y_edges = _load_or_build_bins(
        output_dir
    )
    model_vec_df = _make_model_vectors(model_rdms)
    model_vec_csv = output_dir / f"{output_prefix}_model_vectors.csv"
    model_vec_df.to_csv(model_vec_csv, index=False)

    sessions = load_sessions(load_epochs=False)
    tasks = []
    for session in sessions:
        task = dict(session)
        task["retained_keys"] = retained_keys
        task["x_edges"] = x_edges
        task["y_edges"] = y_edges
        task["window_width_sec"] = window_width_sec
        task["center_step_sec"] = center_step_sec
        task["event_names"] = event_names
        tasks.append(task)
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    t0 = time.time()
    print(
        f"[{log_label}] Processing {len(tasks)} sessions "
        f"({len(retained_bins)} bins, {window_width_sec*1000:.0f} ms windows, "
        f"n_workers={n_workers})...",
        flush=True,
    )

    wrote_rdm = False
    wrote_count = False
    wrote_qc = False
    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    fit_frames = []
    done = 0

    def handle_result(result):
        nonlocal wrote_rdm, wrote_count, wrote_qc, done
        done += 1
        if result["ok"]:
            wrote_rdm = _append_csv(result["rdm_df"], rdm_csv, wrote_rdm)
            wrote_count = _append_csv(result["count_df"], count_csv, wrote_count)
            fit_frames.append(
                compute_model_fit_timecourses(result["rdm_df"], model_vec_df)
            )
        else:
            wrote_qc = _append_csv(
                pd.DataFrame([result["qc"]], columns=qc_columns), qc_csv, wrote_qc
            )
        if done % max(progress_every, 1) == 0:
            elapsed = time.time() - t0
            print(
                f"[{log_label}] complete {done}/{len(tasks)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    def iter_windowed_rsa_jobs():
        for task in tasks:
            yield delayed(process_windowed_rsa_session)(task)

    if n_workers == 1:
        for task in tasks:
            handle_result(process_windowed_rsa_session(task))
    elif threadpool_limits is None:
        result_iter = Parallel(
            n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
        )(iter_windowed_rsa_jobs())
        for result in result_iter:
            handle_result(result)
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(
                n_jobs=n_workers,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(iter_windowed_rsa_jobs())
            for result in result_iter:
                handle_result(result)

    fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame()
    fit_df.to_csv(model_fit_csv, index=False)
    rdm_df = pd.read_csv(rdm_csv) if rdm_csv.exists() else pd.DataFrame()
    cross_day_df = compute_cross_day_geometry_similarity(rdm_df)
    cross_day_df.to_csv(cross_day_csv, index=False)
    if not qc_csv.exists():
        pd.DataFrame(columns=qc_columns).to_csv(qc_csv, index=False)

    elapsed = time.time() - t0
    print(f"[{log_label}] Done in {elapsed/60:.1f} min.", flush=True)

    return {
        "rdm_csv": rdm_csv,
        "count_csv": count_csv,
        "model_fit_csv": model_fit_csv,
        "cross_day_csv": cross_day_csv,
        "model_vec_csv": model_vec_csv,
        "qc_csv": qc_csv,
    }


if __name__ == "__main__":
    raise SystemExit(
        "Use rsa_stim_time_resolved_analysis.py, rsa_stim_time_resolved_figure.py, "
        "rsa_stim_windowed_analysis.py, rsa_stim_windowed_figure.py, "
        "rsa_feedback_time_resolved_analysis.py, rsa_feedback_time_resolved_figure.py, "
        "rsa_feedback_windowed_analysis.py, or rsa_feedback_windowed_figure.py."
    )
