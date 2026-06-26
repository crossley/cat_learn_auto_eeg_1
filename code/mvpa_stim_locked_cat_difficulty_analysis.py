#!/usr/bin/env python3
"""Stimulus-locked category MVPA split by bound-distance difficulty."""

from __future__ import annotations

import json
import os
import time
import warnings
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

from load_project_data import align_behaviour_to_epochs, load_sessions
from mvpa_stim_locked_cat_time_resolved_analysis import decode_timecourse
from stimulus_difficulty import add_bound_difficulty
from util_mvpa import pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
N_JOBS = 8
MIN_EPOCHS = 20
MIN_CLASS_TRIALS = 5
DIFFICULTIES = ["easy", "difficult"]


def process_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    session_file = task["epo_file"]
    random_state = int(task["random_state"])
    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh = align_behaviour_to_epochs(
            task["beh"],
            epochs,
            event_names=("Stim/A", "Stim/B"),
        )
        stim_epochs = stim_epochs.copy().load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        stim_epochs.resample(128, npad="auto")
        beh = add_bound_difficulty(beh)
        codes = stim_epochs.events[:, 2]
        y = np.full(len(codes), -1, dtype=int)
        y[codes == stim_epochs.event_id["Stim/A"]] = 0
        y[codes == stim_epochs.event_id["Stim/B"]] = 1
        keep = y >= 0
        y = y[keep]
        X_all = stim_epochs.get_data()[keep]
        beh = beh.iloc[np.where(keep)[0]].reset_index(drop=True)
        times = stim_epochs.times.copy()
    except Exception as exc:
        return {
            "ok": False,
            "qc": [
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "difficulty": "all",
                    "stage": "preprocess",
                    "reason": "prep_error",
                    "detail": str(exc),
                }
            ],
        }

    rows = []
    qc_rows = []
    for difficulty in DIFFICULTIES:
        idx = np.where(beh["difficulty"].to_numpy() == difficulty)[0]
        y_diff = y[idx]
        n_trials = int(len(y_diff))
        n_a = int(np.sum(y_diff == 0))
        n_b = int(np.sum(y_diff == 1))
        if n_trials < MIN_EPOCHS or min(n_a, n_b) < MIN_CLASS_TRIALS:
            reason = "insufficient_epochs" if n_trials < MIN_EPOCHS else "insufficient_class_trials"
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "difficulty": difficulty,
                    "stage": "epoch_count",
                    "reason": reason,
                    "detail": f"n_trials={n_trials}, n_a={n_a}, n_b={n_b}",
                }
            )
            continue
        X = X_all[idx]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            auc = decode_timecourse(X, y_diff, n_splits=5, random_state=random_state)
        for ti, auc_val in enumerate(auc):
            rows.append(
                {
                    "difficulty": difficulty,
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "time_sec": float(times[ti]),
                    "auc": float(auc_val),
                    "n_trials": n_trials,
                    "n_a": n_a,
                    "n_b": n_b,
                }
            )
    return {"ok": True, "session_rows": rows, "qc": qc_rows}


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def run_mvpa_stim_locked_cat_difficulty(
    output_dir=OUTPUT_DIR,
    random_state=42,
    n_workers=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    sessions = load_sessions()
    tasks = [
        {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "epo_file": item["epo_file"],
            "epo_path": str(item["epo_path"]),
            "beh": item["beh"],
            "random_state": int(random_state),
        }
        for item in sessions
    ]
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    session_csv = output_dir / "mvpa_stim_locked_cat_difficulty_session_timecourse.csv"
    subject_day_csv = output_dir / "mvpa_stim_locked_cat_difficulty_subject_day_timecourse.csv"
    day_mean_csv = output_dir / "mvpa_stim_locked_cat_difficulty_day_means_timecourse.csv"
    qc_csv = output_dir / "mvpa_stim_locked_cat_difficulty_qc_log.csv"
    progress_json = output_dir / "mvpa_stim_locked_cat_difficulty_progress.json"
    t0 = time.time()

    def write_progress(stage, done, total):
        progress_json.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "done": int(done),
                    "total": int(total),
                    "elapsed_sec": float(time.time() - t0),
                    "updated_at_unix": float(time.time()),
                },
                indent=2,
            )
        )

    rows = []
    qc_rows = []

    def handle_result(result, done):
        if result.get("ok"):
            rows.extend(result.get("session_rows", []))
            qc_rows.extend(result.get("qc", []))
        else:
            qc_rows.extend(result.get("qc", []))
        write_progress("running", done, len(tasks))
        if (done % 5) == 0:
            print(
                f"[MVPA difficulty] complete {done}/{len(tasks)} sessions "
                f"(elapsed {(time.time() - t0) / 60.0:.1f} min)",
                flush=True,
            )

    print(
        f"[MVPA difficulty] Running {len(tasks)} sessions "
        f"(n_workers={n_workers})",
        flush=True,
    )
    write_progress("running", 0, len(tasks))

    def iter_jobs():
        for task in tasks:
            yield delayed(process_session)(task)

    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            handle_result(process_session(task), done)
    elif threadpool_limits is None:
        result_iter = Parallel(
            n_jobs=n_workers,
            backend="loky",
            verbose=0,
            return_as="generator_unordered",
        )(iter_jobs())
        for done, result in enumerate(result_iter, start=1):
            handle_result(result, done)
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(
                n_jobs=n_workers,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(iter_jobs())
            for done, result in enumerate(result_iter, start=1):
                handle_result(result, done)

    session_df = pd.DataFrame(rows)
    qc_df = pd.DataFrame(
        qc_rows,
        columns=["session_file", "subject", "day", "difficulty", "stage", "reason", "detail"],
    )
    if session_df.empty:
        session_df.to_csv(session_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        raise RuntimeError("Difficulty MVPA produced no valid session rows")

    subject_day_df = (
        session_df.groupby(["difficulty", "subject", "day", "time_sec"], as_index=False)
        .agg(auc=("auc", "mean"))
        .sort_values(["difficulty", "subject", "day", "time_sec"])
    )
    day_mean_df = (
        subject_day_df.groupby(["difficulty", "day", "time_sec"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sem=("auc", sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["difficulty", "day", "time_sec"])
    )
    session_df.to_csv(session_csv, index=False)
    subject_day_df.to_csv(subject_day_csv, index=False)
    day_mean_df.to_csv(day_mean_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    write_progress("completed", len(tasks), len(tasks))
    print(f"[MVPA difficulty] wrote {day_mean_csv}", flush=True)
    return {
        "session": session_csv,
        "subject_day": subject_day_csv,
        "day_mean": day_mean_csv,
        "qc": qc_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_difficulty()
