#!/usr/bin/env python3
"""Stimulus-locked category transfer in three late subwindows.

This script prepares each EEG session once, extracts all late subwindows from
that prepared data, then parallelizes transfer fits across subject/window jobs.
"""

from __future__ import annotations

import json
import os
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

from load_project_data import load_sessions
from mvpa_stim_locked_cat_late_window_analysis import OUTPUT_DIR, RANDOM_STATE
from mvpa_stim_locked_cat_late_window_transfer_analysis import (
    N_JOBS,
    fit_transfer_subject,
    make_group_summary,
)
from util_mvpa import pick_eeg_interpolate_bads

LATE_SUBWINDOWS = [
    ("late_030_040", 0.30, 0.40),
    ("late_040_050", 0.40, 0.50),
    ("late_050_060", 0.50, 0.60),
]

QC_COLUMNS = [
    "window",
    "session_file",
    "subject",
    "day",
    "classifier",
    "train_day",
    "test_day",
    "stage",
    "reason",
    "detail",
]


def prepare_late_subwindow_session(task):
    session_file = task["epo_file"]
    subject = int(task["subject"])
    day = int(task["day"])
    min_epochs = int(task["min_epochs"])
    subwindows = task["subwindows"]

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_events = []
        for event_name in ["Stim/A", "Stim/B"]:
            if event_name in epochs.event_id:
                stim_events.append(event_name)
        if len(stim_events) < 2:
            return {
                "ok": False,
                "items": [],
                "qc": [
                    {
                        "window": "all",
                        "session_file": session_file,
                        "subject": subject,
                        "day": day,
                        "stage": "event_select",
                        "reason": "missing_stim_labels",
                        "detail": ",".join(stim_events),
                    }
                ],
            }
        stim_epochs = epochs[stim_events].copy()
        stim_epochs.load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        if len(stim_epochs.ch_names) == 0:
            raise RuntimeError("No EEG channels after interpolation.")
        stim_epochs.resample(128, npad="auto")
    except Exception as exc:
        return {
            "ok": False,
            "items": [],
            "qc": [
                {
                    "window": "all",
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "preprocess",
                    "reason": "prep_error",
                    "detail": str(exc),
                }
            ],
        }

    codes = stim_epochs.events[:, 2]
    y = np.full(len(codes), -1, dtype=int)
    y[codes == stim_epochs.event_id["Stim/A"]] = 0
    y[codes == stim_epochs.event_id["Stim/B"]] = 1
    keep = y >= 0
    y = y[keep]
    X = stim_epochs.get_data()[keep]
    times = stim_epochs.times.copy()

    n_a = int(np.sum(y == 0))
    n_b = int(np.sum(y == 1))
    n_trials = int(len(y))
    if n_trials < min_epochs:
        return {
            "ok": False,
            "items": [],
            "qc": [
                {
                    "window": "all",
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "epoch_count",
                    "reason": "insufficient_epochs",
                    "detail": f"n_trials={n_trials} < min_epochs={min_epochs}",
                }
            ],
        }
    if min(n_a, n_b) < 5:
        return {
            "ok": False,
            "items": [],
            "qc": [
                {
                    "window": "all",
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "class_balance",
                    "reason": "insufficient_class_trials",
                    "detail": f"n_a={n_a}, n_b={n_b}; need >=5 in each class",
                }
            ],
        }

    items = []
    qc_rows = []
    for window, window_start_sec, window_end_sec in subwindows:
        time_mask = (times >= window_start_sec) & (times <= window_end_sec)
        if int(np.sum(time_mask)) < 2:
            qc_rows.append(
                {
                    "window": window,
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "window_select",
                    "reason": "insufficient_timepoints",
                    "detail": f"window={window_start_sec}-{window_end_sec}",
                }
            )
            continue
        X_win = X[:, :, time_mask]
        X_flat = X_win.reshape(X_win.shape[0], X_win.shape[1] * X_win.shape[2])
        items.append(
            {
                "subject": subject,
                "day": day,
                "session_file": session_file,
                "X": X_flat,
                "y": y,
                "n_trials": n_trials,
                "n_a": n_a,
                "n_b": n_b,
                "n_channels": int(X_win.shape[1]),
                "n_timepoints": int(X_win.shape[2]),
                "window": window,
                "window_start_sec": float(window_start_sec),
                "window_end_sec": float(window_end_sec),
            }
        )
    return {"ok": bool(items), "items": items, "qc": qc_rows}


def _run_parallel_or_serial(jobs, n_workers, label):
    if n_workers == 1:
        out = []
        for func, args, kwargs in jobs:
            out.append(func(*args, **kwargs))
        return out
    if threadpool_limits is None:
        return Parallel(n_jobs=n_workers, backend="loky", verbose=0)(jobs)
    with threadpool_limits(limits=1):
        return Parallel(n_jobs=n_workers, backend="loky", verbose=0)(jobs)


def run_mvpa_stim_locked_cat_late_subwindow_transfer(
    output_dir: Path | str = OUTPUT_DIR,
    min_epochs: int = 20,
    random_state: int = RANDOM_STATE,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = int(os.environ.get("N_JOBS", N_JOBS))
    n_workers = max(1, int(n_workers))

    mne.set_log_level("ERROR")
    t0 = time.time()
    progress_json = (
        output_dir / "mvpa_stim_locked_cat_late_subwindow_transfer_progress.json"
    )

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

    sessions = load_sessions()
    tasks = [
        {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "epo_file": item["epo_file"],
            "epo_path": str(item["epo_path"]),
            "min_epochs": int(min_epochs),
            "subwindows": LATE_SUBWINDOWS,
        }
        for item in sessions
    ]
    print(
        "[MVPA late-subwindow transfer] Preparing "
        f"{len(tasks)} sessions once for {len(LATE_SUBWINDOWS)} windows "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    write_progress("prepare", 0, len(tasks))
    prepare_jobs = [delayed(prepare_late_subwindow_session)(task) for task in tasks]
    prepared_results = _run_parallel_or_serial(prepare_jobs, n_workers, "prepare")
    write_progress("prepare", len(tasks), len(tasks))

    subject_window_days = {}
    qc_rows = []
    for result in prepared_results:
        qc_rows.extend(result["qc"])
        for item in result["items"]:
            window = str(item["window"])
            subject = int(item["subject"])
            day = int(item["day"])
            subject_window_days.setdefault(window, {})
            subject_window_days[window].setdefault(subject, {})
            subject_window_days[window][subject][day] = item

    transfer_specs = []
    for window in [row[0] for row in LATE_SUBWINDOWS]:
        for subject, day_data in sorted(subject_window_days.get(window, {}).items()):
            transfer_specs.append((window, int(subject), day_data))
    print(
        "[MVPA late-subwindow transfer] Computing "
        f"{len(transfer_specs)} subject/window transfer jobs "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    write_progress("transfer", 0, len(transfer_specs))
    transfer_jobs = [
        delayed(fit_transfer_subject)(subject, day_data, random_state)
        for _window, subject, day_data in transfer_specs
    ]
    transfer_results = _run_parallel_or_serial(transfer_jobs, n_workers, "transfer")
    write_progress("transfer", len(transfer_specs), len(transfer_specs))

    subject_rows = []
    for (window, _subject, _day_data), (rows, new_qc) in zip(
        transfer_specs, transfer_results
    ):
        subject_rows.extend(rows)
        for row in new_qc:
            row = row.copy()
            row["window"] = window
            qc_rows.append(row)
    subject_df = pd.DataFrame(subject_rows)
    qc_df = pd.DataFrame(qc_rows, columns=QC_COLUMNS)
    if subject_df.empty:
        raise RuntimeError("Late-subwindow transfer produced no subject rows")

    paths = {}
    for window, _start_sec, _end_sec in LATE_SUBWINDOWS:
        stem = f"mvpa_stim_locked_cat_{window}_window_transfer"
        subject_csv = output_dir / f"{stem}_subject_pairs.csv"
        group_csv = output_dir / f"{stem}_group_pairs.csv"
        qc_csv = output_dir / f"{stem}_qc_log.csv"

        subject_win = subject_df[subject_df["window"] == window].copy()
        qc_win = qc_df[(qc_df["window"] == window) | (qc_df["window"] == "all")].copy()
        if subject_win.empty:
            subject_win.to_csv(subject_csv, index=False)
            qc_win.to_csv(qc_csv, index=False)
            raise RuntimeError(f"{window} transfer produced no subject rows")
        group_win = make_group_summary(subject_win.dropna(subset=["auc"]))

        subject_win.to_csv(subject_csv, index=False)
        group_win.to_csv(group_csv, index=False)
        qc_win.to_csv(qc_csv, index=False)
        paths[window] = {
            "subject_csv": subject_csv,
            "group_csv": group_csv,
            "qc_csv": qc_csv,
        }
        print(f"[MVPA late-subwindow transfer] Wrote {subject_csv}", flush=True)
        print(f"[MVPA late-subwindow transfer] Wrote {group_csv}", flush=True)
        print(f"[MVPA late-subwindow transfer] Wrote {qc_csv}", flush=True)
    write_progress("completed", len(transfer_specs), len(transfer_specs))
    return paths


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_late_subwindow_transfer()
