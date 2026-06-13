#!/usr/bin/env python3
"""Stimulus-locked category decoding in strict sensor ROIs."""

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

from load_project_data import load_sessions
from mvpa_stim_locked_cat_time_resolved_analysis import decode_timecourse
from sensor_rois import STRICT_SENSOR_ROIS
from util_mvpa import pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
N_JOBS = 8


def process_roi_mvpa_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    session_file = task["epo_file"]
    min_epochs = int(task["min_epochs"])
    random_state = int(task["random_state"])
    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_events = [event for event in ["Stim/A", "Stim/B"] if event in epochs.event_id]
        if len(stim_events) < 2:
            return {
                "ok": False,
                "qc": [{
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "roi": "all",
                    "stage": "event_select",
                    "reason": "missing_stim_labels",
                    "detail": ",".join(stim_events),
                }],
            }
        stim_epochs = epochs[stim_events].copy()
        stim_epochs.load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        stim_epochs.resample(128, npad="auto")
        codes = stim_epochs.events[:, 2]
        y = np.full(len(codes), -1, dtype=int)
        y[codes == stim_epochs.event_id["Stim/A"]] = 0
        y[codes == stim_epochs.event_id["Stim/B"]] = 1
        keep = y >= 0
        y = y[keep]
        X_all = stim_epochs.get_data()[keep]
        times = stim_epochs.times.copy()
        ch_names = list(stim_epochs.ch_names)
    except Exception as exc:
        return {
            "ok": False,
            "qc": [{
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "roi": "all",
                "stage": "preprocess",
                "reason": "prep_error",
                "detail": str(exc),
            }],
        }

    n_trials = int(len(y))
    n_a = int(np.sum(y == 0))
    n_b = int(np.sum(y == 1))
    qc_rows = []
    session_rows = []
    if n_trials < min_epochs or min(n_a, n_b) < 5:
        reason = "insufficient_epochs" if n_trials < min_epochs else "insufficient_class_trials"
        detail = f"n_trials={n_trials}, n_a={n_a}, n_b={n_b}"
        return {
            "ok": False,
            "qc": [{
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "roi": "all",
                "stage": "epoch_count",
                "reason": reason,
                "detail": detail,
            }],
        }

    for roi, roi_channels in STRICT_SENSOR_ROIS.items():
        missing = [ch for ch in roi_channels if ch not in ch_names]
        if missing:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "roi": roi,
                    "stage": "channel_select",
                    "reason": "missing_channels",
                    "detail": ",".join(missing),
                }
            )
            continue
        picks = [ch_names.index(ch) for ch in roi_channels]
        X = X_all[:, picks, :]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                auc = decode_timecourse(X, y, n_splits=5, random_state=random_state)
        except Exception as exc:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "roi": roi,
                    "stage": "decode",
                    "reason": "compute_error",
                    "detail": str(exc),
                }
            )
            continue
        for ti, auc_val in enumerate(auc):
            session_rows.append(
                {
                    "roi": roi,
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "time_sec": float(times[ti]),
                    "auc": float(auc_val),
                    "n_trials": n_trials,
                    "n_a": n_a,
                    "n_b": n_b,
                    "n_channels": int(len(picks)),
                }
            )
    return {"ok": True, "session_rows": session_rows, "qc": qc_rows}


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def run_mvpa_stim_locked_cat_roi_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    min_epochs: int = 20,
    random_state: int = 42,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    session_csv = output_dir / "mvpa_stim_locked_cat_roi_session_timecourse.csv"
    subject_day_csv = output_dir / "mvpa_stim_locked_cat_roi_subject_day_timecourse.csv"
    day_mean_csv = output_dir / "mvpa_stim_locked_cat_roi_day_means_timecourse.csv"
    qc_csv = output_dir / "mvpa_stim_locked_cat_roi_qc_log.csv"
    progress_json = output_dir / "mvpa_stim_locked_cat_roi_progress.json"
    t0 = time.time()

    sessions = load_sessions()
    tasks = [
        {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "epo_file": item["epo_file"],
            "epo_path": str(item["epo_path"]),
            "min_epochs": int(min_epochs),
            "random_state": int(random_state),
        }
        for item in sessions
    ]
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

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

    session_rows = []
    qc_rows = []

    def handle_result(result, done):
        if result.get("ok"):
            session_rows.extend(result.get("session_rows", []))
            qc_rows.extend(result.get("qc", []))
        else:
            qc_rows.extend(result.get("qc", []))
        write_progress("running", done, len(tasks))
        if (done % 5) == 0:
            print(
                f"[MVPA ROI] complete {done}/{len(tasks)} sessions "
                f"(elapsed {(time.time() - t0) / 60.0:.1f} min)",
                flush=True,
            )

    print(
        f"[MVPA ROI] Starting ROI decoding on {len(tasks)} sessions "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    write_progress("running", 0, len(tasks))

    def iter_jobs():
        for task in tasks:
            yield delayed(process_roi_mvpa_session)(task)

    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            handle_result(process_roi_mvpa_session(task), done)
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

    session_df = pd.DataFrame(session_rows)
    qc_df = pd.DataFrame(
        qc_rows,
        columns=["session_file", "subject", "day", "roi", "stage", "reason", "detail"],
    )
    if session_df.empty:
        session_df.to_csv(session_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        pd.DataFrame().to_csv(subject_day_csv, index=False)
        pd.DataFrame().to_csv(day_mean_csv, index=False)
        raise RuntimeError("MVPA ROI decoding produced no valid session rows")

    subject_day_df = (
        session_df.groupby(["roi", "subject", "day", "time_sec"], as_index=False)
        .agg(auc=("auc", "mean"), n_channels=("n_channels", "max"))
        .sort_values(["roi", "subject", "day", "time_sec"])
    )
    day_mean_df = (
        subject_day_df.groupby(["roi", "day", "time_sec"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sem=("auc", sem),
            n_subjects=("subject", "nunique"),
            n_channels=("n_channels", "max"),
        )
        .sort_values(["roi", "day", "time_sec"])
    )

    session_df.to_csv(session_csv, index=False)
    subject_day_df.to_csv(subject_day_csv, index=False)
    day_mean_df.to_csv(day_mean_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    write_progress("completed", len(tasks), len(tasks))
    print(f"[MVPA ROI] wrote {session_csv}", flush=True)
    print(f"[MVPA ROI] wrote {subject_day_csv}", flush=True)
    print(f"[MVPA ROI] wrote {day_mean_csv}", flush=True)
    print(f"[MVPA ROI] wrote {qc_csv}", flush=True)
    return {
        "session": session_csv,
        "subject_day": subject_day_csv,
        "day_mean": day_mean_csv,
        "qc": qc_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_roi_time_resolved()
