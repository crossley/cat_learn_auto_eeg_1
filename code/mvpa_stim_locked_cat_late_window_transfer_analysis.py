#!/usr/bin/env python3
"""Late-window stimulus-locked category transfer across training days."""

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
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from load_project_data import load_sessions
from mvpa_stim_locked_cat_late_window_analysis import (
    CLASSIFIERS,
    DAYS,
    OUTPUT_DIR,
    RANDOM_STATE,
    WINDOW,
    WINDOW_END_SEC,
    WINDOW_START_SEC,
    build_window_clf,
    sem,
)
from util_mvpa import pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8


def classifier_scores(clf, X):
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X)
    elif hasattr(clf, "predict_proba"):
        scores = clf.predict_proba(X)[:, 1]
    else:
        raise ValueError("Classifier has neither decision_function nor predict_proba")
    return np.asarray(scores, dtype=float)


def prepare_late_window_session(task):
    session_file = task["epo_file"]
    subject = int(task["subject"])
    day = int(task["day"])
    min_epochs = int(task["min_epochs"])

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_events = []
        for event_name in ["Stim/A", "Stim/B"]:
            if event_name in epochs.event_id:
                stim_events.append(event_name)
        if len(stim_events) < 2:
            return {
                "ok": False,
                "qc": {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "event_select",
                    "reason": "missing_stim_labels",
                    "detail": ",".join(stim_events),
                },
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
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "preprocess",
                "reason": "prep_error",
                "detail": str(exc),
            },
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
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "epoch_count",
                "reason": "insufficient_epochs",
                "detail": f"n_trials={n_trials} < min_epochs={min_epochs}",
            },
        }
    if min(n_a, n_b) < 5:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "class_balance",
                "reason": "insufficient_class_trials",
                "detail": f"n_a={n_a}, n_b={n_b}; need >=5 in each class",
            },
        }

    time_mask = (times >= WINDOW_START_SEC) & (times <= WINDOW_END_SEC)
    if int(np.sum(time_mask)) < 2:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "window_select",
                "reason": "insufficient_timepoints",
                "detail": f"window={WINDOW_START_SEC}-{WINDOW_END_SEC}",
            },
        }
    X_win = X[:, :, time_mask]
    X_flat = X_win.reshape(X_win.shape[0], X_win.shape[1] * X_win.shape[2])
    return {
        "ok": True,
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
    }


def fit_transfer_subject(subject, day_data, random_state):
    rows = []
    qc_rows = []
    for classifier in CLASSIFIERS:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        for train_day in DAYS:
            if train_day not in day_data:
                continue
            train_item = day_data[train_day]
            X_train = train_item["X"]
            y_train = train_item["y"]
            for test_day in DAYS:
                if test_day not in day_data:
                    continue
                test_item = day_data[test_day]
                if train_day == test_day:
                    clf = build_window_clf(classifier, random_state=random_state)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", SklearnConvergenceWarning)
                            scores = cross_val_score(
                                clf,
                                X_train,
                                y_train,
                                cv=cv,
                                scoring="roc_auc",
                            )
                        auc = float(np.mean(scores))
                        status = "cv"
                        detail = ""
                    except Exception as exc:
                        auc = np.nan
                        status = "error"
                        detail = str(exc)
                else:
                    clf = build_window_clf(classifier, random_state=random_state)
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", SklearnConvergenceWarning)
                            clf.fit(X_train, y_train)
                        scores = classifier_scores(clf, test_item["X"])
                        auc = float(roc_auc_score(test_item["y"], scores))
                        status = "transfer"
                        detail = ""
                    except Exception as exc:
                        auc = np.nan
                        status = "error"
                        detail = str(exc)
                if status == "error":
                    qc_rows.append(
                        {
                            "subject": int(subject),
                            "classifier": classifier,
                            "train_day": int(train_day),
                            "test_day": int(test_day),
                            "stage": "transfer",
                            "reason": "compute_error",
                            "detail": detail,
                        }
                    )
                rows.append(
                    {
                        "subject": int(subject),
                        "classifier": classifier,
                        "train_day": int(train_day),
                        "test_day": int(test_day),
                        "day_distance": int(abs(train_day - test_day)),
                        "window": WINDOW,
                        "window_start_sec": float(WINDOW_START_SEC),
                        "window_end_sec": float(WINDOW_END_SEC),
                        "auc": auc,
                        "fit_status": status,
                        "train_n_trials": int(train_item["n_trials"]),
                        "test_n_trials": int(test_item["n_trials"]),
                        "train_n_a": int(train_item["n_a"]),
                        "train_n_b": int(train_item["n_b"]),
                        "test_n_a": int(test_item["n_a"]),
                        "test_n_b": int(test_item["n_b"]),
                    }
                )
    return rows, qc_rows


def make_group_summary(subject_df):
    rows = []
    for (classifier, train_day, test_day), g in subject_df.groupby(
        ["classifier", "train_day", "test_day"]
    ):
        rows.append(
            {
                "classifier": classifier,
                "train_day": int(train_day),
                "test_day": int(test_day),
                "day_distance": int(abs(train_day - test_day)),
                "window": WINDOW,
                "window_start_sec": float(WINDOW_START_SEC),
                "window_end_sec": float(WINDOW_END_SEC),
                "auc_mean": float(np.mean(g["auc"])),
                "auc_sem": sem(g["auc"]),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    out = pd.DataFrame(rows).sort_values(["classifier", "train_day", "test_day"])
    if out.empty:
        raise ValueError("No late-window transfer group rows were produced")
    return out


def run_mvpa_stim_locked_cat_late_window_transfer(
    output_dir: Path | str = OUTPUT_DIR,
    min_epochs: int = 20,
    random_state: int = RANDOM_STATE,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )

    subject_csv = output_dir / "mvpa_stim_locked_cat_late_window_transfer_subject_pairs.csv"
    group_csv = output_dir / "mvpa_stim_locked_cat_late_window_transfer_group_pairs.csv"
    qc_csv = output_dir / "mvpa_stim_locked_cat_late_window_transfer_qc_log.csv"
    progress_json = output_dir / "mvpa_stim_locked_cat_late_window_transfer_progress.json"

    qc_columns = [
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
    t0 = time.time()

    def write_progress(stage, done, total):
        payload = {
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

    sessions = load_sessions()
    tasks = []
    for item in sessions:
        tasks.append(
            {
                "subject": int(item["subject"]),
                "day": int(item["day"]),
                "epo_file": item["epo_file"],
                "epo_path": str(item["epo_path"]),
                "min_epochs": int(min_epochs),
            }
        )
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    prepared = []
    qc_rows = []
    write_progress("prepare", 0, len(tasks))
    print(
        f"[MVPA late-window transfer] Preparing {len(tasks)} sessions "
        f"(n_workers={n_workers}, window={WINDOW_START_SEC}-{WINDOW_END_SEC}s)...",
        flush=True,
    )

    def iter_prepare_jobs():
        for task in tasks:
            yield delayed(prepare_late_window_session)(task)

    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            result = prepare_late_window_session(task)
            if result["ok"]:
                prepared.append(result)
            else:
                qc_rows.append(result["qc"])
            write_progress("prepare", done, len(tasks))
    elif threadpool_limits is None:
        result_iter = Parallel(
            n_jobs=n_workers,
            backend="loky",
            verbose=0,
            return_as="generator_unordered",
        )(iter_prepare_jobs())
        for done, result in enumerate(result_iter, start=1):
            if result["ok"]:
                prepared.append(result)
            else:
                qc_rows.append(result["qc"])
            write_progress("prepare", done, len(tasks))
            if (done % 5) == 0:
                elapsed = time.time() - t0
                print(
                    f"[MVPA late-window transfer] prepared {done}/{len(tasks)} "
                    f"sessions (elapsed {elapsed/60:.1f} min)",
                    flush=True,
                )
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(
                n_jobs=n_workers,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(iter_prepare_jobs())
            for done, result in enumerate(result_iter, start=1):
                if result["ok"]:
                    prepared.append(result)
                else:
                    qc_rows.append(result["qc"])
                write_progress("prepare", done, len(tasks))
                if (done % 5) == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[MVPA late-window transfer] prepared {done}/{len(tasks)} "
                        f"sessions (elapsed {elapsed/60:.1f} min)",
                        flush=True,
                    )

    subject_days = {}
    for item in prepared:
        subject = int(item["subject"])
        day = int(item["day"])
        if subject not in subject_days:
            subject_days[subject] = {}
        subject_days[subject][day] = item

    subject_rows = []
    subjects = sorted(subject_days.keys())
    write_progress("transfer", 0, len(subjects))
    print(
        f"[MVPA late-window transfer] Computing transfer for {len(subjects)} "
        "subjects...",
        flush=True,
    )
    for done, subject in enumerate(subjects, start=1):
        rows, new_qc = fit_transfer_subject(subject, subject_days[subject], random_state)
        subject_rows.extend(rows)
        for row in new_qc:
            qc_rows.append(row)
        write_progress("transfer", done, len(subjects))
        if (done % 5) == 0:
            elapsed = time.time() - t0
            print(
                f"[MVPA late-window transfer] complete {done}/{len(subjects)} "
                f"subjects (elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    subject_df = pd.DataFrame(subject_rows)
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)
    if subject_df.empty:
        subject_df.to_csv(subject_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        raise RuntimeError("Late-window transfer produced no subject rows")
    group_df = make_group_summary(subject_df.dropna(subset=["auc"]))

    subject_df.to_csv(subject_csv, index=False)
    group_df.to_csv(group_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    write_progress("completed", len(subjects), len(subjects))

    print(f"[MVPA late-window transfer] Wrote {subject_csv}")
    print(f"[MVPA late-window transfer] Wrote {group_csv}")
    print(f"[MVPA late-window transfer] Wrote {qc_csv}")
    return {
        "subject_df": subject_df,
        "group_df": group_df,
        "qc_df": qc_df,
        "subject_csv": subject_csv,
        "group_csv": group_csv,
        "qc_csv": qc_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_late_window_transfer()
