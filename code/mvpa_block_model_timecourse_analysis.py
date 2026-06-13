#!/usr/bin/env python3
"""25-block stimulus category-transfer model-evidence timecourse."""

from __future__ import annotations

from pathlib import Path
import os
import warnings

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from block_model_utils import assign_accumulated_blocks, score_payload, write_scores
from load_project_data import align_behaviour_to_epochs, load_sessions
from mvpa_stim_locked_cat_time_resolved_analysis import build_clf
from util_mvpa import pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
N_JOBS = 4
MIN_CLASS_TRIALS = 5


def load_subject_blocks(subject, items, random_state):
    block_data = {}
    times_ref = None
    for item in items:
        epochs = mne.read_epochs(item["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh = align_behaviour_to_epochs(
            item["beh"],
            epochs,
            event_names=("Stim/A", "Stim/B"),
        )
        stim_epochs = stim_epochs.copy().load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        stim_epochs.resample(128, npad="auto")
        times = stim_epochs.times.copy()
        if times_ref is None:
            times_ref = times
        elif len(times_ref) != len(times) or not np.allclose(times_ref, times):
            raise ValueError(f"Time mismatch for subject={subject}")
        codes = stim_epochs.events[:, 2]
        y = np.full(len(codes), -1, dtype=int)
        y[codes == stim_epochs.event_id["Stim/A"]] = 0
        y[codes == stim_epochs.event_id["Stim/B"]] = 1
        keep = y >= 0
        y = y[keep]
        X = stim_epochs.get_data()[keep]
        beh = beh.copy()
        beh["day"] = int(item["day"])
        beh = assign_accumulated_blocks(beh)
        beh = beh.iloc[np.where(keep)[0]].reset_index(drop=True)
        for block, idx in beh.groupby("block").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            y_block = y[idx]
            if min(np.sum(y_block == 0), np.sum(y_block == 1)) < MIN_CLASS_TRIALS:
                continue
            block_data[int(block)] = {"X": X[idx], "y": y_block}
    return block_data, times_ref


def auc_train_test(X_train, y_train, X_test, y_test, random_state):
    if min(np.sum(y_train == 0), np.sum(y_train == 1)) < MIN_CLASS_TRIALS:
        return np.nan
    if min(np.sum(y_test == 0), np.sum(y_test == 1)) < MIN_CLASS_TRIALS:
        return np.nan
    clf = build_clf(random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X_train, y_train)
        score = clf.predict_proba(X_test)[:, 1]
    return float(roc_auc_score(y_test, score))


def auc_within_block(X, y, random_state):
    if min(np.sum(y == 0), np.sum(y == 1)) < MIN_CLASS_TRIALS:
        return np.nan
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    clf = build_clf(random_state=random_state)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return float(np.mean(scores))


def process_subject(task):
    subject = int(task["subject"])
    random_state = int(task["random_state"])
    try:
        block_data, times = load_subject_blocks(subject, task["items"], random_state)
        if times is None or len(block_data) < 6:
            raise ValueError("insufficient_blocks")
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "subject": subject,
                "reason": "prep_error",
                "detail": str(exc),
            },
        }
    blocks = sorted(block_data)
    rows = []
    for ti, time_sec in enumerate(times):
        pair_rows = []
        y_vals = []
        for block_i in blocks:
            Xi = block_data[block_i]["X"][:, :, ti]
            yi = block_data[block_i]["y"]
            for block_j in blocks:
                Xj = block_data[block_j]["X"][:, :, ti]
                yj = block_data[block_j]["y"]
                if block_i == block_j:
                    auc = auc_within_block(Xi, yi, random_state)
                else:
                    auc = auc_train_test(Xi, yi, Xj, yj, random_state)
                if np.isfinite(auc):
                    pair_rows.append({"block_i": block_i, "block_j": block_j})
                    y_vals.append(float(auc))
        if len(y_vals) >= 40:
            rows.extend(
                score_payload(
                    subject,
                    float(time_sec),
                    pair_rows,
                    y_vals,
                    include_same_block=True,
                )
            )
        if ((ti + 1) % 25) == 0:
            print(f"[MVPA block model] subject {subject}: time {ti + 1}/{len(times)}", flush=True)
    return {"ok": True, "rows": rows}


def run_mvpa_block_model_timecourse(output_dir=OUTPUT_DIR, random_state=42, n_workers=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    sessions = load_sessions()
    by_subject = {}
    for item in sessions:
        by_subject.setdefault(int(item["subject"]), []).append(item)
    tasks = [
        {"subject": subject, "items": items, "random_state": int(random_state)}
        for subject, items in sorted(by_subject.items())
    ]
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    print(
        f"[MVPA block model] Running {len(tasks)} subjects (n_workers={n_workers})",
        flush=True,
    )
    def iter_jobs():
        for task in tasks:
            yield delayed(process_subject)(task)
    if n_workers == 1:
        results = [process_subject(task) for task in tasks]
    elif threadpool_limits is None:
        results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(iter_jobs())
    else:
        with threadpool_limits(limits=1):
            results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(iter_jobs())
    rows = []
    qc_rows = []
    for result in results:
        if result["ok"]:
            rows.extend(result["rows"])
        else:
            qc_rows.append(result["qc"])
    subject_path = output_dir / "mvpa_block_model_timecourse_subject.csv"
    summary_path = output_dir / "mvpa_block_model_timecourse_summary.csv"
    write_scores(rows, subject_path, summary_path)
    pd.DataFrame(qc_rows, columns=["subject", "reason", "detail"]).to_csv(
        output_dir / "mvpa_block_model_timecourse_qc_log.csv",
        index=False,
    )
    print(f"[MVPA block model] wrote {subject_path}", flush=True)
    print(f"[MVPA block model] wrote {summary_path}", flush=True)
    return {"subject": subject_path, "summary": summary_path}


if __name__ == "__main__":
    run_mvpa_block_model_timecourse()
