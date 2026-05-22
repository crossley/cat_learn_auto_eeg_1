#!/usr/bin/env python3
"""Compute core grand-average ERP outputs."""

from pathlib import Path
import json
import os
import time

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

from load_project_data import align_behaviour_to_epochs, load_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
N_JOBS = 8


def evoked_map_to_long(evoked_map, lock_type, condition):
    rows = []
    for day, evoked in evoked_map.items():
        arr = evoked.data
        for i_ch, ch in enumerate(evoked.ch_names):
            rows.append(
                pd.DataFrame(
                    {
                        "day": int(day),
                        "lock_type": lock_type,
                        "condition": condition,
                        "channel": ch,
                        "time_s": evoked.times,
                        "amplitude_v": arr[i_ch, :],
                    }
                )
            )
    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "day",
                "lock_type",
                "condition",
                "channel",
                "time_s",
                "amplitude_v",
            ]
        )
    return pd.concat(rows, ignore_index=True)


def process_erp_session(session_item):
    subject = session_item["subject"]
    day = session_item["day"]
    beh_df = session_item["beh"].copy()
    if "epo_path" in session_item:
        epochs = mne.read_epochs(session_item["epo_path"], preload=False, verbose="ERROR")
    else:
        epochs = session_item["epochs"]
    epochs_stim_all, beh_aligned = align_behaviour_to_epochs(
        beh_df, epochs, event_names=("Stim/A", "Stim/B")
    )
    if len(epochs_stim_all) == 0:
        return None
    epochs_fb_all, beh_fb_aligned = align_behaviour_to_epochs(
        beh_df, epochs, event_names=("FB/Cor", "FB/Inc")
    )

    fb_use = beh_aligned["fb"].astype(str).str.lower().to_numpy()
    cat_use = beh_aligned["cat"].astype(str).to_numpy()
    fb_epoch_use = beh_fb_aligned["fb"].astype(str).str.lower().to_numpy()
    fb_cat_use = beh_fb_aligned["cat"].astype(str).to_numpy()

    idx_cor = np.where(fb_use == "correct")[0]
    idx_inc = np.where(fb_use == "incorrect")[0]
    idx_cat_a = np.where(cat_use == "A")[0]
    idx_cat_b = np.where(cat_use == "B")[0]
    idx_fb_cor = np.where(fb_epoch_use == "correct")[0]
    idx_fb_inc = np.where(fb_epoch_use == "incorrect")[0]
    idx_fb_cat_a = np.where(fb_cat_use == "A")[0]
    idx_fb_cat_b = np.where(fb_cat_use == "B")[0]

    result = {
        "subject": subject,
        "day": day,
        "evoked_stim_all": epochs_stim_all.average(),
    }
    if len(epochs_fb_all) > 0:
        result["evoked_feedback_all"] = epochs_fb_all.average()
    if len(idx_cor) > 0:
        result["evoked_stim_cor"] = epochs_stim_all[idx_cor].average()
    if len(idx_inc) > 0:
        result["evoked_stim_inc"] = epochs_stim_all[idx_inc].average()
    if len(idx_cat_a) > 0:
        result["evoked_stim_cat_a"] = epochs_stim_all[idx_cat_a].average()
    if len(idx_cat_b) > 0:
        result["evoked_stim_cat_b"] = epochs_stim_all[idx_cat_b].average()
    if len(idx_fb_cor) > 0:
        result["evoked_feedback_cor"] = epochs_fb_all[idx_fb_cor].average()
    if len(idx_fb_inc) > 0:
        result["evoked_feedback_inc"] = epochs_fb_all[idx_fb_inc].average()
    if len(idx_fb_cat_a) > 0:
        result["evoked_feedback_cat_a"] = epochs_fb_all[idx_fb_cat_a].average()
    if len(idx_fb_cat_b) > 0:
        result["evoked_feedback_cat_b"] = epochs_fb_all[idx_fb_cat_b].average()
    return result


def process_erp_session_safe(session_item):
    subject = session_item["subject"]
    day = session_item["day"]
    try:
        result = process_erp_session(session_item)
        if result is None:
            return {
                "ok": False,
                "qc": {
                    "subject": subject,
                    "day": day,
                    "stage": "erp_session",
                    "reason": "no_stim_epochs",
                    "detail": "",
                },
            }
        result["ok"] = True
        return result
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "subject": subject,
                "day": day,
                "stage": "erp_session",
                "reason": "extract_error",
                "detail": str(exc),
            },
        }


def _grand_average_records(records, value_key):
    out = {}
    d = pd.DataFrame(records)
    if d.empty:
        return out
    for day, g in d.groupby("day"):
        out[day] = mne.grand_average(g[value_key].tolist())
    return out


def run_erp_grand_average(
    output_dir: Path | str = OUTPUT_DIR,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d_grand_path = output_dir / "erp_grand_average_by_day_lock_condition.csv"
    d_subject_path = output_dir / "erp_grand_average_subject_day_all.csv"
    qc_path = output_dir / "erp_grand_average_qc.csv"
    progress_json = output_dir / "erp_grand_average_progress.json"
    t0 = time.time()

    def write_progress(stage: str, done: int = 0, total: int = 0):
        payload = {
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

    sessions = load_sessions()
    worker_items = []
    for item in sessions:
        worker_items.append(
            {
            "subject": item["subject"],
            "day": item["day"],
            "beh": item["beh"],
            "epo_path": str(item["epo_path"]),
            }
        )
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    write_progress("running", 0, len(worker_items))
    def iter_session_jobs():
        for item in worker_items:
            yield delayed(process_erp_session_safe)(item)

    if n_workers == 1:
        session_results = []
        for i, item in enumerate(worker_items, start=1):
            session_results.append(process_erp_session_safe(item))
            write_progress("running", i, len(worker_items))
    else:
        result_iter = Parallel(
            n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
        )(iter_session_jobs())
        session_results = []
        for i, result in enumerate(result_iter, start=1):
            session_results.append(result)
            write_progress("running", i, len(worker_items))

    qc_rows = []
    records = {
        "evoked_stim_all": [],
        "evoked_stim_cor": [],
        "evoked_stim_inc": [],
        "evoked_stim_cat_a": [],
        "evoked_stim_cat_b": [],
        "evoked_feedback_all": [],
        "evoked_feedback_cor": [],
        "evoked_feedback_inc": [],
        "evoked_feedback_cat_a": [],
        "evoked_feedback_cat_b": [],
    }
    for result in session_results:
        if result is None:
            continue
        if not result.get("ok", True):
            qc_rows.append(result["qc"])
            continue
        subject = result["subject"]
        day = result["day"]
        for key in records:
            if key in result:
                records[key].append({"subject": subject, "day": day, key: result[key]})
    pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

    means = {}
    for key, vals in records.items():
        means[key] = _grand_average_records(vals, key)
    d_grand = pd.concat(
        [
            evoked_map_to_long(means["evoked_stim_all"], "stim", "all"),
            evoked_map_to_long(means["evoked_stim_cor"], "stim", "correct"),
            evoked_map_to_long(means["evoked_stim_inc"], "stim", "incorrect"),
            evoked_map_to_long(means["evoked_stim_cat_a"], "stim", "cat_a"),
            evoked_map_to_long(means["evoked_stim_cat_b"], "stim", "cat_b"),
            evoked_map_to_long(means["evoked_feedback_all"], "feedback", "all"),
            evoked_map_to_long(means["evoked_feedback_cor"], "feedback", "correct"),
            evoked_map_to_long(means["evoked_feedback_inc"], "feedback", "incorrect"),
            evoked_map_to_long(means["evoked_feedback_cat_a"], "feedback", "cat_a"),
            evoked_map_to_long(means["evoked_feedback_cat_b"], "feedback", "cat_b"),
        ],
        ignore_index=True,
    ).sort_values(["lock_type", "condition", "day", "channel", "time_s"])
    d_grand.to_csv(d_grand_path, index=False)

    stim_all = pd.DataFrame(records["evoked_stim_all"])
    feedback_all = pd.DataFrame(records["evoked_feedback_all"])
    subject_rows = []
    if not stim_all.empty:
        for subject in sorted(stim_all["subject"].unique()):
            stim_map = {}
            for _, row in stim_all[stim_all["subject"] == subject].iterrows():
                stim_map[int(row["day"])] = row["evoked_stim_all"]
            d_stim_s = evoked_map_to_long(stim_map, "stim", "all")
            d_feedback_s = pd.DataFrame()
            if not feedback_all.empty:
                feedback_map = {}
                for _, row in feedback_all[
                    feedback_all["subject"] == subject
                ].iterrows():
                    feedback_map[int(row["day"])] = row["evoked_feedback_all"]
                d_feedback_s = evoked_map_to_long(feedback_map, "feedback", "all")
            d_sub_s = pd.concat([d_stim_s, d_feedback_s], ignore_index=True)
            if not d_sub_s.empty:
                subject_rows.append(d_sub_s.assign(subject=int(subject)))
    if len(subject_rows) == 0:
        d_subject = pd.DataFrame(
            columns=[
                "subject",
                "day",
                "lock_type",
                "condition",
                "channel",
                "time_s",
                "amplitude_v",
            ]
        )
    else:
        d_subject = pd.concat(subject_rows, ignore_index=True).sort_values(
            ["subject", "day", "lock_type", "channel", "time_s"]
        )
    d_subject.to_csv(d_subject_path, index=False)
    write_progress("completed", len(worker_items), len(worker_items))
    return {
        "grand_average_csv": d_grand_path,
        "subject_csv": d_subject_path,
        "qc_csv": qc_path,
        "progress_json": progress_json,
    }


if __name__ == "__main__":
    run_erp_grand_average()
