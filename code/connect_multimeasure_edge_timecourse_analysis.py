#!/usr/bin/env python3
"""Strict sensor-ROI edge connectivity for multi-measure robustness checks."""

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
import pandas as pd

from analysis_utils import parallel_collect
from connect_multimeasure_utils import (
    BANDS,
    FEEDBACK_TMAX,
    FEEDBACK_TMIN,
    MEASURES,
    OUTPUT_DIR,
    STEP_SEC,
    STIM_TMAX,
    STIM_TMIN,
    append_csv,
    compute_connectivity_rows,
    strict_roi_channels,
    strict_roi_edge_table,
)
from connect_sensorwide_analysis import channel_xy
from load_project_data import align_behaviour_to_epochs, load_sessions

N_JOBS = 2
OUTPUT_PREFIX = "connect_multimeasure"


def process_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    channel_subset = task["channel_subset"]
    edge_df = pd.DataFrame(task["edge_rows"])
    try:
        epochs_all = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, _ = align_behaviour_to_epochs(
            task["beh"], epochs_all, event_names=("Stim/A", "Stim/B")
        )
        feedback_epochs, _ = align_behaviour_to_epochs(
            task["beh"], epochs_all, event_names=("FB/Cor", "FB/Inc")
        )
        stim_epochs = stim_epochs.load_data()
        feedback_epochs = feedback_epochs.load_data()
        missing = [ch for ch in channel_subset if ch not in stim_epochs.ch_names]
        if missing:
            raise ValueError("missing_channels:" + ",".join(missing))
        stim_epochs.pick(channel_subset)
        feedback_epochs.pick(channel_subset)

        frames = []
        for lock_type, epochs, tmin, tmax in [
            ("stim", stim_epochs, task["stim_tmin"], task["stim_tmax"]),
            ("feedback", feedback_epochs, task["feedback_tmin"], task["feedback_tmax"]),
        ]:
            rows = compute_connectivity_rows(
                epochs=epochs,
                edge_df=edge_df,
                measures=task["measures"],
                bands=task["bands"],
                lock_type=lock_type,
                tmin=float(tmin),
                tmax=float(tmax),
                step_sec=float(task["step_sec"]),
                n_jobs=1,
            )
            rows.insert(0, "day", day)
            rows.insert(0, "subject", subject)
            rows["n_trials"] = len(epochs)
            frames.append(rows)

        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "rows": pd.concat(frames, ignore_index=True),
            "info": stim_epochs.info.copy(),
        }
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "subject": subject,
                "day": day,
                "session_file": task["epo_file"],
                "reason": "compute_error",
                "detail": str(exc),
            },
        }


def make_tasks(subject_ids=None, max_sessions=None):
    sessions = load_sessions()
    if subject_ids is not None:
        subject_ids = {int(subject) for subject in subject_ids}
        sessions = [item for item in sessions if int(item["subject"]) in subject_ids]
    if max_sessions is not None:
        sessions = sessions[: int(max_sessions)]
    if not sessions:
        raise ValueError("No sessions selected for multi-measure connectivity")

    channel_subset = strict_roi_channels()
    edge_df = strict_roi_edge_table(channel_subset)
    tasks = []
    for item in sessions:
        tasks.append(
            {
                "subject": int(item["subject"]),
                "day": int(item["day"]),
                "beh": item["beh"],
                "epo_path": str(item["epo_path"]),
                "epo_file": item["epo_file"],
                "channel_subset": channel_subset,
                "edge_rows": edge_df.to_dict("records"),
                "measures": list(MEASURES),
                "bands": dict(BANDS),
                "step_sec": STEP_SEC,
                "stim_tmin": STIM_TMIN,
                "stim_tmax": STIM_TMAX,
                "feedback_tmin": FEEDBACK_TMIN,
                "feedback_tmax": FEEDBACK_TMAX,
            }
        )
    return tasks, channel_subset


def run_connect_multimeasure_edge_timecourse(
    output_dir=OUTPUT_DIR,
    output_prefix=OUTPUT_PREFIX,
    n_workers=None,
    subject_ids=None,
    max_sessions=None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    mne.set_log_level("ERROR")
    tasks, channel_subset = make_tasks(
        subject_ids=subject_ids, max_sessions=max_sessions
    )
    edge_path = output_dir / f"{output_prefix}_edge_subject_timeseries.csv"
    channel_path = output_dir / f"{output_prefix}_channel_layout.csv"
    progress_path = output_dir / f"{output_prefix}_edge_progress.json"
    qc_path = output_dir / f"{output_prefix}_edge_qc_log.csv"

    if edge_path.exists():
        edge_path.unlink()
    if qc_path.exists():
        qc_path.unlink()

    t0 = time.time()
    write_header = True
    used = 0
    qc_rows = []
    info_subset = None
    for batch_start in range(0, len(tasks), n_workers):
        batch = tasks[batch_start : batch_start + n_workers]
        results = parallel_collect(process_session, batch, n_workers)
        for result in results:
            if not result["ok"]:
                qc_rows.append(result["qc"])
                continue
            used += 1
            if info_subset is None:
                info_subset = result["info"]
            write_header = append_csv(edge_path, result["rows"], write_header)
        print(
            "[connect multimeasure edges] "
            f"sessions {min(batch_start + n_workers, len(tasks))}/{len(tasks)}",
            flush=True,
        )

    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)
    if used == 0 or info_subset is None or not edge_path.exists():
        raise ValueError("No multi-measure edge rows were computed")

    xy = channel_xy(info_subset, channel_subset)
    pd.DataFrame({"channel": channel_subset, "x": xy[:, 0], "y": xy[:, 1]}).to_csv(
        channel_path, index=False
    )
    progress_path.write_text(
        json.dumps(
            {
                "sessions": len(tasks),
                "used": used,
                "skipped": len(qc_rows),
                "measures": MEASURES,
                "bands": list(BANDS.keys()),
                "edge_scope": "strict visual-frontal and visual-central sensor ROI pairs",
                "elapsed_sec": float(time.time() - t0),
            },
            indent=2,
        )
    )
    print(f"[connect multimeasure edges] wrote {edge_path}", flush=True)
    return {
        "edge_subject": edge_path,
        "channels": channel_path,
        "progress": progress_path,
        "qc": qc_path if qc_rows else None,
    }


if __name__ == "__main__":
    run_connect_multimeasure_edge_timecourse()
