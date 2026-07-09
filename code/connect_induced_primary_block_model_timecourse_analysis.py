#!/usr/bin/env python3
"""25-block model evidence for total-vs-induced primary dwPLI."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np
import pandas as pd

from analysis_utils import parallel_collect
from block_model_utils import (
    assign_accumulated_blocks,
    score_payload,
    vector_corr,
    write_grouped_scores,
)
from connect_induced_primary_edge_timecourse_analysis import OUTPUT_PREFIX
from connect_multimeasure_utils import (
    OUTPUT_DIR,
    PRIMARY_BANDS,
    PRIMARY_MEASURE,
    STEP_SEC,
    STIM_TMAX,
    STIM_TMIN,
    compute_connectivity_rows,
    strict_roi_channels,
    strict_roi_edge_table,
    subtract_condition_evoked,
)
from load_project_data import align_behaviour_to_epochs, load_sessions

N_JOBS = 2
MIN_BLOCK_TRIALS = 5


def process_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    channel_subset = task["channel_subset"]
    edge_df = pd.DataFrame(task["edge_rows"])
    try:
        epochs_all = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh = align_behaviour_to_epochs(
            task["beh"], epochs_all, event_names=("Stim/A", "Stim/B")
        )
        stim_epochs = stim_epochs.load_data()
        missing = [ch for ch in channel_subset if ch not in stim_epochs.ch_names]
        if missing:
            raise ValueError("missing_channels:" + ",".join(missing))
        stim_epochs.pick(channel_subset)
        induced_epochs = subtract_condition_evoked(stim_epochs, ("Stim/A", "Stim/B"))
        beh = beh.copy()
        beh["day"] = day
        beh = assign_accumulated_blocks(beh)
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "subject": subject,
                "day": day,
                "session_file": task["epo_file"],
                "reason": "setup_error",
                "detail": str(exc),
            },
        }

    vectors = {}
    try:
        for block, idx in beh.groupby("block").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) < MIN_BLOCK_TRIALS:
                continue
            for signal_estimate, epochs in [
                ("total", stim_epochs[idx]),
                ("induced", induced_epochs[idx]),
            ]:
                rows = compute_connectivity_rows(
                    epochs=epochs,
                    edge_df=edge_df,
                    measures=[PRIMARY_MEASURE],
                    bands=task["bands"],
                    lock_type="stim",
                    tmin=float(task["stim_tmin"]),
                    tmax=float(task["stim_tmax"]),
                    step_sec=float(task["step_sec"]),
                    n_jobs=1,
                )
                for key, group in rows.groupby(["measure", "band", "lock_time"]):
                    measure, band, lock_time = key
                    vals = group.sort_values(["roi_pair", "ch_i", "ch_j"])[
                        "conn_val"
                    ].to_numpy(dtype=np.float32)
                    vectors[
                        (
                            signal_estimate,
                            str(measure),
                            str(band),
                            subject,
                            int(block),
                            float(lock_time),
                        )
                    ] = vals
        return {"ok": True, "vectors": vectors}
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
        raise ValueError("No sessions selected for induced primary block model")

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
                "bands": dict(PRIMARY_BANDS),
                "step_sec": STEP_SEC,
                "stim_tmin": STIM_TMIN,
                "stim_tmax": STIM_TMAX,
            }
        )
    return tasks


def score_vectors(vectors):
    by_group = {}
    for key, vec in vectors.items():
        signal_estimate, measure, band, subject, block, time_sec = key
        by_group.setdefault((signal_estimate, measure, band), {}).setdefault(
            time_sec, {}
        ).setdefault(subject, {})[block] = vec

    rows = []
    for signal_estimate, measure, band in sorted(by_group):
        by_time = by_group[(signal_estimate, measure, band)]
        times = sorted(by_time)
        for time_i, time_sec in enumerate(times, start=1):
            for subject, block_map in sorted(by_time[time_sec].items()):
                blocks = sorted(block_map)
                pair_rows = []
                y_vals = []
                for i, block_i in enumerate(blocks):
                    for block_j in blocks[i + 1 :]:
                        val = vector_corr(block_map[block_i], block_map[block_j])
                        if np.isfinite(val):
                            pair_rows.append(
                                {"block_i": int(block_i), "block_j": int(block_j)}
                            )
                            y_vals.append(float(val))
                if len(y_vals) >= 20:
                    scored = score_payload(subject, time_sec, pair_rows, y_vals)
                    for row in scored:
                        row["signal_estimate"] = signal_estimate
                        row["measure"] = measure
                        row["band"] = band
                    rows.extend(scored)
            if (time_i % 10) == 0:
                print(
                    "[connect induced primary block] "
                    f"{signal_estimate} times {time_i}/{len(times)}",
                    flush=True,
                )
    if not rows:
        raise ValueError("No induced primary block-model rows were produced")
    return rows


def run_connect_induced_primary_block_model_timecourse(
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
    tasks = make_tasks(subject_ids=subject_ids, max_sessions=max_sessions)
    results = []
    qc_rows = []
    for batch_start in range(0, len(tasks), n_workers):
        batch = tasks[batch_start : batch_start + n_workers]
        batch_results = parallel_collect(process_session, batch, n_workers)
        results.extend(batch_results)
        print(
            "[connect induced primary block] "
            f"sessions {min(batch_start + n_workers, len(tasks))}/{len(tasks)}",
            flush=True,
        )

    vectors = {}
    for result in results:
        if result["ok"]:
            vectors.update(result["vectors"])
        else:
            qc_rows.append(result["qc"])
    if qc_rows:
        pd.DataFrame(qc_rows).to_csv(
            output_dir / f"{output_prefix}_block_model_qc_log.csv", index=False
        )
    if not vectors:
        raise ValueError("No induced primary block vectors were computed")

    rows = score_vectors(vectors)
    subject_path = output_dir / f"{output_prefix}_block_model_timecourse_subject.csv"
    summary_path = output_dir / f"{output_prefix}_block_model_timecourse_summary.csv"
    write_grouped_scores(
        rows,
        ["signal_estimate", "measure", "band"],
        subject_path,
        summary_path,
    )
    print(f"[connect induced primary block] wrote {subject_path}", flush=True)
    print(f"[connect induced primary block] wrote {summary_path}", flush=True)
    return {"subject": subject_path, "summary": summary_path}


if __name__ == "__main__":
    run_connect_induced_primary_block_model_timecourse()
