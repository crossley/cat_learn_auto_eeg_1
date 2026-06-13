#!/usr/bin/env python3
"""25-block stimulus ERP model-evidence timecourse."""

from pathlib import Path
import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np

from analysis_utils import parallel_collect
from block_model_utils import assign_accumulated_blocks, score_payload, vector_corr, write_scores
from load_project_data import align_behaviour_to_epochs, load_sessions
from util_mvpa import pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
N_JOBS = 8
WINDOW_WIDTH_SEC = 0.050
WINDOW_STEP_SEC = 0.025


def process_session(item):
    try:
        epochs = mne.read_epochs(item["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh = align_behaviour_to_epochs(
            item["beh"], epochs, event_names=("Stim/A", "Stim/B")
        )
        stim_epochs = stim_epochs.copy().load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        stim_epochs.resample(128, npad="auto")
        beh = beh.copy()
        beh["day"] = int(item["day"])
        beh = assign_accumulated_blocks(beh)
        data = stim_epochs.get_data()
        out = {}
        for block, idx in beh.groupby("block").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) < 5:
                continue
            out[(int(item["subject"]), int(block))] = np.nanmean(data[idx], axis=0)
        return {"ok": True, "patterns": out, "times": stim_epochs.times.copy()}
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "subject": int(item["subject"]),
                "day": int(item["day"]),
                "session_file": item["epo_file"],
                "reason": "compute_error",
                "detail": str(exc),
            },
        }


def run_erp_block_model_timecourse(output_dir=OUTPUT_DIR, n_workers=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    sessions = load_sessions()
    if n_workers is None:
        n_workers = N_JOBS
    results = parallel_collect(process_session, sessions, max(1, int(n_workers)))
    patterns = {}
    times = None
    qc_rows = []
    for result in results:
        if not result["ok"]:
            qc_rows.append(result["qc"])
            continue
        patterns.update(result["patterns"])
        if times is None:
            times = result["times"]
    if times is None or not patterns:
        raise RuntimeError("No ERP block patterns were computed")

    subjects = sorted({subject for subject, _block in patterns})
    centers = np.arange(
        float(times.min()) + WINDOW_WIDTH_SEC / 2.0,
        float(times.max()) - WINDOW_WIDTH_SEC / 2.0 + WINDOW_STEP_SEC / 2.0,
        WINDOW_STEP_SEC,
    )
    rows = []
    for center_i, center in enumerate(centers, start=1):
        keep_time = (times >= center - WINDOW_WIDTH_SEC / 2.0) & (
            times <= center + WINDOW_WIDTH_SEC / 2.0
        )
        if int(np.sum(keep_time)) < 2:
            continue
        for subject in subjects:
            pair_rows = []
            y_vals = []
            blocks = sorted(block for subj, block in patterns if subj == subject)
            for i, block_i in enumerate(blocks):
                for block_j in blocks[i + 1:]:
                    val = vector_corr(
                        patterns[(subject, block_i)][:, keep_time].ravel(),
                        patterns[(subject, block_j)][:, keep_time].ravel(),
                    )
                    if np.isfinite(val):
                        pair_rows.append({"block_i": block_i, "block_j": block_j})
                        y_vals.append(float(val))
            if len(y_vals) >= 20:
                rows.extend(score_payload(subject, float(center), pair_rows, y_vals))
        if (center_i % 10) == 0:
            print(f"[ERP block model] centers {center_i}/{len(centers)}", flush=True)

    subject_path = output_dir / "erp_block_model_timecourse_subject.csv"
    summary_path = output_dir / "erp_block_model_timecourse_summary.csv"
    write_scores(rows, subject_path, summary_path)
    print(f"[ERP block model] wrote {subject_path}", flush=True)
    print(f"[ERP block model] wrote {summary_path}", flush=True)
    return {"subject": subject_path, "summary": summary_path}


if __name__ == "__main__":
    run_erp_block_model_timecourse()
