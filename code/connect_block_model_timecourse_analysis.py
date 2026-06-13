#!/usr/bin/env python3
"""25-block ROI connectivity model-evidence timecourse."""

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
from connect_sensorwide_analysis import compute_coherence_components, OUTPUT_DIR
from load_project_data import align_behaviour_to_epochs, load_sessions
from sensor_rois import STRICT_SENSOR_ROIS, cross_roi_pairs

N_JOBS = 8
WINDOW_SEC = 0.050
STEP_SEC = 0.010
STIM_TMIN = 0.00
STIM_TMAX = 0.80
ROI_PAIR_SPECS = [("visual", "frontal"), ("visual", "central")]


def edge_defs(channel_subset):
    ch_pos = {ch: idx for idx, ch in enumerate(channel_subset)}
    pairs = []
    for source, target in ROI_PAIR_SPECS:
        for ch_i, ch_j in cross_roi_pairs(source, target):
            pairs.append((ch_pos[ch_i], ch_pos[ch_j]))
    return sorted(set(pairs))


def process_session(item):
    channel_subset = sorted(
        set(STRICT_SENSOR_ROIS["visual"])
        | set(STRICT_SENSOR_ROIS["frontal"])
        | set(STRICT_SENSOR_ROIS["central"])
    )
    pairs = edge_defs(channel_subset)
    try:
        epochs = mne.read_epochs(item["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh = align_behaviour_to_epochs(
            item["beh"], epochs, event_names=("Stim/A", "Stim/B")
        )
        stim_epochs = stim_epochs.copy().load_data()
        missing = [ch for ch in channel_subset if ch not in stim_epochs.ch_names]
        if missing:
            raise ValueError("missing_channels:" + ",".join(missing))
        stim_epochs.pick(channel_subset)
        stim_epochs = stim_epochs.apply_hilbert(envelope=False, verbose="ERROR")
        data = stim_epochs.get_data()
        times = stim_epochs.times.copy()
        beh = beh.copy()
        beh["day"] = int(item["day"])
        beh = assign_accumulated_blocks(beh)
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

    starts = np.arange(STIM_TMIN, STIM_TMAX - WINDOW_SEC + 1e-12, STEP_SEC)
    vectors = {}
    for t_start in starts:
        i0 = int(np.searchsorted(times, t_start, side="left"))
        i1 = int(np.searchsorted(times, t_start + WINDOW_SEC, side="left"))
        if i1 - i0 < 2:
            continue
        for block, idx in beh.groupby("block").groups.items():
            idx = np.asarray(list(idx), dtype=int)
            if len(idx) < 5:
                continue
            win = data[idx, :, i0:i1]
            vals = []
            for i, j in pairs:
                val = compute_coherence_components(
                    win[:, i, :].reshape(-1),
                    win[:, j, :].reshape(-1),
                )["conn_val"]
                vals.append(float(val) if np.isfinite(val) else np.nan)
            vectors[(int(item["subject"]), int(block), float(t_start))] = np.asarray(vals, dtype=float)
    return {"ok": True, "vectors": vectors}


def run_connect_block_model_timecourse(output_dir=OUTPUT_DIR, n_workers=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    sessions = load_sessions()
    if n_workers is None:
        n_workers = N_JOBS
    results = parallel_collect(process_session, sessions, max(1, int(n_workers)))
    vectors = {}
    qc_rows = []
    for result in results:
        if result["ok"]:
            vectors.update(result["vectors"])
        else:
            qc_rows.append(result["qc"])
    subjects = sorted({subject for subject, _block, _time in vectors})
    times = sorted({time for _subject, _block, time in vectors})
    rows = []
    for time_i, time_sec in enumerate(times, start=1):
        for subject in subjects:
            blocks = sorted(block for subj, block, t in vectors if subj == subject and t == time_sec)
            pair_rows = []
            y_vals = []
            for i, block_i in enumerate(blocks):
                for block_j in blocks[i + 1:]:
                    val = vector_corr(
                        vectors[(subject, block_i, time_sec)],
                        vectors[(subject, block_j, time_sec)],
                    )
                    if np.isfinite(val):
                        pair_rows.append({"block_i": block_i, "block_j": block_j})
                        y_vals.append(float(val))
            if len(y_vals) >= 20:
                rows.extend(score_payload(subject, time_sec, pair_rows, y_vals))
        if (time_i % 10) == 0:
            print(f"[connect block model] times {time_i}/{len(times)}", flush=True)
    subject_path = output_dir / "connect_block_model_timecourse_subject.csv"
    summary_path = output_dir / "connect_block_model_timecourse_summary.csv"
    write_scores(rows, subject_path, summary_path)
    print(f"[connect block model] wrote {subject_path}", flush=True)
    print(f"[connect block model] wrote {summary_path}", flush=True)
    return {"subject": subject_path, "summary": summary_path}


if __name__ == "__main__":
    run_connect_block_model_timecourse()
