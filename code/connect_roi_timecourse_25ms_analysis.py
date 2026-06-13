#!/usr/bin/env python3
"""25 ms stimulus-locked connectivity time courses for strict sensor ROIs."""

from pathlib import Path
import os
import time

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np
import pandas as pd

from analysis_utils import parallel_collect
from connect_sensorwide_analysis import compute_coherence_components, OUTPUT_DIR
from load_project_data import align_behaviour_to_epochs, load_sessions
from sensor_rois import STRICT_SENSOR_ROIS, cross_roi_pairs

N_JOBS = 8
WINDOW_SEC = 0.025
STEP_SEC = 0.010
STIM_TMIN = 0.00
STIM_TMAX = 0.80
ROI_PAIR_SPECS = {
    "visual_frontal": ("visual", "frontal"),
    "visual_central": ("visual", "central"),
}


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def roi_pair_defs(channel_subset):
    ch_pos = {ch: idx for idx, ch in enumerate(channel_subset)}
    out = {}
    for roi_pair, (source, target) in ROI_PAIR_SPECS.items():
        pairs = []
        for ch_i, ch_j in cross_roi_pairs(source, target):
            pairs.append((ch_pos[ch_i], ch_pos[ch_j]))
        out[roi_pair] = pairs
    return out


def process_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    session_file = task["epo_file"]
    channel_subset = task["channel_subset"]
    pair_defs = task["pair_defs"]
    try:
        epochs_all = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, _ = align_behaviour_to_epochs(
            task["beh"],
            epochs_all,
            event_names=("Stim/A", "Stim/B"),
        )
        stim_epochs = stim_epochs.load_data()
        missing = [ch for ch in channel_subset if ch not in stim_epochs.ch_names]
        if missing:
            return {
                "ok": False,
                "qc": {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "reason": "missing_channels",
                    "detail": ",".join(missing),
                },
            }
        stim_epochs.pick(channel_subset)
        stim_epochs = stim_epochs.apply_hilbert(envelope=False, verbose="ERROR")
        data = stim_epochs.get_data()
        times = stim_epochs.times
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "reason": "compute_error",
                "detail": str(exc),
            },
        }

    rows = []
    starts = np.arange(STIM_TMIN, STIM_TMAX - WINDOW_SEC + 1e-12, STEP_SEC)
    for t_start in starts:
        t_end = float(t_start + WINDOW_SEC)
        i0 = int(np.searchsorted(times, t_start, side="left"))
        i1 = int(np.searchsorted(times, t_end, side="left"))
        if i1 - i0 < 2:
            continue
        win = data[:, :, i0:i1]
        for roi_pair, pairs in pair_defs.items():
            vals = []
            for i, j in pairs:
                components = compute_coherence_components(
                    win[:, i, :].reshape(-1),
                    win[:, j, :].reshape(-1),
                )
                val = components["conn_val"]
                if np.isfinite(val):
                    vals.append(float(val))
            rows.append(
                {
                    "roi_pair": roi_pair,
                    "subject": subject,
                    "day": day,
                    "session_file": session_file,
                    "lock_time": float(t_start),
                    "conn_val": float(np.mean(vals)) if vals else np.nan,
                    "n_edges": int(len(vals)),
                    "window_sec": WINDOW_SEC,
                    "step_sec": STEP_SEC,
                }
            )
    return {"ok": True, "rows": rows}


def summarize_day(subject_df):
    rows = []
    for key, g in subject_df.groupby(["roi_pair", "day", "lock_time"]):
        roi_pair, day, lock_time = key
        vals = g["conn_val"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "roi_pair": roi_pair,
                "day": int(day),
                "lock_time": float(lock_time),
                "conn_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "conn_sem": sem(vals),
                "n_subjects": int(len(vals)),
                "n_edges": int(g["n_edges"].max()),
                "window_sec": WINDOW_SEC,
                "step_sec": STEP_SEC,
            }
        )
    return pd.DataFrame(rows).sort_values(["roi_pair", "day", "lock_time"])


def make_contrast(subject_df):
    wide = (
        subject_df.pivot_table(
            index=["subject", "day", "lock_time"],
            columns="roi_pair",
            values="conn_val",
            aggfunc="mean",
        )
        .reset_index()
        .dropna(subset=["visual_central", "visual_frontal"])
    )
    wide["contrast"] = wide["visual_central"] - wide["visual_frontal"]
    contrast_subject = wide[["subject", "day", "lock_time", "contrast"]].copy()
    rows = []
    for key, g in contrast_subject.groupby(["day", "lock_time"]):
        day, lock_time = key
        vals = g["contrast"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "day": int(day),
                "lock_time": float(lock_time),
                "contrast_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "contrast_sem": sem(vals),
                "n_subjects": int(len(vals)),
                "window_sec": WINDOW_SEC,
                "step_sec": STEP_SEC,
            }
        )
    return contrast_subject, pd.DataFrame(rows).sort_values(["day", "lock_time"])


def run_connect_roi_timecourse_25ms(output_dir=OUTPUT_DIR, n_workers=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    channel_subset = sorted(
        set(STRICT_SENSOR_ROIS["visual"])
        | set(STRICT_SENSOR_ROIS["frontal"])
        | set(STRICT_SENSOR_ROIS["central"])
    )
    pair_defs = roi_pair_defs(channel_subset)
    sessions = load_sessions()
    tasks = []
    for item in sessions:
        tasks.append(
            {
                "subject": int(item["subject"]),
                "day": int(item["day"]),
                "epo_file": item["epo_file"],
                "epo_path": str(item["epo_path"]),
                "beh": item["beh"],
                "channel_subset": channel_subset,
                "pair_defs": pair_defs,
            }
        )
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    print(
        f"[connect ROI 25ms] Running {len(tasks)} sessions "
        f"(window={WINDOW_SEC:.3f}s, step={STEP_SEC:.3f}s, n_workers={n_workers})",
        flush=True,
    )
    t0 = time.time()
    results = parallel_collect(process_session, tasks, n_workers)
    rows = []
    qc_rows = []
    for result in results:
        if result["ok"]:
            rows.extend(result["rows"])
        else:
            qc_rows.append(result["qc"])
    subject_df = pd.DataFrame(rows)
    if subject_df.empty:
        pd.DataFrame(qc_rows).to_csv(
            output_dir / "connect_roi_timecourse_25ms_qc_log.csv",
            index=False,
        )
        raise RuntimeError("25 ms ROI connectivity produced no valid rows")
    day_df = summarize_day(subject_df)
    contrast_subject, contrast_day = make_contrast(subject_df)

    paths = {
        "subject": output_dir / "connect_roi_timecourse_25ms_subject.csv",
        "day_mean": output_dir / "connect_roi_timecourse_25ms_day_mean.csv",
        "contrast_subject": output_dir / "connect_roi_timecourse_25ms_contrast_subject.csv",
        "contrast_day_mean": output_dir / "connect_roi_timecourse_25ms_contrast_day_mean.csv",
        "qc": output_dir / "connect_roi_timecourse_25ms_qc_log.csv",
    }
    subject_df.to_csv(paths["subject"], index=False)
    day_df.to_csv(paths["day_mean"], index=False)
    contrast_subject.to_csv(paths["contrast_subject"], index=False)
    contrast_day.to_csv(paths["contrast_day_mean"], index=False)
    pd.DataFrame(
        qc_rows,
        columns=["session_file", "subject", "day", "reason", "detail"],
    ).to_csv(paths["qc"], index=False)
    for path in paths.values():
        print(f"[connect ROI 25ms] wrote {path}", flush=True)
    print(f"[connect ROI 25ms] elapsed {(time.time() - t0) / 60.0:.1f} min", flush=True)
    return paths


if __name__ == "__main__":
    run_connect_roi_timecourse_25ms()
