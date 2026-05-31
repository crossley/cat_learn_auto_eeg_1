#!/usr/bin/env python3
"""Sensor-wide functional connectivity dynamics (16-channel abs ImCoh carpets)."""

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

from analysis_utils import parallel_collect
from load_project_data import align_behaviour_to_epochs, load_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8

CHANNEL_SUBSET = [
    "Fp1",
    "AF7",
    "AF3",
    "F1",
    "F3",
    "F5",
    "F7",
    "FT7",
    "FC5",
    "FC3",
    "FC1",
    "C1",
    "C3",
    "C5",
    "T7",
    "TP7",
    "CP5",
    "CP3",
    "CP1",
    "P1",
    "P3",
    "P5",
    "P7",
    "P9",
    "PO7",
    "PO3",
    "O1",
    "Iz",
    "Oz",
    "POz",
    "Pz",
    "CPz",
    "Fpz",
    "Fp2",
    "AF8",
    "AF4",
    "AFz",
    "Fz",
    "F2",
    "F4",
    "F6",
    "F8",
    "FT8",
    "FC6",
    "FC4",
    "FC2",
    "FCz",
    "Cz",
    "C2",
    "C4",
    "C6",
    "T8",
    "TP8",
    "CP6",
    "CP4",
    "CP2",
    "P2",
    "P4",
    "P6",
    "P8",
    "P10",
    "PO8",
    "PO4",
    "O2",
]

BANDS = {
    "broadband": None,
}


CONNECTIVITY_VALUE_COLUMNS = [
    "conn_val",
    "imcoh_abs",
    "coh_abs",
    "coh_phase_rad",
    "phase_lag_factor",
]


def compute_coherence_components(x, y):
    sxy = np.mean(x * np.conjugate(y))
    sxx = np.mean(np.abs(x) ** 2)
    syy = np.mean(np.abs(y) ** 2)
    denom = np.sqrt(sxx * syy)
    if (not np.isfinite(denom)) or (denom <= np.finfo(float).eps):
        return {
            "conn_val": np.nan,
            "imcoh_abs": np.nan,
            "coh_abs": np.nan,
            "coh_phase_rad": np.nan,
            "phase_lag_factor": np.nan,
        }
    coh = sxy / denom
    coh_abs = float(np.abs(coh))
    coh_phase = float(np.angle(coh))
    phase_lag_factor = float(np.abs(np.sin(coh_phase)))
    imcoh_abs = float(np.abs(np.imag(coh)))
    return {
        "conn_val": imcoh_abs,
        "imcoh_abs": imcoh_abs,
        "coh_abs": coh_abs,
        "coh_phase_rad": coh_phase,
        "phase_lag_factor": phase_lag_factor,
    }


def compute_abs_imcoh(x, y):
    return compute_coherence_components(x, y)["imcoh_abs"]


def process_sensorwide_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    beh_df = task["beh"]
    epo_path = task["epo_path"]
    channel_subset = task["channel_subset"]
    bands = task["bands"]
    pair_idx = task["pair_idx"]
    window_sec = float(task["window_sec"])
    stim_tmin = float(task["stim_tmin"])
    stim_tmax = float(task["stim_tmax"])
    feedback_tmin = float(task["feedback_tmin"])
    feedback_tmax = float(task["feedback_tmax"])
    step_sec = float(task["step_sec"])

    try:
        epochs_all = mne.read_epochs(epo_path, preload=False, verbose="ERROR")
        stim_epochs, _ = align_behaviour_to_epochs(
            beh_df, epochs_all, event_names=("Stim/A", "Stim/B")
        )
        feedback_epochs, _ = align_behaviour_to_epochs(
            beh_df, epochs_all, event_names=("FB/Cor", "FB/Inc")
        )
        stim_epochs = stim_epochs.load_data()
        feedback_epochs = feedback_epochs.load_data()
        all_channels_present = True
        missing = []
        for ch in channel_subset:
            if ch not in stim_epochs.ch_names:
                all_channels_present = False
                missing.append(ch)
        if not all_channels_present:
            return {
                "ok": False, "subject": subject, "day": day,
                "reason": "missing_channels", "detail": ",".join(missing),
            }
        stim_epochs.pick(channel_subset)
        feedback_epochs.pick(channel_subset)
        info = stim_epochs.info.copy()
        stim_starts = np.arange(stim_tmin, stim_tmax - window_sec + 1e-12, step_sec)
        feedback_starts = np.arange(
            feedback_tmin, feedback_tmax - window_sec + 1e-12, step_sec
        )
        if len(stim_starts) == 0 and len(feedback_starts) == 0:
            return {
                "ok": False,
                "subject": subject,
                "day": day,
                "reason": "no_windows",
                "detail": "",
            }

        agg = {}
        subject_rows = []
        for band_name, band_limits in bands.items():
            for lock_name, epochs_lock, starts in [
                ("stim", stim_epochs, stim_starts),
                ("feedback", feedback_epochs, feedback_starts),
            ]:
                if band_limits is None:
                    epochs_band = epochs_lock.copy()
                else:
                    fmin, fmax = band_limits
                    epochs_band = epochs_lock.copy().filter(
                        l_freq=fmin, h_freq=fmax, method="fir",
                        fir_design="firwin", phase="zero-double", verbose="ERROR",
                    )
                epochs_band = epochs_band.apply_hilbert(envelope=False, verbose="ERROR")
                data = epochs_band.get_data()
                times = epochs_band.times
                n_trials = data.shape[0]
                if n_trials == 0:
                    continue

                for t_start in starts:
                    t_end = t_start + window_sec
                    i0 = int(np.searchsorted(times, t_start, side="left"))
                    i1 = int(np.searchsorted(times, t_end, side="left"))
                    if i1 - i0 < 2:
                        continue
                    win = data[:, :, i0:i1]
                    lock_time = float(t_start)
                    for i, j in pair_idx:
                        components = compute_coherence_components(
                            win[:, i, :].reshape(-1), win[:, j, :].reshape(-1)
                        )
                        val = components["conn_val"]
                        if not np.isfinite(val):
                            continue
                        key = (lock_name, int(day), band_name, lock_time, i, j)
                        if key not in agg:
                            agg[key] = {
                                "count": 0,
                                "phase_unit_sin": 0.0,
                                "phase_unit_cos": 0.0,
                            }
                            for col in CONNECTIVITY_VALUE_COLUMNS:
                                if col != "coh_phase_rad":
                                    agg[key][col] = 0.0
                        agg[key]["count"] += 1
                        for col in CONNECTIVITY_VALUE_COLUMNS:
                            if col == "coh_phase_rad":
                                phase = components[col]
                                agg[key]["phase_unit_sin"] += float(np.sin(phase))
                                agg[key]["phase_unit_cos"] += float(np.cos(phase))
                            else:
                                agg[key][col] += float(components[col])
                        subject_row = {
                            "subject": subject,
                            "day": int(day),
                            "lock_type": lock_name,
                            "band": band_name,
                            "lock_time": lock_time,
                            "ch_i": channel_subset[i],
                            "ch_j": channel_subset[j],
                        }
                        subject_row.update(components)
                        subject_rows.append(subject_row)

        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "agg": agg,
            "subject_rows": subject_rows,
            "info": info,
        }
    except Exception as exc:
        return {
            "ok": False,
            "subject": subject,
            "day": day,
            "reason": "compute_error",
            "detail": str(exc),
        }


def agg_to_edges_df(agg, channel_subset):
    agg_rows = []
    for lock_name, day, band_name, t, i, j in sorted(agg.keys()):
        entry = agg[(lock_name, day, band_name, t, i, j)]
        c = int(entry["count"])
        row = {
            "lock_type": lock_name,
            "day": int(day),
            "band": band_name,
            "lock_time": float(t),
            "ch_i": channel_subset[i],
            "ch_j": channel_subset[j],
            "n_session_contrib": c,
        }
        for col in CONNECTIVITY_VALUE_COLUMNS:
            if col == "coh_phase_rad":
                row[col] = (
                    float(
                        np.arctan2(
                            entry["phase_unit_sin"],
                            entry["phase_unit_cos"],
                        )
                    )
                    if c > 0
                    else np.nan
                )
            else:
                row[col] = float(entry[col] / c) if c > 0 else np.nan
        agg_rows.append(row)
    if not agg_rows:
        return pd.DataFrame(
            columns=[
                "lock_type",
                "day",
                "band",
                "lock_time",
                "ch_i",
                "ch_j",
                "n_session_contrib",
            ]
            + CONNECTIVITY_VALUE_COLUMNS
        )
    return pd.DataFrame(agg_rows).sort_values(
        ["lock_type", "band", "day", "lock_time", "ch_i", "ch_j"]
    )


def channel_xy(info, ch_names):
    montage = info.get_montage()
    if montage is None:
        n = len(ch_names)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        return np.c_[np.cos(ang), np.sin(ang)] * 0.9
    ch_pos = montage.get_positions()["ch_pos"]
    xy_rows = []
    for ch in ch_names:
        xy_rows.append(ch_pos[ch][:2])
    xy = np.array(xy_rows, dtype=float)
    xy = xy - np.mean(xy, axis=0, keepdims=True)
    scale = np.max(np.linalg.norm(xy, axis=1))
    if scale > 0:
        xy = 0.9 * (xy / scale)
    return xy


def run_sensorwide_connectivity_analysis(
    channel_subset: list[str] = CHANNEL_SUBSET,
    bands: dict = BANDS,
    window_sec: float = 0.05,
    step_sec: float = 0.01,
    stim_tmin: float = 0.00,
    stim_tmax: float = 0.80,
    feedback_tmin: float = 0.00,
    feedback_tmax: float = 0.80,
    n_workers: int | None = None,
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    mne.set_log_level("ERROR")
    sessions = load_sessions()
    t0 = time.time()
    progress_json = output_dir / "connect_sensorwide_progress.json"
    carpet_path = output_dir / "sensorwide_carpet_timeseries.csv"
    subject_path = output_dir / "sensorwide_carpet_subject_timeseries.csv"
    channels_path = output_dir / "sensorwide_channel_layout.csv"
    checkpoint_path = output_dir / "sensorwide_carpet_timeseries_checkpoint.csv"

    n_channels = len(channel_subset)
    pair_idx = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pair_idx.append((i, j))

    tasks = []
    for item in sessions:
        tasks.append(
            {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "beh": item["beh"],
            "epo_path": str(item["epo_path"]),
            "channel_subset": channel_subset,
            "bands": bands,
            "pair_idx": pair_idx,
            "window_sec": window_sec,
            "step_sec": step_sec,
            "stim_tmin": stim_tmin,
            "stim_tmax": stim_tmax,
            "feedback_tmin": feedback_tmin,
            "feedback_tmax": feedback_tmax,
            }
        )
    print(
        "[connect_sensorwide] Running on "
        f"{len(tasks)} sessions, {n_channels} channels, "
        f"{len(pair_idx)} pairs (n_workers={n_workers})...",
        flush=True,
    )
    results = parallel_collect(process_sensorwide_session, tasks, n_workers)

    agg = {}
    subject_rows = []
    info_subset = None
    used = 0
    skipped = []
    for r in results:
        if not r["ok"]:
            skipped.append(r)
            continue
        used += 1
        if info_subset is None:
            info_subset = r["info"]
        for key, entry in r["agg"].items():
            if key not in agg:
                agg[key] = {
                    "count": 0,
                    "phase_unit_sin": 0.0,
                    "phase_unit_cos": 0.0,
                }
                for col in CONNECTIVITY_VALUE_COLUMNS:
                    if col != "coh_phase_rad":
                        agg[key][col] = 0.0
            agg[key]["count"] += int(entry["count"])
            agg[key]["phase_unit_sin"] += float(entry["phase_unit_sin"])
            agg[key]["phase_unit_cos"] += float(entry["phase_unit_cos"])
            for col in CONNECTIVITY_VALUE_COLUMNS:
                if col != "coh_phase_rad":
                    agg[key][col] += float(entry[col])
        for row in r["subject_rows"]:
            subject_rows.append(row)

    progress_json.write_text(
        json.dumps(
            {
                "sessions": len(tasks),
                "used": used,
                "skipped": len(skipped),
                "elapsed_sec": float(time.time() - t0),
            },
            indent=2,
        )
    )
    if skipped:
        pd.DataFrame(skipped).to_csv(
            output_dir / "connect_sensorwide_qc_skipped.csv", index=False
        )

    if not agg or info_subset is None:
        print("[connect_sensorwide] No data computed.", flush=True)
        return {"carpet_path": None}

    d_edges = agg_to_edges_df(agg, channel_subset)
    d_edges.to_csv(carpet_path, index=False)
    d_edges.to_csv(checkpoint_path, index=False)
    pd.DataFrame(subject_rows).to_csv(subject_path, index=False)

    xy = channel_xy(info_subset, channel_subset)
    pd.DataFrame({"channel": channel_subset, "x": xy[:, 0], "y": xy[:, 1]}).to_csv(
        channels_path, index=False
    )

    print(
        f"[connect_sensorwide] Done. Used sessions: {used}, skipped: {len(skipped)}.",
        flush=True,
    )
    return {
        "carpet_path": carpet_path,
        "subject_path": subject_path,
        "channels_path": channels_path,
    }


if __name__ == "__main__":
    run_sensorwide_connectivity_analysis()
