#!/usr/bin/env python3
"""Shared settings for multi-measure sensor-ROI connectivity analyses.

Primary analysis specified before looking at the robustness outputs:
debiased wPLI (``wpli2_debiased``, labelled dwPLI), broadband stimulus-locked
connectivity in the 0.25-0.45 s decision window, using visual-frontal and
visual-central strict sensor ROI pairs, followed by the same day/block model
comparison used by the existing imcoh connectivity pipeline.

The remaining measures and frequency bands are robustness/convergence checks.
Coherence and PLV are contrast-only zero-lag-inclusive measures.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd
from mne_connectivity import spectral_connectivity_epochs

from sensor_rois import STRICT_SENSOR_ROIS, cross_roi_pairs

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

ROI_PAIR_SPECS = {
    "visual_frontal": ("visual", "frontal"),
    "visual_central": ("visual", "central"),
}

MEASURES = ["imcoh", "wpli", "wpli2_debiased", "pli", "coh", "plv"]
ZERO_LAG_REJECTING_MEASURES = ["imcoh", "wpli", "wpli2_debiased", "pli"]
CONTRAST_ONLY_MEASURES = ["coh", "plv"]
PRIMARY_MEASURE = "wpli2_debiased"
PRIMARY_MEASURE_LABEL = "dwPLI"
PRIMARY_BAND = "broadband"
PRIMARY_LOCK_TYPE = "stim"
PRIMARY_DECISION_WINDOW = (0.25, 0.45)

MEASURE_LABELS = {
    "imcoh": "abs ImCoh",
    "wpli": "wPLI",
    "wpli2_debiased": "dwPLI",
    "pli": "PLI",
    "coh": "Coherence",
    "plv": "PLV",
}

BANDS = {
    "broadband": (1.0, 40.0),
    "delta": (1.0, 4.0),
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
}
PRIMARY_BANDS = {"broadband": BANDS["broadband"]}
CWT_FREQS = np.arange(1.0, 41.0, 1.0)
CWT_N_CYCLES = 2.0

WINDOW_SEC = 0.050
STEP_SEC = 0.010
STIM_TMIN = 0.00
STIM_TMAX = 0.80
FEEDBACK_TMIN = 0.00
FEEDBACK_TMAX = 0.80


def strict_roi_channels():
    channels = (
        set(STRICT_SENSOR_ROIS["visual"])
        | set(STRICT_SENSOR_ROIS["frontal"])
        | set(STRICT_SENSOR_ROIS["central"])
    )
    return sorted(channels)


def strict_roi_edge_table(channel_subset=None):
    if channel_subset is None:
        channel_subset = strict_roi_channels()
    ch_pos = {ch: idx for idx, ch in enumerate(channel_subset)}
    rows = []
    for roi_pair, (source, target) in ROI_PAIR_SPECS.items():
        for ch_i, ch_j in cross_roi_pairs(source, target):
            rows.append(
                {
                    "roi_pair": roi_pair,
                    "ch_i": ch_i,
                    "ch_j": ch_j,
                    "i": int(ch_pos[ch_i]),
                    "j": int(ch_pos[ch_j]),
                }
            )
    edge_df = pd.DataFrame(rows).drop_duplicates()
    return edge_df.sort_values(["roi_pair", "ch_i", "ch_j"]).reset_index(drop=True)


def target_times(tmin, tmax, step_sec=STEP_SEC):
    return np.round(np.arange(tmin, tmax + 1e-12, step_sec), 10)


def nearest_time_indices(available_times, requested_times):
    available_times = np.asarray(available_times, dtype=float)
    requested_times = np.asarray(requested_times, dtype=float)
    indices = []
    for time_sec in requested_times:
        idx = int(np.argmin(np.abs(available_times - time_sec)))
        indices.append(idx)
    return np.asarray(indices, dtype=int)


def subtract_condition_evoked(epochs, condition_names):
    induced = epochs.copy().load_data()
    data = induced.get_data(copy=False)
    for condition in condition_names:
        if condition not in induced.event_id:
            raise ValueError(f"Missing condition for induced estimate: {condition}")
        code = int(induced.event_id[condition])
        idx = np.where(induced.events[:, 2] == code)[0]
        if len(idx) == 0:
            raise ValueError(f"No epochs for induced condition: {condition}")
        data[idx, :, :] -= np.mean(data[idx, :, :], axis=0, keepdims=True)
    return induced


def compute_connectivity_rows(
    epochs,
    edge_df,
    measures,
    bands,
    lock_type,
    tmin,
    tmax,
    step_sec=STEP_SEC,
    signal_estimate=None,
    n_jobs=1,
):
    epochs = epochs.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)
    requested_times = target_times(tmin, tmax, step_sec)
    indices = (
        edge_df["i"].to_numpy(dtype=int),
        edge_df["j"].to_numpy(dtype=int),
    )
    fmin = tuple(float(bounds[0]) for bounds in bands.values())
    fmax = tuple(float(bounds[1]) for bounds in bands.values())
    con_list = spectral_connectivity_epochs(
        epochs,
        method=list(measures),
        indices=indices,
        mode="cwt_morlet",
        cwt_freqs=CWT_FREQS,
        cwt_n_cycles=CWT_N_CYCLES,
        fmin=fmin,
        fmax=fmax,
        faverage=True,
        n_jobs=max(1, int(n_jobs)),
        verbose="ERROR",
    )
    if len(measures) == 1:
        con_list = [con_list]

    time_idx = nearest_time_indices(con_list[0].times, requested_times)
    rows = []
    base_edges = edge_df[["roi_pair", "ch_i", "ch_j"]].reset_index(drop=True)
    band_names = list(bands.keys())
    for measure, con in zip(measures, con_list):
        values = con.get_data()[:, :, time_idx]
        if measure == "imcoh":
            values = np.abs(values)
        for band_i, band_name in enumerate(band_names):
            band_values = values[:, band_i, :]
            for time_i, lock_time in enumerate(requested_times):
                d = base_edges.copy()
                d["lock_type"] = lock_type
                d["band"] = band_name
                d["measure"] = measure
                d["lock_time"] = float(lock_time)
                d["conn_val"] = band_values[:, time_i].astype(float)
                if signal_estimate is not None:
                    d["signal_estimate"] = signal_estimate
                rows.append(d)
    if not rows:
        raise ValueError("No connectivity rows were computed")
    return pd.concat(rows, ignore_index=True)


def append_csv(path, df, write_header):
    df.to_csv(path, mode="w" if write_header else "a", header=write_header, index=False)
    return False


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
