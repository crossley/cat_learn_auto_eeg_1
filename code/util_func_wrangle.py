#!/usr/bin/env python3
"""Data wrangling utilities for behavioural/session and EEG epoch data."""

import os
import re
import math
from pathlib import Path
import numpy as np
import pandas as pd
import mne

# os.environ["NUMBA_DISABLE_JIT"] = "1"


def util_wrangle_align_beh_to_epochs(beh_df, epochs, event_names=("Stim/A", "Stim/B")):
    """Return stimulus epochs and behavioural rows aligned after epoch rejection."""
    beh_sorted = beh_df.sort_values("trial").reset_index(drop=True) if "trial" in beh_df else beh_df.reset_index(drop=True)
    event_names = [name for name in event_names if name in epochs.event_id]
    if len(event_names) == 0:
        return epochs[:0], beh_sorted.iloc[:0].copy()

    epochs_stim = epochs[event_names]
    if len(epochs_stim) == 0:
        return epochs_stim, beh_sorted.iloc[:0].copy()

    if epochs_stim.metadata is not None and "beh_trial_index" in epochs_stim.metadata:
        trial_idx = epochs_stim.metadata["beh_trial_index"].to_numpy(dtype=int)
    else:
        # Existing epoch files may predate metadata. Their MNE selections are in
        # raw event-index space, where stimulus triggers occur at a fixed stride.
        sel = np.asarray(epochs_stim.selection, dtype=int)
        if len(sel) > 1:
            diffs = np.diff(np.sort(sel))
            step = int(diffs[0])
            for diff in diffs[1:]:
                step = math.gcd(step, int(diff))
            if step > 1:
                offset = int(np.min(sel % step))
                trial_idx = (sel - offset) // step
            else:
                trial_idx = sel.copy()
        else:
            trial_idx = sel.copy()

    valid = (trial_idx >= 0) & (trial_idx < len(beh_sorted))
    if not valid.all():
        epochs_stim = epochs_stim[np.where(valid)[0]]
        trial_idx = trial_idx[valid]

    beh_aligned = beh_sorted.iloc[trial_idx].reset_index(drop=True)
    return epochs_stim, beh_aligned


def util_wrangle_load_sessions():
    """
    Load behavioural and epoch sessions
    """

    beh_dir = Path("../Behavioural")
    epo_dir = Path("../EEG_epo")

    beh_re = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")
    epo_re = re.compile(r"^P(\d+)_D([\d_]+)-epo\.fif$")

    beh_map = {}
    for beh_path in sorted(beh_dir.glob("*.csv")):
        m = beh_re.match(beh_path.name)
        subject = int(m.group(1))
        day_code = int(m.group(2))
        day = day_code // 100
        key = (subject, day)
        beh_map[key] = beh_path

    epo_map = {}
    for epo_path in sorted(epo_dir.glob("*-epo.fif")):
        m = epo_re.match(epo_path.name)
        subject = int(m.group(1))
        day = int(m.group(2).split("_")[0])
        key = (subject, day)
        epo_map[key] = epo_path

    missing_beh = sorted(set(epo_map) - set(beh_map))
    missing_epo = sorted(set(beh_map) - set(epo_map))

    print(f"Missing behavioural files: {missing_beh}")
    print(f"Missing epoch files: {missing_epo}")

    subjects = sorted({s for s, _ in beh_map} | {s for s, _ in epo_map})
    days = [1, 2, 3, 4, 5]

    sessions = []
    for subject in subjects:
        for day in days:
            key = (subject, day)
            beh_path = beh_map.get(key)
            epo_path = epo_map.get(key)

            if (beh_path is not None) and (epo_path is not None):
                beh_df = pd.read_csv(beh_path)
                epochs = mne.read_epochs(epo_path, preload=False, verbose="ERROR")
                sessions.append(
                    {
                        "subject": subject,
                        "day": day,
                        "beh_file": beh_path.name if beh_path is not None else None,
                        "epo_file": epo_path.name if epo_path is not None else None,
                        "beh_df": beh_df,
                        "epochs": epochs,
                        "events": epochs.events.copy() if epochs is not None else None,
                        "event_id": dict(epochs.event_id) if epochs is not None else {},
                    }
                )

    return sessions
