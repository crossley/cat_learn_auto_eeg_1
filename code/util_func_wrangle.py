#!/usr/bin/env python3
"""Data wrangling utilities for behavioural/session and EEG epoch data."""

import os
import re
from pathlib import Path
import pandas as pd
import mne

# os.environ["NUMBA_DISABLE_JIT"] = "1"


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
