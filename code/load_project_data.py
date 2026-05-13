#!/usr/bin/env python3
"""
Load behavioural CSV files and epoched EEG files for the category-learning project.

This file is intentionally brittle. It assumes the documented project layout:

    Behavioural/sub_<subject>_day_<daycode>_data.csv
    EEG_epo/P<subject>_D<daycode>-epo.fif

If files are missing, misnamed, or lack required columns, the code should fail
clearly rather than trying to recover.
"""

from pathlib import Path
import math
import os
import re

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent.parent
BEHAVIOURAL_DIR = PROJECT_DIR / "Behavioural"
EPOCH_DIR = PROJECT_DIR / "EEG_epo"

BEHAVIOURAL_COLUMNS = ["trial", "cat", "x", "y", "xt", "yt", "resp", "rt", "fb"]

BEHAVIOURAL_FILE_RE = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")
EPOCH_FILE_RE = re.compile(r"^P(\d+)_D([\d_]+)-epo\.fif$")


def load_sessions(load_epochs=False):
    """
    Return matched behavioural and epoched EEG sessions.

    day_code is the value in the behavioural filename. day is the analysis day.
    Existing behavioural files use day codes such as 100, 200, ...; ad hoc task
    test files such as day_01 are treated as day 1.
    """
    if not BEHAVIOURAL_DIR.exists():
        raise FileNotFoundError(f"Missing behavioural directory: {BEHAVIOURAL_DIR}")
    if not EPOCH_DIR.exists():
        raise FileNotFoundError(f"Missing epoch directory: {EPOCH_DIR}")

    behavioural_files = {}
    for path in sorted(BEHAVIOURAL_DIR.glob("*.csv")):
        match = BEHAVIOURAL_FILE_RE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected behavioural filename: {path.name}")

        subject = int(match.group(1))
        day_code = int(match.group(2))
        day = day_code // 100 if day_code >= 100 else day_code

        behaviour = pd.read_csv(path)
        missing_columns = [col for col in BEHAVIOURAL_COLUMNS if col not in behaviour.columns]
        if missing_columns:
            raise ValueError(f"{path.name} is missing columns: {missing_columns}")

        behavioural_files[(subject, day)] = {
            "subject": subject,
            "day": day,
            "day_code": day_code,
            "beh_path": path,
            "beh_file": path.name,
            "beh": behaviour,
        }

    epoch_files = {}
    for path in sorted(EPOCH_DIR.glob("*-epo.fif")):
        match = EPOCH_FILE_RE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected epoch filename: {path.name}")

        subject = int(match.group(1))
        day_code_text = match.group(2)
        day = int(day_code_text.split("_")[0])

        epoch_files[(subject, day)] = {
            "epo_path": path,
            "epo_file": path.name,
        }

    missing_behaviour = sorted(set(epoch_files) - set(behavioural_files))
    missing_epochs = sorted(set(behavioural_files) - set(epoch_files))
    if missing_behaviour:
        raise FileNotFoundError(f"Epoch files without matching behavioural files: {missing_behaviour}")
    if missing_epochs:
        raise FileNotFoundError(f"Behavioural files without matching epoch files: {missing_epochs}")

    sessions = []
    for key in sorted(behavioural_files):
        session = behavioural_files[key] | epoch_files[key]
        if load_epochs:
            import mne

            session["epochs"] = mne.read_epochs(session["epo_path"], preload=False, verbose="ERROR")
            session["events"] = session["epochs"].events.copy()
            session["event_id"] = dict(session["epochs"].event_id)
        sessions.append(session)

    if not sessions:
        raise FileNotFoundError(f"No matched sessions found in {BEHAVIOURAL_DIR} and {EPOCH_DIR}")

    return sessions


def align_behaviour_to_epochs(behaviour, epochs, event_names=("Stim/A", "Stim/B")):
    """
    Align behavioural rows to retained epochs after epoch rejection.

    This helper is shared because every EEG analysis needs exactly this operation.
    It fails when alignment cannot be established.
    """
    behaviour = behaviour.sort_values("trial").reset_index(drop=True)

    requested_event_names = list(event_names)
    event_names = [event_name for event_name in requested_event_names if event_name in epochs.event_id]
    if len(event_names) == 0:
        raise ValueError(f"None of the requested events are present: {requested_event_names}")

    selected_epochs = epochs[event_names]
    if len(selected_epochs) == 0:
        raise ValueError("No epochs remain after selecting requested events.")

    if selected_epochs.metadata is not None and "beh_trial_index" in selected_epochs.metadata:
        trial_index = selected_epochs.metadata["beh_trial_index"].to_numpy(dtype=int)
    else:
        selection = np.asarray(selected_epochs.selection, dtype=int)
        if len(selection) > 1:
            differences = np.diff(np.sort(selection))
            step = int(differences[0])
            for difference in differences[1:]:
                step = math.gcd(step, int(difference))
            offset = int(np.min(selection % step)) if step > 1 else 0
            trial_index = (selection - offset) // step if step > 1 else selection.copy()
        else:
            trial_index = selection.copy()

    if np.any(trial_index < 0) or np.any(trial_index >= len(behaviour)):
        raise ValueError("Epoch-to-behaviour alignment produced invalid trial indices.")

    aligned_behaviour = behaviour.iloc[trial_index].reset_index(drop=True)
    return selected_epochs, aligned_behaviour
