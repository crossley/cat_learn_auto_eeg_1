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

    def observed_and_behaviour_labels(candidate):
        event_lookup = {code: name for name, code in selected_epochs.event_id.items()}
        event_labels = np.array(
            [event_lookup.get(int(code), "") for code in selected_epochs.events[:, 2]],
            dtype=object,
        )
        if set(event_labels).issubset({"Stim/A", "Stim/B"}):
            observed = np.where(event_labels == "Stim/A", "A", "B")
            behaviour_labels = behaviour["cat"].astype(str).to_numpy()
            expected = behaviour_labels[candidate]
            return observed, expected, behaviour_labels
        if set(event_labels).issubset({"FB/Cor", "FB/Inc"}):
            observed = np.where(event_labels == "FB/Cor", "correct", "incorrect")
            behaviour_labels = behaviour["fb"].astype(str).str.lower().to_numpy()
            expected = behaviour_labels[candidate]
            return observed, expected, behaviour_labels
        return None, None, None

    def score_trial_index(candidate):
        valid = (candidate >= 0) & (candidate < len(behaviour))
        if not np.all(valid):
            return -np.inf
        observed, expected, _ = observed_and_behaviour_labels(candidate)
        if observed is None:
            return 0.0
        return float(np.mean(observed == expected))

    def refine_trial_index_by_label_sequence(prior):
        valid = (prior >= 0) & (prior < len(behaviour))
        if not np.all(valid):
            return prior
        observed, _, behaviour_labels = observed_and_behaviour_labels(prior)
        if observed is None:
            return prior
        n_events = len(observed)
        n_trials = len(behaviour_labels)
        if n_events == 0 or n_trials == 0 or n_events > n_trials:
            return prior
        inf = 1e15
        mismatch_penalty = 10000.0
        prior_weight = 0.01
        dp = np.full((n_events, n_trials), inf, dtype=float)
        prev = np.full((n_events, n_trials), -1, dtype=int)
        for trial in range(n_trials):
            penalty = 0.0 if observed[0] == behaviour_labels[trial] else mismatch_penalty
            dp[0, trial] = penalty + prior_weight * abs(trial - prior[0])
        for event_idx in range(1, n_events):
            best_cost = inf
            best_arg = -1
            best_cost_before = np.full(n_trials, inf, dtype=float)
            best_arg_before = np.full(n_trials, -1, dtype=int)
            for trial in range(n_trials):
                if dp[event_idx - 1, trial] < best_cost:
                    best_cost = dp[event_idx - 1, trial]
                    best_arg = trial
                best_cost_before[trial] = best_cost
                best_arg_before[trial] = best_arg
            for trial in range(event_idx, n_trials):
                prev_cost = best_cost_before[trial - 1]
                if not np.isfinite(prev_cost):
                    continue
                penalty = (
                    0.0
                    if observed[event_idx] == behaviour_labels[trial]
                    else mismatch_penalty
                )
                dp[event_idx, trial] = (
                    prev_cost + penalty + prior_weight * abs(trial - prior[event_idx])
                )
                prev[event_idx, trial] = best_arg_before[trial - 1]
        trial = int(np.argmin(dp[-1]))
        refined = np.empty(n_events, dtype=int)
        for event_idx in range(n_events - 1, -1, -1):
            refined[event_idx] = trial
            trial = prev[event_idx, trial]
        if np.any(refined < 0):
            return prior
        return refined

    if selected_epochs.metadata is not None and "beh_trial_index" in selected_epochs.metadata:
        trial_index = selected_epochs.metadata["beh_trial_index"].to_numpy(dtype=int)
    else:
        selection = np.asarray(selected_epochs.selection, dtype=int)
        candidates = []
        if len(selection) > 1:
            differences = np.diff(np.sort(selection))
            step = int(differences[0])
            for difference in differences[1:]:
                step = math.gcd(step, int(difference))
            offset = int(np.min(selection % step)) if step > 1 else 0
            candidates.append(
                {
                    "trial_index": (selection - offset) // step if step > 1 else selection.copy(),
                    "method": f"gcd_step_{step}_offset_{offset}",
                }
            )
        else:
            candidates.append({"trial_index": selection.copy(), "method": "selection_direct"})

        # Older epoch files do not always carry metadata. In those files, MNE's
        # selection indexes often refer to a trial event stream with four events
        # per behavioural trial. Rejected intermediate events can make adjacent
        # retained stimulus epochs differ by 3 as well as 4, so the GCD fallback
        # above can collapse to 1. Try plausible fixed strides and choose the
        # mapping that best matches event labels to behavioural labels.
        if len(selection) > 0 and int(np.nanmax(selection)) >= len(behaviour):
            for stride in (4, 3, 2):
                for offset in range(stride):
                    candidates.append(
                        {
                            "trial_index": (selection - offset) // stride,
                            "method": f"fixed_step_{stride}_offset_{offset}",
                        }
                    )

        best = None
        for candidate in candidates:
            trial_candidate = candidate["trial_index"].astype(int)
            score = score_trial_index(trial_candidate)
            valid = (trial_candidate >= 0) & (trial_candidate < len(behaviour))
            item = {
                "trial_index": trial_candidate,
                "score": score,
                "valid_count": int(np.sum(valid)),
                "method": candidate["method"],
            }
            if best is None or (item["score"], item["valid_count"]) > (
                best["score"],
                best["valid_count"],
            ):
                best = item
        trial_index = best["trial_index"]
        if best["score"] < 1.0:
            refined = refine_trial_index_by_label_sequence(trial_index)
            if score_trial_index(refined) >= best["score"]:
                trial_index = refined

    if np.any(trial_index < 0) or np.any(trial_index >= len(behaviour)):
        raise ValueError("Epoch-to-behaviour alignment produced invalid trial indices.")

    aligned_behaviour = behaviour.iloc[trial_index].reset_index(drop=True)
    return selected_epochs, aligned_behaviour
