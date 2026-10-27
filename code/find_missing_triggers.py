#!/usr/bin/env python3
"""Label every trigger pulse with its trial and kind; list the dropped ones.

The behaviour file says which triggers each trial sent: an ITI marker,
the stimulus, the response (none on a timeout) and the feedback, with
START before the first trial and ITI then END after the last. The
recording is walked alongside that expected sequence: each pulse is
matched to the next expected trigger, and when the pulse that arrives
is the one after it, the expected trigger is listed as dropped. A
pulse that matches nothing, or more than a trial's worth of dropped
triggers in a row, means the stream no longer follows the behaviour
file and the session is marked unusable.

Outputs, read by preprocess_task_epochs_v2.py:
  output/trigger_events.csv    one row per matched pulse: subject, day,
                               trial, kind, code, sample (raw rate)
  output/missing_triggers.csv  the dropped triggers: subject, day, trial, kind
  output/trigger_audit.csv     one row per recording: behaviour file(s),
                               status, note, counts

Handled by rule: trigger bits can arrive one sample early, giving a
one-sample stray value just before a real pulse; find_events(min_duration
= 2 samples) drops them. A response before the stimulus overwrites the
stimulus pulse, so that stimulus is listed as dropped. A recording with
an aborted run before the real one is read from its last START.
One ITI trigger was dropped in four sessions; the walk lists it like any
other dropped trigger. The behaviour file records no trigger codes; they
follow from cat, resp and fb as experiment.py sent them.

Reads the Status channel only. Run once after data are added.
"""

from pathlib import Path

import mne
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "EEG"  # P<sub>_D<day>[_<n>].bdf
BEHAVIOUR_DIR = PROJECT_DIR / "Behavioural"  # sub_<sub>_day_<day><sub>_data.csv
EVENTS_PATH = PROJECT_DIR / "output" / "trigger_events.csv"
MISSING_PATH = PROJECT_DIR / "output" / "missing_triggers.csv"
AUDIT_PATH = PROJECT_DIR / "output" / "trigger_audit.csv"

START, ITI, END = 10, 11, 15
MAX_DROPPED_IN_A_ROW = 4  # one trial's worth; more means the pulses no longer follow the file
STIM_CODE = {"A": 20, "B": 21}
RESP_CODE = {"A": 30, "B": 31}
FB_CODE = {"Correct": 40, "Incorrect": 41}


class DataProblem(Exception):
    """The recording and the behaviour file disagree. Report the session."""


def behaviour_files(subject, day):
    path = BEHAVIOUR_DIR / f"sub_{subject:03d}_day_{day}{subject:02d}_data.csv"
    if not path.exists():
        raise DataProblem(f"no behaviour file {path.name}")
    return [path]


def read_behaviour(files):
    """Rows in recorded order, with the trigger code each event should have sent."""
    behaviour = pd.read_csv(files[0])
    behaviour["trigger_stim"] = behaviour.cat.map(STIM_CODE)
    behaviour["trigger_resp"] = behaviour.resp.map(RESP_CODE)
    behaviour["trigger_fb"] = behaviour.fb.map(FB_CODE)
    if behaviour[["trigger_stim", "trigger_resp", "trigger_fb"]].isna().any().any():
        raise DataProblem("behaviour file has a cat, resp or fb value with no trigger code")
    return behaviour


def expected_triggers(behaviour):
    """(trial, kind, code) in the order the task sent them."""
    expected = [(-1, "start", START)]
    for trial, row in behaviour.iterrows():
        expected.append((trial, "iti", ITI))
        expected.append((trial, "stim", int(row.trigger_stim)))
        if pd.notna(row.trigger_resp):  # a timeout sends no response trigger
            expected.append((trial, "resp", int(row.trigger_resp)))
        expected.append((trial, "fb", int(row.trigger_fb)))
    expected += [(len(behaviour), "iti", ITI), (len(behaviour), "end", END)]
    return expected


def recorded_pulses(raw):
    """(sample, value) of every pulse from the last START, and the START count."""
    events = mne.find_events(raw, stim_channel="Status", min_duration=2 / raw.info["sfreq"],
                             verbose="ERROR")
    values = events[:, 2] & 0xFF
    samples, values = events[values != 0, 0], values[values != 0]
    starts = np.flatnonzero(values == START)
    if len(starts) == 0:
        raise DataProblem("no START trigger")
    return samples[starts[-1]:], values[starts[-1]:], len(starts)


def match(samples, values, expected):
    """Walk the pulses alongside the expected triggers."""
    matched, missing, i, dropped_in_a_row = [], [], 0, 0
    for trial, kind, code in expected:
        if i < len(values) and values[i] == code:
            matched.append({"trial": trial, "kind": kind, "code": code, "sample": int(samples[i])})
            i, dropped_in_a_row = i + 1, 0
        else:
            missing.append({"trial": trial, "kind": kind})
            dropped_in_a_row += 1
            if dropped_in_a_row > MAX_DROPPED_IN_A_ROW or kind in ("start", "end"):
                raise DataProblem(f"trial {trial}: pulses stop matching the behaviour file ({kind} expected)")
    if i < len(values):
        raise DataProblem(f"{len(values) - i} pulses after the END trigger")
    return matched, missing


def audit(subject, day, bdf):
    behaviour_paths = behaviour_files(subject, day)
    behaviour = read_behaviour(behaviour_paths)
    raw = mne.io.read_raw_bdf(bdf, preload=False, stim_channel="Status", verbose="ERROR")
    samples, values, n_starts = recorded_pulses(raw)
    matched, missing = match(samples, values, expected_triggers(behaviour))
    row = {"status": "ok", "note": "",
           "behaviour_file": ";".join(str(p.relative_to(PROJECT_DIR)) for p in behaviour_paths),
           "n_rows": len(behaviour), "n_starts": n_starts, "n_pulses": len(values),
           "sfreq": raw.info["sfreq"]}
    for kind in ("iti", "stim", "resp", "fb"):
        row[f"n_missing_{kind}"] = sum(m["kind"] == kind for m in missing)
    events = [m for m in matched if m["kind"] not in ("start", "end")]
    return row, events, missing


if __name__ == "__main__":
    audit_rows, event_rows, missing_rows = [], [], []
    for bdf in sorted(RAW_DIR.glob("*.bdf")):
        subject, day = (int(part[1:]) for part in bdf.stem.split("_")[:2])  # P<sub>_D<day>[_<n>]
        try:
            row, events, missing = audit(subject, day, bdf)
        except DataProblem as problem:
            row, events, missing = {"status": "unusable", "note": str(problem)}, [], []
        audit_rows.append({"subject": subject, "day": day, "file": str(bdf.relative_to(PROJECT_DIR)), **row})
        event_rows += [{"subject": subject, "day": day, **e} for e in events]
        missing_rows += [{"subject": subject, "day": day, **m} for m in missing]
        print(audit_rows[-1], flush=True)
    AUDIT_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(AUDIT_PATH, index=False)
    pd.DataFrame(event_rows, columns=["subject", "day", "trial", "kind", "code", "sample"]).to_csv(EVENTS_PATH, index=False)
    pd.DataFrame(missing_rows, columns=["subject", "day", "trial", "kind"]).to_csv(MISSING_PATH, index=False)
    print(f"{sum(r['status'] == 'ok' for r in audit_rows)} of {len(audit_rows)} recordings ok, "
          f"{len(missing_rows)} dropped triggers")
