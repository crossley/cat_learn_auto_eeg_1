#!/usr/bin/env python3
"""Raw BDF to cleaned response-locked epochs, one session at a time.

The continuous preprocessing is the task pipeline's, imported from
preprocess_task_epochs_v2.py: read, filter, crop to the task, resample,
robust average reference, ICA with automatic removal of ocular
components. Then one epoch per response pulse, TMIN to TMAX around the
response, with no baseline: the pre-response interval is the decision
period, and the pre-stimulus interval lies a different distance back
on every trial. The metadata carries the trial's behaviour row plus
stim_sample, eeg_sample (the response) and fb_sample at SFREQ, so an
analysis can baseline on the pre-stimulus interval itself. A trial
whose stimulus pulse was dropped keeps its response epoch with
stim_sample missing. AutoReject, save.

The triggers come from the tables written by find_missing_triggers.py,
as in the task pipeline. One check remains here: on every trial with
both pulses, response minus stimulus must agree with the RT within
RESP_MINUS_RT_MS.

The checks are plain asserts. A session that fails one is skipped
and the assert message recorded in output/preprocess_resp_epochs_v2_qc.csv.
Any other error is a bug and stops the run. Writes
EEG_epo_v2/resp/P<sub>_D<day>_resp-epo.fif.
"""

from multiprocessing import Pool
import os

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
import pandas as pd

from preprocess_task_epochs_v2 import (AUDIT_PATH, EVENTS_PATH, N_WORKERS, PROJECT_DIR, SFREQ, TASK_MARGIN_S,
                                       EPOCH_DIR as TASK_EPOCH_DIR, clean, prepare_raw, reference,
                                       remove_ocular_components)

EPOCH_DIR = TASK_EPOCH_DIR / "resp"
QC_PATH = PROJECT_DIR / "output" / "preprocess_resp_epochs_v2_qc.csv"

ONLY_SESSIONS = None  # or a list of (subject, day) to process only those

TMIN, TMAX, BASELINE = -1.0, 0.2, None

EVENT_ID = {"Resp/A": 30, "Resp/B": 31}

# The response pulse is sent straight from the key press; the stimulus pulse
# is flip-locked. Clean sessions give -4 to 51 ms across the four experiments.
RESP_MINUS_RT_MS = (-10, 60)


def process(subject, day, audit_row, triggers):
    assert audit_row.status == "ok", f"trigger audit: {audit_row.note}"
    behaviour = pd.concat([pd.read_csv(PROJECT_DIR / f) for f in audit_row.behaviour_file.split(";")],
                          ignore_index=True)
    raw = prepare_raw(PROJECT_DIR / audit_row.file)
    sfreq = raw.info["sfreq"]  # whole seconds, so the sample numbers stay exact after resampling
    raw.crop(max(np.floor(triggers["sample"].min() / sfreq) - TASK_MARGIN_S, 0),
             min(np.ceil(triggers["sample"].max() / sfreq) + TASK_MARGIN_S, raw.times[-1]))
    events = np.column_stack([triggers["sample"], np.zeros(len(triggers), int), triggers["code"]])
    raw, events = raw.resample(SFREQ, events=events, verbose="ERROR")
    triggers = triggers.assign(sample=events[:, 0])  # now at SFREQ
    raw = reference(raw)
    n_ocular = remove_ocular_components(raw)

    stim = triggers[triggers.kind == "stim"].set_index("trial")
    fb = triggers[triggers.kind == "fb"].set_index("trial")
    resp = triggers[triggers.kind == "resp"].set_index("trial")
    behaviour["stim_sample"] = stim["sample"]
    behaviour["fb_sample"] = fb["sample"]
    rows = behaviour.loc[resp.index].assign(eeg_sample=resp["sample"], code=resp["code"])

    both = rows.index.intersection(stim.index)
    latency_ms = (rows.eeg_sample[both] - rows.stim_sample[both]) / SFREQ * 1000
    assert (latency_ms - rows.rt[both]).between(*RESP_MINUS_RT_MS).all(), \
        "stimulus-to-response latency disagrees with RT"

    rows = rows.rename_axis("trial_index").reset_index()
    events = np.column_stack([rows.eeg_sample, np.zeros(len(rows), int), rows.code])
    event_id = {name: code for name, code in EVENT_ID.items() if code in rows.code.values}
    epochs = mne.Epochs(raw, events, event_id, tmin=TMIN, tmax=TMAX, baseline=BASELINE, picks="eeg",
                        metadata=rows.drop(columns="code"), preload=True, verbose="ERROR")
    assert len(epochs) == len(rows), f"{len(rows) - len(epochs)} epochs fell outside the recording"

    n_before = len(epochs)
    epochs = clean(epochs)
    EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    epochs.save(EPOCH_DIR / f"P{subject}_D{day}_resp-epo.fif", overwrite=True, verbose="ERROR")
    return {"status": "ok", "n_trials": len(behaviour), "n_responses": len(rows), "n_ocular_components": n_ocular,
            "n_epochs_before_autoreject": n_before, "n_epochs": len(epochs),
            "rejection_fraction": round(1 - len(epochs) / n_before, 4),
            "bad_channels": ";".join(epochs.info["bads"])}


def process_session(session):
    subject, day, audit_row, triggers = session
    try:
        row = process(subject, day, audit_row, triggers)
    except AssertionError as problem:
        row = {"status": "skipped", "reason": str(problem)}
    return {"subject": subject, "day": day, **row}


if __name__ == "__main__":
    audit = pd.read_csv(AUDIT_PATH).set_index(["subject", "day"])
    triggers = pd.read_csv(EVENTS_PATH)
    sessions = [(subject, day, audit.loc[(subject, day)],
                 triggers[(triggers.subject == subject) & (triggers.day == day)])
                for subject, day in audit.index
                if ONLY_SESSIONS is None or (subject, day) in ONLY_SESSIONS]
    qc = []
    with Pool(N_WORKERS, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(process_session, sessions):
            print(row, flush=True)
            qc.append(row)
    qc = sorted(qc, key=lambda r: (r["subject"], r["day"]))
    QC_PATH.parent.mkdir(exist_ok=True)
    pd.DataFrame(qc).to_csv(QC_PATH, index=False)
    skipped = [r for r in qc if r["status"] == "skipped"]
    print(f"{len(qc) - len(skipped)} sessions processed, {len(skipped)} skipped")
    for r in skipped:
        print(f"  P{r['subject']} day {r['day']}: {r['reason']}")
