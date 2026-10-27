#!/usr/bin/env python3
"""Raw BDF to cleaned task epochs, one session at a time.

The standard pipeline: read, filter, crop to the task, resample, robust
average reference, ICA with automatic removal of ocular components,
epoch on the stimulus and feedback triggers with the behaviour file
attached as metadata, AutoReject, save. The metadata also carries each trial's response
trigger sample (resp_sample) and the epoch's own sample (eeg_sample),
at SFREQ, for response-locked epoching by a later script.

The triggers come from two tables written by find_missing_triggers.py.
trigger_audit.csv names each recording's behaviour file(s) and says
whether it is usable. trigger_events.csv gives every trigger pulse its
trial number and kind, matched against the behaviour file's expected
sequence, so a dropped trigger has no row and that trial simply has no
epoch of that kind. One check remains here: on every trial with a
response, the stimulus-to-feedback latency must be the RT plus the
task's feedback gap.

The checks are plain asserts. A session that fails one is skipped
and the assert message recorded in output/preprocess_task_epochs_v2_qc.csv.
Any other error is a bug and stops the run. Writes
EEG_epo_v2/P<sub>_D<day>-epo.fif.
"""

from multiprocessing import Pool
from pathlib import Path
import os

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import mne
import numpy as np
import pandas as pd
from autoreject import AutoReject
from pyprep.prep_pipeline import PrepPipeline

PROJECT_DIR = Path(__file__).resolve().parent.parent
EPOCH_DIR = PROJECT_DIR / "EEG_epo_v2"
AUDIT_PATH = PROJECT_DIR / "output" / "trigger_audit.csv"
EVENTS_PATH = PROJECT_DIR / "output" / "trigger_events.csv"
QC_PATH = PROJECT_DIR / "output" / "preprocess_task_epochs_v2_qc.csv"

ONLY_SESSIONS = None  # or a list of (subject, day) to process only those
N_WORKERS = 8  # each session holds the raw recording in memory, about 7 GB at 2048 Hz

L_FREQ, H_FREQ = 0.5, 40.0
SFREQ = 256
TMIN, TMAX, BASELINE = -0.2, 0.8, (-0.2, 0.0)
RANDOM_STATE = 42
# bipolar EOG: EXG2 (below left eye) - Fp1 = VEOG, EXG1 (beside left eye) - F8 = HEOG
EOG_ANODES, EOG_CATHODES, EOG_NAMES = ["EXG2", "EXG1"], ["Fp1", "F8"], ["VEOG", "HEOG"]
EOG_CORRELATION = 0.5  # |r| with VEOG or HEOG above which a component is ocular
# Live eye electrodes give 11-196 uV above 1 Hz and correlate at most 0.75 with
# each other (all 183 recordings of Exp1-2); unconnected ones are identical or mV-scale.
EOG_MAX_STD_UV, EOG_MAX_R = 250, 0.9
ICA_HIGHPASS = 1.0
# Recordings start before the task and some run on after the cap came off;
# the reference and the ICA are fitted on the task interval only.
TASK_MARGIN_S = 5

EVENT_ID = {"Stim/A": 20, "Stim/B": 21, "FB/Cor": 40, "FB/Inc": 41}

# Feedback follows the response after a fixed gap plus display lag.
# Clean sessions give 212-419 ms; a misaligned trial gives seconds.
LATENCY_MINUS_RT_MS = (200, 450)


def prepare_raw(bdf):
    """Read one recording, keep the 64-channel cap, filter."""
    montage = mne.channels.make_standard_montage("biosemi64")
    raw = mne.io.read_raw_bdf(bdf, preload=True, stim_channel="Status", verbose="ERROR")
    cap = [ch for ch in raw.ch_names if ch[0] in "AB" and ch[1:].isdigit()]
    assert len(cap) == 64, f"{len(cap)} cap channels, expected 64"
    raw.drop_channels([ch for ch in raw.ch_names if ch[0] in "CDEFGH" and ch[1:].isdigit()])
    raw.rename_channels(dict(zip(cap, montage.ch_names)))
    raw.set_channel_types({ch: "eog" for ch in raw.ch_names if ch.startswith("EXG")})
    raw.set_channel_types({ch: "misc" for ch in ["GSR1", "GSR2", "Erg1", "Erg2", "Resp", "Plet", "Temp"]
                           if ch in raw.ch_names})
    raw.set_montage(montage)
    raw.filter(L_FREQ, H_FREQ, verbose="ERROR")
    return raw


def reference(raw):
    """PyPREP robust average reference; returns the referenced copy."""
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True)]
    prep = PrepPipeline(raw, prep_params={"ref_chs": names, "reref_chs": names, "line_freqs": []},
                        montage=raw.get_montage(), ransac=False, random_state=RANDOM_STATE)
    prep.fit()
    referenced = prep.raw
    if np.abs(referenced.get_data(picks=names).mean(axis=0)).max() > 1e-12:
        raise ValueError("PyPREP output is not average referenced")
    referenced.info["bads"] = sorted(prep.still_noisy_channels)
    referenced.set_eeg_reference(ref_channels=[], verbose="ERROR")
    return referenced


def remove_ocular_components(raw):
    """ICA fitted above 1 Hz; components correlating |r| > 0.5 with VEOG or HEOG are removed."""
    ica_raw = raw.copy().filter(ICA_HIGHPASS, None, picks=["eeg", "eog"], verbose="ERROR")  # eog too
    mne.set_bipolar_reference(ica_raw, EOG_ANODES, EOG_CATHODES, ch_name=EOG_NAMES,
                              drop_refs=False, copy=False, verbose="ERROR")
    ica_raw.set_channel_types({name: "eog" for name in EOG_NAMES})
    eog = ica_raw.get_data(picks=EOG_ANODES) * 1e6
    assert eog.std(axis=1).max() < EOG_MAX_STD_UV, f"EOG channel at {eog.std(axis=1).max():.0f} uV: electrode not recording"
    assert abs(np.corrcoef(eog)[0, 1]) < EOG_MAX_R, "EXG1 and EXG2 carry the same signal: EOG not connected"
    ica = mne.preprocessing.ICA(n_components=mne.compute_rank(ica_raw, verbose="ERROR")["eeg"],
                                method="picard", random_state=RANDOM_STATE, verbose="ERROR")
    ica.fit(ica_raw, picks="eeg", verbose="ERROR")
    ica.exclude, _ = ica.find_bads_eog(ica_raw, ch_name=EOG_NAMES, measure="correlation",
                                       threshold=EOG_CORRELATION, verbose="ERROR")
    assert 1 <= len(ica.exclude) <= 3, f"{len(ica.exclude)} ocular components found"
    ica.apply(raw, verbose="ERROR")
    return len(ica.exclude)


def clean(epochs):
    """AutoReject, then restore the average reference across good channels."""
    ar = AutoReject(cv=3, thresh_method="bayesian_optimization", n_interpolate=[4, 8, 12],
                    consensus=np.linspace(0.5, 1.0, 3), n_jobs=1, random_state=RANDOM_STATE,
                    verbose=False)
    ar.fit(epochs[::10])
    cleaned = ar.transform(epochs)
    cleaned.set_eeg_reference("average", projection=False, verbose="ERROR")
    return cleaned


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
    behaviour["resp_sample"] = resp["sample"]  # for response-locked epoching later
    stim_rows = behaviour.loc[stim.index].assign(event_type="stim", eeg_sample=stim["sample"], code=stim["code"])
    fb_rows = behaviour.loc[fb.index].assign(event_type="fb", eeg_sample=fb["sample"], code=fb["code"])

    both = stim.index.intersection(fb.index)
    latency_ms = (fb_rows.eeg_sample[both] - stim_rows.eeg_sample[both]) / SFREQ * 1000
    assert (latency_ms - behaviour.rt[both]).between(*LATENCY_MINUS_RT_MS).all(), \
        "stimulus-to-feedback latency disagrees with RT"

    rows = pd.concat([stim_rows, fb_rows]).sort_values("eeg_sample").rename_axis("trial_index").reset_index()
    events = np.column_stack([rows.eeg_sample, np.zeros(len(rows), int), rows.code])
    event_id = {name: code for name, code in EVENT_ID.items() if code in rows.code.values}
    epochs = mne.Epochs(raw, events, event_id, tmin=TMIN, tmax=TMAX, baseline=BASELINE, picks="eeg",
                        metadata=rows.drop(columns="code"), preload=True, verbose="ERROR")
    assert len(epochs) == len(rows), f"{len(rows) - len(epochs)} epochs fell outside the recording"

    n_before = len(epochs)
    epochs = clean(epochs)
    EPOCH_DIR.mkdir(parents=True, exist_ok=True)
    epochs.save(EPOCH_DIR / f"P{subject}_D{day}-epo.fif", overwrite=True, verbose="ERROR")
    return {"status": "ok", "n_trials": len(behaviour), "n_ocular_components": n_ocular,
            "n_stim_epochs": int((epochs.metadata.event_type == "stim").sum()),
            "n_fb_epochs": int((epochs.metadata.event_type == "fb").sum()),
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
