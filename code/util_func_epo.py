#!/usr/bin/env python3

from pathlib import Path
import re
import os
import mne
import numpy as np
from joblib import Parallel, delayed
from pyprep.prep_pipeline import PrepPipeline
from autoreject import AutoReject

# os.environ["NUMBA_DISABLE_JIT"] = "1"


def process_single_subject(raw_path, epo_dir):

    raw_path = Path(raw_path)
    epo_dir = Path(epo_dir)

    event_id = {
        "Stim/A": 20,
        "Stim/B": 21,
        "FB/Cor": 40,
        "FB/Inc": 41,
    }

    bdf_re = re.compile(r"^P(\d+)_D([\d_]+)\.bdf$")
    cap_ch_re = re.compile(r"^[AB]\d+$")
    aux_types = {f"EXG{i}": "eog" for i in range(1, 9)}
    aux_types.update(
        {
            "GSR1": "misc",
            "GSR2": "misc",
            "Erg1": "misc",
            "Erg2": "misc",
            "Resp": "misc",
            "Plet": "misc",
            "Temp": "misc",
        }
    )
    biosemi64_montage = mne.channels.make_standard_montage("biosemi64")

    m = bdf_re.match(raw_path.name)
    if m is None:
        return None
    subject = int(m.group(1))
    day_token = m.group(2)

    raw = mne.io.read_raw_bdf(raw_path, preload=True, stim_channel="Status", verbose="ERROR")
    cap_chs = [ch for ch in raw.ch_names if cap_ch_re.match(ch)]
    if len(cap_chs) == 64:
        raw.rename_channels(dict(zip(cap_chs, biosemi64_montage.ch_names)))

    channel_types = {}
    for channel_name, channel_type in aux_types.items():
        if channel_name in raw.ch_names:
            channel_types[channel_name] = channel_type
    raw.set_channel_types(channel_types, verbose="ERROR")
    raw.set_montage(biosemi64_montage, on_missing="ignore")

    raw.filter(l_freq=0.5, h_freq=40.0, verbose="ERROR")

    events = mne.find_events(raw, stim_channel="Status", shortest_event=1, verbose="ERROR")

    # codes, counts = np.unique(events[:, 2], return_counts=True)
    # print(f"[{raw_path.name}] event counts")
    # for code, count in zip(codes, counts):
    #     print(f"event {int(code)}: {int(count)}")

    raw, events = raw.resample(256, npad="auto", events=events, verbose="ERROR")

    eeg_chans = [ch for ch, ch_type in zip(raw.ch_names, raw.get_channel_types()) if ch_type == "eeg"]
    prep_params = {"ref_chs": eeg_chans, "reref_chs": eeg_chans, "line_freqs": []}
    prep = PrepPipeline(
        raw,
        prep_params=prep_params,
        montage=biosemi64_montage,
        ransac=False,
        random_state=42,
    )
    prep.fit()
    raw.info["bads"] = prep.still_noisy_channels

    epochs = mne.Epochs(
        raw,
        events=events,
        event_id=event_id,
        tmin=-0.2,
        tmax=0.8,
        baseline=(-0.2, 0.0),
        picks="eeg",
        preload=True,
        reject_by_annotation=True,
        verbose="ERROR",
    )

    ar = AutoReject(
        cv=3,
        thresh_method="bayesian_optimization",
        n_interpolate=[4, 8, 12],
        consensus=np.linspace(0.5, 1.0, 3),
        n_jobs=1,
        random_state=42,
        verbose=False,
    )
    ar.fit(epochs[::10])
    epochs_clean = ar.transform(epochs)

    epo_name = f"P{subject}_D{day_token}-epo.fif"
    epo_path = epo_dir / epo_name
    epochs_clean.save(epo_path, overwrite=True, verbose="ERROR")
    del raw, epochs, epochs_clean
    return f"Finished sub {subject}, day {day_token}"


def util_epo_make_from_bdf():
    """Create epoched FIF files from raw BDF files."""
    raw_dir = Path("../EEG")
    epo_dir = Path("../EEG_epo")
    epo_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(raw_dir.glob("*.bdf"))
    results = Parallel(n_jobs=10)(
        delayed(process_single_subject)(raw_path, epo_dir) for raw_path in raw_files
    )

    for result in results:
        if result is not None:
            print(result)
