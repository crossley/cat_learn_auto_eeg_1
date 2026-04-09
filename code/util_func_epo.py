#!/usr/bin/env python3

from pathlib import Path
import re
import os
import mne
from pyprep.prep_pipeline import PrepPipeline
from autoreject import AutoReject

os.environ["NUMBA_DISABLE_JIT"] = "1"


def util_epo_make_from_bdf():
    """Create epoched FIF files from raw BDF files."""

    raw_dir = Path("../EEG")
    epo_dir = Path("../EEG_epo")
    epo_dir.mkdir(parents=True, exist_ok=True)

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

    for raw_path in sorted(raw_dir.glob("*.bdf")):

        m = bdf_re.match(raw_path.name)
        if m is None:
            continue

        subject = int(m.group(1))
        day_token = m.group(2)

        raw = mne.io.read_raw_bdf(raw_path, preload=True, stim_channel="Status", verbose="ERROR")

        cap_chs = [ch for ch in raw.ch_names if cap_ch_re.match(ch)]
        if len(cap_chs) == 64:
            raw.rename_channels(dict(zip(cap_chs, biosemi64_montage.ch_names)))

        raw.set_channel_types({k: v for k, v in aux_types.items() if k in raw.ch_names}, verbose="ERROR")
        raw.set_montage(biosemi64_montage, on_missing="ignore")

        raw.notch_filter(freqs=[50, 100], verbose="ERROR")
        raw.filter(l_freq=0.1, h_freq=40.0, verbose="ERROR")

        prep_params = {"ref_chs": "eeg", "reref_chs": "eeg", "line_freqs": [50]}
        prep_raw = raw.copy().pick_types(eeg=True, eog=False, stim=False, exclude=[])
        prep = PrepPipeline(
            prep_raw,
            prep_params=prep_params,
            montage=prep_raw.get_montage(),
            random_state=42,
        )
        prep.fit()
        raw.info["bads"] = prep_raw.info["bads"]
        raw.interpolate_bads(reset_bads=True, verbose="ERROR")

        events = mne.find_events(
            raw,
            stim_channel="Status",
            shortest_event=1,
            verbose="ERROR",
        )
        picks = mne.pick_types(raw.info, eeg=True, eog=False, stim=False, exclude="bads")

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=-0.2,
            tmax=0.8,
            baseline=(-0.2, 0.0),
            picks=picks,
            preload=True,
            reject_by_annotation=True,
            verbose="ERROR",
        )

        ar = AutoReject(random_state=42, verbose=False)
        epochs = ar.fit_transform(epochs)

        epo_name = f"P{subject}_D{day_token}-epo.fif"
        epo_path = epo_dir / epo_name
        epochs.save(epo_path, overwrite=True, verbose="ERROR")
