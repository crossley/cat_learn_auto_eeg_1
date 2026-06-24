#!/usr/bin/env python3
"""Shared trial-sequence feature interface for state-space and sequence analyses."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import StringIO
import json
from pathlib import Path
import os
import warnings

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from connect_sensorwide_analysis import CHANNEL_SUBSET, compute_coherence_components
from load_project_data import align_behaviour_to_epochs, load_sessions
from sensor_rois import STRICT_SENSOR_ROIS, cross_roi_pairs
from util_mvpa import build_clf, pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
FEATURE_CACHE_VERSION = 1
DEFAULT_CACHED_FEATURES = {"time_frequency", "connectivity", "imcoh", "mvpa", "mvpa_decision"}


@dataclass
class SequenceDataset:
    """Standard feature payload: trials x time x features."""

    X: np.ndarray
    y: np.ndarray
    time: np.ndarray
    feature_names: list[str]
    metadata: pd.DataFrame
    subject: int
    session: int
    session_file: str
    feature_kind: str


def _label_vector_from_behaviour(behaviour: pd.DataFrame) -> np.ndarray:
    labels = behaviour["cat"].astype(str).to_numpy()
    y = np.full(len(labels), -1, dtype=int)
    y[labels == "A"] = 0
    y[labels == "B"] = 1
    if np.any(y < 0):
        bad = sorted(set(labels[y < 0].tolist()))
        raise ValueError(f"unexpected category labels: {bad}")
    return y


def _metadata(session_item: dict, behaviour: pd.DataFrame, n_trials: int) -> pd.DataFrame:
    meta = behaviour.reset_index(drop=True).copy()
    meta["subject"] = int(session_item["subject"])
    meta["session"] = int(session_item["day"])
    meta["day"] = int(session_item["day"])
    meta["session_file"] = session_item["epo_file"]
    meta["trial_index_aligned"] = np.arange(n_trials, dtype=int)
    return meta


def _read_stim_epochs(session_item: dict, event_names=("Stim/A", "Stim/B")):
    epochs = mne.read_epochs(session_item["epo_path"], preload=False, verbose="ERROR")
    stim_epochs, beh = align_behaviour_to_epochs(
        session_item["beh"],
        epochs,
        event_names=event_names,
    )
    stim_epochs = stim_epochs.copy().load_data()
    pick_eeg_interpolate_bads(stim_epochs)
    return stim_epochs, beh


def load_sequence_sessions(load_epochs: bool = False) -> list[dict]:
    """Return project sessions using the established loader contract."""

    return load_sessions(load_epochs=load_epochs)


def _json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, tuple):
        return [_json_safe(x) for x in value]
    if isinstance(value, list):
        return [_json_safe(x) for x in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in sorted(value.items())}
    return value


def _normal_feature_kind(feature_kind: str) -> str:
    if feature_kind == "connectivity":
        return "imcoh"
    if feature_kind == "mvpa":
        return "mvpa_decision"
    return str(feature_kind)


def _feature_cache_path(
    session_item: dict,
    feature_kind: str,
    kwargs: dict,
    cache_dir: Path | str = OUTPUT_DIR,
) -> Path:
    cache_dir = Path(cache_dir)
    epo_path = Path(session_item["epo_path"])
    beh_path = Path(session_item["beh_path"]) if "beh_path" in session_item else None
    payload = {
        "version": FEATURE_CACHE_VERSION,
        "feature_kind": _normal_feature_kind(feature_kind),
        "subject": int(session_item["subject"]),
        "day": int(session_item["day"]),
        "epo_file": session_item.get("epo_file", epo_path.name),
        "epo_mtime_ns": epo_path.stat().st_mtime_ns if epo_path.exists() else None,
        "beh_file": session_item.get("beh_file", beh_path.name if beh_path else ""),
        "beh_mtime_ns": beh_path.stat().st_mtime_ns if beh_path and beh_path.exists() else None,
        "kwargs": _json_safe(kwargs),
    }
    token = hashlib.md5(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    label = _normal_feature_kind(feature_kind)
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    return cache_dir / f"sequence_feature_cache_{label}_sub{subject:03d}_day{day}_{token}.npz"


def _write_sequence_cache(path: Path, dataset: SequenceDataset):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        X=dataset.X,
        y=dataset.y,
        time=dataset.time,
        feature_names=np.asarray(dataset.feature_names, dtype=str),
        metadata_json=np.asarray(dataset.metadata.to_json(orient="split")),
        subject=np.asarray(dataset.subject, dtype=int),
        session=np.asarray(dataset.session, dtype=int),
        session_file=np.asarray(dataset.session_file, dtype=str),
        feature_kind=np.asarray(dataset.feature_kind, dtype=str),
    )


def _read_sequence_cache(path: Path) -> SequenceDataset:
    with np.load(path, allow_pickle=False) as z:
        metadata = pd.read_json(StringIO(str(z["metadata_json"])), orient="split")
        return SequenceDataset(
            X=z["X"],
            y=z["y"],
            time=z["time"],
            feature_names=z["feature_names"].astype(str).tolist(),
            metadata=metadata,
            subject=int(z["subject"]),
            session=int(z["session"]),
            session_file=str(z["session_file"]),
            feature_kind=str(z["feature_kind"]),
        )


def _time_mask(times: np.ndarray, tmin: float | None, tmax: float | None) -> np.ndarray:
    mask = np.ones(len(times), dtype=bool)
    if tmin is not None:
        mask &= times >= float(tmin)
    if tmax is not None:
        mask &= times <= float(tmax)
    if not np.any(mask):
        raise ValueError(f"time crop produced no samples: tmin={tmin}, tmax={tmax}")
    return mask


def load_voltage_sequence(
    session_item: dict,
    resample_hz: float | None = 128.0,
    picks: list[str] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
) -> SequenceDataset:
    """Voltage ERP features as trials x time x channels."""

    epochs, beh = _read_stim_epochs(session_item)
    if picks is not None:
        present = [ch for ch in picks if ch in epochs.ch_names]
        if len(present) == 0:
            raise ValueError("none of the requested channels are present")
        epochs.pick(present)
    if resample_hz is not None:
        epochs.resample(float(resample_hz), npad="auto")
    data = epochs.get_data()
    times = epochs.times.copy()
    mask = _time_mask(times, tmin, tmax)
    data = data[:, :, mask]
    times = times[mask]
    X = np.transpose(data, (0, 2, 1))
    y = _label_vector_from_behaviour(beh)
    return SequenceDataset(
        X=X,
        y=y,
        time=times,
        feature_names=list(epochs.ch_names),
        metadata=_metadata(session_item, beh, len(y)),
        subject=int(session_item["subject"]),
        session=int(session_item["day"]),
        session_file=session_item["epo_file"],
        feature_kind="voltage",
    )


def load_time_frequency_sequence(
    session_item: dict,
    freqs: np.ndarray | None = None,
    n_cycles: np.ndarray | float | None = None,
    resample_hz: float | None = 128.0,
    baseline: tuple[float | None, float | None] = (None, 0.0),
    mode: str = "logratio",
    picks: list[str] | None = None,
    tmin: float | None = None,
    tmax: float | None = None,
) -> SequenceDataset:
    """Morlet power features as trials x time x channel-frequency features."""

    from mne.time_frequency import tfr_array_morlet

    epochs, beh = _read_stim_epochs(session_item)
    if picks is not None:
        present = [ch for ch in picks if ch in epochs.ch_names]
        if len(present) == 0:
            raise ValueError("none of the requested channels are present")
        epochs.pick(present)
    if resample_hz is not None:
        epochs.resample(float(resample_hz), npad="auto")
    if freqs is None:
        freqs = np.array([4, 6, 8, 10, 12, 16, 20, 30], dtype=float)
    freqs = np.asarray(freqs, dtype=float)
    if n_cycles is None:
        n_cycles = np.maximum(freqs / 2.0, 3.0)
    power = tfr_array_morlet(
        epochs.get_data(),
        sfreq=float(epochs.info["sfreq"]),
        freqs=freqs,
        n_cycles=n_cycles,
        output="power",
        zero_mean=True,
        n_jobs=1,
        verbose="ERROR",
    )
    # trials x channels x freqs x times -> trials x times x channel-frequency
    times = epochs.times.copy()
    if baseline is not None:
        b0, b1 = baseline
        idx = np.ones(len(times), dtype=bool)
        if b0 is not None:
            idx &= times >= float(b0)
        if b1 is not None:
            idx &= times <= float(b1)
        if np.any(idx):
            base = np.nanmean(power[:, :, :, idx], axis=-1, keepdims=True)
            if mode == "logratio":
                power = np.log10((power + np.finfo(float).eps) / (base + np.finfo(float).eps))
            elif mode == "ratio":
                power = power / (base + np.finfo(float).eps)
            elif mode == "subtract":
                power = power - base
            else:
                raise ValueError(f"unsupported baseline mode: {mode}")
    mask = _time_mask(times, tmin, tmax)
    power = power[:, :, :, mask]
    times = times[mask]
    X = np.transpose(power, (0, 3, 1, 2)).reshape(len(epochs), len(times), -1)
    feature_names = [
        f"{ch}_{freq:g}Hz" for ch in epochs.ch_names for freq in freqs
    ]
    y = _label_vector_from_behaviour(beh)
    return SequenceDataset(
        X=X,
        y=y,
        time=times,
        feature_names=feature_names,
        metadata=_metadata(session_item, beh, len(y)),
        subject=int(session_item["subject"]),
        session=int(session_item["day"]),
        session_file=session_item["epo_file"],
        feature_kind="time_frequency",
    )


def _roi_edge_names(channel_names: list[str], roi_pair: str) -> tuple[list[tuple[int, int]], list[str]]:
    channel_index = {ch: i for i, ch in enumerate(channel_names)}
    if roi_pair == "sensorwide":
        pairs = []
        names = []
        for i, ch_i in enumerate(channel_names):
            for j in range(i + 1, len(channel_names)):
                pairs.append((i, j))
                names.append(f"{ch_i}-{channel_names[j]}")
        return pairs, names
    left, right = roi_pair.split("_to_")
    pairs = []
    names = []
    for ch_i, ch_j in cross_roi_pairs(left, right):
        if ch_i in channel_index and ch_j in channel_index:
            pairs.append((channel_index[ch_i], channel_index[ch_j]))
            names.append(f"{ch_i}-{ch_j}")
    if len(pairs) == 0:
        raise ValueError(f"no valid edges for roi_pair={roi_pair}")
    return pairs, names


def load_connectivity_sequence(
    session_item: dict,
    roi_pair: str = "visual_to_central",
    window_sec: float = 0.05,
    step_sec: float = 0.025,
    tmin: float = 0.0,
    tmax: float = 0.8,
) -> SequenceDataset:
    """Trial-wise abs-imcoh edge features as trials x windows x edges."""

    epochs, beh = _read_stim_epochs(session_item)
    channel_subset = sorted(set(CHANNEL_SUBSET) & set(epochs.ch_names))
    if roi_pair != "sensorwide":
        left, right = roi_pair.split("_to_")
        channel_subset = sorted(
            (set(STRICT_SENSOR_ROIS[left]) | set(STRICT_SENSOR_ROIS[right]))
            & set(epochs.ch_names)
        )
    epochs.pick(channel_subset)
    epochs = epochs.apply_hilbert(envelope=False, verbose="ERROR")
    data = epochs.get_data()
    times = epochs.times.copy()
    pairs, feature_names = _roi_edge_names(list(epochs.ch_names), roi_pair)
    starts = np.arange(tmin, tmax - window_sec + 1e-12, step_sec)
    X = np.full((data.shape[0], len(starts), len(pairs)), np.nan, dtype=float)
    for wi, t_start in enumerate(starts):
        t_end = float(t_start + window_sec)
        i0 = int(np.searchsorted(times, t_start, side="left"))
        i1 = int(np.searchsorted(times, t_end, side="left"))
        if i1 - i0 < 2:
            continue
        for ei, (ch_i, ch_j) in enumerate(pairs):
            for tri in range(data.shape[0]):
                X[tri, wi, ei] = compute_coherence_components(
                    data[tri, ch_i, i0:i1],
                    data[tri, ch_j, i0:i1],
                )["conn_val"]
    y = _label_vector_from_behaviour(beh)
    return SequenceDataset(
        X=X,
        y=y,
        time=starts.copy(),
        feature_names=feature_names,
        metadata=_metadata(session_item, beh, len(y)),
        subject=int(session_item["subject"]),
        session=int(session_item["day"]),
        session_file=session_item["epo_file"],
        feature_kind="connectivity",
    )


def load_mvpa_decision_sequence(
    session_item: dict,
    resample_hz: float | None = 128.0,
    random_state: int = 42,
    tmin: float | None = None,
    tmax: float | None = None,
) -> SequenceDataset:
    """Cross-validated linear MVPA decision values as trials x time x 1."""

    voltage = load_voltage_sequence(
        session_item,
        resample_hz=resample_hz,
        tmin=tmin,
        tmax=tmax,
    )
    n_trials, n_times, _ = voltage.X.shape
    decision = np.full((n_trials, n_times), np.nan, dtype=float)
    min_class = int(min(np.sum(voltage.y == 0), np.sum(voltage.y == 1)))
    if min_class < 2:
        raise ValueError("need at least two trials per class for MVPA decision values")
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for ti in range(n_times):
        clf = build_clf(random_state=random_state)
        Xt = voltage.X[:, ti, :]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            vals = cross_val_predict(
                clf,
                Xt,
                voltage.y,
                cv=cv,
                method="decision_function",
                n_jobs=1,
            )
        decision[:, ti] = vals.astype(float)
    return SequenceDataset(
        X=decision[:, :, None],
        y=voltage.y.copy(),
        time=voltage.time.copy(),
        feature_names=["linear_mvpa_decision"],
        metadata=voltage.metadata.copy(),
        subject=voltage.subject,
        session=voltage.session,
        session_file=voltage.session_file,
        feature_kind="mvpa_decision",
    )


def load_feature_sequence(
    session_item: dict,
    feature_kind: str,
    use_cache: bool = True,
    force_recompute: bool = False,
    cache_dir: Path | str = OUTPUT_DIR,
    verbose: bool = False,
    **kwargs,
) -> SequenceDataset:
    """Dispatch one session into the shared trials x time x features interface.

    Expensive derived features are cached by default as compressed NPZ files in
    the project output directory. Voltage remains uncached unless requested
    through a future caller because it is mostly file I/O and cheap reshaping.
    """

    normalized = _normal_feature_kind(feature_kind)
    cache_path = _feature_cache_path(session_item, normalized, kwargs, cache_dir)
    should_cache = bool(use_cache and normalized in DEFAULT_CACHED_FEATURES)
    if should_cache and cache_path.exists() and not force_recompute:
        if verbose:
            print(f"[sequence features] cache hit {cache_path.name}", flush=True)
        return _read_sequence_cache(cache_path)

    if verbose and should_cache:
        print(f"[sequence features] cache miss {cache_path.name}", flush=True)
    if normalized == "voltage":
        dataset = load_voltage_sequence(session_item, **kwargs)
    elif normalized == "time_frequency":
        dataset = load_time_frequency_sequence(session_item, **kwargs)
    elif normalized == "imcoh":
        dataset = load_connectivity_sequence(session_item, **kwargs)
    elif normalized == "mvpa_decision":
        dataset = load_mvpa_decision_sequence(session_item, **kwargs)
    else:
        raise ValueError(f"unknown feature_kind: {feature_kind}")

    if should_cache:
        _write_sequence_cache(cache_path, dataset)
        if verbose:
            size_mb = cache_path.stat().st_size / (1024.0 * 1024.0)
            print(f"[sequence features] wrote {cache_path.name} ({size_mb:.1f} MB)", flush=True)
    return dataset
