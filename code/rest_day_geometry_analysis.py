#!/usr/bin/env python3
"""Resting-state day-geometry controls for connectivity and spectral power."""

from __future__ import annotations

from pathlib import Path
import json
import os
import re
import time

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from connect_sensorwide_analysis import CHANNEL_SUBSET, OUTPUT_DIR

PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "EEG"

N_JOBS = 8
DAYS = [1, 2, 3, 4, 5]
TASK_EVENT_CODES = [20, 21, 40, 41]
REST_DURATION_SEC = 300.0
REST_END_BUFFER_SEC = 5.0
RESAMPLE_SFREQ = 128.0
CONNECT_WINDOW_SEC = 2.0
CONNECT_STEP_SEC = 1.0

BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 40.0),
}


def raw_file_rows():
    rows = []
    raw_re = re.compile(r"^P(\d+)_D([\d_]+)\.bdf$")
    for path in sorted(RAW_DIR.glob("*.bdf")):
        match = raw_re.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected raw EEG filename: {path.name}")
        subject = int(match.group(1))
        day_token = match.group(2)
        day = int(day_token.split("_")[0])
        rows.append(
            {
                "subject": subject,
                "day": day,
                "day_token": day_token,
                "raw_path": path,
                "raw_file": path.name,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No BDF files found in {RAW_DIR}")
    return rows


def prepare_rest_raw(path):
    cap_re = re.compile(r"^[AB]\d+$")
    aux_types = {}
    for idx in range(1, 9):
        aux_types[f"EXG{idx}"] = "eog"
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
    montage = mne.channels.make_standard_montage("biosemi64")
    raw = mne.io.read_raw_bdf(
        path,
        preload=False,
        stim_channel="Status",
        verbose="ERROR",
    )
    events = mne.find_events(
        raw,
        stim_channel="Status",
        shortest_event=1,
        verbose="ERROR",
    )
    task_idx = np.where(np.isin(events[:, 2], TASK_EVENT_CODES))[0]
    if len(task_idx) == 0:
        raise ValueError(f"No task events found in {Path(path).name}")
    first_task_sec = float(events[task_idx[0], 0] / raw.info["sfreq"])
    rest_stop = first_task_sec - REST_END_BUFFER_SEC
    rest_start = rest_stop - REST_DURATION_SEC
    if rest_start < 0:
        raise ValueError(
            f"{Path(path).name} has only {rest_stop:.1f}s available before task"
        )

    cap_chs = [ch for ch in raw.ch_names if cap_re.match(ch)]
    if len(cap_chs) != 64:
        raise ValueError(f"{Path(path).name} has {len(cap_chs)} cap channels")
    raw.rename_channels(dict(zip(cap_chs, montage.ch_names)))
    channel_types = {}
    for channel_name, channel_type in aux_types.items():
        if channel_name in raw.ch_names:
            channel_types[channel_name] = channel_type
    raw.set_channel_types(channel_types, verbose="ERROR")
    raw.set_montage(montage, on_missing="ignore")
    raw.crop(tmin=rest_start, tmax=rest_stop).load_data(verbose="ERROR")
    raw.pick(CHANNEL_SUBSET)
    raw.set_eeg_reference("average", projection=False, verbose="ERROR")
    raw.resample(RESAMPLE_SFREQ, npad="auto", verbose="ERROR")
    return raw, rest_start, rest_stop, first_task_sec


def band_features(data, sfreq, pair_i, pair_j):
    power = np.log(np.mean(np.abs(data) ** 2, axis=1))
    win_n = int(round(CONNECT_WINDOW_SEC * sfreq))
    step_n = int(round(CONNECT_STEP_SEC * sfreq))
    starts = np.arange(0, data.shape[1] - win_n + 1, step_n)
    edge_sum = np.zeros(len(pair_i), dtype=float)
    edge_count = np.zeros(len(pair_i), dtype=float)
    for start in starts:
        stop = int(start + win_n)
        seg = data[:, start:stop]
        cross = seg @ np.conjugate(seg.T)
        cross = cross / float(seg.shape[1])
        ch_power = np.mean(np.abs(seg) ** 2, axis=1)
        denom = np.sqrt(np.outer(ch_power, ch_power))
        with np.errstate(divide="ignore", invalid="ignore"):
            coh = cross / denom
        vals = np.abs(np.imag(coh[pair_i, pair_j]))
        good = np.isfinite(vals)
        edge_sum[good] += vals[good]
        edge_count[good] += 1.0
    conn = np.full(len(pair_i), np.nan, dtype=float)
    good = edge_count > 0
    conn[good] = edge_sum[good] / edge_count[good]
    return power, conn, int(len(starts))


def process_rest_session(item):
    subject = int(item["subject"])
    day = int(item["day"])
    raw, rest_start, rest_stop, first_task_sec = prepare_rest_raw(item["raw_path"])
    pair_i, pair_j = np.triu_indices(len(CHANNEL_SUBSET), k=1)
    pair_labels = [
        f"{CHANNEL_SUBSET[i]}--{CHANNEL_SUBSET[j]}" for i, j in zip(pair_i, pair_j)
    ]
    feature_rows = []
    qc_rows = []
    for band, (fmin, fmax) in BANDS.items():
        raw_band = raw.copy().filter(
            l_freq=fmin,
            h_freq=fmax,
            method="fir",
            fir_design="firwin",
            phase="zero-double",
            verbose="ERROR",
        )
        raw_band.apply_hilbert(envelope=False, verbose="ERROR")
        data = raw_band.get_data()
        power, conn, n_windows = band_features(
            data,
            float(raw_band.info["sfreq"]),
            pair_i,
            pair_j,
        )
        for channel, value in zip(CHANNEL_SUBSET, power):
            feature_rows.append(
                {
                    "subject": subject,
                    "day": day,
                    "feature_kind": "spectral",
                    "band": band,
                    "feature": channel,
                    "value": float(value),
                }
            )
        for label, value in zip(pair_labels, conn):
            feature_rows.append(
                {
                    "subject": subject,
                    "day": day,
                    "feature_kind": "connectivity",
                    "band": band,
                    "feature": label,
                    "value": float(value),
                }
            )
        qc_rows.append(
            {
                "subject": subject,
                "day": day,
                "band": band,
                "n_windows": n_windows,
                "rest_start_sec": rest_start,
                "rest_stop_sec": rest_stop,
                "first_task_sec": first_task_sec,
            }
        )
    return pd.DataFrame(feature_rows), pd.DataFrame(qc_rows), None


def process_rest_session_safe(item):
    try:
        return process_rest_session(item)
    except Exception as exc:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
            {
                "subject": int(item["subject"]),
                "day": int(item["day"]),
                "raw_file": item["raw_file"],
                "stage": "rest_features",
                "reason": "error",
                "detail": str(exc),
            },
        )


def z_euclidean(vec_a, vec_b):
    a = np.asarray(vec_a, dtype=float)
    b = np.asarray(vec_b, dtype=float)
    good = np.isfinite(a) & np.isfinite(b)
    if int(np.sum(good)) < 2:
        return np.nan
    a = a[good]
    b = b[good]
    if np.std(a) <= np.finfo(float).eps or np.std(b) <= np.finfo(float).eps:
        return np.nan
    a = (a - float(np.mean(a))) / float(np.std(a))
    b = (b - float(np.mean(b))) / float(np.std(b))
    return float(np.sqrt(np.sum((a - b) ** 2)))


def model_distance(model, day_i, day_j, split_day=None):
    if model == "baseline":
        return 0.0
    if model == "gradual":
        return float(abs(day_i - day_j) / 4.0)
    if model == "discrete":
        if split_day is None:
            raise ValueError("discrete model requires split_day")
        gradual = float(abs(day_i - day_j) / 4.0)
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late == j_late:
            return 0.5 * gradual
        return 0.5 + 0.5 * gradual
    raise ValueError(f"Unknown model: {model}")


def model_specs():
    rows = [
        {"model_label": "Baseline", "model": "baseline", "split_day": np.nan},
        {"model_label": "Gradual", "model": "gradual", "split_day": np.nan},
    ]
    for split_day in [1, 2, 3, 4]:
        rows.append(
            {
                "model_label": f"Discrete D{split_day}",
                "model": "discrete",
                "split_day": float(split_day),
            }
        )
    return rows


def fit_bic(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    good = np.isfinite(y)
    for col_i in range(x.shape[1]):
        good &= np.isfinite(x[:, col_i])
    y = y[good]
    x = x[good]
    n_obs = int(len(y))
    if n_obs < 4:
        return {"bic": np.nan, "r2": np.nan, "n_obs": n_obs, "n_params": np.nan}
    keep_cols = [0]
    for col_i in range(1, x.shape[1]):
        col = x[:, col_i]
        if float(np.nanmax(col) - np.nanmin(col)) > np.finfo(float).eps:
            keep_cols.append(col_i)
    x = x[:, keep_cols]
    n_params = int(x.shape[1])
    beta, _resid, rank, _singular = np.linalg.lstsq(x, y, rcond=None)
    if int(rank) < n_params:
        return {"bic": np.nan, "r2": np.nan, "n_obs": n_obs, "n_params": n_params}
    pred = x @ beta
    resid = y - pred
    rss = max(float(np.sum(resid**2)), np.finfo(float).eps)
    tss = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = np.nan
    if tss > np.finfo(float).eps:
        r2 = 1.0 - rss / tss
    bic = float(n_obs * np.log(rss / float(n_obs)) + n_params * np.log(float(n_obs)))
    return {"bic": bic, "r2": r2, "n_obs": n_obs, "n_params": n_params}


def design_matrix(pair_rows, spec):
    intercept = np.ones(len(pair_rows), dtype=float)
    if spec["model"] == "baseline":
        return intercept.reshape(-1, 1)
    split_day = None
    if np.isfinite(spec["split_day"]):
        split_day = int(spec["split_day"])
    pred = []
    for row in pair_rows:
        pred.append(
            model_distance(
                spec["model"],
                int(row["day_i"]),
                int(row["day_j"]),
                split_day,
            )
        )
    return np.column_stack([intercept, np.asarray(pred, dtype=float)])


def feature_vectors(feature_df, subject, feature_kind, band_group):
    d = feature_df[
        (feature_df["subject"] == subject)
        & (feature_df["feature_kind"] == feature_kind)
    ].copy()
    if band_group != "all_bands":
        d = d[d["band"] == band_group].copy()
    vectors = {}
    features = sorted(d["band"].astype(str) + "::" + d["feature"].astype(str))
    features = sorted(set(features))
    for day in DAYS:
        g = d[d["day"] == day].copy()
        if g.empty:
            continue
        vals = {}
        for row in g.itertuples(index=False):
            vals[f"{row.band}::{row.feature}"] = float(row.value)
        vectors[day] = np.asarray([vals.get(feature, np.nan) for feature in features])
    return vectors


def score_feature_group(feature_df, subject, feature_kind, band_group):
    vectors = feature_vectors(feature_df, subject, feature_kind, band_group)
    pair_rows = []
    distances = []
    for day_i in DAYS:
        if day_i not in vectors:
            continue
        for day_j in DAYS:
            if day_j <= day_i or day_j not in vectors:
                continue
            distance = z_euclidean(vectors[day_i], vectors[day_j])
            pair_rows.append({"day_i": day_i, "day_j": day_j})
            distances.append(distance)
    score_rows = []
    for spec in model_specs():
        fit = fit_bic(distances, design_matrix(pair_rows, spec))
        score_rows.append(
            {
                "subject": int(subject),
                "feature_kind": feature_kind,
                "band_group": band_group,
                "model_label": spec["model_label"],
                "model": spec["model"],
                "split_day": spec["split_day"],
                "bic": fit["bic"],
                "r2": fit["r2"],
                "n_obs": fit["n_obs"],
                "n_params": fit["n_params"],
            }
        )
    distance_rows = []
    for row, distance in zip(pair_rows, distances):
        distance_rows.append(
            {
                "subject": int(subject),
                "feature_kind": feature_kind,
                "band_group": band_group,
                "day_i": int(row["day_i"]),
                "day_j": int(row["day_j"]),
                "distance": float(distance),
            }
        )
    return score_rows, distance_rows


def add_delta_bic(score_df):
    frames = []
    group_cols = ["subject", "feature_kind", "band_group"]
    for _key, g in score_df.groupby(group_cols, dropna=False):
        g = g.copy()
        finite = g["bic"].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        g["delta_bic_best"] = np.nan if len(finite) == 0 else g["bic"] - np.min(finite)
        baseline = g[g["model_label"] == "Baseline"]
        if baseline.empty or not np.isfinite(float(baseline["bic"].iloc[0])):
            g["delta_bic_baseline"] = np.nan
        else:
            g["delta_bic_baseline"] = g["bic"] - float(baseline["bic"].iloc[0])
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def summarize_scores(score_df):
    rows = []
    group_cols = ["feature_kind", "band_group", "model_label", "model", "split_day"]
    for key, g in score_df.groupby(group_cols, dropna=False):
        feature_kind, band_group, model_label, model, split_day = key
        delta_base = g["delta_bic_baseline"].to_numpy(float)
        delta_best = g["delta_bic_best"].to_numpy(float)
        r2_vals = g["r2"].to_numpy(float)
        delta_base = delta_base[np.isfinite(delta_base)]
        delta_best = delta_best[np.isfinite(delta_best)]
        r2_vals = r2_vals[np.isfinite(r2_vals)]
        rows.append(
            {
                "feature_kind": feature_kind,
                "band_group": band_group,
                "model_label": model_label,
                "model": model,
                "split_day": split_day,
                "delta_bic_baseline_mean": float(np.mean(delta_base))
                if len(delta_base)
                else np.nan,
                "delta_bic_baseline_sem": sem(delta_base),
                "delta_bic_best_mean": float(np.mean(delta_best))
                if len(delta_best)
                else np.nan,
                "r2_mean": float(np.mean(r2_vals)) if len(r2_vals) else np.nan,
                "r2_sem": sem(r2_vals),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["feature_kind", "band_group", "delta_bic_best_mean"]
    )


def summarize_distances(distance_df):
    rows = []
    group_cols = ["feature_kind", "band_group", "day_i", "day_j"]
    for key, g in distance_df.groupby(group_cols):
        feature_kind, band_group, day_i, day_j = key
        vals = g["distance"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        rows.append(
            {
                "feature_kind": feature_kind,
                "band_group": band_group,
                "day_i": int(day_i),
                "day_j": int(day_j),
                "distance_mean": float(np.mean(vals)) if len(vals) else np.nan,
                "distance_sem": sem(vals),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["feature_kind", "band_group", "day_i", "day_j"])


def run_rest_day_geometry(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_path = output_dir / "rest_day_features.csv"
    qc_path = output_dir / "rest_day_feature_qc.csv"
    error_path = output_dir / "rest_day_feature_errors.csv"
    distance_path = output_dir / "rest_day_geometry_subject_distances.csv"
    group_distance_path = output_dir / "rest_day_geometry_group_distances.csv"
    score_path = output_dir / "rest_day_geometry_model_subject.csv"
    summary_path = output_dir / "rest_day_geometry_model_summary.csv"
    progress_path = output_dir / "rest_day_geometry_progress.json"

    t0 = time.time()
    items = raw_file_rows()
    progress_path.write_text(
        json.dumps({"stage": "features", "done": 0, "total": len(items)}, indent=2)
    )
    print(f"[rest] extracting features from {len(items)} raw files", flush=True)
    results = Parallel(n_jobs=N_JOBS)(
        delayed(process_rest_session_safe)(item) for item in items
    )

    feature_frames = []
    qc_frames = []
    error_rows = []
    for idx, (features, qc, error) in enumerate(results, start=1):
        if not features.empty:
            feature_frames.append(features)
        if not qc.empty:
            qc_frames.append(qc)
        if error is not None:
            error_rows.append(error)
        progress_path.write_text(
            json.dumps(
                {
                    "stage": "features",
                    "done": idx,
                    "total": len(items),
                    "elapsed_sec": time.time() - t0,
                },
                indent=2,
            )
        )

    if not feature_frames:
        raise ValueError("No resting-state feature rows were produced")
    feature_df = pd.concat(feature_frames, ignore_index=True)
    qc_df = pd.concat(qc_frames, ignore_index=True) if qc_frames else pd.DataFrame()
    feature_df.to_csv(feature_path, index=False)
    qc_df.to_csv(qc_path, index=False)
    pd.DataFrame(error_rows).to_csv(error_path, index=False)

    progress_path.write_text(
        json.dumps(
            {
                "stage": "models",
                "done": 0,
                "total": int(feature_df["subject"].nunique()),
                "elapsed_sec": time.time() - t0,
            },
            indent=2,
        )
    )

    subjects = sorted(feature_df["subject"].unique())
    band_groups = ["all_bands"] + list(BANDS.keys())
    score_rows = []
    distance_rows = []
    for idx, subject in enumerate(subjects, start=1):
        for feature_kind in ["connectivity", "spectral"]:
            for band_group in band_groups:
                scores, distances = score_feature_group(
                    feature_df,
                    int(subject),
                    feature_kind,
                    band_group,
                )
                score_rows.extend(scores)
                distance_rows.extend(distances)
        progress_path.write_text(
            json.dumps(
                {
                    "stage": "models",
                    "done": idx,
                    "total": len(subjects),
                    "elapsed_sec": time.time() - t0,
                },
                indent=2,
            )
        )

    score_df = add_delta_bic(pd.DataFrame(score_rows))
    summary_df = summarize_scores(score_df)
    distance_df = pd.DataFrame(distance_rows)
    group_distance_df = summarize_distances(distance_df)
    distance_df.to_csv(distance_path, index=False)
    group_distance_df.to_csv(group_distance_path, index=False)
    score_df.to_csv(score_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    progress_path.write_text(
        json.dumps(
            {
                "stage": "complete",
                "done": len(subjects),
                "total": len(subjects),
                "elapsed_sec": time.time() - t0,
                "feature_rows": int(len(feature_df)),
                "score_rows": int(len(score_df)),
            },
            indent=2,
        )
    )
    print(f"[rest] wrote {feature_path}", flush=True)
    print(f"[rest] wrote {summary_path}", flush=True)
    return feature_path, summary_path


if __name__ == "__main__":
    run_rest_day_geometry()
