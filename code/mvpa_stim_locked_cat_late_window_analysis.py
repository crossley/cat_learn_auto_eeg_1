#!/usr/bin/env python3
"""Late-window stimulus-locked category MVPA using all window time points."""

from __future__ import annotations

import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from load_project_data import load_sessions
from mvpa_stim_locked_cat_tg_window_structure_analysis import MVPA_CAT_TG_WINDOWS
from util_mvpa import pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8
DAYS = [1, 2, 3, 4, 5]
WINDOW = "late"
WINDOW_START_SEC = MVPA_CAT_TG_WINDOWS[WINDOW][0]
WINDOW_END_SEC = MVPA_CAT_TG_WINDOWS[WINDOW][1]
N_BOOTSTRAP = 1000
RANDOM_STATE = 42
CLASSIFIERS = ["logreg", "linear_svm", "shrinkage_lda"]


def build_window_clf(classifier, random_state):
    if classifier == "logreg":
        estimator = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )
    elif classifier == "linear_svm":
        estimator = LinearSVC(
            C=1.0,
            class_weight="balanced",
            dual=True,
            max_iter=5000,
            random_state=random_state,
        )
    elif classifier == "shrinkage_lda":
        estimator = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        raise ValueError(f"Unknown late-window classifier: {classifier}")
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("estimator", estimator),
        ]
    )


def classifier_weight(clf, classifier):
    scaler = clf.named_steps["scaler"]
    estimator = clf.named_steps["estimator"]
    if classifier in ["logreg", "linear_svm"]:
        w_scaled = estimator.coef_.ravel().astype(float)
    elif classifier == "shrinkage_lda":
        w_scaled = estimator.coef_.ravel().astype(float)
    else:
        raise ValueError(f"Unknown late-window classifier: {classifier}")
    scale = np.asarray(scaler.scale_, dtype=float)
    scale[scale == 0] = 1.0
    return w_scaled / scale


def sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def finite_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(valid)) < 3:
        return np.nan
    x = x[valid]
    y = y[valid]
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x * y) / denom)


def compute_haufe_pattern_window(
    X_flat,
    y,
    n_channels,
    n_times,
    random_state,
    classifier,
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    fold_patterns = []
    for tr, _ in cv.split(X_flat, y):
        X_tr = X_flat[tr]
        y_tr = y[tr]
        clf = build_window_clf(classifier, random_state=random_state)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SklearnConvergenceWarning)
                clf.fit(X_tr, y_tr)
        except Exception:
            continue
        w_feature = classifier_weight(clf, classifier)
        if X_tr.shape[0] < 2:
            continue
        X_centered = X_tr - np.mean(X_tr, axis=0)
        projected = X_centered @ w_feature
        pattern_flat = X_centered.T @ projected / float(X_tr.shape[0] - 1)
        fold_patterns.append(pattern_flat.reshape(n_channels, n_times))
    if len(fold_patterns) == 0:
        raise RuntimeError("No valid Haufe folds")
    arr = np.stack(fold_patterns, axis=0)
    return np.nanmean(arr, axis=0)


def process_late_window_session(task):
    session_file = task["epo_file"]
    subject = int(task["subject"])
    day = int(task["day"])
    min_epochs = int(task["min_epochs"])
    random_state = int(task["random_state"])

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_events = []
        for event_name in ["Stim/A", "Stim/B"]:
            if event_name in epochs.event_id:
                stim_events.append(event_name)
        if len(stim_events) < 2:
            return {
                "ok": False,
                "qc": {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "event_select",
                    "reason": "missing_stim_labels",
                    "detail": ",".join(stim_events),
                },
            }
        stim_epochs = epochs[stim_events].copy()
        stim_epochs.load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        if len(stim_epochs.ch_names) == 0:
            raise RuntimeError("No EEG channels after interpolation.")
        stim_epochs.resample(128, npad="auto")
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "preprocess",
                "reason": "prep_error",
                "detail": str(exc),
            },
        }

    codes = stim_epochs.events[:, 2]
    y = np.full(len(codes), -1, dtype=int)
    y[codes == stim_epochs.event_id["Stim/A"]] = 0
    y[codes == stim_epochs.event_id["Stim/B"]] = 1
    keep = y >= 0
    y = y[keep]
    X = stim_epochs.get_data()[keep]
    times = stim_epochs.times.copy()

    n_a = int(np.sum(y == 0))
    n_b = int(np.sum(y == 1))
    n_trials = int(len(y))
    if n_trials < min_epochs:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "epoch_count",
                "reason": "insufficient_epochs",
                "detail": f"n_trials={n_trials} < min_epochs={min_epochs}",
            },
        }
    if min(n_a, n_b) < 5:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "class_balance",
                "reason": "insufficient_class_trials",
                "detail": f"n_a={n_a}, n_b={n_b}; need >=5 in each class",
            },
        }

    time_mask = (times >= WINDOW_START_SEC) & (times <= WINDOW_END_SEC)
    if int(np.sum(time_mask)) < 2:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "window_select",
                "reason": "insufficient_timepoints",
                "detail": f"window={WINDOW_START_SEC}-{WINDOW_END_SEC}",
            },
        }
    X_win = X[:, :, time_mask]
    win_times = times[time_mask]
    n_channels = int(X_win.shape[1])
    n_times = int(X_win.shape[2])
    X_flat = X_win.reshape(X_win.shape[0], n_channels * n_times)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    session_rows = []
    haufe_rows = []
    sensor_rows = []
    qc_rows = []
    channel_pos = []
    for ci, ch in enumerate(stim_epochs.ch_names):
        loc = stim_epochs.info["chs"][stim_epochs.info.ch_names.index(ch)]["loc"][:3]
        channel_pos.append(
            {
                "channel": ch,
                "x": float(loc[0]),
                "y": float(loc[1]),
                "z": float(loc[2]),
            }
        )
    for classifier in CLASSIFIERS:
        clf = build_window_clf(classifier, random_state=random_state)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SklearnConvergenceWarning)
                scores = cross_val_score(clf, X_flat, y, cv=cv, scoring="roc_auc")
            auc = float(np.mean(scores))
        except Exception as exc:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "decode",
                    "reason": "compute_error",
                    "detail": f"{classifier}: {exc}",
                }
            )
            continue

        try:
            patterns = compute_haufe_pattern_window(
                X_flat,
                y,
                n_channels,
                n_times,
                random_state,
                classifier,
            )
        except Exception as exc:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "haufe",
                    "reason": "compute_error",
                    "detail": f"{classifier}: {exc}",
                }
            )
            continue

        session_rows.append(
            {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "classifier": classifier,
                "window": WINDOW,
                "window_start_sec": float(WINDOW_START_SEC),
                "window_end_sec": float(WINDOW_END_SEC),
                "auc": auc,
                "n_trials": n_trials,
                "n_a": n_a,
                "n_b": n_b,
                "n_channels": n_channels,
                "n_timepoints": n_times,
            }
        )
        for ci, ch in enumerate(stim_epochs.ch_names):
            vals = patterns[ci, :]
            sensor_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "classifier": classifier,
                    "window": WINDOW,
                    "channel": ch,
                    "pattern_mean": float(np.nanmean(vals)),
                    "abs_pattern_mean": float(np.nanmean(np.abs(vals))),
                    "n_timepoints": n_times,
                }
            )
            for ti, t_sec in enumerate(win_times):
                val = float(patterns[ci, ti])
                haufe_rows.append(
                    {
                        "session_file": session_file,
                        "subject": subject,
                        "day": day,
                        "classifier": classifier,
                        "window": WINDOW,
                        "channel": ch,
                        "time_sec": float(t_sec),
                        "pattern": val,
                        "abs_pattern": float(np.abs(val)),
                        "n_trials": n_trials,
                        "n_a": n_a,
                        "n_b": n_b,
                    }
                )
    if len(session_rows) == 0:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "classifier_loop",
                "reason": "all_classifiers_failed",
                "detail": "No classifier produced valid AUC and Haufe patterns",
            },
        }
    return {
        "ok": True,
        "session_rows": session_rows,
        "haufe_rows": haufe_rows,
        "sensor_rows": sensor_rows,
        "channel_pos": channel_pos,
        "qc_rows": qc_rows,
    }


def make_day_means(session_df):
    return (
        session_df.groupby(["classifier", "day", "window"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sem=("auc", sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "window"])
    )


def make_sensor_day_mean(sensor_df):
    return (
        sensor_df.groupby(["classifier", "day", "window", "channel"], as_index=False)
        .agg(
            pattern_mean=("pattern_mean", "mean"),
            pattern_sem=("pattern_mean", sem),
            abs_pattern_mean=("abs_pattern_mean", "mean"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "window", "channel"])
    )


def subject_day_vector(sensor_df, subject, day, classifier, channels):
    g = sensor_df[
        (sensor_df["subject"] == int(subject))
        & (sensor_df["day"] == int(day))
        & (sensor_df["classifier"] == classifier)
    ]
    if g.empty:
        raise ValueError(
            f"Missing late-window sensor pattern: subject={subject}, "
            f"day={day}, classifier={classifier}"
        )
    s = g.set_index("channel")["pattern_mean"]
    vals = []
    missing = []
    for ch in channels:
        if ch not in s.index:
            missing.append(ch)
            vals.append(np.nan)
        else:
            vals.append(float(s.loc[ch]))
    if len(missing) > 0:
        raise ValueError(
            f"Missing channels for subject={subject}, day={day}: {missing}"
        )
    return np.asarray(vals, dtype=float)


def complete_subjects(sensor_df, classifier, channels):
    d_classifier = sensor_df[sensor_df["classifier"] == classifier]
    subjects = sorted(d_classifier["subject"].dropna().unique().astype(int))
    retained = []
    for subject in subjects:
        complete = True
        for day in DAYS:
            try:
                subject_day_vector(
                    sensor_df,
                    int(subject),
                    int(day),
                    classifier,
                    channels,
                )
            except ValueError:
                complete = False
        if complete:
            retained.append(int(subject))
    if len(retained) == 0:
        raise ValueError(
            f"No complete subjects for late-window Haufe similarity: {classifier}"
        )
    return retained


def make_symmetrised_similarity(sensor_df):
    channels = sorted(sensor_df["channel"].dropna().unique().tolist())
    if len(channels) < 3:
        raise ValueError("Need at least three channels for late-window similarity")
    rows = []
    for classifier in CLASSIFIERS:
        subjects = complete_subjects(sensor_df, classifier, channels)
        cache = {}
        for subject in subjects:
            day_cache = {}
            for day in DAYS:
                day_cache[int(day)] = subject_day_vector(
                    sensor_df,
                    subject,
                    day,
                    classifier,
                    channels,
                )
            cache[int(subject)] = day_cache
        for subject in subjects:
            for i, day_i in enumerate(DAYS):
                for j in range(i + 1, len(DAYS)):
                    day_j = DAYS[j]
                    sim = finite_corr(
                        cache[int(subject)][int(day_i)],
                        cache[int(subject)][int(day_j)],
                    )
                    if not np.isfinite(sim):
                        raise ValueError(
                            f"Non-finite late-window similarity: "
                            f"classifier={classifier}, subject={subject}, "
                            f"D{day_i}-D{day_j}"
                        )
                    rows.append(
                        {
                            "row_type": "subject",
                            "classifier": classifier,
                            "subject": int(subject),
                            "day_low": int(day_i),
                            "day_high": int(day_j),
                            "similarity": float(sim),
                            "similarity_mean": np.nan,
                            "similarity_sem": np.nan,
                            "n_subjects": np.nan,
                            "n_channels": int(len(channels)),
                        }
                    )
    subject_rows = pd.DataFrame(rows)
    group_rows = (
        subject_rows.groupby(["classifier", "day_low", "day_high"], as_index=False)
        .agg(
            similarity_mean=("similarity", "mean"),
            similarity_sem=("similarity", sem),
            n_subjects=("subject", "nunique"),
            n_channels=("n_channels", "max"),
        )
        .sort_values(["day_low", "day_high"])
    )
    out_rows = []
    for _, row in group_rows.iterrows():
        out_rows.append(
            {
                "row_type": "group",
                "classifier": row["classifier"],
                "subject": np.nan,
                "day_low": int(row["day_low"]),
                "day_high": int(row["day_high"]),
                "similarity": np.nan,
                "similarity_mean": float(row["similarity_mean"]),
                "similarity_sem": float(row["similarity_sem"]),
                "n_subjects": int(row["n_subjects"]),
                "n_channels": int(row["n_channels"]),
            }
        )
    group_df = pd.DataFrame(out_rows)
    return pd.concat([subject_rows, group_df], ignore_index=True)


def group_similarity_matrix(sym_df, classifier):
    g = sym_df[
        (sym_df["row_type"] == "group")
        & (sym_df["classifier"] == classifier)
    ]
    if g.empty:
        raise ValueError(f"Missing group late-window similarity rows: {classifier}")
    mat = np.full((len(DAYS), len(DAYS)), np.nan, dtype=float)
    for _, row in g.iterrows():
        i = DAYS.index(int(row["day_low"]))
        j = DAYS.index(int(row["day_high"]))
        mat[i, j] = float(row["similarity_mean"])
        mat[j, i] = float(row["similarity_mean"])
    return mat


def distance_matrix_from_similarity(sim_mat):
    finite = sim_mat[np.isfinite(sim_mat)]
    if len(finite) == 0:
        raise ValueError("Cannot build distance matrix from empty similarity matrix")
    max_sim = float(np.max(finite))
    dist = np.full_like(sim_mat, np.nan, dtype=float)
    for i in range(sim_mat.shape[0]):
        for j in range(sim_mat.shape[1]):
            if i == j:
                dist[i, j] = 0.0
            elif np.isfinite(sim_mat[i, j]):
                dist[i, j] = max_sim - float(sim_mat[i, j])
    if not np.all(np.isfinite(dist)):
        raise ValueError("Distance matrix contains missing values")
    return dist


def cluster_members(node_id, z):
    node_id = int(node_id)
    n_days = len(DAYS)
    if node_id < n_days:
        return [DAYS[node_id]]
    merge_idx = node_id - n_days
    members = []
    for child_col in [0, 1]:
        child_members = cluster_members(int(z[merge_idx, child_col]), z)
        for day in child_members:
            members.append(day)
    return sorted(members)


def cluster_description_from_distance(dist):
    condensed = squareform(dist, checks=False)
    z = linkage(condensed, method="average")
    order_idx = leaves_list(z)
    order_days = []
    for idx in order_idx:
        order_days.append(str(DAYS[int(idx)]))
    first_members = cluster_members(int(z[0, 0]), z)
    for day in cluster_members(int(z[0, 1]), z):
        first_members.append(day)
    first_members = sorted(first_members)
    first_labels = []
    for day in first_members:
        first_labels.append(f"D{day}")
    first_pair = "-".join(first_labels)
    final_left = cluster_members(int(z[-1, 0]), z)
    final_right = cluster_members(int(z[-1, 1]), z)
    last_singleton_day = np.nan
    if len(final_left) == 1 and len(final_right) > 1:
        last_singleton_day = int(final_left[0])
    elif len(final_right) == 1 and len(final_left) > 1:
        last_singleton_day = int(final_right[0])
    return z, ",".join(order_days), first_pair, last_singleton_day


def make_clusters(sym_df):
    rows = []
    for classifier in CLASSIFIERS:
        sim_mat = group_similarity_matrix(sym_df, classifier)
        dist = distance_matrix_from_similarity(sim_mat)
        z, day_order, first_pair, last_singleton_day = cluster_description_from_distance(dist)
        for merge_idx in range(z.shape[0]):
            rows.append(
                {
                    "row_type": "linkage",
                    "classifier": classifier,
                    "merge_index": int(merge_idx),
                    "child_1": float(z[merge_idx, 0]),
                    "child_2": float(z[merge_idx, 1]),
                    "distance": float(z[merge_idx, 2]),
                    "n_members": int(z[merge_idx, 3]),
                    "day": np.nan,
                    "order_position": np.nan,
                    "day_order": day_order,
                    "first_pair": first_pair,
                    "last_singleton_day": last_singleton_day,
                }
            )
        order_parts = day_order.split(",")
        for pos, day_text in enumerate(order_parts):
            rows.append(
                {
                    "row_type": "order",
                    "classifier": classifier,
                    "merge_index": np.nan,
                    "child_1": np.nan,
                    "child_2": np.nan,
                    "distance": np.nan,
                    "n_members": np.nan,
                    "day": int(day_text),
                    "order_position": int(pos),
                    "day_order": day_order,
                    "first_pair": first_pair,
                    "last_singleton_day": last_singleton_day,
                }
            )
    return pd.DataFrame(rows)


def run_mvpa_stim_locked_cat_late_window(
    output_dir: Path | str = OUTPUT_DIR,
    min_epochs: int = 20,
    random_state: int = RANDOM_STATE,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )

    session_csv = output_dir / "mvpa_stim_locked_cat_late_window_session_auc.csv"
    day_mean_csv = output_dir / "mvpa_stim_locked_cat_late_window_day_auc.csv"
    haufe_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_channel_time.csv"
    sensor_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_sensor.csv"
    sensor_day_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_sensor_day_mean.csv"
    pos_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_channel_positions.csv"
    similarity_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_similarity.csv"
    clusters_csv = output_dir / "mvpa_stim_locked_cat_late_window_haufe_clusters.csv"
    qc_csv = output_dir / "mvpa_stim_locked_cat_late_window_qc_log.csv"
    progress_json = output_dir / "mvpa_stim_locked_cat_late_window_progress.json"

    qc_rows = []
    session_rows = []
    haufe_rows = []
    sensor_rows = []
    channel_pos = {}
    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    t0 = time.time()

    def write_progress(stage, done, total):
        payload = {
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

    sessions = load_sessions()
    tasks = []
    for item in sessions:
        tasks.append(
            {
                "subject": int(item["subject"]),
                "day": int(item["day"]),
                "epo_file": item["epo_file"],
                "epo_path": str(item["epo_path"]),
                "min_epochs": int(min_epochs),
                "random_state": int(random_state),
            }
        )
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    write_progress("running", 0, len(tasks))

    def handle_result(result, done):
        if not result["ok"]:
            qc_rows.append(result["qc"])
        else:
            session_rows.extend(result["session_rows"])
            haufe_rows.extend(result["haufe_rows"])
            sensor_rows.extend(result["sensor_rows"])
            for qc_row in result.get("qc_rows", []):
                qc_rows.append(qc_row)
            for pos_row in result["channel_pos"]:
                ch = pos_row["channel"]
                if ch not in channel_pos:
                    channel_pos[ch] = np.array(
                        [pos_row["x"], pos_row["y"], pos_row["z"]],
                        dtype=float,
                    )
        write_progress("running", done, len(tasks))
        if (done % 5) == 0:
            elapsed = time.time() - t0
            print(
                f"[MVPA late-window] complete {done}/{len(tasks)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    print(
        f"[MVPA late-window] Starting on {len(tasks)} sessions "
        f"(n_workers={n_workers}, window={WINDOW_START_SEC}-{WINDOW_END_SEC}s)...",
        flush=True,
    )

    def iter_jobs():
        for task in tasks:
            yield delayed(process_late_window_session)(task)

    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            handle_result(process_late_window_session(task), done)
    elif threadpool_limits is None:
        result_iter = Parallel(
            n_jobs=n_workers,
            backend="loky",
            verbose=0,
            return_as="generator_unordered",
        )(iter_jobs())
        for done, result in enumerate(result_iter, start=1):
            handle_result(result, done)
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(
                n_jobs=n_workers,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(iter_jobs())
            for done, result in enumerate(result_iter, start=1):
                handle_result(result, done)

    session_df = pd.DataFrame(session_rows)
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)
    if session_df.empty:
        session_df.to_csv(session_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        raise RuntimeError("Late-window MVPA produced no valid session rows")

    day_mean_df = make_day_means(session_df)
    haufe_df = pd.DataFrame(haufe_rows)
    sensor_df = pd.DataFrame(sensor_rows)
    sensor_day_df = make_sensor_day_mean(sensor_df)
    pos_rows = []
    for ch, xyz in sorted(channel_pos.items()):
        pos_rows.append(
            {
                "channel": ch,
                "x": float(xyz[0]),
                "y": float(xyz[1]),
                "z": float(xyz[2]),
            }
        )
    pos_df = pd.DataFrame(pos_rows)
    similarity_df = make_symmetrised_similarity(sensor_df)
    clusters_df = make_clusters(similarity_df)

    session_df.to_csv(session_csv, index=False)
    day_mean_df.to_csv(day_mean_csv, index=False)
    haufe_df.to_csv(haufe_csv, index=False)
    sensor_df.to_csv(sensor_csv, index=False)
    sensor_day_df.to_csv(sensor_day_csv, index=False)
    pos_df.to_csv(pos_csv, index=False)
    similarity_df.to_csv(similarity_csv, index=False)
    clusters_df.to_csv(clusters_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    write_progress("completed", len(tasks), len(tasks))

    print(f"[MVPA late-window] Wrote {session_csv}")
    print(f"[MVPA late-window] Wrote {day_mean_csv}")
    print(f"[MVPA late-window] Wrote {haufe_csv}")
    print(f"[MVPA late-window] Wrote {sensor_csv}")
    print(f"[MVPA late-window] Wrote {sensor_day_csv}")
    print(f"[MVPA late-window] Wrote {pos_csv}")
    print(f"[MVPA late-window] Wrote {similarity_csv}")
    print(f"[MVPA late-window] Wrote {clusters_csv}")
    print(f"[MVPA late-window] Wrote {qc_csv}")

    return {
        "session_df": session_df,
        "day_mean_df": day_mean_df,
        "haufe_df": haufe_df,
        "sensor_df": sensor_df,
        "sensor_day_df": sensor_day_df,
        "pos_df": pos_df,
        "similarity_df": similarity_df,
        "clusters_df": clusters_df,
        "qc_df": qc_df,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_late_window()
