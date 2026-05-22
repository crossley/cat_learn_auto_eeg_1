#!/usr/bin/env python3
"""Stimulus-to-feedback and feedback-to-stimulus cross-epoch TG for category labels."""

from __future__ import annotations

import os
import gc
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
from mne.decoding import GeneralizingEstimator
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from load_project_data import load_sessions
from mvpa_feedback_locked_cat_tg_analysis import prepare_feedback_session_cache
from util_mvpa import balanced_day_subset, build_clf, prepare_session_cache, session_cache_key

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8

TRAIN_TEST_DIRECTIONS = {
    "stim_to_feedback": ("stim", "feedback"),
    "feedback_to_stim": ("feedback", "stim"),
}


def _get_cache_row(result: dict):
    return {
        "session_file": result["session_file"],
        "subject": result["subject"],
        "day": result["day"],
        "cache_path": result["cache_path"],
        "n_trials": result["n_trials"],
        "n_a": result["n_a"],
        "n_b": result["n_b"],
        "n_times": result["n_times"],
    }


def prepare_epoch_caches(session_item: dict, output_dir: Path):
    stim_result = prepare_session_cache(
        session_item,
        cache_dir=output_dir,
        cache_prefix="mvpa_stim_feedback_cat_tg_stim_cache_interp_bads",
    )
    fb_result = prepare_feedback_session_cache(
        session_item,
        cache_dir=output_dir,
        cache_prefix="mvpa_stim_feedback_cat_tg_feedback_cache_interp_bads",
    )
    return stim_result, fb_result


def process_cross_epoch_pair(pair_item: dict, random_state: int):
    subject = int(pair_item["subject"])
    train_day = int(pair_item["train_day"])
    test_day = int(pair_item["test_day"])
    direction = str(pair_item["direction"])
    train_kind = str(pair_item["train_kind"])
    test_kind = str(pair_item["test_kind"])
    train_session_file = pair_item["train_session_file"]
    test_session_file = pair_item["test_session_file"]
    train_cache_path = pair_item["train_cache_path"]
    test_cache_path = pair_item["test_cache_path"]
    pair_seed = int(pair_item["pair_seed"])

    X_train_all = y_train_all = t_train = ch_train = None
    X_test_all = y_test_all = t_test = ch_test = None
    X_train = y_train = X_test = y_test = mat_transfer = None
    out = None

    try:
        with np.load(train_cache_path, allow_pickle=False) as z:
            X_train_all = z["X"]
            y_train_all = z["y"]
            t_train = z["t"]
            ch_train = z["ch_names"] if "ch_names" in z.files else np.array([], dtype=str)
        with np.load(test_cache_path, allow_pickle=False) as z:
            X_test_all = z["X"]
            y_test_all = z["y"]
            t_test = z["t"]
            ch_test = z["ch_names"] if "ch_names" in z.files else np.array([], dtype=str)

        if len(ch_train) == 0 or len(ch_test) == 0 or ch_train.tolist() != ch_test.tolist():
            return {
                "ok": False,
                "qc": {
                    "session_file": f"{train_session_file}->{test_session_file}",
                    "subject": subject,
                    "day": train_day,
                    "stage": "cross_epoch_channels",
                    "reason": "channel_mismatch",
                    "detail": f"train_n={len(ch_train)}, test_n={len(ch_test)}",
                },
            }

        n_per_class = int(
            min(
                np.sum(y_train_all == 0),
                np.sum(y_train_all == 1),
                np.sum(y_test_all == 0),
                np.sum(y_test_all == 1),
            )
        )
        if n_per_class < 5:
            return {
                "ok": False,
                "qc": {
                    "session_file": f"{train_session_file}->{test_session_file}",
                    "subject": subject,
                    "day": train_day,
                    "stage": "cross_epoch_balance",
                    "reason": "insufficient_balanced_trials",
                    "detail": f"n_per_class={n_per_class}",
                },
            }

        rng_pair = np.random.default_rng(pair_seed)
        X_train, y_train = balanced_day_subset(
            X_train_all, y_train_all, n_per_class=n_per_class, rng=rng_pair
        )
        X_test, y_test = balanced_day_subset(
            X_test_all, y_test_all, n_per_class=n_per_class, rng=rng_pair
        )

        clf = build_clf(random_state=random_state)
        ge = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=1, verbose=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            ge.fit(X_train, y_train)
            mat_transfer = ge.score(X_test, y_test)
        out = {
            "ok": True,
            "row": {
                "subject": subject,
                "train_day": train_day,
                "test_day": test_day,
                "direction": direction,
                "train_kind": train_kind,
                "test_kind": test_kind,
                "n_per_class": int(n_per_class),
                "n_train_trials_used": int(len(y_train)),
                "n_test_trials_used": int(len(y_test)),
                "diag_mean_auc": float(np.nanmean(np.diag(mat_transfer))),
            },
            "train_time": np.asarray(t_train, dtype=float),
            "test_time": np.asarray(t_test, dtype=float),
            "mat": np.asarray(mat_transfer, dtype=float),
        }
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": f"{train_session_file}->{test_session_file}",
                "subject": subject,
                "day": train_day,
                "stage": "cross_epoch_tg",
                "reason": "compute_error",
                "detail": str(exc),
            },
        }
    finally:
        X_train_all = y_train_all = t_train = ch_train = None
        X_test_all = y_test_all = t_test = ch_test = None
        X_train = y_train = X_test = y_test = mat_transfer = None
        gc.collect()
    return out


def _append_csv(df: pd.DataFrame, path: Path, wrote_flag: bool):
    if df.empty:
        return wrote_flag
    df.to_csv(path, mode="a", header=not wrote_flag, index=False)
    return True


def run_mvpa_stim_feedback_cat_tg(
    output_dir: Path | str = OUTPUT_DIR,
    random_state: int = 42,
    progress_every: int = 5,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    subject_csv = output_dir / "mvpa_stim_feedback_cat_tg_subject_level.csv"
    day_mean_csv = output_dir / "mvpa_stim_feedback_cat_tg_day_mean.csv"
    matrix_csv = output_dir / "mvpa_stim_feedback_cat_tg_timegen_day_mean.csv"
    day_pair_csv = output_dir / "mvpa_stim_feedback_cat_tg_day_pair_mean.csv"
    matrix_day_pair_csv = output_dir / "mvpa_stim_feedback_cat_tg_timegen_day_pair_mean.csv"
    qc_csv = output_dir / "mvpa_stim_feedback_cat_tg_qc_log.csv"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    wrote_subject = False
    wrote_qc = False
    t0 = time.time()

    sessions = load_sessions(load_epochs=True)
    prepared_map: dict[tuple[int, int, str], dict] = {}
    for item in sessions:
        stim_result, fb_result = prepare_epoch_caches(item, output_dir=output_dir)
        for kind, result in [("stim", stim_result), ("feedback", fb_result)]:
            if not result["ok"]:
                qc_rows.append(result["qc"])
                continue
            prepared_map[(result["subject"], result["day"], kind)] = result
    if qc_rows:
        wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []
    del sessions
    gc.collect()

    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    row_frames = []
    matrix_accum: dict[str, dict[str, np.ndarray]] = {}
    matrix_times: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    day_pair_accum: dict[tuple[str, int, int], dict[str, np.ndarray]] = {}
    day_pair_times: dict[tuple[str, int, int], tuple[np.ndarray, np.ndarray]] = {}
    total_pairs = 0
    cross_done = 0
    subject_set = set()
    for k in prepared_map:
        subject_set.add(k[0])
    subjects = sorted(subject_set)

    def handle_result(result: dict):
        nonlocal wrote_qc, qc_rows, cross_done
        if result["ok"]:
            row_frames.append(pd.DataFrame([result["row"]]))
            direction = result["row"]["direction"]
            train_day = int(result["row"]["train_day"])
            test_day = int(result["row"]["test_day"])
            train_time = result["train_time"]
            test_time = result["test_time"]
            mat = result["mat"]
            if direction not in matrix_accum:
                matrix_accum[direction] = {
                    "sum": np.zeros_like(mat, dtype=float),
                    "count": np.zeros_like(mat, dtype=float),
                }
                matrix_times[direction] = (train_time, test_time)
            valid = np.isfinite(mat)
            matrix_accum[direction]["sum"][valid] += mat[valid]
            matrix_accum[direction]["count"][valid] += 1.0
            key = (direction, train_day, test_day)
            if key not in day_pair_accum:
                day_pair_accum[key] = {
                    "sum": np.zeros_like(mat, dtype=float),
                    "count": np.zeros_like(mat, dtype=float),
                }
                day_pair_times[key] = (train_time, test_time)
            day_pair_accum[key]["sum"][valid] += mat[valid]
            day_pair_accum[key]["count"][valid] += 1.0
        else:
            qc_rows.append(result["qc"])
            if len(qc_rows) >= max(progress_every, 1):
                wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
                qc_rows = []
        cross_done += 1
        if (cross_done % max(progress_every * 2, 1)) == 0:
            elapsed = time.time() - t0
            print(
                f"[TG cross-epoch] processed {cross_done}/{total_pairs} pairs "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    def iter_cross_epoch_pair_jobs(items):
        for item in items:
            yield delayed(process_cross_epoch_pair)(item, random_state=random_state)

    for direction, (train_kind, test_kind) in TRAIN_TEST_DIRECTIONS.items():
        pair_items = []
        for subject in subjects:
            day_set = set()
            for k in prepared_map:
                if k[0] == subject:
                    day_set.add(k[1])
            days = sorted(day_set)
            for train_day in days:
                for test_day in days:
                    if (
                        (subject, train_day, train_kind) not in prepared_map
                        or (subject, test_day, test_kind) not in prepared_map
                    ):
                        continue
                    train_item = prepared_map[(subject, train_day, train_kind)]
                    test_item = prepared_map[(subject, test_day, test_kind)]
                    pair_items.append(
                        {
                            "subject": subject,
                            "train_day": train_day,
                            "test_day": test_day,
                            "direction": direction,
                            "train_kind": train_kind,
                            "test_kind": test_kind,
                            "train_cache_path": train_item["cache_path"],
                            "test_cache_path": test_item["cache_path"],
                            "train_session_file": train_item["session_file"],
                            "test_session_file": test_item["session_file"],
                            "pair_seed": int(
                                np.random.default_rng(
                                    random_state
                                    + subject * 1000
                                    + train_day * 100
                                    + test_day
                                ).integers(0, 2**31 - 1)
                            ),
                        }
                    )
        total_pairs += len(pair_items)
        print(
            f"[TG cross-epoch] Processing {direction} on {len(pair_items)} pairs "
            f"(n_workers={n_workers})...",
            flush=True,
        )
        if not pair_items:
            continue
        if n_workers == 1:
            if threadpool_limits is None:
                for item in pair_items:
                    handle_result(process_cross_epoch_pair(item, random_state=random_state))
            else:
                with threadpool_limits(limits=1):
                    for item in pair_items:
                        handle_result(process_cross_epoch_pair(item, random_state=random_state))
        else:
            if threadpool_limits is None:
                result_iter = Parallel(
                    n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
                )(iter_cross_epoch_pair_jobs(pair_items))
                for result in result_iter:
                    handle_result(result)
            else:
                with threadpool_limits(limits=1):
                    result_iter = Parallel(
                        n_jobs=n_workers,
                        backend="loky",
                        verbose=0,
                        return_as="generator_unordered",
                    )(iter_cross_epoch_pair_jobs(pair_items))
                    for result in result_iter:
                        handle_result(result)
        del pair_items
        gc.collect()

    if qc_rows:
        wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    subject_df = pd.concat(row_frames, ignore_index=True) if row_frames else pd.DataFrame()
    if not subject_df.empty:
        subject_df.to_csv(subject_csv, index=False)
        day_mean_df = (
            subject_df.groupby(["direction"], as_index=False)
            .agg(
                auc_mean=("diag_mean_auc", "mean"),
                auc_sem=(
                    "diag_mean_auc",
                    lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
                ),
                n_subjects=("subject", "nunique"),
            )
            .sort_values("direction")
        )
    else:
        day_mean_df = pd.DataFrame(columns=["direction", "auc_mean", "auc_sem", "n_subjects"])
    day_mean_df.to_csv(day_mean_csv, index=False)

    if not subject_df.empty:
        day_pair_df = (
            subject_df.groupby(["direction", "train_day", "test_day"], as_index=False)
            .agg(
                auc_mean=("diag_mean_auc", "mean"),
                auc_sem=(
                    "diag_mean_auc",
                    lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
                ),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["direction", "train_day", "test_day"])
        )
    else:
        day_pair_df = pd.DataFrame(
            columns=["direction", "train_day", "test_day", "auc_mean", "auc_sem", "n_subjects"]
        )
    day_pair_df.to_csv(day_pair_csv, index=False)

    matrix_rows = []
    for direction, acc in sorted(matrix_accum.items()):
        train_time, test_time = matrix_times[direction]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_mat = acc["sum"] / acc["count"]
        for i, tt in enumerate(train_time):
            for j, te in enumerate(test_time):
                val = mean_mat[i, j]
                if np.isfinite(val):
                    matrix_rows.append(
                        {
                            "direction": direction,
                            "train_time_sec": float(tt),
                            "test_time_sec": float(te),
                            "auc_mean": float(val),
                            "n_subjects": int(acc["count"][i, j]),
                        }
                    )
    matrix_df = pd.DataFrame(matrix_rows)
    matrix_df.to_csv(matrix_csv, index=False)

    matrix_day_pair_rows = []
    for (direction, train_day, test_day), acc in sorted(day_pair_accum.items()):
        train_time, test_time = day_pair_times[(direction, train_day, test_day)]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_mat = acc["sum"] / acc["count"]
        for i, tt in enumerate(train_time):
            for j, te in enumerate(test_time):
                val = mean_mat[i, j]
                if np.isfinite(val):
                    matrix_day_pair_rows.append(
                        {
                            "direction": direction,
                            "train_day": int(train_day),
                            "test_day": int(test_day),
                            "train_time_sec": float(tt),
                            "test_time_sec": float(te),
                            "auc_mean": float(val),
                            "n_subjects": int(acc["count"][i, j]),
                        }
                    )
    matrix_day_pair_df = pd.DataFrame(matrix_day_pair_rows)
    matrix_day_pair_df.to_csv(matrix_day_pair_csv, index=False)
    qc_df = pd.read_csv(qc_csv) if qc_csv.exists() else pd.DataFrame(columns=qc_columns)

    elapsed = time.time() - t0
    print(
        f"[TG cross-epoch] Done: {len(subject_df)} subject-direction rows, "
        f"{len(matrix_df)} matrix rows, {total_pairs} pairs, elapsed {elapsed/60:.1f} min.",
        flush=True,
    )

    return {
        "subject_df": subject_df,
        "day_mean_df": day_mean_df,
        "day_pair_df": day_pair_df,
        "matrix_df": matrix_df,
        "matrix_day_pair_df": matrix_day_pair_df,
        "qc_df": qc_df,
        "subject_csv": subject_csv,
        "day_mean_csv": day_mean_csv,
        "day_pair_csv": day_pair_csv,
        "matrix_csv": matrix_csv,
        "matrix_day_pair_csv": matrix_day_pair_csv,
        "qc_csv": qc_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_feedback_cat_tg()
