#!/usr/bin/env python3
"""Feedback-locked cross-day temporal generalization for category labels."""

from __future__ import annotations

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
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from mne.decoding import GeneralizingEstimator
from sklearn.model_selection import StratifiedKFold

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from load_project_data import align_behaviour_to_epochs, load_sessions
from mvpa_stim_locked_cat_tg_analysis import write_cross_day_outputs
from util_mvpa import (
    balanced_day_subset,
    build_clf,
    pick_eeg_interpolate_bads,
    session_cache_key,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8


def compute_haufe_patterns_from_xy(X, y, random_state: int):
    n_times = X.shape[2]
    n_ch = X.shape[1]
    patterns = np.full((n_ch, n_times), np.nan, dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for ti in range(n_times):
        Xt = X[:, :, ti]
        fold_patterns = []
        for tr, _ in cv.split(Xt, y):
            Xt_tr = Xt[tr]
            y_tr = y[tr]
            clf = build_clf(random_state=random_state)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SklearnConvergenceWarning)
                    clf.fit(Xt_tr, y_tr)
            except Exception:
                continue
            scaler = clf.named_steps["scaler"]
            logreg = clf.named_steps["logreg"]
            w_scaled = logreg.coef_.ravel().astype(float)
            scale = np.asarray(scaler.scale_, dtype=float)
            scale[scale == 0] = 1.0
            w_sensor = w_scaled / scale
            if Xt_tr.shape[0] < 2:
                continue
            cov_x = np.cov(Xt_tr, rowvar=False, ddof=1)
            fold_patterns.append(cov_x @ w_sensor)
        if fold_patterns:
            patterns[:, ti] = np.nanmean(np.vstack(fold_patterns), axis=0)
    return patterns


def _extract_channel_positions(epochs):
    rows = []
    for ch in epochs.ch_names:
        idx = epochs.info.ch_names.index(ch)
        loc = epochs.info["chs"][idx]["loc"][:3]
        rows.append({"channel": ch, "x": float(loc[0]), "y": float(loc[1]), "z": float(loc[2])})
    return rows


def prepare_feedback_data(epochs, behaviour):
    fb_events = [x for x in ["FB/Cor", "FB/Inc"] if x in epochs.event_id]
    if len(fb_events) < 2:
        raise ValueError(f"missing_feedback_labels:{','.join(fb_events)}")
    fb_epochs = epochs[fb_events].copy()
    fb_epochs.load_data()
    pick_eeg_interpolate_bads(fb_epochs)
    fb_epochs.resample(128, npad="auto")
    fb_epochs, beh_aligned = align_behaviour_to_epochs(
        behaviour, fb_epochs, event_names=("FB/Cor", "FB/Inc")
    )
    y = (beh_aligned["cat"].astype(str) == "B").astype(int).to_numpy()
    X = fb_epochs.get_data()
    t = fb_epochs.times.copy()
    ch_names = np.array(fb_epochs.ch_names, dtype=str)
    return X, y, t, ch_names, beh_aligned


def prepare_feedback_session_cache(
    session_item: dict,
    cache_dir: Path,
    cache_prefix: str = "mvpa_feedback_locked_cat_tg_cache_interp_bads",
):
    cache_dir = Path(cache_dir)
    session_file = session_item["epo_file"]
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    epochs = session_item["epochs"]
    behaviour = session_item["beh"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_prefix}_{session_cache_key(session_item)}.npz"

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if "ch_names" in z.files:
                t = z["t"]
                y = z["y"]
                ch_names = z["ch_names"]
                return {
                    "ok": True,
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "cache_path": str(cache_path),
                    "n_trials": int(len(y)),
                    "n_a": int(np.sum(y == 0)),
                    "n_b": int(np.sum(y == 1)),
                    "n_times": int(len(t)),
                    "ch_names": ch_names.tolist(),
                }

    try:
        X, y, t, ch_names, beh_aligned = prepare_feedback_data(epochs, behaviour)
    except Exception as exc:
        msg = str(exc)
        reason = msg.split(":")[0] if ":" in msg else "prep_error"
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "prepare",
                "reason": reason,
                "detail": msg,
            },
        }

    np.savez_compressed(cache_path, X=X, y=y, t=t, ch_names=ch_names)
    return {
        "ok": True,
        "session_file": session_file,
        "subject": subject,
        "day": day,
        "cache_path": str(cache_path),
        "n_trials": int(len(y)),
        "n_a": int(np.sum(y == 0)),
        "n_b": int(np.sum(y == 1)),
        "n_times": int(len(t)),
        "ch_names": ch_names.tolist(),
    }


def process_feedback_day_pair(pair_item: dict, random_state: int):
    subject = int(pair_item["subject"])
    d_train = int(pair_item["train_day"])
    d_test = int(pair_item["test_day"])
    train_session_file = pair_item["train_session_file"]
    test_session_file = pair_item["test_session_file"]
    train_cache_path = pair_item["train_cache_path"]
    test_cache_path = pair_item["test_cache_path"]
    pair_seed = int(pair_item["pair_seed"])

    with np.load(train_cache_path, allow_pickle=False) as z:
        X_train_all = z["X"]
        y_train_all = z["y"]
        t_train = z["t"]
        ch_train = z["ch_names"] if "ch_names" in z.files else np.array([], dtype=str)
    with np.load(test_cache_path, allow_pickle=False) as z:
        X_test_all = z["X"]
        y_test_all = z["y"]
        ch_test = z["ch_names"] if "ch_names" in z.files else np.array([], dtype=str)

    if len(ch_train) == 0 or len(ch_test) == 0 or ch_train.tolist() != ch_test.tolist():
        return {
            "ok": False,
            "qc": {
                "session_file": f"{train_session_file}->{test_session_file}",
                "subject": subject,
                "day": d_train,
                "stage": "cross_day_channels",
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
                "day": d_train,
                "stage": "cross_day_balance",
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
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            ge.fit(X_train, y_train)
            mat_transfer = ge.score(X_test, y_test)
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": f"{train_session_file}->{test_session_file}",
                "subject": subject,
                "day": d_train,
                "stage": "cross_day_tg",
                "reason": "compute_error",
                "detail": str(exc),
            },
        }

    return {
        "ok": True,
        "row": {
            "subject": subject,
            "train_day": d_train,
            "test_day": d_test,
            "n_per_class": int(n_per_class),
            "n_train_trials_used": int(len(y_train)),
            "n_test_trials_used": int(len(y_test)),
            "diag_mean_auc": float(np.nanmean(np.diag(mat_transfer))),
        },
        "t": t_train,
        "mat": mat_transfer,
    }


def run_mvpa_feedback_locked_cat_tg(
    output_dir: Path | str = OUTPUT_DIR,
    min_epochs: int = 20,
    random_state: int = 42,
    progress_every: int = 5,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    cross_subject_csv = output_dir / "mvpa_feedback_locked_cat_tg_subject_level.csv"
    cross_day_mean_csv = output_dir / "mvpa_feedback_locked_cat_tg_day_mean.csv"
    cross_matrix_dir = output_dir
    cross_matrix_day_mean_csv = output_dir / "mvpa_feedback_locked_cat_tg_timegen_day_mean.csv"
    haufe_session_csv = output_dir / "mvpa_feedback_locked_cat_tg_haufe_session_channel_time.csv"
    haufe_day_mean_csv = output_dir / "mvpa_feedback_locked_cat_tg_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_feedback_locked_cat_tg_haufe_channel_positions.csv"
    qc_csv = output_dir / "mvpa_feedback_locked_cat_tg_qc_log.csv"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    wrote_qc = False
    t0 = time.time()

    def append_csv(df: pd.DataFrame, path: Path, wrote_flag: bool):
        if df.empty:
            return wrote_flag
        df.to_csv(path, mode="a", header=not wrote_flag, index=False)
        return True

    session_items = load_sessions(load_epochs=True)
    cache_results = [
        prepare_feedback_session_cache(
            item,
            cache_dir=output_dir,
            cache_prefix="mvpa_feedback_locked_cat_tg_cache_interp_bads",
        )
        for item in session_items
    ]
    prepared_map: dict[tuple, dict] = {}
    for result in cache_results:
        if not result["ok"]:
            qc_rows.append(result["qc"])
        else:
            item = next(
                x
                for x in session_items
                if int(x["subject"]) == int(result["subject"]) and int(x["day"]) == int(result["day"])
            )
            prepared_map[(result["subject"], result["day"])] = {
                **result,
                "epochs": item["epochs"],
            }
    if qc_rows:
        wrote_qc = append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    haufe_rows = []
    channel_pos_rows = {}
    for key in sorted(prepared_map):
        item = prepared_map[key]
        try:
            with np.load(item["cache_path"], allow_pickle=False) as z:
                X = z["X"]
                y = z["y"]
                t = z["t"]
                ch_names = z["ch_names"] if "ch_names" in z.files else np.array([], dtype=str)
            if len(ch_names) == 0:
                continue
            patterns = compute_haufe_patterns_from_xy(X, y, random_state=random_state)
            for ci, ch in enumerate(ch_names):
                loc = item["epochs"].info["chs"][item["epochs"].info.ch_names.index(ch)]["loc"][:3]
                channel_pos_rows.setdefault(
                    ch,
                    {"channel": ch, "x": float(loc[0]), "y": float(loc[1]), "z": float(loc[2])},
                )
                for ti, tsec in enumerate(t):
                    val = float(patterns[ci, ti])
                    haufe_rows.append(
                        {
                            "subject": int(item["subject"]),
                            "day": int(item["day"]),
                            "session_file": item["session_file"],
                            "channel": ch,
                            "time_sec": float(tsec),
                            "pattern": val,
                            "abs_pattern": float(np.abs(val)),
                        }
                    )
        except Exception as exc:
            qc_rows.append(
                {
                    "session_file": item["session_file"],
                    "subject": int(item["subject"]),
                    "day": int(item["day"]),
                    "stage": "haufe",
                    "reason": "compute_error",
                    "detail": str(exc),
                }
            )
    if haufe_rows:
        haufe_session_df = pd.DataFrame(haufe_rows)
        haufe_day_mean_df = (
            haufe_session_df.groupby(["day", "channel", "time_sec"], as_index=False)
            .agg(
                pattern_mean=("pattern", "mean"),
                pattern_sem=(
                    "pattern",
                    lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
                ),
                abs_pattern_mean=("abs_pattern", "mean"),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["day", "channel", "time_sec"])
        )
        haufe_pos_df = pd.DataFrame(
            [channel_pos_rows[ch] for ch in sorted(channel_pos_rows)]
        )
        haufe_session_df.to_csv(haufe_session_csv, index=False)
        haufe_day_mean_df.to_csv(haufe_day_mean_csv, index=False)
        haufe_pos_df.to_csv(haufe_channel_pos_csv, index=False)
    else:
        pd.DataFrame().to_csv(haufe_session_csv, index=False)
        pd.DataFrame().to_csv(haufe_day_mean_csv, index=False)
        pd.DataFrame().to_csv(haufe_channel_pos_csv, index=False)

    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    rng_master = np.random.default_rng(random_state)
    pair_items = []
    cross_matrix_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted({k[0] for k in prepared_map})
    for subject in subjects:
        subject_days = sorted([k[1] for k in prepared_map if k[0] == subject])
        if len(subject_days) < 2:
            continue
        for d_train in subject_days:
            for d_test in subject_days:
                train_item = prepared_map[(subject, d_train)]
                test_item = prepared_map[(subject, d_test)]
                pair_items.append(
                    {
                        "subject": subject,
                        "train_day": d_train,
                        "test_day": d_test,
                        "train_cache_path": train_item["cache_path"],
                        "test_cache_path": test_item["cache_path"],
                        "train_session_file": train_item["session_file"],
                        "test_session_file": test_item["session_file"],
                        "pair_seed": int(rng_master.integers(0, 2**31 - 1)),
                    }
                )

    cross_rows = []
    cross_matrix_accum: dict = {}
    cross_time_template = None
    cross_done = 0
    wrote_cross = False

    def handle_cross_result(result):
        nonlocal wrote_cross, wrote_qc, cross_done, qc_rows, cross_time_template
        if result["ok"]:
            row = result["row"]
            mat = np.asarray(result["mat"], dtype=float)
            t_vec = np.asarray(result["t"], dtype=float)
            cross_rows.append(row)
            wrote_cross = append_csv(pd.DataFrame([row]), cross_subject_csv, wrote_cross)
            matrix_path = cross_matrix_dir / (
                f"mvpa_feedback_locked_cat_tg_matrix_sub_{int(row['subject']):03d}_trainD{int(row['train_day'])}"
                f"_testD{int(row['test_day'])}.npz"
            )
            np.savez_compressed(matrix_path, auc=mat, time_sec=t_vec)
            if cross_time_template is None:
                cross_time_template = t_vec
            key = (int(row["train_day"]), int(row["test_day"]))
            if key not in cross_matrix_accum:
                cross_matrix_accum[key] = {
                    "sum": np.zeros_like(mat, dtype=float),
                    "count": np.zeros_like(mat, dtype=float),
                }
            valid = np.isfinite(mat)
            cross_matrix_accum[key]["sum"][valid] += mat[valid]
            cross_matrix_accum[key]["count"][valid] += 1.0
        else:
            qc_rows.append(result["qc"])
            if len(qc_rows) >= max(progress_every, 1):
                wrote_qc = append_csv(
                    pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc
                )
                qc_rows = []
        cross_done += 1
        if (cross_done % max(progress_every * 2, 1)) == 0:
            elapsed = time.time() - t0
            print(
                f"[TG feedback] processed {cross_done}/{len(pair_items)} pairs "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    print(
        f"[TG feedback] Starting cross-day transfer on {len(subjects)} subjects, "
        f"{len(pair_items)} pairs (n_workers={n_workers})...",
        flush=True,
    )
    if n_workers == 1:
        if threadpool_limits is None:
            for item in pair_items:
                handle_cross_result(process_feedback_day_pair(pair_item=item, random_state=random_state))
        else:
            with threadpool_limits(limits=1):
                for item in pair_items:
                    handle_cross_result(
                        process_feedback_day_pair(pair_item=item, random_state=random_state)
                    )
    elif pair_items:
        if threadpool_limits is None:
            result_iter = Parallel(
                n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
            )(
                delayed(process_feedback_day_pair)(pair_item=item, random_state=random_state)
                for item in pair_items
            )
            for result in result_iter:
                handle_cross_result(result)
        else:
            with threadpool_limits(limits=1):
                result_iter = Parallel(
                    n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
                )(
                    delayed(process_feedback_day_pair)(pair_item=item, random_state=random_state)
                    for item in pair_items
                )
                for result in result_iter:
                    handle_cross_result(result)

    if qc_rows:
        wrote_qc = append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    cross_subject_df, cross_day_mean_df = write_cross_day_outputs(
        cross_rows=cross_rows,
        cross_matrix_accum=cross_matrix_accum,
        cross_time_template=cross_time_template,
        cross_matrix_dir=cross_matrix_dir,
        cross_subject_csv=cross_subject_csv,
        cross_day_mean_csv=cross_day_mean_csv,
        cross_matrix_day_mean_csv=cross_matrix_day_mean_csv,
    )
    qc_df = pd.read_csv(qc_csv) if qc_csv.exists() else pd.DataFrame(columns=qc_columns)

    return {
        "cross_subject_df": cross_subject_df,
        "cross_day_mean_df": cross_day_mean_df,
        "qc_df": qc_df,
        "cross_subject_csv": cross_subject_csv,
        "cross_day_mean_csv": cross_day_mean_csv,
        "cross_matrix_day_mean_csv": cross_matrix_day_mean_csv,
        "qc_csv": qc_csv,
        "haufe_session_csv": haufe_session_csv,
        "haufe_day_mean_csv": haufe_day_mean_csv,
        "haufe_channel_pos_csv": haufe_channel_pos_csv,
    }


if __name__ == "__main__":
    run_mvpa_feedback_locked_cat_tg()
