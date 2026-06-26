#!/usr/bin/env python3
"""GRU early-exit MVPA transfer analysis."""

from __future__ import annotations

import json
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from figure_style import DAYS
from sequence_feature_interface import load_feature_sequence, load_sequence_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"

TMIN = 0.0
TMAX = 0.8
RESAMPLE_HZ = 128.0
END_TIMES_SEC = np.array([0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80])

RANDOM_STATE = 42
HIDDEN_SIZE = 16
DROPOUT = 0.0
MAX_EPOCHS = 35
PATIENCE = 6
BATCH_SIZE = 64
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
MIN_CLASS_TRIALS = 12
PROGRESS_EVERY_SUBJECTS = 1
N_WORKERS = 4

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


class GRUClassifier(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True,
            dropout=DROPOUT,
        )
        self.readout = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        _seq, h = self.gru(x)
        return self.readout(h[-1]).squeeze(-1)


def write_progress(path, payload):
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def class_counts(y):
    y = np.asarray(y, dtype=int)
    return int(np.sum(y == 0)), int(np.sum(y == 1))


def standardize_from_train(x_train, x_apply):
    mean = np.nanmean(x_train, axis=(0, 1), keepdims=True)
    sd = np.nanstd(x_train, axis=(0, 1), keepdims=True)
    sd[sd < np.finfo(float).eps] = 1.0
    return (x_apply - mean) / sd


def make_loader(x, y, shuffle):
    x_t = torch.tensor(x, dtype=torch.float32)
    y_t = torch.tensor(y.astype(float), dtype=torch.float32)
    return DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        drop_last=False,
    )


def train_gru(x_fit, y_fit, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )
    train_idx, val_idx = next(splitter.split(np.zeros(len(y_fit)), y_fit))
    x_train_raw = x_fit[train_idx]
    y_train = y_fit[train_idx]
    x_val_raw = x_fit[val_idx]
    y_val = y_fit[val_idx]

    x_train = standardize_from_train(x_train_raw, x_train_raw)
    x_val = standardize_from_train(x_train_raw, x_val_raw)

    model = GRUClassifier(n_features=x_fit.shape[2])
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = make_loader(x_train, y_train, shuffle=True)
    val_loader = make_loader(x_val, y_val, shuffle=False)

    best_state = None
    best_loss = np.inf
    stale_epochs = 0
    for _epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in train_loader:
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                val_losses.append(float(loss_fn(model(xb), yb).item()))
        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, x_train_raw


def predict_scores(model, x_train_raw, x_apply):
    x_apply = standardize_from_train(x_train_raw, x_apply)
    loader = make_loader(x_apply, np.zeros(x_apply.shape[0], dtype=int), shuffle=False)
    vals = []
    model.eval()
    with torch.no_grad():
        for xb, _yb in loader:
            vals.append(model(xb).detach().cpu().numpy())
    return np.concatenate(vals)


def day_model_value(model, day_i, day_j, split_day=None):
    if model == "baseline":
        return 0.0
    if day_i == day_j:
        return 0.0
    if model == "continuous":
        return float(0.65 * min(day_i, day_j) / float(max(DAYS)))
    if model == "discrete":
        if split_day is None:
            raise ValueError("discrete model requires split_day")
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late != j_late:
            return 0.0
        return float(0.65 * min(day_i, day_j) / float(max(DAYS)))
    raise ValueError(f"Unknown model: {model}")


def day_model_specs():
    rows = [
        {"model_label": "Baseline", "model": "baseline", "split_day": np.nan},
        {"model_label": "Continuous Restructuring", "model": "continuous", "split_day": np.nan},
    ]
    for split_day in range(1, max(DAYS)):
        rows.append(
            {
                "model_label": f"Discrete Restructuring D{split_day}",
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
    return {"bic": bic, "r2": float(r2), "n_obs": n_obs, "n_params": n_params}


def score_day_models(subject, end_time_sec, transfer_rows):
    rows = []
    pair_rows = []
    y_vals = []
    for row in transfer_rows:
        pair_rows.append((int(row["train_day"]), int(row["test_day"])))
        y_vals.append(float(row["auc"]))
    y_vals = np.asarray(y_vals, dtype=float)

    for spec in day_model_specs():
        cols = [np.ones(len(pair_rows), dtype=float)]
        if spec["model"] != "baseline":
            split_day = None
            if np.isfinite(spec["split_day"]):
                split_day = int(spec["split_day"])
            cols.append(
                np.asarray(
                    [
                        day_model_value(
                            spec["model"],
                            train_day,
                            test_day,
                            split_day=split_day,
                        )
                        for train_day, test_day in pair_rows
                    ],
                    dtype=float,
                )
            )
        fit = fit_bic(y_vals, np.column_stack(cols))
        rows.append(
            {
                "subject": int(subject),
                "end_time_sec": float(end_time_sec),
                "model_label": spec["model_label"],
                "model": spec["model"],
                "split_day": spec["split_day"],
                "bic": fit["bic"],
                "r2": fit["r2"],
                "n_obs": fit["n_obs"],
                "n_params": fit["n_params"],
            }
        )
    return rows


def add_delta_bic(score_df):
    frames = []
    for _key, g in score_df.groupby(["subject", "end_time_sec"], dropna=False):
        g = g.copy()
        finite = g["bic"].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        g["delta_bic_best"] = np.nan
        if len(finite):
            g["delta_bic_best"] = g["bic"] - float(np.min(finite))
        base = g[g["model_label"] == "Baseline"]
        g["delta_bic_baseline"] = np.nan
        if not base.empty and np.isfinite(float(base["bic"].iloc[0])):
            g["delta_bic_baseline"] = g["bic"] - float(base["bic"].iloc[0])
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def summarize_scores(score_df):
    rows = []
    group_cols = ["end_time_sec", "model_label", "model", "split_day"]
    for key, g in score_df.groupby(group_cols, dropna=False):
        key_vals = dict(zip(group_cols, key))
        delta = g["delta_bic_baseline"].to_numpy(float)
        delta = delta[np.isfinite(delta)]
        r2 = g["r2"].to_numpy(float)
        r2 = r2[np.isfinite(r2)]
        delta_sem = np.nan
        r2_sem = np.nan
        if len(delta) > 1:
            delta_sem = float(np.std(delta, ddof=1) / np.sqrt(len(delta)))
        if len(r2) > 1:
            r2_sem = float(np.std(r2, ddof=1) / np.sqrt(len(r2)))
        rows.append(
            {
                **key_vals,
                "delta_bic_baseline_mean": float(np.mean(delta)) if len(delta) else np.nan,
                "delta_bic_baseline_sem": delta_sem,
                "r2_mean": float(np.mean(r2)) if len(r2) else np.nan,
                "r2_sem": r2_sem,
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["end_time_sec", "model_label"])


def split_fit_holdout(y, seed):
    n0, n1 = class_counts(y)
    if min(n0, n1) < MIN_CLASS_TRIALS:
        raise ValueError(f"too few trials per class: A={n0}, B={n1}")
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )
    return next(splitter.split(np.zeros(len(y)), y))


def run_subject(subject, day_data):
    transfer_rows = []
    score_rows = []
    qc_rows = []
    for end_time_sec in END_TIMES_SEC:
        day_models = {}
        for train_day in DAYS:
            if train_day not in day_data:
                continue
            ds = day_data[train_day]
            time_mask = ds.time <= end_time_sec + 1e-12
            if not np.any(time_mask):
                raise ValueError(f"no samples through {end_time_sec} s")
            x_all = ds.X[:, time_mask, :]
            y_all = ds.y.astype(int)
            seed = RANDOM_STATE + int(subject) * 1000 + int(train_day) * 100 + int(end_time_sec * 100)
            try:
                fit_idx, holdout_idx = split_fit_holdout(y_all, seed)
                model, x_train_raw = train_gru(x_all[fit_idx], y_all[fit_idx], seed)
            except Exception as exc:
                qc_rows.append(
                    {
                        "subject": int(subject),
                        "day": int(train_day),
                        "end_time_sec": float(end_time_sec),
                        "stage": "train",
                        "reason": "compute_error",
                        "detail": str(exc),
                    }
                )
                continue
            day_models[train_day] = {
                "model": model,
                "x_train_raw": x_train_raw,
                "holdout_idx": holdout_idx,
                "fit_idx": fit_idx,
            }

        subject_time_rows = []
        for train_day, payload in day_models.items():
            for test_day in DAYS:
                if test_day not in day_data:
                    continue
                ds_test = day_data[test_day]
                time_mask = ds_test.time <= end_time_sec + 1e-12
                x_test_all = ds_test.X[:, time_mask, :]
                y_test_all = ds_test.y.astype(int)
                if train_day == test_day:
                    test_idx = payload["holdout_idx"]
                    x_test = x_test_all[test_idx]
                    y_test = y_test_all[test_idx]
                else:
                    x_test = x_test_all
                    y_test = y_test_all
                n0, n1 = class_counts(y_test)
                if min(n0, n1) < 2:
                    continue
                scores = predict_scores(
                    payload["model"],
                    payload["x_train_raw"],
                    x_test,
                )
                auc = float(roc_auc_score(y_test, scores))
                row = {
                    "subject": int(subject),
                    "end_time_sec": float(end_time_sec),
                    "train_day": int(train_day),
                    "test_day": int(test_day),
                    "auc": auc,
                    "n_train_fit": int(len(payload["fit_idx"])),
                    "n_train_holdout": int(len(payload["holdout_idx"])),
                    "n_test": int(len(y_test)),
                    "n_test_A": n0,
                    "n_test_B": n1,
                }
                transfer_rows.append(row)
                subject_time_rows.append(row)
        if len(subject_time_rows) >= 10:
            score_rows.extend(score_day_models(subject, end_time_sec, subject_time_rows))
    return transfer_rows, score_rows, qc_rows


def run_subject_job(subject, subject_items):
    day_data = {}
    for day, item in sorted(subject_items.items()):
        if day not in DAYS:
            continue
        day_data[day] = load_feature_sequence(
            item,
            "voltage",
            use_cache=False,
            resample_hz=RESAMPLE_HZ,
            tmin=TMIN,
            tmax=TMAX,
        )
    transfer_rows, score_rows, qc_rows = run_subject(subject, day_data)
    return int(subject), transfer_rows, score_rows, qc_rows


def run_rnn_mvpa_gru_analysis(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transfer_csv = output_dir / "rnn_mvpa_gru_transfer_subject.csv"
    transfer_summary_csv = output_dir / "rnn_mvpa_gru_transfer_day_mean.csv"
    score_csv = output_dir / "rnn_mvpa_gru_model_timecourse_subject.csv"
    score_summary_csv = output_dir / "rnn_mvpa_gru_model_timecourse_summary.csv"
    qc_csv = output_dir / "rnn_mvpa_gru_qc_log.csv"
    progress_json = output_dir / "rnn_mvpa_gru_progress.json"

    for path in [transfer_csv, transfer_summary_csv, score_csv, score_summary_csv, qc_csv]:
        if path.exists():
            path.unlink()

    t0 = time.time()
    sessions = load_sequence_sessions(load_epochs=False)
    by_subject = {}
    for item in sessions:
        by_subject.setdefault(int(item["subject"]), {})[int(item["day"])] = item

    all_transfer_rows = []
    all_score_rows = []
    all_qc_rows = []
    subjects = sorted(by_subject)
    write_progress(
        progress_json,
        {
            "status": "started",
            "subjects_total": len(subjects),
            "subjects_done": 0,
            "last_subject": None,
            "elapsed_min": 0.0,
        },
    )
    print(
        f"[RNN MVPA] starting {len(subjects)} subjects, "
        f"{len(END_TIMES_SEC)} end times, {N_WORKERS} workers",
        flush=True,
    )

    def record_subject_result(subject_i, subject, transfer_rows, score_rows, qc_rows):
        all_transfer_rows.extend(transfer_rows)
        all_score_rows.extend(score_rows)
        all_qc_rows.extend(qc_rows)

        transfer_df = pd.DataFrame(all_transfer_rows)
        score_df = pd.DataFrame(all_score_rows)
        qc_df = pd.DataFrame(all_qc_rows)
        if not transfer_df.empty:
            transfer_df.to_csv(transfer_csv, index=False)
            transfer_summary = (
                transfer_df.groupby(["end_time_sec", "train_day", "test_day"], as_index=False)
                .agg(
                    auc_mean=("auc", "mean"),
                    auc_sem=(
                        "auc",
                        lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
                        if len(x) > 1
                        else np.nan,
                    ),
                    n_subjects=("subject", "nunique"),
                )
                .sort_values(["end_time_sec", "train_day", "test_day"])
            )
            transfer_summary.to_csv(transfer_summary_csv, index=False)
        if not score_df.empty:
            score_df = add_delta_bic(score_df)
            score_df.to_csv(score_csv, index=False)
            summarize_scores(score_df).to_csv(score_summary_csv, index=False)
        if not qc_df.empty:
            qc_df.to_csv(qc_csv, index=False)

        elapsed_min = (time.time() - t0) / 60.0
        write_progress(
            progress_json,
            {
                "status": "running",
                "subjects_total": len(subjects),
                "subjects_done": subject_i,
                "last_subject": int(subject),
                "elapsed_min": round(elapsed_min, 2),
            },
        )
        if subject_i % PROGRESS_EVERY_SUBJECTS == 0:
            print(
                f"[RNN MVPA] subject {subject_i}/{len(subjects)} done "
                f"(P{subject}, {elapsed_min:.1f} min)",
                flush=True,
            )

    if N_WORKERS == 1:
        for subject_i, subject in enumerate(subjects, start=1):
            subject, transfer_rows, score_rows, qc_rows = run_subject_job(
                subject,
                by_subject[subject],
            )
            record_subject_result(subject_i, subject, transfer_rows, score_rows, qc_rows)
    else:
        with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
            futures = {
                pool.submit(run_subject_job, subject, by_subject[subject]): subject
                for subject in subjects
            }
            for subject_i, fut in enumerate(as_completed(futures), start=1):
                subject, transfer_rows, score_rows, qc_rows = fut.result()
                record_subject_result(subject_i, subject, transfer_rows, score_rows, qc_rows)

    elapsed_min = (time.time() - t0) / 60.0
    write_progress(
        progress_json,
        {
            "status": "complete",
            "subjects_total": len(subjects),
            "subjects_done": len(subjects),
            "last_subject": int(subjects[-1]) if subjects else None,
            "elapsed_min": round(elapsed_min, 2),
        },
    )
    print(f"[RNN MVPA] complete in {elapsed_min:.1f} min", flush=True)
    return {
        "transfer_subject": transfer_csv,
        "transfer_summary": transfer_summary_csv,
        "model_subject": score_csv,
        "model_summary": score_summary_csv,
        "qc": qc_csv,
        "progress": progress_json,
    }


if __name__ == "__main__":
    run_rnn_mvpa_gru_analysis()
