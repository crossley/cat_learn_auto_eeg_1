#!/usr/bin/env python3
"""Sequence classifiers for A-vs-B decoding over shared feature sequences."""

from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd
from types import SimpleNamespace
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

from sequence_feature_interface import (
    FIGURES_DIR,
    OUTPUT_DIR,
    PROJECT_DIR,
    SequenceDataset,
    load_feature_sequence,
    load_sequence_sessions,
)
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_AVAILABLE = True
    TORCH_VERSION = getattr(torch, "__version__", "")
    TORCH_IMPORT_ERROR = ""
except Exception as exc:  # pragma: no cover - depends on local environment
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    TORCH_AVAILABLE = False
    TORCH_VERSION = ""
    TORCH_IMPORT_ERROR = str(exc)

try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover - optional dependency
    threadpool_limits = None


RANDOM_STATE = 42
N_JOBS = 8
DEFAULT_FEATURE_KINDS = ["voltage", "time_frequency", "imcoh", "mvpa_decision"]
QC_COLUMNS = [
    "session_file",
    "subject",
    "day",
    "feature_kind",
    "model",
    "stage",
    "reason",
    "detail",
]


def sem(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def stable_feature_kind(feature_kind: str) -> str:
    if feature_kind == "connectivity":
        return "imcoh"
    return str(feature_kind)


def n_class_min(y) -> int:
    y = np.asarray(y, dtype=int)
    return int(min(np.sum(y == 0), np.sum(y == 1)))


def clean_sequence(X):
    X = np.asarray(X, dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


def maybe_time_shuffle(X, rng):
    X_use = np.asarray(X).copy()
    for i in range(X_use.shape[0]):
        X_use[i] = X_use[i, rng.permutation(X_use.shape[1]), :]
    return X_use


def truncate_sequence(X, prefix_fraction: float):
    X = np.asarray(X)
    n_keep = max(1, int(np.ceil(X.shape[1] * float(prefix_fraction))))
    return X[:, :n_keep, :]


def bag_features(X):
    X = clean_sequence(X)
    return np.concatenate(
        [
            np.nanmean(X, axis=1),
            np.nanstd(X, axis=1),
            X[:, -1, :],
        ],
        axis=1,
    )


def build_bag_clf(random_state: int):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def classifier_scores(clf, X):
    if hasattr(clf, "decision_function"):
        return np.asarray(clf.decision_function(X), dtype=float)
    if hasattr(clf, "predict_proba"):
        return np.asarray(clf.predict_proba(X)[:, 1], dtype=float)
    raise ValueError("classifier has neither decision_function nor predict_proba")


def parallel_collect_rnn(func, items, n_workers):
    n_workers = max(1, int(n_workers))
    if n_workers == 1:
        return [func(item) for item in items]
    work = (delayed(func)(item) for item in items)
    if threadpool_limits is None:
        return Parallel(n_jobs=n_workers, backend="threading", verbose=0)(work)
    with threadpool_limits(limits=1):
        return Parallel(n_jobs=n_workers, backend="threading", verbose=0)(work)


if nn is not None:

    class SmallSequenceNet(nn.Module):
        def __init__(self, input_size: int, hidden_size: int, cell: str, dropout: float):
            super().__init__()
            rnn_cls = nn.GRU if cell == "gru" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                dropout=0.0,
            )
            self.drop = nn.Dropout(float(dropout))
            self.head = nn.Linear(hidden_size, 1)

        def forward(self, x):
            out, _state = self.rnn(x)
            last = out[:, -1, :]
            return self.head(self.drop(last)).squeeze(-1)

else:
    SmallSequenceNet = None


def standardize_sequence_train_test(X_train, X_test):
    X_train = clean_sequence(X_train)
    X_test = clean_sequence(X_test)
    n_features = X_train.shape[2]
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, n_features))
    train = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    test = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    return train.astype(np.float32), test.astype(np.float32)


def fit_predict_rnn(
    X_train,
    y_train,
    X_test,
    cell: str,
    hidden_size: int,
    epochs: int,
    batch_size: int,
    torch_num_threads: int,
    learning_rate: float,
    dropout: float,
    random_state: int,
):
    if torch is None:
        raise RuntimeError(f"torch_unavailable: {TORCH_IMPORT_ERROR}")
    torch.set_num_threads(max(1, int(torch_num_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    X_train, X_test = standardize_sequence_train_test(X_train, X_test)
    y_train = np.asarray(y_train, dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(random_state))
    if torch.cuda.is_available():  # pragma: no cover - hardware dependent
        torch.cuda.manual_seed_all(int(random_state))
    model = SmallSequenceNet(
        input_size=int(X_train.shape[2]),
        hidden_size=int(hidden_size),
        cell=cell,
        dropout=float(dropout),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    loss_fn = nn.BCEWithLogitsLoss()
    ds = TensorDataset(
        torch.as_tensor(X_train, dtype=torch.float32),
        torch.as_tensor(y_train, dtype=torch.float32),
    )
    generator = torch.Generator()
    generator.manual_seed(int(random_state))
    loader = DataLoader(
        ds,
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )
    model.train()
    for _epoch in range(int(epochs)):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
    model.eval()
    with torch.no_grad():
        logits = model(torch.as_tensor(X_test, dtype=torch.float32).to(device))
    return logits.detach().cpu().numpy().astype(float)


def fit_predict_model(
    X_train,
    y_train,
    X_test,
    model: str,
    random_state: int,
    args,
):
    if model == "bag_logreg":
        clf = build_bag_clf(random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            clf.fit(bag_features(X_train), y_train)
        return classifier_scores(clf, bag_features(X_test))
    if model in {"gru", "lstm"}:
        return fit_predict_rnn(
            X_train,
            y_train,
            X_test,
            cell=model,
            hidden_size=args.hidden_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            torch_num_threads=args.torch_num_threads,
            learning_rate=args.learning_rate,
            dropout=args.dropout,
            random_state=random_state,
        )
    raise ValueError(f"unknown model: {model}")


def auc_or_nan(y_true, scores):
    try:
        if n_class_min(y_true) < 1:
            return np.nan
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return np.nan


def make_score_row(
    dataset: SequenceDataset,
    model: str,
    control: str,
    evaluation: str,
    prefix_fraction: float,
    auc: float,
    fit_status: str,
    train_day=None,
    test_day=None,
    train_n_trials=None,
    test_n_trials=None,
):
    return {
        "subject": int(dataset.subject),
        "feature_kind": stable_feature_kind(dataset.feature_kind),
        "model": model,
        "control": control,
        "evaluation": evaluation,
        "train_day": int(train_day if train_day is not None else dataset.session),
        "test_day": int(test_day if test_day is not None else dataset.session),
        "day_distance": int(abs((train_day if train_day is not None else dataset.session) - (test_day if test_day is not None else dataset.session))),
        "prefix_fraction": float(prefix_fraction),
        "time_start_sec": float(np.nanmin(dataset.time)),
        "time_end_sec": float(np.nanmax(dataset.time)),
        "n_timepoints": int(dataset.X.shape[1]),
        "n_features": int(dataset.X.shape[2]),
        "auc": float(auc) if np.isfinite(auc) else np.nan,
        "fit_status": fit_status,
        "train_n_trials": int(train_n_trials if train_n_trials is not None else len(dataset.y)),
        "test_n_trials": int(test_n_trials if test_n_trials is not None else len(dataset.y)),
    }


def within_session_scores(dataset: SequenceDataset, models, prefix_fractions, args, rng):
    rows = []
    qc_rows = []
    if len(dataset.y) < args.min_trials or n_class_min(dataset.y) < args.min_class_trials:
        qc_rows.append(
            {
                "session_file": dataset.session_file,
                "subject": int(dataset.subject),
                "day": int(dataset.session),
                "feature_kind": stable_feature_kind(dataset.feature_kind),
                "model": "all",
                "stage": "within_session",
                "reason": "insufficient_trials",
                "detail": f"n_trials={len(dataset.y)}, min_class={n_class_min(dataset.y)}",
            }
        )
        return rows, qc_rows
    n_splits = min(args.cv_splits, n_class_min(dataset.y))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.random_state)
    for model in models:
        for control in ["intact", "time_shuffled"]:
            fracs = prefix_fractions if control == "intact" else [1.0]
            for prefix_fraction in fracs:
                scores = np.full(len(dataset.y), np.nan, dtype=float)
                for fold_i, (train_idx, test_idx) in enumerate(cv.split(dataset.X, dataset.y)):
                    X_train = truncate_sequence(dataset.X[train_idx], prefix_fraction)
                    X_test = truncate_sequence(dataset.X[test_idx], prefix_fraction)
                    if control == "time_shuffled":
                        X_train = maybe_time_shuffle(X_train, rng)
                        X_test = maybe_time_shuffle(X_test, rng)
                    try:
                        pred = fit_predict_model(
                            X_train,
                            dataset.y[train_idx],
                            X_test,
                            model=model,
                            random_state=args.random_state + fold_i,
                            args=args,
                        )
                        scores[test_idx] = pred
                    except Exception as exc:
                        qc_rows.append(
                            {
                                "session_file": dataset.session_file,
                                "subject": int(dataset.subject),
                                "day": int(dataset.session),
                                "feature_kind": stable_feature_kind(dataset.feature_kind),
                                "model": model,
                                "stage": "within_session",
                                "reason": "compute_error",
                                "detail": str(exc),
                            }
                        )
                        break
                auc = auc_or_nan(dataset.y, scores)
                rows.append(
                    make_score_row(
                        dataset,
                        model,
                        control,
                        "within_session_cv",
                        prefix_fraction,
                        auc,
                        "cv" if np.isfinite(auc) else "error",
                    )
                )
    return rows, qc_rows


def cross_session_scores(subject_datasets, models, prefix_fractions, args, rng):
    rows = []
    qc_rows = []
    for train_day, train_ds in sorted(subject_datasets.items()):
        for test_day, test_ds in sorted(subject_datasets.items()):
            if int(train_day) == int(test_day):
                continue
            if train_ds.X.shape[2] != test_ds.X.shape[2]:
                qc_rows.append(
                    {
                        "session_file": train_ds.session_file,
                        "subject": int(train_ds.subject),
                        "day": int(train_day),
                        "feature_kind": stable_feature_kind(train_ds.feature_kind),
                        "model": "all",
                        "stage": "cross_session",
                        "reason": "feature_dimension_mismatch",
                        "detail": f"train={train_ds.X.shape[2]}, test={test_ds.X.shape[2]}",
                    }
                )
                continue
            if train_ds.X.shape[1] != test_ds.X.shape[1]:
                n_time = min(train_ds.X.shape[1], test_ds.X.shape[1])
                X_train_base = train_ds.X[:, :n_time, :]
                X_test_base = test_ds.X[:, :n_time, :]
            else:
                X_train_base = train_ds.X
                X_test_base = test_ds.X
            for model in models:
                for control in ["intact", "time_shuffled"]:
                    fracs = prefix_fractions if control == "intact" else [1.0]
                    for prefix_fraction in fracs:
                        X_train = truncate_sequence(X_train_base, prefix_fraction)
                        X_test = truncate_sequence(X_test_base, prefix_fraction)
                        if control == "time_shuffled":
                            X_train = maybe_time_shuffle(X_train, rng)
                            X_test = maybe_time_shuffle(X_test, rng)
                        try:
                            pred = fit_predict_model(
                                X_train,
                                train_ds.y,
                                X_test,
                                model=model,
                                random_state=args.random_state + int(train_day) * 101 + int(test_day),
                                args=args,
                            )
                            auc = auc_or_nan(test_ds.y, pred)
                            status = "transfer" if np.isfinite(auc) else "error"
                        except Exception as exc:
                            auc = np.nan
                            status = "error"
                            qc_rows.append(
                                {
                                    "session_file": train_ds.session_file,
                                    "subject": int(train_ds.subject),
                                    "day": int(train_day),
                                    "feature_kind": stable_feature_kind(train_ds.feature_kind),
                                    "model": model,
                                    "stage": "cross_session",
                                    "reason": "compute_error",
                                    "detail": str(exc),
                                }
                            )
                        rows.append(
                            make_score_row(
                                train_ds,
                                model,
                                control,
                                "cross_session_transfer",
                                prefix_fraction,
                                auc,
                                status,
                                train_day=train_day,
                                test_day=test_day,
                                train_n_trials=len(train_ds.y),
                                test_n_trials=len(test_ds.y),
                            )
                        )
    return rows, qc_rows


def load_datasets(feature_kind: str, args):
    sessions = load_sequence_sessions(load_epochs=False)
    sessions = sorted(sessions, key=lambda s: (int(s["subject"]), int(s["day"]), str(s["epo_file"])))
    if args.max_sessions is not None:
        sessions = sessions[: int(args.max_sessions)]
    n_workers = max(1, int(args.n_workers))
    datasets = []
    qc_rows = []
    tasks = [
        {
            "session_item": item,
            "feature_kind": feature_kind,
            "resample_hz": args.resample_hz,
            "connectivity_window_sec": args.connectivity_window_sec,
            "connectivity_step_sec": args.connectivity_step_sec,
            "connectivity_tmin": args.connectivity_tmin,
            "connectivity_tmax": args.connectivity_tmax,
            "no_feature_cache": args.no_feature_cache,
            "force_feature_recompute": args.force_feature_recompute,
            "feature_cache_verbose": args.feature_cache_verbose,
        }
        for item in sessions
    ]
    if n_workers == 1:
        results = []
        for done, task in enumerate(tasks, start=1):
            results.append(load_dataset_task(task))
            if done == len(tasks) or (done % max(args.progress_every, 1)) == 0:
                print(
                    f"[sequence RNN] loaded {done}/{len(tasks)} sessions for {feature_kind}",
                    flush=True,
                )
    else:
        print(
            f"[sequence RNN] loading {len(tasks)} sessions for {feature_kind} "
            f"with {n_workers} worker(s)",
            flush=True,
        )
        results = parallel_collect_rnn(load_dataset_task, tasks, n_workers)
        print(
            f"[sequence RNN] loaded {len(tasks)}/{len(tasks)} sessions for {feature_kind}",
            flush=True,
        )
    for result in results:
        if result["ok"]:
            datasets.append(result["dataset"])
        else:
            qc_rows.append(result["qc"])
    return datasets, qc_rows


def load_dataset_task(task):
    item = task["session_item"]
    feature_kind = task["feature_kind"]
    try:
        kwargs = {}
        if feature_kind in {"voltage", "time_frequency", "mvpa_decision", "mvpa"}:
            kwargs["resample_hz"] = task["resample_hz"]
        if feature_kind in {"connectivity", "imcoh"}:
            kwargs.update(
                {
                    "window_sec": task["connectivity_window_sec"],
                    "step_sec": task["connectivity_step_sec"],
                    "tmin": task["connectivity_tmin"],
                    "tmax": task["connectivity_tmax"],
                }
            )
        ds = load_feature_sequence(
            item,
            feature_kind=feature_kind,
            use_cache=not task["no_feature_cache"],
            force_recompute=task["force_feature_recompute"],
            verbose=task["feature_cache_verbose"],
            **kwargs,
        )
        return {"ok": True, "dataset": ds}
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": item.get("epo_file", ""),
                "subject": int(item.get("subject", -1)),
                "day": int(item.get("day", -1)),
                "feature_kind": stable_feature_kind(feature_kind),
                "model": "all",
                "stage": "load_feature_sequence",
                "reason": "load_error",
                "detail": str(exc),
            },
        }


def within_session_task(task):
    args = SimpleNamespace(**task["args"])
    rng = np.random.default_rng(int(task["seed"]))
    return within_session_scores(
        task["dataset"],
        task["models"],
        task["prefix_fractions"],
        args,
        rng,
    )


def cross_session_task(task):
    args = SimpleNamespace(**task["args"])
    rng = np.random.default_rng(int(task["seed"]))
    return cross_session_scores(
        task["subject_datasets"],
        task["models"],
        task["prefix_fractions"],
        args,
        rng,
    )


def _args_payload(args):
    return {
        "min_trials": args.min_trials,
        "min_class_trials": args.min_class_trials,
        "cv_splits": args.cv_splits,
        "random_state": args.random_state,
        "hidden_size": args.hidden_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "torch_num_threads": args.torch_num_threads,
        "learning_rate": args.learning_rate,
        "dropout": args.dropout,
    }


def score_feature_datasets(feature_kind, datasets, models, prefix_fractions, args):
    n_workers = max(1, int(args.n_workers))
    all_rows = []
    qc_rows = []
    args_payload = _args_payload(args)
    within_tasks = [
        {
            "dataset": ds,
            "models": models,
            "prefix_fractions": prefix_fractions,
            "args": args_payload,
            "seed": int(args.random_state) + int(ds.subject) * 1009 + int(ds.session) * 37,
        }
        for ds in datasets
    ]
    if n_workers == 1:
        within_results = []
        for done, task in enumerate(within_tasks, start=1):
            within_results.append(within_session_task(task))
            if done == len(within_tasks) or (done % max(args.progress_every, 1)) == 0:
                print(
                    f"[sequence RNN] within-session scored {done}/{len(within_tasks)} "
                    f"datasets for {feature_kind}",
                    flush=True,
                )
    else:
        print(
            f"[sequence RNN] within-session scoring {len(within_tasks)} datasets "
            f"for {feature_kind} with {n_workers} worker(s)",
            flush=True,
        )
        within_results = parallel_collect_rnn(within_session_task, within_tasks, n_workers)
        print(
            f"[sequence RNN] within-session scored {len(within_tasks)}/{len(within_tasks)} "
            f"datasets for {feature_kind}",
            flush=True,
        )
    for rows, new_qc in within_results:
        all_rows.extend(rows)
        qc_rows.extend(new_qc)

    by_subject = {}
    for ds in datasets:
        by_subject.setdefault(int(ds.subject), {})[int(ds.session)] = ds
    cross_tasks = [
        {
            "subject": subject,
            "subject_datasets": subject_datasets,
            "models": models,
            "prefix_fractions": prefix_fractions,
            "args": args_payload,
            "seed": int(args.random_state) + int(subject) * 2003,
        }
        for subject, subject_datasets in sorted(by_subject.items())
        if len(subject_datasets) > 1
    ]
    if n_workers == 1:
        cross_results = []
        for done, task in enumerate(cross_tasks, start=1):
            cross_results.append(cross_session_task(task))
            if done == len(cross_tasks) or (done % max(args.progress_every, 1)) == 0:
                print(
                    f"[sequence RNN] cross-session scored {done}/{len(cross_tasks)} "
                    f"subjects for {feature_kind}",
                    flush=True,
                )
    else:
        print(
            f"[sequence RNN] cross-session scoring {len(cross_tasks)} subjects "
            f"for {feature_kind} with {n_workers} worker(s)",
            flush=True,
        )
        cross_results = parallel_collect_rnn(cross_session_task, cross_tasks, n_workers)
        print(
            f"[sequence RNN] cross-session scored {len(cross_tasks)}/{len(cross_tasks)} "
            f"subjects for {feature_kind}",
            flush=True,
        )
    for rows, new_qc in cross_results:
        all_rows.extend(rows)
        qc_rows.extend(new_qc)
    return all_rows, qc_rows


def make_group_summary(subject_df):
    if subject_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = [
        "feature_kind",
        "model",
        "control",
        "evaluation",
        "train_day",
        "test_day",
        "prefix_fraction",
    ]
    for keys, g in subject_df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys))
        row.update(
            {
                "day_distance": int(abs(int(row["train_day"]) - int(row["test_day"]))),
                "auc_mean": float(np.nanmean(g["auc"])),
                "auc_sem": sem(g["auc"]),
                "n_subjects": int(g["subject"].nunique()),
                "n_rows": int(len(g)),
                "n_finite": int(np.sum(np.isfinite(g["auc"].to_numpy(dtype=float)))),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def write_progress(path, stage, done, total, t0):
    path.write_text(
        json.dumps(
            {
                "stage": stage,
                "torch_available": bool(TORCH_AVAILABLE),
                "torch_version": TORCH_VERSION,
                "done": int(done),
                "total": int(total),
                "elapsed_sec": float(time.time() - t0),
                "updated_at_unix": float(time.time()),
            },
            indent=2,
        )
    )


def parse_feature_kinds(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def parse_prefix_fractions(text):
    vals = [float(x.strip()) for x in str(text).split(",") if x.strip()]
    vals = sorted(set(vals))
    if not vals or min(vals) <= 0.0 or max(vals) > 1.0:
        raise ValueError("prefix fractions must be in (0, 1]")
    if 1.0 not in vals:
        vals.append(1.0)
    return vals


def run_sequence_rnn_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    feature_kinds=None,
    smoke: bool = False,
    **kwargs,
):
    args = make_arg_parser().parse_args([])
    for key, value in kwargs.items():
        setattr(args, key, value)
    args.output_dir = str(output_dir)
    args.smoke = bool(smoke)
    if feature_kinds is not None:
        args.feature_kinds = ",".join(feature_kinds) if isinstance(feature_kinds, (list, tuple)) else str(feature_kinds)
    if args.smoke:
        args.epochs = min(int(args.epochs), 1)
        args.max_sessions = args.max_sessions if args.max_sessions is not None else 2
        if "feature_kinds" not in kwargs and feature_kinds is None:
            args.feature_kinds = "voltage,mvpa_decision"
    return main(args)


def make_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--feature-kinds", default=",".join(DEFAULT_FEATURE_KINDS))
    parser.add_argument("--models", default="bag_logreg,gru,lstm")
    parser.add_argument("--prefix-fractions", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--resample-hz", type=float, default=64.0)
    parser.add_argument("--min-trials", type=int, default=20)
    parser.add_argument("--min-class-trials", type=int, default=5)
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--torch-num-threads", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--n-workers", type=int, default=N_JOBS)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--connectivity-window-sec", type=float, default=0.05)
    parser.add_argument("--connectivity-step-sec", type=float, default=0.05)
    parser.add_argument("--connectivity-tmin", type=float, default=0.0)
    parser.add_argument("--connectivity-tmax", type=float, default=0.8)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--force-feature-recompute", action="store_true")
    parser.add_argument("--feature-cache-verbose", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser


def main(args=None):
    if args is None:
        args = make_arg_parser().parse_args()
    if args.smoke:
        args.epochs = min(int(args.epochs), 1)
        args.max_sessions = args.max_sessions if args.max_sessions is not None else 2
        args.n_workers = 1
        if args.feature_kinds == ",".join(DEFAULT_FEATURE_KINDS):
            args.feature_kinds = "voltage,mvpa_decision"
    if torch is not None:
        torch.set_num_threads(max(1, int(args.torch_num_threads)))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    subject_csv = output_dir / "sequence_rnn_subject_scores.csv"
    group_csv = output_dir / "sequence_rnn_group_summary.csv"
    qc_csv = output_dir / "sequence_rnn_qc_log.csv"
    progress_json = output_dir / "sequence_rnn_progress.json"
    t0 = time.time()
    rng = np.random.default_rng(int(args.random_state))
    feature_kinds = parse_feature_kinds(args.feature_kinds)
    models = parse_feature_kinds(args.models)
    prefix_fractions = parse_prefix_fractions(args.prefix_fractions)
    if torch is None:
        models = [m for m in models if m == "bag_logreg"]
    qc_rows = []
    if torch is None:
        qc_rows.append(
            {
                "session_file": "",
                "subject": -1,
                "day": -1,
                "feature_kind": "all",
                "model": "gru,lstm",
                "stage": "import",
                "reason": "torch_unavailable",
                "detail": TORCH_IMPORT_ERROR,
            }
        )
    all_rows = []
    total = len(feature_kinds)
    write_progress(progress_json, "start", 0, total, t0)
    print(
        f"[sequence RNN] feature_kinds={feature_kinds}, models={models}, "
        f"epochs={args.epochs}, smoke={args.smoke}, torch_available={TORCH_AVAILABLE}",
        flush=True,
    )
    for feature_done, feature_kind in enumerate(feature_kinds, start=1):
        datasets, load_qc = load_datasets(feature_kind, args)
        qc_rows.extend(load_qc)
        rows, new_qc = score_feature_datasets(
            feature_kind,
            datasets,
            models,
            prefix_fractions,
            args,
        )
        all_rows.extend(rows)
        qc_rows.extend(new_qc)
        write_progress(progress_json, "feature_kind", feature_done, total, t0)
        print(
            f"[sequence RNN] completed {feature_kind}: "
            f"{len(datasets)} datasets, {len(all_rows)} score rows so far",
            flush=True,
        )
    subject_df = pd.DataFrame(all_rows)
    if not subject_df.empty:
        subject_df = subject_df.sort_values(
            [
                "feature_kind",
                "model",
                "control",
                "evaluation",
                "subject",
                "train_day",
                "test_day",
                "prefix_fraction",
            ]
        )
    subject_df.to_csv(subject_csv, index=False)
    group_df = make_group_summary(subject_df)
    group_df.to_csv(group_csv, index=False)
    qc_df = pd.DataFrame(qc_rows, columns=QC_COLUMNS)
    qc_df.to_csv(qc_csv, index=False)
    write_progress(progress_json, "completed", total, total, t0)
    print(f"[sequence RNN] wrote {subject_csv}", flush=True)
    print(f"[sequence RNN] wrote {group_csv}", flush=True)
    print(f"[sequence RNN] wrote {qc_csv}", flush=True)
    return {
        "subject_csv": subject_csv,
        "group_csv": group_csv,
        "qc_csv": qc_csv,
        "progress_json": progress_json,
        "subject_df": subject_df,
        "group_df": group_df,
        "qc_df": qc_df,
    }


if __name__ == "__main__":
    main()
