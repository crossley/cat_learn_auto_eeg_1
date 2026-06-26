#!/usr/bin/env python3
"""Gaussian HMM/state surrogate analyses for trial-wise feature sequences."""

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
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from hmmlearn.hmm import GaussianHMM

    HMM_BACKEND = "hmmlearn_gaussian_hmm"
    HMM_IMPORT_ERROR = ""
except Exception:
    GaussianHMM = None
    HMM_BACKEND = "gmm_state_surrogate"
    HMM_IMPORT_ERROR = "hmmlearn unavailable"

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from sequence_feature_interface import load_feature_sequence, load_sequence_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8
DEFAULT_TMIN = 0.0
DEFAULT_TMAX = 0.8


def _as_clean_sequence(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim != 3:
        raise ValueError(f"expected X as trials x time x features, got shape={X.shape}")
    valid_obs = np.isfinite(X).any(axis=2)
    X_flat = X.reshape(-1, X.shape[2])
    valid_flat = valid_obs.reshape(-1)
    return X_flat[valid_flat], valid_obs


def make_preprocessor(X_flat: np.ndarray, max_components: int, random_state: int):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    n_features = int(X_flat.shape[1])
    n_samples = int(X_flat.shape[0])
    n_components = min(int(max_components), n_features, max(1, n_samples - 1))
    if n_features > n_components:
        steps.append(
            (
                "pca",
                PCA(n_components=n_components, random_state=random_state, svd_solver="auto"),
            )
        )
    return Pipeline(steps)


def _hmm_param_count(n_states: int, n_features: int, covariance_type: str) -> int:
    start = n_states - 1
    trans = n_states * (n_states - 1)
    means = n_states * n_features
    if covariance_type == "full":
        cov = n_states * n_features * (n_features + 1) // 2
    elif covariance_type == "tied":
        cov = n_features * (n_features + 1) // 2
    elif covariance_type == "spherical":
        cov = n_states
    else:
        cov = n_states * n_features
    return int(start + trans + means + cov)


def _information_criteria(log_likelihood: float, n_params: int, n_obs: int) -> tuple[float, float]:
    aic = 2.0 * n_params - 2.0 * log_likelihood
    bic = np.log(max(n_obs, 1)) * n_params - 2.0 * log_likelihood
    return float(aic), float(bic)


def _state_summary_rows(
    posterior: np.ndarray,
    states: np.ndarray,
    valid_obs: np.ndarray,
    dataset,
    n_states: int,
    model_family: str,
):
    n_trials, n_times = valid_obs.shape
    posterior_full = np.full((n_trials * n_times, n_states), np.nan, dtype=float)
    states_full = np.full(n_trials * n_times, -1, dtype=int)
    valid_flat = valid_obs.reshape(-1)
    posterior_full[valid_flat] = posterior
    states_full[valid_flat] = states
    posterior_full = posterior_full.reshape(n_trials, n_times, n_states)
    states_full = states_full.reshape(n_trials, n_times)

    time_rows = []
    for ti, tsec in enumerate(dataset.time):
        valid_t = states_full[:, ti] >= 0
        for state in range(n_states):
            vals = posterior_full[:, ti, state]
            state_mask = states_full[:, ti] == state
            time_rows.append(
                {
                    "subject": dataset.subject,
                    "day": dataset.session,
                    "session_file": dataset.session_file,
                    "feature_kind": dataset.feature_kind,
                    "model_family": model_family,
                    "n_states": n_states,
                    "time_index": ti,
                    "time_sec": float(tsec),
                    "state": state,
                    "posterior_mean": float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan,
                    "occupancy": float(np.mean(state_mask[valid_t])) if np.any(valid_t) else np.nan,
                    "n_trials": int(n_trials),
                }
            )

    trial_rows = []
    dwell_rows = []
    transition_counts = np.zeros((n_states, n_states), dtype=float)
    times = np.asarray(dataset.time, dtype=float)
    for trial in range(n_trials):
        seq = states_full[trial]
        valid = seq >= 0
        if not np.any(valid):
            continue
        seq_valid = seq[valid]
        time_valid = times[valid]
        meta = dataset.metadata.iloc[trial].to_dict() if len(dataset.metadata) > trial else {}
        label = int(dataset.y[trial]) if len(dataset.y) > trial else -1
        for state in range(n_states):
            in_state = seq_valid == state
            first = np.where(in_state)[0]
            trial_rows.append(
                {
                    "subject": dataset.subject,
                    "day": dataset.session,
                    "session_file": dataset.session_file,
                    "feature_kind": dataset.feature_kind,
                    "model_family": model_family,
                    "n_states": n_states,
                    "trial_index": trial,
                    "trial_index_aligned": int(meta.get("trial_index_aligned", trial)),
                    "label": label,
                    "state": state,
                    "occupancy": float(np.mean(in_state)),
                    "posterior_mean": float(np.nanmean(posterior_full[trial, valid, state])),
                    "onset_time_sec": float(time_valid[first[0]]) if len(first) else np.nan,
                }
            )
            if len(first):
                run_starts = first[np.r_[True, np.diff(first) > 1]]
                run_ends = first[np.r_[np.diff(first) > 1, True]]
                for start_i, end_i in zip(run_starts, run_ends):
                    dwell_rows.append(
                        {
                            "subject": dataset.subject,
                            "day": dataset.session,
                            "session_file": dataset.session_file,
                            "feature_kind": dataset.feature_kind,
                            "model_family": model_family,
                            "n_states": n_states,
                            "trial_index": trial,
                            "trial_index_aligned": int(meta.get("trial_index_aligned", trial)),
                            "label": label,
                            "state": state,
                            "start_time_sec": float(time_valid[start_i]),
                            "end_time_sec": float(time_valid[end_i]),
                            "dwell_steps": int(end_i - start_i + 1),
                            "dwell_sec": float(time_valid[end_i] - time_valid[start_i]),
                        }
                    )
        for prev, nxt in zip(seq_valid[:-1], seq_valid[1:]):
            transition_counts[int(prev), int(nxt)] += 1.0

    transition_rows = []
    row_sums = transition_counts.sum(axis=1, keepdims=True)
    transition_prob = np.divide(
        transition_counts,
        row_sums,
        out=np.full_like(transition_counts, np.nan, dtype=float),
        where=row_sums > 0,
    )
    for i in range(n_states):
        for j in range(n_states):
            transition_rows.append(
                {
                    "subject": dataset.subject,
                    "day": dataset.session,
                    "session_file": dataset.session_file,
                    "feature_kind": dataset.feature_kind,
                    "model_family": model_family,
                    "n_states": n_states,
                    "from_state": i,
                    "to_state": j,
                    "transition_count": float(transition_counts[i, j]),
                    "transition_probability": float(transition_prob[i, j]),
                }
            )

    return time_rows, trial_rows, dwell_rows, transition_rows


def fit_state_models_for_dataset(
    dataset,
    state_grid=range(2, 7),
    covariance_type: str = "diag",
    max_components: int = 12,
    random_state: int = 42,
    heldout_fraction: float = 0.2,
    diagnostic_states: list[int] | None = None,
):
    X_flat, valid_obs = _as_clean_sequence(dataset.X)
    if X_flat.shape[0] < 10:
        raise ValueError(f"insufficient observations: {X_flat.shape[0]}")
    if len(dataset.y) >= 4 and len(np.unique(dataset.y)) > 1 and min(np.bincount(dataset.y)) >= 2:
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=heldout_fraction, random_state=random_state
        )
        train_trials, test_trials = next(splitter.split(np.zeros(len(dataset.y)), dataset.y))
    else:
        splitter = ShuffleSplit(n_splits=1, test_size=heldout_fraction, random_state=random_state)
        train_trials, test_trials = next(splitter.split(np.arange(dataset.X.shape[0])))

    flat_trial = np.repeat(np.arange(dataset.X.shape[0]), dataset.X.shape[1])[valid_obs.reshape(-1)]
    train_mask = np.isin(flat_trial, train_trials)
    test_mask = np.isin(flat_trial, test_trials)
    preprocessor = make_preprocessor(X_flat[train_mask], max_components, random_state)
    X_train = preprocessor.fit_transform(X_flat[train_mask])
    X_test = preprocessor.transform(X_flat[test_mask]) if np.any(test_mask) else np.empty((0, X_train.shape[1]))
    X_all = preprocessor.transform(X_flat)
    lengths_all = valid_obs.sum(axis=1).astype(int).tolist()
    lengths_train = valid_obs[train_trials].sum(axis=1).astype(int).tolist()
    X_train_by_trial = []
    for trial in train_trials:
        obs_idx = flat_trial == trial
        if np.any(obs_idx):
            X_train_by_trial.append(preprocessor.transform(X_flat[obs_idx]))
    X_train_seq = np.vstack(X_train_by_trial)
    lengths_train = [int(x.shape[0]) for x in X_train_by_trial]

    model_rows = []
    fitted = {}
    model_family = HMM_BACKEND
    for n_states in state_grid:
        n_states = int(n_states)
        try:
            if GaussianHMM is not None:
                model = GaussianHMM(
                    n_components=n_states,
                    covariance_type=covariance_type,
                    n_iter=100,
                    tol=1e-3,
                    random_state=random_state,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_train_seq, lengths=lengths_train)
                train_ll = float(model.score(X_train_seq, lengths=lengths_train))
                test_ll = float(model.score(X_test)) if len(X_test) else np.nan
                n_params = _hmm_param_count(n_states, X_train.shape[1], covariance_type)
            else:
                model = GaussianMixture(
                    n_components=n_states,
                    covariance_type=covariance_type,
                    reg_covar=1e-6,
                    n_init=3,
                    random_state=random_state,
                )
                model.fit(X_train)
                train_ll = float(model.score(X_train) * len(X_train))
                test_ll = float(model.score(X_test) * len(X_test)) if len(X_test) else np.nan
                n_params = int(model._n_parameters())
            aic, bic = _information_criteria(train_ll, n_params, len(X_train))
            model_rows.append(
                {
                    "subject": dataset.subject,
                    "day": dataset.session,
                    "session_file": dataset.session_file,
                    "feature_kind": dataset.feature_kind,
                    "model_family": model_family,
                    "n_states": n_states,
                    "covariance_type": covariance_type,
                    "n_trials": int(dataset.X.shape[0]),
                    "n_observations_train": int(len(X_train)),
                    "n_observations_test": int(len(X_test)),
                    "n_features_input": int(dataset.X.shape[2]),
                    "n_features_model": int(X_train.shape[1]),
                    "log_likelihood_train": train_ll,
                    "log_likelihood_test": test_ll,
                    "heldout_log_likelihood_per_obs": float(test_ll / len(X_test)) if len(X_test) else np.nan,
                    "aic": aic,
                    "bic": bic,
                    "status": "ok",
                    "detail": "",
                }
            )
            fitted[n_states] = model
        except Exception as exc:
            model_rows.append(
                {
                    "subject": dataset.subject,
                    "day": dataset.session,
                    "session_file": dataset.session_file,
                    "feature_kind": dataset.feature_kind,
                    "model_family": model_family,
                    "n_states": n_states,
                    "covariance_type": covariance_type,
                    "n_trials": int(dataset.X.shape[0]),
                    "n_observations_train": int(len(X_train)),
                    "n_observations_test": int(len(X_test)),
                    "n_features_input": int(dataset.X.shape[2]),
                    "n_features_model": int(X_train.shape[1]),
                    "log_likelihood_train": np.nan,
                    "log_likelihood_test": np.nan,
                    "heldout_log_likelihood_per_obs": np.nan,
                    "aic": np.nan,
                    "bic": np.nan,
                    "status": "fit_error",
                    "detail": str(exc),
                }
            )

    ok_rows = [r for r in model_rows if r["status"] == "ok" and np.isfinite(r["bic"])]
    if not ok_rows:
        raise RuntimeError("no state model fit succeeded")
    best_states = int(min(ok_rows, key=lambda r: r["bic"])["n_states"])
    for row in model_rows:
        row["selected_by_bic"] = bool(row["n_states"] == best_states and row["status"] == "ok")
    requested_states = {best_states}
    if diagnostic_states is not None:
        requested_states.update(int(x) for x in diagnostic_states)
    requested_states = sorted(x for x in requested_states if x in fitted)

    time_rows = []
    trial_rows = []
    dwell_rows = []
    transition_rows = []
    for n_states in requested_states:
        model = fitted[n_states]
        if GaussianHMM is not None:
            posterior = model.predict_proba(X_all, lengths=lengths_all)
            states = model.predict(X_all, lengths=lengths_all)
        else:
            posterior = model.predict_proba(X_all)
            states = model.predict(X_all)
        state_rows = _state_summary_rows(
            posterior, states, valid_obs, dataset, n_states, model_family
        )
        for rows in state_rows:
            for row in rows:
                row["selected_by_bic"] = bool(n_states == best_states)
        time_rows.extend(state_rows[0])
        trial_rows.extend(state_rows[1])
        dwell_rows.extend(state_rows[2])
        transition_rows.extend(state_rows[3])
    return {
        "model_rows": model_rows,
        "time_rows": time_rows,
        "trial_rows": trial_rows,
        "dwell_rows": dwell_rows,
        "transition_rows": transition_rows,
    }


def process_sequence_hmm_session(task: dict):
    session_item = task["session_item"]
    subject = int(session_item.get("subject", -1))
    day = int(session_item.get("day", -1))
    session_file = session_item.get("epo_file", "")
    try:
        dataset = load_feature_sequence(
            session_item,
            task["feature_kind"],
            use_cache=task.get("use_feature_cache", True),
            force_recompute=task.get("force_feature_recompute", False),
            verbose=task.get("feature_cache_verbose", False),
            **task.get("feature_kwargs", {}),
        )
        result = fit_state_models_for_dataset(
            dataset,
            state_grid=range(task["min_states"], task["max_states"] + 1),
            covariance_type=task["covariance_type"],
            max_components=task["max_components"],
            random_state=task["random_state"],
            heldout_fraction=task["heldout_fraction"],
            diagnostic_states=task.get("diagnostic_states"),
        )
        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "session_file": session_file,
            **result,
        }
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "feature_kind": task["feature_kind"],
                "stage": "fit",
                "reason": "analysis_error",
                "detail": str(exc),
            },
        }


def iter_parallel_hmm_results(tasks: list[dict], n_workers: int):
    def delayed_items():
        for task in tasks:
            yield delayed(process_sequence_hmm_session)(task)

    def make_iterator(backend: str):
        return Parallel(
            n_jobs=int(n_workers),
            backend=backend,
            verbose=0,
            return_as="generator_unordered",
        )(delayed_items())

    try:
        if threadpool_limits is None:
            iterator = make_iterator("loky")
            yield from iterator
        else:
            with threadpool_limits(limits=1):
                iterator = make_iterator("loky")
                yield from iterator
    except PermissionError as exc:
        print(
            "[sequence HMM] loky process backend unavailable "
            f"({exc}); falling back to threading backend",
            flush=True,
        )
        if threadpool_limits is None:
            iterator = make_iterator("threading")
            yield from iterator
        else:
            with threadpool_limits(limits=1):
                iterator = make_iterator("threading")
                yield from iterator


def _parse_feature_kwargs(args) -> dict:
    kwargs = {}
    if args.feature_kind in {"voltage", "time_frequency", "mvpa", "mvpa_decision"}:
        kwargs["resample_hz"] = None if args.resample_hz <= 0 else float(args.resample_hz)
        kwargs["tmin"] = DEFAULT_TMIN if args.tmin is None else float(args.tmin)
        kwargs["tmax"] = DEFAULT_TMAX if args.tmax is None else float(args.tmax)
    if args.feature_kind in {"connectivity", "imcoh"}:
        kwargs.update(
            {
                "roi_pair": args.roi_pair,
                "window_sec": float(args.window_sec),
                "step_sec": float(args.step_sec),
                "tmin": DEFAULT_TMIN if args.tmin is None else float(args.tmin),
                "tmax": DEFAULT_TMAX if args.tmax is None else float(args.tmax),
            }
        )
    return kwargs


def run_sequence_hmm_analysis(
    feature_kind: str = "mvpa_decision",
    output_dir: Path | str = OUTPUT_DIR,
    min_states: int = 2,
    max_states: int = 6,
    covariance_type: str = "diag",
    max_components: int = 12,
    random_state: int = 42,
    n_workers: int | None = None,
    smoke: bool = False,
    max_sessions: int | None = None,
    heldout_fraction: float = 0.2,
    feature_kwargs: dict | None = None,
    diagnostic_states: list[int] | None = None,
    output_label: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n_workers = N_JOBS if n_workers is None else max(1, int(n_workers))
    feature_kwargs = {} if feature_kwargs is None else dict(feature_kwargs)
    if smoke:
        max_sessions = 1 if max_sessions is None else max_sessions
        n_workers = 1
        max_states = min(max_states, 3)

    label = feature_kind if output_label is None else str(output_label)
    prefix = f"sequence_hmm_{label}"
    progress_json = output_dir / f"{prefix}_progress.json"
    qc_csv = output_dir / f"{prefix}_qc_log.csv"
    model_csv = output_dir / f"{prefix}_model_selection.csv"
    time_csv = output_dir / f"{prefix}_state_timecourse.csv"
    trial_csv = output_dir / f"{prefix}_trial_state_summary.csv"
    dwell_csv = output_dir / f"{prefix}_dwell_times.csv"
    transition_csv = output_dir / f"{prefix}_transitions.csv"

    t0 = time.time()

    def write_progress(stage: str, done: int, total: int):
        progress_json.write_text(
            json.dumps(
                {
                    "stage": stage,
                    "feature_kind": feature_kind,
                    "model_family": HMM_BACKEND,
                    "backend_detail": HMM_IMPORT_ERROR,
                    "done": int(done),
                    "total": int(total),
                    "elapsed_sec": float(time.time() - t0),
                    "updated_at_unix": float(time.time()),
                },
                indent=2,
            )
        )

    sessions = load_sequence_sessions(load_epochs=False)
    if max_sessions is not None:
        sessions = sessions[: int(max_sessions)]
    use_feature_cache = bool(feature_kwargs.pop("_use_feature_cache", True))
    force_feature_recompute = bool(feature_kwargs.pop("_force_feature_recompute", False))
    feature_cache_verbose = bool(feature_kwargs.pop("_feature_cache_verbose", False))
    tasks = [
        {
            "session_item": item,
            "feature_kind": feature_kind,
            "feature_kwargs": feature_kwargs,
            "min_states": int(min_states),
            "max_states": int(max_states),
            "covariance_type": covariance_type,
            "max_components": int(max_components),
            "random_state": int(random_state),
            "heldout_fraction": float(heldout_fraction),
            "use_feature_cache": use_feature_cache,
            "force_feature_recompute": force_feature_recompute,
            "feature_cache_verbose": feature_cache_verbose,
            "diagnostic_states": diagnostic_states,
        }
        for item in sessions
    ]
    write_progress("running", 0, len(tasks))
    print(
        f"[sequence HMM] feature={feature_kind} sessions={len(tasks)} "
        f"states={min_states}-{max_states} n_workers={n_workers} backend={HMM_BACKEND}",
        flush=True,
    )
    raw_results = []
    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            raw_results.append(process_sequence_hmm_session(task))
            write_progress("running", done, len(tasks))
            if done == len(tasks) or done % 5 == 0:
                print(
                    f"[sequence HMM] completed {done}/{len(tasks)} sessions",
                    flush=True,
                )
    else:
        print(
            f"[sequence HMM] parallel batch running; progress JSON: {progress_json}",
            flush=True,
        )
        for done, result in enumerate(iter_parallel_hmm_results(tasks, n_workers), start=1):
            raw_results.append(result)
            write_progress("running", done, len(tasks))
            status = "ok" if result.get("ok") else "failed"
            if done == len(tasks) or done % 5 == 0 or not result.get("ok"):
                if result.get("ok"):
                    ident = f"sub={result.get('subject')} day={result.get('day')}"
                else:
                    qc = result.get("qc", {})
                    ident = f"sub={qc.get('subject')} day={qc.get('day')}"
                elapsed = (time.time() - t0) / 60.0
                print(
                    f"[sequence HMM] completed {done}/{len(tasks)} sessions "
                    f"({ident}, {status}, elapsed={elapsed:.1f} min)",
                    flush=True,
                )

    model_rows = []
    time_rows = []
    trial_rows = []
    dwell_rows = []
    transition_rows = []
    qc_rows = []
    for result in raw_results:
        if result["ok"]:
            model_rows.extend(result["model_rows"])
            time_rows.extend(result["time_rows"])
            trial_rows.extend(result["trial_rows"])
            dwell_rows.extend(result["dwell_rows"])
            transition_rows.extend(result["transition_rows"])
        else:
            qc_rows.append(result["qc"])

    model_df = pd.DataFrame(model_rows)
    time_df = pd.DataFrame(time_rows)
    trial_df = pd.DataFrame(trial_rows)
    dwell_df = pd.DataFrame(dwell_rows)
    transition_df = pd.DataFrame(transition_rows)
    qc_df = pd.DataFrame(
        qc_rows,
        columns=["session_file", "subject", "day", "feature_kind", "stage", "reason", "detail"],
    )
    model_df.to_csv(model_csv, index=False)
    time_df.to_csv(time_csv, index=False)
    trial_df.to_csv(trial_csv, index=False)
    dwell_df.to_csv(dwell_csv, index=False)
    transition_df.to_csv(transition_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    write_progress("completed", len(tasks), len(tasks))
    if model_df.empty:
        raise RuntimeError(f"sequence HMM produced no valid model rows; see {qc_csv}")
    return {
        "model_df": model_df,
        "time_df": time_df,
        "trial_df": trial_df,
        "dwell_df": dwell_df,
        "transition_df": transition_df,
        "qc_df": qc_df,
        "model_csv": model_csv,
        "time_csv": time_csv,
        "trial_csv": trial_csv,
        "dwell_csv": dwell_csv,
        "transition_csv": transition_csv,
        "qc_csv": qc_csv,
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-kind", default=os.environ.get("SEQUENCE_HMM_FEATURE_KIND", "mvpa_decision"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--min-states", type=int, default=2)
    parser.add_argument("--max-states", type=int, default=6)
    parser.add_argument("--covariance-type", default="diag", choices=["diag", "full", "tied", "spherical"])
    parser.add_argument("--max-components", type=int, default=12)
    parser.add_argument("--heldout-fraction", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-workers", type=int, default=int(os.environ.get("SEQUENCE_HMM_N_WORKERS", "1")))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--resample-hz", type=float, default=128.0)
    parser.add_argument("--roi-pair", default="visual_to_central")
    parser.add_argument("--window-sec", type=float, default=0.05)
    parser.add_argument("--step-sec", type=float, default=0.025)
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument(
        "--output-label",
        default=None,
        help="Output label used in sequence_hmm_<label> files; defaults to feature kind.",
    )
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--force-feature-recompute", action="store_true")
    parser.add_argument("--feature-cache-verbose", action="store_true")
    parser.add_argument(
        "--diagnostic-states",
        default="",
        help="Comma-separated state counts to save full diagnostics for in addition to the BIC-selected model, e.g. 4,6.",
    )
    return parser


def _parse_diagnostic_states(text: str):
    vals = []
    for item in str(text).split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return vals or None


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_sequence_hmm_analysis(
        feature_kind=args.feature_kind,
        output_dir=args.output_dir,
        min_states=args.min_states,
        max_states=args.max_states,
        covariance_type=args.covariance_type,
        max_components=args.max_components,
        random_state=args.random_state,
        n_workers=args.n_workers,
        smoke=args.smoke,
        max_sessions=args.max_sessions,
        heldout_fraction=args.heldout_fraction,
        feature_kwargs={
            **_parse_feature_kwargs(args),
            "_use_feature_cache": not args.no_feature_cache,
            "_force_feature_recompute": args.force_feature_recompute,
            "_feature_cache_verbose": args.feature_cache_verbose,
        },
        diagnostic_states=_parse_diagnostic_states(args.diagnostic_states),
        output_label=args.output_label,
    )
