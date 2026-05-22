#!/usr/bin/env python3
"""Stimulus-locked time-resolved MVPA (Stim/A vs Stim/B)."""

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
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from load_project_data import load_sessions
from util_mvpa import pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8


def build_clf(random_state: int):
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


def decode_timecourse(X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = build_clf(random_state=random_state)
    n_times = X.shape[2]
    auc = np.full(n_times, np.nan, dtype=float)
    for ti in range(n_times):
        Xt = X[:, :, ti]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                scores = cross_val_score(clf, Xt, y, cv=cv, scoring="roc_auc")
            auc[ti] = float(np.mean(scores))
        except Exception:
            auc[ti] = np.nan
    return auc


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


def process_stim_mvpa_session(task: dict):
    session_file = task["epo_file"]
    subject = int(task["subject"])
    day = int(task["day"])
    min_epochs = int(task["min_epochs"])
    random_state = int(task["random_state"])

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
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

    auc = decode_timecourse(X, y, n_splits=5, random_state=random_state)
    times = stim_epochs.times.copy()
    session_rows = [
        {
            "session_file": session_file,
            "subject": subject,
            "day": day,
            "time_sec": float(times[ti]),
            "auc": float(auc_val),
            "n_trials": n_trials,
            "n_a": n_a,
            "n_b": n_b,
        }
        for ti, auc_val in enumerate(auc)
    ]

    haufe_rows = []
    channel_pos = []
    haufe_qc = None
    try:
        patterns = compute_haufe_patterns_from_xy(X, y, random_state=random_state)
        for ci, ch in enumerate(stim_epochs.ch_names):
            loc = stim_epochs.info["chs"][stim_epochs.info.ch_names.index(ch)]["loc"][:3]
            channel_pos.append({"channel": ch, "x": float(loc[0]), "y": float(loc[1]), "z": float(loc[2])})
            for ti, tsec in enumerate(times):
                val = float(patterns[ci, ti])
                haufe_rows.append(
                    {
                        "subject": subject,
                        "day": day,
                        "session_file": session_file,
                        "channel": ch,
                        "time_sec": float(tsec),
                        "pattern": val,
                        "abs_pattern": float(np.abs(val)),
                        "n_trials": n_trials,
                        "n_a": n_a,
                        "n_b": n_b,
                    }
                )
    except Exception as exc:
        haufe_qc = {
            "session_file": session_file,
            "subject": subject,
            "day": day,
            "stage": "haufe",
            "reason": "compute_error",
            "detail": str(exc),
        }

    return {
        "ok": True,
        "session_rows": session_rows,
        "haufe_rows": haufe_rows,
        "channel_pos": channel_pos,
        "haufe_qc": haufe_qc,
    }


def run_mvpa_stim_locked_cat_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    min_epochs: int = 20,
    random_state: int = 42,
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

    session_csv = output_dir / "mvpa_stim_locked_cat_session_timecourse.csv"
    subject_day_csv = output_dir / "mvpa_stim_locked_cat_subject_day_timecourse.csv"
    day_means_csv = output_dir / "mvpa_stim_locked_cat_day_means_timecourse.csv"
    day_effect_csv = output_dir / "mvpa_stim_locked_cat_day_effect_per_time.csv"
    qc_csv = output_dir / "mvpa_stim_locked_cat_qc_log.csv"
    haufe_session_csv = output_dir / "mvpa_stim_locked_cat_haufe_session_channel_time.csv"
    haufe_day_mean_csv = output_dir / "mvpa_stim_locked_cat_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_stim_locked_cat_haufe_channel_positions.csv"
    progress_json = output_dir / "mvpa_stim_locked_cat_progress.json"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    session_rows = []
    haufe_rows = []
    haufe_channel_pos = {}
    t0 = time.time()

    def write_progress(stage: str, done: int, total: int):
        payload = {
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

    sessions = load_sessions()
    tasks = [
        {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "epo_file": item["epo_file"],
            "epo_path": str(item["epo_path"]),
            "min_epochs": int(min_epochs),
            "random_state": int(random_state),
        }
        for item in sessions
    ]
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    write_progress("running", 0, len(tasks))

    def handle_result(result, done):
        if not result["ok"]:
            qc_rows.append(result["qc"])
        else:
            session_rows.extend(result["session_rows"])
            haufe_rows.extend(result.get("haufe_rows", []))
            if result.get("haufe_qc") is not None:
                qc_rows.append(result["haufe_qc"])
            for pos_row in result.get("channel_pos", []):
                ch = pos_row["channel"]
                if ch not in haufe_channel_pos:
                    haufe_channel_pos[ch] = np.array(
                        [pos_row["x"], pos_row["y"], pos_row["z"]], dtype=float
                    )
        write_progress("running", done, len(tasks))
        if (done % 5) == 0:
            elapsed = time.time() - t0
            print(
                f"[MVPA stimulus] complete {done}/{len(tasks)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    print(
        f"[MVPA stimulus] Starting time-resolved MVPA on {len(tasks)} sessions "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            handle_result(process_stim_mvpa_session(task), done)
    elif threadpool_limits is None:
        result_iter = Parallel(
            n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
        )(delayed(process_stim_mvpa_session)(task) for task in tasks)
        for done, result in enumerate(result_iter, start=1):
            handle_result(result, done)
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(
                n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
            )(delayed(process_stim_mvpa_session)(task) for task in tasks)
            for done, result in enumerate(result_iter, start=1):
                handle_result(result, done)

    session_df = pd.DataFrame(session_rows)
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)

    if session_df.empty:
        session_df.to_csv(session_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        pd.DataFrame().to_csv(subject_day_csv, index=False)
        pd.DataFrame().to_csv(day_means_csv, index=False)
        pd.DataFrame().to_csv(day_effect_csv, index=False)
        raise RuntimeError("MVPA stage produced no valid session rows.")

    subject_day_df = (
        session_df.groupby(["subject", "day", "time_sec"], as_index=False)["auc"]
        .mean()
        .sort_values(["subject", "day", "time_sec"])
    )
    day_means_df = (
        subject_day_df.groupby(["day", "time_sec"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sem=(
                "auc",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            ),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "time_sec"])
    )

    effect_rows = []
    for t, g in subject_day_df.groupby("time_sec"):
        if g["subject"].nunique() < 2 or g["day"].nunique() < 2:
            effect_rows.append(
                {
                    "time_sec": float(t),
                    "n_rows": int(len(g)),
                    "n_subjects": int(g["subject"].nunique()),
                    "day_coef": np.nan,
                    "day_se": np.nan,
                    "day_pvalue": np.nan,
                    "status": "insufficient_data",
                    "detail": "Need >=2 subjects and >=2 day values",
                }
            )
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            try:
                model = smf.mixedlm("auc ~ day", data=g, groups=g["subject"]).fit(
                    reml=False, method="lbfgs", disp=False
                )
                effect_rows.append(
                    {
                        "time_sec": float(t),
                        "n_rows": int(len(g)),
                        "n_subjects": int(g["subject"].nunique()),
                        "day_coef": float(model.params["day"]),
                        "day_se": float(model.bse["day"]),
                        "day_pvalue": float(model.pvalues["day"]),
                        "status": "ok",
                        "detail": "",
                    }
                )
            except Exception as exc:
                try:
                    model = smf.ols("auc ~ day", data=g).fit()
                    effect_rows.append(
                        {
                            "time_sec": float(t),
                            "n_rows": int(len(g)),
                            "n_subjects": int(g["subject"].nunique()),
                            "day_coef": float(model.params["day"]),
                            "day_se": float(model.bse["day"]),
                            "day_pvalue": float(model.pvalues["day"]),
                            "status": "ols_fallback",
                            "detail": str(exc),
                        }
                    )
                except Exception as exc2:
                    effect_rows.append(
                        {
                            "time_sec": float(t),
                            "n_rows": int(len(g)),
                            "n_subjects": int(g["subject"].nunique()),
                            "day_coef": np.nan,
                            "day_se": np.nan,
                            "day_pvalue": np.nan,
                            "status": "fit_error",
                            "detail": f"{exc}; fallback={exc2}",
                        }
                    )

    day_effect_df = pd.DataFrame(effect_rows).sort_values("time_sec")
    day_effect_df["p_fdr"] = np.nan
    day_effect_df["significant_fdr"] = False
    valid = day_effect_df["day_pvalue"].notna()
    if valid.any():
        rej, p_corr = fdrcorrection(day_effect_df.loc[valid, "day_pvalue"].values, alpha=0.05)
        day_effect_df.loc[valid, "p_fdr"] = p_corr
        day_effect_df.loc[valid, "significant_fdr"] = rej

    session_df.to_csv(session_csv, index=False)
    subject_day_df.to_csv(subject_day_csv, index=False)
    day_means_df.to_csv(day_means_csv, index=False)
    day_effect_df.to_csv(day_effect_csv, index=False)

    haufe_session_df = pd.DataFrame(haufe_rows)
    if haufe_session_df.empty:
        haufe_session_df.to_csv(haufe_session_csv, index=False)
        pd.DataFrame().to_csv(haufe_day_mean_csv, index=False)
        pd.DataFrame().to_csv(haufe_channel_pos_csv, index=False)
    else:
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
            [
                {"channel": ch, "x": xyz[0], "y": xyz[1], "z": xyz[2]}
                for ch, xyz in sorted(haufe_channel_pos.items())
            ]
        )
        haufe_session_df.to_csv(haufe_session_csv, index=False)
        haufe_day_mean_df.to_csv(haufe_day_mean_csv, index=False)
        haufe_pos_df.to_csv(haufe_channel_pos_csv, index=False)

    qc_df.to_csv(qc_csv, index=False)
    write_progress("completed", len(tasks), len(tasks))

    return {
        "session_df": session_df,
        "subject_day_df": subject_day_df,
        "day_means_df": day_means_df,
        "day_effect_df": day_effect_df,
        "qc_df": qc_df,
        "time_sec": np.array(sorted(session_df["time_sec"].unique())),
        "session_csv": session_csv,
        "subject_day_csv": subject_day_csv,
        "day_means_csv": day_means_csv,
        "day_effect_csv": day_effect_csv,
        "qc_csv": qc_csv,
    }


if __name__ == "__main__":
    run_mvpa_stim_locked_cat_time_resolved()
