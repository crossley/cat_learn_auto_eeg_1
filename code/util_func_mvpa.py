#!/usr/bin/env python3
"""MVPA utilities for time-resolved and temporal-generalization analyses."""

from pathlib import Path
import os
import time
import warnings
import json
import hashlib
from multiprocessing import cpu_count

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import ConvergenceWarning

os.environ["NUMBA_DISABLE_JIT"] = "1"
import mne
from mne.decoding import GeneralizingEstimator, cross_val_multiscore
from util_func_wrangle import util_wrangle_load_sessions
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

_CODE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CODE_DIR.parent
_OUTPUT_ROOT = _PROJECT_DIR / "output"
_FIGURES_ROOT = _PROJECT_DIR / "figures"


def _apply_single_thread_env_defaults():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")


def _default_n_workers():
    logical = max(1, int(cpu_count() or 1))
    physical_est = max(1, logical // 2)
    return max(1, physical_est - 2)


def _session_cache_key(session_item: dict):
    token = f"{session_item['subject']}_{session_item['day']}_{session_item['epo_file']}"
    return hashlib.md5(token.encode("utf-8")).hexdigest()[:12]


def _decode_timecourse(X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = Pipeline(
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


def util_mvpa_time_resolved(
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa",
    figures_dir: Path | str = _FIGURES_ROOT / "mvpa",
    min_epochs: int = 20,
    random_state: int = 42,
    save_figures: bool = True,
):
    """Compute per-session time-resolved MVPA and day-effect statistics."""
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )

    session_csv = output_dir / "mvpa_session_timecourse.csv"
    subject_day_csv = output_dir / "mvpa_subject_day_timecourse.csv"
    day_means_csv = output_dir / "mvpa_day_means_timecourse.csv"
    day_effect_csv = output_dir / "mvpa_day_effect_per_time.csv"
    qc_csv = output_dir / "mvpa_qc_log.csv"
    fig_day_panels = figures_dir / "mvpa_auc_by_day_panels.png"
    fig_day_slope = figures_dir / "mvpa_day_slope_timecourse.png"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    session_rows = []

    time_sec = None

    sessions = util_wrangle_load_sessions()
    for item in sessions:
        session_file = item["epo_file"]
        subject = item["subject"]
        day = item["day"]
        epochs = item["epochs"]

        stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
        if len(stim_events) < 2:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "event_select",
                    "reason": "missing_stim_labels",
                    "detail": ",".join(stim_events),
                }
            )
            continue

        try:
            stim_epochs = epochs[stim_events].copy()
            stim_epochs.load_data()
            stim_epochs.pick_types(eeg=True, exclude="bads")
            if len(stim_epochs.ch_names) == 0:
                raise RuntimeError("No EEG channels after pick_types.")
            stim_epochs.resample(128, npad="auto")
        except Exception as exc:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "preprocess",
                    "reason": "prep_error",
                    "detail": str(exc),
                }
            )
            continue

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
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "epoch_count",
                    "reason": "insufficient_epochs",
                    "detail": f"n_trials={n_trials} < min_epochs={min_epochs}",
                }
            )
            continue

        if min(n_a, n_b) < 5:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "class_balance",
                    "reason": "insufficient_class_trials",
                    "detail": f"n_a={n_a}, n_b={n_b}; need >=5 in each class",
                }
            )
            continue

        auc = _decode_timecourse(X, y, n_splits=5, random_state=random_state)
        t = stim_epochs.times.copy()
        if time_sec is None:
            time_sec = t

        for ti, auc_val in enumerate(auc):
            session_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "time_sec": float(t[ti]),
                    "auc": float(auc_val),
                    "n_trials": n_trials,
                    "n_a": n_a,
                    "n_b": n_b,
                }
            )

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
            auc_sem=("auc", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
                     if len(x) > 1 else np.nan),
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
                    reml=False,
                    method="lbfgs",
                    disp=False,
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
                continue
            except Exception as exc:
                mixed_err = str(exc)

            try:
                ols = smf.ols("auc ~ day", data=g).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": g["subject"]},
                )
                effect_rows.append(
                    {
                        "time_sec": float(t),
                        "n_rows": int(len(g)),
                        "n_subjects": int(g["subject"].nunique()),
                        "day_coef": float(ols.params["day"]),
                        "day_se": float(ols.bse["day"]),
                        "day_pvalue": float(ols.pvalues["day"]),
                        "status": "ols_fallback",
                        "detail": f"mixedlm_error={mixed_err}",
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
                        "status": "model_error",
                        "detail": f"mixedlm_error={mixed_err}; ols_error={exc2}",
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
    qc_df.to_csv(qc_csv, index=False)

    # Plot from on-disk outputs (two-step pattern: compute/write -> read/plot).
    day_means_plot_df = pd.read_csv(day_means_csv)
    day_effect_plot_df = pd.read_csv(day_effect_csv)

    if not save_figures:
        print("Skipped MVPA figure generation (save_figures=False).")
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
            "figure_paths": {},
        }

    # Figure 1: one panel per day with mean+-SEM AUC over time
    days = sorted(day_means_plot_df["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 3.8), sharey=True, squeeze=False)
    for ax, day in zip(axes.ravel(), days):
        g = day_means_plot_df[day_means_plot_df["day"] == day].sort_values("time_sec")
        x = g["time_sec"].to_numpy()
        y = g["auc_mean"].to_numpy()
        s = g["auc_sem"].to_numpy()
        ax.plot(x, y, color="tab:blue", linewidth=2)
        ax.fill_between(x, y - s, y + s, color="tab:blue", alpha=0.2, linewidth=0)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("ROC-AUC")
    fig.suptitle("Time-resolved Category Decoding (Stim/A vs Stim/B)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_day_panels, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: day-effect slope over time with FDR markers
    g = day_effect_plot_df.sort_values("time_sec")
    x = g["time_sec"].to_numpy()
    y = g["day_coef"].to_numpy()
    sig = g["significant_fdr"].to_numpy(dtype=bool)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="tab:green", linewidth=2, label="Day slope (AUC ~ day)")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    if np.any(sig):
        ax.scatter(x[sig], y[sig], color="red", s=16, label="FDR < 0.05", zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Day coefficient")
    ax.set_title("Day Effect on Decoding Over Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_day_slope, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Wrote MVPA session table: {session_csv}")
    print(f"Wrote MVPA subject-day table: {subject_day_csv}")
    print(f"Wrote MVPA day means table: {day_means_csv}")
    print(f"Wrote MVPA day-effect table: {day_effect_csv}")
    print(f"Wrote MVPA QC log: {qc_csv}")
    print(f"Saved MVPA figures: 2")
    print(f"- day panels: {fig_day_panels}")
    print(f"- day slope: {fig_day_slope}")

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
        "figure_paths": {
            "day_panels": fig_day_panels,
            "day_slope": fig_day_slope,
        },
    }

def _build_clf(random_state: int):
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


def _prepare_stim_data(epochs):
    stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
    if len(stim_events) < 2:
        raise ValueError(f"missing_stim_labels:{','.join(stim_events)}")

    stim_epochs = epochs[stim_events].copy()
    stim_epochs.load_data()
    stim_epochs.pick_types(eeg=True, exclude="bads")
    if len(stim_epochs.ch_names) == 0:
        raise RuntimeError("no_eeg_channels_after_pick")
    stim_epochs.resample(128, npad="auto")

    codes = stim_epochs.events[:, 2]
    y = np.full(len(codes), -1, dtype=int)
    y[codes == stim_epochs.event_id["Stim/A"]] = 0
    y[codes == stim_epochs.event_id["Stim/B"]] = 1
    keep = y >= 0

    X = stim_epochs.get_data()[keep]
    y = y[keep]
    t = stim_epochs.times.copy()

    return X, y, t


def _prepare_session_cache(session_item: dict, cache_dir: Path):
    session_file = session_item["epo_file"]
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    epochs = session_item["epochs"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"stim_cache_{_session_cache_key(session_item)}.npz"

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            t = z["t"]
            y = z["y"]
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
        }

    try:
        X, y, t = _prepare_stim_data(epochs)
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

    np.savez_compressed(cache_path, X=X, y=y, t=t)
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
    }


def _balanced_day_subset(X, y, n_per_class: int, rng: np.random.Generator):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    pick0 = rng.choice(idx0, size=n_per_class, replace=False)
    pick1 = rng.choice(idx1, size=n_per_class, replace=False)
    idx = np.concatenate([pick0, pick1])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _process_within_day_session(
    session_meta: dict,
    min_epochs: int,
    random_state: int,
):
    session_file = session_meta["session_file"]
    subject = int(session_meta["subject"])
    day = int(session_meta["day"])
    cache_path = session_meta["cache_path"]

    with np.load(cache_path, allow_pickle=False) as z:
        X = z["X"]
        y = z["y"]
        t = z["t"]

    if len(y) < min_epochs:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "epoch_count",
                "reason": "insufficient_epochs",
                "detail": f"n_trials={len(y)} < min_epochs={min_epochs}",
            },
        }

    if min(np.sum(y == 0), np.sum(y == 1)) < 5:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "class_balance",
                "reason": "insufficient_class_trials",
                "detail": f"n_a={int(np.sum(y==0))}, n_b={int(np.sum(y==1))}",
            },
        }

    clf = _build_clf(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    ge = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=1, verbose=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            scores = cross_val_multiscore(ge, X, y, cv=cv, n_jobs=1)
        mat = np.nanmean(scores, axis=0)
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "within_day_tg",
                "reason": "compute_error",
                "detail": str(exc),
            },
        }

    return {
        "ok": True,
        "session_file": session_file,
        "subject": subject,
        "day": day,
        "cache_path": cache_path,
        "X": X,
        "y": y,
        "t": t,
        "mat": mat,
    }


def _process_cross_day_pair(
    pair_item: dict,
    random_state: int,
):
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
    with np.load(test_cache_path, allow_pickle=False) as z:
        X_test_all = z["X"]
        y_test_all = z["y"]

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
    X_train, y_train = _balanced_day_subset(X_train_all, y_train_all, n_per_class=n_per_class, rng=rng_pair)
    X_test, y_test = _balanced_day_subset(X_test_all, y_test_all, n_per_class=n_per_class, rng=rng_pair)

    clf = _build_clf(random_state=random_state)
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
    }


def util_mvpa_temporal_generalization(
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa_tg",
    figures_dir: Path | str = _FIGURES_ROOT / "mvpa_tg",
    min_epochs: int = 20,
    random_state: int = 42,
    progress_every: int = 5,
    n_workers: int | None = None,
    save_figures: bool = True,
    run_within_day: bool = True,
    run_cross_day: bool = True,
):
    """Compute within-day and cross-day temporal generalization decoding."""
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    _apply_single_thread_env_defaults()

    within_subject_csv = output_dir / "tg_within_day_subject_level.csv"
    within_day_mean_csv = output_dir / "tg_within_day_day_mean.csv"
    cross_subject_csv = output_dir / "tg_cross_day_subject_level.csv"
    cross_day_mean_csv = output_dir / "tg_cross_day_day_mean.csv"
    qc_csv = output_dir / "tg_qc_log.csv"
    progress_json = output_dir / "tg_progress.json"
    fig_within = figures_dir / "tg_within_day_heatmaps.png"
    fig_cross = figures_dir / "tg_cross_day_transfer_5x4.png"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    t0 = time.time()
    wrote_within_subject = False
    wrote_cross_subject = False
    wrote_qc = False

    def _append_csv(df: pd.DataFrame, path: Path, wrote_flag: bool):
        if df.empty:
            return wrote_flag
        df.to_csv(path, mode="a", header=not wrote_flag, index=False)
        return True

    def _write_progress(stage: str, within_done: int, within_total: int, cross_done: int, cross_total: int):
        elapsed = time.time() - t0
        rate = (cross_done / elapsed * 60.0) if elapsed > 0 else 0.0
        eta_min = ((cross_total - cross_done) / (rate / 60.0) / 60.0) if rate > 0 and cross_total > 0 else None
        payload = {
            "stage": stage,
            "elapsed_sec": elapsed,
            "within_done": int(within_done),
            "within_total": int(within_total),
            "cross_done": int(cross_done),
            "cross_total": int(cross_total),
            "cross_pairs_per_min": float(rate),
            "eta_min": None if eta_min is None else float(eta_min),
            "updated_unix": time.time(),
        }
        with open(progress_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    day_data = {}
    within_mats = []
    time_template = None

    rng_master = np.random.default_rng(random_state)

    session_items = util_wrangle_load_sessions()
    cache_dir = output_dir / "cache_stim_arrays"
    cache_results = [_prepare_session_cache(item, cache_dir=cache_dir) for item in session_items]
    prepared_items = []
    for result in cache_results:
        if not result["ok"]:
            qc_rows.append(result["qc"])
        else:
            prepared_items.append(result)
    if qc_rows:
        wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))
    within_results = []
    if run_within_day:
        print(
            f"[TG] Starting within-day TG on {len(prepared_items)} prepared sessions "
            f"(n_workers={n_workers})...",
            flush=True,
        )
        _write_progress("within_day_running", 0, len(prepared_items), 0, 0)
        if len(prepared_items) == 0:
            within_results = []
        elif n_workers == 1:
            if threadpool_limits is None:
                within_results = [
                    _process_within_day_session(
                        session_meta=item,
                        min_epochs=min_epochs,
                        random_state=random_state,
                    )
                    for item in prepared_items
                ]
            else:
                with threadpool_limits(limits=1):
                    within_results = [
                        _process_within_day_session(
                            session_meta=item,
                            min_epochs=min_epochs,
                            random_state=random_state,
                        )
                        for item in prepared_items
                    ]
        else:
            if threadpool_limits is None:
                within_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                    delayed(_process_within_day_session)(
                        session_meta=item,
                        min_epochs=min_epochs,
                        random_state=random_state,
                    )
                    for item in prepared_items
                )
            else:
                with threadpool_limits(limits=1):
                    within_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                        delayed(_process_within_day_session)(
                            session_meta=item,
                            min_epochs=min_epochs,
                            random_state=random_state,
                        )
                        for item in prepared_items
                    )

    n_done = 0
    for result in within_results:
        if not result["ok"]:
            qc_rows.append(result["qc"])
            if len(qc_rows) >= max(progress_every, 1):
                wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
                qc_rows = []
            continue

        subject = int(result["subject"])
        day = int(result["day"])
        X = result["X"]
        y = result["y"]
        t = result["t"]
        mat = result["mat"]
        session_file = result["session_file"]

        if time_template is None:
            time_template = t
        elif (len(t) != len(time_template)) or (not np.allclose(t, time_template, atol=1e-9)):
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "time_grid",
                    "reason": "inconsistent_time_axis",
                    "detail": "",
                }
            )
            continue

        within_mats.append(
            {
                "subject": subject,
                "day": day,
                "session_file": session_file,
                "mat": mat,
                "n_trials": int(len(y)),
                "n_a": int(np.sum(y == 0)),
                "n_b": int(np.sum(y == 1)),
            }
        )
        day_data[(subject, day)] = {
            "cache_path": result["cache_path"],
            "session_file": session_file,
        }
        # Incrementally append within-day subject-level rows.
        n_t_local = len(t)
        rows_local = []
        for i in range(n_t_local):
            for j in range(n_t_local):
                rows_local.append(
                    {
                        "subject": subject,
                        "day": day,
                        "session_file": session_file,
                        "n_trials": int(len(y)),
                        "n_a": int(np.sum(y == 0)),
                        "n_b": int(np.sum(y == 1)),
                        "train_time_sec": float(t[i]),
                        "test_time_sec": float(t[j]),
                        "auc": float(mat[i, j]),
                    }
                )
        wrote_within_subject = _append_csv(pd.DataFrame(rows_local), within_subject_csv, wrote_within_subject)
        n_done += 1
        _write_progress("within_day_running", n_done, len(prepared_items), 0, 0)
        if (n_done % max(progress_every, 1)) == 0:
            elapsed = time.time() - t0
            print(
                f"[TG] within-day complete {n_done}/{len(prepared_items)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    if qc_rows:
        wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []
    qc_df = pd.read_csv(qc_csv) if qc_csv.exists() else pd.DataFrame(columns=qc_columns)

    if run_within_day and not within_mats:
        pd.DataFrame().to_csv(within_subject_csv, index=False)
        pd.DataFrame().to_csv(within_day_mean_csv, index=False)
        if run_cross_day:
            pd.DataFrame().to_csv(cross_subject_csv, index=False)
            pd.DataFrame().to_csv(cross_day_mean_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        raise RuntimeError("No valid within-day TG matrices were computed.")

    if run_within_day:
        # Flatten within-day subject-level matrices (for in-memory return/final means).
        within_rows = []
        n_t = len(time_template)
        for item in within_mats:
            mat = item["mat"]
            for i in range(n_t):
                for j in range(n_t):
                    within_rows.append(
                        {
                            "subject": item["subject"],
                            "day": item["day"],
                            "session_file": item["session_file"],
                            "n_trials": item["n_trials"],
                            "n_a": item["n_a"],
                            "n_b": item["n_b"],
                            "train_time_sec": float(time_template[i]),
                            "test_time_sec": float(time_template[j]),
                            "auc": float(mat[i, j]),
                        }
                    )
        within_subject_df = pd.DataFrame(within_rows)
        within_day_mean_df = (
            within_subject_df.groupby(["day", "train_time_sec", "test_time_sec"], as_index=False)
            .agg(
                auc_mean=("auc", "mean"),
                auc_sem=(
                    "auc",
                    lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
                ),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["day", "train_time_sec", "test_time_sec"])
        )
        within_day_mean_df.to_csv(within_day_mean_csv, index=False)
    else:
        within_subject_df = (
            pd.read_csv(within_subject_csv) if within_subject_csv.exists() else pd.DataFrame()
        )
        within_day_mean_df = (
            pd.read_csv(within_day_mean_csv) if within_day_mean_csv.exists() else pd.DataFrame()
        )

    if run_cross_day:
        print(
            f"[TG] Starting cross-day transfer on {len(sorted({k[0] for k in day_data}))} subjects...",
            flush=True,
        )

    # Cross-day transfer (within-subject): train day -> test other day.
    cross_rows = []
    cross_total = 0
    pair_items = []
    if run_cross_day:
        subjects = sorted({k[0] for k in day_data})
        for subject in subjects:
            subject_days = sorted([k[1] for k in day_data if k[0] == subject])
            if len(subject_days) < 2:
                continue
            print(f"[TG] cross-day subject {subject} with days {subject_days}", flush=True)
            for d_train in subject_days:
                for d_test in subject_days:
                    if d_test == d_train:
                        continue
                    train_item = day_data[(subject, d_train)]
                    test_item = day_data[(subject, d_test)]
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
        cross_total = len(pair_items)

    if run_cross_day and cross_total > 0:
        if n_workers == 1:
            if threadpool_limits is None:
                cross_results = [
                    _process_cross_day_pair(pair_item=item, random_state=random_state)
                    for item in pair_items
                ]
            else:
                with threadpool_limits(limits=1):
                    cross_results = [
                        _process_cross_day_pair(pair_item=item, random_state=random_state)
                        for item in pair_items
                    ]
        else:
            if threadpool_limits is None:
                cross_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                    delayed(_process_cross_day_pair)(pair_item=item, random_state=random_state)
                    for item in pair_items
                )
            else:
                with threadpool_limits(limits=1):
                    cross_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                        delayed(_process_cross_day_pair)(pair_item=item, random_state=random_state)
                        for item in pair_items
                    )
    elif run_cross_day:
        cross_results = []
    else:
        cross_results = []

    cross_done = 0
    for result in cross_results:
        if result["ok"]:
            cross_rows.append(result["row"])
            cross_done += 1
            wrote_cross_subject = _append_csv(pd.DataFrame([result["row"]]), cross_subject_csv, wrote_cross_subject)
        else:
            qc_rows.append(result["qc"])
            if len(qc_rows) >= max(progress_every, 1):
                wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
                qc_rows = []
        _write_progress("cross_day_running", n_done, len(prepared_items), cross_done, cross_total)
        if (cross_done % max(progress_every * 2, 1)) == 0 and cross_done > 0:
            elapsed = time.time() - t0
            print(
                f"[TG] cross-day complete {cross_done}/{cross_total} pairs "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )
    if qc_rows:
        wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    if run_cross_day:
        cross_subject_df = pd.DataFrame(cross_rows)
        if cross_subject_df.empty:
            pd.DataFrame().to_csv(cross_day_mean_csv, index=False)
            cross_day_mean_df = pd.DataFrame()
        else:
            cross_day_mean_df = (
                cross_subject_df.groupby(["train_day", "test_day"], as_index=False)
                .agg(
                    auc_mean=("diag_mean_auc", "mean"),
                    auc_sem=(
                        "diag_mean_auc",
                        lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
                    ),
                    n_subjects=("subject", "nunique"),
                )
                .sort_values(["train_day", "test_day"])
            )
    else:
        cross_subject_df = pd.read_csv(cross_subject_csv) if cross_subject_csv.exists() else pd.DataFrame()
        cross_day_mean_df = pd.read_csv(cross_day_mean_csv) if cross_day_mean_csv.exists() else pd.DataFrame()

    # Save final derived tables.
    if run_within_day and not within_subject_csv.exists():
        within_subject_df.to_csv(within_subject_csv, index=False)
    if run_cross_day and not cross_subject_csv.exists():
        cross_subject_df.to_csv(cross_subject_csv, index=False)
    if run_within_day:
        within_day_mean_df.to_csv(within_day_mean_csv, index=False)
    if run_cross_day:
        cross_day_mean_df.to_csv(cross_day_mean_csv, index=False)
    qc_df = pd.read_csv(qc_csv) if qc_csv.exists() else pd.DataFrame(columns=qc_columns)
    _write_progress("completed", n_done, len(prepared_items), cross_done, cross_total)

    if not save_figures:
        print("Skipped TG figure generation (save_figures=False).")
        return {
            "within_subject_df": within_subject_df,
            "within_day_mean_df": within_day_mean_df,
            "cross_subject_df": cross_subject_df,
            "cross_day_mean_df": cross_day_mean_df,
            "qc_df": qc_df,
            "time_sec": np.array(time_template, dtype=float),
            "within_subject_csv": within_subject_csv,
            "within_day_mean_csv": within_day_mean_csv,
            "cross_subject_csv": cross_subject_csv,
            "cross_day_mean_csv": cross_day_mean_csv,
            "qc_csv": qc_csv,
            "figure_paths": {},
        }

    # Plot imports are intentionally lazy to avoid startup stalls from
    # font-cache initialization before compute begins.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Plot from on-disk outputs (two-step pattern: compute/write -> read/plot).
    within_day_plot_df = pd.read_csv(within_day_mean_csv) if within_day_mean_csv.exists() else pd.DataFrame()
    cross_day_plot_df = pd.read_csv(cross_day_mean_csv) if cross_day_mean_csv.exists() else pd.DataFrame()

    # Figure: within-day heatmaps (one panel per day).
    if run_within_day and (not within_day_plot_df.empty):
        days = sorted(within_day_plot_df["day"].unique())
        fig, axes = plt.subplots(1, len(days), figsize=(4.6 * len(days), 4), squeeze=False)
        vmin = float(within_day_plot_df["auc_mean"].min())
        vmax = float(within_day_plot_df["auc_mean"].max())
        for ax, day in zip(axes.ravel(), days):
            g = within_day_plot_df[within_day_plot_df["day"] == day]
            pivot = g.pivot(index="train_time_sec", columns="test_time_sec", values="auc_mean")
            mat = pivot.to_numpy()
            im = ax.imshow(
                mat,
                origin="lower",
                aspect="auto",
                extent=[
                    float(pivot.columns.min()),
                    float(pivot.columns.max()),
                    float(pivot.index.min()),
                    float(pivot.index.max()),
                ],
                vmin=vmin,
                vmax=vmax,
                cmap="viridis",
            )
            ax.axvline(0.0, color="white", linestyle=":", linewidth=1)
            ax.axhline(0.0, color="white", linestyle=":", linewidth=1)
            ax.set_title(f"Day {day}")
            ax.set_xlabel("Test Time (s)")
            ax.set_ylabel("Train Time (s)")
        fig.suptitle("Within-Day Temporal Generalization (AUC)")
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="AUC")
        fig.subplots_adjust(top=0.85, wspace=0.28)
        fig.savefig(fig_within, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # Figure: cross-day transfer heatmap (off-diagonal day transfer).
    if run_cross_day:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        day_grid = sorted({1, 2, 3, 4, 5})
        mat = np.full((len(day_grid), len(day_grid)), np.nan)
        if not cross_day_plot_df.empty:
            for _, r in cross_day_plot_df.iterrows():
                i = day_grid.index(int(r["train_day"]))
                j = day_grid.index(int(r["test_day"]))
                mat[i, j] = float(r["auc_mean"])
        masked = np.ma.masked_invalid(mat)
        im = ax.imshow(masked, cmap="magma", aspect="equal")
        ax.set_xticks(range(len(day_grid)))
        ax.set_yticks(range(len(day_grid)))
        ax.set_xticklabels([f"D{d}" for d in day_grid])
        ax.set_yticklabels([f"D{d}" for d in day_grid])
        ax.set_xlabel("Test Day")
        ax.set_ylabel("Train Day")
        ax.set_title("Cross-Day Transfer (Diagonal Mean AUC)")
        for i in range(len(day_grid)):
            for j in range(len(day_grid)):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
                elif i == j:
                    ax.text(j, i, "—", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, shrink=0.9, label="AUC")
        fig.tight_layout()
        fig.savefig(fig_cross, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"Wrote TG within-day subject table: {within_subject_csv}")
    print(f"Wrote TG within-day day-mean table: {within_day_mean_csv}")
    print(f"Wrote TG cross-day subject table: {cross_subject_csv}")
    print(f"Wrote TG cross-day day-mean table: {cross_day_mean_csv}")
    print(f"Wrote TG QC log: {qc_csv}")
    print(f"Saved TG figures: 2")
    print(f"- within-day heatmaps: {fig_within}")
    print(f"- cross-day transfer: {fig_cross}")
    elapsed = time.time() - t0
    print(f"[TG] Done in {elapsed/60:.1f} min", flush=True)

    return {
        "within_subject_df": within_subject_df,
        "within_day_mean_df": within_day_mean_df,
        "cross_subject_df": cross_subject_df,
        "cross_day_mean_df": cross_day_mean_df,
        "qc_df": qc_df,
        "time_sec": np.array(time_template, dtype=float),
        "within_subject_csv": within_subject_csv,
        "within_day_mean_csv": within_day_mean_csv,
        "cross_subject_csv": cross_subject_csv,
        "cross_day_mean_csv": cross_day_mean_csv,
        "qc_csv": qc_csv,
        "figure_paths": {
            "within_day_heatmaps": fig_within,
            "cross_day_transfer": fig_cross,
        },
    }


def run_mvpa_time_resolved(**kwargs):
    """Run time-resolved MVPA analysis."""
    return util_mvpa_time_resolved(save_figures=False, **kwargs)


def save_fig_mvpa_time_resolved(**kwargs):
    """Generate time-resolved MVPA figures."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa"))
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    day_means_csv = output_dir / "mvpa_day_means_timecourse.csv"
    day_effect_csv = output_dir / "mvpa_day_effect_per_time.csv"
    if (not day_means_csv.exists()) or (not day_effect_csv.exists()):
        raise FileNotFoundError(f"Missing MVPA outputs in {output_dir}. Run run_mvpa_time_resolved() first.")

    day_means_df = pd.read_csv(day_means_csv)
    day_effect_df = pd.read_csv(day_effect_csv)
    fig_day_panels = figures_dir / "mvpa_auc_by_day_panels.png"
    fig_day_slope = figures_dir / "mvpa_day_slope_timecourse.png"

    days = sorted(day_means_df["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 3.8), sharey=True, squeeze=False)
    for ax, day in zip(axes.ravel(), days):
        g = day_means_df[day_means_df["day"] == day].sort_values("time_sec")
        x = g["time_sec"].to_numpy()
        y = g["auc_mean"].to_numpy()
        s = g["auc_sem"].to_numpy()
        ax.plot(x, y, color="tab:blue", linewidth=2)
        ax.fill_between(x, y - s, y + s, color="tab:blue", alpha=0.2, linewidth=0)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("ROC-AUC")
    fig.suptitle("Time-resolved Category Decoding (Stim/A vs Stim/B)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_day_panels, dpi=150, bbox_inches="tight")
    plt.close(fig)

    g = day_effect_df.sort_values("time_sec")
    x = g["time_sec"].to_numpy()
    y = g["day_coef"].to_numpy()
    sig = g["significant_fdr"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="tab:green", linewidth=2, label="Day slope (AUC ~ day)")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    if np.any(sig):
        ax.scatter(x[sig], y[sig], color="red", s=16, label="FDR < 0.05", zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Day coefficient")
    ax.set_title("Day Effect on Decoding Over Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_day_slope, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure_paths": {"day_panels": fig_day_panels, "day_slope": fig_day_slope}}


def run_mvpa_temporal_generalization(**kwargs):
    """Run temporal-generalization MVPA analysis."""
    return util_mvpa_temporal_generalization(save_figures=False, **kwargs)


def save_fig_mvpa_temporal_generalization(**kwargs):
    """Generate temporal-generalization MVPA figures."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa_tg"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa_tg"))
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    within_day_mean_csv = output_dir / "tg_within_day_day_mean.csv"
    cross_day_mean_csv = output_dir / "tg_cross_day_day_mean.csv"
    if (not within_day_mean_csv.exists()) or (not cross_day_mean_csv.exists()):
        raise FileNotFoundError(
            f"Missing TG outputs in {output_dir}. Run run_mvpa_temporal_generalization() first."
        )
    out_within = save_fig_mvpa_temporal_generalization_within_day(
        output_dir=output_dir, figures_dir=figures_dir
    )
    out_cross = save_fig_mvpa_temporal_generalization_cross_day(
        output_dir=output_dir, figures_dir=figures_dir
    )
    return {
        "figure_paths": {
            "within_day_heatmaps": out_within["figure_path"],
            "cross_day_transfer": out_cross["figure_path"],
        }
    }


def save_fig_mvpa_temporal_generalization_within_day(**kwargs):
    """Generate only time x time TG (within-day) figure from saved outputs."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa_tg"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa_tg"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    within_day_mean_csv = output_dir / "tg_within_day_day_mean.csv"
    if not within_day_mean_csv.exists():
        raise FileNotFoundError(
            f"Missing TG within-day output in {output_dir}. Run run_mvpa_temporal_generalization() first."
        )
    within_day_mean_df = pd.read_csv(within_day_mean_csv)
    fig_within = figures_dir / "tg_within_day_heatmaps.png"

    days = sorted(within_day_mean_df["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(4.6 * len(days), 4), squeeze=False)
    vmin = float(within_day_mean_df["auc_mean"].min())
    vmax = float(within_day_mean_df["auc_mean"].max())
    for ax, day in zip(axes.ravel(), days):
        g = within_day_mean_df[within_day_mean_df["day"] == day]
        pivot = g.pivot(index="train_time_sec", columns="test_time_sec", values="auc_mean")
        mat = pivot.to_numpy()
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.axvline(0.0, color="white", linestyle=":", linewidth=1)
        ax.axhline(0.0, color="white", linestyle=":", linewidth=1)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Test Time (s)")
        ax.set_ylabel("Train Time (s)")
    fig.suptitle("Within-Day Temporal Generalization (AUC)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="AUC")
    fig.subplots_adjust(top=0.85, wspace=0.28)
    fig.savefig(fig_within, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure_path": fig_within}


def save_fig_mvpa_temporal_generalization_cross_day(**kwargs):
    """Generate only day x day TG transfer figure from saved outputs."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa_tg"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa_tg"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    cross_day_mean_csv = output_dir / "tg_cross_day_day_mean.csv"
    if not cross_day_mean_csv.exists():
        raise FileNotFoundError(
            f"Missing TG cross-day output in {output_dir}. Run run_mvpa_temporal_generalization() first."
        )
    cross_day_mean_df = pd.read_csv(cross_day_mean_csv)
    fig_cross = figures_dir / "tg_cross_day_transfer_5x4.png"

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    day_grid = sorted({1, 2, 3, 4, 5})
    mat = np.full((len(day_grid), len(day_grid)), np.nan)
    if not cross_day_mean_df.empty:
        for _, r in cross_day_mean_df.iterrows():
            i = day_grid.index(int(r["train_day"]))
            j = day_grid.index(int(r["test_day"]))
            mat[i, j] = float(r["auc_mean"])
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="magma", aspect="equal")
    ax.set_xticks(range(len(day_grid)))
    ax.set_yticks(range(len(day_grid)))
    ax.set_xticklabels([f"D{d}" for d in day_grid])
    ax.set_yticklabels([f"D{d}" for d in day_grid])
    ax.set_xlabel("Test Day")
    ax.set_ylabel("Train Day")
    ax.set_title("Cross-Day Transfer (Diagonal Mean AUC)")
    for i in range(len(day_grid)):
        for j in range(len(day_grid)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
            elif i == j:
                ax.text(j, i, "—", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.9, label="AUC")
    fig.tight_layout()
    fig.savefig(fig_cross, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure_path": fig_cross}


def run_mvpa_temporal_generalization_within_day(**kwargs):
    """Run only within-day time x time temporal-generalization analysis."""
    return util_mvpa_temporal_generalization(
        save_figures=False,
        run_within_day=True,
        run_cross_day=False,
        **kwargs,
    )


def run_mvpa_temporal_generalization_cross_day(**kwargs):
    """Run only cross-day day x day temporal-generalization transfer analysis."""
    return util_mvpa_temporal_generalization(
        save_figures=False,
        run_within_day=True,
        run_cross_day=True,
        **kwargs,
    )


def _extract_stim_channel_meta(epochs):
    stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
    if len(stim_events) < 2:
        raise ValueError(f"missing_stim_labels:{','.join(stim_events)}")
    stim_epochs = epochs[stim_events].copy()
    stim_epochs.load_data()
    stim_epochs.pick_types(eeg=True, exclude="bads")
    if len(stim_epochs.ch_names) == 0:
        raise RuntimeError("no_eeg_channels_after_pick")
    stim_epochs.resample(128, npad="auto")
    info = stim_epochs.info.copy()
    ch_names = list(stim_epochs.ch_names)
    times = stim_epochs.times.copy()
    pos = np.array([info["chs"][info.ch_names.index(ch)]["loc"][:3] for ch in ch_names], dtype=float)
    return {
        "ch_names": ch_names,
        "times": times,
        "pos": pos,
    }


def _compute_haufe_patterns_from_xy(X, y, random_state: int):
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
            clf = _build_clf(random_state=random_state)
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


def _process_haufe_session_item(task: dict):
    item = task["session_item"]
    cache_dir = Path(task["cache_dir"])
    min_epochs = int(task["min_epochs"])
    random_state = int(task["random_state"])
    session_file = item["epo_file"]
    subject = int(item["subject"])
    day = int(item["day"])
    cache_path = cache_dir / f"stim_cache_{_session_cache_key(item)}.npz"
    if not cache_path.exists():
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "cache",
                "reason": "missing_cache",
                "detail": str(cache_path),
            },
        }
    try:
        meta = {
            "ch_names": list(task["ch_names"]),
            "pos": np.array(task["pos"], dtype=float),
        }
        with np.load(cache_path, allow_pickle=False) as z:
            X = z["X"]
            y = z["y"]
            t = z["t"]
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "load",
                "reason": "load_error",
                "detail": str(exc),
            },
        }
    if len(y) < min_epochs:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "epoch_count",
                "reason": "insufficient_epochs",
                "detail": f"n_trials={len(y)} < min_epochs={min_epochs}",
            },
        }
    n_a = int(np.sum(y == 0))
    n_b = int(np.sum(y == 1))
    if min(n_a, n_b) < 5:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "class_balance",
                "reason": "insufficient_class_trials",
                "detail": f"n_a={n_a}, n_b={n_b}",
            },
        }
    patterns = _compute_haufe_patterns_from_xy(X, y, random_state=random_state)
    return {
        "ok": True,
        "subject": subject,
        "day": day,
        "session_file": session_file,
        "ch_names": meta["ch_names"],
        "pos": meta["pos"],
        "t": t,
        "patterns": patterns,
        "n_trials": int(len(y)),
        "n_a": n_a,
        "n_b": n_b,
    }


def _process_haufe_cross_pair_item(task: dict):
    subject = int(task["subject"])
    d_train = int(task["train_day"])
    d_test = int(task["test_day"])
    tr_item = task["train_item"]
    te_item = task["test_item"]
    min_epochs = int(task["min_epochs"])
    random_state = int(task["random_state"])
    try:
        meta = {
            "ch_names": list(task["ch_names"]),
            "pos": np.array(task["pos"], dtype=float),
        }
        with np.load(tr_item["cache_path"], allow_pickle=False) as z:
            X_train_all = z["X"]
            y_train_all = z["y"]
            t = z["t"]
        with np.load(te_item["cache_path"], allow_pickle=False) as z:
            X_test_all = z["X"]
            y_test_all = z["y"]
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": f"{tr_item['session_file']}->{te_item['session_file']}",
                "subject": subject,
                "day": d_train,
                "stage": "load",
                "reason": "load_error",
                "detail": str(exc),
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
    if (len(y_train_all) < min_epochs) or (n_per_class < 5):
        return {
            "ok": False,
            "qc": {
                "session_file": f"{tr_item['session_file']}->{te_item['session_file']}",
                "subject": subject,
                "day": d_train,
                "stage": "cross_day_balance",
                "reason": "insufficient_trials",
                "detail": f"len_train={len(y_train_all)}, n_per_class={n_per_class}",
            },
        }
    rng_pair = np.random.default_rng(int(task["pair_seed"]))
    X_train, y_train = _balanced_day_subset(X_train_all, y_train_all, n_per_class=n_per_class, rng=rng_pair)
    patterns = _compute_haufe_patterns_from_xy(X_train, y_train, random_state=random_state)
    return {
        "ok": True,
        "subject": subject,
        "train_day": d_train,
        "test_day": d_test,
        "train_session_file": tr_item["session_file"],
        "test_session_file": te_item["session_file"],
        "ch_names": meta["ch_names"],
        "pos": meta["pos"],
        "t": t,
        "patterns": patterns,
        "n_per_class": int(n_per_class),
    }


def run_mvpa_haufe_patterns(
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa_haufe",
    cache_dir: Path | str = _OUTPUT_ROOT / "mvpa_tg" / "cache_stim_arrays",
    min_epochs: int = 20,
    random_state: int = 42,
    n_workers: int | None = None,
):
    """Compute time-resolved Haufe activation patterns from cached stim arrays."""
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )

    session_csv = output_dir / "haufe_session_channel_time.csv"
    day_mean_csv = output_dir / "haufe_day_mean_channel_time.csv"
    day_abs_mean_csv = output_dir / "haufe_day_abs_mean_channel_time.csv"
    channel_pos_csv = output_dir / "haufe_channel_positions.csv"
    qc_csv = output_dir / "haufe_qc_log.csv"
    progress_json = output_dir / "haufe_progress.json"

    _apply_single_thread_env_defaults()
    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    rows = []
    channel_pos = {}
    t0 = time.time()

    def _write_progress(stage: str, done: int, total: int):
        payload = {
            "stage": stage,
            "total": int(total),
            "done": int(done),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))
    sessions = util_wrangle_load_sessions()
    tasks = []
    for item in sessions:
        lean_item = {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "epo_file": item["epo_file"],
        }
        try:
            meta = _extract_stim_channel_meta(item["epochs"])
            ch_names = meta["ch_names"]
            pos = meta["pos"]
        except Exception:
            ch_names = []
            pos = np.empty((0, 3), dtype=float)
        tasks.append(
            {
                "session_item": lean_item,
                "cache_dir": str(cache_dir),
                "min_epochs": int(min_epochs),
                "random_state": int(random_state),
                "ch_names": ch_names,
                "pos": pos,
            }
        )
    _write_progress("running", 0, len(tasks))
    if n_workers == 1:
        results = [_process_haufe_session_item(t) for t in tasks]
    else:
        if threadpool_limits is None:
            results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                delayed(_process_haufe_session_item)(t) for t in tasks
            )
        else:
            with threadpool_limits(limits=1):
                results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                    delayed(_process_haufe_session_item)(t) for t in tasks
                )

    done = 0
    for result in results:
        done += 1
        if not result["ok"]:
            qc_rows.append(result["qc"])
            _write_progress("running", done, len(tasks))
            continue
        ch_names = result["ch_names"]
        pos = result["pos"]
        for i, ch in enumerate(ch_names):
            if ch not in channel_pos:
                channel_pos[ch] = pos[i]
        t = result["t"]
        patterns = result["patterns"]
        for ci, ch in enumerate(ch_names):
            for ti, tsec in enumerate(t):
                rows.append(
                    {
                        "subject": int(result["subject"]),
                        "day": int(result["day"]),
                        "session_file": result["session_file"],
                        "channel": ch,
                        "time_sec": float(tsec),
                        "pattern": float(patterns[ci, ti]),
                        "abs_pattern": float(np.abs(patterns[ci, ti])),
                        "n_trials": int(result["n_trials"]),
                        "n_a": int(result["n_a"]),
                        "n_b": int(result["n_b"]),
                    }
                )
        _write_progress("running", done, len(tasks))

    session_df = pd.DataFrame(rows)
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)
    if session_df.empty:
        session_df.to_csv(session_csv, index=False)
        pd.DataFrame().to_csv(day_mean_csv, index=False)
        pd.DataFrame().to_csv(day_abs_mean_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        _write_progress("completed_empty", done, len(tasks))
        raise RuntimeError("No valid Haufe patterns computed.")

    day_mean_df = (
        session_df.groupby(["day", "channel", "time_sec"], as_index=False)
        .agg(
            pattern_mean=("pattern", "mean"),
            pattern_sem=("pattern", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "channel", "time_sec"])
    )
    day_abs_mean_df = (
        session_df.groupby(["day", "channel", "time_sec"], as_index=False)
        .agg(
            abs_pattern_mean=("abs_pattern", "mean"),
            abs_pattern_sem=(
                "abs_pattern",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            ),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "channel", "time_sec"])
    )
    pos_df = pd.DataFrame(
        [{"channel": ch, "x": xyz[0], "y": xyz[1], "z": xyz[2]} for ch, xyz in sorted(channel_pos.items())]
    )

    session_df.to_csv(session_csv, index=False)
    day_mean_df.to_csv(day_mean_csv, index=False)
    day_abs_mean_df.to_csv(day_abs_mean_csv, index=False)
    pos_df.to_csv(channel_pos_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    _write_progress("completed", done, len(tasks))
    return {
        "session_csv": session_csv,
        "day_mean_csv": day_mean_csv,
        "day_abs_mean_csv": day_abs_mean_csv,
        "channel_pos_csv": channel_pos_csv,
        "qc_csv": qc_csv,
    }


def save_fig_mvpa_haufe_patterns(
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa_haufe",
    figures_dir: Path | str = _FIGURES_ROOT / "mvpa_haufe",
    time_points: tuple[float, ...] = (0.10, 0.15, 0.20, 0.30),
):
    """Generate day-panel topomaps for signed and abs Haufe patterns from saved outputs."""
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    day_mean_csv = output_dir / "haufe_day_mean_channel_time.csv"
    day_abs_mean_csv = output_dir / "haufe_day_abs_mean_channel_time.csv"
    channel_pos_csv = output_dir / "haufe_channel_positions.csv"
    if (not day_mean_csv.exists()) or (not day_abs_mean_csv.exists()) or (not channel_pos_csv.exists()):
        raise FileNotFoundError(f"Missing Haufe outputs in {output_dir}. Run run_mvpa_haufe_patterns() first.")

    day_mean_df = pd.read_csv(day_mean_csv)
    day_abs_mean_df = pd.read_csv(day_abs_mean_csv)
    pos_df = pd.read_csv(channel_pos_csv)
    ch_names = pos_df["channel"].tolist()
    ch_pos = {r["channel"]: np.array([r["x"], r["y"], r["z"]], dtype=float) for _, r in pos_df.iterrows()}
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="ignore")

    days = sorted(day_mean_df["day"].unique())
    times_all = sorted(day_mean_df["time_sec"].unique())
    picked_times = []
    for tp in time_points:
        idx = int(np.argmin(np.abs(np.array(times_all) - tp)))
        picked_times.append(float(times_all[idx]))

    def _plot_grid(df, value_col, title, fig_path, cmap):
        n_rows = len(days)
        n_cols = len(picked_times)
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.8 * n_cols, 2.6 * n_rows), squeeze=False)
        vmin = float(df[value_col].min())
        vmax = float(df[value_col].max())
        if vmin == vmax:
            vmax = vmin + 1e-12
        for ri, day in enumerate(days):
            for ci, tp in enumerate(picked_times):
                ax = axes[ri, ci]
                g = df[(df["day"] == day) & (np.isclose(df["time_sec"], tp))]
                vals = (
                    g.set_index("channel")
                    .reindex(ch_names)[value_col]
                    .to_numpy(dtype=float)
                )
                im, _ = mne.viz.plot_topomap(
                    vals,
                    info,
                    axes=ax,
                    show=False,
                    contours=0,
                    cmap=cmap,
                    vlim=(vmin, vmax),
                    sphere=(0.0, 0.0, 0.0, 0.095),
                )
                if ri == 0:
                    ax.set_title(f"t={tp:.3f}s")
                if ci == 0:
                    ax.set_ylabel(f"Day {day}")
        fig.suptitle(title)
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.9)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    fig_signed = figures_dir / "haufe_topomap_signed_by_day.png"
    fig_abs = figures_dir / "haufe_topomap_abs_by_day.png"
    _plot_grid(
        day_mean_df,
        "pattern_mean",
        "Haufe Activation Patterns (Signed)",
        fig_signed,
        "RdBu_r",
    )
    _plot_grid(
        day_abs_mean_df,
        "abs_pattern_mean",
        "Haufe Activation Patterns (Absolute)",
        fig_abs,
        "magma",
    )
    return {"figure_paths": {"signed": fig_signed, "abs": fig_abs}}


def run_mvpa_haufe_time_resolved(**kwargs):
    """Haufe patterns aligned to time-resolved MVPA."""
    kwargs.setdefault("output_dir", _OUTPUT_ROOT / "mvpa_haufe_time_resolved")
    return run_mvpa_haufe_patterns(**kwargs)


def save_fig_mvpa_haufe_time_resolved(**kwargs):
    """Figures for Haufe patterns aligned to time-resolved MVPA."""
    kwargs.setdefault("output_dir", _OUTPUT_ROOT / "mvpa_haufe_time_resolved")
    kwargs.setdefault("figures_dir", _FIGURES_ROOT / "mvpa_haufe_time_resolved")
    return save_fig_mvpa_haufe_patterns(**kwargs)


def run_mvpa_haufe_tg_within_day(**kwargs):
    """Haufe train-time patterns aligned to within-day TG."""
    kwargs.setdefault("output_dir", _OUTPUT_ROOT / "mvpa_haufe_tg_within_day")
    return run_mvpa_haufe_patterns(**kwargs)


def save_fig_mvpa_haufe_tg_within_day(**kwargs):
    """Figures for Haufe train-time patterns aligned to within-day TG."""
    kwargs.setdefault("output_dir", _OUTPUT_ROOT / "mvpa_haufe_tg_within_day")
    kwargs.setdefault("figures_dir", _FIGURES_ROOT / "mvpa_haufe_tg_within_day")
    return save_fig_mvpa_haufe_patterns(**kwargs)


def run_mvpa_haufe_tg_cross_day(
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa_haufe_tg_cross_day",
    cache_dir: Path | str = _OUTPUT_ROOT / "mvpa_tg" / "cache_stim_arrays",
    min_epochs: int = 20,
    random_state: int = 42,
    n_workers: int | None = None,
):
    """Haufe train-time patterns aligned to cross-day TG (day x day)."""
    output_dir = Path(output_dir)
    cache_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    session_csv = output_dir / "haufe_crossday_pair_channel_time.csv"
    day_mean_csv = output_dir / "haufe_crossday_train_day_mean_channel_time.csv"
    day_abs_mean_csv = output_dir / "haufe_crossday_train_day_abs_mean_channel_time.csv"
    channel_pos_csv = output_dir / "haufe_channel_positions.csv"
    qc_csv = output_dir / "haufe_qc_log.csv"
    progress_json = output_dir / "haufe_progress.json"

    _apply_single_thread_env_defaults()
    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))
    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    rows = []
    channel_pos = {}
    rng_master = np.random.default_rng(random_state)
    t0 = time.time()

    def _write_progress(stage: str, done: int, total: int):
        payload = {
            "stage": stage,
            "total": int(total),
            "done": int(done),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

    sessions = util_wrangle_load_sessions()
    day_data = {}
    for item in sessions:
        cache_path = cache_dir / f"stim_cache_{_session_cache_key(item)}.npz"
        if cache_path.exists():
            try:
                meta = _extract_stim_channel_meta(item["epochs"])
                ch_names = meta["ch_names"]
                pos = meta["pos"]
            except Exception:
                ch_names = []
                pos = np.empty((0, 3), dtype=float)
            day_data[(int(item["subject"]), int(item["day"]))] = {
                "cache_path": str(cache_path),
                "session_file": item["epo_file"],
                "ch_names": ch_names,
                "pos": pos,
            }

    pair_tasks = []
    subjects = sorted({k[0] for k in day_data})
    for subject in subjects:
        subject_days = sorted([d for (s, d) in day_data if s == subject])
        if len(subject_days) < 2:
            continue
        for d_train in subject_days:
            for d_test in subject_days:
                if d_test == d_train:
                    continue
                pair_tasks.append(
                    {
                        "subject": int(subject),
                        "train_day": int(d_train),
                        "test_day": int(d_test),
                        "train_item": day_data[(subject, d_train)],
                        "test_item": day_data[(subject, d_test)],
                        "ch_names": day_data[(subject, d_train)]["ch_names"],
                        "pos": day_data[(subject, d_train)]["pos"],
                        "pair_seed": int(rng_master.integers(0, 2**31 - 1)),
                        "min_epochs": int(min_epochs),
                        "random_state": int(random_state),
                    }
                )
    _write_progress("running", 0, len(pair_tasks))
    if n_workers == 1:
        pair_results = [_process_haufe_cross_pair_item(t) for t in pair_tasks]
    else:
        if threadpool_limits is None:
            pair_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                delayed(_process_haufe_cross_pair_item)(t) for t in pair_tasks
            )
        else:
            with threadpool_limits(limits=1):
                pair_results = Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                    delayed(_process_haufe_cross_pair_item)(t) for t in pair_tasks
                )

    done = 0
    for result in pair_results:
        done += 1
        if not result["ok"]:
            qc_rows.append(result["qc"])
            _write_progress("running", done, len(pair_tasks))
            continue
        ch_names = result["ch_names"]
        pos = result["pos"]
        for i, ch in enumerate(ch_names):
            if ch not in channel_pos:
                channel_pos[ch] = pos[i]
        t = result["t"]
        patterns = result["patterns"]
        for ci, ch in enumerate(ch_names):
            for ti, tsec in enumerate(t):
                rows.append(
                    {
                        "subject": int(result["subject"]),
                        "train_day": int(result["train_day"]),
                        "test_day": int(result["test_day"]),
                        "train_session_file": result["train_session_file"],
                        "test_session_file": result["test_session_file"],
                        "channel": ch,
                        "time_sec": float(tsec),
                        "pattern": float(patterns[ci, ti]),
                        "abs_pattern": float(np.abs(patterns[ci, ti])),
                        "n_per_class": int(result["n_per_class"]),
                    }
                )
        _write_progress("running", done, len(pair_tasks))

    pair_df = pd.DataFrame(rows)
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)
    if pair_df.empty:
        pair_df.to_csv(session_csv, index=False)
        pd.DataFrame().to_csv(day_mean_csv, index=False)
        pd.DataFrame().to_csv(day_abs_mean_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        _write_progress("completed_empty", done, len(pair_tasks))
        raise RuntimeError("No valid cross-day Haufe patterns computed.")

    day_mean_df = (
        pair_df.groupby(["train_day", "channel", "time_sec"], as_index=False)
        .agg(
            pattern_mean=("pattern", "mean"),
            pattern_sem=("pattern", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["train_day", "channel", "time_sec"])
    )
    day_abs_mean_df = (
        pair_df.groupby(["train_day", "channel", "time_sec"], as_index=False)
        .agg(
            abs_pattern_mean=("abs_pattern", "mean"),
            abs_pattern_sem=(
                "abs_pattern",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            ),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["train_day", "channel", "time_sec"])
    )
    pos_df = pd.DataFrame(
        [{"channel": ch, "x": xyz[0], "y": xyz[1], "z": xyz[2]} for ch, xyz in sorted(channel_pos.items())]
    )
    pair_df.to_csv(session_csv, index=False)
    day_mean_df.to_csv(day_mean_csv, index=False)
    day_abs_mean_df.to_csv(day_abs_mean_csv, index=False)
    pos_df.to_csv(channel_pos_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    _write_progress("completed", done, len(pair_tasks))
    return {
        "pair_csv": session_csv,
        "day_mean_csv": day_mean_csv,
        "day_abs_mean_csv": day_abs_mean_csv,
        "channel_pos_csv": channel_pos_csv,
        "qc_csv": qc_csv,
    }


def save_fig_mvpa_haufe_tg_cross_day(**kwargs):
    """Figures for Haufe patterns aligned to cross-day TG (grouped by train_day)."""
    kwargs.setdefault("output_dir", _OUTPUT_ROOT / "mvpa_haufe_tg_cross_day")
    kwargs.setdefault("figures_dir", _FIGURES_ROOT / "mvpa_haufe_tg_cross_day")
    output_dir = Path(kwargs["output_dir"])
    day_mean_csv = output_dir / "haufe_crossday_train_day_mean_channel_time.csv"
    day_abs_mean_csv = output_dir / "haufe_crossday_train_day_abs_mean_channel_time.csv"
    if (not day_mean_csv.exists()) or (not day_abs_mean_csv.exists()):
        raise FileNotFoundError(
            f"Missing cross-day Haufe outputs in {output_dir}. Run run_mvpa_haufe_tg_cross_day() first."
        )

    tmp_output = output_dir / "_tmp_plot_compat"
    tmp_output.mkdir(parents=True, exist_ok=True)
    (pd.read_csv(day_mean_csv)).rename(columns={"train_day": "day"}).to_csv(
        tmp_output / "haufe_day_mean_channel_time.csv", index=False
    )
    (pd.read_csv(day_abs_mean_csv)).rename(columns={"train_day": "day"}).to_csv(
        tmp_output / "haufe_day_abs_mean_channel_time.csv", index=False
    )
    src_pos = output_dir / "haufe_channel_positions.csv"
    pd.read_csv(src_pos).to_csv(tmp_output / "haufe_channel_positions.csv", index=False)
    kwargs["output_dir"] = tmp_output
    out = save_fig_mvpa_haufe_patterns(**kwargs)
    return out
