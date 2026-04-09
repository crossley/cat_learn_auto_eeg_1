#!/usr/bin/env python3
"""MVPA utilities for time-resolved and temporal-generalization analyses."""

from pathlib import Path
import os
import time
import warnings

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

_CODE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CODE_DIR.parent
_OUTPUT_ROOT = _PROJECT_DIR / "output"
_FIGURES_ROOT = _PROJECT_DIR / "figures"


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
    epo_dir: Path | str = Path("../task_eeg_preprocessed"),
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa",
    figures_dir: Path | str = _FIGURES_ROOT / "mvpa",
    min_epochs: int = 20,
    random_state: int = 42,
):
    """Compute per-session time-resolved MVPA and day-effect statistics."""
    epo_dir = Path(epo_dir)
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

    sessions = util_wrangle_load_sessions(epo_dir=epo_dir, preload=False)
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

    # Figure 1: one panel per day with mean+-SEM AUC over time
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

    # Figure 2: day-effect slope over time with FDR markers
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

    session_df.to_csv(session_csv, index=False)
    subject_day_df.to_csv(subject_day_csv, index=False)
    day_means_df.to_csv(day_means_csv, index=False)
    day_effect_df.to_csv(day_effect_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)

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


def _balanced_day_subset(X, y, n_per_class: int, rng: np.random.Generator):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    pick0 = rng.choice(idx0, size=n_per_class, replace=False)
    pick1 = rng.choice(idx1, size=n_per_class, replace=False)
    idx = np.concatenate([pick0, pick1])
    rng.shuffle(idx)
    return X[idx], y[idx]


def _process_within_day_session(
    session_item: dict,
    min_epochs: int,
    random_state: int,
):
    session_file = session_item["epo_file"]
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    epochs = session_item["epochs"]

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
        "X": X,
        "y": y,
        "t": t,
        "mat": mat,
    }


def util_mvpa_temporal_generalization(
    epo_dir: Path | str = Path("../task_eeg_preprocessed"),
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa_tg",
    figures_dir: Path | str = _FIGURES_ROOT / "mvpa_tg",
    min_epochs: int = 20,
    random_state: int = 42,
    progress_every: int = 5,
    n_workers: int = 1,
):
    """Compute within-day and cross-day temporal generalization decoding."""
    epo_dir = Path(epo_dir)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    within_subject_csv = output_dir / "tg_within_day_subject_level.csv"
    within_day_mean_csv = output_dir / "tg_within_day_day_mean.csv"
    cross_subject_csv = output_dir / "tg_cross_day_subject_level.csv"
    cross_day_mean_csv = output_dir / "tg_cross_day_day_mean.csv"
    qc_csv = output_dir / "tg_qc_log.csv"
    fig_within = figures_dir / "tg_within_day_heatmaps.png"
    fig_cross = figures_dir / "tg_cross_day_transfer_5x4.png"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    t0 = time.time()

    day_data = {}
    within_mats = []
    time_template = None

    rng_master = np.random.default_rng(random_state)

    session_items = util_wrangle_load_sessions(epo_dir=epo_dir, preload=False)

    n_workers = max(1, int(n_workers))
    print(
        f"[TG] Starting within-day TG on {len(session_items)} sessions "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    if len(session_items) == 0:
        within_results = []
    elif n_workers == 1:
        within_results = [
            _process_within_day_session(
                session_item=item,
                min_epochs=min_epochs,
                random_state=random_state,
            )
            for item in session_items
        ]
    else:
        within_results = Parallel(n_jobs=n_workers, prefer="threads", verbose=0)(
            delayed(_process_within_day_session)(
                session_item=item,
                min_epochs=min_epochs,
                random_state=random_state,
            )
            for item in session_items
        )

    n_done = 0
    for result in within_results:
        if not result["ok"]:
            qc_rows.append(result["qc"])
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
        day_data[(subject, day)] = {"X": X, "y": y, "session_file": session_file}
        n_done += 1
        if (n_done % max(progress_every, 1)) == 0:
            elapsed = time.time() - t0
            print(
                f"[TG] within-day complete {n_done}/{len(session_items)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)

    if not within_mats:
        pd.DataFrame().to_csv(within_subject_csv, index=False)
        pd.DataFrame().to_csv(within_day_mean_csv, index=False)
        pd.DataFrame().to_csv(cross_subject_csv, index=False)
        pd.DataFrame().to_csv(cross_day_mean_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        raise RuntimeError("No valid within-day TG matrices were computed.")

    # Flatten within-day subject-level matrices.
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

    # Write within-day outputs early so progress is visible while cross-day runs.
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)
    within_subject_df.to_csv(within_subject_csv, index=False)
    within_day_mean_df.to_csv(within_day_mean_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    print(
        f"[TG] Wrote within-day outputs. Starting cross-day transfer on "
        f"{len(sorted({k[0] for k in day_data}))} subjects...",
        flush=True,
    )

    # Cross-day transfer (within-subject): train day -> test other day.
    cross_rows = []
    subjects = sorted({k[0] for k in day_data})
    cross_done = 0
    cross_total = 0
    for subject in subjects:
        d = sorted([k[1] for k in day_data if k[0] == subject])
        cross_total += len(d) * (len(d) - 1)

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
                X_train_all, y_train_all = train_item["X"], train_item["y"]
                X_test_all, y_test_all = test_item["X"], test_item["y"]

                n_per_class = int(
                    min(
                        np.sum(y_train_all == 0),
                        np.sum(y_train_all == 1),
                        np.sum(y_test_all == 0),
                        np.sum(y_test_all == 1),
                    )
                )

                if n_per_class < 5:
                    qc_rows.append(
                        {
                            "session_file": f"{train_item['session_file']}->{test_item['session_file']}",
                            "subject": subject,
                            "day": d_train,
                            "stage": "cross_day_balance",
                            "reason": "insufficient_balanced_trials",
                            "detail": f"n_per_class={n_per_class}",
                        }
                    )
                    continue

                # Deterministic pair-specific RNG for reproducible balancing.
                pair_seed = int(rng_master.integers(0, 2**31 - 1))
                rng_pair = np.random.default_rng(pair_seed)

                X_train, y_train = _balanced_day_subset(
                    X_train_all, y_train_all, n_per_class=n_per_class, rng=rng_pair
                )
                X_test, y_test = _balanced_day_subset(
                    X_test_all, y_test_all, n_per_class=n_per_class, rng=rng_pair
                )

                clf = _build_clf(random_state=random_state)
                ge = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=1, verbose=False)
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", SklearnConvergenceWarning)
                        ge.fit(X_train, y_train)
                        mat_transfer = ge.score(X_test, y_test)
                except Exception as exc:
                    qc_rows.append(
                        {
                            "session_file": f"{train_item['session_file']}->{test_item['session_file']}",
                            "subject": subject,
                            "day": d_train,
                            "stage": "cross_day_tg",
                            "reason": "compute_error",
                            "detail": str(exc),
                        }
                    )
                    continue

                diag_mean_auc = float(np.nanmean(np.diag(mat_transfer)))
                cross_rows.append(
                    {
                        "subject": subject,
                        "train_day": d_train,
                        "test_day": d_test,
                        "n_per_class": int(n_per_class),
                        "n_train_trials_used": int(len(y_train)),
                        "n_test_trials_used": int(len(y_test)),
                        "diag_mean_auc": diag_mean_auc,
                    }
                )
                cross_done += 1
                if (cross_done % max(progress_every * 2, 1)) == 0:
                    elapsed = time.time() - t0
                    print(
                        f"[TG] cross-day complete {cross_done}/{cross_total} pairs "
                        f"(elapsed {elapsed/60:.1f} min)",
                        flush=True,
                    )
                    pd.DataFrame(cross_rows).to_csv(cross_subject_csv, index=False)

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

    # Save tables.
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)
    within_subject_df.to_csv(within_subject_csv, index=False)
    within_day_mean_df.to_csv(within_day_mean_csv, index=False)
    cross_subject_df.to_csv(cross_subject_csv, index=False)
    cross_day_mean_df.to_csv(cross_day_mean_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)

    # Plot imports are intentionally lazy to avoid startup stalls from
    # font-cache initialization before compute begins.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Figure: within-day heatmaps (one panel per day).
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

    # Figure: cross-day transfer heatmap (off-diagonal day transfer).
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
