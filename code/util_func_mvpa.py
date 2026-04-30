#!/usr/bin/env python3
"""MVPA utilities for time-resolved and temporal-generalization analyses."""

from pathlib import Path
import os
import time
import warnings
import json
import hashlib
from multiprocessing import cpu_count

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

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
from util_func_wrangle import util_wrangle_align_beh_to_epochs, util_wrangle_load_sessions
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
    haufe_session_csv = output_dir / "mvpa_haufe_session_channel_time.csv"
    haufe_day_mean_csv = output_dir / "mvpa_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_haufe_channel_positions.csv"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    session_rows = []
    haufe_rows = []
    haufe_channel_pos = {}

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

        try:
            patterns = _compute_haufe_patterns_from_xy(X, y, random_state=random_state)
            for ci, ch in enumerate(stim_epochs.ch_names):
                if ch not in haufe_channel_pos:
                    loc = stim_epochs.info["chs"][stim_epochs.info.ch_names.index(ch)]["loc"][:3]
                    haufe_channel_pos[ch] = np.array(loc, dtype=float)
                for ti, tsec in enumerate(t):
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
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "haufe",
                    "reason": "compute_error",
                    "detail": str(exc),
                }
            )

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
    ch_names = np.array(stim_epochs.ch_names, dtype=str)

    return X, y, t, ch_names


def _prepare_session_cache(session_item: dict, cache_dir: Path):
    session_file = session_item["epo_file"]
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    epochs = session_item["epochs"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"stim_cache_{_session_cache_key(session_item)}.npz"

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            has_channel_metadata = "ch_names" in z.files
            if has_channel_metadata:
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
        X, y, t, ch_names = _prepare_stim_data(epochs)
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
        "t": t_train,
        "mat": mat_transfer,
    }


def util_mvpa_temporal_generalization(
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa_tg_combined",
    figures_dir: Path | str = _FIGURES_ROOT / "mvpa_tg_combined",
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
    cross_matrix_dir = output_dir / "tg_cross_day_subject_matrices"
    cross_matrix_day_mean_csv = output_dir / "tg_cross_day_timegen_day_mean.csv"
    qc_csv = output_dir / "tg_qc_log.csv"
    progress_json = output_dir / "tg_progress.json"
    fig_within = figures_dir / "tg_within_day_heatmaps.png"
    fig_cross = figures_dir / "tg_cross_day_transfer_5x4.png"
    fig_cross_timegen = figures_dir / "tg_cross_day_timegen_matrices_5x5.png"

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
    n_done = 0

    def _handle_within_result(result):
        nonlocal n_done, time_template, wrote_within_subject, wrote_qc, qc_rows
        if not result["ok"]:
            qc_rows.append(result["qc"])
            if len(qc_rows) >= max(progress_every, 1):
                wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
                qc_rows = []
            return

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
            return

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
        rows_local = []
        for i in range(len(t)):
            for j in range(len(t)):
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

    if run_within_day:
        print(
            f"[TG] Starting within-day TG on {len(prepared_items)} prepared sessions "
            f"(n_workers={n_workers})...",
            flush=True,
        )
        _write_progress("within_day_running", 0, len(prepared_items), 0, 0)
        if n_workers == 1:
            if threadpool_limits is None:
                for item in prepared_items:
                    _handle_within_result(
                        _process_within_day_session(
                            session_meta=item,
                            min_epochs=min_epochs,
                            random_state=random_state,
                        )
                    )
            else:
                with threadpool_limits(limits=1):
                    for item in prepared_items:
                        _handle_within_result(
                            _process_within_day_session(
                                session_meta=item,
                                min_epochs=min_epochs,
                                random_state=random_state,
                            )
                        )
        elif len(prepared_items) > 0:
            if threadpool_limits is None:
                result_iter = Parallel(n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered")(
                    delayed(_process_within_day_session)(
                        session_meta=item,
                        min_epochs=min_epochs,
                        random_state=random_state,
                    )
                    for item in prepared_items
                )
                for result in result_iter:
                    _handle_within_result(result)
            else:
                with threadpool_limits(limits=1):
                    result_iter = Parallel(n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered")(
                        delayed(_process_within_day_session)(
                            session_meta=item,
                            min_epochs=min_epochs,
                            random_state=random_state,
                        )
                        for item in prepared_items
                    )
                    for result in result_iter:
                        _handle_within_result(result)

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
    cross_matrix_accum = {}
    cross_time_template = None
    cross_total = 0
    pair_items = []
    if run_cross_day:
        cross_matrix_dir.mkdir(parents=True, exist_ok=True)
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

    cross_done = 0
    if run_cross_day:
        _write_progress("cross_day_running", n_done, len(prepared_items), 0, cross_total)

    def _handle_cross_result(result):
        nonlocal wrote_cross_subject, wrote_qc, cross_done, qc_rows, cross_time_template
        if result["ok"]:
            row = result["row"]
            mat = np.asarray(result["mat"], dtype=float)
            t_vec = np.asarray(result["t"], dtype=float)
            cross_rows.append(row)
            wrote_cross_subject = _append_csv(pd.DataFrame([row]), cross_subject_csv, wrote_cross_subject)
            matrix_path = cross_matrix_dir / (
                f"sub_{int(row['subject']):03d}_trainD{int(row['train_day'])}_testD{int(row['test_day'])}.npz"
            )
            np.savez_compressed(matrix_path, auc=mat, time_sec=t_vec)
            if cross_time_template is None:
                cross_time_template = t_vec
            key = (int(row["train_day"]), int(row["test_day"]))
            if key not in cross_matrix_accum:
                cross_matrix_accum[key] = {"sum": np.zeros_like(mat, dtype=float), "count": np.zeros_like(mat, dtype=float)}
            valid = np.isfinite(mat)
            cross_matrix_accum[key]["sum"][valid] += mat[valid]
            cross_matrix_accum[key]["count"][valid] += 1.0
        else:
            qc_rows.append(result["qc"])
            if len(qc_rows) >= max(progress_every, 1):
                wrote_qc = _append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
                qc_rows = []
        cross_done += 1
        _write_progress("cross_day_running", n_done, len(prepared_items), cross_done, cross_total)
        if (cross_done % max(progress_every * 2, 1)) == 0:
            elapsed = time.time() - t0
            print(
                f"[TG] cross-day processed {cross_done}/{cross_total} pairs "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    if run_cross_day and cross_total > 0:
        if n_workers == 1:
            if threadpool_limits is None:
                for item in pair_items:
                    _handle_cross_result(_process_cross_day_pair(pair_item=item, random_state=random_state))
            else:
                with threadpool_limits(limits=1):
                    for item in pair_items:
                        _handle_cross_result(_process_cross_day_pair(pair_item=item, random_state=random_state))
        else:
            if threadpool_limits is None:
                result_iter = Parallel(n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered")(
                    delayed(_process_cross_day_pair)(pair_item=item, random_state=random_state)
                    for item in pair_items
                )
                for result in result_iter:
                    _handle_cross_result(result)
            else:
                with threadpool_limits(limits=1):
                    result_iter = Parallel(n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered")(
                        delayed(_process_cross_day_pair)(pair_item=item, random_state=random_state)
                        for item in pair_items
                    )
                    for result in result_iter:
                        _handle_cross_result(result)
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
        cross_matrix_rows = []
        if cross_time_template is not None:
            for (train_day, test_day), acc in sorted(cross_matrix_accum.items()):
                with np.errstate(invalid="ignore", divide="ignore"):
                    mean_mat = acc["sum"] / acc["count"]
                for i, train_t in enumerate(cross_time_template):
                    for j, test_t in enumerate(cross_time_template):
                        val = mean_mat[i, j]
                        if np.isfinite(val):
                            cross_matrix_rows.append(
                                {
                                    "train_day": int(train_day),
                                    "test_day": int(test_day),
                                    "train_time_sec": float(train_t),
                                    "test_time_sec": float(test_t),
                                    "auc_mean": float(val),
                                    "n_subjects": int(acc["count"][i, j]),
                                }
                            )
        pd.DataFrame(cross_matrix_rows).to_csv(cross_matrix_day_mean_csv, index=False)
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
            "cross_matrix_day_mean_csv": cross_matrix_day_mean_csv,
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
    cross_matrix_plot_df = (
        pd.read_csv(cross_matrix_day_mean_csv)
        if cross_matrix_day_mean_csv.exists()
        else pd.DataFrame()
    )

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

        if not cross_matrix_plot_df.empty:
            fig, axes = plt.subplots(5, 5, figsize=(15, 15), squeeze=False)
            vmin = float(cross_matrix_plot_df["auc_mean"].min())
            vmax = float(cross_matrix_plot_df["auc_mean"].max())
            for i, train_day in enumerate(day_grid):
                for j, test_day in enumerate(day_grid):
                    ax = axes[i, j]
                    if train_day == test_day:
                        ax.axis("off")
                        ax.text(0.5, 0.5, f"D{train_day}=D{test_day}", ha="center", va="center")
                        continue
                    g = cross_matrix_plot_df[
                        (cross_matrix_plot_df["train_day"] == train_day)
                        & (cross_matrix_plot_df["test_day"] == test_day)
                    ]
                    if g.empty:
                        ax.axis("off")
                        continue
                    pivot = g.pivot(index="train_time_sec", columns="test_time_sec", values="auc_mean")
                    im = ax.imshow(
                        pivot.to_numpy(),
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
                    ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
                    ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
                    ax.set_title(f"Train D{train_day} -> Test D{test_day}", fontsize=9)
                    if i == len(day_grid) - 1:
                        ax.set_xlabel("Test Time (s)")
                    if j == 0:
                        ax.set_ylabel("Train Time (s)")
            fig.suptitle("Cross-Day Temporal Generalization by Day Pair (AUC)")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="AUC")
            fig.subplots_adjust(top=0.94, wspace=0.30, hspace=0.35)
            fig.savefig(fig_cross_timegen, dpi=150, bbox_inches="tight")
            plt.close(fig)

    print(f"Wrote TG within-day subject table: {within_subject_csv}")
    print(f"Wrote TG within-day day-mean table: {within_day_mean_csv}")
    print(f"Wrote TG cross-day subject table: {cross_subject_csv}")
    print(f"Wrote TG cross-day day-mean table: {cross_day_mean_csv}")
    print(f"Wrote TG QC log: {qc_csv}")
    print(f"Saved TG figures: 3")
    print(f"- within-day heatmaps: {fig_within}")
    print(f"- cross-day transfer: {fig_cross}")
    print(f"- cross-day timegen matrices: {fig_cross_timegen}")
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
        "cross_matrix_day_mean_csv": cross_matrix_day_mean_csv,
        "qc_csv": qc_csv,
        "figure_paths": {
            "within_day_heatmaps": fig_within,
            "cross_day_transfer": fig_cross,
            "cross_day_timegen_matrices": fig_cross_timegen,
        },
    }


def run_mvpa_time_resolved(**kwargs):
    """Run time-resolved MVPA analysis."""
    return util_mvpa_time_resolved(save_figures=False, **kwargs)


def _process_response_mvpa_session(task: dict):
    session_file = task["epo_file"]
    subject = int(task["subject"])
    day = int(task["day"])
    beh_df = task["beh_df"]
    min_epochs = int(task["min_epochs"])
    random_state = int(task["random_state"])

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh_aligned = util_wrangle_align_beh_to_epochs(
            beh_df,
            epochs,
            event_names=("Stim/A", "Stim/B"),
        )
        if len(stim_epochs) == 0:
            raise RuntimeError("No aligned stimulus epochs.")
        stim_epochs = stim_epochs.copy()
        stim_epochs.load_data()
        stim_epochs.pick_types(eeg=True, exclude="bads")
        if len(stim_epochs.ch_names) == 0:
            raise RuntimeError("No EEG channels after pick_types.")
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

    if "resp" not in beh_aligned.columns:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "response_label",
                "reason": "missing_resp_column",
                "detail": "",
            },
        }

    resp = beh_aligned["resp"].astype(str).str.strip().str.upper().to_numpy()
    y = np.full(len(resp), -1, dtype=int)
    y[resp == "A"] = 0
    y[resp == "B"] = 1
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
                "reason": "insufficient_response_trials",
                "detail": f"n_resp_a={n_a}, n_resp_b={n_b}; need >=5 in each class",
            },
        }

    auc = _decode_timecourse(X, y, n_splits=5, random_state=random_state)
    try:
        patterns = _compute_haufe_patterns_from_xy(X, y, random_state=random_state)
    except Exception:
        patterns = None
    rows = []
    for ti, auc_val in enumerate(auc):
        rows.append(
            {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "time_sec": float(stim_epochs.times[ti]),
                "auc": float(auc_val),
                "n_trials": n_trials,
                "n_resp_a": n_a,
                "n_resp_b": n_b,
            }
        )
    haufe_rows = []
    channel_pos = []
    if patterns is not None:
        for ci, ch in enumerate(stim_epochs.ch_names):
            loc = stim_epochs.info["chs"][stim_epochs.info.ch_names.index(ch)]["loc"][:3]
            channel_pos.append({"channel": ch, "x": float(loc[0]), "y": float(loc[1]), "z": float(loc[2])})
            for ti, tsec in enumerate(stim_epochs.times):
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
                        "n_resp_a": n_a,
                        "n_resp_b": n_b,
                    }
                )
    return {"ok": True, "rows": rows, "haufe_rows": haufe_rows, "channel_pos": channel_pos}


def util_mvpa_response_time_resolved(
    output_dir: Path | str = _OUTPUT_ROOT / "mvpa_response",
    figures_dir: Path | str = _FIGURES_ROOT / "mvpa_response",
    min_epochs: int = 20,
    random_state: int = 42,
    save_figures: bool = True,
    n_workers: int | None = None,
):
    """Compute per-session time-resolved MVPA using subject response as the label."""
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

    session_csv = output_dir / "mvpa_response_session_timecourse.csv"
    subject_day_csv = output_dir / "mvpa_response_subject_day_timecourse.csv"
    day_means_csv = output_dir / "mvpa_response_day_means_timecourse.csv"
    day_effect_csv = output_dir / "mvpa_response_day_effect_per_time.csv"
    qc_csv = output_dir / "mvpa_response_qc_log.csv"
    progress_json = output_dir / "mvpa_response_progress.json"
    haufe_session_csv = output_dir / "mvpa_response_haufe_session_channel_time.csv"
    haufe_day_mean_csv = output_dir / "mvpa_response_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_response_haufe_channel_positions.csv"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    session_rows = []
    haufe_rows = []
    channel_pos = {}
    t0 = time.time()

    def _write_progress(stage: str, done: int, total: int):
        payload = {
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

    sessions = util_wrangle_load_sessions()
    tasks = [
        {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "beh_df": item["beh_df"],
            "epo_file": item["epo_file"],
            "epo_path": str(Path("../EEG_epo") / item["epo_file"]),
            "min_epochs": int(min_epochs),
            "random_state": int(random_state),
        }
        for item in sessions
    ]
    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))
    _write_progress("running", 0, len(tasks))

    def _handle_response_result(result, done):
        if result["ok"]:
            session_rows.extend(result["rows"])
            haufe_rows.extend(result.get("haufe_rows", []))
            for pos_row in result.get("channel_pos", []):
                channel_pos.setdefault(pos_row["channel"], pos_row)
        else:
            qc_rows.append(result["qc"])
        _write_progress("running", done, len(tasks))
        if (done % 5) == 0:
            elapsed = time.time() - t0
            print(
                f"[MVPA response] complete {done}/{len(tasks)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    print(
        f"[MVPA response] Starting response-label MVPA on {len(tasks)} sessions "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            _handle_response_result(_process_response_mvpa_session(task), done)
    elif threadpool_limits is None:
        result_iter = Parallel(n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered")(
            delayed(_process_response_mvpa_session)(task) for task in tasks
        )
        for done, result in enumerate(result_iter, start=1):
            _handle_response_result(result, done)
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered")(
                delayed(_process_response_mvpa_session)(task) for task in tasks
            )
            for done, result in enumerate(result_iter, start=1):
                _handle_response_result(result, done)

    session_df = pd.DataFrame(session_rows)
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)
    if session_df.empty:
        session_df.to_csv(session_csv, index=False)
        pd.DataFrame().to_csv(subject_day_csv, index=False)
        pd.DataFrame().to_csv(day_means_csv, index=False)
        pd.DataFrame().to_csv(day_effect_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        raise RuntimeError("Response-label MVPA produced no valid session rows.")

    subject_day_df = (
        session_df.groupby(["subject", "day", "time_sec"], as_index=False)["auc"]
        .mean()
        .sort_values(["subject", "day", "time_sec"])
    )
    day_means_df = (
        subject_day_df.groupby(["day", "time_sec"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sem=("auc", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
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
            except Exception as exc:
                effect_rows.append(
                    {
                        "time_sec": float(t),
                        "n_rows": int(len(g)),
                        "n_subjects": int(g["subject"].nunique()),
                        "day_coef": np.nan,
                        "day_se": np.nan,
                        "day_pvalue": np.nan,
                        "status": "model_error",
                        "detail": str(exc),
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
        pd.DataFrame(list(channel_pos.values())).sort_values("channel").to_csv(haufe_channel_pos_csv, index=False)
        haufe_session_df.to_csv(haufe_session_csv, index=False)
        haufe_day_mean_df.to_csv(haufe_day_mean_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)
    _write_progress("completed", len(tasks), len(tasks))

    if save_figures:
        save_fig_mvpa_response_time_resolved(output_dir=output_dir, figures_dir=figures_dir)

    return {
        "session_df": session_df,
        "subject_day_df": subject_day_df,
        "day_means_df": day_means_df,
        "day_effect_df": day_effect_df,
        "qc_df": qc_df,
        "session_csv": session_csv,
        "subject_day_csv": subject_day_csv,
        "day_means_csv": day_means_csv,
        "day_effect_csv": day_effect_csv,
        "qc_csv": qc_csv,
    }



def save_fig_mvpa_response_time_resolved(**kwargs):
    """Generate response-label time-resolved MVPA figures with integrated Haufe topoplots."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa_response"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa_response"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    session_csv = output_dir / "mvpa_response_session_timecourse.csv"
    day_means_csv = output_dir / "mvpa_response_day_means_timecourse.csv"
    day_effect_csv = output_dir / "mvpa_response_day_effect_per_time.csv"
    haufe_day_mean_csv = output_dir / "mvpa_response_haufe_day_mean_channel_time.csv"
    haufe_session_csv = output_dir / "mvpa_response_haufe_session_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_response_haufe_channel_positions.csv"
    haufe_peak_times_csv = output_dir / "mvpa_response_haufe_subject_day_peak_times.csv"
    haufe_stability_subject_csv = output_dir / "mvpa_response_haufe_peak_stability_subject.csv"
    haufe_stability_summary_csv = output_dir / "mvpa_response_haufe_peak_stability_summary.csv"
    if (not day_means_csv.exists()) or (not day_effect_csv.exists()):
        raise FileNotFoundError(
            f"Missing response-label MVPA outputs in {output_dir}. Run run_mvpa_response_time_resolved() first."
        )
    day_means_df = pd.read_csv(day_means_csv)
    day_effect_df = pd.read_csv(day_effect_csv)
    fig_day_panels = figures_dir / "mvpa_response_auc_by_day_panels.png"
    fig_day_slope = figures_dir / "mvpa_response_day_slope_timecourse.png"
    fig_haufe_stability = figures_dir / "mvpa_response_haufe_peak_stability.png"

    def _make_haufe_info(pos_df):
        ch_names = pos_df["channel"].tolist()
        ch_pos = {
            r["channel"]: np.array([r["x"], r["y"], r["z"]], dtype=float)
            for _, r in pos_df.iterrows()
        }
        info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        info.set_montage(montage, on_missing="ignore")
        return info, ch_names

    def _detect_subject_day_peak_times():
        if not session_csv.exists():
            return pd.DataFrame()
        d_session = pd.read_csv(session_csv)
        rows = []
        for (subject, day), d_sd in d_session.groupby(["subject", "day"]):
            d_sd = d_sd.sort_values("time_sec")
            for lo, hi, label in [(0.0, 0.20, "early"), (0.35, 0.80, "late")]:
                d_win = d_sd[(d_sd["time_sec"] >= lo) & (d_sd["time_sec"] <= hi)]
                if d_win.empty:
                    continue
                row = d_win.loc[d_win["auc"].idxmax()]
                rows.append(
                    {
                        "subject": int(subject),
                        "day": int(day),
                        "peak": label,
                        "peak_time_sec": float(row["time_sec"]),
                        "peak_auc": float(row["auc"]),
                        "window_start_sec": float(lo),
                        "window_end_sec": float(hi),
                    }
                )
        peak_df = pd.DataFrame(rows).sort_values(["day", "peak", "subject"])
        peak_df.to_csv(haufe_peak_times_csv, index=False)
        return peak_df

    def _vector_corr(x_vec, y_vec):
        valid = np.isfinite(x_vec) & np.isfinite(y_vec)
        if int(np.sum(valid)) < 3:
            return np.nan
        x_use = x_vec[valid] - np.nanmean(x_vec[valid])
        y_use = y_vec[valid] - np.nanmean(y_vec[valid])
        denom = np.sqrt(np.sum(x_use ** 2) * np.sum(y_use ** 2))
        if (not np.isfinite(denom)) or denom <= np.finfo(float).eps:
            return np.nan
        return float(np.sum(x_use * y_use) / denom)

    def _write_haufe_stability(peak_df, ch_names):
        if (not haufe_session_csv.exists()) or peak_df.empty:
            return pd.DataFrame()
        d_sub = pd.read_csv(haufe_session_csv)
        if d_sub.empty:
            return pd.DataFrame()
        rows = []
        for day in sorted(peak_df["day"].dropna().unique().astype(int)):
            d_day = d_sub[d_sub["day"] == day].copy()
            times = np.sort(d_day["time_sec"].dropna().unique().astype(float))
            if len(times) == 0:
                continue
            for peak_label in ["early", "late"]:
                d_peak_meta = peak_df[(peak_df["day"] == day) & (peak_df["peak"] == peak_label)]
                vec_rows = []
                for _, peak_row in d_peak_meta.iterrows():
                    subject = int(peak_row["subject"])
                    peak_time = float(peak_row["peak_time_sec"])
                    t_show = float(times[int(np.argmin(np.abs(times - peak_time)))])
                    d_peak = d_day[(d_day["subject"] == subject) & np.isclose(d_day["time_sec"], t_show)].copy()
                    if d_peak.empty:
                        continue
                    vals = d_peak.set_index("channel").reindex(ch_names)["pattern"].to_numpy(dtype=float)
                    vec_rows.append({"subject": subject, "peak_time_sec": peak_time, "used_time_sec": t_show, "values": vals})
                if len(vec_rows) < 3:
                    continue
                values = np.vstack([r["values"] for r in vec_rows])
                subjects = np.array([r["subject"] for r in vec_rows])
                for i_sub, subject in enumerate(subjects):
                    x_vec = values[i_sub, :]
                    loo = np.delete(values, i_sub, axis=0)
                    with np.errstate(invalid="ignore"):
                        y_vec = np.nanmean(loo, axis=0)
                    rows.append(
                        {
                            "subject": int(subject),
                            "day": int(day),
                            "peak": peak_label,
                            "subject_peak_time_sec": float(vec_rows[i_sub]["peak_time_sec"]),
                            "used_time_sec": float(vec_rows[i_sub]["used_time_sec"]),
                            "loo_pattern_r": _vector_corr(x_vec, y_vec),
                            "n_channels": int(np.sum(np.isfinite(x_vec) & np.isfinite(y_vec))),
                        }
                    )
        subject_df = pd.DataFrame(rows)
        if subject_df.empty:
            subject_df.to_csv(haufe_stability_subject_csv, index=False)
            pd.DataFrame().to_csv(haufe_stability_summary_csv, index=False)
            return subject_df
        summary_df = (
            subject_df.groupby(["day", "peak"], as_index=False)
            .agg(
                median_peak_time_sec=("subject_peak_time_sec", "median"),
                q25_peak_time_sec=("subject_peak_time_sec", lambda x: float(np.nanpercentile(x, 25))),
                q75_peak_time_sec=("subject_peak_time_sec", lambda x: float(np.nanpercentile(x, 75))),
                median_used_time_sec=("used_time_sec", "median"),
                median_r=("loo_pattern_r", "median"),
                q25_r=("loo_pattern_r", lambda x: float(np.nanpercentile(x, 25))),
                q75_r=("loo_pattern_r", lambda x: float(np.nanpercentile(x, 75))),
                mean_r=("loo_pattern_r", "mean"),
                prop_positive=("loo_pattern_r", lambda x: float(np.nanmean(np.asarray(x) > 0))),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["day", "peak"])
        )
        subject_df.to_csv(haufe_stability_subject_csv, index=False)
        summary_df.to_csv(haufe_stability_summary_csv, index=False)
        return subject_df

    def _plot_haufe_stability(subject_df):
        if subject_df.empty:
            return
        peak_order = [p for p in ["early", "late"] if p in set(subject_df["peak"])]
        days_plot = sorted(subject_df["day"].dropna().unique().astype(int).tolist())
        fig, axes = plt.subplots(1, len(peak_order), figsize=(4.5 * len(peak_order), 4), sharey=True, squeeze=False)
        rng = np.random.default_rng(42)
        for ax, peak in zip(axes.ravel(), peak_order):
            data = []
            labels = []
            for day in days_plot:
                vals = subject_df[(subject_df["day"] == day) & (subject_df["peak"] == peak)]["loo_pattern_r"].dropna().to_numpy(dtype=float)
                data.append(vals)
                labels.append(f"D{day}")
            ax.boxplot(data, labels=labels, showfliers=False)
            for i_day, vals in enumerate(data, start=1):
                if len(vals) == 0:
                    continue
                ax.scatter(np.full(len(vals), i_day) + rng.normal(0.0, 0.035, size=len(vals)), vals, s=18, alpha=0.55, color="#2f4f4f")
            ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
            ax.set_title(f"{peak} peak")
            ax.set_xlabel("Day")
            ax.grid(axis="y", alpha=0.25)
        axes.ravel()[0].set_ylabel("Subject vs leave-one-subject-out group pattern r")
        fig.suptitle("Response Haufe Pattern Stability at MVPA Peaks")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(fig_haufe_stability, dpi=150, bbox_inches="tight")
        plt.close(fig)

    haufe_df = pd.DataFrame()
    haufe_info = None
    haufe_ch_names = []
    peak_medians = {}
    if haufe_day_mean_csv.exists() and haufe_channel_pos_csv.exists():
        haufe_df = pd.read_csv(haufe_day_mean_csv)
        pos_df = pd.read_csv(haufe_channel_pos_csv)
        if (not haufe_df.empty) and (not pos_df.empty):
            haufe_info, haufe_ch_names = _make_haufe_info(pos_df)
            peak_df = _detect_subject_day_peak_times()
            if not peak_df.empty:
                peak_medians = {
                    (int(r["day"]), str(r["peak"])): float(r["peak_time_sec"])
                    for _, r in peak_df.groupby(["day", "peak"], as_index=False)["peak_time_sec"].median().iterrows()
                }
                stability_df = _write_haufe_stability(peak_df, haufe_ch_names)
                _plot_haufe_stability(stability_df)

    days = sorted(day_means_df["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 5.2), sharey=True, squeeze=False)
    x_all = day_means_df["time_sec"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_all))
    x_max = float(np.nanmax(x_all))
    y_upper = float(np.nanmax(day_means_df["auc_mean"] + day_means_df["auc_sem"].fillna(0.0)))
    y_lower = float(np.nanmin(day_means_df["auc_mean"] - day_means_df["auc_sem"].fillna(0.0)))
    y_pad = max(0.02, 0.20 * (y_upper - y_lower))
    topomap_ims = []
    lim = 1e-12
    if not haufe_df.empty:
        lim = float(np.nanmax(np.abs(haufe_df["pattern_mean"].to_numpy(dtype=float))))
        if not np.isfinite(lim) or lim <= 0:
            lim = 1e-12
    for ax, day in zip(axes.ravel(), days):
        g = day_means_df[day_means_df["day"] == day].sort_values("time_sec")
        x = g["time_sec"].to_numpy()
        y = g["auc_mean"].to_numpy()
        s = g["auc_sem"].to_numpy()
        ax.plot(x, y, color="#8c510a", linewidth=2)
        ax.fill_between(x, y - s, y + s, color="#8c510a", alpha=0.2, linewidth=0)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(y_lower - 0.02, y_upper + y_pad)
        ax.grid(alpha=0.25)
        day_peak_times = [
            (peak_medians[(int(day), peak_label)], peak_label)
            for peak_label in ["early", "late"]
            if (int(day), peak_label) in peak_medians
        ]
        for peak_time, peak_label in day_peak_times:
            if x_max <= x_min:
                continue
            y_peak = float(np.interp(peak_time, x, y))
            ax.axvline(peak_time, color="#b22222", linestyle=":", linewidth=1.2)
            ax.scatter([peak_time], [y_peak], s=36, facecolor="white", edgecolor="#b22222", linewidth=1.2, zorder=4)
            ax.text(peak_time, y_upper + (0.05 * y_pad), peak_label, color="#b22222", fontsize=8, ha="center", va="bottom")
            if haufe_df.empty or haufe_info is None:
                continue
            x_frac = (peak_time - x_min) / (x_max - x_min)
            width = 0.18
            inset = ax.inset_axes([max(0.01, min(0.99 - width, x_frac - width / 2.0)), 1.04, width, 0.36], transform=ax.transAxes)
            d_day = haufe_df[haufe_df["day"] == day]
            times = np.sort(d_day["time_sec"].unique().astype(float))
            t_show = float(times[int(np.argmin(np.abs(times - peak_time)))])
            vals = d_day[np.isclose(d_day["time_sec"], t_show)].set_index("channel").reindex(haufe_ch_names)["pattern_mean"].to_numpy(dtype=float)
            im, _ = mne.viz.plot_topomap(vals, haufe_info, axes=inset, show=False, contours=0, cmap="RdBu_r", vlim=(-lim, lim), sphere=(0.0, 0.0, 0.0, 0.095))
            topomap_ims.append(im)
            inset.set_title(f"{peak_label}\nmedian {peak_time:.3f}s\nmap {t_show:.3f}s", fontsize=7)
    axes.ravel()[0].set_ylabel("ROC-AUC")
    fig.suptitle("Time-resolved Response Decoding (Response A vs Response B)")
    if topomap_ims:
        cax = fig.add_axes([0.32, 0.89, 0.36, 0.025])
        fig.colorbar(topomap_ims[-1], cax=cax, orientation="horizontal", label="Response Haufe pattern")
        fig.tight_layout(rect=[0, 0, 1, 0.78])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_day_panels, dpi=150, bbox_inches="tight")
    plt.close(fig)

    g = day_effect_df.sort_values("time_sec")
    x = g["time_sec"].to_numpy()
    y = g["day_coef"].to_numpy()
    sig = g["significant_fdr"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="#01665e", linewidth=2, label="Day slope (AUC ~ day)")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    if np.any(sig):
        ax.scatter(x[sig], y[sig], color="red", s=16, label="FDR < 0.05", zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Day coefficient")
    ax.set_title("Day Effect on Response Decoding Over Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_day_slope, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {
        "figure_paths": {"day_panels": fig_day_panels, "day_slope": fig_day_slope, "haufe_stability": fig_haufe_stability},
        "haufe_stability_subject_csv": haufe_stability_subject_csv,
        "haufe_stability_summary_csv": haufe_stability_summary_csv,
    }

def run_mvpa_response_time_resolved(**kwargs):
    """Run time-resolved MVPA with subject response as the label."""
    return util_mvpa_response_time_resolved(save_figures=False, **kwargs)


def save_fig_mvpa_time_resolved(**kwargs):
    """Generate time-resolved MVPA figures."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa"))
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    session_csv = output_dir / "mvpa_session_timecourse.csv"
    day_means_csv = output_dir / "mvpa_day_means_timecourse.csv"
    day_effect_csv = output_dir / "mvpa_day_effect_per_time.csv"
    if (not day_means_csv.exists()) or (not day_effect_csv.exists()):
        raise FileNotFoundError(f"Missing MVPA outputs in {output_dir}. Run run_mvpa_time_resolved() first.")

    day_means_df = pd.read_csv(day_means_csv)
    day_effect_df = pd.read_csv(day_effect_csv)
    fig_day_panels = figures_dir / "mvpa_auc_by_day_panels.png"
    fig_day_slope = figures_dir / "mvpa_day_slope_timecourse.png"
    haufe_day_mean_csv = output_dir / "mvpa_haufe_day_mean_channel_time.csv"
    haufe_session_csv = output_dir / "mvpa_haufe_session_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_haufe_channel_positions.csv"
    haufe_peak_times_csv = output_dir / "mvpa_haufe_subject_day_peak_times.csv"
    haufe_stability_subject_csv = output_dir / "mvpa_haufe_peak_stability_subject.csv"
    haufe_stability_summary_csv = output_dir / "mvpa_haufe_peak_stability_summary.csv"
    fig_haufe_stability = figures_dir / "mvpa_haufe_peak_stability.png"

    def _detect_subject_day_peak_times():
        if not session_csv.exists():
            return pd.DataFrame()
        d_session = pd.read_csv(session_csv)
        if d_session.empty:
            return pd.DataFrame()
        rows = []
        for (subject, day), d_sd in d_session.groupby(["subject", "day"]):
            d_sd = d_sd.sort_values("time_sec")
            for lo, hi, label in [(0.0, 0.20, "early"), (0.35, 0.80, "late")]:
                d_win = d_sd[(d_sd["time_sec"] >= lo) & (d_sd["time_sec"] <= hi)]
                if d_win.empty:
                    continue
                row = d_win.loc[d_win["auc"].idxmax()]
                rows.append(
                    {
                        "subject": int(subject),
                        "day": int(day),
                        "peak": label,
                        "peak_time_sec": float(row["time_sec"]),
                        "peak_auc": float(row["auc"]),
                        "window_start_sec": float(lo),
                        "window_end_sec": float(hi),
                    }
                )
        peak_df = pd.DataFrame(rows).sort_values(["day", "peak", "subject"])
        peak_df.to_csv(haufe_peak_times_csv, index=False)
        return peak_df

    def _make_haufe_info(pos_df):
        ch_names = pos_df["channel"].tolist()
        ch_pos = {
            r["channel"]: np.array([r["x"], r["y"], r["z"]], dtype=float)
            for _, r in pos_df.iterrows()
        }
        info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
        montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
        info.set_montage(montage, on_missing="ignore")
        return info, ch_names

    def _vector_corr(x_vec, y_vec):
        valid = np.isfinite(x_vec) & np.isfinite(y_vec)
        if int(np.sum(valid)) < 3:
            return np.nan
        x_use = x_vec[valid] - np.nanmean(x_vec[valid])
        y_use = y_vec[valid] - np.nanmean(y_vec[valid])
        denom = np.sqrt(np.sum(x_use ** 2) * np.sum(y_use ** 2))
        if (not np.isfinite(denom)) or denom <= np.finfo(float).eps:
            return np.nan
        return float(np.sum(x_use * y_use) / denom)

    def _write_haufe_stability(peak_df):
        if (not haufe_session_csv.exists()) or peak_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        d_sub = pd.read_csv(haufe_session_csv)
        if d_sub.empty:
            return pd.DataFrame(), pd.DataFrame()
        rows = []
        for day in sorted(peak_df["day"].dropna().unique().astype(int)):
            d_day = d_sub[d_sub["day"] == day].copy()
            times = np.sort(d_day["time_sec"].dropna().unique().astype(float))
            if len(times) == 0:
                continue
            for peak_label in ["early", "late"]:
                d_peak_meta = peak_df[(peak_df["day"] == day) & (peak_df["peak"] == peak_label)]
                if d_peak_meta.empty:
                    continue
                vec_rows = []
                for _, peak_row in d_peak_meta.iterrows():
                    subject = int(peak_row["subject"])
                    peak_time = float(peak_row["peak_time_sec"])
                    t_show = float(times[int(np.argmin(np.abs(times - peak_time)))])
                    d_peak = d_day[
                        (d_day["subject"] == subject) & np.isclose(d_day["time_sec"], t_show)
                    ].copy()
                    if d_peak.empty:
                        continue
                    vals = (
                        d_peak.set_index("channel")
                        .reindex(haufe_ch_names)["pattern"]
                        .to_numpy(dtype=float)
                    )
                    vec_rows.append(
                        {
                            "subject": subject,
                            "peak_time_sec": peak_time,
                            "used_time_sec": t_show,
                            "values": vals,
                        }
                    )
                if len(vec_rows) < 3:
                    continue
                mat = pd.DataFrame(
                    [r["values"] for r in vec_rows],
                    index=[r["subject"] for r in vec_rows],
                    columns=haufe_ch_names,
                )
                if mat.shape[0] < 3:
                    continue
                values = mat.to_numpy(dtype=float)
                subjects = mat.index.to_numpy()
                for i_sub, subject in enumerate(subjects):
                    x_vec = values[i_sub, :]
                    loo = np.delete(values, i_sub, axis=0)
                    with np.errstate(invalid="ignore"):
                        y_vec = np.nanmean(loo, axis=0)
                    r_val = _vector_corr(x_vec, y_vec)
                    rows.append(
                        {
                            "subject": int(subject),
                            "day": int(day),
                            "peak": peak_label,
                            "subject_peak_time_sec": float(vec_rows[i_sub]["peak_time_sec"]),
                            "used_time_sec": float(vec_rows[i_sub]["used_time_sec"]),
                            "loo_pattern_r": r_val,
                            "n_channels": int(np.sum(np.isfinite(x_vec) & np.isfinite(y_vec))),
                        }
                    )
        subject_df = pd.DataFrame(rows)
        if subject_df.empty:
            subject_df.to_csv(haufe_stability_subject_csv, index=False)
            pd.DataFrame().to_csv(haufe_stability_summary_csv, index=False)
            return subject_df, pd.DataFrame()
        summary_df = (
            subject_df.groupby(["day", "peak"], as_index=False)
            .agg(
                median_peak_time_sec=("subject_peak_time_sec", "median"),
                q25_peak_time_sec=("subject_peak_time_sec", lambda x: float(np.nanpercentile(x, 25))),
                q75_peak_time_sec=("subject_peak_time_sec", lambda x: float(np.nanpercentile(x, 75))),
                median_used_time_sec=("used_time_sec", "median"),
                median_r=("loo_pattern_r", "median"),
                q25_r=("loo_pattern_r", lambda x: float(np.nanpercentile(x, 25))),
                q75_r=("loo_pattern_r", lambda x: float(np.nanpercentile(x, 75))),
                mean_r=("loo_pattern_r", "mean"),
                prop_positive=("loo_pattern_r", lambda x: float(np.nanmean(np.asarray(x) > 0))),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["day", "peak"])
        )
        subject_df.to_csv(haufe_stability_subject_csv, index=False)
        summary_df.to_csv(haufe_stability_summary_csv, index=False)
        return subject_df, summary_df

    def _plot_haufe_stability(subject_df):
        if subject_df.empty:
            return
        peak_order = [p for p in ["early", "late"] if p in set(subject_df["peak"])]
        days_plot = sorted(subject_df["day"].dropna().unique().astype(int).tolist())
        fig, axes = plt.subplots(1, len(peak_order), figsize=(4.5 * len(peak_order), 4), sharey=True, squeeze=False)
        rng = np.random.default_rng(42)
        for ax, peak in zip(axes.ravel(), peak_order):
            data = []
            labels = []
            for day in days_plot:
                vals = subject_df[
                    (subject_df["day"] == day) & (subject_df["peak"] == peak)
                ]["loo_pattern_r"].dropna().to_numpy(dtype=float)
                data.append(vals)
                labels.append(f"D{day}")
            ax.boxplot(data, labels=labels, showfliers=False)
            for i_day, vals in enumerate(data, start=1):
                if len(vals) == 0:
                    continue
                jitter = rng.normal(0.0, 0.035, size=len(vals))
                ax.scatter(np.full(len(vals), i_day) + jitter, vals, s=18, alpha=0.55, color="#2f4f4f")
            ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
            ax.set_title(f"{peak} peak")
            ax.set_xlabel("Day")
            ax.grid(axis="y", alpha=0.25)
        axes.ravel()[0].set_ylabel("Subject vs leave-one-subject-out group pattern r")
        fig.suptitle("Haufe Pattern Stability at MVPA Peaks")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(fig_haufe_stability, dpi=150, bbox_inches="tight")
        plt.close(fig)

    haufe_df = pd.DataFrame()
    haufe_info = None
    haufe_ch_names = []
    peak_df = pd.DataFrame()
    peak_medians = {}
    if haufe_day_mean_csv.exists() and haufe_channel_pos_csv.exists():
        haufe_df = pd.read_csv(haufe_day_mean_csv)
        pos_df = pd.read_csv(haufe_channel_pos_csv)
        if (not haufe_df.empty) and (not pos_df.empty):
            haufe_info, haufe_ch_names = _make_haufe_info(pos_df)
            peak_df = _detect_subject_day_peak_times()
            if not peak_df.empty:
                d_peak_median = (
                    peak_df.groupby(["day", "peak"], as_index=False)["peak_time_sec"]
                    .median()
                    .rename(columns={"peak_time_sec": "median_peak_time_sec"})
                )
                peak_medians = {
                    (int(r["day"]), str(r["peak"])): float(r["median_peak_time_sec"])
                    for _, r in d_peak_median.iterrows()
                }

    haufe_stability_subject_df = pd.DataFrame()
    if (not peak_df.empty) and haufe_ch_names:
        haufe_stability_subject_df, _ = _write_haufe_stability(peak_df)
        _plot_haufe_stability(haufe_stability_subject_df)

    days = sorted(day_means_df["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 5.2), sharey=True, squeeze=False)
    x_all = day_means_df["time_sec"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_all))
    x_max = float(np.nanmax(x_all))
    y_upper = float(np.nanmax(day_means_df["auc_mean"] + day_means_df["auc_sem"].fillna(0.0)))
    y_lower = float(np.nanmin(day_means_df["auc_mean"] - day_means_df["auc_sem"].fillna(0.0)))
    y_pad = max(0.02, 0.20 * (y_upper - y_lower))
    topomap_ims = []
    if not haufe_df.empty:
        lim = float(np.nanmax(np.abs(haufe_df["pattern_mean"].to_numpy(dtype=float))))
        if not np.isfinite(lim) or lim <= 0:
            lim = 1e-12
    else:
        lim = 1e-12
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
        ax.set_ylim(y_lower - 0.02, y_upper + y_pad)
        ax.grid(alpha=0.25)
        day_peak_times = [
            (peak_medians[(int(day), peak_label)], peak_label)
            for peak_label in ["early", "late"]
            if (int(day), peak_label) in peak_medians
        ]
        for peak_time, peak_label in day_peak_times:
            if x_max <= x_min:
                continue
            y_peak = float(np.interp(peak_time, x, y))
            ax.axvline(peak_time, color="#b22222", linestyle=":", linewidth=1.2)
            ax.scatter(
                [peak_time],
                [y_peak],
                s=36,
                facecolor="white",
                edgecolor="#b22222",
                linewidth=1.2,
                zorder=4,
            )
            ax.text(
                peak_time,
                y_upper + (0.05 * y_pad),
                peak_label,
                color="#b22222",
                fontsize=8,
                ha="center",
                va="bottom",
            )
            x_frac = (peak_time - x_min) / (x_max - x_min)
            width = 0.18
            inset = ax.inset_axes(
                [max(0.01, min(0.99 - width, x_frac - width / 2.0)), 1.04, width, 0.36],
                transform=ax.transAxes,
            )
            d_day = haufe_df[haufe_df["day"] == day]
            if d_day.empty:
                inset.axis("off")
                continue
            times = np.sort(d_day["time_sec"].unique().astype(float))
            t_show = float(times[int(np.argmin(np.abs(times - peak_time)))])
            d_topo = d_day[np.isclose(d_day["time_sec"], t_show)]
            vals = (
                d_topo.set_index("channel")
                .reindex(haufe_ch_names)["pattern_mean"]
                .to_numpy(dtype=float)
            )
            im, _ = mne.viz.plot_topomap(
                vals,
                haufe_info,
                axes=inset,
                show=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-lim, lim),
                sphere=(0.0, 0.0, 0.0, 0.095),
            )
            topomap_ims.append(im)
            inset.set_title(f"{peak_label}\nmedian {peak_time:.3f}s\nmap {t_show:.3f}s", fontsize=7)
    axes.ravel()[0].set_ylabel("ROC-AUC")
    fig.suptitle("Time-resolved Category Decoding (Stim/A vs Stim/B)")
    if topomap_ims:
        cax = fig.add_axes([0.32, 0.89, 0.36, 0.025])
        fig.colorbar(topomap_ims[-1], cax=cax, orientation="horizontal", label="Haufe pattern")
        fig.tight_layout(rect=[0, 0, 1, 0.78])
    else:
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
    return {
        "figure_paths": {
            "day_panels": fig_day_panels,
            "day_slope": fig_day_slope,
            "haufe_stability": fig_haufe_stability,
        },
        "haufe_stability_subject_csv": haufe_stability_subject_csv,
        "haufe_stability_summary_csv": haufe_stability_summary_csv,
    }


def save_movie_mvpa_haufe_time_resolved(**kwargs):
    """Generate an animation linking time-resolved MVPA to evolving Haufe topomaps."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa"))
    tmin = float(kwargs.pop("tmin", 0.0))
    tmax = float(kwargs.pop("tmax", 0.80))
    fps = int(kwargs.pop("fps", 8))
    frame_step = int(kwargs.pop("frame_step", 1))
    dpi = int(kwargs.pop("dpi", 120))
    movie_path = figures_dir / kwargs.pop("movie_name", "mvpa_haufe_timecourse.mp4")
    gif_path = figures_dir / "mvpa_haufe_timecourse.gif"
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    day_means_csv = output_dir / "mvpa_day_means_timecourse.csv"
    haufe_day_mean_csv = output_dir / "mvpa_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_haufe_channel_positions.csv"
    required = [day_means_csv, haufe_day_mean_csv, haufe_channel_pos_csv]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs for MVPA/Haufe movie: {missing}")

    day_means_df = pd.read_csv(day_means_csv)
    haufe_df = pd.read_csv(haufe_day_mean_csv)
    pos_df = pd.read_csv(haufe_channel_pos_csv)
    if day_means_df.empty or haufe_df.empty or pos_df.empty:
        raise RuntimeError("MVPA/Haufe movie inputs are empty.")

    ch_names = pos_df["channel"].tolist()
    ch_pos = {
        r["channel"]: np.array([r["x"], r["y"], r["z"]], dtype=float)
        for _, r in pos_df.iterrows()
    }
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="ignore")

    all_times = np.array(sorted(haufe_df["time_sec"].dropna().unique().astype(float)), dtype=float)
    frame_times = all_times[(all_times >= tmin) & (all_times <= tmax)]
    frame_times = frame_times[::max(1, frame_step)]
    if len(frame_times) == 0:
        raise RuntimeError(f"No Haufe time points found in requested range {tmin}-{tmax}s.")

    days = sorted(day_means_df["day"].dropna().unique().astype(int).tolist())
    y_upper = float(np.nanmax(day_means_df["auc_mean"] + day_means_df["auc_sem"].fillna(0.0)))
    y_lower = float(np.nanmin(day_means_df["auc_mean"] - day_means_df["auc_sem"].fillna(0.0)))
    y_pad = max(0.02, 0.08 * (y_upper - y_lower))
    lim = float(np.nanmax(np.abs(haufe_df["pattern_mean"].to_numpy(dtype=float))))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1e-12

    from matplotlib import animation
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(4.2 * len(days), 7.0))
    gs = fig.add_gridspec(2, len(days), height_ratios=[1.0, 1.35], hspace=0.28, wspace=0.25)
    topo_axes = [fig.add_subplot(gs[0, i]) for i in range(len(days))]
    auc_axes = [fig.add_subplot(gs[1, i]) for i in range(len(days))]
    line_artists = []
    point_artists = []

    for ax, day in zip(auc_axes, days):
        g = day_means_df[day_means_df["day"] == day].sort_values("time_sec")
        x = g["time_sec"].to_numpy(dtype=float)
        y = g["auc_mean"].to_numpy(dtype=float)
        s = g["auc_sem"].to_numpy(dtype=float)
        ax.plot(x, y, color="tab:blue", linewidth=2)
        ax.fill_between(x, y - s, y + s, color="tab:blue", alpha=0.2, linewidth=0)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
        line = ax.axvline(frame_times[0], color="#b22222", linestyle="--", linewidth=1.8)
        y_now = float(np.interp(frame_times[0], x, y))
        point = ax.scatter(
            [frame_times[0]],
            [y_now],
            s=40,
            facecolor="white",
            edgecolor="#b22222",
            linewidth=1.3,
            zorder=4,
        )
        line_artists.append((line, x, y))
        point_artists.append(point)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(y_lower - 0.02, y_upper + y_pad)
        ax.grid(alpha=0.25)
    auc_axes[0].set_ylabel("ROC-AUC")

    cax = fig.add_axes([0.35, 0.93, 0.30, 0.025])
    sm = ScalarMappable(norm=Normalize(vmin=-lim, vmax=lim), cmap="RdBu_r")
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="horizontal", label="Haufe pattern")
    time_text = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=13)

    def _draw_frame(frame_i):
        t_current = float(frame_times[frame_i])
        time_text.set_text(f"MVPA/Haufe time = {t_current:.3f} s")
        for ax_topo, day in zip(topo_axes, days):
            ax_topo.clear()
            d_day = haufe_df[haufe_df["day"] == day]
            times = np.sort(d_day["time_sec"].dropna().unique().astype(float))
            t_show = float(times[int(np.argmin(np.abs(times - t_current)))])
            d_topo = d_day[np.isclose(d_day["time_sec"], t_show)]
            vals = (
                d_topo.set_index("channel")
                .reindex(ch_names)["pattern_mean"]
                .to_numpy(dtype=float)
            )
            mne.viz.plot_topomap(
                vals,
                info,
                axes=ax_topo,
                show=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-lim, lim),
                sphere=(0.0, 0.0, 0.0, 0.095),
            )
            ax_topo.set_title(f"Day {day} topomap", fontsize=10)
        for point, (line, x, y) in zip(point_artists, line_artists):
            line.set_xdata([t_current, t_current])
            point.set_offsets([[t_current, float(np.interp(t_current, x, y))]])
        return []

    anim_obj = animation.FuncAnimation(
        fig,
        _draw_frame,
        frames=len(frame_times),
        interval=1000 / max(1, fps),
        blit=False,
    )
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim_obj.save(movie_path, writer=writer, dpi=dpi)
        out_path = movie_path
    except Exception:
        writer = animation.PillowWriter(fps=fps)
        anim_obj.save(gif_path, writer=writer, dpi=dpi)
        out_path = gif_path
    plt.close(fig)
    return {"movie_path": out_path, "n_frames": int(len(frame_times))}


def run_mvpa_temporal_generalization(**kwargs):
    """Run temporal-generalization MVPA analysis."""
    return util_mvpa_temporal_generalization(save_figures=False, **kwargs)


def save_fig_mvpa_temporal_generalization(**kwargs):
    """Generate temporal-generalization MVPA figures."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa_tg_combined"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa_tg_combined"))
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
            "cross_day_timegen_matrices": out_cross.get("timegen_figure_path"),
        }
    }


def save_fig_mvpa_temporal_generalization_within_day(**kwargs):
    """Generate only time x time TG (within-day) figure from saved outputs."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa_tg_within_day"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa_tg_within_day"))
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
    """Generate cross-day scalar transfer and day-pair time x time figures."""
    output_dir = Path(kwargs.pop("output_dir", _OUTPUT_ROOT / "mvpa_tg_cross_day"))
    figures_dir = Path(kwargs.pop("figures_dir", _FIGURES_ROOT / "mvpa_tg_cross_day"))
    figures_dir.mkdir(parents=True, exist_ok=True)
    cross_day_mean_csv = output_dir / "tg_cross_day_day_mean.csv"
    cross_matrix_day_mean_csv = output_dir / "tg_cross_day_timegen_day_mean.csv"
    if not cross_day_mean_csv.exists():
        raise FileNotFoundError(
            f"Missing TG cross-day output in {output_dir}. Run run_mvpa_temporal_generalization() first."
        )
    cross_day_mean_df = pd.read_csv(cross_day_mean_csv)
    fig_cross = figures_dir / "tg_cross_day_transfer_5x4.png"
    fig_cross_timegen = figures_dir / "tg_cross_day_timegen_matrices_5x5.png"

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

    if cross_matrix_day_mean_csv.exists():
        d_mat = pd.read_csv(cross_matrix_day_mean_csv)
        if not d_mat.empty:
            fig, axes = plt.subplots(5, 5, figsize=(15, 15), squeeze=False)
            vmin = float(d_mat["auc_mean"].min())
            vmax = float(d_mat["auc_mean"].max())
            for i, train_day in enumerate(day_grid):
                for j, test_day in enumerate(day_grid):
                    ax = axes[i, j]
                    if train_day == test_day:
                        ax.axis("off")
                        ax.text(0.5, 0.5, f"D{train_day}=D{test_day}", ha="center", va="center")
                        continue
                    g = d_mat[
                        (d_mat["train_day"] == train_day)
                        & (d_mat["test_day"] == test_day)
                    ]
                    if g.empty:
                        ax.axis("off")
                        continue
                    pivot = g.pivot(index="train_time_sec", columns="test_time_sec", values="auc_mean")
                    im = ax.imshow(
                        pivot.to_numpy(),
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
                    ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
                    ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
                    ax.set_title(f"Train D{train_day} -> Test D{test_day}", fontsize=9)
                    if i == len(day_grid) - 1:
                        ax.set_xlabel("Test Time (s)")
                    if j == 0:
                        ax.set_ylabel("Train Time (s)")
            fig.suptitle("Cross-Day Temporal Generalization by Day Pair (AUC)")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="AUC")
            fig.subplots_adjust(top=0.94, wspace=0.30, hspace=0.35)
            fig.savefig(fig_cross_timegen, dpi=150, bbox_inches="tight")
            plt.close(fig)
    return {"figure_path": fig_cross, "timegen_figure_path": fig_cross_timegen}


def run_mvpa_temporal_generalization_within_day(**kwargs):
    """Run only within-day time x time temporal-generalization analysis."""
    kwargs.setdefault("output_dir", _OUTPUT_ROOT / "mvpa_tg_within_day")
    kwargs.setdefault("figures_dir", _FIGURES_ROOT / "mvpa_tg_within_day")
    return util_mvpa_temporal_generalization(
        save_figures=False,
        run_within_day=True,
        run_cross_day=False,
        **kwargs,
    )


def run_mvpa_temporal_generalization_cross_day(**kwargs):
    """Run only cross-day day x day temporal-generalization transfer analysis."""
    kwargs.setdefault("output_dir", _OUTPUT_ROOT / "mvpa_tg_cross_day")
    kwargs.setdefault("figures_dir", _FIGURES_ROOT / "mvpa_tg_cross_day")
    return util_mvpa_temporal_generalization(
        save_figures=False,
        run_within_day=True,
        run_cross_day=True,
        **kwargs,
    )


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
