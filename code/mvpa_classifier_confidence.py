#!/usr/bin/env python3
"""MVPA classifier confidence time-resolved analysis."""

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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.model_selection import StratifiedKFold

from analysis_utils import model_term_summary, parallel_collect
from erp_n2_boundary import load_epoch_beh_sessions_with_boundary
from load_project_data import align_behaviour_to_epochs
from mvpa_tg_within_day import build_clf, pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8

WINDOWS = {
    "early": (0.060, 0.180),
    "late": (0.250, 0.550),
}


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_regression_with_fallback(formula, df, group_col="subject"):
    try:
        model = smf.mixedlm(formula, data=df, groups=df[group_col]).fit(
            reml=False, method="lbfgs", disp=False
        )
        return model, "mixedlm"
    except Exception as exc:
        model = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[group_col]}
        )
        model._fallback_detail = str(exc)
        return model, "ols_cluster_fallback"


def cross_validated_decision_values(X, y, random_state):
    n_trials, _, n_times = X.shape
    decisions = np.full((n_trials, n_times), np.nan, dtype=float)
    min_class = int(min(np.sum(y == 0), np.sum(y == 1)))
    if min_class < 2:
        return decisions
    n_splits = min(5, min_class)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    for ti in range(n_times):
        Xt = X[:, :, ti]
        for train_idx, test_idx in cv.split(Xt, y):
            clf = build_clf(random_state=random_state)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SklearnConvergenceWarning)
                    clf.fit(Xt[train_idx], y[train_idx])
                decisions[test_idx, ti] = clf.decision_function(Xt[test_idx])
            except Exception:
                decisions[test_idx, ti] = np.nan
    return decisions


def _ols_slope(formula, df, term):
    if len(df) < 10:
        return np.nan, np.nan
    try:
        fit = smf.ols(formula, data=df).fit()
        return float(fit.params.get(term, np.nan)), float(fit.pvalues.get(term, np.nan))
    except Exception:
        return np.nan, np.nan


def process_classifier_confidence_session(task):
    session_file = task["epo_file"]
    subject = int(task["subject"])
    day = int(task["day"])
    min_epochs = int(task["min_epochs"])
    random_state = int(task["random_state"])
    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        epochs_stim, beh_aligned = align_behaviour_to_epochs(
            task["beh"], epochs, event_names=("Stim/A", "Stim/B")
        )
        epochs_stim = epochs_stim.copy().load_data()
        pick_eeg_interpolate_bads(epochs_stim)
        epochs_stim.resample(128, npad="auto")
        codes = epochs_stim.events[:, 2]
        y = np.full(len(codes), -1, dtype=int)
        y[codes == epochs_stim.event_id["Stim/A"]] = 0
        y[codes == epochs_stim.event_id["Stim/B"]] = 1
        keep = y >= 0
        y = y[keep]
        X = epochs_stim.get_data()[keep]
        beh = beh_aligned.iloc[np.where(keep)[0]].reset_index(drop=True).copy()
        n_a = int(np.sum(y == 0))
        n_b = int(np.sum(y == 1))
        n_trials = int(len(y))
        if n_trials < min_epochs:
            raise ValueError(f"insufficient_epochs:n_trials={n_trials} < min_epochs={min_epochs}")
        if min(n_a, n_b) < 5:
            raise ValueError(f"insufficient_class_trials:n_a={n_a}, n_b={n_b}")
        decisions = cross_validated_decision_values(X, y, random_state=random_state)
        times = epochs_stim.times.copy()
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "reason": "prep_or_decode_error",
                "detail": str(exc),
            },
        }

    sign = np.where(y == 1, 1.0, -1.0)
    confidence = np.abs(decisions)
    signed_evidence = decisions * sign[:, None]
    session_rows = []
    for ti, time_sec in enumerate(times):
        d = pd.DataFrame(
            {
                "confidence": confidence[:, ti],
                "signed_evidence": signed_evidence[:, ti],
                "boundary_distance": pd.to_numeric(beh["boundary_distance"], errors="coerce"),
                "boundary_distance_abs": pd.to_numeric(
                    beh["boundary_distance_abs"], errors="coerce"
                ),
                "rt_sec": pd.to_numeric(beh["rt_sec"], errors="coerce"),
                "accuracy": pd.to_numeric(beh["accuracy"], errors="coerce"),
            }
        ).dropna()
        boundary_slope, boundary_p = _ols_slope(
            "confidence ~ boundary_distance_abs", d, "boundary_distance_abs"
        )
        evidence_slope, evidence_p = _ols_slope(
            "signed_evidence ~ boundary_distance", d, "boundary_distance"
        )
        rt_slope, rt_p = _ols_slope("rt_sec ~ confidence", d, "confidence")
        acc_slope, acc_p = _ols_slope("accuracy ~ confidence", d, "confidence")
        session_rows.append(
            {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "time_sec": float(time_sec),
                "mean_confidence": float(np.nanmean(confidence[:, ti])),
                "mean_signed_evidence": float(np.nanmean(signed_evidence[:, ti])),
                "confidence_boundary_abs_slope": boundary_slope,
                "confidence_boundary_abs_p": boundary_p,
                "signed_evidence_boundary_slope": evidence_slope,
                "signed_evidence_boundary_p": evidence_p,
                "rt_confidence_slope": rt_slope,
                "rt_confidence_p": rt_p,
                "accuracy_confidence_slope": acc_slope,
                "accuracy_confidence_p": acc_p,
                "n_trials": n_trials,
                "n_a": n_a,
                "n_b": n_b,
            }
        )
    return {"ok": True, "session_rows": session_rows}


def plot_classifier_confidence(day_mean_df, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
    metrics = [
        (axes[0, 0], "mean_confidence", "Classifier confidence"),
        (axes[0, 1], "confidence_boundary_abs_slope", "Confidence slope on boundary distance"),
    ]
    days = sorted(day_mean_df["day"].dropna().unique())
    cmap = plt.get_cmap("viridis", max(len(days), 2))
    for ax, metric, ylabel in metrics:
        for idx, day in enumerate(days):
            g = day_mean_df[day_mean_df["day"] == day].sort_values("time_sec")
            ax.plot(
                g["time_sec"],
                g[metric],
                linewidth=1.6,
                color=cmap(idx),
                label=f"Day {int(day)}",
            )
        ax.axvline(0, color="0.35", linestyle=":", linewidth=1)
        ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
        ax.set_xlabel("Time from stimulus (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(frameon=False, ncol=1)
    fig.suptitle("Time-Resolved MVPA Classifier Confidence")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def summarize_windows(session_df):
    rows = []
    for (subject, day), g in session_df.groupby(["subject", "day"]):
        for window, (tmin, tmax) in WINDOWS.items():
            w = g[(g["time_sec"] >= tmin) & (g["time_sec"] <= tmax)]
            if w.empty:
                continue
            row = {"subject": int(subject), "day": int(day), "window": window}
            for col in [
                "mean_confidence",
                "mean_signed_evidence",
                "confidence_boundary_abs_slope",
                "signed_evidence_boundary_slope",
                "rt_confidence_slope",
                "accuracy_confidence_slope",
            ]:
                row[col] = float(w[col].mean())
            rows.append(row)
    return pd.DataFrame(rows)


def fit_window_models(window_df):
    rows = []
    if window_df.empty:
        return pd.DataFrame()
    for metric in [
        "mean_confidence",
        "confidence_boundary_abs_slope",
        "signed_evidence_boundary_slope",
        "rt_confidence_slope",
    ]:
        d = window_df.dropna(subset=[metric]).copy()
        if d.empty:
            continue
        model, status = _fit_regression_with_fallback(f"{metric} ~ window * day", d)
        for term in model.model.exog_names:
            if term == "Intercept":
                continue
            item = model_term_summary(model, term)
            item.update({"metric": metric, "status": status})
            if hasattr(model, "_fallback_detail"):
                item["fallback_detail"] = model._fallback_detail
            rows.append(item)
    return pd.DataFrame(rows)


def run_classifier_confidence_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    min_epochs: int = 20,
    random_state: int = 42,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    sessions, _ = load_epoch_beh_sessions_with_boundary()
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    tasks = [
        {
            "subject": s["subject"],
            "day": s["day"],
            "epo_file": Path(s["epo_path"]).name,
            "epo_path": str(s["epo_path"]),
            "beh": s["beh"],
            "min_epochs": int(min_epochs),
            "random_state": int(random_state),
        }
        for s in sessions
    ]
    progress_json = output_dir / "mvpa_classifier_confidence_progress.json"
    t0 = time.time()
    progress_json.write_text(
        json.dumps({"stage": "running", "done": 0, "total": len(tasks), "elapsed_sec": 0.0}, indent=2)
    )
    print(
        f"[MVPA classifier confidence] Running {len(tasks)} sessions (n_workers={n_workers})...",
        flush=True,
    )
    results = parallel_collect(process_classifier_confidence_session, tasks, n_workers)
    session_rows = []
    qc = []
    for result in results:
        if result["ok"]:
            session_rows.extend(result["session_rows"])
        else:
            qc.append(result["qc"])
    progress_json.write_text(
        json.dumps(
            {
                "stage": "completed",
                "done": len(tasks),
                "total": len(tasks),
                "elapsed_sec": float(time.time() - t0),
            },
            indent=2,
        )
    )
    session_df = pd.DataFrame(session_rows)
    if session_df.empty:
        raise RuntimeError("No classifier-confidence rows extracted.")
    day_mean_df = (
        session_df.groupby(["day", "time_sec"], as_index=False)
        .agg(
            mean_confidence=("mean_confidence", "mean"),
            mean_signed_evidence=("mean_signed_evidence", "mean"),
            confidence_boundary_abs_slope=("confidence_boundary_abs_slope", "mean"),
            signed_evidence_boundary_slope=("signed_evidence_boundary_slope", "mean"),
            rt_confidence_slope=("rt_confidence_slope", "mean"),
            accuracy_confidence_slope=("accuracy_confidence_slope", "mean"),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "time_sec"])
    )
    window_df = summarize_windows(session_df)
    model_df = fit_window_models(window_df)
    session_csv = output_dir / "mvpa_classifier_confidence_session_timecourse.csv"
    day_mean_csv = output_dir / "mvpa_classifier_confidence_day_mean_timecourse.csv"
    window_csv = output_dir / "mvpa_classifier_confidence_window_subject_day.csv"
    model_csv = output_dir / "mvpa_classifier_confidence_window_model_terms.csv"
    qc_csv = output_dir / "mvpa_classifier_confidence_qc.csv"
    fig_path = figures_dir / "mvpa_classifier_confidence_timecourse.png"
    session_df.to_csv(session_csv, index=False)
    day_mean_df.to_csv(day_mean_csv, index=False)
    window_df.to_csv(window_csv, index=False)
    model_df.to_csv(model_csv, index=False)
    pd.DataFrame(qc).to_csv(qc_csv, index=False)
    plot_classifier_confidence(day_mean_df, fig_path)
    return {
        "session_df": session_df,
        "day_mean_df": day_mean_df,
        "window_df": window_df,
        "model_df": model_df,
        "session_csv": session_csv,
        "day_mean_csv": day_mean_csv,
        "window_csv": window_csv,
        "model_csv": model_csv,
        "qc_csv": qc_csv,
        "progress_json": progress_json,
        "figure": fig_path,
    }


if __name__ == "__main__":
    run_classifier_confidence_analysis()
