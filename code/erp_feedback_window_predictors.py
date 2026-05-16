#!/usr/bin/env python3
"""Feedback-locked window amplitudes x feedback, boundary distance, and RPE."""

from __future__ import annotations

import os
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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from analysis_utils import model_term_summary, parallel_collect
from erp_frn_rpe_day1 import fit_rw_for_subject
from erp_n2_boundary import load_epoch_beh_sessions_with_boundary
from load_project_data import align_behaviour_to_epochs

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8

WINDOWS = {
    "frn_200_300": (0.200, 0.300),
    "positivity_300_400": (0.300, 0.400),
    "slow_450_800": (0.450, 0.800),
}
CHANNELS = ("Fz", "FCz")


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_mixed_or_ols(formula, df):
    try:
        fit = smf.mixedlm(formula, data=df, groups=df["subject"]).fit(
            reml=False, method="lbfgs", disp=False
        )
        return fit, "mixedlm"
    except Exception:
        fit = smf.ols(formula, data=df).fit()
        return fit, "ols_fallback"


def add_difficulty_conditioned_prediction_error(beh_df):
    beh_df = beh_df.copy()
    beh_df["difficulty_pred"] = np.nan
    beh_df["difficulty_rpe"] = np.nan
    needed = ["boundary_distance_abs", "trial", "fb"]
    if any(col not in beh_df.columns for col in needed):
        return beh_df
    d = beh_df.dropna(subset=needed).copy()
    if len(d) < 10:
        return beh_df
    outcome = (d["fb"].astype(str).str.lower() == "correct").astype(int).to_numpy()
    if np.unique(outcome).size < 2:
        return beh_df
    trial = pd.to_numeric(d["trial"], errors="coerce").to_numpy(dtype=float)
    trial_position = (trial - np.nanmin(trial)) / max(np.nanmax(trial) - np.nanmin(trial), 1.0)
    features = np.column_stack(
        [
            pd.to_numeric(d["boundary_distance_abs"], errors="coerce").to_numpy(dtype=float),
            trial_position,
        ]
    )
    valid = np.isfinite(features).all(axis=1)
    if np.sum(valid) < 10 or np.unique(outcome[valid]).size < 2:
        return beh_df
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000),
            ),
        ]
    )
    clf.fit(features[valid], outcome[valid])
    pred = np.full(len(d), np.nan, dtype=float)
    pred[valid] = clf.predict_proba(features[valid])[:, 1]
    beh_df.loc[d.index, "difficulty_pred"] = pred
    beh_df.loc[d.index, "difficulty_rpe"] = outcome - pred
    return beh_df


def extract_feedback_window_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    try:
        beh_df = task["beh"].sort_values("trial").reset_index(drop=True).copy()
        if day == 1:
            outcomes = (
                beh_df["fb"].astype(str).str.lower() == "correct"
            ).astype(float).to_numpy()
            alpha, nll, pred, rpe = fit_rw_for_subject(outcomes)
            beh_df["rw_alpha"] = alpha
            beh_df["rw_nll"] = nll
            beh_df["rw_pred"] = pred
            beh_df["rpe"] = rpe
            beh_df = add_difficulty_conditioned_prediction_error(beh_df)
        else:
            beh_df["rw_alpha"] = np.nan
            beh_df["rw_nll"] = np.nan
            beh_df["rw_pred"] = np.nan
            beh_df["rpe"] = np.nan
            beh_df["difficulty_pred"] = np.nan
            beh_df["difficulty_rpe"] = np.nan

        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        epochs_fb, beh_aligned = align_behaviour_to_epochs(
            beh_df, epochs, event_names=("FB/Cor", "FB/Inc")
        )
        if len(epochs_fb) == 0:
            return {
                "ok": False,
                "qc": {"subject": subject, "day": day, "reason": "no_feedback_epochs", "detail": ""},
            }
        channels = [ch for ch in CHANNELS if ch in epochs_fb.ch_names]
        if len(channels) == 0:
            return {
                "ok": False,
                "qc": {"subject": subject, "day": day, "reason": "missing_channels", "detail": ""},
            }
        epochs_fb = epochs_fb.copy().load_data().pick(channels)
        data = epochs_fb.get_data()
        times = epochs_fb.times
        rows = []
        base = beh_aligned.reset_index(drop=True).copy()
        base["subject"] = subject
        base["day"] = day
        base["feedback"] = base["fb"].astype(str).str.lower()
        base["is_incorrect"] = (base["feedback"] == "incorrect").astype(float)
        base["is_correct"] = (base["feedback"] == "correct").astype(float)
        for window, (tmin, tmax) in WINDOWS.items():
            tmask = (times >= tmin) & (times <= tmax)
            if not np.any(tmask):
                continue
            d = base.copy()
            amp = data[:, :, tmask].mean(axis=(1, 2))
            d["window"] = window
            d["window_tmin"] = tmin
            d["window_tmax"] = tmax
            d["amplitude_v"] = amp
            d["amplitude_uv"] = amp * 1e6
            rows.append(d)
        if len(rows) == 0:
            return {
                "ok": False,
                "qc": {"subject": subject, "day": day, "reason": "no_window_samples", "detail": ""},
            }
        return {"ok": True, "rows": pd.concat(rows, ignore_index=True)}
    except Exception as exc:
        return {
            "ok": False,
            "qc": {"subject": subject, "day": day, "reason": "extract_error", "detail": str(exc)},
        }


def subject_window_coefficients(trial_df):
    rows = []
    predictors = ["is_incorrect", "boundary_distance_abs", "is_incorrect:boundary_distance_abs"]
    for (subject, day, window), g in trial_df.groupby(["subject", "day", "window"]):
        d = g.dropna(subset=["amplitude_uv", "boundary_distance_abs", "is_incorrect"]).copy()
        if len(d) < 10 or d["is_incorrect"].nunique() < 2 or d["boundary_distance_abs"].nunique() < 2:
            continue
        fit = smf.ols(
            "amplitude_uv ~ is_incorrect * boundary_distance_abs", data=d
        ).fit()
        for predictor in predictors:
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "window": window,
                    "predictor": predictor,
                    "estimate": float(fit.params.get(predictor, np.nan)),
                    "p_value": float(fit.pvalues.get(predictor, np.nan)),
                    "n_trials": int(len(d)),
                }
            )
    return pd.DataFrame(rows)


def day1_rpe_coefficients(trial_df):
    rows = []
    d1 = trial_df[trial_df["day"] == 1].dropna(
        subset=["amplitude_uv", "rpe", "boundary_distance_abs", "is_incorrect"]
    )
    for (subject, window), g in d1.groupby(["subject", "window"]):
        if len(g) < 10 or g["rpe"].nunique() < 2:
            continue
        fit = smf.ols(
            "amplitude_uv ~ rpe + boundary_distance_abs + is_incorrect", data=g
        ).fit()
        for predictor in ["rpe", "boundary_distance_abs", "is_incorrect"]:
            rows.append(
                {
                    "subject": int(subject),
                    "window": window,
                    "predictor": predictor,
                    "estimate": float(fit.params.get(predictor, np.nan)),
                    "p_value": float(fit.pvalues.get(predictor, np.nan)),
                    "n_trials": int(len(g)),
                }
            )
    return pd.DataFrame(rows)


def day1_separate_coefficients(trial_df):
    rows = []
    d1 = trial_df[trial_df["day"] == 1].copy()
    models = [
        ("rpe", "amplitude_uv ~ rpe", ["amplitude_uv", "rpe"]),
        (
            "difficulty_rpe",
            "amplitude_uv ~ difficulty_rpe",
            ["amplitude_uv", "difficulty_rpe"],
        ),
        (
            "boundary_distance_abs",
            "amplitude_uv ~ boundary_distance_abs",
            ["amplitude_uv", "boundary_distance_abs"],
        ),
        ("is_incorrect", "amplitude_uv ~ is_incorrect", ["amplitude_uv", "is_incorrect"]),
    ]
    for (subject, window), g in d1.groupby(["subject", "window"]):
        for predictor, formula, cols in models:
            d = g.dropna(subset=cols).copy()
            if len(d) < 10 or d[predictor].nunique() < 2:
                continue
            fit = smf.ols(formula, data=d).fit()
            rows.append(
                {
                    "subject": int(subject),
                    "window": window,
                    "predictor": predictor,
                    "estimate": float(fit.params[predictor]),
                    "p_value": float(fit.pvalues[predictor]),
                    "n_trials": int(len(d)),
                    "model": formula,
                }
            )
    return pd.DataFrame(rows)


def all_days_separate_coefficients(trial_df):
    rows = []
    models = [
        (
            "boundary_distance_abs",
            "amplitude_uv ~ boundary_distance_abs",
            ["amplitude_uv", "boundary_distance_abs"],
        ),
        ("is_incorrect", "amplitude_uv ~ is_incorrect", ["amplitude_uv", "is_incorrect"]),
    ]
    for (subject, day, window), g in trial_df.groupby(["subject", "day", "window"]):
        for predictor, formula, cols in models:
            d = g.dropna(subset=cols).copy()
            if len(d) < 10 or d[predictor].nunique() < 2:
                continue
            fit = smf.ols(formula, data=d).fit()
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "window": window,
                    "predictor": predictor,
                    "estimate": float(fit.params[predictor]),
                    "p_value": float(fit.pvalues[predictor]),
                    "n_trials": int(len(d)),
                    "model": formula,
                }
            )
    return pd.DataFrame(rows)


def correct_only_boundary_coefficients(trial_df):
    rows = []
    correct_df = trial_df[trial_df["feedback"] == "correct"].copy()
    for (subject, day, window), g in correct_df.groupby(["subject", "day", "window"]):
        d = g.dropna(subset=["amplitude_uv", "boundary_distance_abs"]).copy()
        if len(d) < 10 or d["boundary_distance_abs"].nunique() < 2:
            continue
        fit = smf.ols("amplitude_uv ~ boundary_distance_abs", data=d).fit()
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "window": window,
                "predictor": "boundary_distance_abs",
                "estimate": float(fit.params["boundary_distance_abs"]),
                "p_value": float(fit.pvalues["boundary_distance_abs"]),
                "n_trials": int(len(d)),
                "model": "correct_only: amplitude_uv ~ boundary_distance_abs",
            }
        )
    return pd.DataFrame(rows)


def predictor_correlations(trial_df):
    rows = []
    cols = ["rpe", "difficulty_rpe", "boundary_distance_abs", "is_incorrect"]
    for (day, window), g in trial_df.groupby(["day", "window"]):
        d = g[cols].dropna()
        for i, x in enumerate(cols):
            for y in cols[i + 1 :]:
                pair = d[[x, y]].dropna()
                if len(pair) < 3 or pair[x].nunique() < 2 or pair[y].nunique() < 2:
                    r_val = np.nan
                    p_val = np.nan
                else:
                    res = stats.pearsonr(pair[x], pair[y])
                    r_val = float(res.statistic)
                    p_val = float(res.pvalue)
                rows.append(
                    {
                        "day": int(day),
                        "window": window,
                        "x": x,
                        "y": y,
                        "r": r_val,
                        "p_value": p_val,
                        "n_trials": int(len(pair)),
                    }
                )
    return pd.DataFrame(rows)


def group_ttests(coef_df, group_cols=("window", "predictor")):
    rows = []
    if coef_df.empty:
        return pd.DataFrame(rows)
    for keys, g in coef_df.groupby(list(group_cols)):
        vals = g["estimate"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            t_val = np.nan
            p_val = np.nan
        else:
            res = stats.ttest_1samp(vals, 0.0, nan_policy="omit")
            t_val = float(res.statistic)
            p_val = float(res.pvalue)
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        row.update(
            {
                "n_subjects": int(g["subject"].nunique()),
                "mean_estimate": float(np.nanmean(vals)) if len(vals) else np.nan,
                "sem_estimate": _sem(vals),
                "t": t_val,
                "p_value": p_val,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def mixed_model_terms(trial_df):
    rows = []
    formula = "amplitude_uv ~ is_incorrect * boundary_distance_abs * day"
    for window, g in trial_df.groupby("window"):
        d = g.dropna(subset=["amplitude_uv", "is_incorrect", "boundary_distance_abs", "day"]).copy()
        if len(d) < 20 or d["subject"].nunique() < 2:
            continue
        fit, status = _fit_mixed_or_ols(formula, d)
        for term in fit.model.exog_names:
            if term == "Intercept":
                continue
            item = model_term_summary(fit, term)
            item["window"] = window
            item["status"] = status
            item["n_subjects"] = int(d["subject"].nunique())
            item["n_trials"] = int(len(d))
            rows.append(item)
    return pd.DataFrame(rows)


def plot_feedback_difference_by_day(trial_df, fig_path):
    d = trial_df.dropna(subset=["amplitude_uv", "feedback"]).copy()
    summary = (
        d.groupby(["subject", "day", "window", "feedback"], as_index=False)["amplitude_uv"]
        .mean()
        .pivot_table(index=["subject", "day", "window"], columns="feedback", values="amplitude_uv")
        .reset_index()
    )
    if not {"correct", "incorrect"} <= set(summary.columns):
        return
    summary["incorrect_minus_correct"] = summary["incorrect"] - summary["correct"]
    agg = (
        summary.groupby(["day", "window"], as_index=False)
        .agg(
            mean_diff=("incorrect_minus_correct", "mean"),
            sem_diff=("incorrect_minus_correct", _sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["window", "day"])
    )
    windows = list(WINDOWS.keys())
    fig, axes = plt.subplots(1, len(windows), figsize=(4.5 * len(windows), 4), squeeze=False)
    for ax, window in zip(axes[0], windows):
        g = agg[agg["window"] == window]
        ax.errorbar(
            g["day"],
            g["mean_diff"],
            yerr=g["sem_diff"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            color="tab:blue",
        )
        ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
        ax.set_title(window.replace("_", " "))
        ax.set_xlabel("Day")
        ax.set_ylabel("Incorrect - correct (uV)")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_day1_prediction_error_coefficients(ttest_df, fig_path):
    d = ttest_df[ttest_df["predictor"].isin(["rpe", "difficulty_rpe"])].copy()
    if d.empty:
        return
    order = list(WINDOWS.keys())
    d["window"] = pd.Categorical(d["window"], categories=order, ordered=True)
    d = d.sort_values(["window", "predictor"])
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x_base = np.arange(len(order))
    width = 0.34
    colors = {"rpe": "tab:purple", "difficulty_rpe": "tab:green"}
    offsets = {"rpe": -width / 2, "difficulty_rpe": width / 2}
    labels = {"rpe": "RW RPE", "difficulty_rpe": "Difficulty PE"}
    for predictor in ["rpe", "difficulty_rpe"]:
        g = d[d["predictor"] == predictor].set_index("window").reindex(order)
        x = x_base + offsets[predictor]
        ax.bar(
            x,
            g["mean_estimate"],
            width=width,
            yerr=g["sem_estimate"],
            color=colors[predictor],
            alpha=0.8,
            capsize=3,
            label=labels[predictor],
        )
    ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
    ax.set_xticks(x_base)
    ax.set_xticklabels([str(w).replace("_", "\n") for w in order])
    ax.set_ylabel("Day 1 prediction-error coefficient (uV/unit)")
    ax.set_title("Feedback-window amplitude by prediction error")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_feedback_window_predictor_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sessions, _ = load_epoch_beh_sessions_with_boundary()
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    tasks = [
        {"subject": s["subject"], "day": s["day"], "epo_path": s["epo_path"], "beh": s["beh"]}
        for s in sessions
    ]
    print(
        f"[ERP feedback windows] Extracting {len(tasks)} sessions "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    results = parallel_collect(extract_feedback_window_session, tasks, n_workers)
    rows = []
    qc = []
    for result in results:
        if result["ok"]:
            rows.append(result["rows"])
        else:
            qc.append(result["qc"])
    trial_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if trial_df.empty:
        raise RuntimeError("No feedback-window rows extracted.")
    trial_df = trial_df.dropna(subset=["amplitude_uv"]).copy()
    subject_coef_df = subject_window_coefficients(trial_df)
    subject_ttest_df = group_ttests(subject_coef_df)
    rpe_coef_df = day1_rpe_coefficients(trial_df)
    rpe_ttest_df = group_ttests(rpe_coef_df)
    day1_separate_coef_df = day1_separate_coefficients(trial_df)
    day1_separate_ttest_df = group_ttests(day1_separate_coef_df)
    all_days_separate_coef_df = all_days_separate_coefficients(trial_df)
    all_days_separate_ttest_df = group_ttests(
        all_days_separate_coef_df, group_cols=("window", "day", "predictor")
    )
    correct_only_coef_df = correct_only_boundary_coefficients(trial_df)
    correct_only_ttest_df = group_ttests(
        correct_only_coef_df, group_cols=("window", "day", "predictor")
    )
    correlation_df = predictor_correlations(trial_df)
    model_df = mixed_model_terms(trial_df)

    trial_csv = output_dir / "erp_feedback_window_trial_level.csv"
    subject_coef_csv = output_dir / "erp_feedback_window_subject_day_coefficients.csv"
    subject_ttest_csv = output_dir / "erp_feedback_window_subject_day_ttests.csv"
    rpe_coef_csv = output_dir / "erp_feedback_window_day1_rpe_coefficients.csv"
    rpe_ttest_csv = output_dir / "erp_feedback_window_day1_rpe_ttests.csv"
    day1_separate_coef_csv = (
        output_dir / "erp_feedback_window_day1_separate_coefficients.csv"
    )
    day1_separate_ttest_csv = output_dir / "erp_feedback_window_day1_separate_ttests.csv"
    all_days_separate_coef_csv = (
        output_dir / "erp_feedback_window_all_days_separate_coefficients.csv"
    )
    all_days_separate_ttest_csv = (
        output_dir / "erp_feedback_window_all_days_separate_ttests.csv"
    )
    correct_only_coef_csv = output_dir / "erp_feedback_window_correct_only_boundary_coefficients.csv"
    correct_only_ttest_csv = output_dir / "erp_feedback_window_correct_only_boundary_ttests.csv"
    correlation_csv = output_dir / "erp_feedback_window_predictor_correlations.csv"
    model_csv = output_dir / "erp_feedback_window_mixed_model_terms.csv"
    qc_csv = output_dir / "erp_feedback_window_qc.csv"
    diff_fig = figures_dir / "erp_feedback_window_difference_by_day.png"
    rpe_fig = figures_dir / "erp_feedback_window_day1_rpe_coefficients.png"

    trial_df.to_csv(trial_csv, index=False)
    subject_coef_df.to_csv(subject_coef_csv, index=False)
    subject_ttest_df.to_csv(subject_ttest_csv, index=False)
    rpe_coef_df.to_csv(rpe_coef_csv, index=False)
    rpe_ttest_df.to_csv(rpe_ttest_csv, index=False)
    day1_separate_coef_df.to_csv(day1_separate_coef_csv, index=False)
    day1_separate_ttest_df.to_csv(day1_separate_ttest_csv, index=False)
    all_days_separate_coef_df.to_csv(all_days_separate_coef_csv, index=False)
    all_days_separate_ttest_df.to_csv(all_days_separate_ttest_csv, index=False)
    correct_only_coef_df.to_csv(correct_only_coef_csv, index=False)
    correct_only_ttest_df.to_csv(correct_only_ttest_csv, index=False)
    correlation_df.to_csv(correlation_csv, index=False)
    model_df.to_csv(model_csv, index=False)
    pd.DataFrame(qc).to_csv(qc_csv, index=False)
    plot_feedback_difference_by_day(trial_df, diff_fig)
    plot_day1_prediction_error_coefficients(day1_separate_ttest_df, rpe_fig)
    print(
        f"[ERP feedback windows] Done: {trial_df['subject'].nunique()} subjects, "
        f"{trial_df['day'].nunique()} days, {len(trial_df)} window rows.",
        flush=True,
    )
    return {
        "trial_df": trial_df,
        "subject_coef_df": subject_coef_df,
        "subject_ttest_df": subject_ttest_df,
        "rpe_coef_df": rpe_coef_df,
        "rpe_ttest_df": rpe_ttest_df,
        "day1_separate_coef_df": day1_separate_coef_df,
        "day1_separate_ttest_df": day1_separate_ttest_df,
        "all_days_separate_coef_df": all_days_separate_coef_df,
        "all_days_separate_ttest_df": all_days_separate_ttest_df,
        "correct_only_coef_df": correct_only_coef_df,
        "correct_only_ttest_df": correct_only_ttest_df,
        "correlation_df": correlation_df,
        "model_df": model_df,
        "trial_csv": trial_csv,
        "subject_coef_csv": subject_coef_csv,
        "subject_ttest_csv": subject_ttest_csv,
        "rpe_coef_csv": rpe_coef_csv,
        "rpe_ttest_csv": rpe_ttest_csv,
        "day1_separate_coef_csv": day1_separate_coef_csv,
        "day1_separate_ttest_csv": day1_separate_ttest_csv,
        "all_days_separate_coef_csv": all_days_separate_coef_csv,
        "all_days_separate_ttest_csv": all_days_separate_ttest_csv,
        "correct_only_coef_csv": correct_only_coef_csv,
        "correct_only_ttest_csv": correct_only_ttest_csv,
        "correlation_csv": correlation_csv,
        "model_csv": model_csv,
        "qc_csv": qc_csv,
        "figures": [diff_fig, rpe_fig],
    }


if __name__ == "__main__":
    run_feedback_window_predictor_analysis()
