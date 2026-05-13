#!/usr/bin/env python3
"""Frontal N2 × boundary distance analysis and individual differences."""

from __future__ import annotations

import os
import re
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

from analysis_utils import model_term_summary, parallel_collect
from boundary_distance import load_behaviour_with_boundary
from load_project_data import align_behaviour_to_epochs

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "erp_n2_boundary"
FIGURES_DIR = PROJECT_DIR / "figures" / "erp_n2_boundary"
N_JOBS = 8


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_regression(formula, df, group_col="subject", re_formula=None):
    if re_formula is None:
        model = smf.mixedlm(formula, data=df, groups=df[group_col]).fit(
            reml=False, method="lbfgs", disp=False
        )
    else:
        model = smf.mixedlm(
            formula, data=df, groups=df[group_col], re_formula=re_formula
        ).fit(reml=False, method="lbfgs", disp=False)
    return model, "mixedlm"


def load_epoch_beh_sessions_with_boundary(project_dir: Path | str = PROJECT_DIR):
    project_dir = Path(project_dir)
    beh, boundary = load_behaviour_with_boundary(project_dir)
    epo_dir = project_dir / "EEG_epo"
    epo_re = re.compile(r"^P(\d+)_D([\d_]+)-epo\.fif$")
    sessions = []
    for epo_path in sorted(epo_dir.glob("*-epo.fif")):
        m = epo_re.match(epo_path.name)
        if m is None:
            continue
        subject = int(m.group(1))
        day = int(m.group(2).split("_")[0])
        beh_df = beh[(beh["subject"] == subject) & (beh["day"] == day)].copy()
        if beh_df.empty:
            continue
        sessions.append({"subject": subject, "day": day, "epo_path": epo_path, "beh": beh_df})
    return sessions, boundary


def extract_n2_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        epochs_stim, beh_aligned = align_behaviour_to_epochs(
            task["beh"], epochs, event_names=("Stim/A", "Stim/B")
        )
        if len(epochs_stim) == 0:
            return {
                "ok": False,
                "qc": {"subject": subject, "day": day, "reason": "no_stim_epochs", "detail": ""},
            }
        epochs_stim = epochs_stim.copy().load_data()
        channels = [ch for ch in ["Fz", "FCz", "FC1", "FC2"] if ch in epochs_stim.ch_names]
        if len(channels) == 0:
            return {
                "ok": False,
                "qc": {"subject": subject, "day": day, "reason": "missing_channels", "detail": ""},
            }
        data = epochs_stim.copy().pick(channels).get_data()
        times = epochs_stim.times
        tmask = (times >= 0.200) & (times <= 0.300)
        amp = data[:, :, tmask].mean(axis=(1, 2))
        out = beh_aligned.reset_index(drop=True).copy()
        out["subject"] = subject
        out["day"] = day
        out["n2_amplitude_v"] = amp
        out["n2_amplitude_uv"] = amp * 1e6
        return {"ok": True, "rows": out}
    except Exception as exc:
        return {
            "ok": False,
            "qc": {"subject": subject, "day": day, "reason": "extract_error", "detail": str(exc)},
        }


def plot_n2_boundary_slopes(slope_df, fig_path):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    summary = (
        slope_df.groupby("day", as_index=False)
        .agg(
            slope_mean=("slope", "mean"),
            slope_sem=("slope", _sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values("day")
    )
    ax.errorbar(
        summary["day"],
        summary["slope_mean"],
        yerr=summary["slope_sem"],
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="tab:blue",
    )
    ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
    ax.set_xlabel("Day")
    ax.set_ylabel("N2 slope on boundary distance (uV/unit)")
    ax.set_title("Frontal N2 Boundary-Distance Slope")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_n2_boundary_distance_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sessions, boundary = load_epoch_beh_sessions_with_boundary()
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    tasks = [
        {"subject": s["subject"], "day": s["day"], "epo_path": s["epo_path"], "beh": s["beh"]}
        for s in sessions
    ]
    results = parallel_collect(extract_n2_session, tasks, n_workers)
    rows = []
    qc = []
    for result in results:
        if result["ok"]:
            rows.append(result["rows"])
        else:
            qc.append(result["qc"])
    trial_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if trial_df.empty:
        raise RuntimeError("No N2 boundary-distance rows extracted.")
    trial_df = trial_df.dropna(subset=["n2_amplitude_uv", "boundary_distance"]).copy()
    model, status = _fit_regression(
        "n2_amplitude_uv ~ boundary_distance * day",
        trial_df,
        re_formula="~boundary_distance",
    )
    terms = []
    for term in model.model.exog_names:
        if term == "Intercept":
            continue
        item = model_term_summary(model, term)
        item["status"] = status
        terms.append(item)
    slope_rows = []
    for (subject, day), g in trial_df.groupby(["subject", "day"]):
        if len(g) < 5 or g["boundary_distance"].nunique() < 2:
            continue
        fit = smf.ols("n2_amplitude_uv ~ boundary_distance", data=g).fit()
        slope_rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "slope": float(fit.params["boundary_distance"]),
                "intercept": float(fit.params["Intercept"]),
                "p_value": float(fit.pvalues["boundary_distance"]),
                "n_trials": int(len(g)),
            }
        )
    slope_df = pd.DataFrame(slope_rows)
    trial_csv = output_dir / "n2_boundary_trial_level.csv"
    model_csv = output_dir / "n2_boundary_model_terms.csv"
    slope_csv = output_dir / "n2_boundary_subject_day_slopes.csv"
    qc_csv = output_dir / "n2_boundary_qc.csv"
    fig_path = figures_dir / "n2_boundary_slope_by_day.png"
    trial_df.to_csv(trial_csv, index=False)
    pd.DataFrame(terms).to_csv(model_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    pd.DataFrame(qc).to_csv(qc_csv, index=False)
    plot_n2_boundary_slopes(slope_df, fig_path)
    return {
        "trial_df": trial_df,
        "model_df": pd.DataFrame(terms),
        "slope_df": slope_df,
        "trial_csv": trial_csv,
        "model_csv": model_csv,
        "slope_csv": slope_csv,
        "qc_csv": qc_csv,
        "figure": fig_path,
    }


def plot_boundary_tg_individual_differences(corr_df, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, xcol, title in [
        (axes[0, 0], "slope_change", "N2 slope change vs late TG gain"),
        (axes[0, 1], "distance_slope_day1", "Day 1 N2 slope vs late TG gain"),
    ]:
        d = corr_df.dropna(subset=[xcol, "late_tg_gain"])
        ax.scatter(d[xcol], d["late_tg_gain"], color="tab:blue", alpha=0.8)
        if len(d) >= 3:
            fit = smf.ols(f"late_tg_gain ~ {xcol}", data=d).fit()
            x = np.linspace(d[xcol].min(), d[xcol].max(), 100)
            y = fit.predict(pd.DataFrame({xcol: x}))
            ax.plot(x, y, color="black", linestyle="--", linewidth=1.5)
        ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
        ax.axvline(0, color="0.35", linestyle=":", linewidth=1)
        ax.set_xlabel(xcol)
        ax.set_ylabel("Late TG gain")
        ax.set_title(title)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_boundary_tg_individual_difference_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    tg_matrix_dir: Path | str | None = None,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    slope_path = output_dir / "n2_boundary_subject_day_slopes.csv"
    if tg_matrix_dir is None:
        tg_matrix_dir = PROJECT_DIR / "output" / "mvpa_tg_cross_day" / "tg_cross_day_subject_matrices"
    day1_path = output_dir / "tg_day1_window_auc_subject_pairs.csv"
    if not slope_path.exists():
        run_n2_boundary_distance_analysis(output_dir=output_dir, figures_dir=figures_dir)
    if not day1_path.exists():
        raise FileNotFoundError(
            f"Day 1 TG window AUC file not found: {day1_path}. "
            "Run mvpa_tg_day1_distinctiveness first."
        )
    slopes = pd.read_csv(slope_path)
    tg = pd.read_csv(day1_path)
    n2_rows = []
    for subject, g in slopes.groupby("subject"):
        g = g.sort_values("day")
        if g.empty:
            continue
        day1 = g[g["day"] == 1]
        first = day1.iloc[0] if not day1.empty else g.iloc[0]
        last = g.iloc[-1]
        n2_rows.append(
            {
                "subject": int(subject),
                "distance_slope_day1": float(first["slope"]),
                "distance_slope_last": float(last["slope"]),
                "last_day": int(last["day"]),
                "slope_change": float(last["slope"] - first["slope"]),
            }
        )
    n2_df = pd.DataFrame(n2_rows)
    late = tg[(tg["window"] == "late") & (tg["train_day"] != tg["test_day"])].copy()
    late["pair_group"] = np.where(
        (late["train_day"] == 1) | (late["test_day"] == 1), "day1_pair", "later_only"
    )
    tg_subject = late.groupby(["subject", "pair_group"], as_index=False)["mean_auc"].mean()
    tg_wide = tg_subject.pivot(index="subject", columns="pair_group", values="mean_auc").reset_index()
    tg_wide["late_tg_gain"] = tg_wide["later_only"] - tg_wide["day1_pair"]
    corr_df = n2_df.merge(tg_wide, on="subject", how="inner")
    corr_rows = []
    for xcol in ["slope_change", "distance_slope_day1"]:
        d = corr_df.dropna(subset=[xcol, "late_tg_gain"])
        if len(d) >= 3:
            r, pval = stats.pearsonr(d[xcol], d["late_tg_gain"])
        else:
            r, pval = np.nan, np.nan
        corr_rows.append({"predictor": xcol, "r": r, "p_value": pval, "n_subjects": int(len(d))})
    corr_stats = pd.DataFrame(corr_rows)
    corr_csv = output_dir / "n2_boundary_late_tg_individual_differences.csv"
    stats_csv = output_dir / "n2_boundary_late_tg_correlations.csv"
    fig_path = figures_dir / "n2_boundary_late_tg_individual_differences.png"
    corr_df.to_csv(corr_csv, index=False)
    corr_stats.to_csv(stats_csv, index=False)
    plot_boundary_tg_individual_differences(corr_df, fig_path)
    return {
        "corr_df": corr_df,
        "corr_stats": corr_stats,
        "corr_csv": corr_csv,
        "stats_csv": stats_csv,
        "figure": fig_path,
    }


if __name__ == "__main__":
    run_n2_boundary_distance_analysis()
