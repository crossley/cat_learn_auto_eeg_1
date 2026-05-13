#!/usr/bin/env python3
"""Day 1 FRN × Rescorla-Wagner RPE analysis."""

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
from scipy.optimize import minimize_scalar

from analysis_utils import parallel_collect
from boundary_distance import load_behaviour_with_boundary

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "erp_frn_rpe"
FIGURES_DIR = PROJECT_DIR / "figures" / "erp_frn_rpe"
N_JOBS = 8


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def load_day1_sessions(project_dir: Path | str = PROJECT_DIR):
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
        if day != 1:
            continue
        beh_df = beh[(beh["subject"] == subject) & (beh["day"] == day)].copy()
        if beh_df.empty:
            continue
        sessions.append({"subject": subject, "day": day, "epo_path": epo_path, "beh": beh_df})
    return sessions


def fit_rw_for_subject(outcomes):
    outcomes = np.asarray(outcomes, dtype=float)

    def nll(alpha):
        v = 0.5
        total = 0.0
        eps = 1e-6
        for outcome in outcomes:
            p = np.clip(v, eps, 1 - eps)
            total -= outcome * np.log(p) + (1 - outcome) * np.log(1 - p)
            v = v + alpha * (outcome - v)
        return total

    res = minimize_scalar(nll, bounds=(0.01, 0.99), method="bounded")
    alpha = float(res.x)
    v = 0.5
    preds = []
    rpes = []
    for outcome in outcomes:
        preds.append(v)
        rpes.append(outcome - v)
        v = v + alpha * (outcome - v)
    return alpha, float(res.fun), np.asarray(preds), np.asarray(rpes)


def align_feedback_to_beh(beh_df, epochs, event_names=("FB/Cor", "FB/Inc")):
    beh_sorted = beh_df.sort_values("trial").reset_index(drop=True)
    event_names = [name for name in event_names if name in epochs.event_id]
    if not event_names:
        return epochs[:0], beh_sorted.iloc[:0].copy()
    epochs_fb = epochs[event_names]
    if len(epochs_fb) == 0:
        return epochs_fb, beh_sorted.iloc[:0].copy()
    if epochs_fb.metadata is not None and "beh_trial_index" in epochs_fb.metadata:
        trial_idx = epochs_fb.metadata["beh_trial_index"].to_numpy(dtype=int)
    else:
        import math

        sel = np.asarray(epochs_fb.selection, dtype=int)
        if len(sel) > 1:
            diffs = np.diff(np.sort(sel))
            step = int(diffs[0])
            for diff in diffs[1:]:
                step = math.gcd(step, int(diff))
            offset = int(np.min(sel % step)) if step > 1 else 0
            trial_idx = (sel - offset) // step if step > 1 else sel.copy()
        else:
            trial_idx = sel.copy()
    valid = (trial_idx >= 0) & (trial_idx < len(beh_sorted))
    if not valid.all():
        epochs_fb = epochs_fb[np.where(valid)[0]]
        trial_idx = trial_idx[valid]
    return epochs_fb, beh_sorted.iloc[trial_idx].reset_index(drop=True)


def extract_frn_day1_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    if day != 1:
        return {"ok": False, "qc": {"subject": subject, "day": day, "reason": "not_day1", "detail": ""}}
    try:
        beh_df = task["beh"].sort_values("trial").reset_index(drop=True).copy()
        outcomes = (beh_df["fb"].astype(str).str.lower() == "correct").astype(float).to_numpy()
        alpha, nll, pred, rpe = fit_rw_for_subject(outcomes)
        beh_df["rw_alpha"] = alpha
        beh_df["rw_nll"] = nll
        beh_df["rw_pred"] = pred
        beh_df["rpe"] = rpe
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        epochs_fb, beh_aligned = align_feedback_to_beh(beh_df, epochs)
        if len(epochs_fb) == 0:
            return {
                "ok": False,
                "qc": {"subject": subject, "day": day, "reason": "no_feedback_epochs", "detail": ""},
            }
        epochs_fb = epochs_fb.copy().load_data()
        channels = [ch for ch in ["Fz", "FCz"] if ch in epochs_fb.ch_names]
        if len(channels) == 0:
            return {
                "ok": False,
                "qc": {"subject": subject, "day": day, "reason": "missing_frn_channels", "detail": ""},
            }
        data = epochs_fb.copy().pick(channels).get_data()
        times = epochs_fb.times
        tmask = (times >= 0.200) & (times <= 0.300)
        frn = data[:, :, tmask].mean(axis=(1, 2))
        out = beh_aligned.reset_index(drop=True).copy()
        out["subject"] = subject
        out["day"] = day
        out["frn_amplitude_v"] = frn
        out["frn_amplitude_uv"] = frn * 1e6
        waves = epochs_fb.copy().pick(channels).get_data().mean(axis=1) * 1e6
        wave_df = pd.DataFrame(waves, columns=[float(t) for t in times])
        wave_df.insert(0, "rpe", out["rpe"].to_numpy())
        wave_df.insert(0, "subject", subject)
        return {
            "ok": True,
            "rows": out,
            "waves": wave_df,
            "fit": {"subject": subject, "alpha": alpha, "nll": nll},
        }
    except Exception as exc:
        return {
            "ok": False,
            "qc": {"subject": subject, "day": day, "reason": "extract_error", "detail": str(exc)},
        }


def plot_frn_rpe_tertiles(wave_df, fig_path):
    if wave_df.empty:
        return
    wave_df = wave_df.copy()
    wave_df["rpe_tertile"] = pd.qcut(wave_df["rpe"].rank(method="first"), 3, labels=["low", "medium", "high"])
    id_cols = ["subject", "rpe"]
    value_cols = [c for c in wave_df.columns if c not in id_cols + ["rpe_tertile"]]
    long = wave_df.melt(
        id_vars=id_cols + ["rpe_tertile"],
        value_vars=value_cols,
        var_name="time_sec",
        value_name="amp_uv",
    )
    long["time_sec"] = long["time_sec"].astype(float)
    summary = (
        long.groupby(["rpe_tertile", "time_sec"], observed=True, as_index=False)
        .agg(amp_mean=("amp_uv", "mean"), amp_sem=("amp_uv", _sem))
        .sort_values(["rpe_tertile", "time_sec"])
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {"low": "tab:red", "medium": "tab:orange", "high": "tab:green"}
    for tertile, g in summary.groupby("rpe_tertile", observed=True):
        ax.plot(g["time_sec"], g["amp_mean"], color=colors[str(tertile)], linewidth=1.8, label=str(tertile))
        ax.fill_between(
            g["time_sec"],
            g["amp_mean"] - g["amp_sem"],
            g["amp_mean"] + g["amp_sem"],
            color=colors[str(tertile)],
            alpha=0.15,
            linewidth=0,
        )
    ax.axvspan(0.200, 0.300, color="0.5", alpha=0.12, linewidth=0)
    ax.axvline(0, color="0.35", linestyle=":", linewidth=1)
    ax.set_xlabel("Time from feedback (s)")
    ax.set_ylabel("Fz/FCz amplitude (uV)")
    ax.set_title("Day 1 FRN by RPE Tertile")
    ax.legend(title="RPE")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_day1_rw_frn_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    sessions = load_day1_sessions()
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    tasks = [
        {"subject": s["subject"], "day": s["day"], "epo_path": s["epo_path"], "beh": s["beh"]}
        for s in sessions
    ]
    results = parallel_collect(extract_frn_day1_session, tasks, n_workers)
    rows = []
    waves = []
    fits = []
    qc = []
    for result in results:
        if result["ok"]:
            rows.append(result["rows"])
            waves.append(result["waves"])
            fits.append(result["fit"])
        else:
            qc.append(result["qc"])
    trial_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    wave_df = pd.concat(waves, ignore_index=True) if waves else pd.DataFrame()
    fit_df = pd.DataFrame(fits)
    coef_rows = []
    for subject, g in trial_df.groupby("subject"):
        if len(g) < 5 or g["rpe"].nunique() < 2:
            continue
        fit = smf.ols("frn_amplitude_uv ~ rpe", data=g).fit()
        coef_rows.append(
            {
                "subject": int(subject),
                "coef_rpe": float(fit.params["rpe"]),
                "p_value": float(fit.pvalues["rpe"]),
                "n_trials": int(len(g)),
            }
        )
    coef_df = pd.DataFrame(coef_rows)
    if coef_df.empty:
        group_stats = {"n_subjects": 0, "mean_coef": np.nan, "t": np.nan, "p_value": np.nan}
    else:
        t_res = stats.ttest_1samp(coef_df["coef_rpe"].to_numpy(dtype=float), 0.0, nan_policy="omit")
        group_stats = {
            "n_subjects": int(coef_df["subject"].nunique()),
            "mean_coef": float(coef_df["coef_rpe"].mean()),
            "t": float(t_res.statistic),
            "p_value": float(t_res.pvalue),
        }
    trial_csv = output_dir / "day1_rw_frn_trial_level.csv"
    coef_csv = output_dir / "day1_rw_frn_subject_coefficients.csv"
    fit_csv = output_dir / "day1_rw_fitted_alphas.csv"
    group_csv = output_dir / "day1_rw_frn_group_ttest.csv"
    qc_csv = output_dir / "day1_rw_frn_qc.csv"
    fig_path = figures_dir / "day1_frn_by_rpe_tertile.png"
    trial_df.to_csv(trial_csv, index=False)
    coef_df.to_csv(coef_csv, index=False)
    fit_df.to_csv(fit_csv, index=False)
    pd.DataFrame([group_stats]).to_csv(group_csv, index=False)
    pd.DataFrame(qc).to_csv(qc_csv, index=False)
    plot_frn_rpe_tertiles(wave_df, fig_path)
    return {
        "trial_df": trial_df,
        "coef_df": coef_df,
        "fit_df": fit_df,
        "group_stats": group_stats,
        "trial_csv": trial_csv,
        "coef_csv": coef_csv,
        "fit_csv": fit_csv,
        "group_csv": group_csv,
        "qc_csv": qc_csv,
        "figure": fig_path,
    }


if __name__ == "__main__":
    run_day1_rw_frn_analysis()
