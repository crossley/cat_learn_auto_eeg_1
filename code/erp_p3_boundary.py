#!/usr/bin/env python3
"""Parietal P3 x boundary distance analysis."""

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

from analysis_utils import model_term_summary, parallel_collect
from erp_n2_boundary import load_epoch_beh_sessions_with_boundary
from load_project_data import align_behaviour_to_epochs

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8

P3_CHANNELS = ["Pz", "P3", "P4"]
P3_TMIN = 0.300
P3_TMAX = 0.600


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_regression_with_fallback(formula, df, group_col="subject", re_formula=None):
    try:
        if re_formula is None:
            model = smf.mixedlm(formula, data=df, groups=df[group_col]).fit(
                reml=False, method="lbfgs", disp=False
            )
        else:
            model = smf.mixedlm(
                formula, data=df, groups=df[group_col], re_formula=re_formula
            ).fit(reml=False, method="lbfgs", disp=False)
        return model, "mixedlm"
    except Exception as exc:
        model = smf.ols(formula, data=df).fit(
            cov_type="cluster", cov_kwds={"groups": df[group_col]}
        )
        model._fallback_detail = str(exc)
        return model, "ols_cluster_fallback"


def extract_p3_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        epochs_stim, beh_aligned = align_behaviour_to_epochs(
            task["beh"], epochs, event_names=("Stim/A", "Stim/B")
        )
        epochs_stim = epochs_stim.copy().load_data()
        channels = [ch for ch in task["p3_channels"] if ch in epochs_stim.ch_names]
        if len(channels) == 0:
            return {
                "ok": False,
                "qc": {"subject": subject, "day": day, "reason": "missing_channels", "detail": ""},
            }
        times = epochs_stim.times
        tmask = (times >= float(task["p3_tmin"])) & (times <= float(task["p3_tmax"]))
        if not np.any(tmask):
            raise ValueError("empty_p3_time_window")
        data = epochs_stim.copy().pick(channels).get_data()
        amp = data[:, :, tmask].mean(axis=(1, 2))
        out = beh_aligned.reset_index(drop=True).copy()
        out["subject"] = subject
        out["day"] = day
        out["p3_amplitude_v"] = amp
        out["p3_amplitude_uv"] = amp * 1e6
        out["p3_channels"] = ",".join(channels)
        return {"ok": True, "rows": out}
    except Exception as exc:
        return {
            "ok": False,
            "qc": {"subject": subject, "day": day, "reason": "extract_error", "detail": str(exc)},
        }


def plot_p3_boundary_slopes(slope_df, fig_path):
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
        color="tab:purple",
    )
    ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
    ax.set_xlabel("Day")
    ax.set_ylabel("P3 slope on boundary distance (uV/unit)")
    ax.set_title("Parietal P3 Boundary-Distance Slope")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_p3_boundary_analysis(
    p3_channels: list[str] = P3_CHANNELS,
    p3_tmin: float = P3_TMIN,
    p3_tmax: float = P3_TMAX,
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
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
            "epo_path": s["epo_path"],
            "beh": s["beh"],
            "p3_channels": list(p3_channels),
            "p3_tmin": float(p3_tmin),
            "p3_tmax": float(p3_tmax),
        }
        for s in sessions
    ]
    print(f"[ERP P3 boundary] Extracting {len(tasks)} sessions (n_workers={n_workers})...", flush=True)
    results = parallel_collect(extract_p3_session, tasks, n_workers)
    rows = []
    qc = []
    for result in results:
        if result["ok"]:
            rows.append(result["rows"])
        else:
            qc.append(result["qc"])
    trial_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if trial_df.empty:
        raise RuntimeError("No P3 boundary-distance rows extracted.")
    trial_df = trial_df.dropna(subset=["p3_amplitude_uv", "boundary_distance"]).copy()
    model, status = _fit_regression_with_fallback(
        "p3_amplitude_uv ~ boundary_distance * day",
        trial_df,
        re_formula="~boundary_distance",
    )
    terms = []
    for term in model.model.exog_names:
        if term == "Intercept":
            continue
        item = model_term_summary(model, term)
        item["status"] = status
        if hasattr(model, "_fallback_detail"):
            item["fallback_detail"] = model._fallback_detail
        terms.append(item)
    slope_rows = []
    for (subject, day), g in trial_df.groupby(["subject", "day"]):
        if len(g) < 5 or g["boundary_distance"].nunique() < 2:
            continue
        fit = smf.ols("p3_amplitude_uv ~ boundary_distance", data=g).fit()
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
    trial_csv = output_dir / "erp_p3_boundary_trial_level.csv"
    model_csv = output_dir / "erp_p3_boundary_model_terms.csv"
    slope_csv = output_dir / "erp_p3_boundary_subject_day_slopes.csv"
    qc_csv = output_dir / "erp_p3_boundary_qc.csv"
    fig_path = figures_dir / "erp_p3_boundary_slope_by_day.png"
    trial_df.to_csv(trial_csv, index=False)
    pd.DataFrame(terms).to_csv(model_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    pd.DataFrame(qc).to_csv(qc_csv, index=False)
    plot_p3_boundary_slopes(slope_df, fig_path)
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


if __name__ == "__main__":
    run_p3_boundary_analysis()
