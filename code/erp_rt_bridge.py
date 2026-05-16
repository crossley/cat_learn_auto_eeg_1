#!/usr/bin/env python3
"""ERP-RT bridge: link stimulus-locked ERP components to reaction time."""

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

N2_CHANNELS = ["Fz", "FCz", "FC1", "FC2"]
P3_CHANNELS = ["Pz", "P3", "P4"]
N2_TMIN = 0.200
N2_TMAX = 0.300
P3_TMIN = 0.300
P3_TMAX = 0.600


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


def _window_mean(epochs, channels, tmin, tmax):
    use_channels = [ch for ch in channels if ch in epochs.ch_names]
    if len(use_channels) == 0:
        raise ValueError(f"missing_channels:{','.join(channels)}")
    times = epochs.times
    tmask = (times >= float(tmin)) & (times <= float(tmax))
    if not np.any(tmask):
        raise ValueError(f"empty_time_window:{tmin}-{tmax}")
    data = epochs.copy().pick(use_channels).get_data()
    return data[:, :, tmask].mean(axis=(1, 2)), use_channels


def extract_erp_rt_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        epochs_stim, beh_aligned = align_behaviour_to_epochs(
            task["beh"], epochs, event_names=("Stim/A", "Stim/B")
        )
        epochs_stim = epochs_stim.copy().load_data()
        n2_amp, n2_channels = _window_mean(
            epochs_stim, task["n2_channels"], task["n2_tmin"], task["n2_tmax"]
        )
        p3_amp, p3_channels = _window_mean(
            epochs_stim, task["p3_channels"], task["p3_tmin"], task["p3_tmax"]
        )
        out = beh_aligned.reset_index(drop=True).copy()
        out["subject"] = subject
        out["day"] = day
        out["n2_amplitude_v"] = n2_amp
        out["n2_amplitude_uv"] = n2_amp * 1e6
        out["p3_amplitude_v"] = p3_amp
        out["p3_amplitude_uv"] = p3_amp * 1e6
        out["log_rt_sec"] = np.log(pd.to_numeric(out["rt_sec"], errors="coerce"))
        out["n2_channels"] = ",".join(n2_channels)
        out["p3_channels"] = ",".join(p3_channels)
        return {"ok": True, "rows": out}
    except Exception as exc:
        return {
            "ok": False,
            "qc": {"subject": subject, "day": day, "reason": "extract_error", "detail": str(exc)},
        }


def plot_erp_rt_slopes(slope_df, fig_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, metric, ylabel, color in [
        (axes[0, 0], "n2_slope", "RT slope on N2 (log s/uV)", "tab:blue"),
        (axes[0, 1], "p3_slope", "RT slope on P3 (log s/uV)", "tab:purple"),
    ]:
        summary = (
            slope_df.groupby("day", as_index=False)
            .agg(mean=(metric, "mean"), sem=(metric, _sem), n_subjects=("subject", "nunique"))
            .sort_values("day")
        )
        ax.errorbar(
            summary["day"],
            summary["mean"],
            yerr=summary["sem"],
            marker="o",
            linewidth=1.8,
            capsize=3,
            color=color,
        )
        ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
        ax.set_xlabel("Day")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    fig.suptitle("ERP Component Coupling to Reaction Time")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_erp_rt_bridge_analysis(
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
            "n2_channels": N2_CHANNELS,
            "p3_channels": P3_CHANNELS,
            "n2_tmin": N2_TMIN,
            "n2_tmax": N2_TMAX,
            "p3_tmin": P3_TMIN,
            "p3_tmax": P3_TMAX,
        }
        for s in sessions
    ]
    print(f"[ERP-RT bridge] Extracting {len(tasks)} sessions (n_workers={n_workers})...", flush=True)
    results = parallel_collect(extract_erp_rt_session, tasks, n_workers)
    rows = []
    qc = []
    for result in results:
        if result["ok"]:
            rows.append(result["rows"])
        else:
            qc.append(result["qc"])
    trial_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if trial_df.empty:
        raise RuntimeError("No ERP-RT rows extracted.")
    trial_df = trial_df.dropna(
        subset=["log_rt_sec", "n2_amplitude_uv", "p3_amplitude_uv", "boundary_distance_abs"]
    ).copy()
    trial_df = trial_df[np.isfinite(trial_df["log_rt_sec"])].copy()
    model, status = _fit_regression_with_fallback(
        "log_rt_sec ~ n2_amplitude_uv * day + p3_amplitude_uv * day + boundary_distance_abs",
        trial_df,
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
        if len(g) < 10:
            continue
        fit = smf.ols(
            "log_rt_sec ~ n2_amplitude_uv + p3_amplitude_uv + boundary_distance_abs",
            data=g,
        ).fit()
        slope_rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "n2_slope": float(fit.params.get("n2_amplitude_uv", np.nan)),
                "p3_slope": float(fit.params.get("p3_amplitude_uv", np.nan)),
                "boundary_distance_abs_slope": float(
                    fit.params.get("boundary_distance_abs", np.nan)
                ),
                "r_squared": float(fit.rsquared),
                "n_trials": int(len(g)),
            }
        )
    slope_df = pd.DataFrame(slope_rows)
    trial_csv = output_dir / "erp_rt_bridge_trial_level.csv"
    model_csv = output_dir / "erp_rt_bridge_model_terms.csv"
    slope_csv = output_dir / "erp_rt_bridge_subject_day_slopes.csv"
    qc_csv = output_dir / "erp_rt_bridge_qc.csv"
    fig_path = figures_dir / "erp_rt_bridge_component_slopes.png"
    trial_df.to_csv(trial_csv, index=False)
    pd.DataFrame(terms).to_csv(model_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    pd.DataFrame(qc).to_csv(qc_csv, index=False)
    plot_erp_rt_slopes(slope_df, fig_path)
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
    run_erp_rt_bridge_analysis()
