#!/usr/bin/env python3
"""Plot first-pass links between behaviour and neural features."""

from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connect_sensorwide_figure import (
    ACTIVE_PAIR_PEAK_WINDOWS,
    compute_peak_edge_rows,
    subject_peak_edge_vectors,
)
from connect_sensorwide_analysis import BANDS, CHANNEL_SUBSET, OUTPUT_DIR

PROJECT_DIR = Path(__file__).resolve().parent.parent
BEHAVIOURAL_DIR = PROJECT_DIR / "Behavioural"
FIGURES_DIR = PROJECT_DIR / "figures"

BEHAV_RE = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")
ERP_PEAK_WINDOWS = ACTIVE_PAIR_PEAK_WINDOWS
MVPA_WINDOWS = {
    "early": (0.060, 0.180),
    "late": (0.400, 0.600),
}


def sem(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def day_colors(days):
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(days)))
    out = {}
    for idx, day in enumerate(days):
        out[int(day)] = colors[idx]
    return out


def load_behaviour_summary():
    if not BEHAVIOURAL_DIR.exists():
        raise FileNotFoundError(f"Missing behavioural directory: {BEHAVIOURAL_DIR}")
    rows = []
    for path in sorted(BEHAVIOURAL_DIR.glob("*.csv")):
        match = BEHAV_RE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected behavioural filename: {path.name}")
        subject = int(match.group(1))
        day_code = int(match.group(2))
        day = day_code // 100 if day_code >= 100 else day_code
        d = pd.read_csv(path)
        for col in ["fb", "rt"]:
            if col not in d.columns:
                raise ValueError(f"{path.name} missing behavioural column: {col}")
        fb = d["fb"].astype(str).str.lower()
        correct = fb == "correct"
        rt = pd.to_numeric(d["rt"], errors="coerce")
        rt_correct = rt[correct & np.isfinite(rt)]
        rows.append(
            {
                "subject": subject,
                "day": day,
                "accuracy": float(np.mean(correct)),
                "rt_correct": float(np.mean(rt_correct)),
                "n_trials": int(len(d)),
                "n_correct_rt": int(len(rt_correct)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No behavioural rows loaded")
    return out


def subject_line_mean_sem(ax, df, value_col, ylabel, title):
    days = sorted(df["day"].dropna().unique().astype(int).tolist())
    for subject in sorted(df["subject"].dropna().unique().astype(int).tolist()):
        d_sub = df[df["subject"] == subject].sort_values("day")
        ax.plot(
            d_sub["day"],
            d_sub[value_col],
            color="0.65",
            linewidth=0.8,
            alpha=0.45,
        )
    means = []
    errors = []
    for day in days:
        vals = df[df["day"] == day][value_col].to_numpy(dtype=float)
        means.append(float(np.nanmean(vals)))
        errors.append(sem(vals))
    ax.errorbar(days, means, yerr=errors, color="black", linewidth=2.0, marker="o")
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(days)
    ax.grid(alpha=0.25)


def load_erp_features(output_dir):
    path = output_dir / "erp_grand_average_subject_day_all.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ERP subject output: {path}. "
            "Run erp_grand_average_analysis.py first."
        )
    usecols = ["subject", "day", "lock_type", "condition", "channel", "time_s"]
    usecols.append("amplitude_v")
    d = pd.read_csv(path, usecols=usecols)
    d = d[
        (d["lock_type"] == "stim")
        & (d["condition"] == "all")
        & (d["time_s"] >= 0.0)
        & (d["time_s"] <= 0.4)
    ].copy()
    if d.empty:
        raise ValueError(f"No stim/all ERP rows in {path}")
    rows = []
    group_cols = ["subject", "day"]
    for key, g in d.groupby(group_cols):
        subject, day = key
        times = np.sort(g["time_s"].unique().astype(float))
        channels = sorted(g["channel"].unique().tolist())
        mat = np.full((len(channels), len(times)), np.nan, dtype=float)
        for ch_i, ch in enumerate(channels):
            d_ch = g[g["channel"] == ch].sort_values("time_s")
            mat[ch_i, :] = d_ch["amplitude_v"].to_numpy(dtype=float)
        gfp = np.sqrt(np.nanmean(mat ** 2, axis=0)) * 1e6
        for peak_i, window in enumerate(ERP_PEAK_WINDOWS, start=1):
            lo, hi = window
            idx_vals = []
            for idx, t_val in enumerate(times):
                if t_val >= lo and t_val <= hi and np.isfinite(gfp[idx]):
                    idx_vals.append(idx)
            if len(idx_vals) == 0:
                raise ValueError(
                    f"No finite ERP values for subject={subject}, day={day}, "
                    f"window={window}"
                )
            best_idx = idx_vals[0]
            best_val = float(gfp[best_idx])
            for idx in idx_vals:
                val = float(gfp[idx])
                if val > best_val:
                    best_idx = idx
                    best_val = val
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "peak": int(peak_i),
                    "latency_sec": float(times[best_idx]),
                    "amplitude_uv": best_val,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No ERP feature rows computed")
    return out


def load_connectivity_features(output_dir):
    carpet_path = output_dir / "sensorwide_carpet_timeseries.csv"
    subject_path = output_dir / "sensorwide_carpet_subject_timeseries.csv"
    channels_path = output_dir / "sensorwide_channel_layout.csv"
    for path in [carpet_path, subject_path, channels_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Missing connectivity output: {path}. "
                "Run connect_sensorwide_analysis.py first."
            )
    d_carpet = pd.read_csv(carpet_path)
    d_subject = pd.read_csv(subject_path)
    d_channels = pd.read_csv(channels_path)
    ch_names = []
    for ch in d_channels["channel"]:
        ch_names.append(ch)
    n_channels = len(ch_names)
    pair_idx = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pair_idx.append((i, j))
    ch_to_idx = {}
    for i, ch in enumerate(ch_names):
        ch_to_idx[ch] = i

    d_lb = d_carpet[
        (d_carpet["lock_type"] == "stim") & (d_carpet["band"] == "broadband")
    ]
    if d_lb.empty:
        raise ValueError("Missing stim/broadband rows in sensorwide carpet output")
    day_data = {}
    all_days = sorted(d_lb["day"].dropna().unique().astype(int).tolist())
    for day in all_days:
        d_day = d_lb[d_lb["day"] == day]
        times_this = sorted(d_day["lock_time"].dropna().unique().tolist())
        mats = []
        for t in times_this:
            mat = np.full((n_channels, n_channels), np.nan, dtype=float)
            d_t = d_day[d_day["lock_time"] == t]
            for _, row in d_t.iterrows():
                i = ch_to_idx.get(row["ch_i"])
                j = ch_to_idx.get(row["ch_j"])
                if i is None or j is None:
                    continue
                mat[i, j] = float(row["conn_val"])
                mat[j, i] = float(row["conn_val"])
            np.fill_diagonal(mat, 0.0)
            mats.append(mat)
        day_data[day] = {"times": np.asarray(times_this, dtype=float), "mats": mats}

    peak_rows, active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    subjects, vector_map = subject_peak_edge_vectors(
        d_subject, peak_rows, active_pair_idx, pair_idx, ch_names
    )
    rows = []
    for subject in subjects:
        for row in peak_rows:
            day = int(row["day"])
            peak = int(row["peak"])
            key = (int(subject), day, peak)
            if key not in vector_map:
                continue
            vec = vector_map[key]
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "peak": int(peak),
                    "connectivity": float(np.nanmean(vec)),
                    "peak_time_sec": float(row["peak_time"]),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No connectivity feature rows computed")
    return out


def load_mvpa_features(output_dir):
    path = output_dir / "mvpa_stim_locked_cat_subject_day_timecourse.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MVPA subject-day timecourse: {path}. "
            "Run mvpa_stim_locked_cat_time_resolved_analysis.py first."
        )
    d = pd.read_csv(path)
    rows = []
    for key, g in d.groupby(["subject", "day"]):
        subject, day = key
        for label, window in MVPA_WINDOWS.items():
            lo, hi = window
            d_win = g[(g["time_sec"] >= lo) & (g["time_sec"] <= hi)]
            if d_win.empty:
                raise ValueError(
                    f"No MVPA rows for subject={subject}, day={day}, "
                    f"window={label}"
                )
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "window": label,
                    "auc": float(np.nanmean(d_win["auc"])),
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No MVPA feature rows computed")
    return out


def plot_behavior_trajectories(beh, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), squeeze=False)
    subject_line_mean_sem(
        axes[0, 0],
        beh,
        "accuracy",
        "Accuracy",
        "Accuracy",
    )
    subject_line_mean_sem(
        axes[0, 1],
        beh,
        "rt_correct",
        "Correct RT",
        "Response time",
    )
    fig.suptitle("Behaviour")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig_path = figures_dir / "link_behave_neuro_behavior_trajectories.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_erp_trajectories(erp, figures_dir):
    fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.0), squeeze=False)
    for peak in [1, 2, 3]:
        d_peak = erp[erp["peak"] == peak]
        subject_line_mean_sem(
            axes[0, peak - 1],
            d_peak,
            "amplitude_uv",
            "GFP amplitude (uV)",
            f"Peak {peak}",
        )
        subject_line_mean_sem(
            axes[1, peak - 1],
            d_peak,
            "latency_sec",
            "Latency (s)",
            f"Peak {peak}",
        )
    fig.suptitle("ERP Peak Features")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = figures_dir / "link_behave_neuro_erp_peak_trajectories.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_connectivity_trajectories(conn, figures_dir):
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), squeeze=False)
    for peak in [1, 2, 3]:
        d_peak = conn[conn["peak"] == peak]
        subject_line_mean_sem(
            axes[0, peak - 1],
            d_peak,
            "connectivity",
            "Connectivity",
            f"Peak {peak}",
        )
    fig.suptitle("Functional Connectivity Strength")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig_path = figures_dir / "link_behave_neuro_connect_peak_trajectories.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_mvpa_trajectories(mvpa, figures_dir):
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4), squeeze=False)
    windows = ["early", "late"]
    for idx, window in enumerate(windows):
        d_win = mvpa[mvpa["window"] == window]
        subject_line_mean_sem(
            axes[0, idx],
            d_win,
            "auc",
            "AUC",
            f"{window.capitalize()} AUC",
        )
        axes[0, idx].axhline(0.5, color="0.35", linestyle=":", linewidth=0.9)
    fig.suptitle("Stim-Locked Category Decoding")
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig_path = figures_dir / "link_behave_neuro_mvpa_auc_trajectories.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def feature_change(df, value_col, label):
    rows = []
    for subject in sorted(df["subject"].dropna().unique().astype(int).tolist()):
        d_sub = df[df["subject"] == subject]
        d1 = d_sub[d_sub["day"] == 1]
        d5 = d_sub[d_sub["day"] == 5]
        if d1.empty or d5.empty:
            continue
        rows.append(
            {
                "subject": int(subject),
                "feature": label,
                "change": float(d5[value_col].mean() - d1[value_col].mean()),
            }
        )
    return pd.DataFrame(rows)


def plot_change_score_scatter(beh, erp, conn, mvpa, figures_dir):
    beh_acc = feature_change(beh, "accuracy", "accuracy")
    beh_rt = feature_change(beh, "rt_correct", "rt")

    erp_peak3 = erp[erp["peak"] == 3]
    conn_peak3 = conn[conn["peak"] == 3]
    mvpa_late = mvpa[mvpa["window"] == "late"]
    neural_parts = [
        feature_change(erp_peak3, "amplitude_uv", "ERP P3 amp"),
        feature_change(conn_peak3, "connectivity", "Conn P3"),
        feature_change(mvpa_late, "auc", "MVPA late AUC"),
    ]
    neural = pd.concat(neural_parts, ignore_index=True)
    outcomes = [("accuracy", beh_acc, "Accuracy change"), ("rt", beh_rt, "RT change")]
    features = ["ERP P3 amp", "Conn P3", "MVPA late AUC"]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 6.3), squeeze=False)
    for row_i, outcome in enumerate(outcomes):
        outcome_name, beh_change, y_label = outcome
        for col_i, feature in enumerate(features):
            ax = axes[row_i, col_i]
            d_feat = neural[neural["feature"] == feature]
            d_plot = d_feat.merge(
                beh_change[["subject", "change"]],
                on="subject",
                how="inner",
                suffixes=("_neural", "_behaviour"),
            )
            if d_plot.empty:
                raise ValueError(
                    f"No change-score rows for feature={feature}, "
                    f"outcome={outcome_name}"
                )
            ax.scatter(
                d_plot["change_neural"],
                d_plot["change_behaviour"],
                color="black",
                s=24,
                alpha=0.78,
            )
            if len(d_plot) >= 2:
                x = d_plot["change_neural"].to_numpy(dtype=float)
                y = d_plot["change_behaviour"].to_numpy(dtype=float)
                ok = np.isfinite(x) & np.isfinite(y)
                if int(np.sum(ok)) >= 2:
                    coef = np.polyfit(x[ok], y[ok], deg=1)
                    x_min = float(np.min(x[ok]))
                    x_max = float(np.max(x[ok]))
                    x_line = np.linspace(x_min, x_max, 100)
                    y_line = coef[0] * x_line + coef[1]
                    ax.plot(x_line, y_line, color="tab:red", linewidth=1.3)
            ax.axhline(0.0, color="0.55", linestyle=":", linewidth=0.8)
            ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
            ax.set_title(feature)
            ax.set_xlabel("Neural change D5 - D1")
            ax.set_ylabel(y_label)
            ax.grid(alpha=0.25)
    fig.suptitle("Behaviour-Neural Change Scores")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig_path = figures_dir / "link_behave_neuro_change_score_scatter.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_link_behave_neuro(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    beh = load_behaviour_summary()
    erp = load_erp_features(output_dir)
    conn = load_connectivity_features(output_dir)
    mvpa = load_mvpa_features(output_dir)

    paths = {}
    paths["behavior"] = plot_behavior_trajectories(beh, figures_dir)
    paths["erp"] = plot_erp_trajectories(erp, figures_dir)
    paths["connectivity"] = plot_connectivity_trajectories(conn, figures_dir)
    paths["mvpa"] = plot_mvpa_trajectories(mvpa, figures_dir)
    paths["change_score"] = plot_change_score_scatter(
        beh, erp, conn, mvpa, figures_dir
    )
    return paths


if __name__ == "__main__":
    save_fig_link_behave_neuro()
