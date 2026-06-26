#!/usr/bin/env python3
"""Plot stimulus-locked category MVPA summaries."""

from pathlib import Path
import os
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from figure_style import DAYS, DAY_COLORS, FIGURES_DIR, setup_axis
from mvpa_stim_locked_cat_time_resolved_analysis import OUTPUT_DIR

PROJECT_DIR = Path(__file__).resolve().parent.parent
BEHAVIOURAL_DIR = PROJECT_DIR / "Behavioural"
BEHAV_RE = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing MVPA output: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty MVPA output: {path}")
    return d


def load_behavior():
    if not BEHAVIOURAL_DIR.exists():
        raise FileNotFoundError(f"Missing behavioural directory: {BEHAVIOURAL_DIR}")
    rows = []
    for path in sorted(BEHAVIOURAL_DIR.glob("*.csv")):
        match = BEHAV_RE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected behavioural file: {path.name}")
        subject = int(match.group(1))
        day_code = int(match.group(2))
        day = day_code
        if day_code >= 100:
            day = day_code // 100
        d = pd.read_csv(path)
        if "fb" not in d.columns or "rt" not in d.columns:
            raise ValueError(f"Missing fb/rt columns in {path}")
        correct = d["fb"].astype(str).str.lower() == "correct"
        rows.append(
            {
                "subject": subject,
                "day": day,
                "accuracy": float(np.mean(correct)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No behavioural files loaded")
    return out


def correlation_label(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 3:
        return "r = n/a"
    x = x[good] - float(np.mean(x[good]))
    y = y[good] - float(np.mean(y[good]))
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return "r = n/a"
    return f"r = {float(np.sum(x * y) / denom):.2f}"


def mvpa_window_features(output_dir):
    d = require_csv(output_dir / "mvpa_stim_locked_cat_subject_day_timecourse.csv")
    rows = []
    for (subject, day), g in d.groupby(["subject", "day"]):
        for window, lo, hi in [("early", 0.06, 0.18), ("late", 0.40, 0.60)]:
            h = g[(g["time_sec"] >= lo) & (g["time_sec"] <= hi)]
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "window": window,
                    "auc": float(np.nanmean(h["auc"].to_numpy(float))),
                }
            )
    return pd.DataFrame(rows)


def save_fig_mvpa_stim_locked_cat_time_resolved(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "mvpa_stim_locked_cat_day_means_timecourse.csv")

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for day in DAYS:
        g = d[d["day"] == day].sort_values("time_sec")
        if g.empty:
            raise ValueError(f"Missing MVPA AUC rows for day={day}")
        x = g["time_sec"].to_numpy(float)
        y = g["auc_mean"].to_numpy(float)
        err = g["auc_sem"].to_numpy(float)
        ax.plot(x, y, color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
        ax.fill_between(
            x,
            y - err,
            y + err,
            color=DAY_COLORS[day],
            alpha=0.12,
            linewidth=0,
        )
    ax.axhline(0.5, color="0.25", linewidth=0.8)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("AUC")
    ax.set_title("Time-Resolved Category Decoding")
    ax.legend(frameon=False, ncol=1, loc="upper left")
    setup_axis(ax)
    fig.tight_layout()
    path = figures_dir / "mvpa_stim_locked_cat_time_resolved_auc.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[MVPA stimulus figure] wrote {path}", flush=True)
    return path


def save_fig_mvpa_stim_locked_cat_peak_behavior(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    peaks = require_csv(
        output_dir / "mvpa_stim_locked_cat_haufe_subject_day_peak_times.csv"
    )
    behavior = load_behavior()
    mvpa = mvpa_window_features(output_dir)
    merged = mvpa.merge(behavior, on=["subject", "day"], how="inner")

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    for col_i, peak in enumerate(["early", "late"]):
        ax = axes[0, col_i]
        g = peaks[peaks["peak"] == peak]
        means = []
        errs = []
        for day in DAYS:
            vals = g[g["day"] == day]["peak_auc"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            means.append(float(np.nanmean(vals)))
            if len(vals) <= 1:
                errs.append(np.nan)
            else:
                errs.append(float(np.std(vals, ddof=1) / np.sqrt(len(vals))))
        ax.errorbar(DAYS, means, yerr=errs, color="#303030", marker="o")
        ax.set_title(f"{peak} peak AUC")
        ax.set_xlabel("day")
        ax.set_ylabel("peak AUC")
        ax.set_xticks(DAYS)
        setup_axis(ax)

    for col_i, window in enumerate(["early", "late"]):
        ax = axes[1, col_i]
        g = merged[merged["window"] == window]
        for day in DAYS:
            h = g[g["day"] == day]
            ax.scatter(
                h["auc"],
                h["accuracy"],
                color=DAY_COLORS[day],
                s=24,
                alpha=0.8,
                label=f"D{day}",
            )
        ax.set_title(
            f"{window} AUC vs accuracy, "
            f"{correlation_label(g['auc'], g['accuracy'])}"
        )
        ax.set_xlabel("AUC")
        ax.set_ylabel("Human category choice accuracy")
        setup_axis(ax)
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    path = figures_dir / "mvpa_stim_locked_cat_peak_and_behavior.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[MVPA stimulus figure] wrote {path}", flush=True)
    return path


def save_fig_mvpa_stim_locked_cat_summaries(
    output_dir=OUTPUT_DIR,
    figures_dir=FIGURES_DIR,
):
    return {
        "time_resolved": save_fig_mvpa_stim_locked_cat_time_resolved(
            output_dir, figures_dir
        ),
        "peak_behavior": save_fig_mvpa_stim_locked_cat_peak_behavior(
            output_dir, figures_dir
        ),
    }


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_summaries()
