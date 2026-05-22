#!/usr/bin/env python3
"""Plot feedback-locked time-resolved MVPA figures from saved outputs."""

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mvpa_feedback_locked_cat_tg_analysis import FIGURES_DIR, OUTPUT_DIR


def save_fig_mvpa_feedback_locked_cat_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    day_means_csv = output_dir / "mvpa_feedback_locked_cat_time_resolved_day_means_timecourse.csv"
    fig_day_panels = figures_dir / "mvpa_feedback_locked_cat_auc_by_day_panels.png"
    if not day_means_csv.exists():
        raise FileNotFoundError(
            f"Missing feedback time-resolved MVPA output in {output_dir}. "
            "Run mvpa_feedback_locked_cat_time_resolved_analysis.py first."
        )

    day_means_df = pd.read_csv(day_means_csv)
    if day_means_df.empty:
        raise ValueError(f"Empty feedback time-resolved MVPA output: {day_means_csv}")

    days = sorted(day_means_df["day"].unique())
    missing_days = []
    for day in [1, 2, 3, 4, 5]:
        if day not in days:
            missing_days.append(str(day))
    if len(missing_days) > 0:
        raise ValueError(
            f"Missing feedback time-resolved MVPA days in {day_means_csv}: "
            + ", ".join(missing_days)
        )
    fig, axes = plt.subplots(
        1, len(days), figsize=(5 * len(days), 5.2), sharey=True, squeeze=False
    )
    y_upper = float(np.nanmax(day_means_df["auc_mean"] + day_means_df["auc_sem"].fillna(0.0)))
    y_lower = float(np.nanmin(day_means_df["auc_mean"] - day_means_df["auc_sem"].fillna(0.0)))
    y_pad = max(0.02, 0.20 * (y_upper - y_lower))
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
    axes.ravel()[0].set_ylabel("ROC-AUC")
    fig.suptitle("Time-resolved Category Decoding (Feedback-Locked)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_day_panels, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"figure_paths": {"day_panels": fig_day_panels}}


if __name__ == "__main__":
    save_fig_mvpa_feedback_locked_cat_time_resolved()
