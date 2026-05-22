#!/usr/bin/env python3
"""Plot core grand-average ERP figures from saved outputs."""

from pathlib import Path
import os

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"


def available_conditions_message(df):
    d_available = df[["lock_type", "condition"]].drop_duplicates()
    d_available = d_available.sort_values(["lock_type", "condition"])
    lines = []
    for _, row in d_available.iterrows():
        lines.append(f"{row['lock_type']} {row['condition']}")
    return "\n".join(lines)


def long_to_evoked_map(df, lock_type, condition):
    d_sel = df[(df["lock_type"] == lock_type) & (df["condition"] == condition)].copy()
    if d_sel.empty:
        return {}
    ch_names = sorted(d_sel["channel"].unique().tolist())
    times = np.sort(d_sel["time_s"].unique().astype(float))
    if len(times) < 2:
        return {}
    sfreq = 1.0 / float(np.median(np.diff(times)))
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
    info.set_montage(
        mne.channels.make_standard_montage("biosemi64"), on_missing="ignore"
    )
    evoked_map = {}
    for day in sorted(d_sel["day"].unique().astype(int)):
        d_day = d_sel[d_sel["day"] == day]
        mat = np.full((len(ch_names), len(times)), np.nan, dtype=float)
        for i_ch, ch in enumerate(ch_names):
            d_ch = d_day[d_day["channel"] == ch].sort_values("time_s")
            if len(d_ch) == 0:
                continue
            t_ch = d_ch["time_s"].to_numpy(dtype=float)
            y_ch = d_ch["amplitude_v"].to_numpy(dtype=float)
            mat[i_ch, :] = np.interp(times, t_ch, y_ch)
        mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
        evoked_map[day] = mne.EvokedArray(
            mat, info=info.copy(), tmin=float(times[0]), nave=1
        )
    return evoked_map


def require_evoked_map(df, lock_type, condition):
    evoked_map = long_to_evoked_map(df, lock_type, condition)
    if len(evoked_map) == 0:
        available = available_conditions_message(df)
        raise ValueError(
            "Missing ERP condition in "
            "erp_grand_average_by_day_lock_condition.csv: "
            f"lock_type={lock_type}, condition={condition}.\n"
            "Rerun code/erp_grand_average_analysis.py so the output table "
            "contains this condition before plotting.\n"
            f"Available conditions:\n{available}"
        )
    return evoked_map


def plot_day_grid(evoked_map, title, fig_path):
    days_sorted = sorted(evoked_map.keys())
    if len(days_sorted) == 0:
        raise ValueError(f"No ERP data available for figure: {fig_path}")
    fig, axes = plt.subplots(
        1, len(days_sorted), figsize=(5 * len(days_sorted), 4), squeeze=False
    )
    for i, day in enumerate(days_sorted):
        ax = axes[0, i]
        evoked_map[day].plot(
            axes=ax, show=False, spatial_colors=True, titles=f"Day {day}"
        )
        ax.set_title(f"Day {day}")
    fig.suptitle(title)
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def plot_day_condition_grid(
    evoked_by_day_cond, title, fig_path, conds=("correct", "incorrect")
):
    days = set()
    for k in evoked_by_day_cond.keys():
        days.add(k[0])
    days_sorted = sorted(days)
    if len(days_sorted) == 0:
        raise ValueError(f"No ERP data available for figure: {fig_path}")
    missing = []
    for day in days_sorted:
        for cond in conds:
            key = (day, cond)
            if key not in evoked_by_day_cond:
                missing.append(f"day={day}, condition={cond}")
    if len(missing) > 0:
        raise ValueError(
            f"Missing ERP day/condition cells for figure {fig_path}:\n"
            + "\n".join(missing)
        )
    fig, axes = plt.subplots(
        len(conds),
        len(days_sorted),
        figsize=(5 * len(days_sorted), 4 * len(conds)),
        squeeze=False,
    )
    for r, cond in enumerate(conds):
        for c, day in enumerate(days_sorted):
            ax = axes[r, c]
            key = (day, cond)
            evoked_by_day_cond[key].plot(
                axes=ax,
                show=False,
                spatial_colors=True,
                titles=f"Day {day} - {cond}",
            )
            ax.set_title(f"Day {day} - {cond}")
    fig.suptitle(title)
    fig.savefig(fig_path)
    plt.close(fig)
    return fig_path


def save_fig_erp_grand_average(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d_grand_path = output_dir / "erp_grand_average_by_day_lock_condition.csv"
    if not d_grand_path.exists():
        raise FileNotFoundError(
            f"Missing ERP output table in {output_dir}. "
            "Run erp_grand_average_analysis.py first."
        )
    d_grand_plot = pd.read_csv(d_grand_path)
    paths = {}
    paths["stim_all"] = plot_day_grid(
        require_evoked_map(d_grand_plot, "stim", "all"),
        title="Grand Average ERP: stim_all",
        fig_path=figures_dir / "erp_grand_average_stim_all.png",
    )
    stim_correct_incorrect = {}
    for day, ev in require_evoked_map(d_grand_plot, "stim", "correct").items():
        stim_correct_incorrect[(day, "correct")] = ev
    for day, ev in require_evoked_map(d_grand_plot, "stim", "incorrect").items():
        stim_correct_incorrect[(day, "incorrect")] = ev
    paths["stim_correct_vs_incorrect"] = plot_day_condition_grid(
        stim_correct_incorrect,
        title="Grand Average ERP: stim locked by feedback correctness",
        fig_path=figures_dir / "erp_grand_average_stim_correct_vs_incorrect.png",
    )
    stim_cat = {}
    for day, ev in require_evoked_map(d_grand_plot, "stim", "cat_a").items():
        stim_cat[(day, "cat_a")] = ev
    for day, ev in require_evoked_map(d_grand_plot, "stim", "cat_b").items():
        stim_cat[(day, "cat_b")] = ev
    paths["stim_cat_a_vs_cat_b"] = plot_day_condition_grid(
        stim_cat,
        title="Grand Average ERP: stim locked by category",
        fig_path=figures_dir / "erp_grand_average_stim_cat_a_vs_cat_b.png",
        conds=("cat_a", "cat_b"),
    )
    paths["feedback_all"] = plot_day_grid(
        require_evoked_map(d_grand_plot, "feedback", "all"),
        title="Grand Average ERP: feedback locked",
        fig_path=figures_dir / "erp_grand_average_feedback_all.png",
    )
    feedback_correct_incorrect = {}
    for day, ev in require_evoked_map(d_grand_plot, "feedback", "correct").items():
        feedback_correct_incorrect[(day, "correct")] = ev
    for day, ev in require_evoked_map(d_grand_plot, "feedback", "incorrect").items():
        feedback_correct_incorrect[(day, "incorrect")] = ev
    paths["feedback_correct_vs_incorrect"] = plot_day_condition_grid(
        feedback_correct_incorrect,
        title="Grand Average ERP: feedback locked by feedback correctness",
        fig_path=figures_dir / "erp_grand_average_feedback_correct_vs_incorrect.png",
    )
    feedback_cat = {}
    for day, ev in require_evoked_map(d_grand_plot, "feedback", "cat_a").items():
        feedback_cat[(day, "cat_a")] = ev
    for day, ev in require_evoked_map(d_grand_plot, "feedback", "cat_b").items():
        feedback_cat[(day, "cat_b")] = ev
    paths["feedback_cat_a_vs_cat_b"] = plot_day_condition_grid(
        feedback_cat,
        title="Grand Average ERP: feedback locked by category",
        fig_path=figures_dir / "erp_grand_average_feedback_cat_a_vs_cat_b.png",
        conds=("cat_a", "cat_b"),
    )
    return paths


if __name__ == "__main__":
    save_fig_erp_grand_average()
