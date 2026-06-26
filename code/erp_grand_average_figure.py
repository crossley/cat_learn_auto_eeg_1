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

from figure_style import DAYS, DAY_COLORS, setup_axis

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


def plot_publication_day_grid(evoked_map, title, fig_path):
    days_sorted = sorted(evoked_map.keys())
    if len(days_sorted) == 0:
        raise ValueError(f"No ERP data available for figure: {fig_path}")
    font_context = {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
    with plt.rc_context(font_context):
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        fig.subplots_adjust(
            left=0.06, right=0.99, bottom=0.10, top=0.87,
            wspace=0.42, hspace=0.35,
        )
        ax_cmap = axes[0, 0]
        parent_bbox = ax_cmap.get_position()
        axes_before = set(id(a) for a in fig.axes)
        evoked_map[days_sorted[0]].plot(
            axes=ax_cmap,
            show=False,
            spatial_colors=True,
            titles="",
        )
        sensor_axes = [a for a in fig.axes if id(a) not in axes_before]
        ax_cmap.cla()
        ax_cmap.axis("off")
        for sensor_ax in sensor_axes:
            sensor_ax.set_axes_locator(None)
            title_room = 0.04
            margin = 0.005
            sensor_ax.set_position(
                [
                    parent_bbox.x0 + margin,
                    parent_bbox.y0 + margin,
                    parent_bbox.width - 2 * margin,
                    parent_bbox.height - title_room - 2 * margin,
                ]
            )
            for col in sensor_ax.collections:
                col.set_sizes([120.0])

        erp_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for (row, col), day in zip(erp_positions, days_sorted):
            ax = axes[row, col]
            axes_before = set(id(a) for a in fig.axes)
            evoked_map[day].plot(
                axes=ax,
                show=False,
                spatial_colors=True,
                titles=f"Day {day}",
            )
            ax.set_title(f"Day {day}")
            ax.set_xlim(-0.1, 0.8)
            for txt in list(ax.texts):
                if "ave" in txt.get_text():
                    txt.remove()
            new_axes = [a for a in fig.axes if id(a) not in axes_before]
            for child_ax in new_axes:
                child_ax.remove()

        fig.suptitle(title, fontsize=14)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig_path


def plot_gfp_stim_all(output_dir, figures_dir):
    d = pd.read_csv(output_dir / "erp_grand_average_subject_day_all.csv")
    d = d[(d["lock_type"] == "stim") & (d["condition"] == "all")].copy()
    if d.empty:
        raise ValueError("No stim/all rows in subject-level ERP data")
    rows = []
    for (subject, day, time_s), g in d.groupby(["subject", "day", "time_s"]):
        vals = g["amplitude_v"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "time_s": float(time_s),
                "gfp": float(np.std(vals, ddof=1)),
            }
        )
    gfp_df = pd.DataFrame(rows)
    if gfp_df.empty:
        raise ValueError("Could not compute GFP from subject-level data")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for day in DAYS:
        d_day = gfp_df[gfp_df["day"] == day]
        summary = (
            d_day.groupby("time_s")["gfp"]
            .agg(
                mean="mean",
                sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))),
            )
            .reset_index()
            .sort_values("time_s")
        )
        t = summary["time_s"].to_numpy(float)
        y = summary["mean"].to_numpy(float)
        y_sem = summary["sem"].to_numpy(float)
        color = DAY_COLORS[day]
        ax.plot(t, y * 1e6, color=color, linewidth=2.0, label=f"Day {day}")
        ax.fill_between(
            t,
            (y - y_sem) * 1e6,
            (y + y_sem) * 1e6,
            color=color,
            alpha=0.18,
            linewidth=0,
        )
    ax.axvline(0, color="0.3", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.1, 0.8)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("GFP (uV)")
    ax.set_title("Stimulus-Locked Global Field Power")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    setup_axis(ax)
    fig.tight_layout()
    path = figures_dir / "erp_gfp_stim_all.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


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
    paths["stim_all"] = plot_publication_day_grid(
        require_evoked_map(d_grand_plot, "stim", "all"),
        title="Stimulus-Locked ERPs",
        fig_path=figures_dir / "erp_grand_average_stim_all.png",
    )
    paths["gfp_stim_all"] = plot_gfp_stim_all(output_dir, figures_dir)
    paths["feedback_all"] = plot_publication_day_grid(
        require_evoked_map(d_grand_plot, "feedback", "all"),
        title="Feedback-Locked ERPs",
        fig_path=figures_dir / "erp_grand_average_feedback_all.png",
    )
    return paths


if __name__ == "__main__":
    save_fig_erp_grand_average()
