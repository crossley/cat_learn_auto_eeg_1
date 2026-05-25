#!/usr/bin/env python3
"""Plot sensorwide connectivity figures from saved outputs."""

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

from connect_sensorwide_analysis import BANDS, CHANNEL_SUBSET, FIGURES_DIR, OUTPUT_DIR

SNAP_TIMES_SEC = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55]
ACTIVE_PAIR_PCTS = [0.05, 0.10, 0.20, 0.30, 0.80]
ACTIVE_PAIR_PEAK_WINDOWS = [
    (0.050, 0.125),
    (0.145, 0.225),
    (0.250, 0.340),
]


def get_day_colors(days):
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(days)))
    day_colors = {}
    for idx, day in enumerate(days):
        day_colors[day] = colors[idx]
    return day_colors


def get_active_pair_idx_by_pct(carpets, n_pairs):
    active_pair_idx_by_pct = {}
    active_pair_scores = []
    days = sorted(carpets.keys())
    for pair_i in range(n_pairs):
        vals = []
        for day in days:
            _times, d = carpets[day]
            for val in d[pair_i, :]:
                if np.isfinite(val):
                    vals.append(float(val))
        score = np.nan
        if len(vals) > 0:
            vals_arr = np.asarray(vals, dtype=float)
            score = float(np.nanmax(vals_arr) - np.nanmin(vals_arr))
        active_pair_scores.append(score)

    finite_scores = np.asarray(active_pair_scores, dtype=float)
    finite_scores = finite_scores[np.isfinite(finite_scores)]
    if len(finite_scores) == 0:
        raise ValueError("No finite sensor-pair modulation scores for active rows")

    for pct in ACTIVE_PAIR_PCTS:
        top_k = max(1, int(np.ceil(pct * n_pairs)))
        threshold = float(np.sort(finite_scores)[-top_k])
        active_pair_idx = []
        for pair_i, score in enumerate(active_pair_scores):
            if np.isfinite(score) and score >= threshold:
                active_pair_idx.append(int(pair_i))
        if len(active_pair_idx) == 0:
            raise ValueError(f"No active sensor-pairs selected for threshold: {pct}")
        active_pair_idx_by_pct[pct] = active_pair_idx

    return active_pair_idx_by_pct


def get_active_ylim(carpets, active_pair_idx_by_pct):
    active_vals = []
    for pct in active_pair_idx_by_pct.keys():
        active_pair_idx = active_pair_idx_by_pct[pct]
        for day in sorted(carpets.keys()):
            _times, d = carpets[day]
            active_mean = np.nanmean(d[active_pair_idx, :], axis=0)
            for val in active_mean:
                if np.isfinite(val):
                    active_vals.append(float(val))
    if len(active_vals) == 0:
        return None
    y_min = float(np.nanmin(active_vals))
    y_max = float(np.nanmax(active_vals))
    y_pad = 0.08 * max(y_max - y_min, 1e-12)
    return (y_min - y_pad, y_max + y_pad)


def make_carpets(day_data, pair_idx):
    carpets = {}
    for day in sorted(day_data.keys()):
        mats = day_data[day]["mats"]
        times = day_data[day]["times"]
        vals = []
        for mat in mats:
            pair_vals = []
            for i, j in pair_idx:
                pair_vals.append(mat[i, j])
            vals.append(pair_vals)
        carpets[day] = (times, np.asarray(vals).T)
    if not carpets:
        raise ValueError("No sensorwide carpet data available")
    return carpets


def peak_time_in_window(times, signal, window):
    lo, hi = window
    candidate_idx = []
    for idx, time_val in enumerate(times):
        if time_val >= lo and time_val <= hi and np.isfinite(signal[idx]):
            candidate_idx.append(idx)
    if len(candidate_idx) == 0:
        raise ValueError(f"No finite active-pair values in peak window {window}")
    best_idx = candidate_idx[0]
    best_val = float(signal[best_idx])
    for idx in candidate_idx:
        val = float(signal[idx])
        if val > best_val:
            best_idx = idx
            best_val = val
    return best_idx, float(times[best_idx]), best_val


def plot_sensor_pair_carpet(
    day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy
):
    days = sorted(day_data.keys())
    n_days = len(days)
    n_pairs = len(pair_idx)
    add_active_row = (lock_name == "stim") and (band_name == "broadband")
    carpets = make_carpets(day_data, pair_idx)

    vmax_candidates = []
    for _, d in carpets.values():
        if np.isfinite(d).any():
            vmax_candidates.append(float(np.nanmax(d)))
    vmax = max(vmax_candidates) if vmax_candidates else 1e-12
    vmax = max(vmax, 1e-12)

    active_pair_idx_by_pct = {}
    if add_active_row:
        active_pair_idx_by_pct = get_active_pair_idx_by_pct(carpets, n_pairs)

    n_rows = 1
    height = 5.2
    if add_active_row:
        n_rows = 1 + len(ACTIVE_PAIR_PCTS)
        height = 11.0
    gridspec_kw = None
    if add_active_row:
        height_ratios = [3.0]
        for _pct in ACTIVE_PAIR_PCTS:
            height_ratios.append(1.1)
        gridspec_kw = {"height_ratios": height_ratios}
    fig, axes = plt.subplots(
        n_rows,
        n_days,
        figsize=(5 * n_days, height),
        gridspec_kw=gridspec_kw,
        squeeze=False,
    )

    im = None
    active_ylim = None
    if add_active_row:
        active_ylim = get_active_ylim(carpets, active_pair_idx_by_pct)

    for col, day in enumerate(days):
        ax = axes[0, col]
        times, d = carpets[day]
        x_min = float(times.min())
        x_max = float(times.max())

        im = ax.imshow(
            d,
            origin="lower",
            aspect="auto",
            extent=[x_min, x_max, 0, n_pairs],
            cmap="viridis",
            vmin=0,
            vmax=vmax,
        )
        ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
        ax.set_xlabel(f"{lock_name.capitalize()}-locked time (s)")
        ax.set_ylabel("Sensor pair")
        ax.set_title(f"Day {day}")

        if add_active_row:
            for row_i, pct in enumerate(ACTIVE_PAIR_PCTS, start=1):
                ax_curve = axes[row_i, col]
                active_pair_idx = active_pair_idx_by_pct[pct]
                active_mean = np.nanmean(d[active_pair_idx, :], axis=0)
                ax_curve.plot(times, active_mean, color="black", linewidth=1.8)
                ax_curve.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
                ax_curve.set_xlabel(f"{lock_name.capitalize()}-locked time (s)")
                pct_label = int(round(pct * 100))
                ax_curve.set_title(
                    f"Top {pct_label}% active pair mean (n={len(active_pair_idx)})"
                )
                if col == 0:
                    ax_curve.set_ylabel("Connectivity")
                if active_ylim is not None:
                    ax_curve.set_ylim(active_ylim)
                ax_curve.grid(alpha=0.25)

    if add_active_row:
        fig.suptitle(f"Sensorwide Connectivity: {lock_name}, {band_name}", y=0.97)
        fig.subplots_adjust(
            left=0.04,
            right=0.91,
            bottom=0.07,
            top=0.82,
            wspace=0.18,
            hspace=0.72,
        )
        cbar_bottom = 0.56
        cbar_height = 0.20
    else:
        fig.suptitle(f"Sensorwide Connectivity: {lock_name}, {band_name}")
        fig.tight_layout(rect=[0, 0, 0.92, 0.78])
        cbar_bottom = 0.10
        cbar_height = 0.55
    cax = fig.add_axes([0.93, cbar_bottom, 0.012, cbar_height])
    fig.colorbar(im, cax=cax, label="Connectivity")
    fig_path = figures_dir / f"sensorwide_carpet_{lock_name}_{band_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_active_pair_overlay(day_data, pair_idx, lock_name, band_name, figures_dir):
    days = sorted(day_data.keys())
    n_pairs = len(pair_idx)
    carpets = make_carpets(day_data, pair_idx)

    active_pair_idx_by_pct = get_active_pair_idx_by_pct(carpets, n_pairs)
    active_ylim = get_active_ylim(carpets, active_pair_idx_by_pct)
    day_colors = get_day_colors(days)

    fig, axes = plt.subplots(
        1,
        len(ACTIVE_PAIR_PCTS),
        figsize=(4.0 * len(ACTIVE_PAIR_PCTS), 3.8),
        sharey=True,
        squeeze=False,
    )
    for col, pct in enumerate(ACTIVE_PAIR_PCTS):
        ax = axes[0, col]
        active_pair_idx = active_pair_idx_by_pct[pct]
        for day in days:
            times, d = carpets[day]
            active_mean = np.nanmean(d[active_pair_idx, :], axis=0)
            ax.plot(
                times,
                active_mean,
                color=day_colors[day],
                linewidth=1.8,
                label=f"D{day}",
            )
        pct_label = int(round(pct * 100))
        ax.set_title(f"Top {pct_label}% (n={len(active_pair_idx)})")
        ax.set_xlabel(f"{lock_name.capitalize()}-locked time (s)")
        ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
        if active_ylim is not None:
            ax.set_ylim(active_ylim)
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("Connectivity")
            ax.legend(frameon=False, fontsize=8, loc="best")

    fig.suptitle(f"Active Sensor-Pair Connectivity: {lock_name}, {band_name}")
    fig.subplots_adjust(
        top=0.82,
        bottom=0.18,
        left=0.06,
        right=0.99,
        wspace=0.16,
    )
    fig_path = figures_dir / (
        f"sensorwide_active_pair_overlay_{lock_name}_{band_name}.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_active_pair_overlay_single_pct(
    day_data, pair_idx, lock_name, band_name, figures_dir, pct, subject_df
):
    days = sorted(day_data.keys())
    n_pairs = len(pair_idx)
    carpets = make_carpets(day_data, pair_idx)

    active_pair_idx_by_pct = get_active_pair_idx_by_pct(carpets, n_pairs)
    active_pair_idx = active_pair_idx_by_pct[pct]
    active_ylim = get_active_ylim(carpets, {pct: active_pair_idx})
    day_colors = get_day_colors(days)
    pct_label = int(round(pct * 100))

    active_pair_rows = []
    for pair_i in active_pair_idx:
        i, j = pair_idx[pair_i]
        active_pair_rows.append(
            {
                "ch_i": CHANNEL_SUBSET[i],
                "ch_j": CHANNEL_SUBSET[j],
            }
        )
    active_pair_df = pd.DataFrame(active_pair_rows)
    d_subject = subject_df[
        (subject_df["lock_type"] == lock_name) & (subject_df["band"] == band_name)
    ].copy()
    if d_subject.empty:
        raise ValueError(
            f"Missing subject-level connectivity rows: {lock_name}, {band_name}"
        )
    d_active = d_subject.merge(active_pair_df, on=["ch_i", "ch_j"], how="inner")
    if d_active.empty:
        raise ValueError(
            f"No subject-level rows match top {pct_label}% active sensor-pairs"
        )

    session_rows = []
    group_cols = ["subject", "day", "lock_time"]
    for key, g in d_active.groupby(group_cols):
        subject, day, lock_time = key
        session_rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "lock_time": float(lock_time),
                "conn_mean": float(np.mean(g["conn_val"])),
            }
        )
    session_df = pd.DataFrame(session_rows)
    stat_rows = []
    for key, g in session_df.groupby(["day", "lock_time"]):
        day, lock_time = key
        vals = np.asarray(g["conn_mean"], dtype=float)
        sem = np.nan
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        stat_rows.append(
            {
                "day": int(day),
                "lock_time": float(lock_time),
                "mean": float(np.mean(vals)),
                "sem": sem,
                "n": int(len(vals)),
            }
        )
    stat_df = pd.DataFrame(stat_rows)

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    for day in days:
        d_day = stat_df[stat_df["day"] == day].sort_values("lock_time")
        if d_day.empty:
            raise ValueError(f"Missing top {pct_label}% overlay stats for day {day}")
        times = np.asarray(d_day["lock_time"], dtype=float)
        active_mean = np.asarray(d_day["mean"], dtype=float)
        active_sem = np.asarray(d_day["sem"], dtype=float)
        ax.plot(
            times,
            active_mean,
            color=day_colors[day],
            linewidth=2.0,
            label=f"D{day}",
        )
        ax.fill_between(
            times,
            active_mean - active_sem,
            active_mean + active_sem,
            color=day_colors[day],
            alpha=0.16,
            linewidth=0,
        )
    ax.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
    ax.set_xlabel(f"{lock_name.capitalize()}-locked time (s)")
    ax.set_ylabel("Connectivity")
    ax.set_title(f"Top {pct_label}% active pairs (n={len(active_pair_idx)})")
    if active_ylim is not None:
        ax.set_ylim(active_ylim)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, loc="best")
    fig.suptitle(f"Active Sensor-Pair Connectivity: {lock_name}, {band_name}")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = figures_dir / (
        f"sensorwide_active_pair_overlay_top{pct_label}_"
        f"{lock_name}_{band_name}.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_active_pair_peak_edges(
    day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError("Peak-edge figure is only defined for stim broadband")

    days = sorted(day_data.keys())
    n_pairs = len(pair_idx)
    carpets = make_carpets(day_data, pair_idx)
    active_pair_idx_by_pct = get_active_pair_idx_by_pct(carpets, n_pairs)
    active_pair_idx = active_pair_idx_by_pct[0.20]

    peak_rows = []
    edge_vals_all = []
    for day in days:
        times, d = carpets[day]
        active_mean = np.nanmean(d[active_pair_idx, :], axis=0)
        for peak_i, window in enumerate(ACTIVE_PAIR_PEAK_WINDOWS, start=1):
            t_idx, peak_time, peak_val = peak_time_in_window(times, active_mean, window)
            pair_vals = d[active_pair_idx, t_idx]
            for val in pair_vals:
                if np.isfinite(val):
                    edge_vals_all.append(float(val))
            peak_rows.append(
                {
                    "day": day,
                    "peak": peak_i,
                    "time_idx": t_idx,
                    "peak_time": peak_time,
                    "peak_val": peak_val,
                    "pair_vals": pair_vals,
                }
            )

    if len(edge_vals_all) == 0:
        raise ValueError("No finite edge values for top-20 peak-edge figure")
    edge_min = float(np.nanmin(edge_vals_all))
    edge_max = float(np.nanmax(edge_vals_all))
    edge_range = max(edge_max - edge_min, 1e-12)

    fig, axes = plt.subplots(
        len(ACTIVE_PAIR_PEAK_WINDOWS),
        len(days),
        figsize=(2.6 * len(days), 6.8),
        squeeze=False,
    )
    for row in peak_rows:
        day = int(row["day"])
        peak_i = int(row["peak"])
        ax = axes[peak_i - 1, days.index(day)]
        pair_vals = row["pair_vals"]
        ax.scatter(
            ch_xy[:, 0],
            ch_xy[:, 1],
            s=16,
            color="0.25",
            zorder=3,
            linewidths=0,
        )
        for edge_i, pair_i in enumerate(active_pair_idx):
            pi, pj = pair_idx[pair_i]
            val = pair_vals[edge_i]
            if not np.isfinite(val):
                continue
            scaled = (float(val) - edge_min) / edge_range
            lw = 0.4 + 3.2 * scaled
            alpha = 0.28 + 0.62 * scaled
            ax.plot(
                [ch_xy[pi, 0], ch_xy[pj, 0]],
                [ch_xy[pi, 1], ch_xy[pj, 1]],
                color="tab:blue",
                linewidth=lw,
                alpha=alpha,
                zorder=2,
            )
        for ch_i, ch in enumerate(CHANNEL_SUBSET):
            ax.text(
                float(ch_xy[ch_i, 0]),
                float(ch_xy[ch_i, 1]),
                ch,
                fontsize=5,
                ha="center",
                va="center",
                color="white",
                zorder=4,
            )
        peak_ms = int(round(float(row["peak_time"]) * 1000.0))
        ax.set_title(f"D{day}: {peak_ms} ms", fontsize=9)
        if day == days[0]:
            ax.set_ylabel(f"Peak {peak_i}", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle("Top 20% Active Sensor-Pair Edges at Estimated Peaks")
    fig.subplots_adjust(
        top=0.90,
        bottom=0.04,
        left=0.05,
        right=0.99,
        wspace=0.18,
        hspace=0.28,
    )
    fig_path = (
        figures_dir
        / "sensorwide_active_pair_peak_edges_top20_stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_sensorwide_connectivity(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    channel_subset: list[str] = CHANNEL_SUBSET,
    bands: dict = BANDS,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    carpet_path = output_dir / "sensorwide_carpet_timeseries.csv"
    subject_path = output_dir / "sensorwide_carpet_subject_timeseries.csv"
    channels_path = output_dir / "sensorwide_channel_layout.csv"
    if not carpet_path.exists() or not channels_path.exists():
        raise FileNotFoundError(
            f"Missing sensorwide output tables in {output_dir}. "
            "Run connect_sensorwide_analysis.py first."
        )
    if not subject_path.exists():
        raise FileNotFoundError(
            f"Missing subject-level sensorwide output: {subject_path}. "
            "Run connect_sensorwide_analysis.py first."
        )
    d_carpet = pd.read_csv(carpet_path)
    if d_carpet.empty:
        raise ValueError(f"Empty sensorwide carpet output table: {carpet_path}")
    d_subject = pd.read_csv(subject_path)
    if d_subject.empty:
        raise ValueError(f"Empty subject-level sensorwide output: {subject_path}")

    d_channels = pd.read_csv(channels_path)
    ch_pos_map = {}
    for _, row in d_channels.iterrows():
        ch_pos_map[row["channel"]] = np.array([float(row["x"]), float(row["y"])])
    missing_ch = []
    for ch in channel_subset:
        if ch not in ch_pos_map:
            missing_ch.append(ch)
    if missing_ch:
        raise ValueError(f"Channel positions missing for: {missing_ch}")
    ch_xy_rows = []
    for ch in channel_subset:
        ch_xy_rows.append(ch_pos_map[ch])
    ch_xy = np.array(ch_xy_rows)

    n_channels = len(channel_subset)
    pair_idx = []
    for i in range(n_channels):
        for j in range(i + 1, n_channels):
            pair_idx.append((i, j))
    ch_to_idx = {}
    for i, ch in enumerate(channel_subset):
        ch_to_idx[ch] = i

    all_days = sorted(d_carpet["day"].dropna().unique().astype(int).tolist())
    figure_paths = []
    for lock_name in ("stim", "feedback"):
        for band_name in bands.keys():
            day_data = {}
            d_lb = d_carpet[
                (d_carpet["lock_type"] == lock_name) & (d_carpet["band"] == band_name)
            ]
            for day in all_days:
                d_day = d_lb[d_lb["day"] == day]
                times_this = sorted(d_day["lock_time"].dropna().unique().tolist())
                if len(times_this) == 0:
                    raise ValueError(
                        f"Missing sensorwide carpet data in {carpet_path}: "
                        f"lock={lock_name}, band={band_name}, day={day}"
                    )
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
                day_data[day] = {
                    "times": np.array(times_this, dtype=float),
                    "mats": mats,
                }
            fig_path = plot_sensor_pair_carpet(
                day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy
            )
            figure_paths.append(fig_path)
            if lock_name == "stim" and band_name == "broadband":
                fig_path = plot_active_pair_overlay(
                    day_data, pair_idx, lock_name, band_name, figures_dir
                )
                figure_paths.append(fig_path)
                fig_path = plot_active_pair_overlay_single_pct(
                    day_data,
                    pair_idx,
                    lock_name,
                    band_name,
                    figures_dir,
                    0.20,
                    d_subject,
                )
                figure_paths.append(fig_path)
                fig_path = plot_active_pair_peak_edges(
                    day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy
                )
                figure_paths.append(fig_path)
    return {"figure_paths": figure_paths}


if __name__ == "__main__":
    save_fig_sensorwide_connectivity()
