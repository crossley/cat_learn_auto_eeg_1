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


def compute_peak_edge_rows(day_data, pair_idx):
    days = sorted(day_data.keys())
    n_pairs = len(pair_idx)
    carpets = make_carpets(day_data, pair_idx)
    active_pair_idx_by_pct = get_active_pair_idx_by_pct(carpets, n_pairs)
    active_pair_idx = active_pair_idx_by_pct[0.20]

    peak_rows = []
    for day in days:
        times, d = carpets[day]
        active_mean = np.nanmean(d[active_pair_idx, :], axis=0)
        for peak_i, window in enumerate(ACTIVE_PAIR_PEAK_WINDOWS, start=1):
            t_idx, peak_time, peak_val = peak_time_in_window(times, active_mean, window)
            peak_rows.append(
                {
                    "day": day,
                    "peak": peak_i,
                    "time_idx": t_idx,
                    "peak_time": peak_time,
                    "peak_val": peak_val,
                    "pair_vals": d[active_pair_idx, t_idx],
                }
            )
    return peak_rows, active_pair_idx


def get_peak_row(peak_rows, day, peak):
    for row in peak_rows:
        if int(row["day"]) == int(day) and int(row["peak"]) == int(peak):
            return row
    raise ValueError(f"Missing peak edge row for day={day}, peak={peak}")


def finite_abs_max(vals):
    finite_vals = []
    for val in vals:
        if np.isfinite(val):
            finite_vals.append(abs(float(val)))
    if len(finite_vals) == 0:
        raise ValueError("No finite values available for scaling")
    return max(finite_vals)


def draw_sensor_nodes(ax, ch_xy, ch_names):
    ax.scatter(
        ch_xy[:, 0],
        ch_xy[:, 1],
        s=16,
        color="0.25",
        zorder=3,
        linewidths=0,
    )
    for ch_i, ch in enumerate(ch_names):
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


def format_edge_axis(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_signed_edges(ax, values, active_pair_idx, pair_idx, ch_xy, ch_names, vlim):
    draw_sensor_nodes(ax, ch_xy, ch_names)
    denom = max(float(vlim), 1e-12)
    for edge_i, pair_i in enumerate(active_pair_idx):
        val = values[edge_i]
        if not np.isfinite(val):
            continue
        pi, pj = pair_idx[pair_i]
        scaled = min(abs(float(val)) / denom, 1.0)
        color = "tab:red"
        if val < 0:
            color = "tab:blue"
        ax.plot(
            [ch_xy[pi, 0], ch_xy[pj, 0]],
            [ch_xy[pi, 1], ch_xy[pj, 1]],
            color=color,
            linewidth=0.35 + 3.4 * scaled,
            alpha=0.15 + 0.75 * scaled,
            zorder=2,
        )
    format_edge_axis(ax)


def draw_positive_edges(
    ax, values, active_pair_idx, pair_idx, ch_xy, ch_names, vmin, vmax
):
    draw_sensor_nodes(ax, ch_xy, ch_names)
    denom = max(float(vmax) - float(vmin), 1e-12)
    for edge_i, pair_i in enumerate(active_pair_idx):
        val = values[edge_i]
        if not np.isfinite(val):
            continue
        pi, pj = pair_idx[pair_i]
        scaled = min(max((float(val) - float(vmin)) / denom, 0.0), 1.0)
        ax.plot(
            [ch_xy[pi, 0], ch_xy[pj, 0]],
            [ch_xy[pi, 1], ch_xy[pj, 1]],
            color="tab:blue",
            linewidth=0.25 + 1.8 * scaled,
            alpha=0.10 + 0.45 * scaled,
            zorder=2,
        )
    format_edge_axis(ax)


def threshold_values_by_abs(values, keep_prop):
    finite_abs = []
    for val in values:
        if np.isfinite(val):
            finite_abs.append(abs(float(val)))
    if len(finite_abs) == 0:
        raise ValueError("No finite contrast values to threshold")
    top_k = max(1, int(np.ceil(keep_prop * len(finite_abs))))
    threshold = float(np.sort(np.asarray(finite_abs, dtype=float))[-top_k])
    out = np.full(len(values), np.nan, dtype=float)
    for idx, val in enumerate(values):
        if np.isfinite(val) and abs(float(val)) >= threshold:
            out[idx] = float(val)
    return out


def threshold_values_by_value(values, keep_prop):
    finite_vals = []
    for val in values:
        if np.isfinite(val):
            finite_vals.append(float(val))
    if len(finite_vals) == 0:
        raise ValueError("No finite contribution values to threshold")
    top_k = max(1, int(np.ceil(keep_prop * len(finite_vals))))
    threshold = float(np.sort(np.asarray(finite_vals, dtype=float))[-top_k])
    out = np.full(len(values), np.nan, dtype=float)
    for idx, val in enumerate(values):
        if np.isfinite(val) and float(val) >= threshold:
            out[idx] = float(val)
    return out


def zscore_finite_values(values):
    out = np.full(len(values), np.nan, dtype=float)
    finite_vals = []
    finite_idx = []
    for idx, val in enumerate(values):
        if np.isfinite(val):
            finite_idx.append(idx)
            finite_vals.append(float(val))
    if len(finite_vals) < 2:
        return out
    arr = np.asarray(finite_vals, dtype=float)
    std = float(np.std(arr))
    if std <= np.finfo(float).eps:
        return out
    mean = float(np.mean(arr))
    for arr_i, idx in enumerate(finite_idx):
        out[idx] = (float(arr[arr_i]) - mean) / std
    return out


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
    day_data, pair_idx, lock_name, band_name, figures_dir, pct, subject_df, ch_names
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
                "ch_i": ch_names[i],
                "ch_j": ch_names[j],
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
    day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy, ch_names
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError("Peak-edge figure is only defined for stim broadband")

    days = sorted(day_data.keys())
    peak_rows, active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    edge_vals_all = []
    for row in peak_rows:
        for val in row["pair_vals"]:
            if np.isfinite(val):
                edge_vals_all.append(float(val))

    if len(edge_vals_all) == 0:
        raise ValueError("No finite edge values for top-20 peak-edge figure")
    edge_min = float(np.nanmin(edge_vals_all))
    edge_max = float(np.nanmax(edge_vals_all))

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
        draw_positive_edges(
            ax,
            pair_vals,
            active_pair_idx,
            pair_idx,
            ch_xy,
            ch_names,
            edge_min,
            edge_max,
        )
        peak_ms = int(round(float(row["peak_time"]) * 1000.0))
        ax.set_title(f"D{day}: {peak_ms} ms", fontsize=9)
        if day == days[0]:
            ax.set_ylabel(f"Peak {peak_i}", fontsize=9)

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


def plot_active_pair_peak_difference_edges(
    day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy, ch_names
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError(
            "Peak-difference edge figure is only defined for stim broadband"
        )
    days = sorted(day_data.keys())
    peak_rows, active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    contrast_defs = [(2, 1), (3, 1), (3, 2)]
    contrasts = []
    all_vals = []
    for day in days:
        for peak_hi, peak_lo in contrast_defs:
            vals_hi = get_peak_row(peak_rows, day, peak_hi)["pair_vals"]
            vals_lo = get_peak_row(peak_rows, day, peak_lo)["pair_vals"]
            vals = vals_hi - vals_lo
            contrasts.append(
                {
                    "day": day,
                    "peak_hi": peak_hi,
                    "peak_lo": peak_lo,
                    "values": vals,
                }
            )
            for val in vals:
                if np.isfinite(val):
                    all_vals.append(float(val))
    vlim = finite_abs_max(all_vals)
    fig, axes = plt.subplots(
        len(days),
        len(contrast_defs),
        figsize=(2.8 * len(contrast_defs), 10.0),
        squeeze=False,
    )
    for row in contrasts:
        day = int(row["day"])
        col = contrast_defs.index((int(row["peak_hi"]), int(row["peak_lo"])))
        ax = axes[days.index(day), col]
        draw_signed_edges(
            ax, row["values"], active_pair_idx, pair_idx, ch_xy, ch_names, vlim
        )
        ax.set_title(f"P{row['peak_hi']} - P{row['peak_lo']}", fontsize=9)
        if col == 0:
            ax.set_ylabel(f"D{day}", fontsize=9)
    fig.suptitle("Top 20% Edge Change across Peaks")
    fig.subplots_adjust(
        top=0.93,
        bottom=0.04,
        left=0.08,
        right=0.99,
        wspace=0.18,
        hspace=0.34,
    )
    fig_path = (
        figures_dir
        / "sensorwide_active_pair_peak_difference_edges_top20_stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_active_pair_day_pair_difference_edges(
    day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy, ch_names
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError(
            "Day-pair difference edge figure is only defined for stim broadband"
        )
    days = sorted(day_data.keys())
    peak_rows, active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    figure_paths = []
    for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
        all_vals = []
        for day_i in days:
            vals_i = get_peak_row(peak_rows, day_i, peak_i)["pair_vals"]
            for day_j in days:
                if day_i == day_j:
                    continue
                vals_j = get_peak_row(peak_rows, day_j, peak_i)["pair_vals"]
                vals = vals_i - vals_j
                for val in vals:
                    if np.isfinite(val):
                        all_vals.append(float(val))
        vlim = finite_abs_max(all_vals)
        fig, axes = plt.subplots(
            len(days),
            len(days),
            figsize=(2.15 * len(days), 2.15 * len(days)),
            squeeze=False,
        )
        for row_i, day_i in enumerate(days):
            vals_i = get_peak_row(peak_rows, day_i, peak_i)["pair_vals"]
            for col_j, day_j in enumerate(days):
                ax = axes[row_i, col_j]
                if day_i == day_j:
                    ax.axis("off")
                    continue
                vals_j = get_peak_row(peak_rows, day_j, peak_i)["pair_vals"]
                vals = vals_i - vals_j
                draw_signed_edges(
                    ax, vals, active_pair_idx, pair_idx, ch_xy, ch_names, vlim
                )
                if row_i == 0:
                    ax.set_title(f"- D{day_j}", fontsize=8)
                if col_j == 0:
                    ax.set_ylabel(f"D{day_i}", fontsize=8)
        fig.suptitle(f"Top 20% Day-Pair Edge Differences: Peak {peak_i}")
        fig.subplots_adjust(
            top=0.92,
            bottom=0.03,
            left=0.05,
            right=0.99,
            wspace=0.08,
            hspace=0.08,
        )
        fig_path = figures_dir / (
            f"sensorwide_active_pair_day_pair_difference_edges_peak{peak_i}_"
            "top20_stim_broadband.png"
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)
    return figure_paths


def plot_active_pair_day_pair_difference_edges_normalized(
    day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy, ch_names
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError(
            "Normalized day-pair difference edge figure is only defined for "
            "stim broadband"
        )
    days = sorted(day_data.keys())
    peak_rows, active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    figure_paths = []
    for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
        z_rows = {}
        for day in days:
            vals = get_peak_row(peak_rows, day, peak_i)["pair_vals"]
            z_rows[day] = zscore_finite_values(vals)

        all_vals = []
        for day_i in days:
            vals_i = z_rows[day_i]
            for day_j in days:
                if day_i == day_j:
                    continue
                vals_j = z_rows[day_j]
                vals = vals_i - vals_j
                for val in vals:
                    if np.isfinite(val):
                        all_vals.append(float(val))
        vlim = finite_abs_max(all_vals)
        fig, axes = plt.subplots(
            len(days),
            len(days),
            figsize=(2.15 * len(days), 2.15 * len(days)),
            squeeze=False,
        )
        for row_i, day_i in enumerate(days):
            vals_i = z_rows[day_i]
            for col_j, day_j in enumerate(days):
                ax = axes[row_i, col_j]
                if day_i == day_j:
                    ax.axis("off")
                    continue
                vals_j = z_rows[day_j]
                vals = vals_i - vals_j
                draw_signed_edges(
                    ax, vals, active_pair_idx, pair_idx, ch_xy, ch_names, vlim
                )
                if row_i == 0:
                    ax.set_title(f"- D{day_j}", fontsize=8)
                if col_j == 0:
                    ax.set_ylabel(f"D{day_i}", fontsize=8)
        fig.suptitle(
            f"Top 20% Z-Normalized Day-Pair Edge Differences: Peak {peak_i}"
        )
        fig.subplots_adjust(
            top=0.92,
            bottom=0.03,
            left=0.05,
            right=0.99,
            wspace=0.08,
            hspace=0.08,
        )
        fig_path = figures_dir / (
            "sensorwide_active_pair_day_pair_difference_edges_normalized_"
            f"peak{peak_i}_top20_stim_broadband.png"
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)
    return figure_paths


def edge_vector_distance(vec_a, vec_b, metric):
    vals_a = []
    vals_b = []
    for idx, val_a in enumerate(vec_a):
        val_b = vec_b[idx]
        if np.isfinite(val_a) and np.isfinite(val_b):
            vals_a.append(float(val_a))
            vals_b.append(float(val_b))
    if len(vals_a) < 2:
        return np.nan
    arr_a = np.asarray(vals_a, dtype=float)
    arr_b = np.asarray(vals_b, dtype=float)
    if metric == "euclidean":
        return float(np.sqrt(np.sum((arr_a - arr_b) ** 2)))
    if metric == "z_euclidean":
        std_a = float(np.std(arr_a))
        std_b = float(np.std(arr_b))
        if std_a <= np.finfo(float).eps or std_b <= np.finfo(float).eps:
            return np.nan
        z_a = (arr_a - float(np.mean(arr_a))) / std_a
        z_b = (arr_b - float(np.mean(arr_b))) / std_b
        return float(np.sqrt(np.sum((z_a - z_b) ** 2)))
    if metric == "correlation":
        std_a = float(np.std(arr_a))
        std_b = float(np.std(arr_b))
        if std_a <= np.finfo(float).eps or std_b <= np.finfo(float).eps:
            return np.nan
        corr = float(np.corrcoef(arr_a, arr_b)[0, 1])
        return float(1.0 - corr)
    raise ValueError(f"Unknown edge-vector distance metric: {metric}")


def edge_vector_contribution(vec_a, vec_b, metric):
    valid_idx = []
    vals_a = []
    vals_b = []
    for idx, val_a in enumerate(vec_a):
        val_b = vec_b[idx]
        if np.isfinite(val_a) and np.isfinite(val_b):
            valid_idx.append(idx)
            vals_a.append(float(val_a))
            vals_b.append(float(val_b))
    out = np.full(len(vec_a), np.nan, dtype=float)
    if len(vals_a) < 2:
        return out
    arr_a = np.asarray(vals_a, dtype=float)
    arr_b = np.asarray(vals_b, dtype=float)
    if metric == "euclidean":
        contrib = np.abs(arr_a - arr_b)
    elif metric == "z_euclidean":
        std_a = float(np.std(arr_a))
        std_b = float(np.std(arr_b))
        if std_a <= np.finfo(float).eps or std_b <= np.finfo(float).eps:
            return out
        z_a = (arr_a - float(np.mean(arr_a))) / std_a
        z_b = (arr_b - float(np.mean(arr_b))) / std_b
        contrib = np.abs(z_a - z_b)
    else:
        raise ValueError(f"Contribution is not defined for metric: {metric}")
    for contrib_i, edge_i in enumerate(valid_idx):
        out[edge_i] = float(contrib[contrib_i])
    return out


def distance_metric_label(metric):
    if metric == "euclidean":
        return "raw euclidean"
    if metric == "z_euclidean":
        return "z euclidean"
    if metric == "correlation":
        return "correlation"
    raise ValueError(f"Unknown edge-vector distance metric: {metric}")


def plot_active_pair_network_distance_matrices(
    day_data, pair_idx, lock_name, band_name, figures_dir
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError(
            "Network-distance figure is only defined for stim broadband"
    )
    days = sorted(day_data.keys())
    peak_rows, _active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    metrics = ["euclidean", "z_euclidean", "correlation"]
    matrices = {}
    row_vmax = {}
    for metric in metrics:
        finite_vals = []
        for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
            mat = np.full((len(days), len(days)), np.nan, dtype=float)
            for row_i, day_i in enumerate(days):
                vals_i = get_peak_row(peak_rows, day_i, peak_i)["pair_vals"]
                for col_j, day_j in enumerate(days):
                    if day_i == day_j:
                        mat[row_i, col_j] = 0.0
                        continue
                    vals_j = get_peak_row(peak_rows, day_j, peak_i)["pair_vals"]
                    dist = edge_vector_distance(vals_i, vals_j, metric)
                    mat[row_i, col_j] = dist
                    if np.isfinite(dist):
                        finite_vals.append(float(dist))
            matrices[(metric, peak_i)] = mat
        if len(finite_vals) == 0:
            raise ValueError(f"No finite network distances for metric={metric}")
        row_vmax[metric] = float(np.nanmax(np.asarray(finite_vals, dtype=float)))

    labels = []
    for day in days:
        labels.append(f"D{day}")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.82")
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.0), squeeze=False)
    for row_i, metric in enumerate(metrics):
        for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
            ax = axes[row_i, peak_i - 1]
            mat = matrices[(metric, peak_i)]
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                origin="upper",
                cmap=cmap,
                vmin=0.0,
                vmax=row_vmax[metric],
            )
            ax.set_title(f"Peak {peak_i}")
            ax.set_xticks(range(len(days)))
            ax.set_yticks(range(len(days)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Day")
            ax.set_ylabel("Day")
            for r in range(len(days)):
                for c in range(len(days)):
                    if np.isfinite(mat[r, c]):
                        val = float(mat[r, c])
                        color = "white"
                        if val > 0.65 * row_vmax[metric]:
                            color = "black"
                        ax.text(
                            c,
                            r,
                            f"{val:.2f}",
                            ha="center",
                            va="center",
                            fontsize=8,
                            color=color,
                        )
            if peak_i == len(ACTIVE_PAIR_PEAK_WINDOWS):
                cax = fig.add_axes([0.92, 0.68 - row_i * 0.27, 0.015, 0.20])
                fig.colorbar(im, cax=cax, label=distance_metric_label(metric))
            if peak_i == 1:
                ax.text(
                    -0.38,
                    0.50,
                    distance_metric_label(metric),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                )
    fig.suptitle("Top 20% Network Distance across Days")
    fig.subplots_adjust(
        top=0.91,
        bottom=0.06,
        left=0.10,
        right=0.89,
        wspace=0.36,
        hspace=0.44,
    )
    fig_path = (
        figures_dir
        / "sensorwide_active_pair_network_distance_matrices_top20_"
        "stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def active_pair_table(active_pair_idx, pair_idx, ch_names):
    rows = []
    for edge_pos, pair_i in enumerate(active_pair_idx):
        ch_i_idx, ch_j_idx = pair_idx[pair_i]
        rows.append(
            {
                "edge_pos": int(edge_pos),
                "ch_i": ch_names[ch_i_idx],
                "ch_j": ch_names[ch_j_idx],
            }
        )
    return pd.DataFrame(rows)


def subject_peak_edge_vectors(
    subject_df, peak_rows, active_pair_idx, pair_idx, ch_names
):
    active_pairs = active_pair_table(active_pair_idx, pair_idx, ch_names)
    d_subject = subject_df[
        (subject_df["lock_type"] == "stim") & (subject_df["band"] == "broadband")
    ].copy()
    if d_subject.empty:
        raise ValueError("Missing subject-level stim broadband connectivity rows")
    d_active = d_subject.merge(active_pairs, on=["ch_i", "ch_j"], how="inner")
    if d_active.empty:
        raise ValueError("No subject-level rows match top-20% active sensor-pairs")

    subjects = sorted(d_active["subject"].dropna().unique().astype(int).tolist())
    vector_map = {}
    n_edges = len(active_pair_idx)
    for subject in subjects:
        d_subject_one = d_active[d_active["subject"] == subject]
        for row in peak_rows:
            day = int(row["day"])
            peak = int(row["peak"])
            peak_time = float(row["peak_time"])
            d_day = d_subject_one[d_subject_one["day"] == day]
            d_time = d_day[np.isclose(d_day["lock_time"], peak_time)]
            if d_time.empty:
                continue
            vec = np.full(n_edges, np.nan, dtype=float)
            for _, edge_row in d_time.iterrows():
                edge_pos = int(edge_row["edge_pos"])
                vec[edge_pos] = float(edge_row["conn_val"])
            vector_map[(subject, day, peak)] = vec
    if len(vector_map) == 0:
        raise ValueError("No subject-level peak edge vectors could be built")
    return subjects, vector_map


def subject_distance_summary(subjects, vector_map, days, metric, peak):
    mean_mat = np.full((len(days), len(days)), np.nan, dtype=float)
    sem_mat = np.full((len(days), len(days)), np.nan, dtype=float)
    n_mat = np.full((len(days), len(days)), 0, dtype=int)
    for row_i, day_i in enumerate(days):
        for col_j, day_j in enumerate(days):
            if day_i == day_j:
                n_diag = 0
                for subject in subjects:
                    if (subject, day_i, peak) in vector_map:
                        n_diag += 1
                mean_mat[row_i, col_j] = 0.0
                sem_mat[row_i, col_j] = 0.0
                n_mat[row_i, col_j] = int(n_diag)
                continue
            vals = []
            for subject in subjects:
                key_i = (subject, day_i, peak)
                key_j = (subject, day_j, peak)
                if key_i not in vector_map or key_j not in vector_map:
                    continue
                dist = edge_vector_distance(
                    vector_map[key_i], vector_map[key_j], metric
                )
                if np.isfinite(dist):
                    vals.append(float(dist))
            if len(vals) > 0:
                arr = np.asarray(vals, dtype=float)
                mean_mat[row_i, col_j] = float(np.mean(arr))
                n_mat[row_i, col_j] = int(len(arr))
                if len(arr) > 1:
                    sem_mat[row_i, col_j] = float(
                        np.std(arr, ddof=1) / np.sqrt(len(arr))
                    )
    return mean_mat, sem_mat, n_mat


def subject_contribution_summary(subjects, vector_map, days, metric, peak):
    n_edges = None
    for vec in vector_map.values():
        n_edges = len(vec)
        break
    if n_edges is None:
        raise ValueError("No subject edge vectors available for contributions")

    contrib_map = {}
    n_map = {}
    for day_i in days:
        for day_j in days:
            if day_i == day_j:
                continue
            vals = []
            for subject in subjects:
                key_i = (subject, day_i, peak)
                key_j = (subject, day_j, peak)
                if key_i not in vector_map or key_j not in vector_map:
                    continue
                contrib = edge_vector_contribution(
                    vector_map[key_i], vector_map[key_j], metric
                )
                if np.isfinite(contrib).any():
                    vals.append(contrib)
            if len(vals) == 0:
                continue
            arr = np.asarray(vals, dtype=float)
            contrib_map[(day_i, day_j)] = np.nanmean(arr, axis=0)
            n_map[(day_i, day_j)] = int(len(vals))
    if len(contrib_map) == 0:
        raise ValueError(f"No finite subject contributions for metric={metric}")
    return contrib_map, n_map, n_edges


def draw_contribution_edges(
    ax, values, active_pair_idx, pair_idx, ch_xy, ch_names, vmax
):
    draw_sensor_nodes(ax, ch_xy, ch_names)
    denom = max(float(vmax), 1e-12)
    for edge_i, pair_i in enumerate(active_pair_idx):
        val = values[edge_i]
        if not np.isfinite(val):
            continue
        pi, pj = pair_idx[pair_i]
        scaled = min(float(val) / denom, 1.0)
        ax.plot(
            [ch_xy[pi, 0], ch_xy[pj, 0]],
            [ch_xy[pi, 1], ch_xy[pj, 1]],
            color="tab:purple",
            linewidth=0.35 + 3.4 * scaled,
            alpha=0.12 + 0.78 * scaled,
            zorder=2,
        )
    format_edge_axis(ax)


def edge_contributions_to_node_scores(values, active_pair_idx, pair_idx, n_channels):
    scores = np.zeros(n_channels, dtype=float)
    for edge_i, pair_i in enumerate(active_pair_idx):
        val = values[edge_i]
        if not np.isfinite(val):
            continue
        pi, pj = pair_idx[pair_i]
        scores[pi] += float(val)
        scores[pj] += float(val)
    return scores


def draw_node_contribution_scores(ax, scores, ch_xy, ch_names, vmax):
    denom = max(float(vmax), 1e-12)
    scaled = np.asarray(scores, dtype=float) / denom
    scaled = np.clip(scaled, 0.0, 1.0)
    sizes = 20.0 + 210.0 * scaled
    ax.scatter(
        ch_xy[:, 0],
        ch_xy[:, 1],
        s=sizes,
        c=scores,
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
        edgecolors="0.15",
        linewidths=0.35,
        zorder=3,
    )
    for ch_i, ch in enumerate(ch_names):
        color = "black"
        if scaled[ch_i] > 0.55:
            color = "white"
        ax.text(
            float(ch_xy[ch_i, 0]),
            float(ch_xy[ch_i, 1]),
            ch,
            fontsize=5,
            ha="center",
            va="center",
            color=color,
            zorder=4,
        )
    format_edge_axis(ax)


def plot_active_pair_subject_network_distance_matrices(
    day_data, pair_idx, lock_name, band_name, figures_dir, subject_df, ch_names
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError(
            "Subject network-distance figure is only defined for stim broadband"
        )
    days = sorted(day_data.keys())
    peak_rows, active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    subjects, vector_map = subject_peak_edge_vectors(
        subject_df, peak_rows, active_pair_idx, pair_idx, ch_names
    )
    metrics = ["euclidean", "z_euclidean", "correlation"]
    mean_mats = {}
    n_mats = {}
    row_vmax = {}
    for metric in metrics:
        finite_vals = []
        for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
            mean_mat, _sem_mat, n_mat = subject_distance_summary(
                subjects, vector_map, days, metric, peak_i
            )
            mean_mats[(metric, peak_i)] = mean_mat
            n_mats[(metric, peak_i)] = n_mat
            for val in mean_mat[np.isfinite(mean_mat)]:
                finite_vals.append(float(val))
        if len(finite_vals) == 0:
            raise ValueError(f"No finite subject distances for metric={metric}")
        row_vmax[metric] = float(np.nanmax(np.asarray(finite_vals, dtype=float)))

    labels = []
    for day in days:
        labels.append(f"D{day}")
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.82")
    fig, axes = plt.subplots(3, 3, figsize=(10.2, 9.0), squeeze=False)
    for row_i, metric in enumerate(metrics):
        for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
            ax = axes[row_i, peak_i - 1]
            mat = mean_mats[(metric, peak_i)]
            n_mat = n_mats[(metric, peak_i)]
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                origin="upper",
                cmap=cmap,
                vmin=0.0,
                vmax=row_vmax[metric],
            )
            ax.set_title(f"Peak {peak_i}")
            ax.set_xticks(range(len(days)))
            ax.set_yticks(range(len(days)))
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
            ax.set_xlabel("Day")
            ax.set_ylabel("Day")
            for r in range(len(days)):
                for c in range(len(days)):
                    if np.isfinite(mat[r, c]):
                        val = float(mat[r, c])
                        color = "white"
                        if val > 0.65 * row_vmax[metric]:
                            color = "black"
                        ax.text(
                            c,
                            r,
                            f"{val:.2f}\nn={int(n_mat[r, c])}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color=color,
                        )
            if peak_i == len(ACTIVE_PAIR_PEAK_WINDOWS):
                cax = fig.add_axes([0.92, 0.68 - row_i * 0.27, 0.015, 0.20])
                fig.colorbar(im, cax=cax, label=distance_metric_label(metric))
            if peak_i == 1:
                ax.text(
                    -0.38,
                    0.50,
                    distance_metric_label(metric),
                    transform=ax.transAxes,
                    rotation=90,
                    ha="center",
                    va="center",
                    fontsize=11,
                )
    fig.suptitle("Subject-Averaged Top 20% Network Distance")
    fig.subplots_adjust(
        top=0.91,
        bottom=0.06,
        left=0.10,
        right=0.89,
        wspace=0.36,
        hspace=0.44,
    )
    fig_path = (
        figures_dir
        / "sensorwide_active_pair_subject_network_distance_matrices_top20_"
        "stim_broadband.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_active_pair_subject_contribution_edges(
    day_data, pair_idx, lock_name, band_name, figures_dir, subject_df, ch_names, ch_xy
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError(
            "Subject contribution-edge figure is only defined for stim broadband"
        )
    days = sorted(day_data.keys())
    peak_rows, active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    subjects, vector_map = subject_peak_edge_vectors(
        subject_df, peak_rows, active_pair_idx, pair_idx, ch_names
    )
    metric = "z_euclidean"
    figure_paths = []
    for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
        contrib_map, n_map, _n_edges = subject_contribution_summary(
            subjects, vector_map, days, metric, peak_i
        )
        all_vals = []
        plot_map = {}
        for day_i in days:
            for day_j in days:
                if day_i == day_j:
                    continue
                key = (day_i, day_j)
                if key not in contrib_map:
                    continue
                vals = threshold_values_by_value(contrib_map[key], 0.10)
                plot_map[key] = vals
                for val in vals:
                    if np.isfinite(val):
                        all_vals.append(float(val))
        vmax = finite_abs_max(all_vals)
        fig, axes = plt.subplots(
            len(days),
            len(days),
            figsize=(2.15 * len(days), 2.15 * len(days)),
            squeeze=False,
        )
        for row_i, day_i in enumerate(days):
            for col_j, day_j in enumerate(days):
                ax = axes[row_i, col_j]
                if day_i == day_j:
                    ax.axis("off")
                    continue
                key = (day_i, day_j)
                if key not in plot_map:
                    raise ValueError(
                        f"Missing contribution values for D{day_i}-D{day_j}, "
                        f"peak={peak_i}"
                    )
                draw_contribution_edges(
                    ax,
                    plot_map[key],
                    active_pair_idx,
                    pair_idx,
                    ch_xy,
                    ch_names,
                    vmax,
                )
                if row_i == 0:
                    ax.set_title(f"vs D{day_j}", fontsize=8)
                if col_j == 0:
                    ax.set_ylabel(f"D{day_i}", fontsize=8)
                ax.text(
                    0.03,
                    0.05,
                    f"n={n_map[key]}",
                    transform=ax.transAxes,
                    fontsize=6,
                    color="0.15",
                )
        fig.suptitle(
            f"Subject Top Edge Contributions to z-Euclidean Distance: Peak {peak_i}"
        )
        fig.subplots_adjust(
            top=0.92,
            bottom=0.03,
            left=0.05,
            right=0.99,
            wspace=0.08,
            hspace=0.08,
        )
        fig_path = figures_dir / (
            "sensorwide_active_pair_subject_distance_contribution_edges_"
            f"peak{peak_i}_top20_stim_broadband.png"
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)
    return figure_paths


def plot_active_pair_subject_contribution_nodes(
    day_data, pair_idx, lock_name, band_name, figures_dir, subject_df, ch_names, ch_xy
):
    if lock_name != "stim" or band_name != "broadband":
        raise ValueError(
            "Subject contribution-node figure is only defined for stim broadband"
        )
    days = sorted(day_data.keys())
    peak_rows, active_pair_idx = compute_peak_edge_rows(day_data, pair_idx)
    subjects, vector_map = subject_peak_edge_vectors(
        subject_df, peak_rows, active_pair_idx, pair_idx, ch_names
    )
    metric = "z_euclidean"
    figure_paths = []
    for peak_i in range(1, len(ACTIVE_PAIR_PEAK_WINDOWS) + 1):
        contrib_map, n_map, _n_edges = subject_contribution_summary(
            subjects, vector_map, days, metric, peak_i
        )
        node_map = {}
        all_node_vals = []
        for day_i in days:
            for day_j in days:
                if day_i == day_j:
                    continue
                key = (day_i, day_j)
                if key not in contrib_map:
                    continue
                scores = edge_contributions_to_node_scores(
                    contrib_map[key], active_pair_idx, pair_idx, len(ch_names)
                )
                node_map[key] = scores
                for val in scores:
                    if np.isfinite(val):
                        all_node_vals.append(float(val))
        if len(all_node_vals) == 0:
            raise ValueError(f"No finite node contribution scores for peak {peak_i}")
        vmax = float(np.nanmax(np.asarray(all_node_vals, dtype=float)))
        fig, axes = plt.subplots(
            len(days),
            len(days),
            figsize=(2.15 * len(days), 2.15 * len(days)),
            squeeze=False,
        )
        im = None
        for row_i, day_i in enumerate(days):
            for col_j, day_j in enumerate(days):
                ax = axes[row_i, col_j]
                if day_i == day_j:
                    ax.axis("off")
                    continue
                key = (day_i, day_j)
                if key not in node_map:
                    raise ValueError(
                        f"Missing node contribution scores for D{day_i}-D{day_j}, "
                        f"peak={peak_i}"
                    )
                draw_node_contribution_scores(
                    ax, node_map[key], ch_xy, ch_names, vmax
                )
                im = ax.collections[0]
                if row_i == 0:
                    ax.set_title(f"vs D{day_j}", fontsize=8)
                if col_j == 0:
                    ax.set_ylabel(f"D{day_i}", fontsize=8)
                ax.text(
                    0.03,
                    0.05,
                    f"n={n_map[key]}",
                    transform=ax.transAxes,
                    fontsize=6,
                    color="0.15",
                )
        fig.suptitle(
            f"Subject Node Contributions to z-Euclidean Distance: Peak {peak_i}"
        )
        fig.subplots_adjust(
            top=0.92,
            bottom=0.03,
            left=0.05,
            right=0.91,
            wspace=0.08,
            hspace=0.08,
        )
        cax = fig.add_axes([0.93, 0.20, 0.012, 0.55])
        fig.colorbar(im, cax=cax, label="Node contribution")
        fig_path = figures_dir / (
            "sensorwide_active_pair_subject_distance_contribution_nodes_"
            f"peak{peak_i}_top20_stim_broadband.png"
        )
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths.append(fig_path)
    return figure_paths


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
    channel_subset = []
    for ch in d_channels["channel"]:
        channel_subset.append(ch)
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
                    channel_subset,
                )
                figure_paths.append(fig_path)
                fig_path = plot_active_pair_peak_edges(
                    day_data,
                    pair_idx,
                    lock_name,
                    band_name,
                    figures_dir,
                    ch_xy,
                    channel_subset,
                )
                figure_paths.append(fig_path)
                fig_path = plot_active_pair_peak_difference_edges(
                    day_data,
                    pair_idx,
                    lock_name,
                    band_name,
                    figures_dir,
                    ch_xy,
                    channel_subset,
                )
                figure_paths.append(fig_path)
                new_paths = plot_active_pair_day_pair_difference_edges(
                    day_data,
                    pair_idx,
                    lock_name,
                    band_name,
                    figures_dir,
                    ch_xy,
                    channel_subset,
                )
                for fig_path in new_paths:
                    figure_paths.append(fig_path)
                new_paths = plot_active_pair_day_pair_difference_edges_normalized(
                    day_data,
                    pair_idx,
                    lock_name,
                    band_name,
                    figures_dir,
                    ch_xy,
                    channel_subset,
                )
                for fig_path in new_paths:
                    figure_paths.append(fig_path)
                fig_path = plot_active_pair_network_distance_matrices(
                    day_data, pair_idx, lock_name, band_name, figures_dir
                )
                figure_paths.append(fig_path)
                fig_path = plot_active_pair_subject_network_distance_matrices(
                    day_data,
                    pair_idx,
                    lock_name,
                    band_name,
                    figures_dir,
                    d_subject,
                    channel_subset,
                )
                figure_paths.append(fig_path)
                new_paths = plot_active_pair_subject_contribution_edges(
                    day_data,
                    pair_idx,
                    lock_name,
                    band_name,
                    figures_dir,
                    d_subject,
                    channel_subset,
                    ch_xy,
                )
                for fig_path in new_paths:
                    figure_paths.append(fig_path)
                new_paths = plot_active_pair_subject_contribution_nodes(
                    day_data,
                    pair_idx,
                    lock_name,
                    band_name,
                    figures_dir,
                    d_subject,
                    channel_subset,
                    ch_xy,
                )
                for fig_path in new_paths:
                    figure_paths.append(fig_path)
    return {"figure_paths": figure_paths}


if __name__ == "__main__":
    save_fig_sensorwide_connectivity()
