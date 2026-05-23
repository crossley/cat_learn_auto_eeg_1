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


def plot_sensor_pair_carpet(day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy):
    days = sorted(day_data.keys())
    n_days = len(days)
    n_pairs = len(pair_idx)
    add_active_row = (lock_name == "stim") and (band_name == "broadband")

    carpets = {}
    for day in days:
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
        raise ValueError(
            f"No sensorwide carpet data available for lock={lock_name}, band={band_name}"
        )

    vmax_candidates = []
    for _, d in carpets.values():
        if np.isfinite(d).any():
            vmax_candidates.append(float(np.nanmax(d)))
    vmax = max(vmax_candidates) if vmax_candidates else 1e-12
    vmax = max(vmax, 1e-12)

    top_k = max(1, int(np.ceil(0.10 * n_pairs)))

    active_pair_idx = []
    active_pair_scores = []
    if add_active_row:
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
            raise ValueError("No finite sensor-pair modulation scores for active row")
        threshold = float(np.sort(finite_scores)[-top_k])
        for pair_i, score in enumerate(active_pair_scores):
            if np.isfinite(score) and score >= threshold:
                active_pair_idx.append(int(pair_i))
        if len(active_pair_idx) == 0:
            raise ValueError("No active sensor-pairs selected for broadband stim row")

    n_rows = 1
    height = 5.2
    if add_active_row:
        n_rows = 2
        height = 7.0
    gridspec_kw = None
    if add_active_row:
        gridspec_kw = {"height_ratios": [3.0, 1.45]}
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
        active_vals = []
        for day in days:
            _times, d = carpets[day]
            active_mean = np.nanmean(d[active_pair_idx, :], axis=0)
            for val in active_mean:
                if np.isfinite(val):
                    active_vals.append(float(val))
        if len(active_vals) > 0:
            y_min = float(np.nanmin(active_vals))
            y_max = float(np.nanmax(active_vals))
            y_pad = 0.08 * max(y_max - y_min, 1e-12)
            active_ylim = (y_min - y_pad, y_max + y_pad)

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

        mats = day_data[day]["mats"]
        for snap_t in SNAP_TIMES_SEC:
            if snap_t < x_min or snap_t > x_max:
                continue
            t_idx = int(np.argmin(np.abs(times - snap_t)))
            mat = mats[t_idx]
            pair_vals_list = []
            for i, j in pair_idx:
                pair_vals_list.append(mat[i, j])
            pair_vals = np.array(pair_vals_list, dtype=float)

            x_frac = (snap_t - x_min) / (x_max - x_min)
            width = 0.13
            inset = ax.inset_axes(
                [max(0.01, min(0.99 - width, x_frac - width / 2.0)), 1.04, width, 0.30],
                transform=ax.transAxes,
            )

            inset.scatter(ch_xy[:, 0], ch_xy[:, 1], s=6, color="0.3", zorder=3, linewidths=0)

            finite_vals = pair_vals[np.isfinite(pair_vals)]
            if len(finite_vals) >= top_k:
                threshold = np.sort(finite_vals)[-top_k]
                max_val = float(finite_vals.max())
                if max_val > 0:
                    for (pi, pj), pv in zip(pair_idx, pair_vals):
                        if np.isfinite(pv) and pv >= threshold:
                            lw = 0.4 + 2.5 * (pv / max_val)
                            inset.plot(
                                [ch_xy[pi, 0], ch_xy[pj, 0]],
                                [ch_xy[pi, 1], ch_xy[pj, 1]],
                                color="tab:blue",
                                linewidth=lw,
                                alpha=0.8,
                                zorder=2,
                            )

            inset.set_aspect("equal")
            inset.axis("off")
            inset.set_title(f"{int(snap_t * 1000)}ms", fontsize=7)

        if add_active_row:
            ax_curve = axes[1, col]
            active_mean = np.nanmean(d[active_pair_idx, :], axis=0)
            ax_curve.plot(times, active_mean, color="black", linewidth=1.8)
            ax_curve.axvline(0.0, color="0.55", linestyle=":", linewidth=0.8)
            ax_curve.set_xlabel(f"{lock_name.capitalize()}-locked time (s)")
            ax_curve.set_title(f"Active pair mean (n={len(active_pair_idx)})")
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
            bottom=0.11,
            top=0.70,
            wspace=0.18,
            hspace=0.58,
        )
        cbar_bottom = 0.40
        cbar_height = 0.25
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
    channels_path = output_dir / "sensorwide_channel_layout.csv"
    if not carpet_path.exists() or not channels_path.exists():
        raise FileNotFoundError(
            f"Missing sensorwide output tables in {output_dir}. "
            "Run connect_sensorwide_analysis.py first."
        )
    d_carpet = pd.read_csv(carpet_path)
    if d_carpet.empty:
        raise ValueError(f"Empty sensorwide carpet output table: {carpet_path}")

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
                day_data[day] = {"times": np.array(times_this, dtype=float), "mats": mats}
            fig_path = plot_sensor_pair_carpet(
                day_data, pair_idx, lock_name, band_name, figures_dir, ch_xy
            )
            figure_paths.append(fig_path)
    return {"figure_paths": figure_paths}


if __name__ == "__main__":
    save_fig_sensorwide_connectivity()
