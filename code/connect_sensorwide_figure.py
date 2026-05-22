#!/usr/bin/env python3
"""Plot sensorwide connectivity figures from saved outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connect_sensorwide_analysis import BANDS, CHANNEL_SUBSET, FIGURES_DIR, OUTPUT_DIR


def plot_sensor_pair_carpet(day_data, pair_idx, lock_name, band_name, figures_dir):
    rows = []
    for day in sorted(day_data):
        mats = day_data[day]["mats"]
        times = day_data[day]["times"]
        if len(mats) == 0:
            raise ValueError(
                f"Missing sensorwide carpet matrices: lock={lock_name}, "
                f"band={band_name}, day={day}"
            )
        vals = []
        for mat in mats:
            pair_vals = []
            for i, j in pair_idx:
                pair_vals.append(mat[i, j])
            vals.append(pair_vals)
        d = np.asarray(vals).T
        rows.append((day, times, d))

    if not rows:
        raise ValueError(
            f"No sensorwide carpet data available for lock={lock_name}, band={band_name}"
        )

    n_pairs = len(pair_idx)
    fig, axes = plt.subplots(len(rows), 1, figsize=(10, 2.1 * len(rows)), sharex=True)
    if len(rows) == 1:
        axes = [axes]
    vmax_candidates = []
    for _, _, d in rows:
        if np.isfinite(d).any():
            vmax_candidates.append(float(np.nanmax(np.abs(d))))
    vmax = max(vmax_candidates)
    vmax = max(vmax, 1e-12)
    for ax, (day, times, d) in zip(axes, rows):
        im = ax.imshow(
            d,
            origin="lower",
            aspect="auto",
            extent=[float(times.min()), float(times.max()), 0, n_pairs],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axvline(0.0, color="k", linestyle=":", linewidth=0.8)
        ax.set_ylabel(f"Day {day}\nPair")
    axes[-1].set_xlabel(f"{lock_name.capitalize()}-locked time (s)")
    fig.suptitle(f"Sensorwide Connectivity Carpet: {lock_name}, {band_name}")
    fig.subplots_adjust(right=0.88, hspace=0.25)
    cax = fig.add_axes([0.90, 0.15, 0.02, 0.70])
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
                if len(times_this) == 0:
                    raise ValueError(
                        f"Missing sensorwide carpet data in {carpet_path}: "
                        f"lock={lock_name}, band={band_name}, day={day}"
                    )
                day_data[day] = {"times": np.array(times_this, dtype=float), "mats": mats}
            fig_path = plot_sensor_pair_carpet(
                day_data, pair_idx, lock_name, band_name, figures_dir
            )
            figure_paths.append(fig_path)
    return {"figure_paths": figure_paths}


if __name__ == "__main__":
    save_fig_sensorwide_connectivity()
