#!/usr/bin/env python3
"""Compute visual-motor functional connectivity across days."""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from util_func_wrangle import util_wrangle_load_sessions

# os.environ["NUMBA_DISABLE_JIT"] = "1"


def util_connect_compute_visual_motor():
    """Compute broadband Hilbert-phase imaginary coherence for visual-motor ROIs."""

    figures_dir = Path("../figures/connectivity")
    figures_dir.mkdir(parents=True, exist_ok=True)

    mne.set_log_level("ERROR")

    roi_visual = ["O1", "Oz", "O2"]
    roi_motor = ["C3", "Cz", "C4"]

    broad_fmin = 12.0
    broad_fmax = 30.0
    window_sec = 0.12
    step_sec = 0.02
    plot_tmin = 0.00
    plot_tmax = 0.60
    baseline_tmin = -0.15
    baseline_tmax = 0.00
    analysis_tmin = min(plot_tmin, baseline_tmin)
    analysis_tmax = plot_tmax
    edge_buffer_sec = 0.00

    sessions = util_wrangle_load_sessions()
    d = pd.DataFrame(sessions)
    d_connect_rec = []
    for sbj in d["subject"].unique():
        ds = d[d["subject"] == sbj]
        for day in ds["day"].unique():
            dsd = ds[ds["day"] == day]
            epochs = dsd["epochs"].iloc[0]
            epochs = epochs[["Stim/A", "Stim/B"]]
            epochs = epochs.load_data()
            epochs.pick(roi_visual + roi_motor)

            vis_idx = [epochs.ch_names.index(ch) for ch in roi_visual]
            mot_idx = [epochs.ch_names.index(ch) for ch in roi_motor]

            indices = (
                np.repeat(vis_idx, len(mot_idx)),
                np.tile(mot_idx, len(vis_idx)),
            )

            epochs = (
                epochs.copy()
                .filter(
                    l_freq=broad_fmin,
                    h_freq=broad_fmax,
                    method="fir",
                    fir_design="firwin",
                    phase="zero-double",
                    verbose="ERROR",
                )
                .apply_hilbert(envelope=False, verbose="ERROR")
            )
            data = epochs.get_data()
            times = epochs.times

            safe_tmin = max(analysis_tmin, float(times[0]) + edge_buffer_sec)
            safe_tmax = min(analysis_tmax, float(times[-1]) - edge_buffer_sec)
            start_times = np.arange(safe_tmin, safe_tmax - window_sec + 1e-12, step_sec)

            if len(start_times) == 0:
                continue

            for t_start in start_times:
                t_end = t_start + window_sec
                i0 = int(np.searchsorted(times, t_start, side="left"))
                i1 = int(np.searchsorted(times, t_end, side="left"))
                if i1 - i0 < 2:
                    continue

                win = data[:, :, i0:i1]
                pair_vals = []
                for i_vis, i_mot in zip(indices[0], indices[1]):
                    x = win[:, i_vis, :].reshape(-1)
                    y = win[:, i_mot, :].reshape(-1)

                    sxy = np.mean(x * np.conjugate(y))
                    sxx = np.mean(np.abs(x) ** 2)
                    syy = np.mean(np.abs(y) ** 2)
                    denom = np.sqrt(sxx * syy)
                    # EEG is in Volts, so power terms are tiny; use a strict floor instead of np.isclose().
                    if (not np.isfinite(denom)) or (denom <= np.finfo(float).eps):
                        continue

                    coh = sxy / denom
                    pair_vals.append(float(np.abs(np.imag(coh))))

                if len(pair_vals) == 0:
                    continue

                d_connect_rec.append(
                    {
                        "subject": sbj,
                        "day": day,
                        "time_window": f"{t_start:.3f}-{t_end:.3f}",
                        "window_start": float(t_start),
                        "window_center": float(t_start + window_sec / 2.0),
                        "conn_val": float(np.nanmean(pair_vals)),
                    }
                )

    d_connect = pd.DataFrame(d_connect_rec)

    if d_connect.empty:
        print("No connectivity values computed. Check epoch time span and window settings.")
        return

    d_connect = d_connect.sort_values(["subject", "day", "window_center"])

    # results figure
    fig, ax = plt.subplots(1, 1, squeeze=False, figsize=(8, 5))
    d_time = (
        d_connect.groupby(["day", "window_center"], as_index=False)["conn_val"]
        .mean()
        .sort_values(["day", "window_center"])
    )
    d_base = (
        d_time[
            (d_time["window_center"] >= baseline_tmin)
            & (d_time["window_center"] <= baseline_tmax)
        ]
        .groupby("day", as_index=False)["conn_val"]
        .mean()
        .rename(columns={"conn_val": "baseline_conn"})
    )
    d_base_fallback = (
        d_time[
            (d_time["window_center"] >= plot_tmin)
            & (d_time["window_center"] <= min(plot_tmin + 0.10, plot_tmax))
        ]
        .groupby("day", as_index=False)["conn_val"]
        .mean()
        .rename(columns={"conn_val": "baseline_fallback"})
    )
    d_time = d_time.merge(d_base, on="day", how="left")
    d_time = d_time.merge(d_base_fallback, on="day", how="left")
    d_time["baseline_conn"] = d_time["baseline_conn"].fillna(d_time["baseline_fallback"])
    d_time["baseline_conn"] = d_time["baseline_conn"].fillna(0.0)
    d_time["conn_val_bc"] = d_time["conn_val"] - d_time["baseline_conn"]
    d_peak = (
        d_time.groupby("day", as_index=False)["conn_val_bc"]
        .max()
        .rename(columns={"conn_val_bc": "peak_conn"})
    )
    d_time = d_time.merge(d_peak, on="day", how="left")
    d_time["peak_conn"] = d_time["peak_conn"].where(d_time["peak_conn"] > np.finfo(float).eps, np.nan)
    d_time["conn_val_bc_norm"] = d_time["conn_val_bc"] / d_time["peak_conn"]
    d_time_plot = d_time[
        (d_time["window_center"] >= plot_tmin) & (d_time["window_center"] <= plot_tmax)
    ].copy()

    sns.lineplot(
        data=d_time_plot,
        x="window_center",
        y="conn_val_bc_norm",
        hue="day",
        ax=ax[0, 0],
    )
    ax[0, 0].set_title("Sliding-window baseline-corrected abs(ImCoh), peak-normalized")
    ax[0, 0].set_ylabel("normalized delta abs(Imaginary coherence)")
    ax[0, 0].set_xlabel("Time (s)")
    fig_path = figures_dir / "connectivity.png"
    try:
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    except PermissionError:
        fallback_dir = Path("./figures/connectivity")
        fallback_dir.mkdir(parents=True, exist_ok=True)
        fig_path = fallback_dir / "connectivity.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Primary figure path not writable; saved to {fig_path}")
    plt.close(fig)
