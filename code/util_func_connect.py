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
    """Compute stim-locked and response-locked abs(ImCoh) profiles across bands."""

    figures_dir = Path("../figures/connectivity")
    figures_dir.mkdir(parents=True, exist_ok=True)

    mne.set_log_level("ERROR")

    roi_visual = ["O1", "Oz", "O2"]
    roi_motor = ["C3", "Cz", "C4"]

    bands = {
        "all": (None, None),
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
        "low_beta": (12.0, 20.0),
        "high_beta": (20.0, 30.0),
        "beta": (12.0, 30.0),
        "gamma": (30.0, 45.0),
    }
    window_sec = 0.12
    step_sec = 0.01
    stim_plot_tmin = 0.00
    stim_plot_tmax = 0.60
    resp_plot_tmin = 0.00
    resp_plot_tmax = 0.60
    analysis_tmin = stim_plot_tmin
    analysis_tmax = max(stim_plot_tmax, resp_plot_tmax)
    edge_buffer_sec = 0.00

    def _compute_abs_imcoh(x, y):
        sxy = np.mean(x * np.conjugate(y))
        sxx = np.mean(np.abs(x) ** 2)
        syy = np.mean(np.abs(y) ** 2)
        denom = np.sqrt(sxx * syy)
        if (not np.isfinite(denom)) or (denom <= np.finfo(float).eps):
            return np.nan
        coh = sxy / denom
        return float(np.abs(np.imag(coh)))

    def _save_profile_figure(d_lock, fig_name, suptitle, x_label):
        if d_lock.empty:
            print(f"No connectivity values computed for {fig_name}.")
            return

        d_time = (
            d_lock.groupby(["band", "day", "lock_time"], as_index=False)["conn_val"]
            .mean()
            .sort_values(["band", "day", "lock_time"])
        )
        d_peak = (
            d_time.groupby(["band", "day"], as_index=False)["conn_val"]
            .max()
            .rename(columns={"conn_val": "peak_conn"})
        )
        d_time = d_time.merge(d_peak, on=["band", "day"], how="left")
        d_time["peak_conn"] = d_time["peak_conn"].where(
            d_time["peak_conn"] > np.finfo(float).eps,
            np.nan,
        )
        d_time["conn_val_norm"] = d_time["conn_val"] / d_time["peak_conn"]

        n_bands = len(bands)
        n_cols = 3
        n_rows = int(np.ceil(n_bands / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(5 * n_cols, 3.5 * n_rows))

        band_names = list(bands.keys())
        for i_band, band_name in enumerate(band_names):
            r = i_band // n_cols
            c = i_band % n_cols
            ax = axes[r, c]
            d_band = d_time[d_time["band"] == band_name]
            if d_band.empty:
                ax.set_title(f"{band_name} (no data)")
                ax.set_xlabel(x_label)
                ax.set_ylabel("normalized abs(ImCoh)")
                continue
            sns.lineplot(
                data=d_band,
                x="lock_time",
                y="conn_val_norm",
                hue="day",
                ax=ax,
                legend=(i_band == 0),
            )
            ax.set_title(band_name)
            ax.set_ylabel("normalized abs(ImCoh)")
            ax.set_xlabel(x_label)
            if i_band != 0 and ax.get_legend() is not None:
                ax.get_legend().remove()

        for i_extra in range(n_bands, n_rows * n_cols):
            r = i_extra // n_cols
            c = i_extra % n_cols
            axes[r, c].axis("off")

        fig.suptitle(suptitle, y=1.02)
        fig.tight_layout()
        fig_path = figures_dir / fig_name
        try:
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        except PermissionError:
            fallback_dir = Path("./figures/connectivity")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fallback_dir / fig_name
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Primary figure path not writable; saved to {fig_path}")
        plt.close(fig)

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
            rt_sec = (
                dsd["beh_df"].iloc[0]
                .sort_values("trial")["rt"]
                .astype(float)
                .to_numpy()
                / 1000.0
            )

            vis_idx = [epochs.ch_names.index(ch) for ch in roi_visual]
            mot_idx = [epochs.ch_names.index(ch) for ch in roi_motor]

            indices = (
                np.repeat(vis_idx, len(mot_idx)),
                np.tile(mot_idx, len(vis_idx)),
            )

            times = epochs.times

            safe_tmin = max(analysis_tmin, float(times[0]) + edge_buffer_sec)
            safe_tmax = min(analysis_tmax, float(times[-1]) - edge_buffer_sec)
            stim_start_times = np.arange(
                max(stim_plot_tmin, safe_tmin),
                min(stim_plot_tmax, safe_tmax) - window_sec + 1e-12,
                step_sec,
            )
            resp_start_times = np.arange(
                resp_plot_tmin,
                resp_plot_tmax - window_sec + 1e-12,
                step_sec,
            )

            if len(stim_start_times) == 0 and len(resp_start_times) == 0:
                continue

            for band_name, (fmin, fmax) in bands.items():
                epochs_band = epochs.copy()
                if fmin is not None and fmax is not None:
                    epochs_band = epochs_band.filter(
                        l_freq=fmin,
                        h_freq=fmax,
                        method="fir",
                        fir_design="firwin",
                        phase="zero-double",
                        verbose="ERROR",
                    )
                epochs_band = epochs_band.apply_hilbert(envelope=False, verbose="ERROR")
                data = epochs_band.get_data()
                n_trials = min(data.shape[0], len(rt_sec))
                if n_trials == 0:
                    continue
                data = data[:n_trials, :, :]
                rt_sec_use = rt_sec[:n_trials]

                for t_start in stim_start_times:
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
                        val = _compute_abs_imcoh(x, y)
                        if np.isfinite(val):
                            pair_vals.append(val)

                    if len(pair_vals) == 0:
                        continue

                    d_connect_rec.append(
                        {
                            "subject": sbj,
                            "day": day,
                            "lock_type": "stim",
                            "band": band_name,
                            "time_window": f"{t_start:.3f}-{t_end:.3f}",
                            "window_start": float(t_start),
                            "lock_time": float(t_start),
                            "conn_val": float(np.nanmean(pair_vals)),
                        }
                    )

                for tau_start in resp_start_times:
                    tau_end = tau_start + window_sec
                    pair_vals = []
                    for i_vis, i_mot in zip(indices[0], indices[1]):
                        x_chunks = []
                        y_chunks = []
                        for i_trial in range(n_trials):
                            rt = rt_sec_use[i_trial]
                            if (not np.isfinite(rt)) or (rt <= 0):
                                continue

                            seg_tmin = rt - tau_end
                            seg_tmax = rt - tau_start
                            if (seg_tmin < times[0]) or (seg_tmax > times[-1]):
                                continue

                            i0 = int(np.searchsorted(times, seg_tmin, side="left"))
                            i1 = int(np.searchsorted(times, seg_tmax, side="left"))
                            if i1 - i0 < 2:
                                continue

                            x_chunks.append(data[i_trial, i_vis, i0:i1])
                            y_chunks.append(data[i_trial, i_mot, i0:i1])

                        if not x_chunks:
                            continue

                        x = np.concatenate(x_chunks)
                        y = np.concatenate(y_chunks)
                        val = _compute_abs_imcoh(x, y)
                        if np.isfinite(val):
                            pair_vals.append(val)

                    if len(pair_vals) == 0:
                        continue

                    d_connect_rec.append(
                        {
                            "subject": sbj,
                            "day": day,
                            "lock_type": "response",
                            "band": band_name,
                            "time_window": f"{tau_start:.3f}-{tau_end:.3f}",
                            "window_start": float(tau_start),
                            "lock_time": float(tau_start),
                            "conn_val": float(np.nanmean(pair_vals)),
                        }
                    )

    d_connect = pd.DataFrame(d_connect_rec)

    if d_connect.empty:
        print("No connectivity values computed. Check epoch time span and window settings.")
        return

    d_connect = d_connect.sort_values(["lock_type", "band", "subject", "day", "lock_time"])

    _save_profile_figure(
        d_connect[d_connect["lock_type"] == "stim"].copy(),
        fig_name="connectivity_stim_locked.png",
        suptitle="Stim-locked abs(ImCoh), peak-normalized",
        x_label="Time from stimulus onset (s)",
    )
    _save_profile_figure(
        d_connect[d_connect["lock_type"] == "response"].copy(),
        fig_name="connectivity_response_locked.png",
        suptitle="Response-locked abs(ImCoh), peak-normalized",
        x_label="Time before response (s)",
    )


def util_connect_explore_sensorwide_dynamics():
    """Explore 16-channel sensor-space connectivity dynamics for stim and response locks."""

    figures_dir = Path("../figures/connectivity_sensorwide")
    figures_dir.mkdir(parents=True, exist_ok=True)

    mne.set_log_level("ERROR")

    channel_subset = [
        "Fp1", "Fp2", "F7", "F8",
        "Fz", "C3", "Cz", "C4",
        "T7", "T8", "P3", "P4",
        "P7", "P8", "O1", "O2",
    ]
    bands = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 12.0),
    }
    locks = ("stim", "response")

    window_sec = 0.12
    step_sec = 0.01
    stim_tmin = 0.00
    stim_tmax = 0.60
    resp_tmin = 0.00
    resp_tmax = 0.60

    snapshot_targets = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    top_edge_quantile = 0.90

    def _compute_abs_imcoh(x, y):
        sxy = np.mean(x * np.conjugate(y))
        sxx = np.mean(np.abs(x) ** 2)
        syy = np.mean(np.abs(y) ** 2)
        denom = np.sqrt(sxx * syy)
        if (not np.isfinite(denom)) or (denom <= np.finfo(float).eps):
            return np.nan
        coh = sxy / denom
        return float(np.abs(np.imag(coh)))

    def _safe_fig_save(fig, fig_name):
        fig_path = figures_dir / fig_name
        try:
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        except PermissionError:
            fallback_dir = Path("./figures/connectivity_sensorwide")
            fallback_dir.mkdir(parents=True, exist_ok=True)
            fig_path = fallback_dir / fig_name
            fig.savefig(fig_path, dpi=150, bbox_inches="tight")
            print(f"Primary figure path not writable; saved to {fig_path}")
        plt.close(fig)

    def _channel_xy(info, ch_names):
        montage = info.get_montage()
        if montage is None:
            n = len(ch_names)
            ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
            return np.c_[np.cos(ang), np.sin(ang)] * 0.9
        ch_pos = montage.get_positions()["ch_pos"]
        xy = np.array([ch_pos[ch][:2] for ch in ch_names], dtype=float)
        xy = xy - np.mean(xy, axis=0, keepdims=True)
        scale = np.max(np.linalg.norm(xy, axis=1))
        if scale > 0:
            xy = 0.9 * (xy / scale)
        return xy

    def _plot_edge_time_carpet(day_data, pair_idx, lock_name, band_name):
        day_keys = sorted(day_data.keys())
        if not day_keys:
            return
        fig, axes = plt.subplots(len(day_keys), 1, squeeze=False, figsize=(12, 2.5 * len(day_keys)))
        for i_day, day in enumerate(day_keys):
            ax = axes[i_day, 0]
            times = day_data[day]["times"]
            mats = day_data[day]["mats"]
            if (len(times) == 0) or (len(mats) == 0):
                ax.set_title(f"Day {day} (no data)")
                continue
            edge_carpet = np.array([[mat[i, j] for mat in mats] for i, j in pair_idx], dtype=float)
            im = ax.imshow(
                edge_carpet,
                origin="lower",
                aspect="auto",
                extent=[times[0], times[-1], 0, len(pair_idx)],
                cmap="viridis",
            )
            ax.set_title(f"Day {day}")
            ax.set_ylabel("Edge index")
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
        axes[-1, 0].set_xlabel(
            "Time from stimulus onset (s)" if lock_name == "stim" else "Time before response (s)"
        )
        fig.suptitle(f"Edge-time carpet | {lock_name} | {band_name}", y=1.01)
        fig.tight_layout()
        _safe_fig_save(fig, f"carpet_{lock_name}_{band_name}.png")

    def _plot_graph_snapshots(day_data, xy, pair_idx, lock_name, band_name):
        day_keys = sorted(day_data.keys())
        if not day_keys:
            return
        fig, axes = plt.subplots(
            len(day_keys),
            len(snapshot_targets),
            squeeze=False,
            figsize=(3.0 * len(snapshot_targets), 2.8 * len(day_keys)),
        )
        for r, day in enumerate(day_keys):
            times = day_data[day]["times"]
            mats = day_data[day]["mats"]
            for c, t_target in enumerate(snapshot_targets):
                ax = axes[r, c]
                ax.set_aspect("equal")
                ax.axis("off")
                if len(times) == 0:
                    continue
                idx = int(np.argmin(np.abs(times - t_target)))
                t_show = times[idx]
                mat = mats[idx].copy()
                np.fill_diagonal(mat, np.nan)
                upper_vals = np.array([mat[i, j] for i, j in pair_idx], dtype=float)
                finite_vals = upper_vals[np.isfinite(upper_vals)]
                if finite_vals.size == 0:
                    ax.set_title(f"D{day} t={t_show:.2f}")
                    continue
                thr = np.quantile(finite_vals, top_edge_quantile)

                head = plt.Circle((0, 0), 1.0, fill=False, color="black", lw=1.0)
                ax.add_patch(head)
                for i, j in pair_idx:
                    val = mat[i, j]
                    if np.isfinite(val) and (val >= thr):
                        ax.plot(
                            [xy[i, 0], xy[j, 0]],
                            [xy[i, 1], xy[j, 1]],
                            color="#4c72b0",
                            alpha=0.65,
                            lw=1.2,
                        )
                ax.scatter(xy[:, 0], xy[:, 1], s=25, c="black")
                ax.set_xlim(-1.05, 1.05)
                ax.set_ylim(-1.05, 1.05)
                ax.set_title(f"D{day} t={t_show:.2f}")
        fig.suptitle(f"Top-edge graph snapshots | {lock_name} | {band_name}", y=1.01)
        fig.tight_layout()
        _safe_fig_save(fig, f"graphs_{lock_name}_{band_name}.png")

    def _plot_node_strength_topomaps(day_data, info_subset, lock_name, band_name):
        day_keys = sorted(day_data.keys())
        if not day_keys:
            return
        fig, axes = plt.subplots(
            len(day_keys),
            len(snapshot_targets),
            squeeze=False,
            figsize=(3.0 * len(snapshot_targets), 2.8 * len(day_keys)),
        )
        for r, day in enumerate(day_keys):
            times = day_data[day]["times"]
            mats = day_data[day]["mats"]
            for c, t_target in enumerate(snapshot_targets):
                ax = axes[r, c]
                if len(times) == 0:
                    ax.axis("off")
                    continue
                idx = int(np.argmin(np.abs(times - t_target)))
                t_show = times[idx]
                mat = mats[idx].copy()
                mat = np.where(np.isfinite(mat), mat, 0.0)
                np.fill_diagonal(mat, 0.0)
                strength = np.sum(mat, axis=1)
                mne.viz.plot_topomap(strength, info_subset, axes=ax, show=False, contours=0, sensors=True)
                ax.set_title(f"D{day} t={t_show:.2f}")
        fig.suptitle(f"Node-strength topomaps | {lock_name} | {band_name}", y=1.01)
        fig.tight_layout()
        _safe_fig_save(fig, f"topomap_{lock_name}_{band_name}.png")

    sessions = util_wrangle_load_sessions()
    d = pd.DataFrame(sessions)
    if d.empty:
        print("No sessions found.")
        return

    n_channels = len(channel_subset)
    pair_idx = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]

    # key -> [sum, count]
    agg = {}
    info_subset = None
    skipped_sessions = 0
    used_sessions = 0

    for sbj in d["subject"].unique():
        ds = d[d["subject"] == sbj]
        for day in ds["day"].unique():
            dsd = ds[ds["day"] == day]
            epochs = dsd["epochs"].iloc[0]
            epochs = epochs[["Stim/A", "Stim/B"]]
            epochs = epochs.load_data()
            if not all(ch in epochs.ch_names for ch in channel_subset):
                skipped_sessions += 1
                continue
            epochs.pick(channel_subset)
            if info_subset is None:
                info_subset = epochs.info.copy()

            rt_sec = (
                dsd["beh_df"].iloc[0]
                .sort_values("trial")["rt"]
                .astype(float)
                .to_numpy()
                / 1000.0
            )

            times = epochs.times
            stim_starts = np.arange(stim_tmin, stim_tmax - window_sec + 1e-12, step_sec)
            resp_starts = np.arange(resp_tmin, resp_tmax - window_sec + 1e-12, step_sec)

            if (len(stim_starts) == 0) and (len(resp_starts) == 0):
                continue

            for band_name, (fmin, fmax) in bands.items():
                epochs_band = epochs.copy().filter(
                    l_freq=fmin,
                    h_freq=fmax,
                    method="fir",
                    fir_design="firwin",
                    phase="zero-double",
                    verbose="ERROR",
                )
                epochs_band = epochs_band.apply_hilbert(envelope=False, verbose="ERROR")
                data = epochs_band.get_data()
                n_trials = min(data.shape[0], len(rt_sec))
                if n_trials == 0:
                    continue
                data = data[:n_trials, :, :]
                rt_use = rt_sec[:n_trials]

                # Stim-locked windows
                for t_start in stim_starts:
                    t_end = t_start + window_sec
                    i0 = int(np.searchsorted(times, t_start, side="left"))
                    i1 = int(np.searchsorted(times, t_end, side="left"))
                    if i1 - i0 < 2:
                        continue
                    win = data[:, :, i0:i1]
                    lock_time = float(t_start)
                    for i, j in pair_idx:
                        val = _compute_abs_imcoh(win[:, i, :].reshape(-1), win[:, j, :].reshape(-1))
                        if not np.isfinite(val):
                            continue
                        key = ("stim", int(day), band_name, lock_time, i, j)
                        if key not in agg:
                            agg[key] = [0.0, 0]
                        agg[key][0] += val
                        agg[key][1] += 1

                # Response-locked windows (0 at response, increasing rightward means further pre-response)
                for tau_start in resp_starts:
                    tau_end = tau_start + window_sec
                    lock_time = float(tau_start)
                    for i, j in pair_idx:
                        x_chunks = []
                        y_chunks = []
                        for i_trial in range(n_trials):
                            rt = rt_use[i_trial]
                            if (not np.isfinite(rt)) or (rt <= 0):
                                continue
                            seg_tmin = rt - tau_end
                            seg_tmax = rt - tau_start
                            if (seg_tmin < times[0]) or (seg_tmax > times[-1]):
                                continue
                            i0 = int(np.searchsorted(times, seg_tmin, side="left"))
                            i1 = int(np.searchsorted(times, seg_tmax, side="left"))
                            if i1 - i0 < 2:
                                continue
                            x_chunks.append(data[i_trial, i, i0:i1])
                            y_chunks.append(data[i_trial, j, i0:i1])
                        if not x_chunks:
                            continue
                        val = _compute_abs_imcoh(np.concatenate(x_chunks), np.concatenate(y_chunks))
                        if not np.isfinite(val):
                            continue
                        key = ("response", int(day), band_name, lock_time, i, j)
                        if key not in agg:
                            agg[key] = [0.0, 0]
                        agg[key][0] += val
                        agg[key][1] += 1

            used_sessions += 1

    if not agg:
        print("No sensor-wide connectivity values computed.")
        return
    if info_subset is None:
        print("No channel info available for plotting.")
        return

    xy = _channel_xy(info_subset, channel_subset)
    all_days = sorted({k[1] for k in agg.keys()})

    for lock_name in locks:
        for band_name in bands.keys():
            day_data = {}
            for day in all_days:
                times_this = sorted({
                    k[3] for k in agg.keys()
                    if (k[0] == lock_name and k[1] == day and k[2] == band_name)
                })
                mats = []
                for t in times_this:
                    mat = np.full((n_channels, n_channels), np.nan, dtype=float)
                    for i, j in pair_idx:
                        key = (lock_name, day, band_name, t, i, j)
                        if key in agg:
                            s, c = agg[key]
                            mat[i, j] = s / c if c > 0 else np.nan
                            mat[j, i] = mat[i, j]
                    np.fill_diagonal(mat, 0.0)
                    mats.append(mat)
                day_data[day] = {
                    "times": np.array(times_this, dtype=float),
                    "mats": mats,
                }

            _plot_edge_time_carpet(day_data, pair_idx, lock_name, band_name)
            _plot_graph_snapshots(day_data, xy, pair_idx, lock_name, band_name)
            _plot_node_strength_topomaps(day_data, info_subset, lock_name, band_name)

    print(
        f"Sensor-wide dynamics complete. Used sessions: {used_sessions}, "
        f"skipped sessions: {skipped_sessions}. Output dir: {figures_dir}"
    )
