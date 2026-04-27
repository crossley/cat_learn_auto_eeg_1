#!/usr/bin/env python3
"""ERP utilities to create evoked responses and write figure outputs."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from util_func_wrangle import util_wrangle_load_sessions

# os.environ["NUMBA_DISABLE_JIT"] = "1"


def util_erp_make_figures():
    """
    Build grand-average ERPs and save figures.
    """

    output_dir = Path("../output/erp")
    figures_dir = Path("../figures/erp")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    event_map = {
        "stim_a": ["Stim/A"],
        "stim_b": ["Stim/B"],
        "stim_all": ["Stim/A", "Stim/B"],
        "fb_cor": ["FB/Cor"],
        "fb_inc": ["FB/Inc"],
        "fb_all": ["FB/Cor", "FB/Inc"],
    }

    evoked_stim_all_rec = []
    evoked_stim_cor_rec = []
    evoked_stim_inc_rec = []
    evoked_resp_all_rec = []
    evoked_resp_cor_rec = []
    evoked_resp_inc_rec = []
    sessions = util_wrangle_load_sessions()
    d = pd.DataFrame(sessions)

    def _make_response_locked_evoked(epochs_stim, rt_sec, t_before=0.6):
        """Build response-locked evoked with x-axis as time before response (0 -> t_before)."""
        data = epochs_stim.get_data()
        times = epochs_stim.times
        info = epochs_stim.info.copy()
        sfreq = info["sfreq"]
        tau = np.arange(0.0, t_before + (1.0 / sfreq) / 2.0, 1.0 / sfreq)

        n_trials, n_ch, _ = data.shape
        aligned = np.full((n_trials, n_ch, len(tau)), np.nan, dtype=float)
        for i_trial in range(n_trials):
            rt = rt_sec[i_trial]
            if (not np.isfinite(rt)) or (rt <= 0):
                continue
            sample_t = rt - tau
            valid = (sample_t >= times[0]) & (sample_t <= times[-1])
            if not np.any(valid):
                continue
            for i_ch in range(n_ch):
                aligned[i_trial, i_ch, valid] = np.interp(sample_t[valid], times, data[i_trial, i_ch, :])

        with np.errstate(invalid="ignore"):
            mean_data = np.nanmean(aligned, axis=0)
        # Some response-locked bins can have no valid contributing trials; keep plotting robust.
        mean_data = np.nan_to_num(mean_data, nan=0.0, posinf=0.0, neginf=0.0)
        return mne.EvokedArray(mean_data, info=info, tmin=0.0, nave=n_trials)

    def _plot_day_grid(evoked_map, title, fig_name):
        days_sorted = sorted(evoked_map.keys())
        if len(days_sorted) == 0:
            return
        fig, axes = plt.subplots(1, len(days_sorted), figsize=(5 * len(days_sorted), 4), squeeze=False)
        for i, day in enumerate(days_sorted):
            ax = axes[0, i]
            evoked_map[day].plot(
                axes=ax,
                show=False,
                spatial_colors=True,
                titles=f"Day {day}",
            )
            ax.set_title(f"Day {day}")
        fig.suptitle(title)
        fig.savefig(figures_dir / fig_name)
        plt.close(fig)

    def _plot_day_condition_grid(evoked_by_day_cond, title, fig_name):
        days_sorted = sorted({k[0] for k in evoked_by_day_cond.keys()})
        conds = ["correct", "incorrect"]
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
                if key not in evoked_by_day_cond:
                    ax.set_axis_off()
                    continue
                evoked_by_day_cond[key].plot(
                    axes=ax,
                    show=False,
                    spatial_colors=True,
                    titles=f"Day {day} - {cond}",
                )
                ax.set_title(f"Day {day} - {cond}")
        fig.suptitle(title)
        fig.savefig(figures_dir / fig_name)
        plt.close(fig)

    def _evoked_map_to_long(evoked_map, lock_type, condition):
        rows = []
        for day, evoked in evoked_map.items():
            arr = evoked.data
            for i_ch, ch in enumerate(evoked.ch_names):
                rows.append(
                    pd.DataFrame(
                        {
                            "day": int(day),
                            "lock_type": lock_type,
                            "condition": condition,
                            "channel": ch,
                            "time_s": evoked.times,
                            "amplitude_v": arr[i_ch, :],
                        }
                    )
                )
        if len(rows) == 0:
            return pd.DataFrame(
                columns=["day", "lock_type", "condition", "channel", "time_s", "amplitude_v"]
            )
        return pd.concat(rows, ignore_index=True)

    for sbj in d["subject"].unique():
        ds = d[d["subject"] == sbj]
        for day in ds["day"].unique():
            dsd = ds[ds["day"] == day]
            beh_df = dsd["beh_df"].iloc[0].copy()
            epochs = dsd["epochs"].iloc[0]
            beh_df = beh_df.sort_values("trial").reset_index(drop=True)
            rt_sec = beh_df["rt"].astype(float).to_numpy() / 1000.0
            fb = beh_df["fb"].astype(str).str.lower().to_numpy()

            epochs_stim_all = epochs[event_map["stim_all"]]
            n = min(len(epochs_stim_all), len(beh_df))
            if n == 0:
                continue
            epochs_stim_all = epochs_stim_all[:n]
            rt_use = rt_sec[:n]
            fb_use = fb[:n]
            idx_cor = np.where(fb_use == "correct")[0]
            idx_inc = np.where(fb_use == "incorrect")[0]

            evoked_stim_all = epochs_stim_all.average()
            evoked_stim_all_rec.append(
                {
                    "subject": sbj,
                    "day": day,
                    "evoked_stim_all": evoked_stim_all,
                }
            )
            if len(idx_cor) > 0:
                evoked_stim_cor_rec.append(
                    {
                        "subject": sbj,
                        "day": day,
                        "evoked_stim_cor": epochs_stim_all[idx_cor].average(),
                    }
                )
            if len(idx_inc) > 0:
                evoked_stim_inc_rec.append(
                    {
                        "subject": sbj,
                        "day": day,
                        "evoked_stim_inc": epochs_stim_all[idx_inc].average(),
                    }
                )

            evoked_resp_all = _make_response_locked_evoked(epochs_stim_all, rt_use, t_before=0.6)
            evoked_resp_all_rec.append(
                {
                    "subject": sbj,
                    "day": day,
                    "evoked_resp_all": evoked_resp_all,
                }
            )
            if len(idx_cor) > 0:
                evoked_resp_cor_rec.append(
                    {
                        "subject": sbj,
                        "day": day,
                        "evoked_resp_cor": _make_response_locked_evoked(
                            epochs_stim_all[idx_cor], rt_use[idx_cor], t_before=0.6
                        ),
                    }
                )
            if len(idx_inc) > 0:
                evoked_resp_inc_rec.append(
                    {
                        "subject": sbj,
                        "day": day,
                        "evoked_resp_inc": _make_response_locked_evoked(
                            epochs_stim_all[idx_inc], rt_use[idx_inc], t_before=0.6
                        ),
                    }
                )

    evoked_stim_all_rec = pd.DataFrame(evoked_stim_all_rec)
    evoked_stim_cor_rec = pd.DataFrame(evoked_stim_cor_rec)
    evoked_stim_inc_rec = pd.DataFrame(evoked_stim_inc_rec)
    evoked_resp_all_rec = pd.DataFrame(evoked_resp_all_rec)
    evoked_resp_cor_rec = pd.DataFrame(evoked_resp_cor_rec)
    evoked_resp_inc_rec = pd.DataFrame(evoked_resp_inc_rec)

    for s in evoked_stim_all_rec["subject"].unique():
        ds = evoked_stim_all_rec[evoked_stim_all_rec["subject"] == s]
        days_sorted = sorted(ds["day"].unique())
        fig, axes = plt.subplots(1, len(days_sorted), figsize=(5 * len(days_sorted), 4), squeeze=False)
        for i, day in enumerate(days_sorted):
            ax = axes[0, i]
            evoked = ds.loc[ds["day"] == day, "evoked_stim_all"].iloc[0]
            evoked.plot(
                axes=ax,
                show=False,
                spatial_colors=True,
                titles=f"Day {day}",
            )
            ax.set_title(f"Day {day}")
        fig.suptitle(f"ERP: stim_all -- subject {s}")
        fig.savefig(figures_dir / f"erp_stim_all_sub_{s}.png")
        plt.close(fig)

    evoked_stim_all_mean = {}
    for day, g in evoked_stim_all_rec.groupby("day"):
        evoked_stim_all_mean[day] = mne.grand_average(g["evoked_stim_all"].tolist())

    _plot_day_grid(
        evoked_stim_all_mean,
        title="Grand Average ERP: stim_all",
        fig_name="erp_stim_all.png",
    )

    evoked_stim_cor_mean = {}
    for day, g in evoked_stim_cor_rec.groupby("day"):
        evoked_stim_cor_mean[day] = mne.grand_average(g["evoked_stim_cor"].tolist())
    evoked_stim_inc_mean = {}
    for day, g in evoked_stim_inc_rec.groupby("day"):
        evoked_stim_inc_mean[day] = mne.grand_average(g["evoked_stim_inc"].tolist())

    evoked_stim_by_day_cond = {}
    for day in sorted(set(evoked_stim_all_mean.keys())):
        if day in evoked_stim_cor_mean:
            evoked_stim_by_day_cond[(day, "correct")] = evoked_stim_cor_mean[day]
        if day in evoked_stim_inc_mean:
            evoked_stim_by_day_cond[(day, "incorrect")] = evoked_stim_inc_mean[day]
    _plot_day_condition_grid(
        evoked_stim_by_day_cond,
        title="Grand Average ERP: stim locked by feedback correctness",
        fig_name="erp_stim_correct_vs_incorrect.png",
    )

    evoked_resp_all_mean = {}
    for day, g in evoked_resp_all_rec.groupby("day"):
        evoked_resp_all_mean[day] = mne.grand_average(g["evoked_resp_all"].tolist())
    _plot_day_grid(
        evoked_resp_all_mean,
        title="Grand Average ERP: response locked (time before response)",
        fig_name="erp_response_all.png",
    )

    evoked_resp_cor_mean = {}
    for day, g in evoked_resp_cor_rec.groupby("day"):
        evoked_resp_cor_mean[day] = mne.grand_average(g["evoked_resp_cor"].tolist())
    evoked_resp_inc_mean = {}
    for day, g in evoked_resp_inc_rec.groupby("day"):
        evoked_resp_inc_mean[day] = mne.grand_average(g["evoked_resp_inc"].tolist())
    evoked_resp_by_day_cond = {}
    for day in sorted(set(evoked_resp_all_mean.keys())):
        if day in evoked_resp_cor_mean:
            evoked_resp_by_day_cond[(day, "correct")] = evoked_resp_cor_mean[day]
        if day in evoked_resp_inc_mean:
            evoked_resp_by_day_cond[(day, "incorrect")] = evoked_resp_inc_mean[day]
    _plot_day_condition_grid(
        evoked_resp_by_day_cond,
        title="Grand Average ERP: response locked by feedback correctness",
        fig_name="erp_response_correct_vs_incorrect.png",
    )

    d_out = pd.concat(
        [
            _evoked_map_to_long(evoked_stim_all_mean, "stim", "all"),
            _evoked_map_to_long(evoked_stim_cor_mean, "stim", "correct"),
            _evoked_map_to_long(evoked_stim_inc_mean, "stim", "incorrect"),
            _evoked_map_to_long(evoked_resp_all_mean, "response", "all"),
            _evoked_map_to_long(evoked_resp_cor_mean, "response", "correct"),
            _evoked_map_to_long(evoked_resp_inc_mean, "response", "incorrect"),
        ],
        ignore_index=True,
    ).sort_values(["lock_type", "condition", "day", "channel", "time_s"])
    d_out.to_csv(output_dir / "erp_grand_averages_by_day_lock_condition.csv", index=False)
