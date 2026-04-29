#!/usr/bin/env python3
"""ERP utilities to create evoked responses and write figure outputs."""

from pathlib import Path
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from joblib import Parallel, delayed
from util_func_wrangle import util_wrangle_align_beh_to_epochs, util_wrangle_load_sessions

# os.environ["NUMBA_DISABLE_JIT"] = "1"


def _default_erp_n_workers():
    logical = os.cpu_count() or 1
    return min(8, max(1, (logical // 2) - 2))


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


def _process_erp_session(session_item):
    event_names = ["Stim/A", "Stim/B"]
    subject = session_item["subject"]
    day = session_item["day"]
    beh_df = session_item["beh_df"].copy()
    if "epo_path" in session_item:
        epochs = mne.read_epochs(session_item["epo_path"], preload=False, verbose="ERROR")
    else:
        epochs = session_item["epochs"]
    epochs_stim_all, beh_aligned = util_wrangle_align_beh_to_epochs(
        beh_df,
        epochs,
        event_names=event_names,
    )
    if len(epochs_stim_all) == 0:
        return None

    rt_use = beh_aligned["rt"].astype(float).to_numpy() / 1000.0
    fb_use = beh_aligned["fb"].astype(str).str.lower().to_numpy()
    idx_cor = np.where(fb_use == "correct")[0]
    idx_inc = np.where(fb_use == "incorrect")[0]

    result = {
        "subject": subject,
        "day": day,
        "evoked_stim_all": epochs_stim_all.average(),
        "evoked_resp_all": _make_response_locked_evoked(epochs_stim_all, rt_use, t_before=0.6),
    }
    if len(idx_cor) > 0:
        result["evoked_stim_cor"] = epochs_stim_all[idx_cor].average()
        result["evoked_resp_cor"] = _make_response_locked_evoked(
            epochs_stim_all[idx_cor],
            rt_use[idx_cor],
            t_before=0.6,
        )
    if len(idx_inc) > 0:
        result["evoked_stim_inc"] = epochs_stim_all[idx_inc].average()
        result["evoked_resp_inc"] = _make_response_locked_evoked(
            epochs_stim_all[idx_inc],
            rt_use[idx_inc],
            t_before=0.6,
        )
    return result


def util_erp_make_figures(save_figures: bool = True, run_compute: bool = True, n_workers: int | None = None):
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

    d_grand_path = output_dir / "erp_grand_averages_by_day_lock_condition.csv"
    d_subject_path = output_dir / "erp_subject_day_stim_all.csv"
    progress_json = output_dir / "erp_progress.json"
    t0 = time.time()

    def _write_progress(stage: str, done: int = 0, total: int = 0):
        payload = {
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

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

    def _long_to_evoked_map(df, lock_type, condition):
        d_sel = df[(df["lock_type"] == lock_type) & (df["condition"] == condition)].copy()
        if d_sel.empty:
            return {}
        ch_names = sorted(d_sel["channel"].unique().tolist())
        times = np.sort(d_sel["time_s"].unique().astype(float))
        if len(times) < 2:
            return {}
        sfreq = 1.0 / float(np.median(np.diff(times)))
        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types="eeg")
        info.set_montage(mne.channels.make_standard_montage("biosemi64"), on_missing="ignore")
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
            evoked_map[day] = mne.EvokedArray(mat, info=info.copy(), tmin=float(times[0]), nave=1)
        return evoked_map

    if run_compute:
        evoked_stim_all_rec = []
        evoked_stim_cor_rec = []
        evoked_stim_inc_rec = []
        evoked_resp_all_rec = []
        evoked_resp_cor_rec = []
        evoked_resp_inc_rec = []
        sessions = util_wrangle_load_sessions()
        worker_items = [
            {
                "subject": item["subject"],
                "day": item["day"],
                "beh_df": item["beh_df"],
                "epo_path": str(Path("../EEG_epo") / item["epo_file"]),
            }
            for item in sessions
        ]
        if n_workers is None:
            n_workers = _default_erp_n_workers()
        n_workers = max(1, int(n_workers))
        _write_progress("running", 0, len(worker_items))
        if n_workers == 1:
            session_results = []
            for i, item in enumerate(worker_items, start=1):
                session_results.append(_process_erp_session(item))
                _write_progress("running", i, len(worker_items))
        else:
            try:
                result_iter = Parallel(n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered")(
                    delayed(_process_erp_session)(item) for item in worker_items
                )
                session_results = []
                for i, result in enumerate(result_iter, start=1):
                    session_results.append(result)
                    _write_progress("running", i, len(worker_items))
            except PermissionError:
                session_results = []
                for i, item in enumerate(worker_items, start=1):
                    session_results.append(_process_erp_session(item))
                    _write_progress("running", i, len(worker_items))

        for result in session_results:
            if result is None:
                continue
            subject = result["subject"]
            day = result["day"]
            evoked_stim_all_rec.append(
                {
                    "subject": subject,
                    "day": day,
                    "evoked_stim_all": result["evoked_stim_all"],
                }
            )
            evoked_resp_all_rec.append(
                {
                    "subject": subject,
                    "day": day,
                    "evoked_resp_all": result["evoked_resp_all"],
                }
            )
            if "evoked_stim_cor" in result:
                evoked_stim_cor_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_stim_cor": result["evoked_stim_cor"],
                    }
                )
            if "evoked_stim_inc" in result:
                evoked_stim_inc_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_stim_inc": result["evoked_stim_inc"],
                    }
                )
            if "evoked_resp_cor" in result:
                evoked_resp_cor_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_resp_cor": result["evoked_resp_cor"],
                    }
                )
            if "evoked_resp_inc" in result:
                evoked_resp_inc_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_resp_inc": result["evoked_resp_inc"],
                    }
                )

        evoked_stim_all_rec = pd.DataFrame(evoked_stim_all_rec)
        evoked_stim_cor_rec = pd.DataFrame(evoked_stim_cor_rec)
        evoked_stim_inc_rec = pd.DataFrame(evoked_stim_inc_rec)
        evoked_resp_all_rec = pd.DataFrame(evoked_resp_all_rec)
        evoked_resp_cor_rec = pd.DataFrame(evoked_resp_cor_rec)
        evoked_resp_inc_rec = pd.DataFrame(evoked_resp_inc_rec)

        evoked_stim_all_mean = {}
        for day, g in evoked_stim_all_rec.groupby("day"):
            evoked_stim_all_mean[day] = mne.grand_average(g["evoked_stim_all"].tolist())

        evoked_stim_cor_mean = {}
        for day, g in evoked_stim_cor_rec.groupby("day"):
            evoked_stim_cor_mean[day] = mne.grand_average(g["evoked_stim_cor"].tolist())
        evoked_stim_inc_mean = {}
        for day, g in evoked_stim_inc_rec.groupby("day"):
            evoked_stim_inc_mean[day] = mne.grand_average(g["evoked_stim_inc"].tolist())


        evoked_resp_all_mean = {}
        for day, g in evoked_resp_all_rec.groupby("day"):
            evoked_resp_all_mean[day] = mne.grand_average(g["evoked_resp_all"].tolist())

        evoked_resp_cor_mean = {}
        for day, g in evoked_resp_cor_rec.groupby("day"):
            evoked_resp_cor_mean[day] = mne.grand_average(g["evoked_resp_cor"].tolist())
        evoked_resp_inc_mean = {}
        for day, g in evoked_resp_inc_rec.groupby("day"):
            evoked_resp_inc_mean[day] = mne.grand_average(g["evoked_resp_inc"].tolist())
        d_grand = pd.concat(
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
        d_grand.to_csv(d_grand_path, index=False)

        subject_rows = []
        for s in sorted(evoked_stim_all_rec["subject"].unique()):
            d_sub_s = _evoked_map_to_long(
                {
                    int(row["day"]): row["evoked_stim_all"]
                    for _, row in evoked_stim_all_rec[evoked_stim_all_rec["subject"] == s].iterrows()
                },
                "stim",
                "all",
            )
            if not d_sub_s.empty:
                subject_rows.append(d_sub_s.assign(subject=int(s)))
        if len(subject_rows) == 0:
            d_subject = pd.DataFrame(
                columns=["subject", "day", "lock_type", "condition", "channel", "time_s", "amplitude_v"]
            )
        else:
            d_subject = pd.concat(subject_rows, ignore_index=True).sort_values(
                ["subject", "day", "channel", "time_s"]
            )
        d_subject.to_csv(d_subject_path, index=False)
        _write_progress("completed", len(worker_items), len(worker_items))

    # Plot from on-disk outputs (two-step pattern: compute/write -> read/plot).
    if not d_grand_path.exists() or not d_subject_path.exists():
        raise FileNotFoundError(f"Missing ERP output tables in {output_dir}. Run with run_compute=True first.")
    if not save_figures:
        return

    d_grand_plot = pd.read_csv(d_grand_path)
    d_subject_plot = pd.read_csv(d_subject_path)

    evoked_stim_all_plot = _long_to_evoked_map(d_grand_plot, "stim", "all")
    _plot_day_grid(
        evoked_stim_all_plot,
        title="Grand Average ERP: stim_all",
        fig_name="erp_stim_all.png",
    )
    _plot_day_condition_grid(
        {
            (day, "correct"): ev
            for day, ev in _long_to_evoked_map(d_grand_plot, "stim", "correct").items()
        }
        | {
            (day, "incorrect"): ev
            for day, ev in _long_to_evoked_map(d_grand_plot, "stim", "incorrect").items()
        },
        title="Grand Average ERP: stim locked by feedback correctness",
        fig_name="erp_stim_correct_vs_incorrect.png",
    )

    _plot_day_grid(
        _long_to_evoked_map(d_grand_plot, "response", "all"),
        title="Grand Average ERP: response locked (time before response)",
        fig_name="erp_response_all.png",
    )
    _plot_day_condition_grid(
        {
            (day, "correct"): ev
            for day, ev in _long_to_evoked_map(d_grand_plot, "response", "correct").items()
        }
        | {
            (day, "incorrect"): ev
            for day, ev in _long_to_evoked_map(d_grand_plot, "response", "incorrect").items()
        },
        title="Grand Average ERP: response locked by feedback correctness",
        fig_name="erp_response_correct_vs_incorrect.png",
    )

    # Subject-wise stim_all figure from on-disk subject/day table.
    for s in sorted(d_subject_plot["subject"].unique().astype(int)):
        d_sub = d_subject_plot[d_subject_plot["subject"] == s].copy()
        d_sub["lock_type"] = "stim"
        d_sub["condition"] = "all"
        evoked_sub = _long_to_evoked_map(d_sub, "stim", "all")
        days_sorted = sorted(evoked_sub.keys())
        if len(days_sorted) == 0:
            continue
        fig, axes = plt.subplots(1, len(days_sorted), figsize=(5 * len(days_sorted), 4), squeeze=False)
        for i, day in enumerate(days_sorted):
            ax = axes[0, i]
            evoked_sub[day].plot(
                axes=ax,
                show=False,
                spatial_colors=True,
                titles=f"Day {day}",
            )
            ax.set_title(f"Day {day}")
        fig.suptitle(f"ERP: stim_all -- subject {s}")
        fig.savefig(figures_dir / f"erp_stim_all_sub_{s}.png")
        plt.close(fig)


def run_erp(**kwargs):
    """Run ERP analysis."""
    return util_erp_make_figures(save_figures=False, run_compute=True, **kwargs)


def save_fig_erp():
    """Generate ERP figures."""
    return util_erp_make_figures(save_figures=True, run_compute=False)
