#!/usr/bin/env python3
"""Build grand-average ERPs and save per-subject and group figures."""

from pathlib import Path
import os
import json
import time
import math
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
from joblib import Parallel, delayed

from load_project_data import align_behaviour_to_epochs, load_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8


def make_response_locked_evoked(epochs_stim, rt_sec, t_before=0.6):
    """Build response-locked evoked with x-axis as time before response."""
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
            aligned[i_trial, i_ch, valid] = np.interp(
                sample_t[valid], times, data[i_trial, i_ch, :]
            )

    with np.errstate(invalid="ignore"):
        mean_data = np.nanmean(aligned, axis=0)
    mean_data = np.nan_to_num(mean_data, nan=0.0, posinf=0.0, neginf=0.0)
    return mne.EvokedArray(mean_data, info=info, tmin=0.0, nave=n_trials)


def align_feedback_to_beh(behaviour, epochs, event_names=("FB/Cor", "FB/Inc")):
    behaviour = behaviour.sort_values("trial").reset_index(drop=True)
    event_names = [name for name in event_names if name in epochs.event_id]
    if len(event_names) == 0:
        return epochs[:0], behaviour.iloc[:0].copy()
    epochs_fb = epochs[event_names]
    if len(epochs_fb) == 0:
        return epochs_fb, behaviour.iloc[:0].copy()
    if epochs_fb.metadata is not None and "beh_trial_index" in epochs_fb.metadata:
        trial_index = epochs_fb.metadata["beh_trial_index"].to_numpy(dtype=int)
    else:
        selection = np.asarray(epochs_fb.selection, dtype=int)
        if len(selection) > 1:
            differences = np.diff(np.sort(selection))
            step = int(differences[0])
            for difference in differences[1:]:
                step = math.gcd(step, int(difference))
            offset = int(np.min(selection % step)) if step > 1 else 0
            trial_index = (selection - offset) // step if step > 1 else selection.copy()
        else:
            trial_index = selection.copy()
    valid = (trial_index >= 0) & (trial_index < len(behaviour))
    if not valid.all():
        epochs_fb = epochs_fb[np.where(valid)[0]]
        trial_index = trial_index[valid]
    return epochs_fb, behaviour.iloc[trial_index].reset_index(drop=True)


def process_erp_session(session_item):
    event_names = ["Stim/A", "Stim/B"]
    subject = session_item["subject"]
    day = session_item["day"]
    beh_df = session_item["beh"].copy()
    if "epo_path" in session_item:
        epochs = mne.read_epochs(session_item["epo_path"], preload=False, verbose="ERROR")
    else:
        epochs = session_item["epochs"]
    epochs_stim_all, beh_aligned = align_behaviour_to_epochs(
        beh_df, epochs, event_names=event_names
    )
    if len(epochs_stim_all) == 0:
        return None
    epochs_fb_all, beh_fb_aligned = align_behaviour_to_epochs(
        beh_df, epochs, event_names=("FB/Cor", "FB/Inc")
    )

    rt_use = beh_aligned["rt"].astype(float).to_numpy() / 1000.0
    fb_use = beh_aligned["fb"].astype(str).str.lower().to_numpy()
    cat_use = beh_aligned["cat"].astype(str).to_numpy()
    idx_cor = np.where(fb_use == "correct")[0]
    idx_inc = np.where(fb_use == "incorrect")[0]
    idx_cat_a = np.where(cat_use == "A")[0]
    idx_cat_b = np.where(cat_use == "B")[0]
    fb_epoch_use = beh_fb_aligned["fb"].astype(str).str.lower().to_numpy()
    idx_fb_cor = np.where(fb_epoch_use == "correct")[0]
    idx_fb_inc = np.where(fb_epoch_use == "incorrect")[0]

    result = {
        "subject": subject,
        "day": day,
        "evoked_stim_all": epochs_stim_all.average(),
        "evoked_resp_all": make_response_locked_evoked(epochs_stim_all, rt_use, t_before=0.6),
    }
    if len(epochs_fb_all) > 0:
        result["evoked_feedback_all"] = epochs_fb_all.average()
    if len(idx_cor) > 0:
        result["evoked_stim_cor"] = epochs_stim_all[idx_cor].average()
        result["evoked_resp_cor"] = make_response_locked_evoked(
            epochs_stim_all[idx_cor], rt_use[idx_cor], t_before=0.6
        )
    if len(idx_inc) > 0:
        result["evoked_stim_inc"] = epochs_stim_all[idx_inc].average()
        result["evoked_resp_inc"] = make_response_locked_evoked(
            epochs_stim_all[idx_inc], rt_use[idx_inc], t_before=0.6
        )
    if len(idx_cat_a) > 0:
        result["evoked_stim_cat_a"] = epochs_stim_all[idx_cat_a].average()
    if len(idx_cat_b) > 0:
        result["evoked_stim_cat_b"] = epochs_stim_all[idx_cat_b].average()
    if len(idx_fb_cor) > 0:
        result["evoked_feedback_cor"] = epochs_fb_all[idx_fb_cor].average()
    if len(idx_fb_inc) > 0:
        result["evoked_feedback_inc"] = epochs_fb_all[idx_fb_inc].average()
    return result


def process_erp_session_safe(session_item):
    subject = session_item["subject"]
    day = session_item["day"]
    try:
        result = process_erp_session(session_item)
        if result is None:
            return {
                "ok": False,
                "qc": {
                    "subject": subject,
                    "day": day,
                    "stage": "erp_session",
                    "reason": "no_stim_epochs",
                    "detail": "",
                },
            }
        result["ok"] = True
        return result
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "subject": subject,
                "day": day,
                "stage": "erp_session",
                "reason": "extract_error",
                "detail": str(exc),
            },
        }


def run_erp_grand_average(
    save_figures: bool = True,
    run_compute: bool = True,
    n_workers: int | None = None,
):
    """Build grand-average ERPs and save figures."""
    output_dir = OUTPUT_DIR
    figures_dir = FIGURES_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    d_grand_path = output_dir / "erp_grand_average_by_day_lock_condition.csv"
    d_subject_path = output_dir / "erp_grand_average_subject_day_all.csv"
    qc_path = output_dir / "erp_grand_average_qc.csv"
    progress_json = output_dir / "erp_grand_average_progress.json"
    t0 = time.time()

    def write_progress(stage: str, done: int = 0, total: int = 0):
        payload = {
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

    def plot_day_grid(evoked_map, title, fig_name):
        days_sorted = sorted(evoked_map.keys())
        if len(days_sorted) == 0:
            return
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
        fig.savefig(figures_dir / fig_name)
        plt.close(fig)

    def plot_day_condition_grid(
        evoked_by_day_cond, title, fig_name, conds=("correct", "incorrect")
    ):
        days_sorted = sorted({k[0] for k in evoked_by_day_cond.keys()})
        if len(days_sorted) == 0:
            return
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

    def plot_feedback_frn_difference(df, fig_name):
        channels = ["Fz", "FCz"]
        d = df[
            (df["lock_type"] == "feedback")
            & (df["condition"].isin(["correct", "incorrect"]))
            & (df["channel"].isin(channels))
        ].copy()
        if d.empty:
            return
        d = (
            d.groupby(["day", "condition", "time_s"], as_index=False)["amplitude_v"]
            .mean()
            .sort_values(["day", "condition", "time_s"])
        )
        wide = d.pivot_table(
            index=["day", "time_s"], columns="condition", values="amplitude_v"
        ).reset_index()
        if not {"correct", "incorrect"} <= set(wide.columns):
            return
        wide["difference_uv"] = (wide["incorrect"] - wide["correct"]) * 1e6
        fig, ax = plt.subplots(figsize=(7, 4))
        days = sorted(wide["day"].unique().astype(int))
        cmap = plt.get_cmap("viridis", max(len(days), 2))
        for idx, day in enumerate(days):
            g = wide[wide["day"] == day].sort_values("time_s")
            ax.plot(
                g["time_s"],
                g["difference_uv"],
                linewidth=1.8,
                color=cmap(idx),
                label=f"Day {day}",
            )
        ax.axvline(0, color="0.35", linestyle=":", linewidth=1)
        ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
        ax.axvspan(0.200, 0.300, color="0.5", alpha=0.12, linewidth=0)
        ax.set_xlabel("Time from feedback (s)")
        ax.set_ylabel("Incorrect - correct amplitude (uV)")
        ax.set_title("Feedback FRN Difference at Fz/FCz")
        ax.legend(frameon=False, ncol=1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / fig_name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def plot_stim_difference(df, conditions, channels, fig_name, title, ylabel):
        cond_a, cond_b = conditions
        d = df[
            (df["lock_type"] == "stim")
            & (df["condition"].isin([cond_a, cond_b]))
            & (df["channel"].isin(channels))
        ].copy()
        if d.empty:
            return
        d = (
            d.groupby(["day", "condition", "time_s"], as_index=False)["amplitude_v"]
            .mean()
            .sort_values(["day", "condition", "time_s"])
        )
        wide = d.pivot_table(
            index=["day", "time_s"], columns="condition", values="amplitude_v"
        ).reset_index()
        if not {cond_a, cond_b} <= set(wide.columns):
            return
        wide["difference_uv"] = (wide[cond_a] - wide[cond_b]) * 1e6
        fig, ax = plt.subplots(figsize=(7, 4))
        days = sorted(wide["day"].unique().astype(int))
        cmap = plt.get_cmap("viridis", max(len(days), 2))
        for idx, day in enumerate(days):
            g = wide[wide["day"] == day].sort_values("time_s")
            ax.plot(
                g["time_s"],
                g["difference_uv"],
                linewidth=1.8,
                color=cmap(idx),
                label=f"Day {day}",
            )
        ax.axvline(0, color="0.35", linestyle=":", linewidth=1)
        ax.axhline(0, color="0.35", linestyle=":", linewidth=1)
        ax.axvspan(0.060, 0.180, color="0.5", alpha=0.10, linewidth=0)
        ax.axvspan(0.250, 0.550, color="0.5", alpha=0.07, linewidth=0)
        ax.set_xlabel("Time from stimulus (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False, ncol=1)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / fig_name, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def evoked_map_to_long(evoked_map, lock_type, condition):
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

    if run_compute:
        evoked_stim_all_rec = []
        evoked_stim_cor_rec = []
        evoked_stim_inc_rec = []
        evoked_stim_cat_a_rec = []
        evoked_stim_cat_b_rec = []
        evoked_resp_all_rec = []
        evoked_resp_cor_rec = []
        evoked_resp_inc_rec = []
        evoked_feedback_all_rec = []
        evoked_feedback_cor_rec = []
        evoked_feedback_inc_rec = []
        sessions = load_sessions()
        worker_items = [
            {
                "subject": item["subject"],
                "day": item["day"],
                "beh": item["beh"],
                "epo_path": str(item["epo_path"]),
            }
            for item in sessions
        ]
        if n_workers is None:
            n_workers = N_JOBS
        n_workers = max(1, int(n_workers))
        write_progress("running", 0, len(worker_items))
        if n_workers == 1:
            session_results = []
            for i, item in enumerate(worker_items, start=1):
                session_results.append(process_erp_session_safe(item))
                write_progress("running", i, len(worker_items))
        else:
            result_iter = Parallel(
                n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
            )(delayed(process_erp_session_safe)(item) for item in worker_items)
            session_results = []
            for i, result in enumerate(result_iter, start=1):
                session_results.append(result)
                write_progress("running", i, len(worker_items))

        qc_rows = []
        for result in session_results:
            if result is None:
                continue
            if not result.get("ok", True):
                qc_rows.append(result["qc"])
                continue
            subject = result["subject"]
            day = result["day"]
            evoked_stim_all_rec.append(
                {"subject": subject, "day": day, "evoked_stim_all": result["evoked_stim_all"]}
            )
            evoked_resp_all_rec.append(
                {"subject": subject, "day": day, "evoked_resp_all": result["evoked_resp_all"]}
            )
            if "evoked_feedback_all" in result:
                evoked_feedback_all_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_feedback_all": result["evoked_feedback_all"],
                    }
                )
            if "evoked_stim_cor" in result:
                evoked_stim_cor_rec.append(
                    {"subject": subject, "day": day, "evoked_stim_cor": result["evoked_stim_cor"]}
                )
            if "evoked_stim_inc" in result:
                evoked_stim_inc_rec.append(
                    {"subject": subject, "day": day, "evoked_stim_inc": result["evoked_stim_inc"]}
                )
            if "evoked_stim_cat_a" in result:
                evoked_stim_cat_a_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_stim_cat_a": result["evoked_stim_cat_a"],
                    }
                )
            if "evoked_stim_cat_b" in result:
                evoked_stim_cat_b_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_stim_cat_b": result["evoked_stim_cat_b"],
                    }
                )
            if "evoked_resp_cor" in result:
                evoked_resp_cor_rec.append(
                    {"subject": subject, "day": day, "evoked_resp_cor": result["evoked_resp_cor"]}
                )
            if "evoked_resp_inc" in result:
                evoked_resp_inc_rec.append(
                    {"subject": subject, "day": day, "evoked_resp_inc": result["evoked_resp_inc"]}
                )
            if "evoked_feedback_cor" in result:
                evoked_feedback_cor_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_feedback_cor": result["evoked_feedback_cor"],
                    }
                )
            if "evoked_feedback_inc" in result:
                evoked_feedback_inc_rec.append(
                    {
                        "subject": subject,
                        "day": day,
                        "evoked_feedback_inc": result["evoked_feedback_inc"],
                    }
                )
        pd.DataFrame(qc_rows).to_csv(qc_path, index=False)

        evoked_stim_all_rec = pd.DataFrame(evoked_stim_all_rec)
        evoked_stim_cor_rec = pd.DataFrame(evoked_stim_cor_rec)
        evoked_stim_inc_rec = pd.DataFrame(evoked_stim_inc_rec)
        evoked_stim_cat_a_rec = pd.DataFrame(evoked_stim_cat_a_rec)
        evoked_stim_cat_b_rec = pd.DataFrame(evoked_stim_cat_b_rec)
        evoked_resp_all_rec = pd.DataFrame(evoked_resp_all_rec)
        evoked_resp_cor_rec = pd.DataFrame(evoked_resp_cor_rec)
        evoked_resp_inc_rec = pd.DataFrame(evoked_resp_inc_rec)
        evoked_feedback_all_rec = pd.DataFrame(evoked_feedback_all_rec)
        evoked_feedback_cor_rec = pd.DataFrame(evoked_feedback_cor_rec)
        evoked_feedback_inc_rec = pd.DataFrame(evoked_feedback_inc_rec)

        evoked_stim_all_mean = {}
        for day, g in evoked_stim_all_rec.groupby("day"):
            evoked_stim_all_mean[day] = mne.grand_average(g["evoked_stim_all"].tolist())

        evoked_stim_cor_mean = {}
        for day, g in evoked_stim_cor_rec.groupby("day"):
            evoked_stim_cor_mean[day] = mne.grand_average(g["evoked_stim_cor"].tolist())
        evoked_stim_inc_mean = {}
        for day, g in evoked_stim_inc_rec.groupby("day"):
            evoked_stim_inc_mean[day] = mne.grand_average(g["evoked_stim_inc"].tolist())
        evoked_stim_cat_a_mean = {}
        for day, g in evoked_stim_cat_a_rec.groupby("day"):
            evoked_stim_cat_a_mean[day] = mne.grand_average(
                g["evoked_stim_cat_a"].tolist()
            )
        evoked_stim_cat_b_mean = {}
        for day, g in evoked_stim_cat_b_rec.groupby("day"):
            evoked_stim_cat_b_mean[day] = mne.grand_average(
                g["evoked_stim_cat_b"].tolist()
            )

        evoked_resp_all_mean = {}
        for day, g in evoked_resp_all_rec.groupby("day"):
            evoked_resp_all_mean[day] = mne.grand_average(g["evoked_resp_all"].tolist())

        evoked_resp_cor_mean = {}
        for day, g in evoked_resp_cor_rec.groupby("day"):
            evoked_resp_cor_mean[day] = mne.grand_average(g["evoked_resp_cor"].tolist())
        evoked_resp_inc_mean = {}
        for day, g in evoked_resp_inc_rec.groupby("day"):
            evoked_resp_inc_mean[day] = mne.grand_average(g["evoked_resp_inc"].tolist())

        evoked_feedback_all_mean = {}
        for day, g in evoked_feedback_all_rec.groupby("day"):
            evoked_feedback_all_mean[day] = mne.grand_average(g["evoked_feedback_all"].tolist())

        evoked_feedback_cor_mean = {}
        for day, g in evoked_feedback_cor_rec.groupby("day"):
            evoked_feedback_cor_mean[day] = mne.grand_average(g["evoked_feedback_cor"].tolist())
        evoked_feedback_inc_mean = {}
        for day, g in evoked_feedback_inc_rec.groupby("day"):
            evoked_feedback_inc_mean[day] = mne.grand_average(g["evoked_feedback_inc"].tolist())

        d_grand = pd.concat(
            [
                evoked_map_to_long(evoked_stim_all_mean, "stim", "all"),
                evoked_map_to_long(evoked_stim_cor_mean, "stim", "correct"),
                evoked_map_to_long(evoked_stim_inc_mean, "stim", "incorrect"),
                evoked_map_to_long(evoked_stim_cat_a_mean, "stim", "cat_a"),
                evoked_map_to_long(evoked_stim_cat_b_mean, "stim", "cat_b"),
                evoked_map_to_long(evoked_resp_all_mean, "response", "all"),
                evoked_map_to_long(evoked_resp_cor_mean, "response", "correct"),
                evoked_map_to_long(evoked_resp_inc_mean, "response", "incorrect"),
                evoked_map_to_long(evoked_feedback_all_mean, "feedback", "all"),
                evoked_map_to_long(evoked_feedback_cor_mean, "feedback", "correct"),
                evoked_map_to_long(evoked_feedback_inc_mean, "feedback", "incorrect"),
            ],
            ignore_index=True,
        ).sort_values(["lock_type", "condition", "day", "channel", "time_s"])
        d_grand.to_csv(d_grand_path, index=False)

        subject_rows = []
        for s in sorted(evoked_stim_all_rec["subject"].unique()):
            d_stim_s = evoked_map_to_long(
                {
                    int(row["day"]): row["evoked_stim_all"]
                    for _, row in evoked_stim_all_rec[
                        evoked_stim_all_rec["subject"] == s
                    ].iterrows()
                },
                "stim",
                "all",
            )
            d_resp_s = evoked_map_to_long(
                {
                    int(row["day"]): row["evoked_resp_all"]
                    for _, row in evoked_resp_all_rec[
                        evoked_resp_all_rec["subject"] == s
                    ].iterrows()
                },
                "response",
                "all",
            )
            d_feedback_s = evoked_map_to_long(
                {
                    int(row["day"]): row["evoked_feedback_all"]
                    for _, row in evoked_feedback_all_rec[
                        evoked_feedback_all_rec["subject"] == s
                    ].iterrows()
                },
                "feedback",
                "all",
            )
            d_sub_s = pd.concat([d_stim_s, d_resp_s, d_feedback_s], ignore_index=True)
            if not d_sub_s.empty:
                subject_rows.append(d_sub_s.assign(subject=int(s)))

        if len(subject_rows) == 0:
            d_subject = pd.DataFrame(
                columns=[
                    "subject", "day", "lock_type", "condition", "channel", "time_s", "amplitude_v"
                ]
            )
        else:
            d_subject = pd.concat(subject_rows, ignore_index=True).sort_values(
                ["subject", "day", "channel", "time_s"]
            )
        d_subject.to_csv(d_subject_path, index=False)
        write_progress("completed", len(worker_items), len(worker_items))

    old_subject_path = output_dir / "erp_grand_average_subject_day_stim_all.csv"
    if old_subject_path.exists() and not d_subject_path.exists():
        pd.read_csv(old_subject_path).to_csv(d_subject_path, index=False)
    if not d_grand_path.exists() or not d_subject_path.exists():
        raise FileNotFoundError(
            f"Missing ERP output tables in {output_dir}. Run with run_compute=True first."
        )
    if not save_figures:
        return

    d_grand_plot = pd.read_csv(d_grand_path)
    d_subject_plot = pd.read_csv(d_subject_path)

    plot_day_grid(
        long_to_evoked_map(d_grand_plot, "stim", "all"),
        title="Grand Average ERP: stim_all",
        fig_name="erp_grand_average_stim_all.png",
    )
    plot_day_condition_grid(
        {
            (day, "correct"): ev
            for day, ev in long_to_evoked_map(d_grand_plot, "stim", "correct").items()
        }
        | {
            (day, "incorrect"): ev
            for day, ev in long_to_evoked_map(d_grand_plot, "stim", "incorrect").items()
        },
        title="Grand Average ERP: stim locked by feedback correctness",
        fig_name="erp_grand_average_stim_correct_vs_incorrect.png",
    )
    plot_day_condition_grid(
        {
            (day, "cat_a"): ev
            for day, ev in long_to_evoked_map(d_grand_plot, "stim", "cat_a").items()
        }
        | {
            (day, "cat_b"): ev
            for day, ev in long_to_evoked_map(d_grand_plot, "stim", "cat_b").items()
        },
        title="Grand Average ERP: stim locked by category",
        fig_name="erp_grand_average_stim_cat_a_vs_cat_b.png",
        conds=("cat_a", "cat_b"),
    )
    plot_stim_difference(
        d_grand_plot,
        conditions=("incorrect", "correct"),
        channels=["Fz", "FCz"],
        fig_name="erp_grand_average_stim_correctness_difference.png",
        title="Stim-Locked Correctness Difference at Fz/FCz",
        ylabel="Incorrect - correct amplitude (uV)",
    )
    plot_stim_difference(
        d_grand_plot,
        conditions=("cat_a", "cat_b"),
        channels=["Oz", "O1", "O2", "POz"],
        fig_name="erp_grand_average_stim_category_difference.png",
        title="Stim-Locked Category Difference at Posterior Channels",
        ylabel="Category A - category B amplitude (uV)",
    )
    plot_day_grid(
        long_to_evoked_map(d_grand_plot, "response", "all"),
        title="Grand Average ERP: response locked (time before response)",
        fig_name="erp_grand_average_response_all.png",
    )
    plot_day_condition_grid(
        {
            (day, "correct"): ev
            for day, ev in long_to_evoked_map(d_grand_plot, "response", "correct").items()
        }
        | {
            (day, "incorrect"): ev
            for day, ev in long_to_evoked_map(d_grand_plot, "response", "incorrect").items()
        },
        title="Grand Average ERP: response locked by feedback correctness",
        fig_name="erp_grand_average_response_correct_vs_incorrect.png",
    )
    plot_day_grid(
        long_to_evoked_map(d_grand_plot, "feedback", "all"),
        title="Grand Average ERP: feedback locked",
        fig_name="erp_grand_average_feedback_all.png",
    )
    plot_day_condition_grid(
        {
            (day, "correct"): ev
            for day, ev in long_to_evoked_map(d_grand_plot, "feedback", "correct").items()
        }
        | {
            (day, "incorrect"): ev
            for day, ev in long_to_evoked_map(d_grand_plot, "feedback", "incorrect").items()
        },
        title="Grand Average ERP: feedback locked by feedback correctness",
        fig_name="erp_grand_average_feedback_correct_vs_incorrect.png",
    )
    plot_feedback_frn_difference(
        d_grand_plot,
        fig_name="erp_grand_average_feedback_frn_difference.png",
    )

    for s in sorted(d_subject_plot["subject"].unique().astype(int)):
        d_sub = d_subject_plot[d_subject_plot["subject"] == s].copy()
        for lock_type, fig_prefix, title_lock in [
            ("stim", "erp_grand_average_stim_sub", "stim_all"),
            ("response", "erp_grand_average_response_sub", "response_all"),
            ("feedback", "erp_grand_average_feedback_sub", "feedback_all"),
        ]:
            evoked_sub = long_to_evoked_map(d_sub, lock_type, "all")
            days_sorted = sorted(evoked_sub.keys())
            if len(days_sorted) == 0:
                continue
            fig, axes = plt.subplots(
                1, len(days_sorted), figsize=(5 * len(days_sorted), 4), squeeze=False
            )
            for i, day in enumerate(days_sorted):
                ax = axes[0, i]
                evoked_sub[day].plot(
                    axes=ax, show=False, spatial_colors=True, titles=f"Day {day}"
                )
                ax.set_title(f"Day {day}")
            fig.suptitle(f"ERP: {title_lock} -- subject {s}")
            fig.savefig(figures_dir / f"{fig_prefix}_{s}.png")
            plt.close(fig)


if __name__ == "__main__":
    run_erp_grand_average()
