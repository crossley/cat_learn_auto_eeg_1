#!/usr/bin/env python3
"""Response-locked visual-motor functional connectivity (abs ImCoh)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import seaborn as sns

from analysis_utils import parallel_collect
from load_project_data import align_behaviour_to_epochs, load_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8

ROI_VISUAL = ["O1", "Oz", "O2"]
ROI_MOTOR = ["C3", "Cz", "C4"]

BANDS = {
    "broadband_0p5_40": (None, None),
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "low_beta": (12.0, 20.0),
    "high_beta": (20.0, 30.0),
    "beta": (12.0, 30.0),
}


def compute_abs_imcoh(x, y):
    sxy = np.mean(x * np.conjugate(y))
    sxx = np.mean(np.abs(x) ** 2)
    syy = np.mean(np.abs(y) ** 2)
    denom = np.sqrt(sxx * syy)
    if (not np.isfinite(denom)) or (denom <= np.finfo(float).eps):
        return np.nan
    coh = sxy / denom
    return float(np.abs(np.imag(coh)))


def process_response_locked_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    beh_df = task["beh"]
    epo_path = task["epo_path"]
    roi_visual = task["roi_visual"]
    roi_motor = task["roi_motor"]
    bands = task["bands"]
    window_sec = float(task["window_sec"])
    step_sec = float(task["step_sec"])
    tau_tmin = float(task["tau_tmin"])
    tau_tmax = float(task["tau_tmax"])

    try:
        epochs = mne.read_epochs(epo_path, preload=False, verbose="ERROR")
        epochs, beh_aligned = align_behaviour_to_epochs(
            beh_df, epochs, event_names=("Stim/A", "Stim/B")
        )
        epochs = epochs.load_data()
        epochs.pick(roi_visual + roi_motor)
        times = epochs.times
        rt_sec = beh_aligned["rt"].astype(float).to_numpy() / 1000.0

        vis_idx = [epochs.ch_names.index(ch) for ch in roi_visual]
        mot_idx = [epochs.ch_names.index(ch) for ch in roi_motor]
        indices = (np.repeat(vis_idx, len(mot_idx)), np.tile(mot_idx, len(vis_idx)))
        tau_starts = np.arange(tau_tmin, tau_tmax - window_sec + 1e-12, step_sec)
        if len(tau_starts) == 0:
            return {"ok": True, "rows": []}

        rows = []
        for band_name, (fmin, fmax) in bands.items():
            epochs_band = epochs.copy()
            if fmin is not None and fmax is not None:
                epochs_band = epochs_band.filter(
                    l_freq=fmin, h_freq=fmax, method="fir",
                    fir_design="firwin", phase="zero-double", verbose="ERROR",
                )
            epochs_band = epochs_band.apply_hilbert(envelope=False, verbose="ERROR")
            data = epochs_band.get_data()
            n_trials = data.shape[0]
            if n_trials == 0:
                continue
            for tau_start in tau_starts:
                tau_end = tau_start + window_sec
                pair_vals = []
                for i_vis, i_mot in zip(indices[0], indices[1]):
                    x_chunks = []
                    y_chunks = []
                    for i_trial in range(n_trials):
                        rt = rt_sec[i_trial]
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
                    val = compute_abs_imcoh(np.concatenate(x_chunks), np.concatenate(y_chunks))
                    if np.isfinite(val):
                        pair_vals.append(val)
                if pair_vals:
                    rows.append(
                        {
                            "subject": subject,
                            "day": day,
                            "band": band_name,
                            "lock_time": float(tau_start),
                            "conn_val": float(np.nanmean(pair_vals)),
                        }
                    )
        return {"ok": True, "rows": rows}
    except Exception as exc:
        return {"ok": False, "subject": subject, "day": day, "reason": "compute_error", "detail": str(exc)}


def run_response_locked_connectivity_analysis(
    roi_visual: list[str] = ROI_VISUAL,
    roi_motor: list[str] = ROI_MOTOR,
    bands: dict = BANDS,
    window_sec: float = 0.12,
    step_sec: float = 0.01,
    tau_tmin: float = 0.00,
    tau_tmax: float = 0.80,
    n_workers: int | None = None,
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    mne.set_log_level("ERROR")
    sessions = load_sessions()
    t0 = time.time()
    progress_json = output_dir / "connect_response_locked_progress.json"

    tasks = [
        {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "beh": item["beh"],
            "epo_path": str(item["epo_path"]),
            "roi_visual": roi_visual,
            "roi_motor": roi_motor,
            "bands": bands,
            "window_sec": window_sec,
            "step_sec": step_sec,
            "tau_tmin": tau_tmin,
            "tau_tmax": tau_tmax,
        }
        for item in sessions
    ]
    print(
        f"[connect_response_locked] Running on {len(tasks)} sessions (n_workers={n_workers})...",
        flush=True,
    )
    results = parallel_collect(process_response_locked_session, tasks, n_workers)

    all_rows = []
    skipped = []
    for r in results:
        if r["ok"]:
            all_rows.extend(r.get("rows", []))
        else:
            skipped.append(r)

    progress_json.write_text(
        json.dumps(
            {
                "sessions": len(tasks),
                "skipped": len(skipped),
                "elapsed_sec": float(time.time() - t0),
            },
            indent=2,
        )
    )
    if skipped:
        pd.DataFrame(skipped).to_csv(
            output_dir / "connect_response_locked_qc_skipped.csv", index=False
        )

    d = pd.DataFrame(all_rows)
    output_csv = output_dir / "connectivity_response_locked_subject_day.csv"
    d.to_csv(output_csv, index=False)

    if d.empty:
        print("[connect_response_locked] No data to plot.", flush=True)
        return {"output_csv": output_csv, "figure": None}

    d_subj = (
        d.groupby(["band", "day", "subject", "lock_time"], as_index=False)["conn_val"]
        .mean()
        .sort_values(["band", "day", "subject", "lock_time"])
    )
    d_peak = (
        d_subj.groupby(["band", "day", "subject"], as_index=False)["conn_val"]
        .max()
        .rename(columns={"conn_val": "peak_conn"})
    )
    d_subj = d_subj.merge(d_peak, on=["band", "day", "subject"], how="left")
    d_subj["peak_conn"] = d_subj["peak_conn"].where(
        d_subj["peak_conn"] > np.finfo(float).eps, np.nan
    )
    d_subj["conn_norm"] = d_subj["conn_val"] / d_subj["peak_conn"]
    d_time = (
        d_subj.groupby(["band", "day", "lock_time"], as_index=False)
        .agg(
            conn_mean=("conn_norm", "mean"),
            conn_sem=(
                "conn_norm",
                lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(np.sum(np.isfinite(x))))
                if np.sum(np.isfinite(x)) > 1
                else np.nan,
            ),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["band", "day", "lock_time"])
    )

    n_bands = len(bands)
    n_cols = 3
    n_rows = int(np.ceil(n_bands / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(5 * n_cols, 3.5 * n_rows))
    day_values = sorted(d_time["day"].dropna().unique().astype(int).tolist())
    palette = dict(zip(day_values, sns.color_palette("cividis", n_colors=len(day_values))))

    for i_band, band_name in enumerate(bands.keys()):
        r = i_band // n_cols
        c = i_band % n_cols
        ax = axes[r, c]
        d_band = d_time[d_time["band"] == band_name]
        if d_band.empty:
            ax.set_title(f"{band_name} (no data)")
            ax.set_xlabel("Time before response (s)")
            ax.set_ylabel("normalized abs(ImCoh)")
            continue
        for day in day_values:
            d_day = d_band[d_band["day"] == day].sort_values("lock_time")
            if d_day.empty:
                continue
            x = d_day["lock_time"].to_numpy(dtype=float)
            y = d_day["conn_mean"].to_numpy(dtype=float)
            sem = d_day["conn_sem"].to_numpy(dtype=float)
            color = palette[day]
            ax.plot(x, y, color=color, linewidth=2.0, label=f"Day {day}")
            ax.fill_between(x, y - sem, y + sem, color=color, alpha=0.18, linewidth=0)
        ax.set_xlim(tau_tmin, tau_tmax)
        ax.set_title(band_name)
        ax.set_ylabel("subject-normalized abs(ImCoh)")
        ax.set_xlabel("Time before response (s)")
        if i_band == 0:
            ax.legend(loc="best", title="day")

    for i_extra in range(n_bands, n_rows * n_cols):
        axes[i_extra // n_cols, i_extra % n_cols].axis("off")

    fig.suptitle("Response-locked abs(ImCoh), peak-normalized", y=1.02)
    fig.tight_layout()
    fig_path = figures_dir / "connectivity_response_locked.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"output_csv": output_csv, "figure": fig_path}


if __name__ == "__main__":
    run_response_locked_connectivity_analysis()
