#!/usr/bin/env python3
"""Sensor-wide functional connectivity dynamics (16-channel abs ImCoh, carpet/graph/topomap)."""

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

from analysis_utils import parallel_collect
from load_project_data import align_behaviour_to_epochs, load_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8

CHANNEL_SUBSET = [
    "Fp1", "Fp2", "F7", "F8",
    "Fz", "C3", "Cz", "C4",
    "T7", "T8", "P3", "P4",
    "P7", "P8", "O1", "O2",
]

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
}

SNAPSHOT_TARGETS = np.array([0.10, 0.25, 0.40, 0.55, 0.68])
TOP_N_EDGES = 5


def compute_abs_imcoh(x, y):
    sxy = np.mean(x * np.conjugate(y))
    sxx = np.mean(np.abs(x) ** 2)
    syy = np.mean(np.abs(y) ** 2)
    denom = np.sqrt(sxx * syy)
    if (not np.isfinite(denom)) or (denom <= np.finfo(float).eps):
        return np.nan
    coh = sxy / denom
    return float(np.abs(np.imag(coh)))


def process_sensorwide_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    beh_df = task["beh"]
    epo_path = task["epo_path"]
    channel_subset = task["channel_subset"]
    bands = task["bands"]
    pair_idx = task["pair_idx"]
    window_sec = float(task["window_sec"])
    stim_tmin = float(task["stim_tmin"])
    stim_tmax = float(task["stim_tmax"])
    resp_tmin = float(task["resp_tmin"])
    resp_tmax = float(task["resp_tmax"])
    step_sec = float(task["step_sec"])

    try:
        epochs = mne.read_epochs(epo_path, preload=False, verbose="ERROR")
        epochs, beh_aligned = align_behaviour_to_epochs(
            beh_df, epochs, event_names=("Stim/A", "Stim/B")
        )
        epochs = epochs.load_data()
        if not all(ch in epochs.ch_names for ch in channel_subset):
            missing = [ch for ch in channel_subset if ch not in epochs.ch_names]
            return {
                "ok": False, "subject": subject, "day": day,
                "reason": "missing_channels", "detail": ",".join(missing),
            }
        epochs.pick(channel_subset)
        info = epochs.info.copy()
        rt_sec = beh_aligned["rt"].astype(float).to_numpy() / 1000.0
        times = epochs.times
        stim_starts = np.arange(stim_tmin, stim_tmax - window_sec + 1e-12, step_sec)
        resp_starts = np.arange(resp_tmin, resp_tmax - window_sec + 1e-12, step_sec)
        if len(stim_starts) == 0 and len(resp_starts) == 0:
            return {"ok": False, "subject": subject, "day": day, "reason": "no_windows", "detail": ""}

        agg = {}
        for band_name, (fmin, fmax) in bands.items():
            epochs_band = epochs.copy().filter(
                l_freq=fmin, h_freq=fmax, method="fir",
                fir_design="firwin", phase="zero-double", verbose="ERROR",
            )
            epochs_band = epochs_band.apply_hilbert(envelope=False, verbose="ERROR")
            data = epochs_band.get_data()
            n_trials = data.shape[0]
            if n_trials == 0:
                continue

            for t_start in stim_starts:
                t_end = t_start + window_sec
                i0 = int(np.searchsorted(times, t_start, side="left"))
                i1 = int(np.searchsorted(times, t_end, side="left"))
                if i1 - i0 < 2:
                    continue
                win = data[:, :, i0:i1]
                lock_time = float(t_start)
                for i, j in pair_idx:
                    val = compute_abs_imcoh(win[:, i, :].reshape(-1), win[:, j, :].reshape(-1))
                    if not np.isfinite(val):
                        continue
                    key = ("stim", int(day), band_name, lock_time, i, j)
                    if key not in agg:
                        agg[key] = [0.0, 0]
                    agg[key][0] += val
                    agg[key][1] += 1

            for tau_start in resp_starts:
                tau_end = tau_start + window_sec
                lock_time = float(tau_start)
                for i, j in pair_idx:
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
                        x_chunks.append(data[i_trial, i, i0:i1])
                        y_chunks.append(data[i_trial, j, i0:i1])
                    if not x_chunks:
                        continue
                    val = compute_abs_imcoh(np.concatenate(x_chunks), np.concatenate(y_chunks))
                    if not np.isfinite(val):
                        continue
                    key = ("response", int(day), band_name, lock_time, i, j)
                    if key not in agg:
                        agg[key] = [0.0, 0]
                    agg[key][0] += val
                    agg[key][1] += 1

        return {"ok": True, "subject": subject, "day": day, "agg": agg, "info": info}
    except Exception as exc:
        return {"ok": False, "subject": subject, "day": day, "reason": "compute_error", "detail": str(exc)}


def agg_to_edges_df(agg, channel_subset):
    agg_rows = []
    for lock_name, day, band_name, t, i, j in sorted(agg.keys()):
        s, c = agg[(lock_name, day, band_name, t, i, j)]
        agg_rows.append(
            {
                "lock_type": lock_name,
                "day": int(day),
                "band": band_name,
                "lock_time": float(t),
                "ch_i": channel_subset[i],
                "ch_j": channel_subset[j],
                "conn_val": float(s / c) if c > 0 else np.nan,
                "n_session_contrib": int(c),
            }
        )
    if not agg_rows:
        return pd.DataFrame(
            columns=["lock_type", "day", "band", "lock_time", "ch_i", "ch_j", "conn_val", "n_session_contrib"]
        )
    return pd.DataFrame(agg_rows).sort_values(["lock_type", "band", "day", "lock_time", "ch_i", "ch_j"])


def channel_xy(info, ch_names):
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


def plot_edge_time_carpet(day_data, pair_idx, lock_name, band_name, figures_dir):
    day_keys = sorted(day_data.keys())
    if not day_keys:
        return
    fig, axes = plt.subplots(len(day_keys), 1, squeeze=False, figsize=(12, 2.5 * len(day_keys)))
    for i_day, day in enumerate(day_keys):
        ax = axes[i_day, 0]
        times = day_data[day]["times"]
        mats = day_data[day]["mats"]
        if len(times) == 0 or len(mats) == 0:
            ax.set_title(f"Day {day} (no data)")
            continue
        edge_carpet = np.array([[mat[i, j] for mat in mats] for i, j in pair_idx], dtype=float)
        im = ax.imshow(
            edge_carpet, origin="lower", aspect="auto",
            extent=[times[0], times[-1], 0, len(pair_idx)], cmap="viridis",
        )
        ax.set_title(f"Day {day}")
        ax.set_ylabel("Edge index")
        ax.set_xlim(0.0, 0.80)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    axes[-1, 0].set_xlabel(
        "Time from stimulus onset (s)" if lock_name == "stim" else "Time before response (s)"
    )
    fig.suptitle(f"Edge-time carpet | {lock_name} | {band_name}", y=1.01)
    fig.tight_layout()
    fig_path = figures_dir / f"sensorwide_carpet_{lock_name}_{band_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_graph_snapshots(day_data, xy, pair_idx, lock_name, band_name, figures_dir,
                         snapshot_targets=SNAPSHOT_TARGETS, top_n_edges=TOP_N_EDGES):
    day_keys = sorted(day_data.keys())
    if not day_keys:
        return
    fig, axes = plt.subplots(
        len(day_keys), len(snapshot_targets), squeeze=False,
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
            top_order = np.argsort(upper_vals)[::-1]
            keep_pairs = set()
            for edge_idx in top_order:
                if len(keep_pairs) >= top_n_edges:
                    break
                if np.isfinite(upper_vals[edge_idx]):
                    keep_pairs.add(pair_idx[edge_idx])
            head = plt.Circle((0, 0), 1.0, fill=False, color="black", lw=1.0)
            ax.add_patch(head)
            for i, j in pair_idx:
                if np.isfinite(mat[i, j]) and (i, j) in keep_pairs:
                    ax.plot([xy[i, 0], xy[j, 0]], [xy[i, 1], xy[j, 1]],
                            color="#4c72b0", alpha=0.65, lw=1.2)
            ax.scatter(xy[:, 0], xy[:, 1], s=25, c="black")
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            ax.set_title(f"D{day} t={t_show:.2f}")
    fig.suptitle(f"Top-edge graph snapshots | {lock_name} | {band_name}", y=1.01)
    fig.tight_layout()
    fig_path = figures_dir / f"sensorwide_graphs_{lock_name}_{band_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_node_strength_topomaps(day_data, info_subset, lock_name, band_name, figures_dir,
                                snapshot_targets=SNAPSHOT_TARGETS):
    day_keys = sorted(day_data.keys())
    if not day_keys:
        return
    fig, axes = plt.subplots(
        len(day_keys), len(snapshot_targets), squeeze=False,
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
    fig_path = figures_dir / f"sensorwide_topomap_{lock_name}_{band_name}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_sensorwide_connectivity_analysis(
    channel_subset: list[str] = CHANNEL_SUBSET,
    bands: dict = BANDS,
    window_sec: float = 0.12,
    step_sec: float = 0.01,
    stim_tmin: float = 0.00,
    stim_tmax: float = 0.80,
    resp_tmin: float = 0.00,
    resp_tmax: float = 0.80,
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
    progress_json = output_dir / "connect_sensorwide_progress.json"
    edges_path = output_dir / "sensorwide_edge_timeseries.csv"
    node_path = output_dir / "sensorwide_node_strength_timeseries.csv"
    channels_path = output_dir / "sensorwide_channel_layout.csv"
    checkpoint_path = output_dir / "sensorwide_edge_timeseries_checkpoint.csv"

    n_channels = len(channel_subset)
    pair_idx = [(i, j) for i in range(n_channels) for j in range(i + 1, n_channels)]

    tasks = [
        {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "beh": item["beh"],
            "epo_path": str(item["epo_path"]),
            "channel_subset": channel_subset,
            "bands": bands,
            "pair_idx": pair_idx,
            "window_sec": window_sec,
            "step_sec": step_sec,
            "stim_tmin": stim_tmin,
            "stim_tmax": stim_tmax,
            "resp_tmin": resp_tmin,
            "resp_tmax": resp_tmax,
        }
        for item in sessions
    ]
    print(
        f"[connect_sensorwide] Running on {len(tasks)} sessions (n_workers={n_workers})...",
        flush=True,
    )
    results = parallel_collect(process_sensorwide_session, tasks, n_workers)

    agg = {}
    info_subset = None
    used = 0
    skipped = []
    for r in results:
        if not r["ok"]:
            skipped.append(r)
            continue
        used += 1
        if info_subset is None:
            info_subset = r["info"]
        for key, (val_sum, count) in r["agg"].items():
            if key not in agg:
                agg[key] = [0.0, 0]
            agg[key][0] += val_sum
            agg[key][1] += count

    progress_json.write_text(
        json.dumps(
            {
                "sessions": len(tasks),
                "used": used,
                "skipped": len(skipped),
                "elapsed_sec": float(time.time() - t0),
            },
            indent=2,
        )
    )
    if skipped:
        pd.DataFrame(skipped).to_csv(
            output_dir / "connect_sensorwide_qc_skipped.csv", index=False
        )

    if not agg or info_subset is None:
        print("[connect_sensorwide] No data computed.", flush=True)
        return {"edges_path": None, "node_path": None}

    d_edges = agg_to_edges_df(agg, channel_subset)
    d_edges.to_csv(edges_path, index=False)
    d_edges.to_csv(checkpoint_path, index=False)

    d_node_strength = (
        d_edges.melt(
            id_vars=["lock_type", "day", "band", "lock_time", "conn_val"],
            value_vars=["ch_i", "ch_j"],
            value_name="channel",
            var_name="channel_role",
        )
        .drop(columns=["channel_role"])
        .groupby(["lock_type", "day", "band", "lock_time", "channel"], as_index=False)["conn_val"]
        .sum()
        .rename(columns={"conn_val": "node_strength"})
        .sort_values(["lock_type", "band", "day", "lock_time", "channel"])
    )
    d_node_strength.to_csv(node_path, index=False)

    xy = channel_xy(info_subset, channel_subset)
    pd.DataFrame({"channel": channel_subset, "x": xy[:, 0], "y": xy[:, 1]}).to_csv(
        channels_path, index=False
    )

    d_channels_plot = pd.read_csv(channels_path)
    xy_plot = np.c_[
        d_channels_plot.set_index("channel").loc[channel_subset, "x"].to_numpy(),
        d_channels_plot.set_index("channel").loc[channel_subset, "y"].to_numpy(),
    ]
    ch_to_idx = {ch: i for i, ch in enumerate(channel_subset)}
    all_days = sorted(d_edges["day"].dropna().unique().astype(int).tolist())

    for lock_name in ("stim", "response"):
        for band_name in bands.keys():
            day_data = {}
            d_lb = d_edges[(d_edges["lock_type"] == lock_name) & (d_edges["band"] == band_name)]
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
                day_data[day] = {"times": np.array(times_this, dtype=float), "mats": mats}

            plot_edge_time_carpet(day_data, pair_idx, lock_name, band_name, figures_dir)
            plot_graph_snapshots(day_data, xy_plot, pair_idx, lock_name, band_name, figures_dir)
            info_plot = mne.create_info(ch_names=channel_subset, sfreq=256.0, ch_types="eeg")
            info_plot.set_montage(mne.channels.make_standard_montage("biosemi64"), on_missing="ignore")
            plot_node_strength_topomaps(day_data, info_plot, lock_name, band_name, figures_dir)

    print(
        f"[connect_sensorwide] Done. Used sessions: {used}, skipped: {len(skipped)}. "
        f"Figures: {figures_dir}",
        flush=True,
    )
    return {"edges_path": edges_path, "node_path": node_path, "channels_path": channels_path}


if __name__ == "__main__":
    run_sensorwide_connectivity_analysis()
