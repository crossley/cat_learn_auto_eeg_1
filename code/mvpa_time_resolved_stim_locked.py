#!/usr/bin/env python3
"""Stimulus-locked time-resolved MVPA (Stim/A vs Stim/B)."""

from __future__ import annotations

import json
import os
import time
import warnings
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
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import fdrcorrection
from statsmodels.tools.sm_exceptions import ConvergenceWarning

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from load_project_data import load_sessions
from mvpa_tg_within_day import pick_eeg_interpolate_bads

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8


def build_clf(random_state: int):
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "logreg",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def vector_corr(x_vec, y_vec):
    valid = np.isfinite(x_vec) & np.isfinite(y_vec)
    if int(np.sum(valid)) < 3:
        return np.nan
    x_use = x_vec[valid] - np.nanmean(x_vec[valid])
    y_use = y_vec[valid] - np.nanmean(y_vec[valid])
    denom = np.sqrt(np.sum(x_use**2) * np.sum(y_use**2))
    if (not np.isfinite(denom)) or denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x_use * y_use) / denom)


def make_haufe_info_from_pos_df(pos_df):
    ch_names = pos_df["channel"].tolist()
    ch_pos = {
        r["channel"]: np.array([r["x"], r["y"], r["z"]], dtype=float)
        for _, r in pos_df.iterrows()
    }
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="ignore")
    return info, ch_names


def plot_peak_latency_trajectory(peak_df, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "mvpa_peak_latency_trajectory.png"

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True, squeeze=False)
    peak_order = [p for p in ["early", "late"] if p in set(peak_df["peak"])]
    if not peak_order:
        fig.text(0.5, 0.5, "No peak data available", ha="center", va="center")
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    for ax, peak_label in zip(axes.ravel(), peak_order):
        d_peak = peak_df[peak_df["peak"] == peak_label].copy()
        days = sorted(d_peak["day"].dropna().unique().astype(int).tolist())
        subj_days = (
            d_peak.groupby(["subject", "day"], as_index=False)["peak_time_sec"]
            .mean()
            .sort_values(["subject", "day"])
        )
        for _, d_sub in subj_days.groupby("subject"):
            ax.plot(
                d_sub["day"].to_numpy(dtype=float),
                d_sub["peak_time_sec"].to_numpy(dtype=float),
                color="#2f4f4f",
                alpha=0.28,
                linewidth=1.0,
            )

        day_mean = (
            d_peak.groupby("day", as_index=False)
            .agg(
                peak_time_sec_mean=("peak_time_sec", "mean"),
                peak_time_sec_sem=(
                    "peak_time_sec",
                    lambda x: (
                        float(np.std(x, ddof=1) / np.sqrt(len(x)))
                        if len(x) > 1
                        else np.nan
                    ),
                ),
                n_subjects=("subject", "nunique"),
            )
            .sort_values("day")
        )
        x = day_mean["day"].to_numpy(dtype=float)
        y = day_mean["peak_time_sec_mean"].to_numpy(dtype=float)
        yerr = day_mean["peak_time_sec_sem"].to_numpy(dtype=float)
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            color="tab:blue",
            marker="o",
            markersize=5,
            linewidth=2,
            capsize=3,
            label="Group mean \u00b1 SEM",
        )
        valid = np.isfinite(x) & np.isfinite(y)
        if int(np.sum(valid)) >= 2:
            coef = np.polyfit(x[valid], y[valid], 1)
            x_fit = np.linspace(
                float(np.nanmin(x[valid])),
                float(np.nanmax(x[valid])),
                100,
            )
            y_fit = np.polyval(coef, x_fit)
            ax.plot(
                x_fit,
                y_fit,
                color="tab:red",
                linestyle="--",
                linewidth=1.8,
                label="OLS trend",
            )
        ax.set_title(f"{peak_label.capitalize()} peak")
        ax.set_xlabel("Day")
        ax.set_xticks(days)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")

    axes.ravel()[0].set_ylabel("Peak latency (s)")
    for ax in axes.ravel()[len(peak_order) :]:
        ax.axis("off")
    fig.suptitle("Stimulus-Locked MVPA Peak Latency Trajectory")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_haufe_topo_at_peak(peak_df, haufe_day_mean_df, pos_df, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "mvpa_haufe_topo_at_peak.png"

    if peak_df.empty or haufe_day_mean_df.empty or pos_df.empty:
        fig = plt.figure(figsize=(10, 4))
        fig.text(0.5, 0.5, "No Haufe peak data available", ha="center", va="center")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    info, ch_names = make_haufe_info_from_pos_df(pos_df)
    day_order = [
        day for day in range(1, 6) if day in set(peak_df["day"].dropna().astype(int))
    ]
    peak_order = [p for p in ["early", "late"] if p in set(peak_df["peak"])]

    topo_items = []
    for peak_label in peak_order:
        for day in day_order:
            d_peak = peak_df[(peak_df["day"] == day) & (peak_df["peak"] == peak_label)]
            d_day = haufe_day_mean_df[haufe_day_mean_df["day"] == day].copy()
            if d_peak.empty or d_day.empty:
                topo_items.append(
                    {
                        "peak": peak_label,
                        "day": day,
                        "peak_time_sec": np.nan,
                        "used_time_sec": np.nan,
                        "values": None,
                    }
                )
                continue
            peak_time = float(d_peak["peak_time_sec"].median())
            times = np.sort(d_day["time_sec"].dropna().unique().astype(float))
            if len(times) == 0:
                topo_items.append(
                    {
                        "peak": peak_label,
                        "day": day,
                        "peak_time_sec": peak_time,
                        "used_time_sec": np.nan,
                        "values": None,
                    }
                )
                continue
            t_show = float(times[int(np.argmin(np.abs(times - peak_time)))])
            d_topo = d_day[np.isclose(d_day["time_sec"], t_show)]
            vals = (
                d_topo.set_index("channel")
                .reindex(ch_names)["pattern_mean"]
                .to_numpy(dtype=float)
            )
            topo_items.append(
                {
                    "peak": peak_label,
                    "day": day,
                    "peak_time_sec": peak_time,
                    "used_time_sec": t_show,
                    "values": vals,
                }
            )

    valid_vals = [item["values"] for item in topo_items if item["values"] is not None]
    lim = (
        float(np.nanmax(np.abs(np.vstack(valid_vals))))
        if valid_vals
        else 1e-12
    )
    if not np.isfinite(lim) or lim <= 0:
        lim = 1e-12

    fig, axes = plt.subplots(
        len(peak_order),
        len(day_order),
        figsize=(2.55 * len(day_order), 5.6),
        squeeze=False,
    )
    im_last = None
    for r, peak_label in enumerate(peak_order):
        for c, day in enumerate(day_order):
            ax = axes[r, c]
            item = next(
                x for x in topo_items if x["peak"] == peak_label and x["day"] == day
            )
            if item["values"] is None:
                ax.axis("off")
                continue
            im_last, _ = mne.viz.plot_topomap(
                item["values"],
                info,
                axes=ax,
                show=False,
                contours=0,
                cmap="RdBu_r",
                vlim=(-lim, lim),
                sphere=(0.0, 0.0, 0.0, 0.095),
            )
            ax.set_title(
                f"Day {day}\n{item['peak_time_sec'] * 1000.0:.0f} ms",
                fontsize=9,
            )
            if c == 0:
                ax.text(
                    -0.16,
                    0.5,
                    f"{peak_label.capitalize()} peak",
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="right",
                    fontsize=10,
                )

    fig.suptitle("Haufe Topographies at Subject-Median Peak Latency", y=0.98)
    cax = fig.add_axes([0.92, 0.18, 0.02, 0.64])
    if im_last is not None:
        fig.colorbar(im_last, cax=cax, label="Haufe pattern")
    else:
        cax.axis("off")
    fig.tight_layout(rect=[0, 0, 0.90, 0.95])
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_haufe_similarity_day_pairs(peak_df, haufe_day_mean_df, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "mvpa_haufe_similarity_timegen_matrices_5x5.png"

    if haufe_day_mean_df.empty:
        fig = plt.figure(figsize=(10, 4))
        fig.text(0.5, 0.5, "No Haufe similarity data available", ha="center", va="center")
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return fig_path

    day_grid = [1, 2, 3, 4, 5]
    channel_order = sorted(haufe_day_mean_df["channel"].dropna().unique().tolist())
    similarity_maps = {}
    time_maps = {}
    for day in day_grid:
        d_day = haufe_day_mean_df[haufe_day_mean_df["day"] == day].copy()
        if d_day.empty:
            continue
        times = np.sort(d_day["time_sec"].dropna().unique().astype(float))
        time_maps[day] = times
        vec_map = {}
        for t in times:
            d_time = d_day[np.isclose(d_day["time_sec"], t)]
            vec_map[float(t)] = (
                d_time.set_index("channel").reindex(channel_order)["pattern_mean"]
                .to_numpy(dtype=float)
            )
        similarity_maps[day] = vec_map

    peak_medians = {}
    if not peak_df.empty:
        peak_medians = {
            (int(r["day"]), str(r["peak"])): float(r["peak_time_sec"])
            for _, r in (
                peak_df.groupby(["day", "peak"], as_index=False)["peak_time_sec"]
                .median()
                .iterrows()
            )
        }

    fig, axes = plt.subplots(5, 5, figsize=(18.0, 16.0), squeeze=False)
    im = None
    for i, train_day in enumerate(day_grid):
        for j, test_day in enumerate(day_grid):
            ax = axes[i, j]
            train_times = time_maps.get(train_day)
            test_times = time_maps.get(test_day)
            if train_times is None or test_times is None:
                ax.axis("off")
                continue
            mat = np.full((len(train_times), len(test_times)), np.nan)
            train_vecs = similarity_maps[train_day]
            test_vecs = similarity_maps[test_day]
            for ti, t_train in enumerate(train_times):
                x_vec = train_vecs[float(t_train)]
                for tj, t_test in enumerate(test_times):
                    y_vec = test_vecs[float(t_test)]
                    mat[ti, tj] = vector_corr(x_vec, y_vec)
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                origin="lower",
                aspect="auto",
                extent=[
                    float(test_times.min()),
                    float(test_times.max()),
                    float(train_times.min()),
                    float(train_times.max()),
                ],
                vmin=-1.0,
                vmax=1.0,
                cmap="RdBu_r",
            )
            ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
            ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
            for peak_label, color in [("early", "#b22222"), ("late", "#ff7f0e")]:
                if (test_day, peak_label) in peak_medians:
                    ax.axvline(
                        peak_medians[(test_day, peak_label)],
                        color=color,
                        linestyle="--",
                        linewidth=0.9,
                    )
                if (train_day, peak_label) in peak_medians:
                    ax.axhline(
                        peak_medians[(train_day, peak_label)],
                        color=color,
                        linestyle="--",
                        linewidth=0.9,
                    )
            if i == 0:
                ax.set_title(f"Test D{test_day}", fontsize=9)
            if j == 0:
                ax.set_ylabel(f"Train D{train_day}", fontsize=9)
            if i == 4:
                ax.set_xlabel("Test Time (s)")
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

    fig.suptitle("Haufe Pattern Similarity by Day Pair (A/B)", y=0.98)
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.90, wspace=0.26, hspace=0.36)
    cax = fig.add_axes([0.92, 0.14, 0.015, 0.72])
    if im is not None:
        fig.colorbar(im, cax=cax, label="Pattern correlation")
    else:
        cax.axis("off")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def decode_timecourse(X, y, n_splits=5, random_state=42):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    clf = build_clf(random_state=random_state)
    n_times = X.shape[2]
    auc = np.full(n_times, np.nan, dtype=float)
    for ti in range(n_times):
        Xt = X[:, :, ti]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                scores = cross_val_score(clf, Xt, y, cv=cv, scoring="roc_auc")
            auc[ti] = float(np.mean(scores))
        except Exception:
            auc[ti] = np.nan
    return auc


def compute_haufe_patterns_from_xy(X, y, random_state: int):
    n_times = X.shape[2]
    n_ch = X.shape[1]
    patterns = np.full((n_ch, n_times), np.nan, dtype=float)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for ti in range(n_times):
        Xt = X[:, :, ti]
        fold_patterns = []
        for tr, _ in cv.split(Xt, y):
            Xt_tr = Xt[tr]
            y_tr = y[tr]
            clf = build_clf(random_state=random_state)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", SklearnConvergenceWarning)
                    clf.fit(Xt_tr, y_tr)
            except Exception:
                continue
            scaler = clf.named_steps["scaler"]
            logreg = clf.named_steps["logreg"]
            w_scaled = logreg.coef_.ravel().astype(float)
            scale = np.asarray(scaler.scale_, dtype=float)
            scale[scale == 0] = 1.0
            w_sensor = w_scaled / scale
            if Xt_tr.shape[0] < 2:
                continue
            cov_x = np.cov(Xt_tr, rowvar=False, ddof=1)
            fold_patterns.append(cov_x @ w_sensor)
        if fold_patterns:
            patterns[:, ti] = np.nanmean(np.vstack(fold_patterns), axis=0)
    return patterns


def process_stim_mvpa_session(task: dict):
    session_file = task["epo_file"]
    subject = int(task["subject"])
    day = int(task["day"])
    min_epochs = int(task["min_epochs"])
    random_state = int(task["random_state"])

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
        if len(stim_events) < 2:
            return {
                "ok": False,
                "qc": {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "event_select",
                    "reason": "missing_stim_labels",
                    "detail": ",".join(stim_events),
                },
            }
        stim_epochs = epochs[stim_events].copy()
        stim_epochs.load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        if len(stim_epochs.ch_names) == 0:
            raise RuntimeError("No EEG channels after interpolation.")
        stim_epochs.resample(128, npad="auto")
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "preprocess",
                "reason": "prep_error",
                "detail": str(exc),
            },
        }

    codes = stim_epochs.events[:, 2]
    y = np.full(len(codes), -1, dtype=int)
    y[codes == stim_epochs.event_id["Stim/A"]] = 0
    y[codes == stim_epochs.event_id["Stim/B"]] = 1
    keep = y >= 0
    y = y[keep]
    X = stim_epochs.get_data()[keep]

    n_a = int(np.sum(y == 0))
    n_b = int(np.sum(y == 1))
    n_trials = int(len(y))
    if n_trials < min_epochs:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "epoch_count",
                "reason": "insufficient_epochs",
                "detail": f"n_trials={n_trials} < min_epochs={min_epochs}",
            },
        }
    if min(n_a, n_b) < 5:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "class_balance",
                "reason": "insufficient_class_trials",
                "detail": f"n_a={n_a}, n_b={n_b}; need >=5 in each class",
            },
        }

    auc = decode_timecourse(X, y, n_splits=5, random_state=random_state)
    times = stim_epochs.times.copy()
    session_rows = [
        {
            "session_file": session_file,
            "subject": subject,
            "day": day,
            "time_sec": float(times[ti]),
            "auc": float(auc_val),
            "n_trials": n_trials,
            "n_a": n_a,
            "n_b": n_b,
        }
        for ti, auc_val in enumerate(auc)
    ]

    haufe_rows = []
    channel_pos = []
    haufe_qc = None
    try:
        patterns = compute_haufe_patterns_from_xy(X, y, random_state=random_state)
        for ci, ch in enumerate(stim_epochs.ch_names):
            loc = stim_epochs.info["chs"][stim_epochs.info.ch_names.index(ch)]["loc"][:3]
            channel_pos.append({"channel": ch, "x": float(loc[0]), "y": float(loc[1]), "z": float(loc[2])})
            for ti, tsec in enumerate(times):
                val = float(patterns[ci, ti])
                haufe_rows.append(
                    {
                        "subject": subject,
                        "day": day,
                        "session_file": session_file,
                        "channel": ch,
                        "time_sec": float(tsec),
                        "pattern": val,
                        "abs_pattern": float(np.abs(val)),
                        "n_trials": n_trials,
                        "n_a": n_a,
                        "n_b": n_b,
                    }
                )
    except Exception as exc:
        haufe_qc = {
            "session_file": session_file,
            "subject": subject,
            "day": day,
            "stage": "haufe",
            "reason": "compute_error",
            "detail": str(exc),
        }

    return {
        "ok": True,
        "session_rows": session_rows,
        "haufe_rows": haufe_rows,
        "channel_pos": channel_pos,
        "haufe_qc": haufe_qc,
    }


def run_stimulus_locked_mvpa_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    min_epochs: int = 20,
    random_state: int = 42,
    save_figures: bool = True,
    n_workers: int | None = None,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    warnings.filterwarnings(
        "ignore",
        message=".*'penalty' was deprecated.*",
        category=FutureWarning,
        module=r"sklearn\.linear_model\._logistic",
    )

    session_csv = output_dir / "mvpa_session_timecourse.csv"
    subject_day_csv = output_dir / "mvpa_subject_day_timecourse.csv"
    day_means_csv = output_dir / "mvpa_day_means_timecourse.csv"
    day_effect_csv = output_dir / "mvpa_day_effect_per_time.csv"
    qc_csv = output_dir / "mvpa_qc_log.csv"
    haufe_session_csv = output_dir / "mvpa_haufe_session_channel_time.csv"
    haufe_day_mean_csv = output_dir / "mvpa_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_haufe_channel_positions.csv"
    progress_json = output_dir / "mvpa_progress.json"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    session_rows = []
    haufe_rows = []
    haufe_channel_pos = {}
    t0 = time.time()

    def write_progress(stage: str, done: int, total: int):
        payload = {
            "stage": stage,
            "done": int(done),
            "total": int(total),
            "elapsed_sec": float(time.time() - t0),
            "updated_at_unix": float(time.time()),
        }
        progress_json.write_text(json.dumps(payload, indent=2))

    sessions = load_sessions()
    tasks = [
        {
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "epo_file": item["epo_file"],
            "epo_path": str(item["epo_path"]),
            "min_epochs": int(min_epochs),
            "random_state": int(random_state),
        }
        for item in sessions
    ]
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    write_progress("running", 0, len(tasks))

    def handle_result(result, done):
        if not result["ok"]:
            qc_rows.append(result["qc"])
        else:
            session_rows.extend(result["session_rows"])
            haufe_rows.extend(result.get("haufe_rows", []))
            if result.get("haufe_qc") is not None:
                qc_rows.append(result["haufe_qc"])
            for pos_row in result.get("channel_pos", []):
                ch = pos_row["channel"]
                if ch not in haufe_channel_pos:
                    haufe_channel_pos[ch] = np.array(
                        [pos_row["x"], pos_row["y"], pos_row["z"]], dtype=float
                    )
        write_progress("running", done, len(tasks))
        if (done % 5) == 0:
            elapsed = time.time() - t0
            print(
                f"[MVPA stimulus] complete {done}/{len(tasks)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    print(
        f"[MVPA stimulus] Starting time-resolved MVPA on {len(tasks)} sessions "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    if n_workers == 1:
        for done, task in enumerate(tasks, start=1):
            handle_result(process_stim_mvpa_session(task), done)
    elif threadpool_limits is None:
        result_iter = Parallel(
            n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
        )(delayed(process_stim_mvpa_session)(task) for task in tasks)
        for done, result in enumerate(result_iter, start=1):
            handle_result(result, done)
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(
                n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
            )(delayed(process_stim_mvpa_session)(task) for task in tasks)
            for done, result in enumerate(result_iter, start=1):
                handle_result(result, done)

    session_df = pd.DataFrame(session_rows)
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)

    if session_df.empty:
        session_df.to_csv(session_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        pd.DataFrame().to_csv(subject_day_csv, index=False)
        pd.DataFrame().to_csv(day_means_csv, index=False)
        pd.DataFrame().to_csv(day_effect_csv, index=False)
        raise RuntimeError("MVPA stage produced no valid session rows.")

    subject_day_df = (
        session_df.groupby(["subject", "day", "time_sec"], as_index=False)["auc"]
        .mean()
        .sort_values(["subject", "day", "time_sec"])
    )
    day_means_df = (
        subject_day_df.groupby(["day", "time_sec"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sem=(
                "auc",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            ),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "time_sec"])
    )

    effect_rows = []
    for t, g in subject_day_df.groupby("time_sec"):
        if g["subject"].nunique() < 2 or g["day"].nunique() < 2:
            effect_rows.append(
                {
                    "time_sec": float(t),
                    "n_rows": int(len(g)),
                    "n_subjects": int(g["subject"].nunique()),
                    "day_coef": np.nan,
                    "day_se": np.nan,
                    "day_pvalue": np.nan,
                    "status": "insufficient_data",
                    "detail": "Need >=2 subjects and >=2 day values",
                }
            )
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            warnings.simplefilter("ignore", UserWarning)
            try:
                model = smf.mixedlm("auc ~ day", data=g, groups=g["subject"]).fit(
                    reml=False, method="lbfgs", disp=False
                )
                effect_rows.append(
                    {
                        "time_sec": float(t),
                        "n_rows": int(len(g)),
                        "n_subjects": int(g["subject"].nunique()),
                        "day_coef": float(model.params["day"]),
                        "day_se": float(model.bse["day"]),
                        "day_pvalue": float(model.pvalues["day"]),
                        "status": "ok",
                        "detail": "",
                    }
                )
            except Exception as exc:
                try:
                    model = smf.ols("auc ~ day", data=g).fit()
                    effect_rows.append(
                        {
                            "time_sec": float(t),
                            "n_rows": int(len(g)),
                            "n_subjects": int(g["subject"].nunique()),
                            "day_coef": float(model.params["day"]),
                            "day_se": float(model.bse["day"]),
                            "day_pvalue": float(model.pvalues["day"]),
                            "status": "ols_fallback",
                            "detail": str(exc),
                        }
                    )
                except Exception as exc2:
                    effect_rows.append(
                        {
                            "time_sec": float(t),
                            "n_rows": int(len(g)),
                            "n_subjects": int(g["subject"].nunique()),
                            "day_coef": np.nan,
                            "day_se": np.nan,
                            "day_pvalue": np.nan,
                            "status": "fit_error",
                            "detail": f"{exc}; fallback={exc2}",
                        }
                    )

    day_effect_df = pd.DataFrame(effect_rows).sort_values("time_sec")
    day_effect_df["p_fdr"] = np.nan
    day_effect_df["significant_fdr"] = False
    valid = day_effect_df["day_pvalue"].notna()
    if valid.any():
        rej, p_corr = fdrcorrection(day_effect_df.loc[valid, "day_pvalue"].values, alpha=0.05)
        day_effect_df.loc[valid, "p_fdr"] = p_corr
        day_effect_df.loc[valid, "significant_fdr"] = rej

    session_df.to_csv(session_csv, index=False)
    subject_day_df.to_csv(subject_day_csv, index=False)
    day_means_df.to_csv(day_means_csv, index=False)
    day_effect_df.to_csv(day_effect_csv, index=False)

    haufe_session_df = pd.DataFrame(haufe_rows)
    if haufe_session_df.empty:
        haufe_session_df.to_csv(haufe_session_csv, index=False)
        pd.DataFrame().to_csv(haufe_day_mean_csv, index=False)
        pd.DataFrame().to_csv(haufe_channel_pos_csv, index=False)
    else:
        haufe_day_mean_df = (
            haufe_session_df.groupby(["day", "channel", "time_sec"], as_index=False)
            .agg(
                pattern_mean=("pattern", "mean"),
                pattern_sem=(
                    "pattern",
                    lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
                ),
                abs_pattern_mean=("abs_pattern", "mean"),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["day", "channel", "time_sec"])
        )
        haufe_pos_df = pd.DataFrame(
            [
                {"channel": ch, "x": xyz[0], "y": xyz[1], "z": xyz[2]}
                for ch, xyz in sorted(haufe_channel_pos.items())
            ]
        )
        haufe_session_df.to_csv(haufe_session_csv, index=False)
        haufe_day_mean_df.to_csv(haufe_day_mean_csv, index=False)
        haufe_pos_df.to_csv(haufe_channel_pos_csv, index=False)

    qc_df.to_csv(qc_csv, index=False)
    write_progress("completed", len(tasks), len(tasks))

    if save_figures:
        save_fig_mvpa_time_resolved(output_dir=output_dir, figures_dir=figures_dir)

    return {
        "session_df": session_df,
        "subject_day_df": subject_day_df,
        "day_means_df": day_means_df,
        "day_effect_df": day_effect_df,
        "qc_df": qc_df,
        "time_sec": np.array(sorted(session_df["time_sec"].unique())),
        "session_csv": session_csv,
        "subject_day_csv": subject_day_csv,
        "day_means_csv": day_means_csv,
        "day_effect_csv": day_effect_csv,
        "qc_csv": qc_csv,
    }


def save_fig_mvpa_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    session_csv = output_dir / "mvpa_session_timecourse.csv"
    day_means_csv = output_dir / "mvpa_day_means_timecourse.csv"
    day_effect_csv = output_dir / "mvpa_day_effect_per_time.csv"
    haufe_day_mean_csv = output_dir / "mvpa_haufe_day_mean_channel_time.csv"
    haufe_session_csv = output_dir / "mvpa_haufe_session_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_haufe_channel_positions.csv"
    haufe_peak_times_csv = output_dir / "mvpa_haufe_subject_day_peak_times.csv"
    haufe_stability_subject_csv = output_dir / "mvpa_haufe_peak_stability_subject.csv"
    haufe_stability_summary_csv = output_dir / "mvpa_haufe_peak_stability_summary.csv"
    fig_day_panels = figures_dir / "mvpa_auc_by_day_panels.png"
    fig_day_slope = figures_dir / "mvpa_day_slope_timecourse.png"
    fig_haufe_stability = figures_dir / "mvpa_haufe_peak_stability.png"
    fig_haufe_similarity = figures_dir / "mvpa_haufe_similarity_timegen_matrices_5x5.png"

    if (not day_means_csv.exists()) or (not day_effect_csv.exists()):
        raise FileNotFoundError(
            "Missing MVPA outputs in "
            f"{output_dir}. Run run_stimulus_locked_mvpa_analysis() first."
        )

    day_means_df = pd.read_csv(day_means_csv)
    day_effect_df = pd.read_csv(day_effect_csv)

    def detect_subject_day_peak_times():
        if not session_csv.exists():
            return pd.DataFrame()
        d_session = pd.read_csv(session_csv)
        if d_session.empty:
            return pd.DataFrame()
        rows = []
        for (subject, day), d_sd in d_session.groupby(["subject", "day"]):
            d_sd = d_sd.sort_values("time_sec")
            for lo, hi, label in [(0.0, 0.20, "early"), (0.35, 0.80, "late")]:
                d_win = d_sd[(d_sd["time_sec"] >= lo) & (d_sd["time_sec"] <= hi)]
                if d_win.empty:
                    continue
                row = d_win.loc[d_win["auc"].idxmax()]
                rows.append(
                    {
                        "subject": int(subject),
                        "day": int(day),
                        "peak": label,
                        "peak_time_sec": float(row["time_sec"]),
                        "peak_auc": float(row["auc"]),
                        "window_start_sec": float(lo),
                        "window_end_sec": float(hi),
                    }
                )
        peak_df = pd.DataFrame(rows).sort_values(["day", "peak", "subject"])
        peak_df.to_csv(haufe_peak_times_csv, index=False)
        return peak_df

    def write_haufe_stability(peak_df, haufe_ch_names):
        if (not haufe_session_csv.exists()) or peak_df.empty:
            return pd.DataFrame(), pd.DataFrame()
        d_sub = pd.read_csv(haufe_session_csv)
        if d_sub.empty:
            return pd.DataFrame(), pd.DataFrame()
        rows = []
        for day in sorted(peak_df["day"].dropna().unique().astype(int)):
            d_day = d_sub[d_sub["day"] == day].copy()
            times = np.sort(d_day["time_sec"].dropna().unique().astype(float))
            if len(times) == 0:
                continue
            for peak_label in ["early", "late"]:
                d_peak_meta = peak_df[
                    (peak_df["day"] == day) & (peak_df["peak"] == peak_label)
                ]
                if d_peak_meta.empty:
                    continue
                vec_rows = []
                for _, peak_row in d_peak_meta.iterrows():
                    subject = int(peak_row["subject"])
                    peak_time = float(peak_row["peak_time_sec"])
                    t_show = float(times[int(np.argmin(np.abs(times - peak_time)))])
                    d_peak = d_day[
                        (d_day["subject"] == subject)
                        & np.isclose(d_day["time_sec"], t_show)
                    ].copy()
                    if d_peak.empty:
                        continue
                    vals = (
                        d_peak.set_index("channel")
                        .reindex(haufe_ch_names)["pattern"]
                        .to_numpy(dtype=float)
                    )
                    vec_rows.append(
                        {
                            "subject": subject,
                            "peak_time_sec": peak_time,
                            "used_time_sec": t_show,
                            "values": vals,
                        }
                    )
                if len(vec_rows) < 3:
                    continue
                mat = pd.DataFrame(
                    [r["values"] for r in vec_rows],
                    index=[r["subject"] for r in vec_rows],
                    columns=haufe_ch_names,
                )
                if mat.shape[0] < 3:
                    continue
                values = mat.to_numpy(dtype=float)
                subjects = mat.index.to_numpy()
                for i_sub, subject in enumerate(subjects):
                    x_vec = values[i_sub, :]
                    loo = np.delete(values, i_sub, axis=0)
                    with np.errstate(invalid="ignore"):
                        y_vec = np.nanmean(loo, axis=0)
                    rows.append(
                        {
                            "subject": int(subject),
                            "day": int(day),
                            "peak": peak_label,
                            "subject_peak_time_sec": float(
                                vec_rows[i_sub]["peak_time_sec"]
                            ),
                            "used_time_sec": float(vec_rows[i_sub]["used_time_sec"]),
                            "loo_pattern_r": vector_corr(x_vec, y_vec),
                            "n_channels": int(
                                np.sum(np.isfinite(x_vec) & np.isfinite(y_vec))
                            ),
                        }
                    )
        subject_df = pd.DataFrame(rows)
        if subject_df.empty:
            subject_df.to_csv(haufe_stability_subject_csv, index=False)
            pd.DataFrame().to_csv(haufe_stability_summary_csv, index=False)
            return subject_df, pd.DataFrame()
        summary_df = (
            subject_df.groupby(["day", "peak"], as_index=False)
            .agg(
                median_peak_time_sec=("subject_peak_time_sec", "median"),
                q25_peak_time_sec=(
                    "subject_peak_time_sec",
                    lambda x: float(np.nanpercentile(x, 25)),
                ),
                q75_peak_time_sec=(
                    "subject_peak_time_sec",
                    lambda x: float(np.nanpercentile(x, 75)),
                ),
                median_used_time_sec=("used_time_sec", "median"),
                median_r=("loo_pattern_r", "median"),
                q25_r=("loo_pattern_r", lambda x: float(np.nanpercentile(x, 25))),
                q75_r=("loo_pattern_r", lambda x: float(np.nanpercentile(x, 75))),
                mean_r=("loo_pattern_r", "mean"),
                prop_positive=(
                    "loo_pattern_r",
                    lambda x: float(np.nanmean(np.asarray(x) > 0)),
                ),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["day", "peak"])
        )
        subject_df.to_csv(haufe_stability_subject_csv, index=False)
        summary_df.to_csv(haufe_stability_summary_csv, index=False)
        return subject_df, summary_df

    def plot_haufe_stability(subject_df):
        if subject_df.empty:
            return
        peak_order = [p for p in ["early", "late"] if p in set(subject_df["peak"])]
        days_plot = sorted(subject_df["day"].dropna().unique().astype(int).tolist())
        fig, axes = plt.subplots(
            1, len(peak_order), figsize=(4.5 * len(peak_order), 4), sharey=True, squeeze=False
        )
        rng = np.random.default_rng(42)
        for ax, peak in zip(axes.ravel(), peak_order):
            data = []
            labels = []
            for day in days_plot:
                vals = (
                    subject_df[(subject_df["day"] == day) & (subject_df["peak"] == peak)][
                        "loo_pattern_r"
                    ]
                    .dropna()
                    .to_numpy(dtype=float)
                )
                data.append(vals)
                labels.append(f"D{day}")
            ax.boxplot(data, labels=labels, showfliers=False)
            for i_day, vals in enumerate(data, start=1):
                if len(vals) == 0:
                    continue
                jitter = rng.normal(0.0, 0.035, size=len(vals))
                ax.scatter(
                    np.full(len(vals), i_day) + jitter,
                    vals,
                    s=18,
                    alpha=0.55,
                    color="#2f4f4f",
                )
            ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
            ax.set_title(f"{peak} peak")
            ax.set_xlabel("Day")
            ax.grid(axis="y", alpha=0.25)
        axes.ravel()[0].set_ylabel("Subject vs leave-one-subject-out group pattern r")
        fig.suptitle("Haufe Pattern Stability at MVPA Peaks")
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fig.savefig(fig_haufe_stability, dpi=150, bbox_inches="tight")
        plt.close(fig)

    haufe_df = pd.DataFrame()
    haufe_info = None
    haufe_ch_names = []
    peak_df = pd.DataFrame()
    peak_medians = {}
    pos_df = pd.DataFrame()
    if haufe_day_mean_csv.exists() and haufe_channel_pos_csv.exists():
        haufe_df = pd.read_csv(haufe_day_mean_csv)
        pos_df = pd.read_csv(haufe_channel_pos_csv)
        if (not haufe_df.empty) and (not pos_df.empty):
            haufe_info, haufe_ch_names = make_haufe_info_from_pos_df(pos_df)
            peak_df = detect_subject_day_peak_times()
            if not peak_df.empty:
                d_peak_median = (
                    peak_df.groupby(["day", "peak"], as_index=False)["peak_time_sec"]
                    .median()
                    .rename(columns={"peak_time_sec": "median_peak_time_sec"})
                )
                peak_medians = {
                    (int(r["day"]), str(r["peak"])): float(r["median_peak_time_sec"])
                    for _, r in d_peak_median.iterrows()
                }

    if (not peak_df.empty) and haufe_ch_names:
        stability_df, _ = write_haufe_stability(peak_df, haufe_ch_names)
        plot_haufe_stability(stability_df)
    peak_latency_path = None
    if not peak_df.empty:
        peak_latency_path = plot_peak_latency_trajectory(peak_df, figures_dir)
    haufe_topo_peak_path = None
    if (not peak_df.empty) and (not haufe_df.empty) and (not pos_df.empty):
        haufe_topo_peak_path = plot_haufe_topo_at_peak(
            peak_df, haufe_df, pos_df, figures_dir
        )
    haufe_similarity_path = None
    if not haufe_df.empty:
        haufe_similarity_path = plot_haufe_similarity_day_pairs(
            peak_df, haufe_df, figures_dir
        )

    days = sorted(day_means_df["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 5.2), sharey=True, squeeze=False)
    x_all = day_means_df["time_sec"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_all))
    x_max = float(np.nanmax(x_all))
    y_upper = float(np.nanmax(day_means_df["auc_mean"] + day_means_df["auc_sem"].fillna(0.0)))
    y_lower = float(np.nanmin(day_means_df["auc_mean"] - day_means_df["auc_sem"].fillna(0.0)))
    y_pad = max(0.02, 0.20 * (y_upper - y_lower))
    topomap_ims = []
    if not haufe_df.empty:
        lim = float(np.nanmax(np.abs(haufe_df["pattern_mean"].to_numpy(dtype=float))))
        if not np.isfinite(lim) or lim <= 0:
            lim = 1e-12
    else:
        lim = 1e-12
    for ax, day in zip(axes.ravel(), days):
        g = day_means_df[day_means_df["day"] == day].sort_values("time_sec")
        x = g["time_sec"].to_numpy()
        y = g["auc_mean"].to_numpy()
        s = g["auc_sem"].to_numpy()
        ax.plot(x, y, color="tab:blue", linewidth=2)
        ax.fill_between(x, y - s, y + s, color="tab:blue", alpha=0.2, linewidth=0)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(y_lower - 0.02, y_upper + y_pad)
        ax.grid(alpha=0.25)
        day_peak_times = [
            (peak_medians[(int(day), peak_label)], peak_label)
            for peak_label in ["early", "late"]
            if (int(day), peak_label) in peak_medians
        ]
        for peak_time, peak_label in day_peak_times:
            if x_max <= x_min:
                continue
            y_peak = float(np.interp(peak_time, x, y))
            ax.axvline(peak_time, color="#b22222", linestyle=":", linewidth=1.2)
            ax.scatter(
                [peak_time], [y_peak], s=36, facecolor="white",
                edgecolor="#b22222", linewidth=1.2, zorder=4,
            )
            ax.text(
                peak_time, y_upper + (0.05 * y_pad), peak_label,
                color="#b22222", fontsize=8, ha="center", va="bottom",
            )
            x_frac = (peak_time - x_min) / (x_max - x_min)
            width = 0.18
            inset = ax.inset_axes(
                [max(0.01, min(0.99 - width, x_frac - width / 2.0)), 1.04, width, 0.36],
                transform=ax.transAxes,
            )
            if haufe_df.empty or haufe_info is None:
                inset.axis("off")
                continue
            d_day = haufe_df[haufe_df["day"] == day]
            if d_day.empty:
                inset.axis("off")
                continue
            times = np.sort(d_day["time_sec"].unique().astype(float))
            t_show = float(times[int(np.argmin(np.abs(times - peak_time)))])
            d_topo = d_day[np.isclose(d_day["time_sec"], t_show)]
            vals = (
                d_topo.set_index("channel")
                .reindex(haufe_ch_names)["pattern_mean"]
                .to_numpy(dtype=float)
            )
            im, _ = mne.viz.plot_topomap(
                vals, haufe_info, axes=inset, show=False, contours=0,
                cmap="RdBu_r", vlim=(-lim, lim), sphere=(0.0, 0.0, 0.0, 0.095),
            )
            topomap_ims.append(im)
            inset.set_title(
                f"{peak_label}\nmedian {peak_time:.3f}s\nmap {t_show:.3f}s", fontsize=7
            )
    axes.ravel()[0].set_ylabel("ROC-AUC")
    fig.suptitle("Time-resolved Category Decoding (Stim/A vs Stim/B)")
    if topomap_ims:
        cax = fig.add_axes([0.32, 0.89, 0.36, 0.025])
        fig.colorbar(topomap_ims[-1], cax=cax, orientation="horizontal", label="Haufe pattern")
        fig.tight_layout(rect=[0, 0, 1, 0.78])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_day_panels, dpi=150, bbox_inches="tight")
    plt.close(fig)

    g = day_effect_df.sort_values("time_sec")
    x = g["time_sec"].to_numpy()
    y = g["day_coef"].to_numpy()
    sig = g["significant_fdr"].to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, y, color="tab:green", linewidth=2, label="Day slope (AUC ~ day)")
    ax.axhline(0.0, color="k", linestyle="--", linewidth=1)
    ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
    if np.any(sig):
        ax.scatter(x[sig], y[sig], color="red", s=16, label="FDR < 0.05", zorder=3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Day coefficient")
    ax.set_title("Day Effect on Decoding Over Time")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(fig_day_slope, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_paths": {
            "day_panels": fig_day_panels,
            "day_slope": fig_day_slope,
            "haufe_stability": fig_haufe_stability,
            "haufe_similarity": fig_haufe_similarity,
            "peak_latency_trajectory": peak_latency_path,
            "haufe_topo_at_peak": haufe_topo_peak_path,
            "haufe_similarity_day_pairs": haufe_similarity_path,
        },
        "haufe_stability_subject_csv": haufe_stability_subject_csv,
        "haufe_stability_summary_csv": haufe_stability_summary_csv,
    }


def save_movie_mvpa_haufe_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    tmin: float = 0.0,
    tmax: float = 0.80,
    fps: int = 8,
    frame_step: int = 1,
    dpi: int = 120,
    movie_name: str = "mvpa_haufe_timecourse.mp4",
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    movie_path = figures_dir / movie_name
    gif_path = figures_dir / "mvpa_haufe_timecourse.gif"

    day_means_csv = output_dir / "mvpa_day_means_timecourse.csv"
    haufe_day_mean_csv = output_dir / "mvpa_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_haufe_channel_positions.csv"
    required = [day_means_csv, haufe_day_mean_csv, haufe_channel_pos_csv]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing inputs for MVPA/Haufe movie: {missing}")

    day_means_df = pd.read_csv(day_means_csv)
    haufe_df = pd.read_csv(haufe_day_mean_csv)
    pos_df = pd.read_csv(haufe_channel_pos_csv)
    if day_means_df.empty or haufe_df.empty or pos_df.empty:
        raise RuntimeError("MVPA/Haufe movie inputs are empty.")

    ch_names = pos_df["channel"].tolist()
    ch_pos = {
        r["channel"]: np.array([r["x"], r["y"], r["z"]], dtype=float)
        for _, r in pos_df.iterrows()
    }
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="ignore")

    all_times = np.array(
        sorted(haufe_df["time_sec"].dropna().unique().astype(float)), dtype=float
    )
    frame_times = all_times[(all_times >= tmin) & (all_times <= tmax)]
    frame_times = frame_times[:: max(1, frame_step)]
    if len(frame_times) == 0:
        raise RuntimeError(f"No Haufe time points found in requested range {tmin}-{tmax}s.")

    days = sorted(day_means_df["day"].dropna().unique().astype(int).tolist())
    y_upper = float(np.nanmax(day_means_df["auc_mean"] + day_means_df["auc_sem"].fillna(0.0)))
    y_lower = float(np.nanmin(day_means_df["auc_mean"] - day_means_df["auc_sem"].fillna(0.0)))
    y_pad = max(0.02, 0.08 * (y_upper - y_lower))
    lim = float(np.nanmax(np.abs(haufe_df["pattern_mean"].to_numpy(dtype=float))))
    if not np.isfinite(lim) or lim <= 0:
        lim = 1e-12

    from matplotlib import animation
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig = plt.figure(figsize=(4.2 * len(days), 7.0))
    gs = fig.add_gridspec(2, len(days), height_ratios=[1.0, 1.35], hspace=0.28, wspace=0.25)
    topo_axes = [fig.add_subplot(gs[0, i]) for i in range(len(days))]
    auc_axes = [fig.add_subplot(gs[1, i]) for i in range(len(days))]
    line_artists = []
    point_artists = []

    for ax, day in zip(auc_axes, days):
        g = day_means_df[day_means_df["day"] == day].sort_values("time_sec")
        x = g["time_sec"].to_numpy(dtype=float)
        y = g["auc_mean"].to_numpy(dtype=float)
        s = g["auc_sem"].to_numpy(dtype=float)
        ax.plot(x, y, color="tab:blue", linewidth=2)
        ax.fill_between(x, y - s, y + s, color="tab:blue", alpha=0.2, linewidth=0)
        ax.axhline(0.5, color="k", linestyle="--", linewidth=1)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1)
        line = ax.axvline(frame_times[0], color="#b22222", linestyle="--", linewidth=1.8)
        y_now = float(np.interp(frame_times[0], x, y))
        point = ax.scatter(
            [frame_times[0]], [y_now], s=40, facecolor="white",
            edgecolor="#b22222", linewidth=1.3, zorder=4,
        )
        line_artists.append((line, x, y))
        point_artists.append(point)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Time (s)")
        ax.set_ylim(y_lower - 0.02, y_upper + y_pad)
        ax.grid(alpha=0.25)
    auc_axes[0].set_ylabel("ROC-AUC")

    cax = fig.add_axes([0.35, 0.93, 0.30, 0.025])
    sm = ScalarMappable(norm=Normalize(vmin=-lim, vmax=lim), cmap="RdBu_r")
    sm.set_array([])
    fig.colorbar(sm, cax=cax, orientation="horizontal", label="Haufe pattern")
    time_text = fig.text(0.5, 0.985, "", ha="center", va="top", fontsize=13)

    def draw_frame(frame_i):
        t_current = float(frame_times[frame_i])
        time_text.set_text(f"MVPA/Haufe time = {t_current:.3f} s")
        for ax_topo, day in zip(topo_axes, days):
            ax_topo.clear()
            d_day = haufe_df[haufe_df["day"] == day]
            times = np.sort(d_day["time_sec"].dropna().unique().astype(float))
            t_show = float(times[int(np.argmin(np.abs(times - t_current)))])
            d_topo = d_day[np.isclose(d_day["time_sec"], t_show)]
            vals = (
                d_topo.set_index("channel")
                .reindex(ch_names)["pattern_mean"]
                .to_numpy(dtype=float)
            )
            mne.viz.plot_topomap(
                vals, info, axes=ax_topo, show=False, contours=0,
                cmap="RdBu_r", vlim=(-lim, lim), sphere=(0.0, 0.0, 0.0, 0.095),
            )
            ax_topo.set_title(f"Day {day} topomap", fontsize=10)
        for point, (line, x, y) in zip(point_artists, line_artists):
            line.set_xdata([t_current, t_current])
            point.set_offsets([[t_current, float(np.interp(t_current, x, y))]])
        return []

    anim_obj = animation.FuncAnimation(
        fig, draw_frame, frames=len(frame_times), interval=1000 / max(1, fps), blit=False,
    )
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=1800)
        anim_obj.save(movie_path, writer=writer, dpi=dpi)
        out_path = movie_path
    except Exception:
        writer = animation.PillowWriter(fps=fps)
        anim_obj.save(gif_path, writer=writer, dpi=dpi)
        out_path = gif_path
    plt.close(fig)
    return {"movie_path": out_path, "n_frames": int(len(frame_times))}


if __name__ == "__main__":
    run_stimulus_locked_mvpa_analysis()
