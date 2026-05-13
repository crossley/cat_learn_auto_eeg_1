#!/usr/bin/env python3
"""Within-day temporal generalization MVPA and session cache infrastructure."""

from __future__ import annotations

import hashlib
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
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from mne.decoding import GeneralizingEstimator, cross_val_multiscore

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from load_project_data import load_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8


def session_cache_key(session_item: dict):
    token = f"{session_item['subject']}_{session_item['day']}_{session_item['epo_file']}"
    return hashlib.md5(token.encode("utf-8")).hexdigest()[:12]


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


def pick_eeg_interpolate_bads(epochs):
    epochs.pick("eeg", exclude=[])
    if len(epochs.ch_names) == 0:
        raise RuntimeError("no_eeg_channels_after_pick")
    if len(epochs.info.get("bads", [])):
        epochs.interpolate_bads(reset_bads=True, verbose="ERROR")
    return epochs


def prepare_stim_data(epochs):
    stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
    if len(stim_events) < 2:
        raise ValueError(f"missing_stim_labels:{','.join(stim_events)}")
    stim_epochs = epochs[stim_events].copy()
    stim_epochs.load_data()
    pick_eeg_interpolate_bads(stim_epochs)
    stim_epochs.resample(128, npad="auto")
    codes = stim_epochs.events[:, 2]
    y = np.full(len(codes), -1, dtype=int)
    y[codes == stim_epochs.event_id["Stim/A"]] = 0
    y[codes == stim_epochs.event_id["Stim/B"]] = 1
    keep = y >= 0
    X = stim_epochs.get_data()[keep]
    y = y[keep]
    t = stim_epochs.times.copy()
    ch_names = np.array(stim_epochs.ch_names, dtype=str)
    return X, y, t, ch_names


def prepare_session_cache(
    session_item: dict,
    cache_dir: Path,
    cache_prefix: str = "stim_cache_interp_bads",
):
    session_file = session_item["epo_file"]
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    epochs = session_item["epochs"]
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_prefix}_{session_cache_key(session_item)}.npz"

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            if "ch_names" in z.files:
                t = z["t"]
                y = z["y"]
                ch_names = z["ch_names"]
                return {
                    "ok": True,
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "cache_path": str(cache_path),
                    "n_trials": int(len(y)),
                    "n_a": int(np.sum(y == 0)),
                    "n_b": int(np.sum(y == 1)),
                    "n_times": int(len(t)),
                    "ch_names": ch_names.tolist(),
                }

    try:
        X, y, t, ch_names = prepare_stim_data(epochs)
    except Exception as exc:
        msg = str(exc)
        reason = msg.split(":")[0] if ":" in msg else "prep_error"
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "prepare",
                "reason": reason,
                "detail": msg,
            },
        }

    np.savez_compressed(cache_path, X=X, y=y, t=t, ch_names=ch_names)
    return {
        "ok": True,
        "session_file": session_file,
        "subject": subject,
        "day": day,
        "cache_path": str(cache_path),
        "n_trials": int(len(y)),
        "n_a": int(np.sum(y == 0)),
        "n_b": int(np.sum(y == 1)),
        "n_times": int(len(t)),
        "ch_names": ch_names.tolist(),
    }


def balanced_day_subset(X, y, n_per_class: int, rng: np.random.Generator):
    idx0 = np.where(y == 0)[0]
    idx1 = np.where(y == 1)[0]
    pick0 = rng.choice(idx0, size=n_per_class, replace=False)
    pick1 = rng.choice(idx1, size=n_per_class, replace=False)
    idx = np.concatenate([pick0, pick1])
    rng.shuffle(idx)
    return X[idx], y[idx]


def process_within_day_session(session_meta: dict, min_epochs: int, random_state: int):
    session_file = session_meta["session_file"]
    subject = int(session_meta["subject"])
    day = int(session_meta["day"])
    cache_path = session_meta["cache_path"]

    with np.load(cache_path, allow_pickle=False) as z:
        X = z["X"]
        y = z["y"]
        t = z["t"]

    if len(y) < min_epochs:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "epoch_count",
                "reason": "insufficient_epochs",
                "detail": f"n_trials={len(y)} < min_epochs={min_epochs}",
            },
        }
    if min(np.sum(y == 0), np.sum(y == 1)) < 5:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "class_balance",
                "reason": "insufficient_class_trials",
                "detail": f"n_a={int(np.sum(y==0))}, n_b={int(np.sum(y==1))}",
            },
        }

    clf = build_clf(random_state=random_state)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    ge = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=1, verbose=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            scores = cross_val_multiscore(ge, X, y, cv=cv, n_jobs=1)
        mat = np.nanmean(scores, axis=0)
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "within_day_tg",
                "reason": "compute_error",
                "detail": str(exc),
            },
        }

    return {
        "ok": True,
        "session_file": session_file,
        "subject": subject,
        "day": day,
        "cache_path": cache_path,
        "X": X,
        "y": y,
        "t": t,
        "mat": mat,
    }


def run_mvpa_tg_within_day(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    min_epochs: int = 20,
    random_state: int = 42,
    progress_every: int = 5,
    n_workers: int | None = None,
    save_figures: bool = True,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    within_subject_csv = output_dir / "tg_within_day_subject_level.csv"
    within_day_mean_csv = output_dir / "tg_within_day_day_mean.csv"
    qc_csv = output_dir / "tg_within_day_qc_log.csv"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    t0 = time.time()
    wrote_within_subject = False
    wrote_qc = False

    def append_csv(df: pd.DataFrame, path: Path, wrote_flag: bool):
        if df.empty:
            return wrote_flag
        df.to_csv(path, mode="a", header=not wrote_flag, index=False)
        return True

    session_items = load_sessions(load_epochs=True)
    cache_results = [
        prepare_session_cache(
            item,
            cache_dir=output_dir,
            cache_prefix="tg_within_day_stim_cache_interp_bads",
        )
        for item in session_items
    ]
    prepared_items = []
    for result in cache_results:
        if not result["ok"]:
            qc_rows.append(result["qc"])
        else:
            prepared_items.append(result)
    if qc_rows:
        wrote_qc = append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    within_mats = []
    time_template = None
    n_done = 0

    def handle_within_result(result):
        nonlocal n_done, time_template, wrote_within_subject, wrote_qc, qc_rows
        if not result["ok"]:
            qc_rows.append(result["qc"])
            if len(qc_rows) >= max(progress_every, 1):
                wrote_qc = append_csv(
                    pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc
                )
                qc_rows = []
            return
        subject = int(result["subject"])
        day = int(result["day"])
        X = result["X"]
        y = result["y"]
        t = result["t"]
        mat = result["mat"]
        session_file = result["session_file"]
        if time_template is None:
            time_template = t
        elif (len(t) != len(time_template)) or (not np.allclose(t, time_template, atol=1e-9)):
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "time_grid",
                    "reason": "inconsistent_time_axis",
                    "detail": "",
                }
            )
            return
        within_mats.append(
            {
                "subject": subject,
                "day": day,
                "session_file": session_file,
                "mat": mat,
                "n_trials": int(len(y)),
                "n_a": int(np.sum(y == 0)),
                "n_b": int(np.sum(y == 1)),
            }
        )
        rows_local = []
        for i in range(len(t)):
            for j in range(len(t)):
                rows_local.append(
                    {
                        "subject": subject,
                        "day": day,
                        "session_file": session_file,
                        "n_trials": int(len(y)),
                        "n_a": int(np.sum(y == 0)),
                        "n_b": int(np.sum(y == 1)),
                        "train_time_sec": float(t[i]),
                        "test_time_sec": float(t[j]),
                        "auc": float(mat[i, j]),
                    }
                )
        wrote_within_subject = append_csv(
            pd.DataFrame(rows_local), within_subject_csv, wrote_within_subject
        )
        n_done += 1
        if (n_done % max(progress_every, 1)) == 0:
            elapsed = time.time() - t0
            print(
                f"[TG within] complete {n_done}/{len(prepared_items)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    print(
        f"[TG within] Starting within-day TG on {len(prepared_items)} sessions "
        f"(n_workers={n_workers})...",
        flush=True,
    )
    if n_workers == 1:
        runner = (lambda: [handle_within_result(
            process_within_day_session(
                session_meta=item, min_epochs=min_epochs, random_state=random_state
            )
        ) for item in prepared_items])
        if threadpool_limits is None:
            runner()
        else:
            with threadpool_limits(limits=1):
                runner()
    elif len(prepared_items) > 0:
        if threadpool_limits is None:
            result_iter = Parallel(
                n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
            )(
                delayed(process_within_day_session)(
                    session_meta=item, min_epochs=min_epochs, random_state=random_state
                )
                for item in prepared_items
            )
            for result in result_iter:
                handle_within_result(result)
        else:
            with threadpool_limits(limits=1):
                result_iter = Parallel(
                    n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
                )(
                    delayed(process_within_day_session)(
                        session_meta=item, min_epochs=min_epochs, random_state=random_state
                    )
                    for item in prepared_items
                )
                for result in result_iter:
                    handle_within_result(result)

    if qc_rows:
        wrote_qc = append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    if not within_mats:
        pd.DataFrame().to_csv(within_subject_csv, index=False)
        pd.DataFrame().to_csv(within_day_mean_csv, index=False)
        raise RuntimeError("No valid within-day TG matrices were computed.")

    n_t = len(time_template)
    within_rows = []
    for item in within_mats:
        mat = item["mat"]
        for i in range(n_t):
            for j in range(n_t):
                within_rows.append(
                    {
                        "subject": item["subject"],
                        "day": item["day"],
                        "session_file": item["session_file"],
                        "n_trials": item["n_trials"],
                        "n_a": item["n_a"],
                        "n_b": item["n_b"],
                        "train_time_sec": float(time_template[i]),
                        "test_time_sec": float(time_template[j]),
                        "auc": float(mat[i, j]),
                    }
                )
    within_subject_df = pd.DataFrame(within_rows)
    within_day_mean_df = (
        within_subject_df.groupby(["day", "train_time_sec", "test_time_sec"], as_index=False)
        .agg(
            auc_mean=("auc", "mean"),
            auc_sem=(
                "auc",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            ),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "train_time_sec", "test_time_sec"])
    )
    within_day_mean_df.to_csv(within_day_mean_csv, index=False)
    qc_df = pd.read_csv(qc_csv) if qc_csv.exists() else pd.DataFrame(columns=qc_columns)

    if save_figures:
        save_fig_mvpa_temporal_generalization_within_day(
            output_dir=output_dir, figures_dir=figures_dir
        )

    return {
        "within_subject_df": within_subject_df,
        "within_day_mean_df": within_day_mean_df,
        "qc_df": qc_df,
        "time_sec": np.array(time_template, dtype=float),
        "within_subject_csv": within_subject_csv,
        "within_day_mean_csv": within_day_mean_csv,
        "qc_csv": qc_csv,
        "prepared_items": prepared_items,
    }


def save_fig_mvpa_temporal_generalization_within_day(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    within_day_mean_csv = output_dir / "tg_within_day_day_mean.csv"
    if not within_day_mean_csv.exists():
        raise FileNotFoundError(
            f"Missing TG within-day output in {output_dir}. "
            "Run run_mvpa_tg_within_day() first."
        )
    within_day_mean_df = pd.read_csv(within_day_mean_csv)
    fig_within = figures_dir / "tg_within_day_heatmaps.png"

    days = sorted(within_day_mean_df["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(4.6 * len(days), 4), squeeze=False)
    vmin = float(within_day_mean_df["auc_mean"].min())
    vmax = float(within_day_mean_df["auc_mean"].max())
    for ax, day in zip(axes.ravel(), days):
        g = within_day_mean_df[within_day_mean_df["day"] == day]
        pivot = g.pivot(index="train_time_sec", columns="test_time_sec", values="auc_mean")
        mat = pivot.to_numpy()
        im = ax.imshow(
            mat,
            origin="lower",
            aspect="auto",
            extent=[
                float(pivot.columns.min()),
                float(pivot.columns.max()),
                float(pivot.index.min()),
                float(pivot.index.max()),
            ],
            vmin=vmin,
            vmax=vmax,
            cmap="viridis",
        )
        ax.axvline(0.0, color="white", linestyle=":", linewidth=1)
        ax.axhline(0.0, color="white", linestyle=":", linewidth=1)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("Test Time (s)")
        ax.set_ylabel("Train Time (s)")
    fig.suptitle("Within-Day Temporal Generalization (AUC)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="AUC")
    fig.subplots_adjust(top=0.85, wspace=0.28)
    fig.savefig(fig_within, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"figure_path": fig_within}


if __name__ == "__main__":
    run_mvpa_tg_within_day()
