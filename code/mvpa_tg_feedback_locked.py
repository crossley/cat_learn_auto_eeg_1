#!/usr/bin/env python3
"""Feedback-locked cross-day temporal generalization for category labels."""

from __future__ import annotations

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
from mne.decoding import GeneralizingEstimator

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from load_project_data import align_behaviour_to_epochs, load_sessions
from mvpa_tg_cross_day import write_cross_day_outputs
from mvpa_tg_within_day import (
    balanced_day_subset,
    build_clf,
    pick_eeg_interpolate_bads,
    session_cache_key,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8


def prepare_feedback_data(epochs, behaviour):
    fb_events = [x for x in ["FB/Cor", "FB/Inc"] if x in epochs.event_id]
    if len(fb_events) < 2:
        raise ValueError(f"missing_feedback_labels:{','.join(fb_events)}")
    fb_epochs = epochs[fb_events].copy()
    fb_epochs.load_data()
    pick_eeg_interpolate_bads(fb_epochs)
    fb_epochs.resample(128, npad="auto")
    fb_epochs, beh_aligned = align_behaviour_to_epochs(
        behaviour, fb_epochs, event_names=("FB/Cor", "FB/Inc")
    )
    y = (beh_aligned["cat"].astype(str) == "B").astype(int).to_numpy()
    X = fb_epochs.get_data()
    t = fb_epochs.times.copy()
    ch_names = np.array(fb_epochs.ch_names, dtype=str)
    return X, y, t, ch_names, beh_aligned


def prepare_feedback_session_cache(
    session_item: dict,
    cache_dir: Path,
    cache_prefix: str = "tg_feedback_cat_cache_interp_bads",
):
    cache_dir = Path(cache_dir)
    session_file = session_item["epo_file"]
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    epochs = session_item["epochs"]
    behaviour = session_item["beh"]
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
        X, y, t, ch_names, beh_aligned = prepare_feedback_data(epochs, behaviour)
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


def process_feedback_day_pair(pair_item: dict, random_state: int):
    subject = int(pair_item["subject"])
    d_train = int(pair_item["train_day"])
    d_test = int(pair_item["test_day"])
    train_session_file = pair_item["train_session_file"]
    test_session_file = pair_item["test_session_file"]
    train_cache_path = pair_item["train_cache_path"]
    test_cache_path = pair_item["test_cache_path"]
    pair_seed = int(pair_item["pair_seed"])

    with np.load(train_cache_path, allow_pickle=False) as z:
        X_train_all = z["X"]
        y_train_all = z["y"]
        t_train = z["t"]
        ch_train = z["ch_names"] if "ch_names" in z.files else np.array([], dtype=str)
    with np.load(test_cache_path, allow_pickle=False) as z:
        X_test_all = z["X"]
        y_test_all = z["y"]
        ch_test = z["ch_names"] if "ch_names" in z.files else np.array([], dtype=str)

    if len(ch_train) == 0 or len(ch_test) == 0 or ch_train.tolist() != ch_test.tolist():
        return {
            "ok": False,
            "qc": {
                "session_file": f"{train_session_file}->{test_session_file}",
                "subject": subject,
                "day": d_train,
                "stage": "cross_day_channels",
                "reason": "channel_mismatch",
                "detail": f"train_n={len(ch_train)}, test_n={len(ch_test)}",
            },
        }

    n_per_class = int(
        min(
            np.sum(y_train_all == 0),
            np.sum(y_train_all == 1),
            np.sum(y_test_all == 0),
            np.sum(y_test_all == 1),
        )
    )
    if n_per_class < 5:
        return {
            "ok": False,
            "qc": {
                "session_file": f"{train_session_file}->{test_session_file}",
                "subject": subject,
                "day": d_train,
                "stage": "cross_day_balance",
                "reason": "insufficient_balanced_trials",
                "detail": f"n_per_class={n_per_class}",
            },
        }

    rng_pair = np.random.default_rng(pair_seed)
    X_train, y_train = balanced_day_subset(
        X_train_all, y_train_all, n_per_class=n_per_class, rng=rng_pair
    )
    X_test, y_test = balanced_day_subset(
        X_test_all, y_test_all, n_per_class=n_per_class, rng=rng_pair
    )

    clf = build_clf(random_state=random_state)
    ge = GeneralizingEstimator(clf, scoring="roc_auc", n_jobs=1, verbose=False)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            ge.fit(X_train, y_train)
            mat_transfer = ge.score(X_test, y_test)
    except Exception as exc:
        return {
            "ok": False,
            "qc": {
                "session_file": f"{train_session_file}->{test_session_file}",
                "subject": subject,
                "day": d_train,
                "stage": "cross_day_tg",
                "reason": "compute_error",
                "detail": str(exc),
            },
        }

    return {
        "ok": True,
        "row": {
            "subject": subject,
            "train_day": d_train,
            "test_day": d_test,
            "n_per_class": int(n_per_class),
            "n_train_trials_used": int(len(y_train)),
            "n_test_trials_used": int(len(y_test)),
            "diag_mean_auc": float(np.nanmean(np.diag(mat_transfer))),
        },
        "t": t_train,
        "mat": mat_transfer,
    }


def save_fig_mvpa_temporal_generalization_feedback_cross_day(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    cross_day_mean_csv = output_dir / "tg_feedback_cat_day_mean.csv"
    cross_matrix_day_mean_csv = output_dir / "tg_feedback_cat_timegen_day_mean.csv"
    if not cross_day_mean_csv.exists():
        raise FileNotFoundError(
            f"Missing TG feedback-cat output in {output_dir}. Run run_mvpa_tg_feedback_locked() first."
        )
    cross_day_mean_df = pd.read_csv(cross_day_mean_csv)
    fig_cross = figures_dir / "tg_feedback_cat_transfer_5x4.png"
    fig_cross_timegen = figures_dir / "tg_feedback_cat_timegen_matrices_5x5.png"

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    day_grid = sorted({1, 2, 3, 4, 5})
    mat = np.full((len(day_grid), len(day_grid)), np.nan)
    if not cross_day_mean_df.empty:
        for _, r in cross_day_mean_df.iterrows():
            i = day_grid.index(int(r["train_day"]))
            j = day_grid.index(int(r["test_day"]))
            mat[i, j] = float(r["auc_mean"])
    masked = np.ma.masked_invalid(mat)
    im = ax.imshow(masked, cmap="magma", aspect="equal")
    ax.set_xticks(range(len(day_grid)))
    ax.set_yticks(range(len(day_grid)))
    ax.set_xticklabels([f"D{d}" for d in day_grid])
    ax.set_yticklabels([f"D{d}" for d in day_grid])
    ax.set_xlabel("Test Day")
    ax.set_ylabel("Train Day")
    ax.set_title("Feedback-Locked Transfer (Diagonal Mean AUC)")
    for i in range(len(day_grid)):
        for j in range(len(day_grid)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color="white")
            elif i == j:
                ax.text(j, i, "—", ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, shrink=0.9, label="AUC")
    fig.tight_layout()
    fig.savefig(fig_cross, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if cross_matrix_day_mean_csv.exists():
        d_mat = pd.read_csv(cross_matrix_day_mean_csv)
        if not d_mat.empty:
            fig, axes = plt.subplots(5, 5, figsize=(15, 15), squeeze=False)
            vmin = float(d_mat["auc_mean"].min())
            vmax = float(d_mat["auc_mean"].max())
            for i, train_day in enumerate(day_grid):
                for j, test_day in enumerate(day_grid):
                    ax = axes[i, j]
                    if train_day == test_day:
                        ax.axis("off")
                        ax.text(0.5, 0.5, f"D{train_day}=D{test_day}", ha="center", va="center")
                        continue
                    g = d_mat[
                        (d_mat["train_day"] == train_day) & (d_mat["test_day"] == test_day)
                    ]
                    if g.empty:
                        ax.axis("off")
                        continue
                    pivot = g.pivot(
                        index="train_time_sec", columns="test_time_sec", values="auc_mean"
                    )
                    im = ax.imshow(
                        pivot.to_numpy(),
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
                    ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
                    ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
                    ax.set_title(f"Train D{train_day} -> Test D{test_day}", fontsize=9)
                    if i == len(day_grid) - 1:
                        ax.set_xlabel("Test Time (s)")
                    if j == 0:
                        ax.set_ylabel("Train Time (s)")
            fig.suptitle("Feedback-Locked Temporal Generalization by Day Pair (AUC)")
            fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.75, label="AUC")
            fig.subplots_adjust(top=0.94, wspace=0.30, hspace=0.35)
            fig.savefig(fig_cross_timegen, dpi=150, bbox_inches="tight")
            plt.close(fig)

    return {"figure_path": fig_cross, "timegen_figure_path": fig_cross_timegen}


def run_mvpa_tg_feedback_locked(
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

    cross_subject_csv = output_dir / "tg_feedback_cat_subject_level.csv"
    cross_day_mean_csv = output_dir / "tg_feedback_cat_day_mean.csv"
    cross_matrix_dir = output_dir
    cross_matrix_day_mean_csv = output_dir / "tg_feedback_cat_timegen_day_mean.csv"
    qc_csv = output_dir / "tg_feedback_cat_qc_log.csv"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    wrote_qc = False
    t0 = time.time()

    def append_csv(df: pd.DataFrame, path: Path, wrote_flag: bool):
        if df.empty:
            return wrote_flag
        df.to_csv(path, mode="a", header=not wrote_flag, index=False)
        return True

    session_items = load_sessions(load_epochs=True)
    cache_results = [
        prepare_feedback_session_cache(
            item,
            cache_dir=output_dir,
            cache_prefix="tg_feedback_cat_cache_interp_bads",
        )
        for item in session_items
    ]
    prepared_map: dict[tuple, dict] = {}
    for result in cache_results:
        if not result["ok"]:
            qc_rows.append(result["qc"])
        else:
            prepared_map[(result["subject"], result["day"])] = result
    if qc_rows:
        wrote_qc = append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))

    rng_master = np.random.default_rng(random_state)
    pair_items = []
    cross_matrix_dir.mkdir(parents=True, exist_ok=True)
    subjects = sorted({k[0] for k in prepared_map})
    for subject in subjects:
        subject_days = sorted([k[1] for k in prepared_map if k[0] == subject])
        if len(subject_days) < 2:
            continue
        for d_train in subject_days:
            for d_test in subject_days:
                if d_test == d_train:
                    continue
                train_item = prepared_map[(subject, d_train)]
                test_item = prepared_map[(subject, d_test)]
                pair_items.append(
                    {
                        "subject": subject,
                        "train_day": d_train,
                        "test_day": d_test,
                        "train_cache_path": train_item["cache_path"],
                        "test_cache_path": test_item["cache_path"],
                        "train_session_file": train_item["session_file"],
                        "test_session_file": test_item["session_file"],
                        "pair_seed": int(rng_master.integers(0, 2**31 - 1)),
                    }
                )

    cross_rows = []
    cross_matrix_accum: dict = {}
    cross_time_template = None
    cross_done = 0
    wrote_cross = False

    def handle_cross_result(result):
        nonlocal wrote_cross, wrote_qc, cross_done, qc_rows, cross_time_template
        if result["ok"]:
            row = result["row"]
            mat = np.asarray(result["mat"], dtype=float)
            t_vec = np.asarray(result["t"], dtype=float)
            cross_rows.append(row)
            wrote_cross = append_csv(pd.DataFrame([row]), cross_subject_csv, wrote_cross)
            matrix_path = cross_matrix_dir / (
                f"tg_feedback_cat_matrix_sub_{int(row['subject']):03d}_trainD{int(row['train_day'])}"
                f"_testD{int(row['test_day'])}.npz"
            )
            np.savez_compressed(matrix_path, auc=mat, time_sec=t_vec)
            if cross_time_template is None:
                cross_time_template = t_vec
            key = (int(row["train_day"]), int(row["test_day"]))
            if key not in cross_matrix_accum:
                cross_matrix_accum[key] = {
                    "sum": np.zeros_like(mat, dtype=float),
                    "count": np.zeros_like(mat, dtype=float),
                }
            valid = np.isfinite(mat)
            cross_matrix_accum[key]["sum"][valid] += mat[valid]
            cross_matrix_accum[key]["count"][valid] += 1.0
        else:
            qc_rows.append(result["qc"])
            if len(qc_rows) >= max(progress_every, 1):
                wrote_qc = append_csv(
                    pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc
                )
                qc_rows = []
        cross_done += 1
        if (cross_done % max(progress_every * 2, 1)) == 0:
            elapsed = time.time() - t0
            print(
                f"[TG feedback] processed {cross_done}/{len(pair_items)} pairs "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    print(
        f"[TG feedback] Starting cross-day transfer on {len(subjects)} subjects, "
        f"{len(pair_items)} pairs (n_workers={n_workers})...",
        flush=True,
    )
    if n_workers == 1:
        if threadpool_limits is None:
            for item in pair_items:
                handle_cross_result(process_feedback_day_pair(pair_item=item, random_state=random_state))
        else:
            with threadpool_limits(limits=1):
                for item in pair_items:
                    handle_cross_result(
                        process_feedback_day_pair(pair_item=item, random_state=random_state)
                    )
    elif pair_items:
        if threadpool_limits is None:
            result_iter = Parallel(
                n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
            )(
                delayed(process_feedback_day_pair)(pair_item=item, random_state=random_state)
                for item in pair_items
            )
            for result in result_iter:
                handle_cross_result(result)
        else:
            with threadpool_limits(limits=1):
                result_iter = Parallel(
                    n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
                )(
                    delayed(process_feedback_day_pair)(pair_item=item, random_state=random_state)
                    for item in pair_items
                )
                for result in result_iter:
                    handle_cross_result(result)

    if qc_rows:
        wrote_qc = append_csv(pd.DataFrame(qc_rows, columns=qc_columns), qc_csv, wrote_qc)
        qc_rows = []

    cross_subject_df, cross_day_mean_df = write_cross_day_outputs(
        cross_rows=cross_rows,
        cross_matrix_accum=cross_matrix_accum,
        cross_time_template=cross_time_template,
        cross_matrix_dir=cross_matrix_dir,
        cross_subject_csv=cross_subject_csv,
        cross_day_mean_csv=cross_day_mean_csv,
        cross_matrix_day_mean_csv=cross_matrix_day_mean_csv,
    )
    qc_df = pd.read_csv(qc_csv) if qc_csv.exists() else pd.DataFrame(columns=qc_columns)

    if save_figures:
        save_fig_mvpa_temporal_generalization_feedback_cross_day(
            output_dir=output_dir, figures_dir=figures_dir
        )

    return {
        "cross_subject_df": cross_subject_df,
        "cross_day_mean_df": cross_day_mean_df,
        "qc_df": qc_df,
        "cross_subject_csv": cross_subject_csv,
        "cross_day_mean_csv": cross_day_mean_csv,
        "cross_matrix_day_mean_csv": cross_matrix_day_mean_csv,
        "qc_csv": qc_csv,
    }


if __name__ == "__main__":
    run_mvpa_tg_feedback_locked()
