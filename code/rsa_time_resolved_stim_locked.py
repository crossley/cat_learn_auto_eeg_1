#!/usr/bin/env python3
"""Time-resolved RSA for stimulus-locked EEG responses."""

from __future__ import annotations

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
from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from boundary_distance import load_behaviour_with_boundary
from load_project_data import align_behaviour_to_epochs, load_sessions
from mvpa_tg_within_day import pick_eeg_interpolate_bads
from rsa_model_predictions import (
    MIN_TRIALS_PER_BIN_SESSION,
    assign_grid_bins,
    choose_grid,
    make_bin_table,
    make_model_rdms,
    run_rsa_model_predictions,
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8
MIN_EPOCHS_PER_BIN = 5
SNAPSHOT_TIMES = [0.10, 0.20, 0.35, 0.60]
GEOMETRY_WINDOWS = {
    "early": (0.06, 0.18),
    "late": (0.30, 0.60),
}


def _vector_corr(x_vec, y_vec):
    x_vec = np.asarray(x_vec, dtype=float)
    y_vec = np.asarray(y_vec, dtype=float)
    good = np.isfinite(x_vec) & np.isfinite(y_vec)
    if np.sum(good) < 3:
        return np.nan
    x = x_vec[good] - np.nanmean(x_vec[good])
    y = y_vec[good] - np.nanmean(y_vec[good])
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom == 0.0:
        return np.nan
    return float(np.sum(x * y) / denom)


def _append_csv(df: pd.DataFrame, path: Path, wrote_flag: bool):
    if df.empty:
        return wrote_flag
    df.to_csv(path, mode="a", header=not wrote_flag, index=False)
    return True


def _load_or_build_bins(output_dir):
    bins_csv = output_dir / "rsa_model_stimulus_bins.csv"
    diagnostics_csv = output_dir / "rsa_model_grid_diagnostics.csv"
    rdm_csv = output_dir / "rsa_model_rdms.csv"
    if not bins_csv.exists() or not diagnostics_csv.exists() or not rdm_csv.exists():
        run_rsa_model_predictions(output_dir=output_dir, figures_dir=FIGURES_DIR)

    beh, boundary = load_behaviour_with_boundary()
    selected_n, _ = choose_grid(beh)
    beh_binned, x_edges, y_edges = assign_grid_bins(beh, selected_n)
    _, retained_bins = make_bin_table(beh, boundary, selected_n)
    model_rdms = make_model_rdms(retained_bins)
    retained_keys = [
        (int(row.sf_bin), int(row.ori_bin)) for row in retained_bins.itertuples()
    ]
    return beh_binned, retained_bins, retained_keys, model_rdms, x_edges, y_edges


def _assign_existing_grid_bins(beh, x_edges, y_edges):
    d = beh.copy()
    d["sf_bin"] = pd.cut(
        d["x"], bins=x_edges, labels=False, include_lowest=True
    ).astype("Int64")
    d["ori_bin"] = pd.cut(
        d["y"], bins=y_edges, labels=False, include_lowest=True
    ).astype("Int64")
    return d


def _make_model_vectors(model_rdms):
    rows = []
    for name, mat in model_rdms.items():
        for i in range(mat.shape[0]):
            for j in range(i + 1, mat.shape[1]):
                rows.append(
                    {
                        "model": name,
                        "bin_i": int(i),
                        "bin_j": int(j),
                        "model_dissimilarity": float(mat[i, j]),
                    }
                )
    return pd.DataFrame(rows)


def process_rsa_session(task):
    subject = int(task["subject"])
    day = int(task["day"])
    session_file = task["epo_file"]
    retained_keys = task["retained_keys"]
    x_edges = task["x_edges"]
    y_edges = task["y_edges"]

    try:
        epochs = mne.read_epochs(task["epo_path"], preload=False, verbose="ERROR")
        stim_epochs, beh_aligned = align_behaviour_to_epochs(
            task["beh"], epochs, event_names=("Stim/A", "Stim/B")
        )
        stim_epochs = stim_epochs.copy()
        stim_epochs.load_data()
        pick_eeg_interpolate_bads(stim_epochs)
        stim_epochs.resample(128, npad="auto")
        beh_aligned = _assign_existing_grid_bins(beh_aligned, x_edges, y_edges)
        X = stim_epochs.get_data()
        times = stim_epochs.times.copy()
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

    bin_patterns = []
    bin_counts = []
    for sf_bin, ori_bin in retained_keys:
        keep = (
            (beh_aligned["sf_bin"].astype("Int64") == sf_bin)
            & (beh_aligned["ori_bin"].astype("Int64") == ori_bin)
        ).to_numpy()
        n_epochs = int(np.sum(keep))
        bin_counts.append(n_epochs)
        if n_epochs < MIN_EPOCHS_PER_BIN:
            bin_patterns.append(np.full((X.shape[1], X.shape[2]), np.nan))
        else:
            bin_patterns.append(np.nanmean(X[keep], axis=0))

    patterns = np.stack(bin_patterns, axis=0)
    n_bins = patterns.shape[0]
    rdm_rows = []
    corr_rows = []
    for ti, time_sec in enumerate(times):
        model_vec_rows = []
        for i in range(n_bins):
            x_vec = patterns[i, :, ti]
            for j in range(i + 1, n_bins):
                sim = _vector_corr(x_vec, patterns[j, :, ti])
                if not np.isfinite(sim):
                    continue
                dissim = 1.0 - sim
                row = {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "time_sec": float(time_sec),
                    "bin_i": int(i),
                    "bin_j": int(j),
                    "dissimilarity": float(dissim),
                    "n_i": int(bin_counts[i]),
                    "n_j": int(bin_counts[j]),
                }
                rdm_rows.append(row)
                model_vec_rows.append(row)
        if model_vec_rows:
            corr_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "time_sec": float(time_sec),
                    "n_pairs": int(len(model_vec_rows)),
                }
            )

    count_rows = [
        {
            "session_file": session_file,
            "subject": subject,
            "day": day,
            "bin_i": int(i),
            "sf_bin": int(retained_keys[i][0]),
            "ori_bin": int(retained_keys[i][1]),
            "n_epochs": int(bin_counts[i]),
            "usable": bool(bin_counts[i] >= MIN_EPOCHS_PER_BIN),
        }
        for i in range(n_bins)
    ]

    return {
        "ok": True,
        "rdm_df": pd.DataFrame(rdm_rows),
        "count_df": pd.DataFrame(count_rows),
        "time_df": pd.DataFrame(corr_rows),
    }


def compute_model_fit_timecourses(rdm_df, model_vec_df):
    if rdm_df.empty:
        return pd.DataFrame()
    merged = rdm_df.merge(model_vec_df, on=["bin_i", "bin_j"], how="inner")
    rows = []
    for (subject, day, time_sec, model), g in merged.groupby(
        ["subject", "day", "time_sec", "model"], sort=False
    ):
        if len(g) < 8:
            rho = np.nan
        else:
            rho = g["dissimilarity"].corr(
                g["model_dissimilarity"], method="spearman"
            )
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "time_sec": float(time_sec),
                "model": model,
                "rho": float(rho) if np.isfinite(rho) else np.nan,
                "n_pairs": int(len(g)),
            }
        )
    return pd.DataFrame(rows)


def compute_cross_day_geometry_similarity(rdm_df):
    if rdm_df.empty:
        return pd.DataFrame()
    vec_map = {}
    for key, g in rdm_df.groupby(["subject", "day", "time_sec"], sort=False):
        g = g.sort_values(["bin_i", "bin_j"])
        vec_map[key] = g[["bin_i", "bin_j", "dissimilarity"]].reset_index(drop=True)

    rows = []
    for subject in sorted(rdm_df["subject"].dropna().unique().astype(int)):
        days = sorted(rdm_df.loc[rdm_df["subject"] == subject, "day"].dropna().unique().astype(int))
        times = sorted(
            rdm_df.loc[rdm_df["subject"] == subject, "time_sec"].dropna().unique().astype(float)
        )
        for d_train in days:
            for d_test in days:
                if d_train == d_test:
                    continue
                for time_sec in times:
                    key_train = (subject, d_train, time_sec)
                    key_test = (subject, d_test, time_sec)
                    if key_train not in vec_map or key_test not in vec_map:
                        continue
                    merged = vec_map[key_train].merge(
                        vec_map[key_test],
                        on=["bin_i", "bin_j"],
                        suffixes=("_train", "_test"),
                    )
                    if len(merged) < 8:
                        rho = np.nan
                    else:
                        rho = merged["dissimilarity_train"].corr(
                            merged["dissimilarity_test"], method="spearman"
                        )
                    rows.append(
                        {
                            "subject": int(subject),
                            "train_day": int(d_train),
                            "test_day": int(d_test),
                            "day_distance": int(abs(d_train - d_test)),
                            "day_pair_type": (
                                "day1_involving"
                                if d_train == 1 or d_test == 1
                                else "later_only"
                            ),
                            "time_sec": float(time_sec),
                            "rho": float(rho) if np.isfinite(rho) else np.nan,
                            "n_pairs": int(len(merged)),
                        }
                    )
    return pd.DataFrame(rows)


def save_cross_day_geometry_figures(
    similarity_df,
    figures_dir: Path | str = FIGURES_DIR,
):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    matrix_fig = figures_dir / "rsa_stim_cross_day_geometry_similarity.png"
    timecourse_fig = figures_dir / "rsa_stim_cross_day_geometry_timecourse.png"
    timecourse_5x5_fig = figures_dir / "rsa_stim_cross_day_geometry_timecourse_5x5.png"
    if similarity_df.empty:
        return {
            "cross_day_geometry_figure": None,
            "cross_day_geometry_timecourse_figure": None,
            "cross_day_geometry_timecourse_5x5_figure": None,
        }

    day_grid = sorted(
        set(similarity_df["train_day"].dropna().astype(int))
        | set(similarity_df["test_day"].dropna().astype(int))
    )
    fig, axes = plt.subplots(1, len(GEOMETRY_WINDOWS), figsize=(10.5, 4.4), squeeze=False)
    plot_values = []
    for _window_name, (tmin, tmax) in GEOMETRY_WINDOWS.items():
        g = similarity_df[
            (similarity_df["time_sec"] >= tmin) & (similarity_df["time_sec"] <= tmax)
        ]
        summary = (
            g.groupby(["train_day", "test_day"], as_index=False)
            .agg(rho=("rho", "mean"))
        )
        plot_values.extend(summary["rho"].dropna().tolist())
    vmin = float(np.nanquantile(plot_values, 0.02))
    vmax = float(np.nanquantile(plot_values, 0.98))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad(color="0.82")
    im = None
    for ax, (window_name, (tmin, tmax)) in zip(axes.ravel(), GEOMETRY_WINDOWS.items()):
        g = similarity_df[
            (similarity_df["time_sec"] >= tmin) & (similarity_df["time_sec"] <= tmax)
        ]
        summary = (
            g.groupby(["train_day", "test_day"], as_index=False)
            .agg(rho=("rho", "mean"))
        )
        mat = np.full((len(day_grid), len(day_grid)), np.nan)
        for row in summary.itertuples():
            i = day_grid.index(int(row.train_day))
            j = day_grid.index(int(row.test_day))
            mat[i, j] = float(row.rho)
        np.fill_diagonal(mat, np.nan)
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            origin="upper",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(f"{window_name}: {tmin*1000:.0f}-{tmax*1000:.0f} ms")
        ax.set_xticks(range(len(day_grid)))
        ax.set_yticks(range(len(day_grid)))
        ax.set_xticklabels([f"D{d}" for d in day_grid])
        ax.set_yticklabels([f"D{d}" for d in day_grid])
        ax.set_xlabel("Test day")
        ax.set_ylabel("Train day")
        for i in range(len(day_grid)):
            for j in range(len(day_grid)):
                if np.isfinite(mat[i, j]):
                    color = "white" if mat[i, j] < (vmin + vmax) / 2 else "black"
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color=color)
    fig.suptitle("Cross-day neural RDM similarity")
    fig.subplots_adjust(top=0.84, bottom=0.16, left=0.08, right=0.88, wspace=0.35)
    cax = fig.add_axes([0.90, 0.20, 0.015, 0.58])
    fig.colorbar(im, cax=cax, label="Spearman rho")
    fig.savefig(matrix_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4), squeeze=False, sharex=True)
    ax = axes[0, 0]
    for distance, g_distance in similarity_df.groupby("day_distance"):
        if int(distance) == 0:
            continue
        summary = (
            g_distance.groupby("time_sec", as_index=False)
            .agg(rho=("rho", "mean"))
            .sort_values("time_sec")
        )
        ax.plot(summary["time_sec"], summary["rho"], label=f"Distance {int(distance)}")
    ax.axvline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.set_title("By day distance")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RDM similarity")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    for pair_type, label in [
        ("day1_involving", "Day 1 involving"),
        ("later_only", "Later only"),
    ]:
        g_type = similarity_df[similarity_df["day_pair_type"] == pair_type]
        summary = (
            g_type.groupby("time_sec", as_index=False)
            .agg(rho=("rho", "mean"))
            .sort_values("time_sec")
        )
        ax.plot(summary["time_sec"], summary["rho"], label=label)
    ax.axvline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.set_title("Day 1 vs later-only pairs")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RDM similarity")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("Cross-day RDM similarity over time")
    fig.tight_layout()
    fig.savefig(timecourse_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    ymin = float(similarity_df["rho"].quantile(0.02))
    ymax = float(similarity_df["rho"].quantile(0.98))
    root_colors = {
        1: ["#08306b", "#2171b5", "#6baed6", "#bdd7e7"],
        2: ["#006d2c", "#31a354", "#a1d99b"],
        3: ["#a50f15", "#ef3b2c"],
        4: ["#54278f"],
    }
    for root_day in day_grid[:-1]:
        partners = [day for day in day_grid if day > root_day]
        for color, test_day in zip(root_colors[root_day], partners):
            g = similarity_df[
                (
                    (similarity_df["train_day"] == root_day)
                    & (similarity_df["test_day"] == test_day)
                )
                | (
                    (similarity_df["train_day"] == test_day)
                    & (similarity_df["test_day"] == root_day)
                )
            ]
            if g.empty:
                continue
            summary = (
                g.groupby("time_sec", as_index=False)
                .agg(rho=("rho", "mean"))
                .sort_values("time_sec")
            )
            ax.plot(
                summary["time_sec"],
                summary["rho"],
                color=color,
                linewidth=1.5,
                label=f"D{root_day}-D{test_day}",
            )
    ax.axvline(0.0, color="0.5", linestyle=":", linewidth=0.9)
    ax.axhline(0.0, color="0.5", linestyle=":", linewidth=0.9)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RDM similarity")
    ax.set_title("Cross-day RDM similarity by day pair")
    ax.legend(frameon=False, fontsize=8, ncol=5)
    fig.tight_layout()
    fig.savefig(timecourse_5x5_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "cross_day_geometry_figure": matrix_fig,
        "cross_day_geometry_timecourse_figure": timecourse_fig,
        "cross_day_geometry_timecourse_5x5_figure": timecourse_5x5_fig,
    }


def save_fig_rsa_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rdm_csv = output_dir / "rsa_stim_time_resolved_rdms.csv"
    model_fit_csv = output_dir / "rsa_stim_model_fit_timecourses.csv"
    if not rdm_csv.exists() or not model_fit_csv.exists():
        raise FileNotFoundError(
            "Missing RSA time-resolved outputs. Run run_rsa_time_resolved() first."
        )
    rdm_df = pd.read_csv(rdm_csv)
    fit_df = pd.read_csv(model_fit_csv)
    cross_day_csv = output_dir / "rsa_stim_cross_day_geometry_similarity.csv"

    fit_fig = figures_dir / "rsa_stim_model_fit_timecourses.png"
    if not fit_df.empty:
        models = list(fit_df["model"].dropna().unique())
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False, sharex=True)
        for ax, model in zip(axes.ravel(), models):
            g_model = fit_df[fit_df["model"] == model]
            for day, g_day in g_model.groupby("day"):
                summary = (
                    g_day.groupby("time_sec", as_index=False)
                    .agg(
                        mean=("rho", "mean"),
                        sem=(
                            "rho",
                            lambda x: float(np.nanstd(x, ddof=1) / np.sqrt(np.sum(np.isfinite(x))))
                            if np.sum(np.isfinite(x)) > 1
                            else np.nan,
                        ),
                    )
                    .sort_values("time_sec")
                )
                ax.plot(summary["time_sec"], summary["mean"], label=f"D{int(day)}")
            ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
            ax.axvline(0.0, color="0.4", linestyle=":", linewidth=1)
            ax.set_title(model)
            ax.set_ylabel("Spearman rho")
            ax.set_xlabel("Time (s)")
        axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
        fig.suptitle("Time-resolved neural RSA model fits")
        fig.tight_layout()
        fig.savefig(fit_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

    snapshot_fig = figures_dir / "rsa_stim_neural_rdm_snapshots.png"
    if not rdm_df.empty:
        days = sorted(rdm_df["day"].dropna().unique().astype(int))
        bins = sorted(
            set(rdm_df["bin_i"].dropna().astype(int))
            | set(rdm_df["bin_j"].dropna().astype(int))
        )
        n_bins = max(bins) + 1
        fig, axes = plt.subplots(
            len(days), len(SNAPSHOT_TIMES), figsize=(14, 3.0 * len(days)), squeeze=False
        )
        snapshot_mats = []
        snapshot_items = []
        for day in days:
            g_day = rdm_df[rdm_df["day"] == day]
            day_times = np.sort(g_day["time_sec"].dropna().unique())
            for target_time in SNAPSHOT_TIMES:
                if len(day_times) == 0:
                    snapshot_items.append((day, None, None))
                    continue
                time_sec = float(day_times[np.argmin(np.abs(day_times - target_time))])
                g = (
                    g_day[np.isclose(g_day["time_sec"], time_sec)]
                    .groupby(["bin_i", "bin_j"], as_index=False)
                    .agg(dissimilarity=("dissimilarity", "mean"))
                )
                mat = np.zeros((n_bins, n_bins), dtype=float)
                mat[:] = np.nan
                np.fill_diagonal(mat, 0.0)
                for row in g.itertuples():
                    mat[int(row.bin_i), int(row.bin_j)] = float(row.dissimilarity)
                    mat[int(row.bin_j), int(row.bin_i)] = float(row.dissimilarity)
                snapshot_mats.append(mat)
                snapshot_items.append((day, time_sec, mat))
        finite_snapshot = np.concatenate(
            [mat[np.isfinite(mat)] for mat in snapshot_mats if np.any(np.isfinite(mat))]
        )
        vmin = float(np.nanquantile(finite_snapshot, 0.02))
        vmax = float(np.nanquantile(finite_snapshot, 0.98))
        im = None
        for i, day in enumerate(days):
            for j, _target_time in enumerate(SNAPSHOT_TIMES):
                ax = axes[i, j]
                _, time_sec, mat = snapshot_items[i * len(SNAPSHOT_TIMES) + j]
                if mat is None:
                    ax.axis("off")
                    continue
                im = ax.imshow(
                    np.ma.masked_invalid(mat),
                    origin="lower",
                    cmap="viridis",
                    vmin=vmin,
                    vmax=vmax,
                )
                if i == 0:
                    ax.set_title(f"{time_sec * 1000:.0f} ms")
                if j == 0:
                    ax.set_ylabel(f"Day {day}")
                ax.set_xlabel("Stimulus bin")
        fig.suptitle("Group-average neural RDM snapshots")
        fig.subplots_adjust(top=0.94, bottom=0.05, left=0.06, right=0.90, wspace=0.25, hspace=0.35)
        cax = fig.add_axes([0.92, 0.12, 0.015, 0.76])
        if im is not None:
            fig.colorbar(im, cax=cax, label="Neural dissimilarity")
        else:
            cax.axis("off")
        fig.savefig(snapshot_fig, dpi=150, bbox_inches="tight")
        plt.close(fig)

    if cross_day_csv.exists():
        cross_day_figs = save_cross_day_geometry_figures(
            pd.read_csv(cross_day_csv), figures_dir=figures_dir
        )
    else:
        cross_day_figs = {}

    return {
        "model_fit_figure": fit_fig,
        "snapshot_figure": snapshot_fig,
        "cross_day_geometry_figure": cross_day_figs.get("cross_day_geometry_figure"),
        "cross_day_geometry_timecourse_figure": cross_day_figs.get(
            "cross_day_geometry_timecourse_figure"
        ),
        "cross_day_geometry_timecourse_5x5_figure": cross_day_figs.get(
            "cross_day_geometry_timecourse_5x5_figure"
        ),
    }


def run_rsa_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    n_workers: int | None = None,
    progress_every: int = 5,
    save_figures: bool = True,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    rdm_csv = output_dir / "rsa_stim_time_resolved_rdms.csv"
    count_csv = output_dir / "rsa_stim_bin_epoch_counts.csv"
    model_fit_csv = output_dir / "rsa_stim_model_fit_timecourses.csv"
    cross_day_csv = output_dir / "rsa_stim_cross_day_geometry_similarity.csv"
    qc_csv = output_dir / "rsa_stim_time_resolved_qc_log.csv"
    for path in [rdm_csv, count_csv, model_fit_csv, cross_day_csv, qc_csv]:
        if path.exists():
            path.unlink()

    _, retained_bins, retained_keys, model_rdms, x_edges, y_edges = _load_or_build_bins(
        output_dir
    )
    model_vec_df = _make_model_vectors(model_rdms)
    model_vec_csv = output_dir / "rsa_stim_model_vectors.csv"
    model_vec_df.to_csv(model_vec_csv, index=False)

    sessions = load_sessions(load_epochs=False)
    tasks = [
        {
            **session,
            "retained_keys": retained_keys,
            "x_edges": x_edges,
            "y_edges": y_edges,
        }
        for session in sessions
    ]
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    t0 = time.time()
    print(
        f"[RSA stim] Processing {len(tasks)} sessions "
        f"({len(retained_bins)} bins, n_workers={n_workers})...",
        flush=True,
    )

    wrote_rdm = False
    wrote_count = False
    wrote_qc = False
    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    fit_frames = []
    done = 0

    def handle_result(result):
        nonlocal wrote_rdm, wrote_count, wrote_qc, done
        done += 1
        if result["ok"]:
            wrote_rdm = _append_csv(result["rdm_df"], rdm_csv, wrote_rdm)
            wrote_count = _append_csv(result["count_df"], count_csv, wrote_count)
            fit_frames.append(
                compute_model_fit_timecourses(result["rdm_df"], model_vec_df)
            )
        else:
            wrote_qc = _append_csv(
                pd.DataFrame([result["qc"]], columns=qc_columns), qc_csv, wrote_qc
            )
        if done % max(progress_every, 1) == 0:
            elapsed = time.time() - t0
            print(
                f"[RSA stim] complete {done}/{len(tasks)} sessions "
                f"(elapsed {elapsed/60:.1f} min)",
                flush=True,
            )

    if n_workers == 1:
        for task in tasks:
            handle_result(process_rsa_session(task))
    elif threadpool_limits is None:
        result_iter = Parallel(
            n_jobs=n_workers, backend="loky", verbose=0, return_as="generator_unordered"
        )(delayed(process_rsa_session)(task) for task in tasks)
        for result in result_iter:
            handle_result(result)
    else:
        with threadpool_limits(limits=1):
            result_iter = Parallel(
                n_jobs=n_workers,
                backend="loky",
                verbose=0,
                return_as="generator_unordered",
            )(delayed(process_rsa_session)(task) for task in tasks)
            for result in result_iter:
                handle_result(result)

    fit_df = pd.concat(fit_frames, ignore_index=True) if fit_frames else pd.DataFrame()
    fit_df.to_csv(model_fit_csv, index=False)
    rdm_df = pd.read_csv(rdm_csv) if rdm_csv.exists() else pd.DataFrame()
    cross_day_df = compute_cross_day_geometry_similarity(rdm_df)
    cross_day_df.to_csv(cross_day_csv, index=False)
    if not qc_csv.exists():
        pd.DataFrame(columns=qc_columns).to_csv(qc_csv, index=False)

    fig_result = (
        save_fig_rsa_time_resolved(output_dir=output_dir, figures_dir=figures_dir)
        if save_figures
        else {}
    )
    elapsed = time.time() - t0
    print(f"[RSA stim] Done in {elapsed/60:.1f} min.", flush=True)

    return {
        "rdm_csv": rdm_csv,
        "count_csv": count_csv,
        "model_fit_csv": model_fit_csv,
        "cross_day_csv": cross_day_csv,
        "model_vec_csv": model_vec_csv,
        "qc_csv": qc_csv,
        "model_fit_figure": fig_result.get("model_fit_figure"),
        "snapshot_figure": fig_result.get("snapshot_figure"),
        "cross_day_geometry_figure": fig_result.get("cross_day_geometry_figure"),
        "cross_day_geometry_timecourse_figure": fig_result.get(
            "cross_day_geometry_timecourse_figure"
        ),
        "cross_day_geometry_timecourse_5x5_figure": fig_result.get(
            "cross_day_geometry_timecourse_5x5_figure"
        ),
    }


if __name__ == "__main__":
    run_rsa_time_resolved()
