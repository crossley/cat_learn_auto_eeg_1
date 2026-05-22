#!/usr/bin/env python3
"""Shared RSA figure functions."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util_rsa_time_resolved import (
    FIGURES_DIR,
    GEOMETRY_WINDOWS,
    OUTPUT_DIR,
    SNAPSHOT_TIMES,
    _copy_with_backup,
    _restore_backups,
)


def save_cross_day_geometry_figure(
    similarity_df,
    figures_dir: Path | str = FIGURES_DIR,
):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    matrix_fig = figures_dir / "rsa_stim_cross_day_geometry_similarity.png"
    timecourse_fig = figures_dir / "rsa_stim_cross_day_geometry_timecourse.png"
    timecourse_5x5_fig = figures_dir / "rsa_stim_cross_day_geometry_timecourse_5x5.png"
    if similarity_df.empty:
        raise ValueError("Empty RSA cross-day geometry similarity table")

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
        if g.empty:
            raise ValueError(
                "Missing RSA cross-day geometry rows for window "
                f"{_window_name}: {tmin}-{tmax}"
            )
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
        missing_pairs = []
        for train_day in day_grid:
            for test_day in day_grid:
                if train_day == test_day:
                    continue
                i_pair = day_grid.index(train_day)
                j_pair = day_grid.index(test_day)
                if not np.isfinite(mat[i_pair, j_pair]):
                    missing_pairs.append(
                        f"train_day={train_day}, test_day={test_day}"
                    )
        if len(missing_pairs) > 0:
            raise ValueError(
                "Missing RSA cross-day geometry day pairs for window "
                f"{window_name}:\n" + "\n".join(missing_pairs)
            )
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
        x_labels = []
        for d in day_grid:
            x_labels.append(f"D{d}")
        y_labels = []
        for d in day_grid:
            y_labels.append(f"D{d}")
        ax.set_xticklabels(x_labels)
        ax.set_yticklabels(y_labels)
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
        partners = []
        for day in day_grid:
            if day > root_day:
                partners.append(day)
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
                raise ValueError(
                    "Missing RSA cross-day geometry timecourse pair: "
                    f"day_pair=D{root_day}-D{test_day}"
                )
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
    cross_day_csv = output_dir / "rsa_stim_cross_day_geometry_similarity.csv"
    if (not rdm_csv.exists()) or (not model_fit_csv.exists()) or (not cross_day_csv.exists()):
        raise FileNotFoundError(
            "Missing RSA time-resolved outputs. Run run_rsa_time_resolved() first."
        )
    rdm_df = pd.read_csv(rdm_csv)
    fit_df = pd.read_csv(model_fit_csv)
    if rdm_df.empty:
        raise ValueError(f"Empty RSA RDM output table: {rdm_csv}")
    if fit_df.empty:
        raise ValueError(f"Empty RSA model-fit output table: {model_fit_csv}")

    fit_fig = figures_dir / "rsa_stim_model_fit_timecourses.png"
    models = list(fit_df["model"].dropna().unique())
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False, sharex=True)
    for ax, model in zip(axes.ravel(), models):
        g_model = fit_df[fit_df["model"] == model]
        if g_model.empty:
            raise ValueError(f"Missing RSA model rows in {model_fit_csv}: {model}")
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
                raise ValueError(f"Missing RSA RDM times in {rdm_csv}: day={day}")
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
    finite_parts = []
    for mat in snapshot_mats:
        if np.any(np.isfinite(mat)):
            finite_parts.append(mat[np.isfinite(mat)])
    if len(finite_parts) == 0:
        raise ValueError(f"No finite RSA snapshot values in {rdm_csv}")
    finite_snapshot = np.concatenate(finite_parts)
    vmin = float(np.nanquantile(finite_snapshot, 0.02))
    vmax = float(np.nanquantile(finite_snapshot, 0.98))
    im = None
    for i, day in enumerate(days):
        for j, _target_time in enumerate(SNAPSHOT_TIMES):
            ax = axes[i, j]
            _, time_sec, mat = snapshot_items[i * len(SNAPSHOT_TIMES) + j]
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
    fig.colorbar(im, cax=cax, label="Neural dissimilarity")
    fig.savefig(snapshot_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    cross_day_figs = save_cross_day_geometry_figure(
        pd.read_csv(cross_day_csv), figures_dir=figures_dir
    )

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


def save_fig_rsa_windowed(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rdm_csv = output_dir / "rsa_stim_windowed_rdms.csv"
    model_fit_csv = output_dir / "rsa_stim_windowed_model_fit_timecourses.csv"
    cross_day_csv = output_dir / "rsa_stim_windowed_cross_day_geometry_similarity.csv"
    if (not rdm_csv.exists()) or (not model_fit_csv.exists()) or (not cross_day_csv.exists()):
        raise FileNotFoundError(
            "Missing windowed RSA outputs. Run run_rsa_windowed() first."
        )

    original_paths = {
        "fit": figures_dir / "rsa_stim_model_fit_timecourses.png",
        "snapshot": figures_dir / "rsa_stim_neural_rdm_snapshots.png",
        "cross_matrix": figures_dir / "rsa_stim_cross_day_geometry_similarity.png",
        "cross_time": figures_dir / "rsa_stim_cross_day_geometry_timecourse.png",
        "cross_pair": figures_dir / "rsa_stim_cross_day_geometry_timecourse_5x5.png",
    }
    windowed_paths = {
        "fit": figures_dir / "rsa_stim_windowed_model_fit_timecourses.png",
        "snapshot": figures_dir / "rsa_stim_windowed_neural_rdm_snapshots.png",
        "cross_matrix": figures_dir / "rsa_stim_windowed_cross_day_geometry_similarity.png",
        "cross_time": figures_dir / "rsa_stim_windowed_cross_day_geometry_timecourse.png",
        "cross_pair": figures_dir / "rsa_stim_windowed_cross_day_geometry_timecourse_pairs.png",
    }

    temp_rdm = output_dir / "rsa_stim_time_resolved_rdms.csv"
    temp_fit = output_dir / "rsa_stim_model_fit_timecourses.csv"
    temp_cross = output_dir / "rsa_stim_cross_day_geometry_similarity.csv"
    saved_temp = {}
    for path in [temp_rdm, temp_fit, temp_cross]:
        if path.exists():
            backup = path.with_suffix(path.suffix + ".single_time_backup")
            path.replace(backup)
            saved_temp[path] = backup
    rdm_csv.replace(temp_rdm)
    model_fit_csv.replace(temp_fit)
    cross_day_csv.replace(temp_cross)

    try:
        save_fig_rsa_time_resolved(output_dir=output_dir, figures_dir=figures_dir)
        for key, original_path in original_paths.items():
            if original_path.exists():
                if key == "cross_time":
                    original_path.unlink()
                else:
                    original_path.replace(windowed_paths[key])
    finally:
        temp_rdm.replace(rdm_csv)
        temp_fit.replace(model_fit_csv)
        if temp_cross.exists():
            temp_cross.replace(cross_day_csv)
        for path, backup in saved_temp.items():
            backup.replace(path)
        if saved_temp:
            save_fig_rsa_time_resolved(output_dir=output_dir, figures_dir=figures_dir)

    return {
        "model_fit_figure": windowed_paths["fit"],
        "snapshot_figure": windowed_paths["snapshot"],
        "cross_day_geometry_figure": windowed_paths["cross_matrix"],
        "cross_day_geometry_timecourse_figure": None,
        "cross_day_geometry_timecourse_5x5_figure": windowed_paths["cross_pair"],
    }


def _save_fig_rsa_prefixed(
    output_dir,
    figures_dir,
    data_prefix,
    figure_prefix,
    windowed=False,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    if windowed:
        rdm_csv = output_dir / f"{data_prefix}_rdms.csv"
    else:
        rdm_csv = output_dir / f"{data_prefix}_time_resolved_rdms.csv"
    model_fit_csv = output_dir / f"{data_prefix}_model_fit_timecourses.csv"
    cross_day_csv = output_dir / f"{data_prefix}_cross_day_geometry_similarity.csv"
    if (not rdm_csv.exists()) or (not model_fit_csv.exists()) or (not cross_day_csv.exists()):
        raise FileNotFoundError(f"Missing RSA outputs for prefix {data_prefix}.")

    temp_paths = {
        "rdm": output_dir / "rsa_stim_time_resolved_rdms.csv",
        "fit": output_dir / "rsa_stim_model_fit_timecourses.csv",
        "cross": output_dir / "rsa_stim_cross_day_geometry_similarity.csv",
    }
    original_figs = {
        "fit": figures_dir / "rsa_stim_model_fit_timecourses.png",
        "snapshot": figures_dir / "rsa_stim_neural_rdm_snapshots.png",
        "cross_matrix": figures_dir / "rsa_stim_cross_day_geometry_similarity.png",
        "cross_time": figures_dir / "rsa_stim_cross_day_geometry_timecourse.png",
        "cross_pair": figures_dir / "rsa_stim_cross_day_geometry_timecourse_5x5.png",
    }
    target_figs = {
        "fit": figures_dir / f"{figure_prefix}_model_fit_timecourses.png",
        "snapshot": figures_dir / f"{figure_prefix}_neural_rdm_snapshots.png",
        "cross_matrix": figures_dir / f"{figure_prefix}_cross_day_geometry_similarity.png",
        "cross_time": figures_dir / f"{figure_prefix}_cross_day_geometry_timecourse.png",
        "cross_pair": figures_dir / f"{figure_prefix}_cross_day_geometry_timecourse_pairs.png",
    }

    backups = {}
    fig_backups = {}
    try:
        _copy_with_backup(rdm_csv, temp_paths["rdm"], backups)
        _copy_with_backup(model_fit_csv, temp_paths["fit"], backups)
        _copy_with_backup(cross_day_csv, temp_paths["cross"], backups)
        for path in original_figs.values():
            if path.exists():
                backup = path.with_suffix(path.suffix + ".rsa_backup")
                path.replace(backup)
                fig_backups[path] = backup
        save_fig_rsa_time_resolved(output_dir=output_dir, figures_dir=figures_dir)
        for key, original_path in original_figs.items():
            if original_path.exists():
                if key == "cross_time":
                    original_path.unlink()
                else:
                    original_path.replace(target_figs[key])
    finally:
        _restore_backups(backups)
        for path, backup in fig_backups.items():
            if path.exists():
                path.unlink()
            backup.replace(path)

    return {
        "model_fit_figure": target_figs["fit"],
        "snapshot_figure": target_figs["snapshot"],
        "cross_day_geometry_figure": target_figs["cross_matrix"],
        "cross_day_geometry_timecourse_figure": None,
        "cross_day_geometry_timecourse_5x5_figure": target_figs["cross_pair"],
    }


def save_fig_rsa_feedback_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    return _save_fig_rsa_prefixed(
        output_dir=output_dir,
        figures_dir=figures_dir,
        data_prefix="rsa_feedback",
        figure_prefix="rsa_feedback",
    )


def save_fig_rsa_feedback_windowed(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    return _save_fig_rsa_prefixed(
        output_dir=output_dir,
        figures_dir=figures_dir,
        data_prefix="rsa_feedback_windowed",
        figure_prefix="rsa_feedback_windowed",
        windowed=True,
    )
