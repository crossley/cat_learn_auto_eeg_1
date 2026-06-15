#!/usr/bin/env python3
"""Plot ERP-like time-frequency figures from saved analysis outputs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from presentation_figure import DAY_COLORS, setup_axis
from sequence_feature_interface import FIGURES_DIR, OUTPUT_DIR
from time_frequency_erp_like_analysis import PREFIX, run_time_frequency_erp_like


def require_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing time-frequency ERP-like output: {path}. "
            "Run time_frequency_erp_like_analysis.py first."
        )
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty time-frequency ERP-like output: {path}")
    return df


def _pivot_tf(df: pd.DataFrame, value_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = df.groupby(["freq_hz", "time_s"], as_index=False)[value_col].mean()
    freqs = np.sort(grouped["freq_hz"].unique().astype(float))
    times = np.sort(grouped["time_s"].unique().astype(float))
    mat = grouped.pivot(index="freq_hz", columns="time_s", values=value_col)
    mat = mat.reindex(index=freqs, columns=times).to_numpy(dtype=float)
    return freqs, times, mat


def _tf_extent(times: np.ndarray, freqs: np.ndarray) -> list[float]:
    return [float(times.min()), float(times.max()), float(freqs.min()), float(freqs.max())]


def save_condition_tf_maps(day_condition_df: pd.DataFrame, figures_dir: Path) -> Path:
    days = sorted(day_condition_df["day"].dropna().unique().astype(int))
    conditions = ["cat_a", "cat_b"]
    vmax = np.nanpercentile(np.abs(day_condition_df["power_mean"].to_numpy(dtype=float)), 98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(
        len(conditions),
        len(days),
        figsize=(3.8 * len(days), 5.8),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    im = None
    for r, condition in enumerate(conditions):
        for c, day in enumerate(days):
            ax = axes[r, c]
            df = day_condition_df[
                (day_condition_df["day"] == day) & (day_condition_df["condition"] == condition)
            ]
            if df.empty:
                ax.set_axis_off()
                continue
            freqs, times, mat = _pivot_tf(df, "power_mean")
            im = ax.imshow(
                np.ma.masked_invalid(mat),
                origin="lower",
                aspect="auto",
                extent=_tf_extent(times, freqs),
                cmap="RdBu_r",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.axvline(0.0, color="0.15", linewidth=0.8)
            ax.set_title(f"D{day}" if r == 0 else "")
            if c == 0:
                ax.set_ylabel(f"{condition}\nfrequency (Hz)")
            if r == len(conditions) - 1:
                ax.set_xlabel("time from stimulus (s)")
    fig.suptitle("Time-Frequency Power by Category and Day")
    fig.subplots_adjust(right=0.92, wspace=0.12, hspace=0.18)
    if im is not None:
        cax = fig.add_axes([0.94, 0.18, 0.014, 0.64])
        fig.colorbar(im, cax=cax, label="baseline-corrected power")
    path = figures_dir / f"{PREFIX}_condition_tf_maps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{PREFIX}] wrote {path}", flush=True)
    return path


def save_difference_tf_maps(day_condition_df: pd.DataFrame, figures_dir: Path) -> Path:
    keys = ["day", "practice_stage", "channel", "freq_hz", "time_s"]
    wide = day_condition_df.pivot_table(index=keys, columns="condition", values="power_mean").reset_index()
    if "cat_a" not in wide or "cat_b" not in wide:
        raise ValueError("Cannot plot A-B maps without both category conditions")
    wide["diff"] = wide["cat_a"] - wide["cat_b"]
    days = sorted(wide["day"].dropna().unique().astype(int))
    vmax = np.nanpercentile(np.abs(wide["diff"].to_numpy(dtype=float)), 98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, len(days), figsize=(3.8 * len(days), 3.2), sharex=True, sharey=True, squeeze=False)
    im = None
    for ax, day in zip(axes[0], days):
        df = wide[wide["day"] == day]
        freqs, times, mat = _pivot_tf(df, "diff")
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            origin="lower",
            aspect="auto",
            extent=_tf_extent(times, freqs),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axvline(0.0, color="0.15", linewidth=0.8)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("time from stimulus (s)")
    axes[0, 0].set_ylabel("frequency (Hz)")
    fig.suptitle("A Minus B Time-Frequency Difference")
    fig.subplots_adjust(right=0.91, wspace=0.14)
    if im is not None:
        cax = fig.add_axes([0.93, 0.18, 0.016, 0.64])
        fig.colorbar(im, cax=cax, label="A-B power")
    path = figures_dir / f"{PREFIX}_a_minus_b_tf_maps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{PREFIX}] wrote {path}", flush=True)
    return path


def save_change_tf_maps(day_condition_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    keys = ["day", "channel", "freq_hz", "time_s"]
    wide = day_condition_df.pivot_table(index=keys, columns="condition", values="power_mean").reset_index()
    if "cat_a" not in wide or "cat_b" not in wide or 1 not in set(wide["day"].astype(int)):
        return None
    wide["diff"] = wide["cat_a"] - wide["cat_b"]
    base = wide[wide["day"] == 1].groupby(["channel", "freq_hz", "time_s"], as_index=False)["diff"].mean()
    rows = []
    for day in sorted(wide["day"].dropna().unique().astype(int)):
        if day == 1:
            continue
        current = wide[wide["day"] == day].groupby(["channel", "freq_hz", "time_s"], as_index=False)["diff"].mean()
        merged = current.merge(base, on=["channel", "freq_hz", "time_s"], suffixes=("", "_day1"))
        merged["change"] = merged["diff"] - merged["diff_day1"]
        merged["day"] = day
        rows.append(merged)
    if not rows:
        return None
    change_df = pd.concat(rows, ignore_index=True)
    days = sorted(change_df["day"].unique().astype(int))
    vmax = np.nanpercentile(np.abs(change_df["change"].to_numpy(dtype=float)), 98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    fig, axes = plt.subplots(1, len(days), figsize=(3.8 * len(days), 3.2), sharex=True, sharey=True, squeeze=False)
    im = None
    for ax, day in zip(axes[0], days):
        df = change_df[change_df["day"] == day]
        freqs, times, mat = _pivot_tf(df, "change")
        im = ax.imshow(
            np.ma.masked_invalid(mat),
            origin="lower",
            aspect="auto",
            extent=_tf_extent(times, freqs),
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.axvline(0.0, color="0.15", linewidth=0.8)
        ax.set_title(f"D{day} - D1")
        ax.set_xlabel("time from stimulus (s)")
    axes[0, 0].set_ylabel("frequency (Hz)")
    fig.suptitle("Session Change in A-B Time-Frequency Difference")
    fig.subplots_adjust(right=0.91, wspace=0.14)
    if im is not None:
        cax = fig.add_axes([0.93, 0.18, 0.016, 0.64])
        fig.colorbar(im, cax=cax, label="change in A-B power")
    path = figures_dir / f"{PREFIX}_session_change_tf_maps.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{PREFIX}] wrote {path}", flush=True)
    return path


def save_band_timecourses(timecourse_day_df: pd.DataFrame, figures_dir: Path) -> Path:
    rois = [roi for roi in ["visual", "central", "frontal", "sensorwide"] if roi in set(timecourse_day_df["roi"])]
    bands = sorted(timecourse_day_df["band"].dropna().unique().tolist())
    if not rois or not bands:
        raise ValueError("No ROI/band time-course rows available")
    roi = rois[0]
    fig, axes = plt.subplots(len(bands), 1, figsize=(8.4, 2.8 * len(bands)), sharex=True, squeeze=False)
    for ax, band in zip(axes[:, 0], bands):
        for day in sorted(timecourse_day_df["day"].dropna().unique().astype(int)):
            for condition, linestyle in [("cat_a", "-"), ("cat_b", "--")]:
                df = timecourse_day_df[
                    (timecourse_day_df["roi"] == roi)
                    & (timecourse_day_df["band"] == band)
                    & (timecourse_day_df["day"] == day)
                    & (timecourse_day_df["condition"] == condition)
                ].sort_values("time_s")
                if df.empty:
                    continue
                color = DAY_COLORS.get(int(day), "0.35")
                x = df["time_s"].to_numpy(dtype=float)
                y = df["power_mean_mean"].to_numpy(dtype=float)
                err = df["power_mean_sem"].to_numpy(dtype=float)
                ax.plot(x, y, color=color, linestyle=linestyle, linewidth=1.7, label=f"D{day} {condition}")
                good = np.isfinite(err)
                if np.any(good):
                    ax.fill_between(x[good], y[good] - err[good], y[good] + err[good], color=color, alpha=0.08, linewidth=0)
        ax.axhline(0.0, color="0.25", linewidth=0.8)
        ax.axvline(0.0, color="0.25", linewidth=0.8)
        ax.set_ylabel(f"{band}\npower")
        setup_axis(ax)
    axes[-1, 0].set_xlabel("time from stimulus (s)")
    axes[0, 0].legend(frameon=False, ncol=3, fontsize=7, loc="upper right")
    fig.suptitle(f"Band-Limited Power Time Courses: {roi}")
    fig.tight_layout()
    path = figures_dir / f"{PREFIX}_band_power_timecourses.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{PREFIX}] wrote {path}", flush=True)
    return path


def save_cluster_summary(day_contrast_df: pd.DataFrame, figures_dir: Path) -> Path:
    df = day_contrast_df[day_contrast_df["summary_level"] == "strict_roi"].copy()
    if df.empty:
        raise ValueError("No strict ROI contrast rows available")
    df["cell"] = df["band"].astype(str) + " / " + df["window"].astype(str)
    rows = sorted(df["roi"].unique().tolist())
    cols = sorted(df["cell"].unique().tolist())
    days = sorted(df["day"].dropna().unique().astype(int))
    fig, axes = plt.subplots(1, len(days), figsize=(4.0 * len(days), 3.8), sharey=True, squeeze=False)
    vmax = np.nanpercentile(np.abs(df["power_diff_mean"].to_numpy(dtype=float)), 98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = None
    for ax, day in zip(axes[0], days):
        mat_df = df[df["day"] == day].pivot_table(index="roi", columns="cell", values="power_diff_mean")
        mat = mat_df.reindex(index=rows, columns=cols).to_numpy(dtype=float)
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Day {day}")
        ax.set_xticks(np.arange(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(np.arange(len(rows)))
        ax.set_yticklabels(rows)
    fig.suptitle("Strict ROI A-B Window Summaries")
    fig.subplots_adjust(right=0.91, bottom=0.26, wspace=0.16)
    if im is not None:
        cax = fig.add_axes([0.93, 0.24, 0.016, 0.56])
        fig.colorbar(im, cax=cax, label="A-B power")
    path = figures_dir / f"{PREFIX}_strict_roi_window_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{PREFIX}] wrote {path}", flush=True)
    return path


def save_electrode_summary(day_contrast_df: pd.DataFrame, figures_dir: Path) -> Path | None:
    df = day_contrast_df[day_contrast_df["summary_level"] == "electrode"].copy()
    if df.empty:
        return None
    bands = sorted(df["band"].unique().tolist())
    windows = sorted(df["window"].unique().tolist())
    band = "alpha" if "alpha" in bands else bands[0]
    window = "late" if "late" in windows else windows[-1]
    df = df[(df["band"] == band) & (df["window"] == window)]
    days = sorted(df["day"].dropna().unique().astype(int))
    electrodes = sorted(df["roi"].unique().tolist())
    fig, axes = plt.subplots(1, len(days), figsize=(3.6 * len(days), 6.8), sharey=True, squeeze=False)
    vmax = np.nanpercentile(np.abs(df["power_diff_mean"].to_numpy(dtype=float)), 98)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    im = None
    for ax, day in zip(axes[0], days):
        mat_df = df[df["day"] == day].set_index("roi")
        mat = mat_df.reindex(electrodes)["power_diff_mean"].to_numpy(dtype=float)[:, None]
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(f"Day {day}")
        ax.set_xticks([0])
        ax.set_xticklabels([f"{band} {window}"])
        ax.set_yticks(np.arange(len(electrodes)))
        ax.set_yticklabels(electrodes, fontsize=6)
    fig.suptitle("Electrode-Level A-B Window Summary")
    fig.subplots_adjust(right=0.90, wspace=0.12)
    if im is not None:
        cax = fig.add_axes([0.92, 0.18, 0.016, 0.66])
        fig.colorbar(im, cax=cax, label="A-B power")
    path = figures_dir / f"{PREFIX}_electrode_window_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[{PREFIX}] wrote {path}", flush=True)
    return path


def save_fig_time_frequency_erp_like(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    run_analysis: bool = False,
    smoke: bool = False,
    max_sessions: int | None = None,
) -> dict[str, Path | None]:
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    if run_analysis:
        run_time_frequency_erp_like(output_dir=output_dir, smoke=smoke, max_sessions=max_sessions)

    day_condition = require_csv(output_dir / f"{PREFIX}_day_condition_channel_freq_time.csv")
    day_contrast = require_csv(output_dir / f"{PREFIX}_day_window_band_roi_contrast.csv")
    timecourse_day = require_csv(output_dir / f"{PREFIX}_day_band_roi_timecourse.csv")
    paths = {
        "condition_tf_maps": save_condition_tf_maps(day_condition, figures_dir),
        "a_minus_b_tf_maps": save_difference_tf_maps(day_condition, figures_dir),
        "session_change_tf_maps": save_change_tf_maps(day_condition, figures_dir),
        "band_power_timecourses": save_band_timecourses(timecourse_day, figures_dir),
        "strict_roi_window_summary": save_cluster_summary(day_contrast, figures_dir),
        "electrode_window_summary": save_electrode_summary(day_contrast, figures_dir),
    }
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-analysis", action="store_true", help="Run analysis before plotting.")
    parser.add_argument("--smoke", action="store_true", help="Use the analysis smoke path when --run-analysis is set.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Limit sessions when --run-analysis is set.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--figures-dir", type=Path, default=FIGURES_DIR)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    save_fig_time_frequency_erp_like(
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        run_analysis=args.run_analysis,
        smoke=args.smoke,
        max_sessions=args.max_sessions,
    )
