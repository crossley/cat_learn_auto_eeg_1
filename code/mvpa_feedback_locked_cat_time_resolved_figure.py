#!/usr/bin/env python3
"""Plot feedback-locked time-resolved MVPA figures from saved outputs."""

import os
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

from mvpa_feedback_locked_cat_tg_analysis import FIGURES_DIR, OUTPUT_DIR


def make_haufe_info_from_pos_df(pos_df):
    ch_names = pos_df["channel"].tolist()
    ch_pos = {}
    for _, r in pos_df.iterrows():
        ch_pos[r["channel"]] = np.array([r["x"], r["y"], r["z"]], dtype=float)
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="ignore")
    return info, ch_names


def save_fig_mvpa_feedback_locked_cat_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    day_means_csv = output_dir / "mvpa_feedback_locked_cat_time_resolved_day_means_timecourse.csv"
    haufe_day_mean_csv = output_dir / "mvpa_feedback_locked_cat_tg_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_feedback_locked_cat_tg_haufe_channel_positions.csv"
    fig_day_panels = figures_dir / "mvpa_feedback_locked_cat_auc_by_day_panels.png"

    required_paths = [day_means_csv, haufe_day_mean_csv, haufe_channel_pos_csv]
    missing_paths = [str(p) for p in required_paths if not p.exists()]
    if missing_paths:
        raise FileNotFoundError(
            "Missing feedback time-resolved MVPA outputs in "
            f"{output_dir}. Run mvpa_feedback_locked_cat_tg_analysis.py and "
            "mvpa_feedback_locked_cat_time_resolved_analysis.py first.\n"
            + "\n".join(missing_paths)
        )

    day_means_df = pd.read_csv(day_means_csv)
    if day_means_df.empty:
        raise ValueError(f"Empty feedback time-resolved MVPA output: {day_means_csv}")

    haufe_df = pd.read_csv(haufe_day_mean_csv)
    pos_df = pd.read_csv(haufe_channel_pos_csv)
    if haufe_df.empty or pos_df.empty:
        raise ValueError(
            f"Empty Haufe output table: {haufe_day_mean_csv} or {haufe_channel_pos_csv}"
        )
    haufe_info, haufe_ch_names = make_haufe_info_from_pos_df(pos_df)

    peak_medians = {}
    for day, d_day in day_means_df.groupby("day"):
        d_day = d_day.sort_values("time_sec")
        for lo, hi, label in [(0.0, 0.20, "early"), (0.35, 0.80, "late")]:
            d_win = d_day[(d_day["time_sec"] >= lo) & (d_day["time_sec"] <= hi)]
            if d_win.empty:
                raise ValueError(
                    f"Missing feedback MVPA peak window: "
                    f"day={day}, peak={label}, window={lo}-{hi}"
                )
            peak_medians[(int(day), label)] = float(
                d_win.loc[d_win["auc_mean"].idxmax(), "time_sec"]
            )

    days = sorted(day_means_df["day"].unique())
    missing_days = [str(d) for d in [1, 2, 3, 4, 5] if d not in days]
    if missing_days:
        raise ValueError(
            f"Missing feedback time-resolved MVPA days in {day_means_csv}: "
            + ", ".join(missing_days)
        )

    fig, axes = plt.subplots(1, len(days), figsize=(5 * len(days), 5.2), sharey=True, squeeze=False)
    x_all = day_means_df["time_sec"].to_numpy(dtype=float)
    x_min = float(np.nanmin(x_all))
    x_max = float(np.nanmax(x_all))
    y_upper = float(np.nanmax(day_means_df["auc_mean"] + day_means_df["auc_sem"].fillna(0.0)))
    y_lower = float(np.nanmin(day_means_df["auc_mean"] - day_means_df["auc_sem"].fillna(0.0)))
    y_pad = max(0.02, 0.20 * (y_upper - y_lower))
    topomap_ims = []
    lim = float(np.nanmax(np.abs(haufe_df["pattern_mean"].to_numpy(dtype=float))))
    if not np.isfinite(lim) or lim <= 0:
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
        day_peak_times = []
        for peak_label in ["early", "late"]:
            if (int(day), peak_label) in peak_medians:
                day_peak_times.append((peak_medians[(int(day), peak_label)], peak_label))
            else:
                raise ValueError(
                    f"Missing feedback MVPA peak: day={day}, peak={peak_label}"
                )
        for peak_time, peak_label in day_peak_times:
            if x_max <= x_min:
                raise ValueError(
                    f"Invalid feedback MVPA time axis in {day_means_csv}: "
                    f"x_min={x_min}, x_max={x_max}"
                )
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
            d_day_haufe = haufe_df[haufe_df["day"] == day]
            if d_day_haufe.empty:
                raise ValueError(f"Missing Haufe data for topomap inset: day={day}")
            times = np.sort(d_day_haufe["time_sec"].unique().astype(float))
            t_show = float(times[int(np.argmin(np.abs(times - peak_time)))])
            d_topo = d_day_haufe[np.isclose(d_day_haufe["time_sec"], t_show)]
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
                f"{peak_label}\npeak {peak_time:.3f}s\nmap {t_show:.3f}s", fontsize=7
            )
    axes.ravel()[0].set_ylabel("ROC-AUC")
    fig.suptitle("Time-resolved Category Decoding (Feedback-Locked)")
    if topomap_ims:
        cax = fig.add_axes([0.32, 0.89, 0.36, 0.025])
        fig.colorbar(topomap_ims[-1], cax=cax, orientation="horizontal", label="Haufe pattern")
        fig.tight_layout(rect=[0, 0, 1, 0.78])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(fig_day_panels, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {"figure_paths": {"day_panels": fig_day_panels}}


if __name__ == "__main__":
    save_fig_mvpa_feedback_locked_cat_time_resolved()
