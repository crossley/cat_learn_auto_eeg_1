#!/usr/bin/env python3
"""Plot stimulus-locked time-resolved MVPA figures from saved outputs."""

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

from mvpa_stim_locked_cat_time_resolved_analysis import FIGURES_DIR, OUTPUT_DIR


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
    ch_pos = {}
    for _, r in pos_df.iterrows():
        ch_pos[r["channel"]] = np.array([r["x"], r["y"], r["z"]], dtype=float)
    info = mne.create_info(ch_names=ch_names, sfreq=128.0, ch_types="eeg")
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info.set_montage(montage, on_missing="ignore")
    return info, ch_names


def plot_haufe_similarity_day_pairs(peak_df, haufe_day_mean_df, figures_dir):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "mvpa_stim_locked_cat_haufe_similarity_timegen_matrices_5x5.png"

    if haufe_day_mean_df.empty:
        raise ValueError("Empty Haufe day-mean table for pattern-similarity figure")

    day_grid = [1, 2, 3, 4, 5]
    channel_order = sorted(haufe_day_mean_df["channel"].dropna().unique().tolist())
    similarity_maps = {}
    time_maps = {}
    for day in day_grid:
        d_day = haufe_day_mean_df[haufe_day_mean_df["day"] == day].copy()
        if d_day.empty:
            raise ValueError(f"Missing Haufe day in day-mean table: day={day}")
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
    if peak_df.empty:
        raise ValueError("Empty peak-time table for Haufe pattern-similarity figure")
    peak_summary = peak_df.groupby(["day", "peak"], as_index=False)[
        "peak_time_sec"
    ].median()
    for _, r in peak_summary.iterrows():
        peak_medians[(int(r["day"]), str(r["peak"]))] = float(r["peak_time_sec"])

    fig, axes = plt.subplots(5, 5, figsize=(18.0, 16.0), squeeze=False)
    im = None
    for i, train_day in enumerate(day_grid):
        for j, test_day in enumerate(day_grid):
            ax = axes[i, j]
            train_times = time_maps.get(train_day)
            test_times = time_maps.get(test_day)
            if train_times is None or test_times is None:
                raise ValueError(
                    "Missing Haufe time map for day pair: "
                    f"train_day={train_day}, test_day={test_day}"
                )
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
                ax.set_ylabel(f"Train-day time on D{train_day} (s)", fontsize=9)
            if i == 4:
                ax.set_xlabel("Test-day time (s)")
            else:
                ax.set_xticklabels([])
            if j != 0:
                ax.set_yticklabels([])

    fig.suptitle("Haufe Pattern Similarity by Day Pair (A/B)", y=0.98)
    fig.subplots_adjust(top=0.94, bottom=0.06, left=0.06, right=0.90, wspace=0.26, hspace=0.36)
    cax = fig.add_axes([0.92, 0.14, 0.015, 0.72])
    fig.colorbar(im, cax=cax, label="Pattern correlation")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_mvpa_stim_locked_cat_time_resolved(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    session_csv = output_dir / "mvpa_stim_locked_cat_session_timecourse.csv"
    day_means_csv = output_dir / "mvpa_stim_locked_cat_day_means_timecourse.csv"
    haufe_day_mean_csv = output_dir / "mvpa_stim_locked_cat_haufe_day_mean_channel_time.csv"
    haufe_channel_pos_csv = output_dir / "mvpa_stim_locked_cat_haufe_channel_positions.csv"
    haufe_peak_times_csv = output_dir / "mvpa_stim_locked_cat_haufe_subject_day_peak_times.csv"
    fig_day_panels = figures_dir / "mvpa_stim_locked_cat_auc_by_day_panels.png"

    required_paths = [
        session_csv,
        day_means_csv,
        haufe_day_mean_csv,
        haufe_channel_pos_csv,
    ]
    missing_paths = []
    for path in required_paths:
        if not path.exists():
            missing_paths.append(str(path))
    if len(missing_paths) > 0:
        raise FileNotFoundError(
            "Missing MVPA outputs in "
            f"{output_dir}. Run mvpa_stim_locked_cat_time_resolved_analysis.py first.\n"
            + "\n".join(missing_paths)
        )

    day_means_df = pd.read_csv(day_means_csv)
    if day_means_df.empty:
        raise ValueError(f"Empty stimulus time-resolved MVPA output: {day_means_csv}")

    def detect_subject_day_peak_times():
        d_session = pd.read_csv(session_csv)
        if d_session.empty:
            raise ValueError(f"Empty stimulus MVPA session table: {session_csv}")
        rows = []
        for (subject, day), d_sd in d_session.groupby(["subject", "day"]):
            d_sd = d_sd.sort_values("time_sec")
            for lo, hi, label in [(0.0, 0.20, "early"), (0.35, 0.80, "late")]:
                d_win = d_sd[(d_sd["time_sec"] >= lo) & (d_sd["time_sec"] <= hi)]
                if d_win.empty:
                    raise ValueError(
                        f"Missing MVPA peak window in {session_csv}: "
                        f"subject={subject}, day={day}, peak={label}, "
                        f"window={lo}-{hi}"
                    )
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

    haufe_df = pd.DataFrame()
    haufe_info = None
    haufe_ch_names = []
    peak_df = pd.DataFrame()
    peak_medians = {}
    pos_df = pd.DataFrame()
    haufe_df = pd.read_csv(haufe_day_mean_csv)
    pos_df = pd.read_csv(haufe_channel_pos_csv)
    if haufe_df.empty or pos_df.empty:
        raise ValueError(
            f"Empty Haufe output table: {haufe_day_mean_csv} "
            f"or {haufe_channel_pos_csv}"
        )
    haufe_info, haufe_ch_names = make_haufe_info_from_pos_df(pos_df)
    peak_df = detect_subject_day_peak_times()
    d_peak_median = (
        peak_df.groupby(["day", "peak"], as_index=False)["peak_time_sec"]
        .median()
        .rename(columns={"peak_time_sec": "median_peak_time_sec"})
    )
    peak_medians = {}
    for _, r in d_peak_median.iterrows():
        peak_medians[(int(r["day"]), str(r["peak"]))] = float(
            r["median_peak_time_sec"]
        )

    haufe_similarity_path = plot_haufe_similarity_day_pairs(
        peak_df, haufe_df, figures_dir
    )

    days = sorted(day_means_df["day"].unique())
    missing_days = []
    for day in [1, 2, 3, 4, 5]:
        if day not in days:
            missing_days.append(str(day))
    if len(missing_days) > 0:
        raise ValueError(
            f"Missing stimulus time-resolved MVPA days in {day_means_csv}: "
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
                    f"Missing stimulus MVPA peak median: day={day}, "
                    f"peak={peak_label}"
                )
        for peak_time, peak_label in day_peak_times:
            if x_max <= x_min:
                raise ValueError(
                    f"Invalid stimulus MVPA time axis in {day_means_csv}: "
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
            d_day = haufe_df[haufe_df["day"] == day]
            if d_day.empty:
                raise ValueError(f"Missing Haufe data for topomap inset: day={day}")
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

    return {
        "figure_paths": {
            "day_panels": fig_day_panels,
            "haufe_similarity_day_pairs": haufe_similarity_path,
        },
    }




if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_time_resolved()
