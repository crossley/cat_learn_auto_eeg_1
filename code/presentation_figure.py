#!/usr/bin/env python3
"""Create presentation-focused figures from existing analysis outputs."""

from pathlib import Path
import os
import re
import shutil

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Circle, Polygon
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import find_peaks

from erp_grand_average_figure import require_evoked_map

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
BEHAVIOURAL_DIR = PROJECT_DIR / "Behavioural"
BEHAV_RE = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")
DAYS = [1, 2, 3, 4, 5]
DAY_COLORS = {
    1: "#440154",
    2: "#3b528b",
    3: "#21918c",
    4: "#5ec962",
    5: "#fde725",
}


def require_csv(path, message):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing {message}: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty {message}: {path}")
    return d


def require_csv_any(paths, message):
    tried = []
    for path in paths:
        tried.append(str(path))
        if Path(path).exists():
            return require_csv(path, message)
    raise FileNotFoundError(
        f"Missing {message}. Tried:\n" + "\n".join(tried)
    )


def sem(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) <= 1:
        return np.nan
    return float(np.std(arr, ddof=1) / np.sqrt(len(arr)))


def corr_text(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 3:
        return "r = n/a"
    x = x[good] - float(np.mean(x[good]))
    y = y[good] - float(np.mean(y[good]))
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return "r = n/a"
    return f"r = {float(np.sum(x * y) / denom):.2f}"


def setup_axis(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_presentation_erp_stim(output_dir, figures_dir):
    d = require_csv(
        output_dir / "erp_grand_average_by_day_lock_condition.csv",
        "ERP grand-average output",
    )
    evoked_map = require_evoked_map(d, "stim", "all")
    days_sorted = sorted(evoked_map.keys())
    if len(days_sorted) == 0:
        raise ValueError("No stim/all ERP evoked data available")
    font_context = {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
    }
    fig_path = figures_dir / "presentation_erp_stim_all.png"
    with plt.rc_context(font_context):
        # 2×3: [0,0] = sensor colormap; [0,1],[0,2],[1,0],[1,1],[1,2] = days 1-5
        fig, axes = plt.subplots(2, 3, figsize=(13, 7), squeeze=False)
        # Adjust first so parent_bbox is accurate when we position the sensor inset
        fig.subplots_adjust(
            left=0.06, right=0.99, bottom=0.10, top=0.87,
            wspace=0.42, hspace=0.35,
        )

        # Panel [0,0]: sensor colormap only — plot Day 1 to capture inset, then clear ERP
        ax_cmap = axes[0, 0]
        parent_bbox = ax_cmap.get_position()
        axes_before = set(id(a) for a in fig.axes)
        evoked_map[days_sorted[0]].plot(
            axes=ax_cmap,
            show=False,
            spatial_colors=True,
            titles="",
        )
        sensor_axes = [a for a in fig.axes if id(a) not in axes_before]
        ax_cmap.cla()
        ax_cmap.axis("off")
        for s_ax in sensor_axes:
            # Remove AnchoredSizeLocator so set_position is honoured
            s_ax.set_axes_locator(None)
            title_room = 0.04
            margin = 0.005
            s_ax.set_position([
                parent_bbox.x0 + margin,
                parent_bbox.y0 + margin,
                parent_bbox.width - 2 * margin,
                parent_bbox.height - title_room - 2 * margin,
            ])
            # Scale up channel dots (designed for tiny inset, now in a full panel)
            for col in s_ax.collections:
                col.set_sizes([120.0])

        # Panels [0,1],[0,2],[1,0],[1,1],[1,2]: one ERP panel per day, no colormap inset
        erp_positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
        for (row, col), day in zip(erp_positions, days_sorted):
            ax = axes[row, col]
            axes_before = set(id(a) for a in fig.axes)
            evoked_map[day].plot(
                axes=ax,
                show=False,
                spatial_colors=True,
                titles=f"Day {day}",
            )
            ax.set_title(f"Day {day}")
            ax.set_xlim(-0.1, 0.8)
            for txt in list(ax.texts):
                if "ave" in txt.get_text():
                    txt.remove()
            new_axes = [a for a in fig.axes if id(a) not in axes_before]
            for child_ax in new_axes:
                child_ax.remove()

        fig.suptitle("Stimulus-Locked ERPs", fontsize=14)
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return fig_path


def plot_presentation_erp_gfp(output_dir, figures_dir):
    d = require_csv(
        output_dir / "erp_grand_average_subject_day_all.csv",
        "ERP subject-level output",
    )
    d = d[(d["lock_type"] == "stim") & (d["condition"] == "all")].copy()
    if d.empty:
        raise ValueError("No stim/all rows in subject-level ERP data")

    # GFP = std across channels per subject/day/time
    gfp_rows = []
    for (subject, day, time_s), g in d.groupby(["subject", "day", "time_s"]):
        vals = g["amplitude_v"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) < 2:
            continue
        gfp_rows.append({
            "subject": int(subject),
            "day": int(day),
            "time_s": float(time_s),
            "gfp": float(np.std(vals, ddof=1)),
        })
    gfp_df = pd.DataFrame(gfp_rows)
    if gfp_df.empty:
        raise ValueError("Could not compute GFP from subject-level data")

    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for day in DAYS:
        d_day = gfp_df[gfp_df["day"] == day]
        summary = (
            d_day.groupby("time_s")["gfp"]
            .agg(mean="mean", sem=lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))))
            .reset_index()
            .sort_values("time_s")
        )
        t = summary["time_s"].to_numpy(float)
        y = summary["mean"].to_numpy(float)
        y_sem = summary["sem"].to_numpy(float)
        color = DAY_COLORS[day]
        ax.plot(t, y * 1e6, color=color, linewidth=2.0, label=f"Day {day}")
        ax.fill_between(
            t,
            (y - y_sem) * 1e6,
            (y + y_sem) * 1e6,
            color=color,
            alpha=0.18,
            linewidth=0,
        )

    ax.axvline(0, color="0.3", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.1, 0.8)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("GFP (µV)")
    ax.set_title("Stimulus-Locked Global Field Power")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_erp_gfp.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_presentation_connect_overlay(output_dir, figures_dir):
    active = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_active_pairs.csv",
        "connectivity active-pair output",
    )
    carpet = require_csv(
        output_dir / "sensorwide_carpet_timeseries.csv",
        "connectivity carpet output",
    )
    subject = require_csv(
        output_dir / "sensorwide_carpet_subject_timeseries.csv",
        "connectivity subject carpet output",
    )
    active = active[np.isclose(active["active_pct"].astype(float), 0.10)].copy()
    if active.empty:
        raise ValueError("Missing active-pair rows for active_pct=0.10")
    pair_labels = set(active["pair_label"].tolist())
    d = carpet[
        (carpet["lock_type"] == "stim")
        & (carpet["band"] == "broadband")
        & (carpet["ch_i"] + "--" + carpet["ch_j"]).isin(pair_labels)
    ].copy()
    if d.empty:
        raise ValueError("No top-10% stim/broadband connectivity rows found")
    d_subject = subject[
        (subject["lock_type"] == "stim")
        & (subject["band"] == "broadband")
        & (subject["ch_i"] + "--" + subject["ch_j"]).isin(pair_labels)
    ].copy()
    if d_subject.empty:
        raise ValueError("No top-10% subject connectivity rows found")
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for day in DAYS:
        d_day = d[d["day"] == day]
        rows = []
        for time_s in sorted(d_day["lock_time"].drop_duplicates().tolist()):
            vals = d_day[d_day["lock_time"] == time_s]["conn_val"].to_numpy(float)
            g_time = d_subject[
                (d_subject["day"] == day)
                & np.isclose(d_subject["lock_time"].astype(float), float(time_s))
            ]
            subject_vals = []
            for subject_id, g_subject in g_time.groupby("subject"):
                vals_sub = g_subject["conn_val"].to_numpy(float)
                subject_vals.append(float(np.nanmean(vals_sub)))
            rows.append(
                {
                    "time": time_s,
                    "mean": float(np.nanmean(vals)),
                    "sem": sem(subject_vals),
                }
            )
        plot_df = pd.DataFrame(rows)
        x = plot_df["time"].to_numpy(float)
        y = plot_df["mean"].to_numpy(float)
        y_sem = plot_df["sem"].to_numpy(float)
        ax.plot(
            x,
            y,
            color=DAY_COLORS[day],
            linewidth=2.0,
            label=f"D{day}",
        )
        ax.fill_between(
            x,
            y - y_sem,
            y + y_sem,
            color=DAY_COLORS[day],
            alpha=0.13,
            linewidth=0,
        )
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("mean connectivity")
    ax.set_title("Functional Connectivity Time Series")
    ax.legend(frameon=False, ncol=1, loc="upper right")
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_connectivity_top10_overlay.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_presentation_connect_decomposition_overlay(output_dir, figures_dir):
    active = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_active_pairs.csv",
        "connectivity active-pair output",
    )
    carpet = require_csv(
        output_dir / "sensorwide_carpet_timeseries.csv",
        "connectivity carpet output",
    )
    subject = require_csv(
        output_dir / "sensorwide_carpet_subject_timeseries.csv",
        "connectivity subject carpet output",
    )
    metric_specs = [
        ("imcoh_abs", "Abs ImCoh", "non-zero-lag coupling pattern"),
        ("coh_abs", "Coherence magnitude", "strength-like component"),
        ("phase_lag_factor", "Abs sin(phase lag)", "phase-lag contribution"),
    ]
    missing = []
    for metric, _ylabel, _title in metric_specs:
        if metric not in carpet.columns:
            missing.append(f"sensorwide_carpet_timeseries.csv:{metric}")
        if metric not in subject.columns:
            missing.append(f"sensorwide_carpet_subject_timeseries.csv:{metric}")
    if missing:
        raise ValueError(
            "Missing connectivity decomposition columns. Re-run "
            "code/connect_sensorwide_analysis.py before making this figure. "
            "Missing: " + ", ".join(missing)
        )

    active = active[np.isclose(active["active_pct"].astype(float), 0.10)].copy()
    if active.empty:
        raise ValueError("Missing active-pair rows for active_pct=0.10")
    pair_labels = set(active["pair_label"].tolist())

    d = carpet[
        (carpet["lock_type"] == "stim")
        & (carpet["band"] == "broadband")
        & (carpet["ch_i"] + "--" + carpet["ch_j"]).isin(pair_labels)
    ].copy()
    if d.empty:
        raise ValueError("No top-10% stim/broadband connectivity rows found")
    d_subject = subject[
        (subject["lock_type"] == "stim")
        & (subject["band"] == "broadband")
        & (subject["ch_i"] + "--" + subject["ch_j"]).isin(pair_labels)
    ].copy()
    if d_subject.empty:
        raise ValueError("No top-10% subject connectivity rows found")

    fig, axes = plt.subplots(
        len(metric_specs),
        1,
        figsize=(7.8, 7.2),
        sharex=True,
        constrained_layout=True,
    )
    for ax, (metric, ylabel, title) in zip(axes, metric_specs):
        for day in DAYS:
            d_day = d[d["day"] == day]
            rows = []
            for time_s in sorted(d_day["lock_time"].drop_duplicates().tolist()):
                vals = d_day[d_day["lock_time"] == time_s][metric].to_numpy(float)
                g_time = d_subject[
                    (d_subject["day"] == day)
                    & np.isclose(d_subject["lock_time"].astype(float), float(time_s))
                ]
                subject_vals = []
                for _subject_id, g_subject in g_time.groupby("subject"):
                    vals_sub = g_subject[metric].to_numpy(float)
                    subject_vals.append(float(np.nanmean(vals_sub)))
                rows.append(
                    {
                        "time": time_s,
                        "mean": float(np.nanmean(vals)),
                        "sem": sem(subject_vals),
                    }
                )
            plot_df = pd.DataFrame(rows)
            x = plot_df["time"].to_numpy(float)
            y = plot_df["mean"].to_numpy(float)
            y_sem = plot_df["sem"].to_numpy(float)
            ax.plot(
                x,
                y,
                color=DAY_COLORS[day],
                linewidth=1.9,
                label=f"D{day}",
            )
            ax.fill_between(
                x,
                y - y_sem,
                y + y_sem,
                color=DAY_COLORS[day],
                alpha=0.12,
                linewidth=0,
            )
        ax.axvline(0, color="0.25", linewidth=0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        setup_axis(ax)
    axes[-1].set_xlabel("time from stimulus (s)")
    axes[0].legend(frameon=False, ncol=5, loc="upper right", fontsize=8)
    fig.suptitle("Connectivity Decomposition: Strength vs Phase-Lag Geometry")
    fig_path = figures_dir / "presentation_connectivity_top10_decomposition_overlay.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _example_coherence_components(strength, phase_lag_rad):
    phase_factor = float(abs(np.sin(phase_lag_rad)))
    return {
        "delayed": float(strength * phase_factor),
        "strength": float(strength),
        "timing": phase_factor,
    }


def plot_presentation_connectivity_decomposition_examples(figures_dir):
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 1.0, 500)
    base = np.sin(2.0 * np.pi * 6.0 * t)
    examples = [
        {
            "title": "High strength only",
            "phase": 0.0,
            "strength": 0.90,
            "noise": 0.04,
            "note": "signals match, but with no delay",
        },
        {
            "title": "High phase alignment only",
            "phase": np.pi / 2.0,
            "strength": 0.25,
            "noise": 0.34,
            "note": "right delay shape, weak shared signal",
        },
        {
            "title": "High delayed coordination",
            "phase": np.pi / 2.0,
            "strength": 0.90,
            "noise": 0.04,
            "note": "strong signal with quarter-cycle delay",
        },
        {
            "title": "High strength, opposite phase",
            "phase": np.pi,
            "strength": 0.90,
            "noise": 0.04,
            "note": "signals match, but not as a delay",
        },
    ]
    metric_specs = [
        ("delayed", "Delayed\ncoordination", "#7b3294"),
        ("strength", "Overall\nstrength", "#1b9e77"),
        ("timing", "Phase\nalignment", "#d6604d"),
    ]

    fig, axes = plt.subplots(
        len(examples),
        2,
        figsize=(10.5, 7.6),
        gridspec_kw={"width_ratios": [2.8, 1.1]},
        constrained_layout=True,
    )
    for row_i, spec in enumerate(examples):
        phase = float(spec["phase"])
        strength = float(spec["strength"])
        noise = float(spec["noise"])
        signal_a = base
        signal_b = strength * np.sin(2.0 * np.pi * 6.0 * t - phase)
        signal_b = signal_b + noise * rng.standard_normal(len(t))
        signal_b = signal_b / max(np.nanmax(np.abs(signal_b)), 1e-12)

        ax_sig = axes[row_i, 0]
        ax_sig.plot(t, signal_a, color="#303030", linewidth=1.8, label="Signal A")
        ax_sig.plot(t, signal_b, color="#377eb8", linewidth=1.8, label="Signal B")
        ax_sig.axhline(0, color="0.78", linewidth=0.7)
        ax_sig.set_ylim(-1.25, 1.25)
        ax_sig.set_yticks([])
        ax_sig.set_title(spec["title"], loc="left", fontsize=10)
        ax_sig.text(
            0.99,
            0.08,
            spec["note"],
            transform=ax_sig.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="0.30",
        )
        setup_axis(ax_sig)
        if row_i == len(examples) - 1:
            ax_sig.set_xlabel("time")
        else:
            ax_sig.set_xticklabels([])
        if row_i == 0:
            ax_sig.legend(frameon=False, ncol=2, loc="upper right", fontsize=8)

        metrics = _example_coherence_components(strength, phase)
        ax_bar = axes[row_i, 1]
        labels = [label for _key, label, _color in metric_specs]
        vals = [metrics[key] for key, _label, _color in metric_specs]
        colors = [color for _key, _label, color in metric_specs]
        ax_bar.bar(labels, vals, color=colors, width=0.72)
        ax_bar.set_ylim(0.0, 1.0)
        ax_bar.set_yticks([0.0, 0.5, 1.0])
        ax_bar.tick_params(axis="x", labelsize=8)
        ax_bar.tick_params(axis="y", labelsize=8)
        setup_axis(ax_bar)
        if row_i == 0:
            ax_bar.set_title("What the metrics see", fontsize=10)
    fig.suptitle("Delayed Coordination Requires Both Strength and Phase Alignment")
    fig_path = figures_dir / "presentation_connectivity_decomposition_examples.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def model_color(model_label):
    if model_label == "gradual":
        return "#303030"
    if "binary_D1" in model_label:
        return "#d6604d"
    if "hybrid_D1" in model_label:
        return "#7b3294"
    if "binary" in model_label:
        return "#f4a582"
    if "hybrid" in model_label:
        return "#c2a5cf"
    return "0.65"


def plot_presentation_connect_model_timecourse(output_dir, figures_dir):
    d = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_summary.csv",
        "connectivity model-timecourse output",
    )
    d = d[
        np.isclose(d["active_pct"].astype(float), 0.10)
        & (d["metric"] == "z_euclidean")
    ].copy()
    if d.empty:
        raise ValueError("Missing top-10% z-euclidean model-timecourse rows")
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    conn_models = [
        ("gradual",            "Continuous Restructuring", "#1f1f1f"),
        ("two_stage_hybrid_D1", "Discrete Restructuring (D1)", "#6a3d9a"),
        ("two_stage_hybrid_D2", "Discrete Restructuring (D2)", "#1b9e77"),
        ("two_stage_hybrid_D3", "Discrete Restructuring (D3)", "#377eb8"),
        ("two_stage_hybrid_D4", "Discrete Restructuring (D4)", "#a6cee3"),
    ]
    for model_key, plot_label, color in conn_models:
        g = d[d["model_label"] == model_key].sort_values("time_center_sec")
        if g.empty:
            continue
        x = g["time_center_sec"].to_numpy(float)
        y = g["rho_mean"].to_numpy(float)
        y_sem = g["rho_sem"].to_numpy(float)
        ax.plot(x, y, color=color, linewidth=1.8, alpha=0.92, label=plot_label)
        ax.fill_between(x, y - y_sem, y + y_sem, color=color, alpha=0.12, linewidth=0)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("model correlation")
    ax.set_title("Connectivity Model Evidence Over Time")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower center")
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_connectivity_model_timecourse_top10.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def draw_head_outline(ax):
    head = Circle((0, 0), 1.02, fill=False, color="0.30", linewidth=1.0, zorder=1)
    nose = Polygon(
        [[-0.08, 1.00], [0.00, 1.13], [0.08, 1.00]],
        closed=False,
        fill=False,
        color="0.30",
        linewidth=1.0,
        zorder=1,
    )
    left_ear = Circle((-1.04, 0.0), 0.08, fill=False, color="0.30", linewidth=1.0)
    right_ear = Circle((1.04, 0.0), 0.08, fill=False, color="0.30", linewidth=1.0)
    ax.add_patch(head)
    ax.add_patch(nose)
    ax.add_patch(left_ear)
    ax.add_patch(right_ear)


def draw_edge_panel(ax, rows, layout, value_col, title, vlim, signed=True):
    draw_head_outline(ax)
    ax.scatter(layout["x"], layout["y"], s=18, color="0.25", zorder=3)
    ch_pos = {}
    for row in layout.itertuples(index=False):
        ch_pos[str(row.channel)] = (float(row.x), float(row.y))
    for row in rows.itertuples(index=False):
        if row.ch_i not in ch_pos or row.ch_j not in ch_pos:
            continue
        x1, y1 = ch_pos[row.ch_i]
        x2, y2 = ch_pos[row.ch_j]
        val = float(getattr(row, value_col))
        scaled = min(abs(val) / max(vlim, 1e-12), 1.0)
        color = "#b2182b"
        if signed:
            if val < 0:
                color = "#2166ac"
        else:
            scaled = min(max(val, 0.0) / max(vlim, 1e-12), 1.0)
            color = "#2c7fb8"
        ax.plot(
            [x1, x2],
            [y1, y2],
            color=color,
            alpha=0.16 + 0.42 * scaled,
            linewidth=0.25 + 1.25 * scaled,
            zorder=2,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.12, 1.18)
    for spine in ax.spines.values():
        spine.set_visible(False)


def panel_vlim(rows, value_col):
    vals = []
    for val in rows[value_col].to_numpy(float):
        if np.isfinite(val):
            vals.append(abs(float(val)))
    if len(vals) == 0:
        raise ValueError(f"No finite edge values for {value_col}")
    return float(np.nanpercentile(vals, 95))


def minmax_normalized_edges(d):
    norm_rows = []
    group_cols = ["subject", "day"]
    for key, g in d.groupby(group_cols):
        subject, day = key
        vals = g["conn_mean"].to_numpy(float)
        min_val = float(np.nanmin(vals))
        max_val = float(np.nanmax(vals))
        denom = max_val - min_val
        if denom <= np.finfo(float).eps:
            continue
        for row in g.itertuples(index=False):
            norm_rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "pair_label": row.pair_label,
                    "ch_i": row.ch_i,
                    "ch_j": row.ch_j,
                    "conn_norm": (float(row.conn_mean) - min_val) / denom,
                }
            )
    out = pd.DataFrame(norm_rows)
    if out.empty:
        raise ValueError("No normalized edge rows available")
    return out


def plot_presentation_connect_edges_for_window(
    edges,
    active,
    layout,
    figures_dir,
    window,
):
    active = active[np.isclose(active["active_pct"].astype(float), 0.10)].copy()
    pair_labels = set(active["pair_label"].tolist())
    d = edges[(edges["window"] == window) & edges["pair_label"].isin(pair_labels)]
    if d.empty:
        raise ValueError(f"No {window}-window top-10% edge rows available")
    d_norm = minmax_normalized_edges(d)
    rows = []
    for pair_label in sorted(pair_labels):
        g = d_norm[d_norm["pair_label"] == pair_label]
        d1 = g[g["day"] == 1]["conn_norm"].to_numpy(float)
        dl = g[g["day"] > 1]["conn_norm"].to_numpy(float)
        if len(d1) == 0 or len(dl) == 0:
            continue
        row0 = g.iloc[0]
        rows.append(
            {
                "pair_label": pair_label,
                "ch_i": row0["ch_i"],
                "ch_j": row0["ch_j"],
                "day1": float(np.nanmean(d1)),
                "later": float(np.nanmean(dl)),
                "difference": float(np.nanmean(d1) - np.nanmean(dl)),
            }
        )
    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.5))
    draw_edge_panel(
        axes[0],
        plot_df,
        layout,
        "day1",
        "Day 1",
        panel_vlim(plot_df, "day1"),
        signed=False,
    )
    draw_edge_panel(
        axes[1],
        plot_df,
        layout,
        "later",
        "Days 2-5",
        panel_vlim(plot_df, "later"),
        signed=False,
    )
    draw_edge_panel(
        axes[2],
        plot_df,
        layout,
        "difference",
        "Day 1 - Days 2-5",
        panel_vlim(plot_df, "difference"),
        signed=True,
    )
    fig.suptitle(
        f"{window.title()}-Window Normalized Connectivity Edges, Top 10%"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig_path = figures_dir / (
        f"presentation_connectivity_d1_later_edges_{window}_top10.png"
    )
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def connect_edge_difference_rows(edges, active, window):
    active = active[np.isclose(active["active_pct"].astype(float), 0.10)].copy()
    pair_labels = set(active["pair_label"].tolist())
    d = edges[(edges["window"] == window) & edges["pair_label"].isin(pair_labels)]
    if d.empty:
        raise ValueError(f"No {window}-window top-10% edge rows available")
    d_norm = minmax_normalized_edges(d)
    rows = []
    for pair_label in sorted(pair_labels):
        g = d_norm[d_norm["pair_label"] == pair_label]
        d1 = g[g["day"] == 1]["conn_norm"].to_numpy(float)
        dl = g[g["day"] > 1]["conn_norm"].to_numpy(float)
        if len(d1) == 0 or len(dl) == 0:
            continue
        row0 = g.iloc[0]
        rows.append(
            {
                "pair_label": pair_label,
                "ch_i": row0["ch_i"],
                "ch_j": row0["ch_j"],
                "difference": float(np.nanmean(d1) - np.nanmean(dl)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No {window}-window difference rows available")
    return out


def plot_presentation_connect_difference_row(
    early_edges,
    middle_edges,
    late_edges,
    active,
    layout,
    figures_dir,
    top_n=10,
):
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.2))
    panels = [
        ("early", early_edges, "Early"),
        ("middle", middle_edges, "Middle"),
        ("late", late_edges, "Late"),
    ]
    for idx, panel in enumerate(panels):
        window, edges, title = panel
        rows = connect_edge_difference_rows(edges, active, window)
        rows = (
            rows.assign(abs_diff=rows["difference"].abs())
            .sort_values("abs_diff", ascending=False)
            .head(top_n)
            .drop(columns="abs_diff")
            .reset_index(drop=True)
        )
        draw_edge_panel(
            axes[idx],
            rows,
            layout,
            "difference",
            title,
            panel_vlim(rows, "difference"),
            signed=True,
        )
    fig.suptitle("Day 1 - Days 2-5 Connectivity Edge Differences (Top 10)")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig_path = figures_dir / "presentation_connectivity_difference_edges_3windows.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def get_connect_three_window_bounds(output_dir):
    shape = require_csv_any(
        [
            output_dir / "connect_sensorwide_model_posterior_shape_summary_top10.csv",
            output_dir / "connect_sensorwide_model_posterior_shape_summary.csv",
        ],
        "connectivity posterior-shape output",
    )
    g = shape[
        np.isclose(shape["active_pct"].astype(float), 0.10)
        & (shape["contrast"] == "two_stage_hybrid_D1_minus_gradual")
        & (shape["shape_model"] == "three_window")
    ]
    if g.empty:
        raise ValueError(
            "Missing top-10% three-window posterior-shape row. Run "
            "ACTIVE_PCT=0.10 "
            "python code/connect_sensorwide_model_posterior_shape_analysis.py first."
        )
    row = g.iloc[0]
    return {
        "early": (float(row["lb_early"]), float(row["ub_early"])),
        "middle": (float(row["lb_middle"]), float(row["ub_middle"])),
        "late": (float(row["lb_late"]), float(row["ub_late"])),
    }


def make_subject_window_edges(subject_df, active, window, bounds):
    active = active[np.isclose(active["active_pct"].astype(float), 0.10)].copy()
    pair_labels = set(active["pair_label"].tolist())
    d = subject_df[
        (subject_df["lock_type"] == "stim")
        & (subject_df["band"] == "broadband")
    ].copy()
    d["pair_label"] = d["ch_i"].astype(str) + "--" + d["ch_j"].astype(str)
    d = d[d["pair_label"].isin(pair_labels)].copy()
    d["window_center_sec"] = d["lock_time"].astype(float) + 0.025
    lo, hi = bounds
    d = d[(d["window_center_sec"] >= lo) & (d["window_center_sec"] <= hi)].copy()
    if d.empty:
        raise ValueError(f"No subject connectivity rows for {window} bounds")
    rows = []
    group_cols = ["subject", "day", "pair_label", "ch_i", "ch_j"]
    for key, g in d.groupby(group_cols):
        subject, day, pair_label, ch_i, ch_j = key
        vals = g["conn_val"].to_numpy(float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "subject": int(subject),
                "day": int(day),
                "window": window,
                "window_start_sec": float(lo),
                "window_end_sec": float(hi),
                "pair_label": str(pair_label),
                "ch_i": str(ch_i),
                "ch_j": str(ch_j),
                "conn_mean": float(np.mean(vals)),
                "n_time_bins": int(len(vals)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError(f"No subject window edge rows for {window}")
    return out


def plot_presentation_connect_edges(output_dir, figures_dir):
    edges = require_csv(
        output_dir / "connect_sensorwide_window_average_subject_edges.csv",
        "connectivity window-average edge output",
    )
    active = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_active_pairs.csv",
        "connectivity active-pair output",
    )
    subject = require_csv(
        output_dir / "sensorwide_carpet_subject_timeseries.csv",
        "connectivity subject carpet output",
    )
    layout = require_csv(
        output_dir / "sensorwide_channel_layout.csv",
        "connectivity channel layout output",
    )
    paths = []
    for window in ["early", "late"]:
        paths.append(
            plot_presentation_connect_edges_for_window(
                edges,
                active,
                layout,
                figures_dir,
                window,
            )
        )
    bounds = get_connect_three_window_bounds(output_dir)
    middle_edges = make_subject_window_edges(
        subject,
        active,
        "middle",
        bounds["middle"],
    )
    paths.insert(
        1,
        plot_presentation_connect_edges_for_window(
            middle_edges,
            active,
            layout,
            figures_dir,
            "middle",
        ),
    )
    paths.append(
        plot_presentation_connect_difference_row(
            edges,
            middle_edges,
            edges,
            active,
            layout,
            figures_dir,
        )
    )
    legacy_path = figures_dir / "presentation_connectivity_d1_later_edges_top10.png"
    shutil.copyfile(paths[2], legacy_path)
    paths.append(legacy_path)
    return paths


def plot_presentation_mvpa_auc(output_dir, figures_dir):
    d = require_csv(
        output_dir / "mvpa_stim_locked_cat_day_means_timecourse.csv",
        "stim MVPA time-resolved output",
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for day in DAYS:
        g = d[d["day"] == day].sort_values("time_sec")
        if g.empty:
            raise ValueError(f"Missing MVPA AUC rows for day={day}")
        x = g["time_sec"].to_numpy(float)
        y = g["auc_mean"].to_numpy(float)
        ax.plot(x, y, color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
    ax.axhline(0.5, color="0.25", linewidth=0.8)
    ax.axvspan(0.06, 0.18, color="0.75", alpha=0.18, linewidth=0)
    ax.axvspan(0.40, 0.60, color="0.55", alpha=0.14, linewidth=0)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("AUC")
    ax.set_title("Time-Resolved Category Decoding")
    ax.legend(frameon=False, ncol=1, loc="upper left")
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_mvpa_time_resolved_auc.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def load_behavior():
    if not BEHAVIOURAL_DIR.exists():
        raise FileNotFoundError(f"Missing behavioural directory: {BEHAVIOURAL_DIR}")
    rows = []
    for path in sorted(BEHAVIOURAL_DIR.glob("*.csv")):
        match = BEHAV_RE.match(path.name)
        if match is None:
            raise ValueError(f"Unexpected behavioural file: {path.name}")
        subject = int(match.group(1))
        day_code = int(match.group(2))
        day = day_code
        if day_code >= 100:
            day = day_code // 100
        d = pd.read_csv(path)
        if "fb" not in d.columns or "rt" not in d.columns:
            raise ValueError(f"Missing fb/rt columns in {path}")
        correct = d["fb"].astype(str).str.lower() == "correct"
        rt = pd.to_numeric(d["rt"], errors="coerce")
        rows.append(
            {
                "subject": subject,
                "day": day,
                "accuracy": float(np.mean(correct)),
                "rt": float(np.nanmean(rt[correct])),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No behavioural files loaded")
    return out


def mvpa_window_features(output_dir):
    d = require_csv(
        output_dir / "mvpa_stim_locked_cat_subject_day_timecourse.csv",
        "stim MVPA subject-day timecourse output",
    )
    rows = []
    for key, g in d.groupby(["subject", "day"]):
        subject, day = key
        for window, lo, hi in [("early", 0.06, 0.18), ("late", 0.40, 0.60)]:
            h = g[(g["time_sec"] >= lo) & (g["time_sec"] <= hi)]
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "window": window,
                    "auc": float(np.nanmean(h["auc"].to_numpy(float))),
                }
            )
    return pd.DataFrame(rows)


def plot_presentation_mvpa_peak_behavior(output_dir, figures_dir):
    peaks = require_csv(
        output_dir / "mvpa_stim_locked_cat_haufe_subject_day_peak_times.csv",
        "MVPA peak-latency output",
    )
    behavior = load_behavior()
    mvpa = mvpa_window_features(output_dir)
    merged = mvpa.merge(behavior, on=["subject", "day"], how="inner")
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.2))
    for col_i, peak in enumerate(["early", "late"]):
        ax = axes[0, col_i]
        g = peaks[peaks["peak"] == peak]
        means = []
        errs = []
        for day in DAYS:
            vals = g[g["day"] == day]["peak_auc"].to_numpy(float)
            means.append(float(np.nanmean(vals)))
            errs.append(sem(vals))
        ax.errorbar(DAYS, means, yerr=errs, color="#303030", marker="o")
        ax.set_title(f"{peak} peak AUC")
        ax.set_xlabel("day")
        ax.set_ylabel("peak AUC")
        ax.set_xticks(DAYS)
        setup_axis(ax)
    for col_i, window in enumerate(["early", "late"]):
        ax = axes[1, col_i]
        g = merged[merged["window"] == window]
        for day in DAYS:
            h = g[g["day"] == day]
            ax.scatter(h["auc"], h["accuracy"], color=DAY_COLORS[day], s=24,
                       alpha=0.8, label=f"D{day}")
        ax.set_title(f"{window} AUC vs accuracy, {corr_text(g['auc'], g['accuracy'])}")
        ax.set_xlabel("AUC")
        ax.set_ylabel("Human Category Choice Accuracy")
        setup_axis(ax)
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig_path = figures_dir / "presentation_mvpa_peak_and_behavior.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_presentation_mvpa_model_timecourse(output_dir, figures_dir):
    d = require_csv(
        output_dir / "mvpa_tg_diagonal_presentation_model_bic_summary.csv",
        "presentation MVPA BIC model-timecourse output",
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    mvpa_models = [
        ("Continuous Restructuring",    "Continuous Restructuring",    "#1f1f1f"),
        ("Discrete Restructuring D1",   "Discrete Restructuring (D1)", "#6a3d9a"),
        ("Discrete Restructuring D2",   "Discrete Restructuring (D2)", "#1b9e77"),
        ("Discrete Restructuring D3",   "Discrete Restructuring (D3)", "#377eb8"),
        ("Discrete Restructuring D4",   "Discrete Restructuring (D4)", "#a6cee3"),
    ]
    y_vals = []
    for label, plot_label, color in mvpa_models:
        g = d[
            (d["classifier"] == "logreg")
            & (d["model_label"] == label)
        ].sort_values("time_sec")
        if g.empty:
            continue
        x = g["time_sec"].to_numpy(float)
        y = -g["delta_bic_null_mean"].to_numpy(float)
        for val in y:
            if np.isfinite(val):
                y_vals.append(float(val))
        ax.plot(x, y, color=color, linewidth=1.8, alpha=0.92, label=plot_label)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.axvspan(0.06, 0.18, color="0.75", alpha=0.18, linewidth=0)
    ax.axvspan(0.40, 0.60, color="0.55", alpha=0.14, linewidth=0)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("BIC improvement over same-day baseline")
    ax.set_title("MVPA Transfer Model Evidence Over Time")
    if len(y_vals) > 0:
        ymin = float(np.nanmin(y_vals))
        ymax = float(np.nanmax(y_vals))
        pad = 0.12 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="best")
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_mvpa_model_timecourse.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_presentation_mvpa_tg_diagonal_day_matrices(output_dir, figures_dir):
    d = require_csv(
        output_dir / "mvpa_stim_locked_cat_tg_timegen_day_mean.csv",
        "stim MVPA TG day-mean matrix output",
    )
    d = d[
        np.isclose(d["train_time_sec"].astype(float), d["test_time_sec"].astype(float))
    ].copy()
    if d.empty:
        raise ValueError("Missing same-time TG diagonal day-matrix rows")
    requested_times = np.round(np.arange(-0.2, 0.81, 0.1), 1)
    available_times = np.sort(d["train_time_sec"].dropna().unique().astype(float))
    snapshot_times = []
    for time_s in requested_times:
        nearest = float(available_times[np.argmin(np.abs(available_times - time_s))])
        if nearest not in snapshot_times:
            snapshot_times.append(nearest)

    matrices = []
    for time_s in snapshot_times:
        g = d[np.isclose(d["train_time_sec"].astype(float), time_s)]
        mat = np.full((5, 5), np.nan)
        for row in g.itertuples(index=False):
            mat[int(row.train_day) - 1, int(row.test_day) - 1] = float(row.auc_mean)
        matrices.append((time_s, mat))

    fig, axes = plt.subplots(3, 4, figsize=(11.0, 7.7))
    fig.subplots_adjust(
        left=0.06,
        right=0.86,
        bottom=0.08,
        top=0.88,
        wspace=0.45,
        hspace=0.35,
    )
    axes_flat = axes.ravel()
    image = None
    for ax, (time_s, mat) in zip(axes_flat, matrices):
        image = plot_matrix(
            ax,
            mat,
            f"{time_s * 1000:.0f} ms",
            "viridis",
            0.50,
            0.72,
            annotate=False,
        )
    for ax in axes_flat[len(matrices):]:
        ax.axis("off")
    if image is not None:
        cax = fig.add_axes([0.89, 0.18, 0.018, 0.62])
        fig.colorbar(image, cax=cax)
    fig.suptitle("MVPA Same-Time Day Transfer Matrices")
    fig_path = figures_dir / "presentation_mvpa_tg_diagonal_day_matrices_100ms.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def transfer_matrix(group, classifier, window):
    mat = np.full((5, 5), np.nan)
    g = group[(group["classifier"] == classifier) & (group["window"] == window)]
    if g.empty:
        raise ValueError(f"Missing transfer rows for {classifier}/{window}")
    for row in g.itertuples(index=False):
        mat[int(row.train_day) - 1, int(row.test_day) - 1] = float(row.auc_mean)
    return mat


def template_matrix(kind, split_day=None):
    mat = np.full((5, 5), np.nan)
    for train_day in DAYS:
        for test_day in DAYS:
            if kind == "gradual":
                val = 0.65 * min(train_day, test_day) / float(max(DAYS))
                if train_day == test_day:
                    val = train_day / float(max(DAYS))
            elif kind == "split_gradual":
                if split_day is None:
                    raise ValueError("split_gradual requires split_day")
                train_late = train_day > split_day
                test_late = test_day > split_day
                if train_late != test_late:
                    val = 0.0
                else:
                    val = 0.65 * min(train_day, test_day) / float(max(DAYS))
                    if train_day == test_day:
                        val = train_day / float(max(DAYS))
            elif kind == "split_binary":
                if split_day is None:
                    raise ValueError("split_binary requires split_day")
                val = 1.0
                train_late = train_day > split_day
                test_late = test_day > split_day
                if train_late != test_late:
                    val = 0.0
            else:
                raise ValueError(f"Unknown template: {kind}")
            mat[train_day - 1, test_day - 1] = val
    return mat


def plot_matrix(ax, mat, title, cmap, vmin=None, vmax=None, annotate=True):
    image = ax.imshow(mat, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(["D1", "D2", "D3", "D4", "D5"], fontsize=8)
    ax.set_yticklabels(["D1", "D2", "D3", "D4", "D5"], fontsize=8)
    ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if annotate:
        for i in range(5):
            for j in range(5):
                color = "white"
                if np.isfinite(mat[i, j]) and float(mat[i, j]) > 0.68:
                    color = "black"
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=color,
                )
    return image


def conn_template_matrix(kind, split_day=None):
    """5×5 day-similarity matrix matching the model_distance() formula in
    connect_sensorwide_model_timecourse_analysis.py, converted to similarity
    via 1 − distance so higher values mean more similar (consistent with MVPA matrices)."""
    mat = np.full((5, 5), np.nan)
    for d1 in DAYS:
        for d2 in DAYS:
            gradual_dist = abs(d1 - d2) / 4.0
            if kind == "gradual":
                dist = gradual_dist
            elif kind == "split_gradual":
                if split_day is None:
                    raise ValueError("split_gradual requires split_day")
                if (d1 > split_day) == (d2 > split_day):
                    dist = 0.5 * gradual_dist
                else:
                    dist = 0.5 + 0.5 * gradual_dist
            else:
                raise ValueError(f"Unknown connectivity template: {kind}")
            mat[d1 - 1, d2 - 1] = 1.0 - dist
    return mat


def plot_presentation_connectivity_model_predictions(figures_dir):
    fig = plt.figure(figsize=(11.0, 5.5))
    gs = gridspec.GridSpec(
        2, 8, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.05, right=0.99, top=0.88, bottom=0.07,
    )
    ax_top = fig.add_subplot(gs[0, 3:5])
    plot_matrix(ax_top, conn_template_matrix("gradual"),
                "Continuous Restructuring", "viridis", 0, 1, annotate=False)
    for idx, split_day in enumerate([1, 2, 3, 4]):
        ax = fig.add_subplot(gs[1, idx * 2: idx * 2 + 2])
        plot_matrix(ax, conn_template_matrix("split_gradual", split_day=split_day),
                    f"Discrete Restructuring (D{split_day})", "viridis", 0, 1, annotate=False)
    fig.suptitle("Connectivity Day-Similarity Model Predictions", y=0.97)
    fig_path = figures_dir / "presentation_connectivity_model_predictions.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_presentation_mvpa_window_model(output_dir, figures_dir):
    early = require_csv(
        output_dir / "mvpa_stim_locked_cat_early_window_transfer_group_pairs.csv",
        "early MVPA transfer output",
    )
    late = require_csv(
        output_dir / "mvpa_stim_locked_cat_late_window_transfer_group_pairs.csv",
        "late MVPA transfer output",
    )
    group = pd.concat([early, late], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for row_i, window in enumerate(["early", "late"]):
        mat = transfer_matrix(group, "logreg", window)
        image = plot_matrix(axes[row_i], mat, f"{window} observed AUC",
                            "viridis", 0.50, 0.62)
        fig.colorbar(image, ax=axes[row_i], fraction=0.046)
    fig.suptitle("Windowed MVPA Transfer")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = figures_dir / "presentation_mvpa_window_transfer_empirical.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig = plt.figure(figsize=(11.0, 5.5))
    gs_m = gridspec.GridSpec(
        2, 8, figure=fig,
        hspace=0.45, wspace=0.35,
        left=0.05, right=0.99, top=0.88, bottom=0.07,
    )
    ax_top = fig.add_subplot(gs_m[0, 3:5])
    plot_matrix(ax_top, template_matrix("gradual"),
                "Continuous Restructuring", "viridis", 0, 1, annotate=False)
    for idx, split_day in enumerate([1, 2, 3, 4]):
        ax = fig.add_subplot(gs_m[1, idx * 2: idx * 2 + 2])
        plot_matrix(ax, template_matrix("split_gradual", split_day=split_day),
                    f"Discrete Restructuring (D{split_day})", "viridis", 0, 1, annotate=False)
    fig.suptitle("MVPA Transfer Model Predictions", y=0.97)
    model_path = figures_dir / "presentation_mvpa_window_transfer_model_predictions.png"
    fig.savefig(model_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [fig_path, model_path]


def plot_presentation_rsa(output_dir, figures_dir):
    d = require_csv(
        output_dir / "rsa_stim_model_fit_timecourses.csv",
        "stim RSA model-fit output",
    )
    fig, axes = plt.subplots(1, 5, figsize=(16.0, 3.4), sharex=True, sharey=True)
    models = [
        "Physical distance",
        "Category / response",
        "Boundary difficulty",
        "Signed boundary position",
    ]
    colors = {
        "Physical distance": "#4c78a8",
        "Category / response": "#f58518",
        "Boundary difficulty": "#54a24b",
        "Signed boundary position": "#b279a2",
    }
    for ax_i, day in enumerate(DAYS):
        ax = axes[ax_i]
        d_day = d[d["day"] == day]
        if d_day.empty:
            raise ValueError(f"Missing RSA rows for day={day}")
        for model in models:
            rows = []
            g_model = d_day[d_day["model"] == model]
            for time_s in sorted(g_model["time_sec"].drop_duplicates().tolist()):
                vals = g_model[g_model["time_sec"] == time_s]["rho"].to_numpy(float)
                rows.append(
                    {
                        "time": time_s,
                        "mean": float(np.nanmean(vals)),
                        "sem": sem(vals),
                    }
                )
            g = pd.DataFrame(rows)
            ax.plot(
                g["time"],
                g["mean"],
                color=colors[model],
                linewidth=1.8,
                label=model,
            )
            ax.fill_between(
                g["time"],
                g["mean"] - g["sem"],
                g["mean"] + g["sem"],
                color=colors[model],
                alpha=0.12,
                linewidth=0,
            )
        ax.axhline(0, color="0.25", linewidth=0.8)
        ax.set_title(f"Day {day}")
        ax.set_xlabel("time from stimulus (s)")
        setup_axis(ax)
        if ax_i == 0:
            ax.set_ylabel("RSA model fit")
        if ax_i == 4:
            ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Stimulus-Locked RSA Model Fits")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig_path = figures_dir / "presentation_rsa_model_fit_timecourses.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    source_path = figures_dir / "rsa_model_prediction_rdms.png"
    if not source_path.exists():
        raise FileNotFoundError(
            f"Missing RSA prediction figure: {source_path}. "
            "Run code/rsa_model_prediction_figure.py first."
        )
    pred_path = figures_dir / "presentation_rsa_model_prediction_rdms.png"
    shutil.copyfile(source_path, pred_path)
    return [fig_path, pred_path]


PEAK_LABELS = ["Peak 1", "Peak 2", "Peak 3"]
PEAK_COLORS = ["#4c78a8", "#f58518", "#54a24b"]


def _grand_avg_connectivity_peak_windows(d, n_peaks=3, half_win=0.05):
    """Detect n_peaks from the grand-average timecourse and return search windows."""
    tc = (
        d.groupby("lock_time", as_index=False)["conn_val"]
        .mean()
        .sort_values("lock_time")
    )
    times = tc["lock_time"].to_numpy(float)
    vals = tc["conn_val"].to_numpy(float)
    peaks, props = find_peaks(vals, distance=5, prominence=0.005)
    if len(peaks) < n_peaks:
        raise ValueError(
            f"Only {len(peaks)} peaks found in grand-average connectivity timecourse; "
            f"expected at least {n_peaks}."
        )
    top_idx = peaks[np.argsort(props["prominences"])[::-1][:n_peaks]]
    top_idx = sorted(top_idx)
    windows = []
    for p in top_idx:
        windows.append((float(times[p]) - half_win, float(times[p]) + half_win))
    return windows


def subject_day_connectivity_peaks(output_dir):
    """Per-subject per-day peak latency and amplitude for the three connectivity peaks."""
    active = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_active_pairs.csv",
        "connectivity active-pair output",
    )
    subject_df = require_csv(
        output_dir / "sensorwide_carpet_subject_timeseries.csv",
        "connectivity subject carpet output",
    )

    active = active[np.isclose(active["active_pct"].astype(float), 0.10)].copy()
    pair_labels = set(active["pair_label"].tolist())

    d = subject_df[
        (subject_df["lock_type"] == "stim")
        & (subject_df["band"] == "broadband")
    ].copy()
    d["pair_label"] = d["ch_i"].astype(str) + "--" + d["ch_j"].astype(str)
    d = d[d["pair_label"].isin(pair_labels)].copy()
    if d.empty:
        raise ValueError("No stim broadband top-10% subject connectivity data found")

    windows = _grand_avg_connectivity_peak_windows(d)
    peak_windows = list(zip(PEAK_LABELS, windows))

    rows = []
    for (subject, day), d_sd in d.groupby(["subject", "day"]):
        tc = (
            d_sd.groupby("lock_time", as_index=False)["conn_val"]
            .mean()
            .sort_values("lock_time")
        )
        times = tc["lock_time"].to_numpy(float)
        vals = tc["conn_val"].to_numpy(float)
        for peak_label, (lo, hi) in peak_windows:
            mask = (times >= lo) & (times <= hi)
            if not np.any(mask):
                continue
            t_win = times[mask]
            v_win = vals[mask]
            peak_idx = int(np.argmax(v_win))
            rows.append(
                {
                    "subject": int(subject),
                    "day": int(day),
                    "peak": peak_label,
                    "latency": float(t_win[peak_idx]),
                    "amplitude": float(v_win[peak_idx]),
                }
            )
    return pd.DataFrame(rows)


def _regression_line(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 3:
        return None
    slope, intercept, r, p, _ = stats.linregress(x[good], y[good])
    x_line = np.linspace(x[good].min(), x[good].max(), 200)
    return x_line, intercept + slope * x_line, r, p


def plot_presentation_connectivity_peak_day(output_dir, figures_dir):
    peak_df = subject_day_connectivity_peaks(output_dir)
    peak_color = dict(zip(PEAK_LABELS, PEAK_COLORS))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))

    for feature, ylabel, title, ax in [
        ("latency", "Latency (s)", "Peak Latency by Day", axes[0]),
        ("amplitude", "Connectivity amplitude", "Peak Amplitude by Day", axes[1]),
    ]:
        for peak in PEAK_LABELS:
            g = peak_df[peak_df["peak"] == peak]
            means, errs = [], []
            for day in DAYS:
                vals = g[g["day"] == day][feature].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                means.append(float(np.nanmean(vals)) if len(vals) > 0 else np.nan)
                errs.append(sem(vals) if len(vals) > 1 else np.nan)
            ax.errorbar(
                DAYS, means, yerr=errs,
                color=peak_color[peak], marker="o", linewidth=2,
                markersize=7, capsize=4, label=peak,
            )
        ax.set_xlabel("Day")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(DAYS)
        ax.legend(frameon=False, fontsize=9)
        setup_axis(ax)

    fig.suptitle("Connectivity Peak Features Across Days", fontsize=12)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_connectivity_peak_day.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_presentation_connectivity_peak_behavior(output_dir, figures_dir):
    from matplotlib.lines import Line2D

    peak_df = subject_day_connectivity_peaks(output_dir)
    behavior = load_behavior()
    merged = peak_df.merge(behavior, on=["subject", "day"], how="inner")

    measures = [
        ("accuracy", "Accuracy",      "lower right"),
        ("rt",       "Response Time (s)", "upper right"),
    ]
    fig, axes = plt.subplots(
        len(measures), len(DAYS),
        figsize=(4.0 * len(DAYS), 3.8 * len(measures)),
        squeeze=False,
    )

    annot_spacing = 0.11
    annot_fontsize = 8.0

    for row_i, (measure_col, measure_label, annot_corner) in enumerate(measures):
        annot_top = annot_corner == "upper right"
        for col_i, day in enumerate(DAYS):
            ax = axes[row_i, col_i]
            d_day = merged[merged["day"] == day]
            annot_lines = []
            for peak, color in zip(PEAK_LABELS, PEAK_COLORS):
                g = d_day[d_day["peak"] == peak]
                x = g["amplitude"].to_numpy(float)
                y = g[measure_col].to_numpy(float)
                ax.scatter(x, y, color=color, s=30, alpha=0.45, zorder=3)
                reg = _regression_line(x, y)
                if reg is not None:
                    x_line, y_line, r, p = reg
                    ax.plot(x_line, y_line, color=color, linewidth=1.6, alpha=0.85)
                    annot_lines.append((peak, color, r, p))
            for k, (peak, color, r, p) in enumerate(annot_lines):
                if annot_top:
                    y_pos = 0.97 - k * annot_spacing
                    va = "top"
                else:
                    y_pos = 0.03 + k * annot_spacing
                    va = "bottom"
                ax.text(
                    0.97, y_pos,
                    f"{peak}: r={r:.2f}, p={p:.2f}",
                    transform=ax.transAxes,
                    fontsize=annot_fontsize, color=color,
                    ha="right", va=va,
                )
            if col_i == 0:
                ax.set_ylabel(measure_label)
            if row_i == 0:
                ax.set_title(f"Day {day}")
            if row_i == len(measures) - 1:
                ax.set_xlabel("Peak amplitude")
            setup_axis(ax)

    legend_handles = [
        Line2D([0], [0], color=color, linewidth=2.2, label=peak)
        for peak, color in zip(PEAK_LABELS, PEAK_COLORS)
    ]
    fig.suptitle("Connectivity Peak Amplitude vs Behavior by Day", fontsize=12)
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig_path = figures_dir / "presentation_connectivity_peak_behavior.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_fig_presentation(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["erp"] = plot_presentation_erp_stim(output_dir, figures_dir)
    paths["erp_gfp"] = plot_presentation_erp_gfp(output_dir, figures_dir)
    paths["connect_overlay"] = plot_presentation_connect_overlay(
        output_dir, figures_dir
    )
    paths["connect_decomposition_overlay"] = (
        plot_presentation_connect_decomposition_overlay(output_dir, figures_dir)
    )
    paths["connect_decomposition_examples"] = (
        plot_presentation_connectivity_decomposition_examples(figures_dir)
    )
    paths["connect_model"] = plot_presentation_connect_model_timecourse(
        output_dir, figures_dir
    )
    connect_edge_paths = plot_presentation_connect_edges(output_dir, figures_dir)
    paths["connect_edges_early"] = connect_edge_paths[0]
    paths["connect_edges_middle"] = connect_edge_paths[1]
    paths["connect_edges_late"] = connect_edge_paths[2]
    paths["connect_difference_row"] = connect_edge_paths[3]
    paths["connect_edges"] = connect_edge_paths[4]
    paths["mvpa_auc"] = plot_presentation_mvpa_auc(output_dir, figures_dir)
    paths["mvpa_peak_behavior"] = plot_presentation_mvpa_peak_behavior(
        output_dir, figures_dir
    )
    paths["mvpa_model_timecourse"] = plot_presentation_mvpa_model_timecourse(
        output_dir, figures_dir
    )
    paths["mvpa_tg_diagonal_day_matrices"] = (
        plot_presentation_mvpa_tg_diagonal_day_matrices(output_dir, figures_dir)
    )
    mvpa_window_paths = plot_presentation_mvpa_window_model(
        output_dir, figures_dir
    )
    paths["mvpa_window_empirical"] = mvpa_window_paths[0]
    paths["mvpa_window_predictions"] = mvpa_window_paths[1]
    rsa_paths = plot_presentation_rsa(output_dir, figures_dir)
    paths["rsa_model_fit"] = rsa_paths[0]
    paths["rsa_model_predictions"] = rsa_paths[1]
    paths["connectivity_model_predictions"] = plot_presentation_connectivity_model_predictions(
        figures_dir
    )
    paths["connectivity_peak_day"] = plot_presentation_connectivity_peak_day(
        output_dir, figures_dir
    )
    paths["connectivity_peak_behavior"] = plot_presentation_connectivity_peak_behavior(
        output_dir, figures_dir
    )
    for key, path in paths.items():
        print(f"[presentation] {key}: {path}", flush=True)
    return paths


if __name__ == "__main__":
    save_fig_presentation()
