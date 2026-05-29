#!/usr/bin/env python3
"""Create presentation-focused figures from existing analysis outputs."""

from pathlib import Path
import os
import re

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    ax.grid(alpha=0.22)


def plot_presentation_erp_stim(output_dir, figures_dir):
    d = require_csv(
        output_dir / "erp_grand_average_by_day_lock_condition.csv",
        "ERP grand-average output",
    )
    d = d[(d["lock_type"] == "stim") & (d["condition"] == "all")].copy()
    if d.empty:
        raise ValueError("No stim/all rows in ERP grand-average output")
    fig, axes = plt.subplots(1, len(DAYS), figsize=(15, 3.2), sharex=True, sharey=True)
    channels = sorted(d["channel"].drop_duplicates().tolist())
    for ax_i, day in enumerate(DAYS):
        ax = axes[ax_i]
        d_day = d[d["day"] == day]
        if d_day.empty:
            raise ValueError(f"Missing stim ERP rows for day={day}")
        for channel in channels:
            d_ch = d_day[d_day["channel"] == channel].sort_values("time_s")
            ax.plot(
                d_ch["time_s"].to_numpy(dtype=float),
                d_ch["amplitude_v"].to_numpy(dtype=float) * 1e6,
                color="0.35",
                alpha=0.22,
                linewidth=0.55,
            )
        gfp = []
        times = sorted(d_day["time_s"].drop_duplicates().tolist())
        for time_s in times:
            vals = d_day[d_day["time_s"] == time_s]["amplitude_v"].to_numpy(float)
            gfp.append(float(np.sqrt(np.nanmean(vals**2))) * 1e6)
        ax.plot(times, gfp, color="#c23b22", linewidth=2.1, label="GFP")
        ax.axvline(0, color="0.25", linewidth=0.8)
        ax.set_title(f"Day {day}")
        ax.set_xlim(-0.1, 0.8)
        setup_axis(ax)
        if ax_i == 0:
            ax.set_ylabel("amplitude / GFP (uV)")
        ax.set_xlabel("time from stimulus (s)")
    fig.suptitle("Stimulus-Locked ERPs")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig_path = figures_dir / "presentation_erp_stim_all.png"
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
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for day in DAYS:
        d_day = d[d["day"] == day]
        rows = []
        for time_s in sorted(d_day["lock_time"].drop_duplicates().tolist()):
            vals = d_day[d_day["lock_time"] == time_s]["conn_val"].to_numpy(float)
            rows.append({"time": time_s, "mean": float(np.nanmean(vals))})
        plot_df = pd.DataFrame(rows)
        ax.plot(
            plot_df["time"],
            plot_df["mean"],
            color=DAY_COLORS[day],
            linewidth=2.0,
            label=f"D{day}",
        )
    ax.axvline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("mean connectivity")
    ax.set_title("Broadband Connectivity, Top 10% Active Edges")
    ax.legend(frameon=False, ncol=5, loc="upper left")
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_connectivity_top10_overlay.png"
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
    labels = [
        "gradual",
        "two_stage_binary_D1",
        "two_stage_hybrid_D1",
        "two_stage_binary_D2",
        "two_stage_hybrid_D2",
        "two_stage_binary_D3",
        "two_stage_hybrid_D3",
        "two_stage_binary_D4",
        "two_stage_hybrid_D4",
    ]
    for label in labels:
        g = d[d["model_label"] == label].sort_values("time_center_sec")
        if g.empty:
            continue
        lw = 1.0
        alpha = 0.55
        zorder = 1
        if label in ["gradual", "two_stage_binary_D1", "two_stage_hybrid_D1"]:
            lw = 2.4
            alpha = 1.0
            zorder = 3
        plot_label = label.replace("two_stage_", "")
        plot_label = plot_label.replace("_", " ")
        ax.plot(
            g["time_center_sec"],
            g["rho_mean"],
            color=model_color(label),
            linewidth=lw,
            alpha=alpha,
            label=plot_label,
            zorder=zorder,
        )
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
        & (shape["shape_model"] == "two_window")
    ]
    if g.empty:
        raise ValueError(
            "Missing top-10% posterior-shape row. Run "
            "ACTIVE_PCT=0.10 "
            "python code/connect_sensorwide_model_posterior_shape_analysis.py first."
        )
    row = g.iloc[0]
    for lo_col, hi_col in [("lb_early", "ub_early"), ("lb_late", "ub_late")]:
        ax.axvspan(
            float(row[lo_col]),
            float(row[hi_col]),
            color="#7b3294",
            alpha=0.12,
            linewidth=0,
        )
    odds = float(row["posterior_model_prob"])
    text = f"P(two-window)={odds:.2f}"
    ax.text(0.98, 0.94, text, transform=ax.transAxes, ha="right", va="top")
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("model correlation")
    ax.set_title("Connectivity Model Evidence Over Time")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="lower right")
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_connectivity_model_timecourse_top10.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def draw_edge_panel(ax, rows, layout, value_col, title, vlim):
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
        if val < 0:
            color = "#2166ac"
        ax.plot([x1, x2], [y1, y2], color=color, alpha=0.25 + 0.55 * scaled,
                linewidth=0.4 + 2.2 * scaled, zorder=2)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_presentation_connect_edges(output_dir, figures_dir):
    edges = require_csv(
        output_dir / "connect_sensorwide_window_average_subject_edges.csv",
        "connectivity window-average edge output",
    )
    active = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_active_pairs.csv",
        "connectivity active-pair output",
    )
    layout = require_csv(
        output_dir / "sensorwide_channel_layout.csv",
        "connectivity channel layout output",
    )
    active = active[np.isclose(active["active_pct"].astype(float), 0.10)].copy()
    pair_labels = set(active["pair_label"].tolist())
    d = edges[(edges["window"] == "late") & edges["pair_label"].isin(pair_labels)]
    if d.empty:
        raise ValueError("No late-window top-10% edge rows available")
    rows = []
    for pair_label in sorted(pair_labels):
        g = d[d["pair_label"] == pair_label]
        d1 = g[g["day"] == 1]["conn_mean"].to_numpy(float)
        dl = g[g["day"] > 1]["conn_mean"].to_numpy(float)
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
    vals = []
    for col in ["day1", "later", "difference"]:
        vals.extend(plot_df[col].to_numpy(float).tolist())
    vlim = float(np.nanpercentile(np.abs(vals), 95))
    fig, axes = plt.subplots(1, 3, figsize=(9.8, 3.5))
    draw_edge_panel(axes[0], plot_df, layout, "day1", "Day 1", vlim)
    draw_edge_panel(axes[1], plot_df, layout, "later", "Days 2-5", vlim)
    draw_edge_panel(axes[2], plot_df, layout, "difference", "Day 1 - Days 2-5", vlim)
    fig.suptitle("Late-Window Connectivity Edges, Top 10%")
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig_path = figures_dir / "presentation_connectivity_d1_later_edges_top10.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


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
        sem_vals = g["auc_sem"].to_numpy(float)
        ax.plot(x, y, color=DAY_COLORS[day], linewidth=2.0, label=f"D{day}")
        ax.fill_between(x, y - sem_vals, y + sem_vals, color=DAY_COLORS[day],
                        alpha=0.12, linewidth=0)
    ax.axhline(0.5, color="0.25", linewidth=0.8)
    ax.axvspan(0.06, 0.18, color="0.75", alpha=0.18, linewidth=0)
    ax.axvspan(0.40, 0.60, color="0.55", alpha=0.14, linewidth=0)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("AUC")
    ax.set_title("Time-Resolved Category Decoding")
    ax.legend(frameon=False, ncol=5, loc="upper left")
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
        ax.set_ylabel("accuracy")
        setup_axis(ax)
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig_path = figures_dir / "presentation_mvpa_peak_and_behavior.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_presentation_mvpa_model_timecourse(output_dir, figures_dir):
    d = require_csv(
        output_dir / "presentation_mvpa_model_timecourse.csv",
        "presentation MVPA model-timecourse output",
    )
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    labels = [
        "one_stage_bottleneck",
        "one_stage_closeness",
        "two_stage_binary_D1",
        "two_stage_bottleneck_D1",
    ]
    for label in labels:
        g = d[d["model_label"] == label].sort_values("time_sec")
        if g.empty:
            continue
        color = "#303030"
        if "closeness" in label:
            color = "#4c78a8"
        if "binary" in label:
            color = "#d6604d"
        if "bottleneck_D1" in label:
            color = "#7b3294"
        ax.plot(g["time_sec"], g["rho"], color=color, linewidth=2.0,
                label=label.replace("_", " "))
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.axvspan(0.06, 0.18, color="0.75", alpha=0.18, linewidth=0)
    ax.axvspan(0.40, 0.60, color="0.55", alpha=0.14, linewidth=0)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("model correlation")
    ax.set_title("MVPA Transfer Model Evidence Over Time")
    ax.legend(frameon=False, fontsize=8)
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_mvpa_model_timecourse.png"
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


def template_matrix(kind):
    mat = np.full((5, 5), np.nan)
    for train_day in DAYS:
        for test_day in DAYS:
            if kind == "gradual":
                val = (min(train_day, test_day) - 1.0) / 4.0
            elif kind == "d1_split":
                val = 1.0
                if (train_day == 1 and test_day > 1) or (
                    train_day > 1 and test_day == 1
                ):
                    val = 0.0
            else:
                raise ValueError(f"Unknown template: {kind}")
            mat[train_day - 1, test_day - 1] = val
    return mat


def plot_matrix(ax, mat, title, cmap, vmin=None, vmax=None):
    image = ax.imshow(mat, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(5))
    ax.set_yticks(range(5))
    ax.set_xticklabels(["D1", "D2", "D3", "D4", "D5"])
    ax.set_yticklabels(["D1", "D2", "D3", "D4", "D5"])
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                    fontsize=8, color="white")
    return image


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
    fig, axes = plt.subplots(2, 3, figsize=(9.8, 6.0))
    for row_i, window in enumerate(["early", "late"]):
        mat = transfer_matrix(group, "logreg", window)
        im0 = plot_matrix(axes[row_i, 0], mat, f"{window} observed AUC",
                          "viridis", 0.50, 0.62)
        plot_matrix(axes[row_i, 1], template_matrix("gradual"),
                    "gradual prediction", "Greys", 0, 1)
        plot_matrix(axes[row_i, 2], template_matrix("d1_split"),
                    "D1 split prediction", "Greys", 0, 1)
        fig.colorbar(im0, ax=axes[row_i, 0], fraction=0.046)
    fig.suptitle("Windowed MVPA Transfer Matrices")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig_path = figures_dir / "presentation_mvpa_window_transfer_model_compare.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def plot_presentation_rsa(output_dir, figures_dir):
    d = require_csv(
        output_dir / "rsa_stim_model_fit_timecourses.csv",
        "stim RSA model-fit output",
    )
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    models = ["Physical distance", "Category / response"]
    for model in models:
        rows = []
        g_model = d[d["model"] == model]
        for time_s in sorted(g_model["time_sec"].drop_duplicates().tolist()):
            vals = g_model[g_model["time_sec"] == time_s]["rho"].to_numpy(float)
            rows.append({"time": time_s, "mean": float(np.nanmean(vals)),
                         "sem": sem(vals)})
        g = pd.DataFrame(rows)
        color = "#4c78a8"
        if model == "Category / response":
            color = "#f58518"
        ax.plot(g["time"], g["mean"], color=color, linewidth=2.0, label=model)
        ax.fill_between(g["time"], g["mean"] - g["sem"], g["mean"] + g["sem"],
                        color=color, alpha=0.15, linewidth=0)
    ax.axhline(0, color="0.25", linewidth=0.8)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("RSA model fit")
    ax.set_title("RSA Mainly Tracks Stimulus Geometry")
    ax.legend(frameon=False)
    setup_axis(ax)
    fig.tight_layout()
    fig_path = figures_dir / "presentation_rsa_model_fit_timecourses.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    comp = require_csv(
        output_dir / "day_rdm_model_compare_summary.csv",
        "day-RDM model comparison output",
    )
    comp = comp[
        (comp["modality"] == "rsa")
        & (comp["measure"] == "stim_windowed")
        & (comp["window"] == "late")
        & (comp["value_kind"] == "similarity")
    ]
    if comp.empty:
        raise ValueError("Missing RSA day-RDM model comparison rows")
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    labels = []
    vals = []
    errs = []
    for row in comp.itertuples(index=False):
        label = str(row.model)
        if np.isfinite(float(row.split_day)):
            label = f"{label} D{int(row.split_day)}"
        labels.append(label)
        vals.append(float(row.mean_rho))
        errs.append(float(row.sem_rho))
    x = np.arange(len(vals))
    ax.bar(x, vals, yerr=errs, color="0.45", error_kw={"capsize": 2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_ylabel("model correlation")
    ax.set_title("RSA Cross-Day Geometry Model Comparison")
    setup_axis(ax)
    fig.tight_layout()
    fig_path2 = figures_dir / "presentation_rsa_cross_day_model_compare.png"
    fig.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [fig_path, fig_path2]


def save_fig_presentation(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    paths["erp"] = plot_presentation_erp_stim(output_dir, figures_dir)
    paths["connect_overlay"] = plot_presentation_connect_overlay(
        output_dir, figures_dir
    )
    paths["connect_model"] = plot_presentation_connect_model_timecourse(
        output_dir, figures_dir
    )
    paths["connect_edges"] = plot_presentation_connect_edges(output_dir, figures_dir)
    paths["mvpa_auc"] = plot_presentation_mvpa_auc(output_dir, figures_dir)
    paths["mvpa_peak_behavior"] = plot_presentation_mvpa_peak_behavior(
        output_dir, figures_dir
    )
    paths["mvpa_model_timecourse"] = plot_presentation_mvpa_model_timecourse(
        output_dir, figures_dir
    )
    paths["mvpa_window_model"] = plot_presentation_mvpa_window_model(
        output_dir, figures_dir
    )
    rsa_paths = plot_presentation_rsa(output_dir, figures_dir)
    paths["rsa_model_fit"] = rsa_paths[0]
    paths["rsa_day_model"] = rsa_paths[1]
    for key, path in paths.items():
        print(f"[presentation] {key}: {path}", flush=True)
    return paths


if __name__ == "__main__":
    save_fig_presentation()
