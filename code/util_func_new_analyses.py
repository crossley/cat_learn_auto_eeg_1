#!/usr/bin/env python3
"""Additional TG and ERP analyses for learning-process dissociation tests."""

from __future__ import annotations

import json
import os
import re
import time
from multiprocessing import cpu_count
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from joblib import Parallel, delayed
try:
    from threadpoolctl import threadpool_limits
except Exception:  # pragma: no cover
    threadpool_limits = None

from util_func_mvpa import _process_cross_day_pair, _session_cache_key


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "new_analyses"
FIGURES_DIR = PROJECT_DIR / "figures" / "new_analyses"

TG_WINDOWS = {
    "early": (0.060, 0.180),
    "late": (0.250, 0.550),
}

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _default_n_workers():
    logical = max(1, int(cpu_count() or 1))
    physical_est = max(1, logical // 2)
    return max(1, physical_est - 2)


def _parallel_collect(func, items, n_workers):
    if n_workers == 1:
        return [func(item) for item in items]
    try:
        if threadpool_limits is None:
            return Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                delayed(func)(item) for item in items
            )
        with threadpool_limits(limits=1):
            return Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
                delayed(func)(item) for item in items
            )
    except PermissionError:
        print("[parallel] loky unavailable; falling back to serial execution.", flush=True)
        return [func(item) for item in items]


def _fit_ols(formula, df, cluster_subject=True):
    model = smf.ols(formula, data=df).fit()
    if cluster_subject and "subject" in df.columns and df["subject"].nunique() > 1:
        return model.get_robustcov_results(cov_type="cluster", groups=df["subject"])
    return model


def _model_term_summary(model, term):
    names = list(model.model.exog_names)
    idx = names.index(term)
    ci = model.conf_int()
    if hasattr(ci, "iloc"):
        lo, hi = ci.iloc[idx]
    else:
        lo, hi = ci[idx]
    return {
        "term": term,
        "estimate": float(model.params[idx]),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(model.pvalues[idx]),
    }


def _parse_matrix_path(path):
    m = re.match(r"sub_(\d+)_trainD(\d+)_testD(\d+)\.npz$", path.name)
    if m is None:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


def _summarize_tg_window_matrix(win, train_times=None, test_times=None, summary="square_mean", top_prop=0.10):
    win = np.asarray(win, dtype=float)
    if summary == "square_mean":
        vals = win[np.isfinite(win)]
    elif summary == "diagonal_mean":
        vals = np.diag(win)
        vals = vals[np.isfinite(vals)]
    elif summary == "top10_mean":
        vals = win[np.isfinite(win)]
        if len(vals) > 0:
            n_top = max(1, int(np.ceil(len(vals) * top_prop)))
            vals = np.sort(vals)[-n_top:]
    else:
        raise ValueError(f"Unknown TG window summary: {summary}")
    if len(vals) == 0:
        return np.nan, 0
    return float(np.nanmean(vals)), int(len(vals))


def extract_tg_window_auc(
    matrix_dir: Path | str = PROJECT_DIR / "output" / "mvpa_tg_cross_day" / "tg_cross_day_subject_matrices",
    windows: dict[str, tuple[float, float]] = TG_WINDOWS,
    summary: str = "square_mean",
):
    """Extract subject-level cross-day window means from saved TG matrices."""
    matrix_dir = Path(matrix_dir)
    rows = []
    for path in sorted(matrix_dir.glob("sub_*_trainD*_testD*.npz")):
        parsed = _parse_matrix_path(path)
        if parsed is None:
            continue
        subject, train_day, test_day = parsed
        if train_day == test_day:
            continue
        with np.load(path, allow_pickle=False) as z:
            auc = np.asarray(z["auc"], dtype=float)
            time_sec = np.asarray(z["time_sec"], dtype=float)
        for window_name, (tmin, tmax) in windows.items():
            mask = (time_sec >= tmin) & (time_sec <= tmax)
            if not np.any(mask):
                mean_auc = np.nan
                n_cells = 0
            else:
                win = auc[np.ix_(mask, mask)]
                mean_auc, n_cells = _summarize_tg_window_matrix(win, summary=summary)
            rows.append(
                {
                    "subject": subject,
                    "train_day": train_day,
                    "test_day": test_day,
                    "day_distance": abs(train_day - test_day),
                    "window": window_name,
                    "window_tmin": tmin,
                    "window_tmax": tmax,
                    "mean_auc": mean_auc,
                    "n_cells": n_cells,
                    "summary": summary,
                    "matrix_file": path.name,
                }
            )
    return pd.DataFrame(rows)


def fit_tg_window_gradients(window_df):
    """Fit day-distance gradients by TG window and the window x distance interaction."""
    d = window_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_auc", "day_distance", "window"]).copy()
    d = d[d["day_distance"] > 0]
    summaries = []
    models = {}
    for window_name, g in d.groupby("window"):
        model = _fit_ols("mean_auc ~ day_distance", g)
        item = _model_term_summary(model, "day_distance")
        item.update({"window": window_name, "n_rows": int(len(g)), "n_subjects": int(g["subject"].nunique())})
        summaries.append(item)
        models[window_name] = model

    d["window"] = pd.Categorical(d["window"], categories=["early", "late"], ordered=False)
    interaction = _fit_ols("mean_auc ~ day_distance * window", d)
    interaction_term = "day_distance:window[T.late]"
    interaction_summary = _model_term_summary(interaction, interaction_term)
    interaction_summary.update({"comparison": "late_minus_early_slope"})
    return pd.DataFrame(summaries).sort_values("window"), interaction_summary, models, interaction


def plot_tg_window_gradients(window_df, slope_df, fig_path):
    d = window_df.dropna(subset=["mean_auc"]).copy()
    d = d[d["day_distance"] > 0]
    subject_distance = (
        d.groupby(["subject", "window", "day_distance"], as_index=False)["mean_auc"]
        .mean()
        .sort_values(["window", "day_distance"])
    )
    plot_df = (
        subject_distance.groupby(["window", "day_distance"], as_index=False)
        .agg(
            auc_mean=("mean_auc", "mean"),
            auc_sem=("mean_auc", _sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["window", "day_distance"])
    )

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4), sharey=True, squeeze=False)
    colors = {"early": "tab:blue", "late": "tab:orange"}
    for ax, window_name in zip(axes.ravel(), ["early", "late"]):
        g = plot_df[plot_df["window"] == window_name]
        raw = subject_distance[subject_distance["window"] == window_name]
        color = colors.get(window_name, "black")
        ax.errorbar(
            g["day_distance"],
            g["auc_mean"],
            yerr=g["auc_sem"],
            marker="o",
            color=color,
            capsize=3,
            linewidth=1.8,
        )
        if len(raw) >= 2 and raw["day_distance"].nunique() >= 2:
            model = smf.ols("mean_auc ~ day_distance", data=raw).fit()
            x = np.linspace(raw["day_distance"].min(), raw["day_distance"].max(), 100)
            pred = model.predict(pd.DataFrame({"day_distance": x}))
            ax.plot(x, pred, color=color, linestyle="--", linewidth=1.6)
        s = slope_df[slope_df["window"] == window_name]
        if not s.empty:
            slope = float(s["estimate"].iloc[0])
            lo = float(s["ci_low"].iloc[0])
            hi = float(s["ci_high"].iloc[0])
            ax.set_title(f"{window_name.title()} ({slope:.4f} [{lo:.4f}, {hi:.4f}])")
        else:
            ax.set_title(window_name.title())
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.set_xlabel("Day distance")
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("Mean off-diagonal AUC")
    fig.suptitle("Cross-Day TG Window Gradients")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_cross_day_tg_window_structure(
    matrix_dir: Path | str = PROJECT_DIR / "output" / "mvpa_tg_cross_day" / "tg_cross_day_subject_matrices",
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    window_df = extract_tg_window_auc(matrix_dir=matrix_dir)
    slope_df, interaction_summary, _, _ = fit_tg_window_gradients(window_df)
    window_csv = output_dir / "tg_cross_day_window_auc_subject_pairs.csv"
    slope_csv = output_dir / "tg_cross_day_window_gradient_slopes.csv"
    interaction_json = output_dir / "tg_cross_day_window_slope_difference.json"
    fig_path = figures_dir / "tg_cross_day_window_gradients.png"
    window_df.to_csv(window_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    interaction_json.write_text(json.dumps(interaction_summary, indent=2))
    plot_tg_window_gradients(window_df, slope_df, fig_path)
    return {
        "window_df": window_df,
        "slope_df": slope_df,
        "interaction_summary": interaction_summary,
        "window_csv": window_csv,
        "slope_csv": slope_csv,
        "interaction_json": interaction_json,
        "figure": fig_path,
    }


def add_day1_pair_labels(window_df):
    d = window_df.copy()
    d["includes_day1"] = (d["train_day"] == 1) | (d["test_day"] == 1)
    d["pair_group"] = np.where(d["includes_day1"], "day1_pair", "later_only")
    d["day1_pair_type"] = "later_only"
    d.loc[(d["train_day"] == 1) & (d["test_day"] != 1), "day1_pair_type"] = "day1_forward"
    d.loc[(d["train_day"] != 1) & (d["test_day"] == 1), "day1_pair_type"] = "day1_backward"
    d["other_day"] = np.nan
    d.loc[d["train_day"] == 1, "other_day"] = d.loc[d["train_day"] == 1, "test_day"]
    d.loc[d["test_day"] == 1, "other_day"] = d.loc[d["test_day"] == 1, "train_day"]
    return d


def fit_day1_distinctiveness(window_df):
    d = add_day1_pair_labels(window_df)
    d = d.replace([np.inf, -np.inf], np.nan).dropna(subset=["mean_auc", "window"]).copy()
    d["window"] = pd.Categorical(d["window"], categories=["early", "late"])
    d["pair_group"] = pd.Categorical(d["pair_group"], categories=["later_only", "day1_pair"])
    d["day1_pair_type"] = pd.Categorical(
        d["day1_pair_type"],
        categories=["later_only", "day1_forward", "day1_backward"],
    )

    model_group = _fit_ols("mean_auc ~ window * pair_group", d)
    group_terms = []
    for term in model_group.model.exog_names:
        if term == "Intercept":
            continue
        item = _model_term_summary(model_group, term)
        item["model"] = "day1_pair_vs_later_only"
        group_terms.append(item)

    model_direction = _fit_ols("mean_auc ~ window * day1_pair_type", d)
    direction_terms = []
    for term in model_direction.model.exog_names:
        if term == "Intercept":
            continue
        item = _model_term_summary(model_direction, term)
        item["model"] = "day1_forward_backward_vs_later_only"
        direction_terms.append(item)

    return pd.DataFrame(group_terms + direction_terms), model_group, model_direction


def plot_day1_pair_group_bars(window_df, fig_path):
    d = add_day1_pair_labels(window_df).dropna(subset=["mean_auc"]).copy()
    subject_group = (
        d.groupby(["subject", "window", "day1_pair_type"], as_index=False)["mean_auc"]
        .mean()
        .sort_values(["window", "day1_pair_type"])
    )
    plot_df = (
        subject_group.groupby(["window", "day1_pair_type"], as_index=False)
        .agg(auc_mean=("mean_auc", "mean"), auc_sem=("mean_auc", _sem), n_subjects=("subject", "nunique"))
    )
    pair_order = ["later_only", "day1_forward", "day1_backward"]
    labels = ["Later only", "D1 -> later", "Later -> D1"]
    colors = {"early": "tab:blue", "late": "tab:orange"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4), sharey=True, squeeze=False)
    for ax, window_name in zip(axes.ravel(), ["early", "late"]):
        g = plot_df[plot_df["window"] == window_name].set_index("day1_pair_type").reindex(pair_order)
        x = np.arange(len(pair_order))
        ax.bar(x, g["auc_mean"], yerr=g["auc_sem"], color=colors[window_name], alpha=0.75, capsize=3)
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_title(window_name.title())
        ax.grid(axis="y", alpha=0.25)
    axes.ravel()[0].set_ylabel("Mean window AUC")
    fig.suptitle("Day 1 Distinctiveness: Pair-Type Contrast")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_day1_anchored_trajectories(window_df, fig_path):
    d = add_day1_pair_labels(window_df).dropna(subset=["mean_auc", "other_day"]).copy()
    d = d[d["day1_pair_type"].isin(["day1_forward", "day1_backward"])]
    subject_day = (
        d.groupby(["subject", "window", "day1_pair_type", "other_day"], as_index=False)["mean_auc"]
        .mean()
        .sort_values(["window", "day1_pair_type", "other_day"])
    )
    plot_df = (
        subject_day.groupby(["window", "day1_pair_type", "other_day"], as_index=False)
        .agg(auc_mean=("mean_auc", "mean"), auc_sem=("mean_auc", _sem), n_subjects=("subject", "nunique"))
    )
    colors = {"day1_forward": "tab:purple", "day1_backward": "tab:green"}
    labels = {"day1_forward": "D1 -> later", "day1_backward": "Later -> D1"}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4), sharey=True, squeeze=False)
    for ax, window_name in zip(axes.ravel(), ["early", "late"]):
        for pair_type in ["day1_forward", "day1_backward"]:
            g = plot_df[(plot_df["window"] == window_name) & (plot_df["day1_pair_type"] == pair_type)]
            if g.empty:
                continue
            ax.errorbar(
                g["other_day"],
                g["auc_mean"],
                yerr=g["auc_sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=colors[pair_type],
                label=labels[pair_type],
            )
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.set_xticks([2, 3, 4, 5])
        ax.set_xlabel("Other day")
        ax.set_title(window_name.title())
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("Mean window AUC")
    axes.ravel()[0].legend(loc="best")
    fig.suptitle("Day 1 Anchored Cross-Day TG")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def extract_within_day_window_auc(
    within_csv: Path | str = PROJECT_DIR / "output" / "mvpa_tg_cross_day" / "tg_within_day_subject_level.csv",
    windows: dict[str, tuple[float, float]] = TG_WINDOWS,
    summary: str = "square_mean",
):
    within_csv = Path(within_csv)
    if not within_csv.exists():
        return pd.DataFrame(
            columns=["subject", "train_day", "test_day", "day_distance", "window", "mean_auc", "n_cells"]
        )
    d = pd.read_csv(within_csv, low_memory=False)
    for col in ["subject", "day", "train_time_sec", "test_time_sec", "auc"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=["subject", "day", "train_time_sec", "test_time_sec", "auc"]).copy()
    rows = []
    for (subject, day), g_day in d.groupby(["subject", "day"]):
        pivot = g_day.pivot_table(index="train_time_sec", columns="test_time_sec", values="auc", aggfunc="mean")
        train_axis = pivot.index.to_numpy(dtype=float)
        test_axis = pivot.columns.to_numpy(dtype=float)
        auc_mat = pivot.to_numpy(dtype=float)
        for window_name, (tmin, tmax) in windows.items():
            train_mask = (train_axis >= tmin) & (train_axis <= tmax)
            test_mask = (test_axis >= tmin) & (test_axis <= tmax)
            win = auc_mat[np.ix_(train_mask, test_mask)]
            mean_auc, n_cells = _summarize_tg_window_matrix(win, summary=summary)
            rows.append(
                {
                    "subject": int(subject),
                    "train_day": int(day),
                    "test_day": int(day),
                    "day_distance": 0,
                    "window": window_name,
                    "window_tmin": tmin,
                    "window_tmax": tmax,
                    "mean_auc": mean_auc,
                    "n_cells": n_cells,
                    "summary": summary,
                    "matrix_file": "",
                    "day1_pair_type": "within_day",
                    "pair_group": "within_day",
                    "includes_day1": int(day) == 1,
                }
            )
    return pd.DataFrame(rows)


def plot_day_pair_window_matrices(window_df, fig_path, within_df=None):
    d = window_df.dropna(subset=["mean_auc"]).copy()
    if within_df is not None and not within_df.empty:
        d = pd.concat([d, within_df.dropna(subset=["mean_auc"]).copy()], ignore_index=True)
    pair_mean = (
        d.groupby(["window", "train_day", "test_day"], as_index=False)["mean_auc"]
        .mean()
        .sort_values(["window", "train_day", "test_day"])
    )
    days = [1, 2, 3, 4, 5]
    vmin = float(pair_mean["mean_auc"].min()) if not pair_mean.empty else 0.45
    vmax = float(pair_mean["mean_auc"].max()) if not pair_mean.empty else 0.55
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), squeeze=False)
    for ax, window_name in zip(axes.ravel(), ["early", "late"]):
        mat = np.full((len(days), len(days)), np.nan)
        g = pair_mean[pair_mean["window"] == window_name]
        for _, row in g.iterrows():
            i = days.index(int(row["train_day"]))
            j = days.index(int(row["test_day"]))
            mat[i, j] = float(row["mean_auc"])
        im = ax.imshow(np.ma.masked_invalid(mat), origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(days)))
        ax.set_yticks(range(len(days)))
        ax.set_xticklabels([f"D{day}" for day in days])
        ax.set_yticklabels([f"D{day}" for day in days])
        ax.set_xlabel("Test day")
        ax.set_ylabel("Train day")
        ax.set_title(window_name.title())
        for i in range(len(days)):
            for j in range(len(days)):
                if np.isfinite(mat[i, j]):
                    color = "black" if mat[i, j] > (vmin + 0.65 * (vmax - vmin)) else "white"
                    ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
    fig.suptitle("TG Window AUC by Day Pair (Diagonal = Within-Day)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, label="AUC")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_day_pair_window_matrices_by_summary(matrix_df, fig_path):
    summaries = ["square_mean", "diagonal_mean", "top10_mean"]
    summaries = [s for s in summaries if s in set(matrix_df["summary"])]
    days = [1, 2, 3, 4, 5]
    fig, axes = plt.subplots(len(summaries), 2, figsize=(9.4, 4.1 * len(summaries)), squeeze=False)
    for r, summary in enumerate(summaries):
        d_summary = matrix_df[matrix_df["summary"] == summary]
        vmin = float(d_summary["auc_mean"].min()) if not d_summary.empty else 0.45
        vmax = float(d_summary["auc_mean"].max()) if not d_summary.empty else 0.55
        for c, window_name in enumerate(["early", "late"]):
            ax = axes[r, c]
            mat = np.full((len(days), len(days)), np.nan)
            g = d_summary[d_summary["window"] == window_name]
            for _, row in g.iterrows():
                i = days.index(int(row["train_day"]))
                j = days.index(int(row["test_day"]))
                mat[i, j] = float(row["auc_mean"])
            im = ax.imshow(np.ma.masked_invalid(mat), origin="upper", cmap="viridis", vmin=vmin, vmax=vmax)
            ax.set_xticks(range(len(days)))
            ax.set_yticks(range(len(days)))
            ax.set_xticklabels([f"D{day}" for day in days])
            ax.set_yticklabels([f"D{day}" for day in days])
            ax.set_xlabel("Test day")
            ax.set_ylabel("Train day")
            ax.set_title(f"{summary} | {window_name}")
            for i in range(len(days)):
                for j in range(len(days)):
                    if np.isfinite(mat[i, j]):
                        color = "black" if mat[i, j] > (vmin + 0.65 * (vmax - vmin)) else "white"
                        ax.text(j, i, f"{mat[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)
            fig.colorbar(im, ax=ax, shrink=0.75, label="AUC")
    fig.suptitle("TG Window AUC by Day Pair and Summary (Diagonal = Within-Day)", y=1.0)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_day1_distinctiveness_analysis(
    matrix_dir: Path | str = PROJECT_DIR / "output" / "mvpa_tg_cross_day" / "tg_cross_day_subject_matrices",
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    window_df = add_day1_pair_labels(extract_tg_window_auc(matrix_dir=matrix_dir, summary="square_mean"))
    within_df = extract_within_day_window_auc(summary="square_mean")
    summary_matrix_rows = []
    summary_window_rows = []
    for summary_name in ["square_mean", "diagonal_mean", "top10_mean"]:
        d_cross = add_day1_pair_labels(extract_tg_window_auc(matrix_dir=matrix_dir, summary=summary_name))
        d_within = extract_within_day_window_auc(summary=summary_name)
        d_all = pd.concat(
            [d_cross.dropna(subset=["mean_auc"]), d_within.dropna(subset=["mean_auc"])],
            ignore_index=True,
        )
        summary_window_rows.append(d_all)
        summary_matrix_rows.append(
            d_all.groupby(["summary", "window", "train_day", "test_day"], as_index=False)
            .agg(
                auc_mean=("mean_auc", "mean"),
                auc_sem=("mean_auc", _sem),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["summary", "window", "train_day", "test_day"])
        )
    stats_df, _, _ = fit_day1_distinctiveness(window_df)
    group_summary = (
        window_df.dropna(subset=["mean_auc"])
        .groupby(["window", "day1_pair_type"], as_index=False)
        .agg(
            auc_mean=("mean_auc", "mean"),
            auc_sem=("mean_auc", _sem),
            n_subjects=("subject", "nunique"),
            n_rows=("mean_auc", "size"),
        )
        .sort_values(["window", "day1_pair_type"])
    )
    pair_matrix_source = pd.concat(
        [window_df.dropna(subset=["mean_auc"]), within_df.dropna(subset=["mean_auc"])],
        ignore_index=True,
    )
    pair_matrix = (
        pair_matrix_source
        .groupby(["window", "train_day", "test_day"], as_index=False)
        .agg(
            auc_mean=("mean_auc", "mean"),
            auc_sem=("mean_auc", _sem),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["window", "train_day", "test_day"])
    )

    window_csv = output_dir / "tg_day1_window_auc_subject_pairs.csv"
    stats_csv = output_dir / "tg_day1_distinctiveness_model_terms.csv"
    summary_csv = output_dir / "tg_day1_pair_type_summary.csv"
    matrix_csv = output_dir / "tg_day_pair_window_auc_matrix.csv"
    summary_window_csv = output_dir / "tg_day_pair_window_auc_subject_pairs_by_summary.csv"
    summary_matrix_csv = output_dir / "tg_day_pair_window_auc_matrix_by_summary.csv"
    fig_pair_types = figures_dir / "tg_day1_pair_type_contrast.png"
    fig_anchored = figures_dir / "tg_day1_anchored_trajectories.png"
    fig_matrices = figures_dir / "tg_day_pair_window_matrices.png"
    fig_matrices_by_summary = figures_dir / "tg_day_pair_window_matrices_by_summary.png"

    window_df.to_csv(window_csv, index=False)
    stats_df.to_csv(stats_csv, index=False)
    group_summary.to_csv(summary_csv, index=False)
    pair_matrix.to_csv(matrix_csv, index=False)
    pd.concat(summary_window_rows, ignore_index=True).to_csv(summary_window_csv, index=False)
    pd.concat(summary_matrix_rows, ignore_index=True).to_csv(summary_matrix_csv, index=False)
    plot_day1_pair_group_bars(window_df, fig_pair_types)
    plot_day1_anchored_trajectories(window_df, fig_anchored)
    plot_day_pair_window_matrices(window_df, fig_matrices, within_df=within_df)
    plot_day_pair_window_matrices_by_summary(pd.concat(summary_matrix_rows, ignore_index=True), fig_matrices_by_summary)

    return {
        "window_df": window_df,
        "stats_df": stats_df,
        "group_summary": group_summary,
        "pair_matrix": pair_matrix,
        "window_csv": window_csv,
        "stats_csv": stats_csv,
        "summary_csv": summary_csv,
        "matrix_csv": matrix_csv,
        "summary_window_csv": summary_window_csv,
        "summary_matrix_csv": summary_matrix_csv,
        "figures": {
            "pair_types": fig_pair_types,
            "anchored": fig_anchored,
            "matrices": fig_matrices,
            "matrices_by_summary": fig_matrices_by_summary,
        },
    }


def _load_sessions_for_band_tg(project_dir=PROJECT_DIR):
    beh_dir = Path(project_dir) / "Behavioural"
    epo_dir = Path(project_dir) / "EEG_epo"
    beh_re = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")
    epo_re = re.compile(r"^P(\d+)_D([\d_]+)-epo\.fif$")

    beh_map = {}
    for beh_path in sorted(beh_dir.glob("*.csv")):
        m = beh_re.match(beh_path.name)
        if m is None:
            continue
        subject = int(m.group(1))
        day = int(m.group(2)) // 100
        beh_map[(subject, day)] = beh_path

    epo_map = {}
    for epo_path in sorted(epo_dir.glob("*-epo.fif")):
        m = epo_re.match(epo_path.name)
        if m is None:
            continue
        subject = int(m.group(1))
        day = int(m.group(2).split("_")[0])
        epo_map[(subject, day)] = epo_path

    sessions = []
    for key in sorted(set(beh_map) & set(epo_map)):
        subject, day = key
        sessions.append(
            {
                "subject": subject,
                "day": day,
                "beh_file": beh_map[key].name,
                "epo_file": epo_map[key].name,
                "epo_path": str(epo_map[key]),
            }
        )
    return sessions


def _prepare_band_envelope_cache(session_item, cache_dir, band_name, fmin, fmax, min_epochs, random_state):
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    session_file = session_item["epo_file"]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{band_name}_cache_{_session_cache_key(session_item)}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            y = z["y"]
            t = z["t"]
            ch_names = z["ch_names"]
        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "session_file": session_file,
            "cache_path": str(cache_path),
            "n_trials": int(len(y)),
            "n_a": int(np.sum(y == 0)),
            "n_b": int(np.sum(y == 1)),
            "n_times": int(len(t)),
            "ch_names": ch_names.tolist(),
        }

    try:
        epochs = mne.read_epochs(session_item["epo_path"], preload=False, verbose="ERROR")
        stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
        if len(stim_events) < 2:
            raise ValueError(f"missing_stim_labels:{','.join(stim_events)}")
        epochs = epochs[stim_events].copy().load_data()
        epochs.pick_types(eeg=True, exclude="bads")
        if len(epochs.ch_names) == 0:
            raise RuntimeError("no_eeg_channels_after_pick")

        analysis_tmin, analysis_tmax = -0.2, 0.8
        if band_name == "delta":
            left_margin = analysis_tmin - float(epochs.times[0])
            right_margin = float(epochs.times[-1]) - analysis_tmax
            if left_margin < 1.0 or right_margin < 1.0:
                return {
                    "ok": False,
                    "qc": {
                        "session_file": session_file,
                        "subject": subject,
                        "day": day,
                        "stage": "filter_margin",
                        "reason": "delta_skipped_insufficient_epoch_margin",
                        "detail": (
                            f"Need >=1 s outside analysis window; "
                            f"left={left_margin:.3f}, right={right_margin:.3f}"
                        ),
                    },
                }

        epochs.filter(
            l_freq=fmin,
            h_freq=fmax,
            method="fir",
            fir_design="firwin",
            phase="zero-double",
            pad="reflect_limited",
            verbose="ERROR",
        )
        epochs.apply_hilbert(envelope=True, verbose="ERROR")
        epochs.resample(128, npad="auto")

        codes = epochs.events[:, 2]
        y = np.full(len(codes), -1, dtype=int)
        y[codes == epochs.event_id["Stim/A"]] = 0
        y[codes == epochs.event_id["Stim/B"]] = 1
        keep = y >= 0
        y = y[keep]
        X = epochs.get_data()[keep]
        t = epochs.times.copy()
        ch_names = np.array(epochs.ch_names, dtype=str)
        if len(y) < min_epochs:
            raise ValueError(f"insufficient_epochs:n_trials={len(y)} < min_epochs={min_epochs}")
        if min(np.sum(y == 0), np.sum(y == 1)) < 5:
            raise ValueError(f"insufficient_class_trials:n_a={int(np.sum(y == 0))}, n_b={int(np.sum(y == 1))}")
        np.savez_compressed(cache_path, X=X, y=y, t=t, ch_names=ch_names)
        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "session_file": session_file,
            "cache_path": str(cache_path),
            "n_trials": int(len(y)),
            "n_a": int(np.sum(y == 0)),
            "n_b": int(np.sum(y == 1)),
            "n_times": int(len(t)),
            "ch_names": ch_names.tolist(),
        }
    except Exception as exc:
        msg = str(exc)
        reason = msg.split(":")[0] if ":" in msg else "prep_error"
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "prepare_band_envelope",
                "reason": reason,
                "detail": msg,
            },
        }


def _prepare_band_envelope_cache_from_task(task):
    return _prepare_band_envelope_cache(
        task["session_item"],
        task["cache_dir"],
        task["band_name"],
        task["fmin"],
        task["fmax"],
        task["min_epochs"],
        task["random_state"],
    )


def _prepare_band_signed_cache(session_item, cache_dir, band_name, fmin, fmax, min_epochs, random_state):
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    session_file = session_item["epo_file"]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{band_name}_signed_cache_{_session_cache_key(session_item)}.npz"
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            y = z["y"]
            t = z["t"]
            ch_names = z["ch_names"]
        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "session_file": session_file,
            "cache_path": str(cache_path),
            "n_trials": int(len(y)),
            "n_a": int(np.sum(y == 0)),
            "n_b": int(np.sum(y == 1)),
            "n_times": int(len(t)),
            "ch_names": ch_names.tolist(),
        }

    try:
        epochs = mne.read_epochs(session_item["epo_path"], preload=False, verbose="ERROR")
        stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
        if len(stim_events) < 2:
            raise ValueError(f"missing_stim_labels:{','.join(stim_events)}")
        epochs = epochs[stim_events].copy().load_data()
        epochs.pick_types(eeg=True, exclude="bads")
        if len(epochs.ch_names) == 0:
            raise RuntimeError("no_eeg_channels_after_pick")

        analysis_tmin, analysis_tmax = -0.2, 0.8
        if band_name == "delta":
            left_margin = analysis_tmin - float(epochs.times[0])
            right_margin = float(epochs.times[-1]) - analysis_tmax
            if left_margin < 1.0 or right_margin < 1.0:
                return {
                    "ok": False,
                    "qc": {
                        "session_file": session_file,
                        "subject": subject,
                        "day": day,
                        "stage": "filter_margin",
                        "reason": "delta_skipped_insufficient_epoch_margin",
                        "detail": (
                            f"Need >=1 s outside analysis window; "
                            f"left={left_margin:.3f}, right={right_margin:.3f}"
                        ),
                    },
                }

        epochs.filter(
            l_freq=fmin,
            h_freq=fmax,
            method="fir",
            fir_design="firwin",
            phase="zero-double",
            pad="reflect_limited",
            verbose="ERROR",
        )
        epochs.resample(128, npad="auto")

        codes = epochs.events[:, 2]
        y = np.full(len(codes), -1, dtype=int)
        y[codes == epochs.event_id["Stim/A"]] = 0
        y[codes == epochs.event_id["Stim/B"]] = 1
        keep = y >= 0
        y = y[keep]
        X = epochs.get_data()[keep]
        t = epochs.times.copy()
        ch_names = np.array(epochs.ch_names, dtype=str)
        if len(y) < min_epochs:
            raise ValueError(f"insufficient_epochs:n_trials={len(y)} < min_epochs={min_epochs}")
        if min(np.sum(y == 0), np.sum(y == 1)) < 5:
            raise ValueError(f"insufficient_class_trials:n_a={int(np.sum(y == 0))}, n_b={int(np.sum(y == 1))}")
        np.savez_compressed(cache_path, X=X, y=y, t=t, ch_names=ch_names)
        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "session_file": session_file,
            "cache_path": str(cache_path),
            "n_trials": int(len(y)),
            "n_a": int(np.sum(y == 0)),
            "n_b": int(np.sum(y == 1)),
            "n_times": int(len(t)),
            "ch_names": ch_names.tolist(),
        }
    except Exception as exc:
        msg = str(exc)
        reason = msg.split(":")[0] if ":" in msg else "prep_error"
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "prepare_band_signed_voltage",
                "reason": reason,
                "detail": msg,
            },
        }


def _prepare_band_signed_cache_from_task(task):
    return _prepare_band_signed_cache(
        task["session_item"],
        task["cache_dir"],
        task["band_name"],
        task["fmin"],
        task["fmax"],
        task["min_epochs"],
        task["random_state"],
    )


def _process_cross_day_pair_from_task(task):
    return _process_cross_day_pair(task["pair_item"], random_state=task["random_state"])


def _write_cross_day_outputs(results, output_dir):
    output_dir = Path(output_dir)
    matrix_dir = output_dir / "tg_cross_day_subject_matrices"
    matrix_dir.mkdir(parents=True, exist_ok=True)
    subject_rows = []
    matrix_accum = {}
    qc_rows = []
    time_template = None

    for result in results:
        if not result["ok"]:
            qc_rows.append(result["qc"])
            continue
        row = result["row"]
        mat = np.asarray(result["mat"], dtype=float)
        t_vec = np.asarray(result["t"], dtype=float)
        subject_rows.append(row)
        np.savez_compressed(
            matrix_dir / f"sub_{int(row['subject']):03d}_trainD{int(row['train_day'])}_testD{int(row['test_day'])}.npz",
            auc=mat,
            time_sec=t_vec,
        )
        if time_template is None:
            time_template = t_vec
        key = (int(row["train_day"]), int(row["test_day"]))
        if key not in matrix_accum:
            matrix_accum[key] = {"sum": np.zeros_like(mat, dtype=float), "count": np.zeros_like(mat, dtype=float)}
        valid = np.isfinite(mat)
        matrix_accum[key]["sum"][valid] += mat[valid]
        matrix_accum[key]["count"][valid] += 1.0

    subject_df = pd.DataFrame(subject_rows)
    subject_csv = output_dir / "tg_cross_day_subject_level.csv"
    day_mean_csv = output_dir / "tg_cross_day_day_mean.csv"
    matrix_day_mean_csv = output_dir / "tg_cross_day_timegen_day_mean.csv"
    qc_csv = output_dir / "tg_qc_log.csv"
    subject_df.to_csv(subject_csv, index=False)
    if subject_df.empty:
        pd.DataFrame().to_csv(day_mean_csv, index=False)
    else:
        (
            subject_df.groupby(["train_day", "test_day"], as_index=False)
            .agg(
                auc_mean=("diag_mean_auc", "mean"),
                auc_sem=("diag_mean_auc", _sem),
                n_subjects=("subject", "nunique"),
            )
            .sort_values(["train_day", "test_day"])
            .to_csv(day_mean_csv, index=False)
        )

    matrix_rows = []
    if time_template is not None:
        for (train_day, test_day), acc in sorted(matrix_accum.items()):
            with np.errstate(invalid="ignore", divide="ignore"):
                mean_mat = acc["sum"] / acc["count"]
            for i, train_t in enumerate(time_template):
                for j, test_t in enumerate(time_template):
                    val = mean_mat[i, j]
                    if np.isfinite(val):
                        matrix_rows.append(
                            {
                                "train_day": train_day,
                                "test_day": test_day,
                                "train_time_sec": float(train_t),
                                "test_time_sec": float(test_t),
                                "auc_mean": float(val),
                                "n_subjects": int(acc["count"][i, j]),
                            }
                        )
    pd.DataFrame(matrix_rows).to_csv(matrix_day_mean_csv, index=False)
    pd.DataFrame(qc_rows).to_csv(qc_csv, index=False)
    return {
        "subject_csv": subject_csv,
        "day_mean_csv": day_mean_csv,
        "matrix_day_mean_csv": matrix_day_mean_csv,
        "qc_csv": qc_csv,
        "matrix_dir": matrix_dir,
        "subject_df": subject_df,
        "qc_df": pd.DataFrame(qc_rows),
    }


def run_band_envelope_cross_day_tg(
    bands: dict[str, tuple[float, float]] = BANDS,
    min_epochs: int = 20,
    random_state: int = 42,
    n_workers: int | None = None,
    output_root: Path | str = PROJECT_DIR / "output",
    figures_dir: Path | str = FIGURES_DIR,
):
    """Run cross-day TG on band-limited amplitude envelopes."""
    output_root = Path(output_root)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    sessions = _load_sessions_for_band_tg()
    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))
    band_results = {}
    progress = {}
    for band_name, (fmin, fmax) in bands.items():
        t0 = time.time()
        band_dir = output_root / f"mvpa_tg_band_{band_name}"
        cache_dir = band_dir / "cache_band_envelope_arrays"
        band_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[Band TG] Preparing {band_name} envelope caches for {len(sessions)} sessions "
            f"(n_workers={n_workers})...",
            flush=True,
        )
        prep_tasks = [
            {
                "session_item": item,
                "cache_dir": cache_dir,
                "band_name": band_name,
                "fmin": fmin,
                "fmax": fmax,
                "min_epochs": min_epochs,
                "random_state": random_state,
            }
            for item in sessions
        ]
        prep_results = _parallel_collect(_prepare_band_envelope_cache_from_task, prep_tasks, n_workers)
        prepared = [r for r in prep_results if r["ok"]]
        qc_rows = [r["qc"] for r in prep_results if not r["ok"]]
        day_data = {
            (int(r["subject"]), int(r["day"])): {
                "cache_path": r["cache_path"],
                "session_file": r["session_file"],
            }
            for r in prepared
        }
        pair_items = []
        rng = np.random.default_rng(random_state)
        for subject in sorted({k[0] for k in day_data}):
            days = sorted(k[1] for k in day_data if k[0] == subject)
            for train_day in days:
                for test_day in days:
                    if train_day == test_day:
                        continue
                    pair_items.append(
                        {
                            "subject": subject,
                            "train_day": train_day,
                            "test_day": test_day,
                            "train_cache_path": day_data[(subject, train_day)]["cache_path"],
                            "test_cache_path": day_data[(subject, test_day)]["cache_path"],
                            "train_session_file": day_data[(subject, train_day)]["session_file"],
                            "test_session_file": day_data[(subject, test_day)]["session_file"],
                            "pair_seed": int(rng.integers(0, 2**31 - 1)),
                        }
                    )

        print(
            f"[Band TG] Running {band_name} cross-day TG on {len(pair_items)} directed pairs "
            f"(prepared_sessions={len(prepared)}, n_workers={n_workers})...",
            flush=True,
        )

        if len(pair_items) == 0:
            out = _write_cross_day_outputs([{"ok": False, "qc": r} for r in qc_rows], band_dir)
        else:
            pair_tasks = [{"pair_item": item, "random_state": random_state} for item in pair_items]
            results = _parallel_collect(_process_cross_day_pair_from_task, pair_tasks, n_workers)
            results.extend({"ok": False, "qc": r} for r in qc_rows)
            out = _write_cross_day_outputs(results, band_dir)

        band_results[band_name] = out
        progress[band_name] = {
            "prepared_sessions": len(prepared),
            "cross_day_pairs": len(pair_items),
            "elapsed_sec": time.time() - t0,
            "output_dir": str(band_dir),
            "n_workers": n_workers,
        }
        (band_dir / "band_tg_progress.json").write_text(json.dumps(progress[band_name], indent=2))

    summary = summarize_band_tg_outputs(output_root=output_root, bands=bands, figures_dir=figures_dir)
    band_gradient = run_band_tg_window_gradients(output_root=output_root, bands=bands, figures_dir=figures_dir)
    return {"band_results": band_results, "progress": progress, "summary": summary, "band_gradient": band_gradient}


def run_band_signed_voltage_cross_day_tg(
    bands: dict[str, tuple[float, float]] = BANDS,
    min_epochs: int = 20,
    random_state: int = 42,
    n_workers: int | None = None,
    output_root: Path | str = PROJECT_DIR / "output",
    figures_dir: Path | str = FIGURES_DIR,
):
    """Run cross-day TG on signed band-limited voltages without Hilbert envelope."""
    output_root = Path(output_root)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    sessions = _load_sessions_for_band_tg()
    if n_workers is None:
        n_workers = _default_n_workers()
    n_workers = max(1, int(n_workers))
    band_results = {}
    progress = {}
    for band_name, (fmin, fmax) in bands.items():
        t0 = time.time()
        band_dir = output_root / f"mvpa_tg_band_signed_{band_name}"
        cache_dir = band_dir / "cache_band_signed_arrays"
        band_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"[Band signed TG] Preparing {band_name} signed-voltage caches for {len(sessions)} sessions "
            f"(n_workers={n_workers})...",
            flush=True,
        )
        prep_tasks = [
            {
                "session_item": item,
                "cache_dir": cache_dir,
                "band_name": band_name,
                "fmin": fmin,
                "fmax": fmax,
                "min_epochs": min_epochs,
                "random_state": random_state,
            }
            for item in sessions
        ]
        prep_results = _parallel_collect(_prepare_band_signed_cache_from_task, prep_tasks, n_workers)
        prepared = [r for r in prep_results if r["ok"]]
        qc_rows = [r["qc"] for r in prep_results if not r["ok"]]
        day_data = {
            (int(r["subject"]), int(r["day"])): {
                "cache_path": r["cache_path"],
                "session_file": r["session_file"],
            }
            for r in prepared
        }
        pair_items = []
        rng = np.random.default_rng(random_state)
        for subject in sorted({k[0] for k in day_data}):
            days = sorted(k[1] for k in day_data if k[0] == subject)
            for train_day in days:
                for test_day in days:
                    if train_day == test_day:
                        continue
                    pair_items.append(
                        {
                            "subject": subject,
                            "train_day": train_day,
                            "test_day": test_day,
                            "train_cache_path": day_data[(subject, train_day)]["cache_path"],
                            "test_cache_path": day_data[(subject, test_day)]["cache_path"],
                            "train_session_file": day_data[(subject, train_day)]["session_file"],
                            "test_session_file": day_data[(subject, test_day)]["session_file"],
                            "pair_seed": int(rng.integers(0, 2**31 - 1)),
                        }
                    )

        print(
            f"[Band signed TG] Running {band_name} cross-day TG on {len(pair_items)} directed pairs "
            f"(prepared_sessions={len(prepared)}, n_workers={n_workers})...",
            flush=True,
        )
        if len(pair_items) == 0:
            out = _write_cross_day_outputs([{"ok": False, "qc": r} for r in qc_rows], band_dir)
        else:
            pair_tasks = [{"pair_item": item, "random_state": random_state} for item in pair_items]
            results = _parallel_collect(_process_cross_day_pair_from_task, pair_tasks, n_workers)
            results.extend({"ok": False, "qc": r} for r in qc_rows)
            out = _write_cross_day_outputs(results, band_dir)

        band_results[band_name] = out
        progress[band_name] = {
            "prepared_sessions": len(prepared),
            "cross_day_pairs": len(pair_items),
            "elapsed_sec": time.time() - t0,
            "output_dir": str(band_dir),
            "n_workers": n_workers,
        }
        (band_dir / "band_signed_tg_progress.json").write_text(json.dumps(progress[band_name], indent=2))

    summary = summarize_band_signed_tg_outputs(output_root=output_root, bands=bands, figures_dir=figures_dir)
    return {"band_results": band_results, "progress": progress, "summary": summary}


def summarize_band_tg_outputs(
    output_root: Path | str = PROJECT_DIR / "output",
    bands: dict[str, tuple[float, float]] = BANDS,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_root = Path(output_root)
    figures_dir = Path(figures_dir)
    rows = []
    for band_name in bands:
        matrix_dir = output_root / f"mvpa_tg_band_{band_name}" / "tg_cross_day_subject_matrices"
        for path in sorted(matrix_dir.glob("sub_*_trainD*_testD*.npz")):
            parsed = _parse_matrix_path(path)
            if parsed is None:
                continue
            subject, train_day, test_day = parsed
            with np.load(path, allow_pickle=False) as z:
                mat = np.asarray(z["auc"], dtype=float)
                t = np.asarray(z["time_sec"], dtype=float)
            diag = np.diag(mat)
            for time_sec, auc in zip(t, diag):
                rows.append(
                    {
                        "band": band_name,
                        "subject": subject,
                        "train_day": train_day,
                        "test_day": test_day,
                        "time_sec": float(time_sec),
                        "auc": float(auc),
                    }
                )
    diag_df = pd.DataFrame(rows)
    diag_csv = OUTPUT_DIR / "band_tg_diagonal_timecourse_subject_pairs.csv"
    mean_csv = OUTPUT_DIR / "band_tg_diagonal_timecourse_mean.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(diag_csv, index=False)
    if diag_df.empty:
        pd.DataFrame().to_csv(mean_csv, index=False)
        return {"diag_csv": diag_csv, "mean_csv": mean_csv, "figure": None}
    subject_df = (
        diag_df.groupby(["band", "subject", "time_sec"], as_index=False)["auc"]
        .mean()
        .sort_values(["band", "subject", "time_sec"])
    )
    mean_df = (
        subject_df.groupby(["band", "time_sec"], as_index=False)
        .agg(auc_mean=("auc", "mean"), auc_sem=("auc", _sem), n_subjects=("subject", "nunique"))
        .sort_values(["band", "time_sec"])
    )
    mean_df.to_csv(mean_csv, index=False)

    fig_path = figures_dir / "band_tg_diagonal_timecourses.png"
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for band_name, g in mean_df.groupby("band"):
        g = g.sort_values("time_sec")
        ax.plot(g["time_sec"], g["auc_mean"], linewidth=1.8, label=band_name)
        ax.fill_between(g["time_sec"], g["auc_mean"] - g["auc_sem"], g["auc_mean"] + g["auc_sem"], alpha=0.15)
    ax.axhline(0.5, color="0.3", linestyle=":", linewidth=1)
    ax.axvline(0.0, color="0.5", linestyle=":", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Diagonal AUC")
    ax.set_title("Band-Envelope Cross-Day TG Diagonal Timecourse")
    ax.legend(title="Band")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"diag_csv": diag_csv, "mean_csv": mean_csv, "figure": fig_path}


def summarize_band_signed_tg_outputs(
    output_root: Path | str = PROJECT_DIR / "output",
    bands: dict[str, tuple[float, float]] = BANDS,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_root = Path(output_root)
    figures_dir = Path(figures_dir)
    rows = []
    for band_name in bands:
        matrix_dir = output_root / f"mvpa_tg_band_signed_{band_name}" / "tg_cross_day_subject_matrices"
        for path in sorted(matrix_dir.glob("sub_*_trainD*_testD*.npz")):
            parsed = _parse_matrix_path(path)
            if parsed is None:
                continue
            subject, train_day, test_day = parsed
            with np.load(path, allow_pickle=False) as z:
                mat = np.asarray(z["auc"], dtype=float)
                t = np.asarray(z["time_sec"], dtype=float)
            diag = np.diag(mat)
            for time_sec, auc in zip(t, diag):
                rows.append(
                    {
                        "band": band_name,
                        "subject": subject,
                        "train_day": train_day,
                        "test_day": test_day,
                        "time_sec": float(time_sec),
                        "auc": float(auc),
                    }
                )
    diag_df = pd.DataFrame(rows)
    diag_csv = OUTPUT_DIR / "band_signed_tg_diagonal_timecourse_subject_pairs.csv"
    mean_csv = OUTPUT_DIR / "band_signed_tg_diagonal_timecourse_mean.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diag_df.to_csv(diag_csv, index=False)
    if diag_df.empty:
        pd.DataFrame().to_csv(mean_csv, index=False)
        return {"diag_csv": diag_csv, "mean_csv": mean_csv, "figure": None}
    subject_df = (
        diag_df.groupby(["band", "subject", "time_sec"], as_index=False)["auc"]
        .mean()
        .sort_values(["band", "subject", "time_sec"])
    )
    mean_df = (
        subject_df.groupby(["band", "time_sec"], as_index=False)
        .agg(auc_mean=("auc", "mean"), auc_sem=("auc", _sem), n_subjects=("subject", "nunique"))
        .sort_values(["band", "time_sec"])
    )
    mean_df.to_csv(mean_csv, index=False)

    fig_path = figures_dir / "band_signed_tg_diagonal_timecourses.png"
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for band_name, g in mean_df.groupby("band"):
        g = g.sort_values("time_sec")
        ax.plot(g["time_sec"], g["auc_mean"], linewidth=1.8, label=band_name)
        ax.fill_between(g["time_sec"], g["auc_mean"] - g["auc_sem"], g["auc_mean"] + g["auc_sem"], alpha=0.15)
    ax.axhline(0.5, color="0.3", linestyle=":", linewidth=1)
    ax.axvline(0.0, color="0.5", linestyle=":", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Diagonal AUC")
    ax.set_title("Band-Limited Signed-Voltage Cross-Day TG Diagonal Timecourse")
    ax.legend(title="Band")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"diag_csv": diag_csv, "mean_csv": mean_csv, "figure": fig_path}


def run_band_tg_window_gradients(
    output_root: Path | str = PROJECT_DIR / "output",
    bands: dict[str, tuple[float, float]] = BANDS,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_root = Path(output_root)
    figures_dir = Path(figures_dir)
    rows = []
    slope_rows = []
    interaction_rows = []
    for band_name in bands:
        matrix_dir = output_root / f"mvpa_tg_band_{band_name}" / "tg_cross_day_subject_matrices"
        if not matrix_dir.exists():
            continue
        d = extract_tg_window_auc(matrix_dir=matrix_dir)
        if d.empty:
            continue
        d["band"] = band_name
        rows.append(d)
        slope_df, interaction_summary, _, _ = fit_tg_window_gradients(d)
        slope_df["band"] = band_name
        slope_rows.append(slope_df)
        interaction_summary["band"] = band_name
        interaction_rows.append(interaction_summary)
    window_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    slope_df = pd.concat(slope_rows, ignore_index=True) if slope_rows else pd.DataFrame()
    interaction_df = pd.DataFrame(interaction_rows)
    window_csv = OUTPUT_DIR / "band_tg_window_auc_subject_pairs.csv"
    slope_csv = OUTPUT_DIR / "band_tg_window_gradient_slopes.csv"
    interaction_csv = OUTPUT_DIR / "band_tg_window_slope_differences.csv"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    window_df.to_csv(window_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    interaction_df.to_csv(interaction_csv, index=False)
    fig_path = None
    if not slope_df.empty:
        fig_path = figures_dir / "band_tg_window_gradient_slopes.png"
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        band_order = [band for band in bands if band in set(slope_df["band"])]
        offsets = {"early": -0.16, "late": 0.16}
        colors = {"early": "tab:blue", "late": "tab:orange"}
        x_base = np.arange(len(band_order), dtype=float)
        for window_name in ["early", "late"]:
            g = slope_df[slope_df["window"] == window_name].set_index("band").reindex(band_order)
            x = x_base + offsets[window_name]
            y = g["estimate"].to_numpy(dtype=float)
            yerr = np.vstack(
                [
                    y - g["ci_low"].to_numpy(dtype=float),
                    g["ci_high"].to_numpy(dtype=float) - y,
                ]
            )
            ax.errorbar(
                x,
                y,
                yerr=yerr,
                fmt="o",
                capsize=3,
                color=colors[window_name],
                label=window_name,
            )
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1)
        ax.set_xticks(x_base)
        ax.set_xticklabels(band_order)
        ax.set_xlabel("Band")
        ax.set_ylabel("AUC slope per day distance")
        ax.set_title("Band-Envelope TG Window Gradients")
        ax.legend(title="Window")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return {
        "window_csv": window_csv,
        "slope_csv": slope_csv,
        "interaction_csv": interaction_csv,
        "figure": fig_path,
        "window_df": window_df,
        "slope_df": slope_df,
        "interaction_df": interaction_df,
    }


def _load_diagonal_timecourses_from_matrices(matrix_dir, signal_name):
    rows = []
    matrix_dir = Path(matrix_dir)
    for path in sorted(matrix_dir.glob("sub_*_trainD*_testD*.npz")):
        parsed = _parse_matrix_path(path)
        if parsed is None:
            continue
        subject, train_day, test_day = parsed
        with np.load(path, allow_pickle=False) as z:
            mat = np.asarray(z["auc"], dtype=float)
            time_sec = np.asarray(z["time_sec"], dtype=float)
        diag = np.diag(mat)
        if (train_day == 1) and (test_day != 1):
            pair_type = "day1_forward"
        elif (train_day != 1) and (test_day == 1):
            pair_type = "day1_backward"
        elif (train_day != 1) and (test_day != 1):
            pair_type = "later_only"
        else:
            continue
        for t, auc in zip(time_sec, diag):
            rows.append(
                {
                    "signal": signal_name,
                    "subject": subject,
                    "train_day": train_day,
                    "test_day": test_day,
                    "day_distance": abs(train_day - test_day),
                    "pair_type": pair_type,
                    "time_sec": float(t),
                    "auc": float(auc),
                }
            )
    return rows


def _summarize_diagonal_by_signal_pair(diag_df):
    subject_df = (
        diag_df.groupby(["signal", "pair_type", "subject", "time_sec"], as_index=False)["auc"]
        .mean()
        .sort_values(["signal", "pair_type", "subject", "time_sec"])
    )
    mean_df = (
        subject_df.groupby(["signal", "pair_type", "time_sec"], as_index=False)
        .agg(auc_mean=("auc", "mean"), auc_sem=("auc", _sem), n_subjects=("subject", "nunique"))
        .sort_values(["signal", "pair_type", "time_sec"])
    )
    window_rows = []
    for signal, pair_type, subject in subject_df[["signal", "pair_type", "subject"]].drop_duplicates().itertuples(index=False):
        g = subject_df[
            (subject_df["signal"] == signal)
            & (subject_df["pair_type"] == pair_type)
            & (subject_df["subject"] == subject)
        ]
        for window_name, (tmin, tmax) in TG_WINDOWS.items():
            w = g[(g["time_sec"] >= tmin) & (g["time_sec"] <= tmax)]
            if w.empty:
                continue
            window_rows.append(
                {
                    "signal": signal,
                    "pair_type": pair_type,
                    "subject": subject,
                    "window": window_name,
                    "auc": float(w["auc"].mean()),
                }
            )
    window_subject_df = pd.DataFrame(window_rows)
    window_mean_df = (
        window_subject_df.groupby(["signal", "pair_type", "window"], as_index=False)
        .agg(auc_mean=("auc", "mean"), auc_sem=("auc", _sem), n_subjects=("subject", "nunique"))
        .sort_values(["signal", "pair_type", "window"])
    )
    return subject_df, mean_df, window_subject_df, window_mean_df


def plot_broadband_vs_band_pair_diagnostics(mean_df, fig_path):
    signal_order = [
        "broadband",
        "theta_envelope",
        "theta_signed",
        "alpha_envelope",
        "alpha_signed",
        "beta_envelope",
        "beta_signed",
        "gamma_envelope",
        "gamma_signed",
    ]
    signal_order = [signal for signal in signal_order if signal in set(mean_df["signal"])]
    pair_order = ["day1_forward", "day1_backward", "later_only"]
    pair_labels = {
        "day1_forward": "D1 -> later",
        "day1_backward": "Later -> D1",
        "later_only": "Later only",
    }
    colors = {
        "broadband": "black",
        "theta_envelope": "tab:blue",
        "theta_signed": "navy",
        "alpha_envelope": "tab:orange",
        "alpha_signed": "darkorange",
        "beta_envelope": "tab:green",
        "beta_signed": "darkgreen",
        "gamma_envelope": "tab:red",
        "gamma_signed": "darkred",
    }

    fig, axes = plt.subplots(len(pair_order), 1, figsize=(10.5, 9.2), sharex=True, sharey=True)
    if len(pair_order) == 1:
        axes = [axes]
    for ax, pair_type in zip(axes, pair_order):
        for signal in signal_order:
            g = mean_df[(mean_df["signal"] == signal) & (mean_df["pair_type"] == pair_type)].sort_values("time_sec")
            if g.empty:
                continue
            ax.plot(g["time_sec"], g["auc_mean"], color=colors.get(signal, None), linewidth=1.8, label=signal)
            ax.fill_between(
                g["time_sec"],
                g["auc_mean"] - g["auc_sem"],
                g["auc_mean"] + g["auc_sem"],
                color=colors.get(signal, None),
                alpha=0.10,
                linewidth=0,
            )
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.axvspan(TG_WINDOWS["early"][0], TG_WINDOWS["early"][1], color="tab:blue", alpha=0.08, linewidth=0)
        ax.axvspan(TG_WINDOWS["late"][0], TG_WINDOWS["late"][1], color="tab:orange", alpha=0.08, linewidth=0)
        ax.set_title(pair_labels[pair_type])
        ax.set_ylabel("Diagonal AUC")
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("Time from stimulus onset (s)")
    axes[0].legend(loc="best", ncol=3, fontsize=8)
    fig.suptitle("Broadband vs Band-Envelope/Signed Cross-Day TG by Day-Pair Type")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_broadband_vs_band_window_bars(window_mean_df, fig_path):
    signal_order = [
        "broadband",
        "theta_envelope",
        "theta_signed",
        "alpha_envelope",
        "alpha_signed",
        "beta_envelope",
        "beta_signed",
        "gamma_envelope",
        "gamma_signed",
    ]
    signal_order = [signal for signal in signal_order if signal in set(window_mean_df["signal"])]
    pair_order = ["day1_forward", "day1_backward", "later_only"]
    pair_labels = ["D1 -> later", "Later -> D1", "Later only"]
    colors = {"early": "tab:blue", "late": "tab:orange"}

    fig, axes = plt.subplots(1, len(pair_order), figsize=(13, 4), sharey=True, squeeze=False)
    width = 0.35
    x = np.arange(len(signal_order), dtype=float)
    for ax, pair_type, pair_label in zip(axes.ravel(), pair_order, pair_labels):
        for i_window, window_name in enumerate(["early", "late"]):
            g = (
                window_mean_df[
                    (window_mean_df["pair_type"] == pair_type)
                    & (window_mean_df["window"] == window_name)
                ]
                .set_index("signal")
                .reindex(signal_order)
            )
            offset = (-width / 2) if window_name == "early" else (width / 2)
            ax.bar(
                x + offset,
                g["auc_mean"],
                width=width,
                yerr=g["auc_sem"],
                color=colors[window_name],
                alpha=0.75,
                capsize=3,
                label=window_name if pair_type == pair_order[0] else None,
            )
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels(signal_order, rotation=35, ha="right")
        ax.set_title(pair_label)
        ax.grid(axis="y", alpha=0.22)
    axes.ravel()[0].set_ylabel("Mean window diagonal AUC")
    axes.ravel()[0].legend(loc="best")
    fig.suptitle("Early/Late TG by Signal Type and Day-Pair Type")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_broadband_vs_band_diagnostic(
    output_root: Path | str = PROJECT_DIR / "output",
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_root = Path(output_root)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_diagonal_timecourses_from_matrices(
        output_root / "mvpa_tg_cross_day" / "tg_cross_day_subject_matrices",
        "broadband",
    )
    for band_name in ["theta", "alpha", "beta", "gamma"]:
        rows.extend(
            _load_diagonal_timecourses_from_matrices(
                output_root / f"mvpa_tg_band_{band_name}" / "tg_cross_day_subject_matrices",
                f"{band_name}_envelope",
            )
        )
        rows.extend(
            _load_diagonal_timecourses_from_matrices(
                output_root / f"mvpa_tg_band_signed_{band_name}" / "tg_cross_day_subject_matrices",
                f"{band_name}_signed",
            )
        )
    diag_df = pd.DataFrame(rows)
    if diag_df.empty:
        raise RuntimeError("No TG diagonal matrices found for broadband-vs-band diagnostic.")
    subject_df, mean_df, window_subject_df, window_mean_df = _summarize_diagonal_by_signal_pair(diag_df)

    diag_csv = output_dir / "tg_broadband_vs_band_diagonal_subject_pairs.csv"
    mean_csv = output_dir / "tg_broadband_vs_band_diagonal_mean.csv"
    window_subject_csv = output_dir / "tg_broadband_vs_band_window_subject.csv"
    window_mean_csv = output_dir / "tg_broadband_vs_band_window_mean.csv"
    fig_timecourse = figures_dir / "tg_broadband_vs_band_pair_type_timecourses.png"
    fig_window = figures_dir / "tg_broadband_vs_band_window_bars.png"

    diag_df.to_csv(diag_csv, index=False)
    mean_df.to_csv(mean_csv, index=False)
    window_subject_df.to_csv(window_subject_csv, index=False)
    window_mean_df.to_csv(window_mean_csv, index=False)
    plot_broadband_vs_band_pair_diagnostics(mean_df, fig_timecourse)
    plot_broadband_vs_band_window_bars(window_mean_df, fig_window)
    return {
        "diag_csv": diag_csv,
        "mean_csv": mean_csv,
        "window_subject_csv": window_subject_csv,
        "window_mean_csv": window_mean_csv,
        "figures": {
            "timecourse": fig_timecourse,
            "window": fig_window,
        },
        "window_mean_df": window_mean_df,
    }


def _flat_peak(values, eps=1e-12):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return True
    return float(np.nanmax(finite) - np.nanmin(finite)) <= eps


def _erp_component_for_group(group, component, level):
    if component == "frontal_n2":
        channels = ["Fz", "FCz", "FC1", "FC2"]
        g = group[group["channel"].isin(channels)]
        if g["channel"].nunique() < len(channels):
            return None
        wave = g.groupby("time_s", as_index=False)["amplitude_v"].mean().sort_values("time_s")
        search = wave[(wave["time_s"] >= 0.150) & (wave["time_s"] <= 0.400)]
        if search.empty or _flat_peak(search["amplitude_v"]):
            peak_time = np.nan
            peak_amp = np.nan
        else:
            idx = search["amplitude_v"].idxmin()
            peak_time = float(search.loc[idx, "time_s"])
            peak_amp = float(search.loc[idx, "amplitude_v"])
        return {
            "component": component,
            "lock_type": "stim",
            "latency_s": peak_time,
            "peak_amplitude_v": peak_amp,
            "level": level,
            "n_channels": int(g["channel"].nunique()),
        }

    if component in {"motor_lrp_stim", "motor_lrp_response"}:
        left = ["C3", "CP3", "C5"]
        right = ["C4", "CP4", "C6"]
        g = group[group["channel"].isin(left + right)]
        if any(ch not in set(g["channel"]) for ch in left + right):
            return None
        left_wave = g[g["channel"].isin(left)].groupby("time_s")["amplitude_v"].mean()
        right_wave = g[g["channel"].isin(right)].groupby("time_s")["amplitude_v"].mean()
        wave = (left_wave - right_wave).rename("lateralisation_v").reset_index().sort_values("time_s")
        if component == "motor_lrp_stim":
            search = wave[(wave["time_s"] >= 0.150) & (wave["time_s"] <= 0.500)]
            lock_type = "stim"
            latency_sign = 1.0
        else:
            search = wave[(wave["time_s"] >= 0.0) & (wave["time_s"] <= 0.400)]
            lock_type = "response"
            latency_sign = -1.0
        if search.empty or _flat_peak(search["lateralisation_v"]):
            peak_time = np.nan
            peak_amp = np.nan
        else:
            idx = search["lateralisation_v"].abs().idxmax()
            peak_time = latency_sign * float(search.loc[idx, "time_s"])
            peak_amp = float(search.loc[idx, "lateralisation_v"])
        return {
            "component": component,
            "lock_type": lock_type,
            "latency_s": peak_time,
            "peak_amplitude_v": peak_amp,
            "level": level,
            "n_channels": int(g["channel"].nunique()),
        }
    raise ValueError(component)


def compute_erp_peak_latencies(
    erp_subject_csv: Path | str = PROJECT_DIR / "output" / "erp" / "erp_subject_day_all.csv",
    erp_grand_csv: Path | str = PROJECT_DIR / "output" / "erp" / "erp_grand_averages_by_day_lock_condition.csv",
):
    rows = []
    subject_path = Path(erp_subject_csv)
    if subject_path.exists():
        d_sub = pd.read_csv(subject_path)
        d_sub = d_sub[d_sub["condition"] == "all"].copy()
        for (subject, day), g_day in d_sub.groupby(["subject", "day"]):
            stim = g_day[g_day["lock_type"] == "stim"]
            resp = g_day[g_day["lock_type"] == "response"]
            for component, group in [
                ("frontal_n2", stim),
                ("motor_lrp_stim", stim),
                ("motor_lrp_response", resp),
            ]:
                item = _erp_component_for_group(group, component, "subject")
                if item is not None:
                    item.update({"subject": int(subject), "day": int(day)})
                    rows.append(item)

    grand_path = Path(erp_grand_csv)
    if grand_path.exists():
        d_grand = pd.read_csv(grand_path)
        d_grand = d_grand[d_grand["condition"] == "all"].copy()
        for day, g_day in d_grand.groupby("day"):
            stim = g_day[g_day["lock_type"] == "stim"]
            resp = g_day[g_day["lock_type"] == "response"]
            for component, group in [
                ("frontal_n2", stim),
                ("motor_lrp_stim", stim),
                ("motor_lrp_response", resp),
            ]:
                item = _erp_component_for_group(group, component, "grand")
                if item is not None:
                    item.update({"subject": np.nan, "day": int(day)})
                    rows.append(item)
    return pd.DataFrame(rows)


def fit_erp_latency_slopes(latency_df):
    rows = []
    models = {}
    d = latency_df[(latency_df["level"] == "subject") & latency_df["latency_s"].notna()].copy()
    for component, g in d.groupby("component"):
        if g["day"].nunique() < 2 or g["subject"].nunique() < 2:
            continue
        model = _fit_ols("latency_s ~ day", g)
        item = _model_term_summary(model, "day")
        item.update({"component": component, "n_rows": int(len(g)), "n_subjects": int(g["subject"].nunique())})
        rows.append(item)
        models[component] = model

    comparisons = []
    for a, b in [("frontal_n2", "motor_lrp_stim"), ("frontal_n2", "motor_lrp_response")]:
        g = d[d["component"].isin([a, b])].copy()
        if g["component"].nunique() < 2:
            continue
        g["component"] = pd.Categorical(g["component"], categories=[a, b])
        model = _fit_ols("latency_s ~ day * component", g)
        term = f"day:component[T.{b}]"
        item = _model_term_summary(model, term)
        item.update({"comparison": f"{b}_minus_{a}_slope"})
        comparisons.append(item)
    return pd.DataFrame(rows), pd.DataFrame(comparisons), models


def plot_erp_latency_trajectories(latency_df, slope_df, fig_path):
    d_sub = latency_df[(latency_df["level"] == "subject") & latency_df["latency_s"].notna()].copy()
    d_grand = latency_df[(latency_df["level"] == "grand") & latency_df["latency_s"].notna()].copy()
    components = ["frontal_n2", "motor_lrp_stim", "motor_lrp_response"]
    labels = {
        "frontal_n2": "Frontal N2",
        "motor_lrp_stim": "Motor lateralisation (stim)",
        "motor_lrp_response": "Motor lateralisation (response)",
    }
    colors = {
        "frontal_n2": "tab:blue",
        "motor_lrp_stim": "tab:orange",
        "motor_lrp_response": "tab:green",
    }
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    for component in components:
        g = d_sub[d_sub["component"] == component]
        if not g.empty:
            for _, gs in g.groupby("subject"):
                ax.plot(gs["day"], gs["latency_s"] * 1000.0, color=colors[component], alpha=0.12, linewidth=0.8)
            mean = (
                g.groupby("day", as_index=False)
                .agg(latency_mean=("latency_s", "mean"), latency_sem=("latency_s", _sem))
                .sort_values("day")
            )
            ax.errorbar(
                mean["day"],
                mean["latency_mean"] * 1000.0,
                yerr=mean["latency_sem"] * 1000.0,
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=colors[component],
                label=labels[component],
            )
        gg = d_grand[d_grand["component"] == component].sort_values("day")
        if not gg.empty:
            ax.plot(
                gg["day"],
                gg["latency_s"] * 1000.0,
                color=colors[component],
                linestyle="--",
                linewidth=1.4,
                alpha=0.85,
            )
    ax.axhline(0.0, color="0.4", linestyle=":", linewidth=1)
    ax.set_xlabel("Day")
    ax.set_ylabel("Peak latency (ms)")
    ax.set_title("ERP Peak Latency Trajectories")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_erp_peak_latency_trajectories(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    latency_df = compute_erp_peak_latencies()
    slope_df, comparison_df, _ = fit_erp_latency_slopes(latency_df)
    latency_csv = output_dir / "erp_peak_latencies.csv"
    slope_csv = output_dir / "erp_latency_day_slopes.csv"
    comparison_csv = output_dir / "erp_latency_slope_comparisons.csv"
    fig_path = figures_dir / "erp_peak_latency_trajectories.png"
    latency_df.to_csv(latency_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    comparison_df.to_csv(comparison_csv, index=False)
    plot_erp_latency_trajectories(latency_df, slope_df, fig_path)
    return {
        "latency_df": latency_df,
        "slope_df": slope_df,
        "comparison_df": comparison_df,
        "latency_csv": latency_csv,
        "slope_csv": slope_csv,
        "comparison_csv": comparison_csv,
        "figure": fig_path,
    }


def run_all_new_analyses(run_band_tg=False, n_workers=None):
    results = {
        "tg_window_structure": run_cross_day_tg_window_structure(),
        "day1_distinctiveness": run_day1_distinctiveness_analysis(),
        "erp_peak_latencies": run_erp_peak_latency_trajectories(),
    }
    if run_band_tg:
        results["band_tg"] = run_band_envelope_cross_day_tg(n_workers=n_workers)
        results["band_signed_tg"] = run_band_signed_voltage_cross_day_tg(n_workers=n_workers)
    results["broadband_vs_band_diagnostic"] = run_broadband_vs_band_diagnostic()
    return results


if __name__ == "__main__":
    run_all_new_analyses(run_band_tg=False)
