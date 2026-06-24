#!/usr/bin/env python3
"""Diagnostic figures linking optimal GLC evidence to MVPA block-model evidence."""

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

from block_model_utils import sem

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"


def _prefix(output_label):
    return "decision_bound" if not output_label else f"decision_bound_{output_label}"


def _require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing input: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty input: {path}")
    return d


def load_inputs(output_dir, output_label):
    prefix = _prefix(output_label)
    weights = _require_csv(output_dir / f"{prefix}_block_model_weights.csv")
    mvpa = _require_csv(output_dir / "mvpa_block_model_timecourse_subject.csv")
    return weights, mvpa


def glc_matrix(weights):
    glc = weights[weights["model"] == "glc"].copy()
    mat = glc.pivot_table(
        index="subject",
        columns="block",
        values="glc_bic_advantage_vs_best_1d",
        aggfunc="mean",
    )
    mat = mat.reindex(columns=np.arange(1, 26))
    return mat.sort_index()


def classify_subjects(glc_mat, threshold=6.0):
    rows = []
    for subject, values in glc_mat.iterrows():
        v = values.to_numpy(dtype=float)
        strong = np.isfinite(v) & (v >= float(threshold))
        group = "weak_or_no_glc"
        first_pair = np.nan
        for i in range(len(strong) - 1):
            if strong[i] and strong[i + 1]:
                first_pair = i + 1
                group = "already_glc" if first_pair <= 1 else "later_glc"
                break
        rows.append(
            {
                "subject": int(subject),
                "behavior_group": group,
                "first_strong_glc_pair_block": first_pair,
                "early_glc_mean_blocks_1_2": float(np.nanmean(v[:2])),
                "early_glc_mean_blocks_1_5": float(np.nanmean(v[:5])),
                "max_glc_advantage": float(np.nanmax(v)),
            }
        )
    d = pd.DataFrame(rows)
    group_order = {"already_glc": 0, "later_glc": 1, "weak_or_no_glc": 2}
    d["group_order"] = d["behavior_group"].map(group_order)
    return d.sort_values(["group_order", "early_glc_mean_blocks_1_2", "subject"], ascending=[True, False, True])


def mvpa_split_evidence(mvpa, tmin=0.3, tmax=0.6, metric="baseline"):
    d = mvpa[
        (mvpa["model"] == "discrete")
        & np.isfinite(mvpa["split_block"])
        & (mvpa["time_sec"] >= float(tmin))
        & (mvpa["time_sec"] <= float(tmax))
    ].copy()
    if d.empty:
        raise ValueError("No MVPA discrete rows in requested diagnostic window")
    if metric == "baseline":
        value_col = "evidence_over_baseline"
        d[value_col] = -d["delta_bic_baseline"].astype(float)
    elif metric == "relative_best":
        value_col = "evidence_relative_best"
        d[value_col] = -d["delta_bic_best"].astype(float)
    else:
        raise ValueError(f"unknown MVPA split metric: {metric}")
    split = (
        d.groupby(["subject", "split_block"], as_index=False)
        .agg(mean_evidence=(value_col, "mean"))
        .sort_values(["subject", "split_block"])
    )
    mat = split.pivot_table(
        index="subject",
        columns="split_block",
        values="mean_evidence",
        aggfunc="mean",
    )
    mat = mat.reindex(columns=np.arange(1, 25))
    return mat.sort_index()


def save_glc_heatmap(glc_mat, groups, figures_dir, output_label, threshold):
    fig_path = figures_dir / f"{_prefix(output_label)}_diagnostic_glc_evidence_heatmap.png"
    order = groups["subject"].tolist()
    mat = glc_mat.reindex(order)
    finite = mat.to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    vmax = max(10.0, min(60.0, float(np.nanpercentile(np.abs(finite), 90)))) if len(finite) else 20.0
    fig, ax = plt.subplots(figsize=(10.8, max(4.8, 0.28 * len(order) + 1.4)))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for boundary in [4.5, 9.5, 14.5, 19.5]:
        ax.axvline(boundary, color="0.65", linewidth=0.9)
    for y, group in enumerate(groups["behavior_group"].tolist()):
        if y > 0 and group != groups["behavior_group"].iloc[y - 1]:
            ax.axhline(y - 0.5, color="black", linewidth=1.0)
    ax.set_xticks(np.arange(0, 25, 2))
    ax.set_xticklabels([str(x) for x in range(1, 26, 2)])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(s) for s in order])
    ax.set_xlabel("Accumulated block")
    ax.set_ylabel("Subject")
    ax.set_title(f"Optimal GLC Evidence by Subject and Block (threshold={threshold:g})")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("GLC BIC advantage over best 1D")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_linked_heatmap(glc_mat, mvpa_mat, groups, figures_dir, output_label, tmin, tmax, suffix, mvpa_label, relative=False):
    fig_path = figures_dir / f"{_prefix(output_label)}_diagnostic_glc_mvpa_{suffix}_linked_heatmaps.png"
    order = groups["subject"].tolist()
    glc = glc_mat.reindex(order)
    mvpa = mvpa_mat.reindex(order)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, max(4.8, 0.28 * len(order) + 1.4)), sharey=True)

    glc_vals = glc.to_numpy(dtype=float)
    finite_glc = glc_vals[np.isfinite(glc_vals)]
    glc_vmax = max(10.0, min(60.0, float(np.nanpercentile(np.abs(finite_glc), 90)))) if len(finite_glc) else 20.0
    im0 = axes[0].imshow(glc_vals, aspect="auto", cmap="RdBu_r", vmin=-glc_vmax, vmax=glc_vmax)

    mvpa_vals = mvpa.to_numpy(dtype=float)
    finite_mvpa = mvpa_vals[np.isfinite(mvpa_vals)]
    if relative:
        mvpa_vmin = min(-5.0, max(-80.0, float(np.nanpercentile(finite_mvpa, 10)))) if len(finite_mvpa) else -10.0
        mvpa_vmax = 0.0
        mvpa_cmap = "magma"
    else:
        mvpa_vmin = np.nanmin(finite_mvpa) if len(finite_mvpa) else 0.0
        mvpa_vmax = max(5.0, min(80.0, float(np.nanpercentile(np.abs(finite_mvpa), 90)))) if len(finite_mvpa) else 10.0
        mvpa_cmap = "viridis"
    im1 = axes[1].imshow(mvpa_vals, aspect="auto", cmap=mvpa_cmap, vmin=mvpa_vmin, vmax=mvpa_vmax)

    for ax, n_cols, title in [
        (axes[0], 25, "Behavior: optimal GLC evidence"),
        (axes[1], 24, f"MVPA split evidence: {mvpa_label} ({tmin:.1f}-{tmax:.1f} s)"),
    ]:
        for boundary in [4.5, 9.5, 14.5, 19.5]:
            ax.axvline(boundary, color="0.65" if ax is axes[0] else "white", linewidth=0.9)
        for y, group in enumerate(groups["behavior_group"].tolist()):
            if y > 0 and group != groups["behavior_group"].iloc[y - 1]:
                ax.axhline(y - 0.5, color="black", linewidth=1.0)
        ax.set_xticks(np.arange(0, n_cols, 2))
        ax.set_xticklabels([str(x) for x in range(1, n_cols + 1, 2)])
        ax.set_xlabel("Accumulated block" if ax is axes[0] else "Candidate MVPA split block")
        ax.set_title(title)
    axes[0].set_yticks(np.arange(len(order)))
    axes[0].set_yticklabels([str(s) for s in order])
    axes[0].set_ylabel("Subject, sorted by behavioral GLC group")
    c0 = fig.colorbar(im0, ax=axes[0], pad=0.02)
    c0.set_label("GLC BIC advantage")
    c1 = fig.colorbar(im1, ax=axes[1], pad=0.02)
    c1.set_label(mvpa_label)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def _mvpa_timecourse_table(mvpa, metric="baseline"):
    d = mvpa[mvpa["model"].isin(["continuous", "discrete"])].copy()
    if metric == "baseline":
        value_col = "evidence_over_baseline"
        d[value_col] = -d["delta_bic_baseline"].astype(float)
    elif metric == "relative_best":
        value_col = "evidence_relative_best"
        d[value_col] = -d["delta_bic_best"].astype(float)
    else:
        raise ValueError(f"unknown MVPA timecourse metric: {metric}")
    cont = d[d["model"] == "continuous"].copy()
    cont["mvpa_model_set"] = "continuous"
    disc = d[d["model"] == "discrete"].copy()
    early = (
        disc[disc["split_block"].astype(float) <= 5]
        .groupby(["subject", "time_sec"], as_index=False)
        .agg(evidence=(value_col, "max"))
    )
    early["mvpa_model_set"] = "day1_discrete"
    late = (
        disc[disc["split_block"].astype(float) > 5]
        .groupby(["subject", "time_sec"], as_index=False)
        .agg(evidence=(value_col, "max"))
    )
    late["mvpa_model_set"] = "later_discrete"
    cont = cont[["subject", "time_sec", "mvpa_model_set", value_col]].rename(columns={value_col: "evidence"})
    return pd.concat([cont, early, late], ignore_index=True)


def save_group_mvpa_timecourses(mvpa, groups, figures_dir, output_label, metric="baseline"):
    suffix = "relative_group_mvpa_timecourses" if metric == "relative_best" else "group_mvpa_timecourses"
    fig_path = figures_dir / f"{_prefix(output_label)}_diagnostic_{suffix}.png"
    d = _mvpa_timecourse_table(mvpa, metric=metric).merge(groups[["subject", "behavior_group"]], on="subject", how="inner")
    model_sets = ["continuous", "day1_discrete", "later_discrete"]
    group_order = ["already_glc", "later_glc", "weak_or_no_glc"]
    colors = {"already_glc": "#4c78a8", "later_glc": "#f58518", "weak_or_no_glc": "#54a24b"}
    labels = {"already_glc": "Already GLC", "later_glc": "Later GLC", "weak_or_no_glc": "Weak/no GLC"}
    titles = {
        "continuous": "Continuous restructuring",
        "day1_discrete": "Best day-1 discrete split",
        "later_discrete": "Best later discrete split",
    }
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.7), sharey=True)
    for ax, model_set in zip(axes, model_sets):
        dd = d[d["mvpa_model_set"] == model_set]
        for group in group_order:
            g = dd[dd["behavior_group"] == group]
            if g.empty:
                continue
            summary = (
                g.groupby("time_sec", as_index=False)
                .agg(
                    mean_evidence=("evidence", "mean"),
                    sem_evidence=("evidence", sem),
                    n_subjects=("subject", "nunique"),
                )
                .sort_values("time_sec")
            )
            x = summary["time_sec"].to_numpy(dtype=float)
            y = summary["mean_evidence"].to_numpy(dtype=float)
            err = summary["sem_evidence"].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, color=colors[group], linewidth=1.8, label=f"{labels[group]} (n={int(summary['n_subjects'].max())})")
            ax.fill_between(x, y - err, y + err, color=colors[group], alpha=0.15, linewidth=0)
        ax.axhline(0, color="0.45", linewidth=0.8)
        ax.axvline(0, color="0.55", linewidth=0.8)
        ax.set_title(titles[model_set])
        ax.set_xlabel("time from stimulus (s)")
        ax.grid(alpha=0.25)
    y_label = "MVPA evidence over baseline" if metric == "baseline" else "MVPA evidence relative to best model"
    axes[0].set_ylabel(y_label)
    axes[-1].legend(frameon=False, fontsize=8)
    title_suffix = "over baseline" if metric == "baseline" else "relative to best model"
    fig.suptitle(f"MVPA Model Evidence by Behavioral GLC Group ({title_suffix})", y=1.02)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def best_mvpa_model_identity(mvpa):
    d = mvpa.copy()
    discrete = d["model"].eq("discrete")
    d["model_set"] = d["model"].astype(str)
    d.loc[discrete & (d["split_block"].astype(float) <= 5), "model_set"] = "day1_discrete"
    d.loc[discrete & (d["split_block"].astype(float) > 5), "model_set"] = "later_discrete"
    idx = d.groupby(["subject", "time_sec"])["bic"].idxmin()
    best = d.loc[idx, ["subject", "time_sec", "model_set"]].copy()
    labels = ["baseline", "continuous", "day1_discrete", "later_discrete"]
    label_to_val = {label: i for i, label in enumerate(labels)}
    best["value"] = best["model_set"].map(label_to_val)
    mat = best.pivot_table(index="subject", columns="time_sec", values="value", aggfunc="first")
    return mat.sort_index(), labels


def save_best_mvpa_identity_heatmap(mvpa, groups, figures_dir, output_label):
    fig_path = figures_dir / f"{_prefix(output_label)}_diagnostic_best_mvpa_model_identity.png"
    mat, labels = best_mvpa_model_identity(mvpa)
    order = groups["subject"].tolist()
    mat = mat.reindex(order)
    times = mat.columns.to_numpy(dtype=float)
    cmap = plt.matplotlib.colors.ListedColormap(["#bab0ac", "#4c78a8", "#f58518", "#54a24b"])
    fig, ax = plt.subplots(figsize=(11.5, max(4.8, 0.28 * len(order) + 1.4)))
    im = ax.imshow(mat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap=cmap, vmin=-0.5, vmax=3.5)
    for y, group in enumerate(groups["behavior_group"].tolist()):
        if y > 0 and group != groups["behavior_group"].iloc[y - 1]:
            ax.axhline(y - 0.5, color="black", linewidth=1.0)
    ax.axvline(np.argmin(np.abs(times - 0.0)), color="white", linewidth=0.8)
    tick_idx = np.linspace(0, len(times) - 1, 7, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{times[i]:.2f}" for i in tick_idx])
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels([str(s) for s in order])
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("Subject, sorted by behavioral GLC group")
    ax.set_title("Best MVPA Model Identity Over Time")
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(labels)), pad=0.02)
    cbar.ax.set_yticklabels(["baseline", "continuous", "day-1 discrete", "later discrete"])
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig_path


def save_diagnostics(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR, output_label="optimal_fixed", threshold=6.0, mvpa_tmin=0.3, mvpa_tmax=0.6):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    weights, mvpa = load_inputs(output_dir, output_label)
    glc_mat = glc_matrix(weights)
    groups = classify_subjects(glc_mat, threshold=threshold)
    mvpa_baseline_mat = mvpa_split_evidence(mvpa, tmin=mvpa_tmin, tmax=mvpa_tmax, metric="baseline")
    mvpa_relative_mat = mvpa_split_evidence(mvpa, tmin=mvpa_tmin, tmax=mvpa_tmax, metric="relative_best")
    prefix = _prefix(output_label)
    group_path = output_dir / f"{prefix}_diagnostic_behavior_groups.csv"
    groups.to_csv(group_path, index=False)
    paths = {
        "behavior_groups": group_path,
        "glc_heatmap": save_glc_heatmap(glc_mat, groups, figures_dir, output_label, threshold),
        "baseline_linked_heatmaps": save_linked_heatmap(
            glc_mat,
            mvpa_baseline_mat,
            groups,
            figures_dir,
            output_label,
            mvpa_tmin,
            mvpa_tmax,
            suffix="baseline",
            mvpa_label="MVPA evidence over baseline",
            relative=False,
        ),
        "relative_linked_heatmaps": save_linked_heatmap(
            glc_mat,
            mvpa_relative_mat,
            groups,
            figures_dir,
            output_label,
            mvpa_tmin,
            mvpa_tmax,
            suffix="relative",
            mvpa_label="MVPA evidence relative to best model",
            relative=True,
        ),
        "best_mvpa_identity": save_best_mvpa_identity_heatmap(mvpa, groups, figures_dir, output_label),
        "group_mvpa_timecourses": save_group_mvpa_timecourses(mvpa, groups, figures_dir, output_label, metric="baseline"),
        "relative_group_mvpa_timecourses": save_group_mvpa_timecourses(mvpa, groups, figures_dir, output_label, metric="relative_best"),
    }
    for path in paths.values():
        print(f"[decision bound diagnostic] wrote {path}", flush=True)
    return paths


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--output-label", default="optimal_fixed")
    parser.add_argument("--glc-threshold", type=float, default=6.0)
    parser.add_argument("--mvpa-tmin", type=float, default=0.3)
    parser.add_argument("--mvpa-tmax", type=float, default=0.6)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    save_diagnostics(
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        output_label=args.output_label,
        threshold=args.glc_threshold,
        mvpa_tmin=args.mvpa_tmin,
        mvpa_tmax=args.mvpa_tmax,
    )
