#!/usr/bin/env python3
"""Generate RSA model-prediction figures for stimulus-bin geometry."""

from __future__ import annotations

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

from boundary_distance import load_behaviour_with_boundary

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"

GRID_CANDIDATES = [10, 8, 6, 5, 4, 3]
MIN_TRIALS_PER_BIN_SESSION = 8
MIN_SESSION_PASS_PROP = 0.80
MIN_RETAINED_BIN_PROP = 0.50
MIN_RETAINED_BINS = 12


def _normalize_rdm(mat):
    mat = np.asarray(mat, dtype=float)
    finite = np.isfinite(mat)
    if not np.any(finite):
        return mat
    lo = float(np.nanmin(mat[finite]))
    hi = float(np.nanmax(mat[finite]))
    if np.isclose(lo, hi):
        return np.zeros_like(mat, dtype=float)
    return (mat - lo) / (hi - lo)


def assign_grid_bins(beh, n_bins):
    d = beh.copy()
    x_edges = np.linspace(d["xt"].min(), d["xt"].max(), n_bins + 1)
    y_edges = np.linspace(d["yt"].min(), d["yt"].max(), n_bins + 1)
    d["sf_bin"] = pd.cut(
        d["xt"], bins=x_edges, labels=False, include_lowest=True
    ).astype(int)
    d["ori_bin"] = pd.cut(
        d["yt"], bins=y_edges, labels=False, include_lowest=True
    ).astype(int)
    d["stim_bin"] = d["sf_bin"].astype(str) + "_" + d["ori_bin"].astype(str)
    return d, x_edges, y_edges


def evaluate_grid_candidate(beh, n_bins):
    d, _, _ = assign_grid_bins(beh, n_bins)
    subject_days = d[["subject", "day"]].drop_duplicates().assign(k=1)
    bins = (
        pd.MultiIndex.from_product(
            [range(n_bins), range(n_bins)], names=["sf_bin", "ori_bin"]
        )
        .to_frame(index=False)
        .assign(k=1)
    )
    counts = (
        d.groupby(["subject", "day", "sf_bin", "ori_bin"])
        .size()
        .rename("n_trials")
        .reset_index()
    )
    full = (
        subject_days.merge(bins, on="k")
        .drop(columns="k")
        .merge(counts, how="left")
        .fillna({"n_trials": 0})
    )
    by_bin = (
        full.groupby(["sf_bin", "ori_bin"], as_index=False)
        .agg(
            median_trials=("n_trials", "median"),
            min_trials=("n_trials", "min"),
            pass_session_prop=(
                "n_trials",
                lambda x: float((x >= MIN_TRIALS_PER_BIN_SESSION).mean()),
            ),
            total_trials=("n_trials", "sum"),
        )
    )
    retained = by_bin[
        (by_bin["total_trials"] > 0)
        & (by_bin["median_trials"] >= MIN_TRIALS_PER_BIN_SESSION)
        & (by_bin["pass_session_prop"] >= MIN_SESSION_PASS_PROP)
    ].copy()
    return {
        "n_bins": int(n_bins),
        "n_cells": int(n_bins * n_bins),
        "n_nonempty_bins": int((by_bin["total_trials"] > 0).sum()),
        "n_retained_bins": int(len(retained)),
        "retained_bin_prop": float(len(retained) / (n_bins * n_bins)),
        "median_trials_nonempty": float(
            by_bin.loc[by_bin["total_trials"] > 0, "median_trials"].median()
        ),
        "min_pass_session_prop_retained": float(
            retained["pass_session_prop"].min() if len(retained) else np.nan
        ),
    }


def choose_grid(beh):
    diagnostics = pd.DataFrame(
        [evaluate_grid_candidate(beh, n_bins) for n_bins in GRID_CANDIDATES]
    )
    passing = diagnostics[
        (diagnostics["n_retained_bins"] >= MIN_RETAINED_BINS)
        & (diagnostics["retained_bin_prop"] >= MIN_RETAINED_BIN_PROP)
    ]
    if passing.empty:
        selected = diagnostics.sort_values(
            ["n_retained_bins", "n_bins"], ascending=[False, False]
        ).iloc[0]
    else:
        selected = passing.sort_values("n_bins", ascending=False).iloc[0]
    return int(selected["n_bins"]), diagnostics


def make_bin_table(beh, boundary, n_bins):
    d, x_edges, y_edges = assign_grid_bins(beh, n_bins)
    counts = (
        d.groupby(["subject", "day", "sf_bin", "ori_bin"])
        .size()
        .rename("n_trials")
        .reset_index()
    )
    by_bin_count = (
        counts.groupby(["sf_bin", "ori_bin"], as_index=False)
        .agg(
            median_trials=("n_trials", "median"),
            pass_session_prop=(
                "n_trials",
                lambda x: float((x >= MIN_TRIALS_PER_BIN_SESSION).mean()),
            ),
        )
    )
    summary = (
        d.groupby(["sf_bin", "ori_bin"], as_index=False)
        .agg(
            xt_mean=("xt", "mean"),
            yt_mean=("yt", "mean"),
            n_trials=("trial", "size"),
            prop_cat_b=("cat_binary", "mean"),
            prop_resp_b=("resp", lambda x: float((x.astype(str) == "B").mean())),
            boundary_signed=("boundary_decision_distance", "mean"),
            boundary_abs=("boundary_distance_abs", "mean"),
        )
        .merge(by_bin_count, on=["sf_bin", "ori_bin"], how="left")
    )
    summary["retained"] = (
        (summary["median_trials"] >= MIN_TRIALS_PER_BIN_SESSION)
        & (summary["pass_session_prop"] >= MIN_SESSION_PASS_PROP)
    )
    summary["category_label"] = np.where(summary["prop_cat_b"] >= 0.5, "B", "A")
    summary["response_label"] = np.where(summary["prop_resp_b"] >= 0.5, "B", "A")
    summary["x_center"] = [
        float((x_edges[int(i)] + x_edges[int(i) + 1]) / 2.0)
        for i in summary["sf_bin"]
    ]
    summary["y_center"] = [
        float((y_edges[int(i)] + y_edges[int(i) + 1]) / 2.0)
        for i in summary["ori_bin"]
    ]
    w = np.array([boundary["coef_xt"], boundary["coef_yt"]], dtype=float)
    b = float(boundary["intercept"])
    norm = float(boundary["norm"])
    centers = summary[["x_center", "y_center"]].to_numpy(dtype=float)
    summary["boundary_signed_center"] = (centers @ w + b) / norm
    summary["boundary_abs_center"] = np.abs(summary["boundary_signed_center"])
    retained = summary[summary["retained"]].copy()
    retained = retained.sort_values(
        ["category_label", "boundary_signed_center", "sf_bin", "ori_bin"]
    ).reset_index(drop=True)
    retained["rdm_order"] = np.arange(len(retained))
    return summary, retained


def make_model_rdms(bin_df):
    xy = bin_df[["x_center", "y_center"]].to_numpy(dtype=float)
    signed = bin_df["boundary_signed_center"].to_numpy(dtype=float)
    abs_boundary = bin_df["boundary_abs_center"].to_numpy(dtype=float)
    category = bin_df["category_label"].to_numpy(dtype=str)
    response = bin_df["response_label"].to_numpy(dtype=str)

    physical = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    category_same = (category[:, None] != category[None, :]).astype(float)
    boundary_difficulty = np.abs(abs_boundary[:, None] - abs_boundary[None, :])
    signed_boundary = np.abs(signed[:, None] - signed[None, :])
    response_mapping = (response[:, None] != response[None, :]).astype(float)

    return {
        "Physical distance": _normalize_rdm(physical),
        "Category membership": category_same,
        "Boundary difficulty": _normalize_rdm(boundary_difficulty),
        "Signed boundary position": _normalize_rdm(signed_boundary),
        "Response mapping": response_mapping,
    }


def plot_grid_diagnostics(diagnostics, selected_n, fig_path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), squeeze=False)
    axes = axes.ravel()
    x = diagnostics["n_bins"].to_numpy()
    for ax, col, ylabel in [
        (axes[0], "n_retained_bins", "Retained bins"),
        (axes[1], "retained_bin_prop", "Retained bin proportion"),
        (axes[2], "median_trials_nonempty", "Median trials / subject-day bin"),
    ]:
        ax.plot(x, diagnostics[col], marker="o", color="black")
        ax.axvline(selected_n, color="tab:red", linestyle="--", linewidth=1)
        ax.set_xlabel("Grid dimension")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
    axes[1].axhline(MIN_RETAINED_BIN_PROP, color="0.5", linestyle=":", linewidth=1)
    axes[2].axhline(MIN_TRIALS_PER_BIN_SESSION, color="0.5", linestyle=":", linewidth=1)
    fig.suptitle(f"RSA stimulus-bin grid selection: selected {selected_n}x{selected_n}")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_predictions(all_bins, retained_bins, rdms, fig_path):
    fig = plt.figure(figsize=(18, 9.4))
    gs = fig.add_gridspec(
        2,
        4,
        left=0.06,
        right=0.90,
        top=0.88,
        bottom=0.09,
        wspace=0.38,
        hspace=0.38,
    )
    ax_space = fig.add_subplot(gs[:, 0])
    colors = {"A": "tab:blue", "B": "tab:orange"}
    for label, group in all_bins.groupby("category_label"):
        ax_space.scatter(
            group["x_center"],
            group["y_center"],
            s=120 * np.sqrt(group["n_trials"] / group["n_trials"].max()),
            alpha=0.20,
            color=colors[label],
            edgecolor="none",
        )
    for label, group in retained_bins.groupby("category_label"):
        ax_space.scatter(
            group["x_center"],
            group["y_center"],
            s=42,
            alpha=0.95,
            color=colors[label],
            label=f"Category {label}",
        )
    for _, row in retained_bins.iterrows():
        ax_space.text(
            row["x_center"],
            row["y_center"],
            str(int(row["rdm_order"])),
            fontsize=6,
            ha="center",
            va="center",
            color="white",
        )
    ax_space.set_xlabel("x")
    ax_space.set_ylabel("y")
    ax_space.set_title("Retained stimulus bins")
    ax_space.legend(frameon=False, loc="best")

    axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[0, 3]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    for ax, (name, mat) in zip(axes, rdms.items()):
        im = ax.imshow(mat, origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel("Stimulus bin")
        ax.set_ylabel("Stimulus bin")
    ax_empty = fig.add_subplot(gs[1, 3])
    ax_empty.axis("off")
    cax = fig.add_axes([0.94, 0.18, 0.012, 0.62])
    fig.colorbar(im, cax=cax, label="Model dissimilarity")
    fig.suptitle("RSA model RDM predictions")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_rsa_model_predictions(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    beh, boundary = load_behaviour_with_boundary()
    selected_n, diagnostics = choose_grid(beh)
    all_bins, retained_bins = make_bin_table(beh, boundary, selected_n)
    rdms = make_model_rdms(retained_bins)

    diagnostics_csv = output_dir / "rsa_model_grid_diagnostics.csv"
    bins_csv = output_dir / "rsa_model_stimulus_bins.csv"
    rdm_csv = output_dir / "rsa_model_rdms.csv"
    diagnostics.to_csv(diagnostics_csv, index=False)
    all_bins.to_csv(bins_csv, index=False)

    rows = []
    for model_name, mat in rdms.items():
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                rows.append(
                    {
                        "model": model_name,
                        "bin_i": int(i),
                        "bin_j": int(j),
                        "dissimilarity": float(mat[i, j]),
                    }
                )
    pd.DataFrame(rows).to_csv(rdm_csv, index=False)

    diagnostics_fig = figures_dir / "rsa_model_grid_diagnostics.png"
    prediction_fig = figures_dir / "rsa_model_prediction_rdms.png"
    plot_grid_diagnostics(diagnostics, selected_n, diagnostics_fig)
    plot_model_predictions(all_bins, retained_bins, rdms, prediction_fig)

    print(
        f"[RSA models] Selected {selected_n}x{selected_n} grid with "
        f"{len(retained_bins)} retained bins.",
        flush=True,
    )
    print(f"[RSA models] Wrote {prediction_fig}", flush=True)

    return {
        "selected_grid": selected_n,
        "diagnostics": diagnostics,
        "stimulus_bins": all_bins,
        "retained_bins": retained_bins,
        "rdms": rdms,
        "diagnostics_csv": diagnostics_csv,
        "bins_csv": bins_csv,
        "rdm_csv": rdm_csv,
        "diagnostics_figure": diagnostics_fig,
        "prediction_figure": prediction_fig,
    }


if __name__ == "__main__":
    run_rsa_model_predictions()
