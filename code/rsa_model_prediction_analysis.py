#!/usr/bin/env python3
"""Generate RSA model-prediction figures for stimulus-bin geometry."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from util_boundary_distance import load_behaviour_with_boundary

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
    x_edges = np.linspace(d["x"].min(), d["x"].max(), n_bins + 1)
    y_edges = np.linspace(d["y"].min(), d["y"].max(), n_bins + 1)
    d["sf_bin"] = pd.cut(
        d["x"], bins=x_edges, labels=False, include_lowest=True
    ).astype(int)
    d["ori_bin"] = pd.cut(
        d["y"], bins=y_edges, labels=False, include_lowest=True
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
            x_mean=("x", "mean"),
            y_mean=("y", "mean"),
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
    centers_transformed = summary[["xt_mean", "yt_mean"]].to_numpy(dtype=float)
    summary["boundary_signed_center"] = (centers_transformed @ w + b) / norm
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

    physical = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    category_same = (category[:, None] != category[None, :]).astype(float)
    boundary_difficulty = np.abs(abs_boundary[:, None] - abs_boundary[None, :])
    signed_boundary = np.abs(signed[:, None] - signed[None, :])

    return {
        "Physical distance": _normalize_rdm(physical),
        "Category / response": category_same,
        "Boundary difficulty": _normalize_rdm(boundary_difficulty),
        "Signed boundary position": _normalize_rdm(signed_boundary),
    }


def run_rsa_model_predictions(
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(
        f"[RSA models] Selected {selected_n}x{selected_n} grid with "
        f"{len(retained_bins)} retained bins.",
        flush=True,
    )
    print(f"[RSA models] Wrote {rdm_csv}", flush=True)

    return {
        "selected_grid": selected_n,
        "diagnostics": diagnostics,
        "stimulus_bins": all_bins,
        "retained_bins": retained_bins,
        "rdms": rdms,
        "diagnostics_csv": diagnostics_csv,
        "bins_csv": bins_csv,
        "rdm_csv": rdm_csv,
    }


if __name__ == "__main__":
    run_rsa_model_predictions()
