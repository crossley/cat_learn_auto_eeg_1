#!/usr/bin/env python3
"""Fit the group decision boundary and annotate trials with boundary distance."""

from __future__ import annotations

import os
import re
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
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression

from analysis_utils import model_term_summary

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output" / "boundary_distance"
FIGURES_DIR = PROJECT_DIR / "figures" / "boundary_distance"


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_regression(formula, df, group_col="subject", re_formula=None):
    if re_formula is None:
        model = smf.mixedlm(formula, data=df, groups=df[group_col]).fit(
            reml=False, method="lbfgs", disp=False
        )
    else:
        model = smf.mixedlm(
            formula, data=df, groups=df[group_col], re_formula=re_formula
        ).fit(reml=False, method="lbfgs", disp=False)
    return model, "mixedlm"


def load_behaviour_with_boundary(project_dir: Path | str = PROJECT_DIR):
    project_dir = Path(project_dir)
    beh_dir = project_dir / "Behavioural"
    beh_re = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")
    rows = []
    for path in sorted(beh_dir.glob("*.csv")):
        m = beh_re.match(path.name)
        if m is None:
            continue
        d = pd.read_csv(path)
        subject = int(m.group(1))
        day_code = int(m.group(2))
        day = day_code // 100
        d["subject"] = subject
        d["day_code"] = day_code
        d["day"] = day
        d["beh_file"] = path.name
        rows.append(d)
    if not rows:
        raise FileNotFoundError(f"No behavioural CSV files found in {beh_dir}")
    beh = pd.concat(rows, ignore_index=True)
    beh["cat_binary"] = (beh["cat"].astype(str) == "B").astype(int)
    clf = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    X = beh[["xt", "yt"]].to_numpy(dtype=float)
    y = beh["cat_binary"].to_numpy(dtype=int)
    clf.fit(X, y)
    w = clf.coef_[0].astype(float)
    b = float(clf.intercept_[0])
    norm = float(np.linalg.norm(w))
    decision_distance = (X @ w + b) / norm
    correct_side_distance = np.where(y == 1, decision_distance, -decision_distance)
    beh["boundary_distance"] = correct_side_distance
    beh["boundary_distance_abs"] = np.abs(correct_side_distance)
    beh["boundary_decision_distance"] = decision_distance
    beh["accuracy"] = (beh["fb"].astype(str).str.lower() == "correct").astype(float)
    beh["rt_sec"] = pd.to_numeric(beh["rt"], errors="coerce") / 1000.0
    boundary = {
        "coef_xt": float(w[0]),
        "coef_yt": float(w[1]),
        "intercept": b,
        "norm": norm,
        "classes": "A=0,B=1",
    }
    return beh, boundary


def add_distance_tertiles(beh):
    d = beh.copy()

    def bin_group(g):
        g = g.copy()
        ranks = g["boundary_distance_abs"].rank(method="first")
        try:
            g["distance_tertile"] = pd.qcut(ranks, 3, labels=["hard", "medium", "easy"])
        except ValueError:
            g["distance_tertile"] = pd.cut(
                ranks, 3, labels=["hard", "medium", "easy"], include_lowest=True
            )
        return g

    return pd.concat(
        [bin_group(g) for _, g in d.groupby(["subject", "day"], sort=False)],
        ignore_index=True,
    )


def plot_boundary_behaviour(agg_df, fig_path):
    tertile_order = ["hard", "medium", "easy"]
    colors = {"hard": "tab:red", "medium": "tab:orange", "easy": "tab:green"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), squeeze=False)
    for ax, metric, ylabel in [
        (axes[0, 0], "rt_sec", "RT (s)"),
        (axes[0, 1], "accuracy", "Accuracy"),
    ]:
        for tertile in tertile_order:
            g = agg_df[agg_df["distance_tertile"] == tertile]
            summary = (
                g.groupby("day", as_index=False)
                .agg(
                    mean=(metric, "mean"),
                    sem=(metric, _sem),
                    n_subjects=("subject", "nunique"),
                )
                .sort_values("day")
            )
            ax.errorbar(
                summary["day"],
                summary["mean"],
                yerr=summary["sem"],
                marker="o",
                linewidth=1.8,
                capsize=3,
                color=colors[tertile],
                label=tertile,
            )
        ax.set_xlabel("Day")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(title="Boundary distance")
    fig.suptitle("Behaviour by Boundary-Distance Tertile")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_boundary_behaviour_analysis(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    beh, boundary = load_behaviour_with_boundary()
    beh = add_distance_tertiles(beh)
    agg = (
        beh.groupby(["subject", "day", "distance_tertile"], observed=True, as_index=False)
        .agg(
            rt_sec=("rt_sec", "mean"),
            accuracy=("accuracy", "mean"),
            mean_boundary_distance=("boundary_distance_abs", "mean"),
            n_trials=("trial", "size"),
        )
        .sort_values(["subject", "day", "distance_tertile"])
    )
    model_rows = []
    for metric in ["rt_sec", "accuracy"]:
        d_model = agg.dropna(subset=[metric]).copy()
        d_model["distance_tertile"] = pd.Categorical(
            d_model["distance_tertile"],
            categories=["hard", "medium", "easy"],
            ordered=True,
        )
        model, status = _fit_regression(f"{metric} ~ distance_tertile * day", d_model)
        for term in model.model.exog_names:
            if term == "Intercept":
                continue
            item = model_term_summary(model, term)
            item.update({"metric": metric, "status": status})
            model_rows.append(item)
    boundary_csv = output_dir / "boundary_model_params.csv"
    trial_csv = output_dir / "behaviour_with_boundary_distance.csv"
    agg_csv = output_dir / "boundary_behaviour_tertile_subject_day.csv"
    model_csv = output_dir / "boundary_behaviour_model_terms.csv"
    fig_path = figures_dir / "boundary_behaviour_by_tertile.png"
    pd.DataFrame([boundary]).to_csv(boundary_csv, index=False)
    beh.to_csv(trial_csv, index=False)
    agg.to_csv(agg_csv, index=False)
    model_df = pd.DataFrame(model_rows)
    model_df.to_csv(model_csv, index=False)
    plot_boundary_behaviour(agg, fig_path)
    return {
        "beh": beh,
        "agg_df": agg,
        "model_df": model_df,
        "boundary": boundary,
        "boundary_csv": boundary_csv,
        "trial_csv": trial_csv,
        "agg_csv": agg_csv,
        "model_csv": model_csv,
        "figure": fig_path,
    }


if __name__ == "__main__":
    run_boundary_behaviour_analysis()
