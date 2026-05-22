#!/usr/bin/env python3
"""Plot cross-day TG window-structure figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from mvpa_stim_locked_cat_tg_window_structure_analysis import FIGURES_DIR, OUTPUT_DIR, _sem


def plot_tg_window_gradients(window_df, slope_df, fig_path):
    if window_df.empty:
        raise ValueError("Empty TG window-structure AUC table")
    if slope_df.empty:
        raise ValueError("Empty TG window-structure slope table")
    d = window_df.dropna(subset=["mean_auc"]).copy()
    d = d[d["day_distance"] > 0]
    if d.empty:
        raise ValueError("No off-diagonal TG window-structure rows to plot")
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
        if s.empty:
            raise ValueError(f"Missing TG window slope row: window={window_name}")
        slope = float(s["estimate"].iloc[0])
        lo = float(s["ci_low"].iloc[0])
        hi = float(s["ci_high"].iloc[0])
        ax.set_title(f"{window_name.title()} ({slope:.4f} [{lo:.4f}, {hi:.4f}])")
        ax.axhline(0.5, color="0.35", linestyle=":", linewidth=1)
        ax.set_xlabel("Day distance")
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("Mean off-diagonal AUC")
    fig.suptitle("Cross-Day TG Window Gradients")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fig_mvpa_stim_locked_cat_tg_window_structure(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    window_csv = output_dir / "mvpa_stim_locked_cat_tg_window_auc_subject_pairs.csv"
    slope_csv = output_dir / "mvpa_stim_locked_cat_tg_window_gradient_slopes.csv"
    if not window_csv.exists() or not slope_csv.exists():
        raise FileNotFoundError(
            "Missing TG window-structure outputs. "
            "Run mvpa_stim_locked_cat_tg_window_structure_analysis.py first."
        )
    fig_path = figures_dir / "mvpa_stim_locked_cat_tg_window_gradients.png"
    plot_tg_window_gradients(pd.read_csv(window_csv), pd.read_csv(slope_csv), fig_path)
    return {"window_gradient_figure": fig_path}


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_tg_window_structure()
