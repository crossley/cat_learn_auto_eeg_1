#!/usr/bin/env python3
"""Plot sequence HMM/state-surrogate outputs."""

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

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"


def save_sequence_hmm_figures(
    feature_kind: str = "mvpa_decision",
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"sequence_hmm_{feature_kind}"
    model_csv = output_dir / f"{prefix}_model_selection.csv"
    time_csv = output_dir / f"{prefix}_state_timecourse.csv"
    transition_csv = output_dir / f"{prefix}_transitions.csv"
    for path in [model_csv, time_csv, transition_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Missing HMM output: {path}")

    model_df = pd.read_csv(model_csv)
    time_df = pd.read_csv(time_csv)
    transition_df = pd.read_csv(transition_csv)
    if model_df.empty or time_df.empty or transition_df.empty:
        raise ValueError(f"Empty sequence HMM output for feature_kind={feature_kind}")

    model_fig = figures_dir / f"{prefix}_model_selection.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), squeeze=False)
    ok = model_df[model_df["status"] == "ok"].copy()
    summary = (
        ok.groupby(["model_family", "n_states"], as_index=False)
        .agg(
            bic_mean=("bic", "mean"),
            bic_sem=("bic", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
            heldout_mean=("heldout_log_likelihood_per_obs", "mean"),
            heldout_sem=(
                "heldout_log_likelihood_per_obs",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan,
            ),
        )
        .sort_values("n_states")
    )
    for family, d_family in summary.groupby("model_family"):
        axes[0, 0].errorbar(
            d_family["n_states"],
            d_family["bic_mean"],
            yerr=d_family["bic_sem"],
            marker="o",
            linewidth=1.8,
            label=family,
        )
        axes[0, 1].errorbar(
            d_family["n_states"],
            d_family["heldout_mean"],
            yerr=d_family["heldout_sem"],
            marker="o",
            linewidth=1.8,
            label=family,
        )
    axes[0, 0].set_title("BIC by State Count")
    axes[0, 0].set_xlabel("States")
    axes[0, 0].set_ylabel("BIC")
    axes[0, 1].set_title("Held-out Log Likelihood")
    axes[0, 1].set_xlabel("States")
    axes[0, 1].set_ylabel("Log likelihood per observation")
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"Sequence State Model Selection: {feature_kind}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(model_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    occupancy_fig = figures_dir / f"{prefix}_state_occupancy_timecourse.png"
    selected = time_df.copy()
    selected_summary = (
        selected.groupby(["day", "time_sec", "state"], as_index=False)
        .agg(
            occupancy_mean=("occupancy", "mean"),
            occupancy_sem=("occupancy", lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan),
        )
        .sort_values(["day", "time_sec", "state"])
    )
    days = sorted(selected_summary["day"].dropna().unique())
    fig, axes = plt.subplots(1, len(days), figsize=(4.6 * len(days), 4.6), sharey=True, squeeze=False)
    for ax, day in zip(axes.ravel(), days):
        d_day = selected_summary[selected_summary["day"] == day]
        for state, d_state in d_day.groupby("state"):
            ax.plot(d_state["time_sec"], d_state["occupancy_mean"], linewidth=1.8, label=f"S{int(state)}")
            sem = d_state["occupancy_sem"].fillna(0.0).to_numpy(dtype=float)
            y = d_state["occupancy_mean"].to_numpy(dtype=float)
            x = d_state["time_sec"].to_numpy(dtype=float)
            ax.fill_between(x, y - sem, y + sem, alpha=0.15, linewidth=0)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"Day {int(day)}")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("State occupancy")
    axes.ravel()[-1].legend(title="State", fontsize=8)
    fig.suptitle(f"Selected State Occupancy: {feature_kind}")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(occupancy_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    transition_fig = figures_dir / f"{prefix}_transition_matrix.png"
    trans_summary = (
        transition_df.groupby(["from_state", "to_state"], as_index=False)["transition_probability"]
        .mean()
        .dropna()
    )
    n_states = int(max(trans_summary["from_state"].max(), trans_summary["to_state"].max()) + 1)
    mat = np.full((n_states, n_states), np.nan)
    for _, row in trans_summary.iterrows():
        mat[int(row["from_state"]), int(row["to_state"])] = float(row["transition_probability"])
    fig, ax = plt.subplots(figsize=(5.2, 4.8))
    im = ax.imshow(np.ma.masked_invalid(mat), vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_xticks(np.arange(n_states))
    ax.set_yticks(np.arange(n_states))
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(f"Mean Transition Matrix: {feature_kind}")
    fig.colorbar(im, ax=ax, label="Transition probability")
    fig.tight_layout()
    fig.savefig(transition_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_paths": {
            "model_selection": model_fig,
            "state_occupancy_timecourse": occupancy_fig,
            "transition_matrix": transition_fig,
        }
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-kind", default=os.environ.get("SEQUENCE_HMM_FEATURE_KIND", "mvpa_decision"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    save_sequence_hmm_figures(
        feature_kind=args.feature_kind,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
    )
