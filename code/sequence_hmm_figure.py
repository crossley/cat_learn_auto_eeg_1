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


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def choose_state_count(model_df: pd.DataFrame, requested: int | None) -> int:
    if requested is not None:
        return int(requested)
    ok = model_df[(model_df["status"] == "ok") & (model_df["selected_by_bic"].astype(bool))]
    if ok.empty:
        ok = model_df[model_df["status"] == "ok"]
        if ok.empty:
            raise ValueError("No successful HMM model rows available")
        return int(ok.groupby("n_states")["bic"].mean().idxmin())
    counts = ok["n_states"].astype(int).value_counts()
    return int(counts.index[0])


def add_state_count_to_name(path: Path, n_states: int) -> Path:
    return path.with_name(path.stem + f"_{n_states}states" + path.suffix)


def save_sequence_hmm_figures(
    feature_kind: str = "mvpa_decision",
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
    n_states: int | None = None,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"sequence_hmm_{feature_kind}"
    model_csv = output_dir / f"{prefix}_model_selection.csv"
    time_csv = output_dir / f"{prefix}_state_timecourse.csv"
    transition_csv = output_dir / f"{prefix}_transitions.csv"
    dwell_csv = output_dir / f"{prefix}_dwell_times.csv"
    for path in [model_csv, time_csv, transition_csv, dwell_csv]:
        if not path.exists():
            raise FileNotFoundError(f"Missing HMM output: {path}")

    model_df = pd.read_csv(model_csv)
    time_df = pd.read_csv(time_csv)
    transition_df = pd.read_csv(transition_csv)
    dwell_df = pd.read_csv(dwell_csv)
    if model_df.empty or time_df.empty or transition_df.empty:
        raise ValueError(f"Empty sequence HMM output for feature_kind={feature_kind}")
    plot_states = choose_state_count(model_df, n_states)
    if plot_states not in set(time_df["n_states"].astype(int)):
        available = sorted(time_df["n_states"].astype(int).unique().tolist())
        raise ValueError(
            f"No saved state diagnostics for n_states={plot_states}. "
            f"Available: {available}. Rerun analysis with --diagnostic-states {plot_states}."
        )

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

    occupancy_fig = add_state_count_to_name(
        figures_dir / f"{prefix}_state_occupancy_timecourse.png", plot_states
    )
    selected = time_df[time_df["n_states"].astype(int) == plot_states].copy()
    selected_summary = (
        selected.groupby(["day", "time_sec", "state"], as_index=False)
        .agg(
            occupancy_mean=("occupancy", "mean"),
            occupancy_sem=("occupancy", sem),
            posterior_mean=("posterior_mean", "mean"),
            posterior_sem=("posterior_mean", sem),
        )
        .sort_values(["day", "time_sec", "state"])
    )
    days = sorted(selected_summary["day"].dropna().unique())
    fig, axes = plt.subplots(1, len(days), figsize=(4.6 * len(days), 4.6), sharey=True, squeeze=False)
    for ax, day in zip(axes.ravel(), days):
        d_day = selected_summary[selected_summary["day"] == day]
        for state, d_state in d_day.groupby("state"):
            ax.plot(d_state["time_sec"], d_state["occupancy_mean"], linewidth=1.8, label=f"S{int(state)}")
            err = d_state["occupancy_sem"].fillna(0.0).to_numpy(dtype=float)
            y = d_state["occupancy_mean"].to_numpy(dtype=float)
            x = d_state["time_sec"].to_numpy(dtype=float)
            ax.fill_between(x, y - err, y + err, alpha=0.15, linewidth=0)
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"Day {int(day)}")
        ax.set_xlabel("Time (s)")
        ax.grid(alpha=0.25)
    axes.ravel()[0].set_ylabel("State occupancy")
    axes.ravel()[-1].legend(title="State", fontsize=8)
    chance = 1.0 / float(plot_states)
    for ax in axes.ravel():
        ax.axhline(chance, color="0.25", linestyle="--", linewidth=0.8, alpha=0.65)
    fig.suptitle(f"State Occupancy: {feature_kind}, {plot_states} states")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(occupancy_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    transition_fig = add_state_count_to_name(
        figures_dir / f"{prefix}_transition_matrix.png", plot_states
    )
    transition_use = transition_df[transition_df["n_states"].astype(int) == plot_states].copy()
    trans_summary = (
        transition_use.groupby(["from_state", "to_state"], as_index=False)["transition_probability"]
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
    ax.set_title(f"Mean Transition Matrix: {feature_kind}, {plot_states} states")
    fig.colorbar(im, ax=ax, label="Transition probability")
    fig.tight_layout()
    fig.savefig(transition_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    dominance_fig = add_state_count_to_name(
        figures_dir / f"{prefix}_dominance_entropy_timecourse.png", plot_states
    )
    pivot = (
        selected.groupby(["subject", "day", "time_sec", "state"], as_index=False)["posterior_mean"]
        .mean()
        .pivot_table(index=["subject", "day", "time_sec"], columns="state", values="posterior_mean")
        .reset_index()
    )
    state_cols = [c for c in pivot.columns if isinstance(c, (int, np.integer)) or str(c).isdigit()]
    probs = pivot[state_cols].to_numpy(dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    row_sums = np.nansum(probs, axis=1, keepdims=True)
    probs = np.divide(probs, row_sums, out=np.full_like(probs, np.nan), where=row_sums > 0)
    pivot["max_posterior"] = np.nanmax(probs, axis=1)
    entropy = -np.nansum(probs * np.log(probs + np.finfo(float).eps), axis=1)
    pivot["normalized_entropy"] = entropy / np.log(float(plot_states))
    diag_summary = (
        pivot.groupby(["day", "time_sec"], as_index=False)
        .agg(
            max_posterior_mean=("max_posterior", "mean"),
            max_posterior_sem=("max_posterior", sem),
            entropy_mean=("normalized_entropy", "mean"),
            entropy_sem=("normalized_entropy", sem),
        )
        .sort_values(["day", "time_sec"])
    )
    days = sorted(diag_summary["day"].dropna().unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True)
    for day, d_day in diag_summary.groupby("day"):
        x = d_day["time_sec"].to_numpy(dtype=float)
        for ax, mean_col, sem_col, label in [
            (axes[0], "max_posterior_mean", "max_posterior_sem", "max state posterior"),
            (axes[1], "entropy_mean", "entropy_sem", "state entropy"),
        ]:
            y = d_day[mean_col].to_numpy(dtype=float)
            s = d_day[sem_col].fillna(0.0).to_numpy(dtype=float)
            ax.plot(x, y, linewidth=1.8, label=f"D{int(day)}")
            ax.fill_between(x, y - s, y + s, alpha=0.12, linewidth=0)
            ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
            ax.set_title(label)
            ax.set_xlabel("Time (s)")
            ax.grid(alpha=0.25)
    axes[0].axhline(1.0 / float(plot_states), color="0.2", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("Probability")
    axes[1].set_ylabel("Normalized entropy")
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(title="Day", fontsize=8)
    fig.suptitle(f"State Dominance Diagnostics: {feature_kind}, {plot_states} states")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(dominance_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    dwell_fig = add_state_count_to_name(
        figures_dir / f"{prefix}_dwell_time_by_state_day.png", plot_states
    )
    dwell_use = dwell_df[dwell_df["n_states"].astype(int) == plot_states].copy()
    dwell_summary = (
        dwell_use.groupby(["day", "state"], as_index=False)
        .agg(dwell_sec_mean=("dwell_sec", "mean"), dwell_sec_sem=("dwell_sec", sem))
        .sort_values(["day", "state"])
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    width = 0.8 / max(plot_states, 1)
    day_vals = sorted(dwell_summary["day"].dropna().unique())
    for state, d_state in dwell_summary.groupby("state"):
        x = np.arange(len(day_vals)) + (int(state) - (plot_states - 1) / 2.0) * width
        d_state = d_state.set_index("day").reindex(day_vals)
        ax.bar(
            x,
            d_state["dwell_sec_mean"].to_numpy(dtype=float),
            yerr=d_state["dwell_sec_sem"].fillna(0.0).to_numpy(dtype=float),
            width=width,
            label=f"S{int(state)}",
            alpha=0.85,
        )
    ax.set_xticks(np.arange(len(day_vals)))
    ax.set_xticklabels([f"D{int(d)}" for d in day_vals])
    ax.set_ylabel("Mean dwell duration (s)")
    ax.set_title(f"Dwell Time by Day: {feature_kind}, {plot_states} states")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="State", fontsize=8, ncol=min(plot_states, 6))
    fig.tight_layout()
    fig.savefig(dwell_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    persistence_fig = add_state_count_to_name(
        figures_dir / f"{prefix}_self_transition_by_state_day.png", plot_states
    )
    diag = transition_use[transition_use["from_state"] == transition_use["to_state"]].copy()
    pers_summary = (
        diag.groupby(["day", "from_state"], as_index=False)
        .agg(persistence_mean=("transition_probability", "mean"), persistence_sem=("transition_probability", sem))
        .sort_values(["day", "from_state"])
    )
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    for state, d_state in pers_summary.groupby("from_state"):
        d_state = d_state.sort_values("day")
        ax.errorbar(
            d_state["day"],
            d_state["persistence_mean"],
            yerr=d_state["persistence_sem"],
            marker="o",
            linewidth=1.7,
            label=f"S{int(state)}",
        )
    ax.set_xticks(day_vals)
    ax.set_xticklabels([f"D{int(d)}" for d in day_vals])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("Self-transition probability")
    ax.set_title(f"State Persistence by Day: {feature_kind}, {plot_states} states")
    ax.grid(alpha=0.25)
    ax.legend(title="State", fontsize=8, ncol=min(plot_states, 6))
    fig.tight_layout()
    fig.savefig(persistence_fig, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "figure_paths": {
            "model_selection": model_fig,
            "state_occupancy_timecourse": occupancy_fig,
            "transition_matrix": transition_fig,
            "dominance_entropy_timecourse": dominance_fig,
            "dwell_time_by_state_day": dwell_fig,
            "self_transition_by_state_day": persistence_fig,
        }
    }


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-kind", default=os.environ.get("SEQUENCE_HMM_FEATURE_KIND", "mvpa_decision"))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--figures-dir", default=str(FIGURES_DIR))
    parser.add_argument("--n-states", type=int, default=None)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    save_sequence_hmm_figures(
        feature_kind=args.feature_kind,
        output_dir=args.output_dir,
        figures_dir=args.figures_dir,
        n_states=args.n_states,
    )
