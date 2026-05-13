#!/usr/bin/env python3
"""Broadband vs band-envelope/signed cross-day TG diagonal diagnostic."""

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

from mvpa_tg_window_structure import TG_CROSS_DAY_MATRIX_GLOB, TG_WINDOWS, parse_matrix_path

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def load_diagonal_timecourses_from_matrices(
    matrix_dir,
    signal_name,
    matrix_glob: str = "sub_*_trainD*_testD*.npz",
):
    rows = []
    matrix_dir = Path(matrix_dir)
    for path in sorted(matrix_dir.glob(matrix_glob)):
        parsed = parse_matrix_path(path)
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


def summarize_diagonal_by_signal_pair(diag_df):
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
    for signal, pair_type, subject in (
        subject_df[["signal", "pair_type", "subject"]].drop_duplicates().itertuples(index=False)
    ):
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
    signal_order = [s for s in signal_order if s in set(mean_df["signal"])]
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
            g = (
                mean_df[(mean_df["signal"] == signal) & (mean_df["pair_type"] == pair_type)]
                .sort_values("time_sec")
            )
            if g.empty:
                continue
            ax.plot(
                g["time_sec"], g["auc_mean"],
                color=colors.get(signal), linewidth=1.8, label=signal,
            )
            ax.fill_between(
                g["time_sec"],
                g["auc_mean"] - g["auc_sem"],
                g["auc_mean"] + g["auc_sem"],
                color=colors.get(signal),
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
    signal_order = [s for s in signal_order if s in set(window_mean_df["signal"])]
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

    rows = load_diagonal_timecourses_from_matrices(
        output_root,
        "broadband",
        matrix_glob=TG_CROSS_DAY_MATRIX_GLOB,
    )
    for band_name in ["theta", "alpha", "beta", "gamma"]:
        rows.extend(
            load_diagonal_timecourses_from_matrices(
                output_root,
                f"{band_name}_envelope",
                matrix_glob=f"band_tg_{band_name}_matrix_sub_*_trainD*_testD*.npz",
            )
        )
        rows.extend(
            load_diagonal_timecourses_from_matrices(
                output_root,
                f"{band_name}_signed",
                matrix_glob=f"band_signed_tg_{band_name}_matrix_sub_*_trainD*_testD*.npz",
            )
        )
    diag_df = pd.DataFrame(rows)
    if diag_df.empty:
        raise RuntimeError("No TG diagonal matrices found for broadband-vs-band diagnostic.")
    subject_df, mean_df, window_subject_df, window_mean_df = summarize_diagonal_by_signal_pair(diag_df)

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


if __name__ == "__main__":
    run_broadband_vs_band_diagnostic()
