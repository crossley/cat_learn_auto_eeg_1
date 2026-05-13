#!/usr/bin/env python3
"""ERP peak latency trajectories across days (N2, LRP stim- and response-locked)."""

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
import statsmodels.formula.api as smf

from analysis_utils import model_term_summary

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def _fit_ols(formula, df, cluster_subject=True):
    model = smf.ols(formula, data=df).fit()
    if cluster_subject and "subject" in df.columns and df["subject"].nunique() > 1:
        return model.get_robustcov_results(cov_type="cluster", groups=df["subject"])
    return model


def _flat_peak(values, eps=1e-12):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) < 3:
        return True
    return float(np.nanmax(finite) - np.nanmin(finite)) <= eps


def erp_component_for_group(group, component, level):
    if component == "frontal_n2":
        channels = ["Fz", "FCz", "FC1", "FC2"]
        g = group[group["channel"].isin(channels)]
        if g["channel"].nunique() < len(channels):
            return None
        wave = (
            g.groupby("time_s", as_index=False)["amplitude_v"]
            .mean()
            .sort_values("time_s")
        )
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
        wave = (
            (left_wave - right_wave)
            .rename("lateralisation_v")
            .reset_index()
            .sort_values("time_s")
        )
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
    erp_subject_csv: Path | str = PROJECT_DIR / "output" / "erp_grand_average_subject_day_all.csv",
    erp_grand_csv: Path | str = (
        PROJECT_DIR / "output" / "erp_grand_average_by_day_lock_condition.csv"
    ),
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
                item = erp_component_for_group(group, component, "subject")
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
                item = erp_component_for_group(group, component, "grand")
                if item is not None:
                    item.update({"subject": np.nan, "day": int(day)})
                    rows.append(item)
    return pd.DataFrame(rows)


def fit_erp_latency_slopes(latency_df):
    rows = []
    models = {}
    d = latency_df[
        (latency_df["level"] == "subject") & latency_df["latency_s"].notna()
    ].copy()
    for component, g in d.groupby("component"):
        if g["day"].nunique() < 2 or g["subject"].nunique() < 2:
            continue
        model = _fit_ols("latency_s ~ day", g)
        item = model_term_summary(model, "day")
        item.update(
            {"component": component, "n_rows": int(len(g)), "n_subjects": int(g["subject"].nunique())}
        )
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
        item = model_term_summary(model, term)
        item.update({"comparison": f"{b}_minus_{a}_slope"})
        comparisons.append(item)
    return pd.DataFrame(rows), pd.DataFrame(comparisons), models


def plot_erp_latency_trajectories(latency_df, slope_df, fig_path):
    d_sub = latency_df[
        (latency_df["level"] == "subject") & latency_df["latency_s"].notna()
    ].copy()
    d_grand = latency_df[
        (latency_df["level"] == "grand") & latency_df["latency_s"].notna()
    ].copy()
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
                ax.plot(
                    gs["day"],
                    gs["latency_s"] * 1000.0,
                    color=colors[component],
                    alpha=0.12,
                    linewidth=0.8,
                )
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
    latency_csv = output_dir / "erp_latency_trajectories_peak_latencies.csv"
    slope_csv = output_dir / "erp_latency_trajectories_day_slopes.csv"
    comparison_csv = output_dir / "erp_latency_trajectories_slope_comparisons.csv"
    fig_path = figures_dir / "erp_latency_trajectories_peak_latency.png"
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


if __name__ == "__main__":
    run_erp_peak_latency_trajectories()
