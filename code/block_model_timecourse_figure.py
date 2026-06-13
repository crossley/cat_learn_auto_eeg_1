#!/usr/bin/env python3
"""Plot 25-block model-evidence timecourses."""

from pathlib import Path
import os
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from connect_sensorwide_analysis import OUTPUT_DIR, FIGURES_DIR
from presentation_figure import setup_axis

MODALITIES = {
    "erp": {
        "summary": "erp_block_model_timecourse_summary.csv",
        "figure": "erp_block_model_timecourse.png",
        "title": "ERP Block Model Evidence Over Time",
    },
    "connectivity": {
        "summary": "connect_block_model_timecourse_summary.csv",
        "figure": "connect_block_model_timecourse.png",
        "title": "Connectivity Block Model Evidence Over Time",
    },
    "mvpa": {
        "summary": "mvpa_block_model_timecourse_summary.csv",
        "figure": "mvpa_block_model_timecourse.png",
        "title": "MVPA Block Model Evidence Over Time",
    },
}


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing block model output: {path}")
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty block model output: {path}")
    return d


def line_label(row):
    label = str(row["model_label"])
    if label.startswith("Discrete Restructuring B"):
        return label.replace("Discrete Restructuring B", "Transition B")
    return label


def split_block_day(split_block):
    split_block = int(split_block)
    return ((split_block - 1) // 5) + 1


def split_block_color(split_block):
    split_block = int(split_block)
    day = split_block_day(split_block)
    day_cmaps = {
        1: plt.cm.Blues,
        2: plt.cm.Greens,
        3: plt.cm.Oranges,
        4: plt.cm.Purples,
        5: plt.cm.Reds,
    }
    day_start = (day - 1) * 5 + 1
    day_end = min(day * 5, 24)
    n_in_day = max(1, day_end - day_start + 1)
    pos = (split_block - day_start) / max(1, n_in_day - 1)
    return day_cmaps[day](0.42 + 0.48 * pos)


def save_block_model_figure(modality, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    spec = MODALITIES[modality]
    d = require_csv(output_dir / spec["summary"])
    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    continuous = d[d["model_label"] == "Continuous Restructuring"].sort_values("time_sec")
    if not continuous.empty:
        x = continuous["time_sec"].to_numpy(float)
        y = -continuous["delta_bic_baseline_mean"].to_numpy(float)
        err = continuous["delta_bic_baseline_sem"].to_numpy(float)
        ax.plot(x, y, color="#1f1f1f", linewidth=2.1, label="Continuous Restructuring")
        good = np.isfinite(err)
        if np.any(good):
            ax.fill_between(x[good], y[good] - err[good], y[good] + err[good], color="#1f1f1f", alpha=0.10, linewidth=0)

    split_rows = d[d["model"] == "discrete"].copy()
    split_rows = split_rows[np.isfinite(split_rows["split_block"].astype(float))]
    if not split_rows.empty:
        split_blocks = sorted(split_rows["split_block"].astype(float).dropna().unique())
        for split_block in split_blocks:
            g = split_rows[np.isclose(split_rows["split_block"].astype(float), float(split_block))].sort_values("time_sec")
            x = g["time_sec"].to_numpy(float)
            y = -g["delta_bic_baseline_mean"].to_numpy(float)
            ax.plot(
                x,
                y,
                color=split_block_color(split_block),
                linewidth=1.1,
                alpha=0.82,
                label=f"B{int(split_block)}",
            )

    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("Evidence above baseline model")
    ax.set_title(spec["title"])
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title="Continuous and transition after block",
        frameon=False,
        fontsize=7,
        title_fontsize=8,
        ncol=5,
        loc="center left",
        bbox_to_anchor=(1.01, 0.5),
        borderaxespad=0.0,
    )
    setup_axis(ax)
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])
    path = figures_dir / spec["figure"]
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[block model figure] wrote {path}", flush=True)
    return path


def save_all_block_model_figures(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    return {
        modality: save_block_model_figure(modality, output_dir, figures_dir)
        for modality in MODALITIES
    }


if __name__ == "__main__":
    requested = sys.argv[1:]
    if requested:
        for modality in requested:
            if modality not in MODALITIES:
                raise ValueError(
                    f"Unknown modality: {modality}. "
                    f"Choose from {sorted(MODALITIES)}"
                )
            save_block_model_figure(modality)
    else:
        save_all_block_model_figures()
