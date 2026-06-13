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


def save_block_model_figure(modality, output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    spec = MODALITIES[modality]
    d = require_csv(output_dir / spec["summary"])
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
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
        best = (
            split_rows.groupby("split_block", as_index=False)
            .agg(best_evidence=("delta_bic_baseline_mean", lambda x: float(np.nanmax(-np.asarray(x, dtype=float)))))
            .sort_values("best_evidence", ascending=False)
            .head(6)
        )
        colors = plt.cm.viridis(np.linspace(0.10, 0.90, len(best)))
        for color, split_block in zip(colors, best["split_block"].tolist()):
            g = split_rows[np.isclose(split_rows["split_block"].astype(float), float(split_block))].sort_values("time_sec")
            x = g["time_sec"].to_numpy(float)
            y = -g["delta_bic_baseline_mean"].to_numpy(float)
            ax.plot(x, y, color=color, linewidth=1.5, alpha=0.9, label=f"Transition B{int(split_block)}")

    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("Evidence above baseline model")
    ax.set_title(spec["title"])
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    setup_axis(ax)
    fig.tight_layout()
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
