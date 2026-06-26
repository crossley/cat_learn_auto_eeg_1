#!/usr/bin/env python3
"""Plot time-resolved ERP day-structure model evidence."""

from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from erp_model_timecourse_analysis import OUTPUT_DIR
from figure_style import FIGURES_DIR, setup_axis


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing ERP model-timecourse output: {path}. "
            "Run erp_model_timecourse_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty ERP model-timecourse output: {path}")
    return d


def save_fig_erp_model_timecourse(output_dir=OUTPUT_DIR, figures_dir=FIGURES_DIR):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    d = require_csv(output_dir / "erp_model_timecourse_summary.csv")
    models = [
        ("Continuous Restructuring", "#1f1f1f"),
        ("Discrete Restructuring D1", "#6a3d9a"),
        ("Discrete Restructuring D2", "#1b9e77"),
        ("Discrete Restructuring D3", "#377eb8"),
        ("Discrete Restructuring D4", "#a6cee3"),
    ]
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    y_vals = []
    for label, color in models:
        g = d[d["model_label"] == label].sort_values("time_center_sec")
        if g.empty:
            continue
        x = g["time_center_sec"].to_numpy(dtype=float)
        y = -g["delta_bic_baseline_mean"].to_numpy(dtype=float)
        err = g["delta_bic_baseline_sem"].to_numpy(dtype=float)
        y_vals.extend([float(v) for v in y if np.isfinite(v)])
        ax.plot(x, y, color=color, linewidth=1.8, alpha=0.92, label=label.replace(" D", " (D") + (")" if label.endswith(("D1", "D2", "D3", "D4")) else ""))
        good = np.isfinite(err)
        if np.any(good):
            ax.fill_between(
                x[good],
                y[good] - err[good],
                y[good] + err[good],
                color=color,
                alpha=0.10,
                linewidth=0,
            )
    ax.axhline(0.0, color="0.25", linewidth=0.8)
    ax.axvline(0.0, color="0.25", linewidth=0.8, linestyle="--")
    ax.set_xlim(-0.1, 0.8)
    if y_vals:
        ymin = float(np.nanmin(y_vals))
        ymax = float(np.nanmax(y_vals))
        pad = 0.12 * max(ymax - ymin, 1.0)
        ax.set_ylim(ymin - pad, ymax + pad)
    ax.set_xlabel("time from stimulus (s)")
    ax.set_ylabel("Evidence above baseline model")
    ax.set_title("ERP Model Evidence Over Time")
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    setup_axis(ax)
    fig.tight_layout()
    path = figures_dir / "erp_model_timecourse.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[ERP model timecourse] wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    save_fig_erp_model_timecourse()
