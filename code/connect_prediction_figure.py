#!/usr/bin/env python3
"""Plot schematic one-stage and two-stage connectivity predictions."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
DAYS = [1, 2, 3, 4, 5]


def gaussian(t, mu, sigma):
    return np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def save_fig_connect_prediction(figures_dir=FIGURES_DIR):
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "connect_prediction_models.png"

    t = np.linspace(0.0, 0.8, 300)
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(DAYS)))

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.8), sharex=True, sharey=True)

    ax = axes[0]
    for idx, day in enumerate(DAYS):
        amp = 0.60 + 0.08 * float(day - 1)
        y = amp * gaussian(t, 0.48, 0.10)
        ax.plot(t, y, color=colors[idx], linewidth=1.8, label=f"D{day}")
    ax.set_title("One-stage connectivity")
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.set_ylabel("Predicted connectivity")
    ax.axvspan(0.40, 0.60, color="0.90", zorder=0)
    ax.grid(alpha=0.25)

    ax = axes[1]
    for idx, day in enumerate(DAYS):
        late_amp = 1.00 - 0.18 * float(day - 1)
        early_amp = 0.08 + 0.20 * float(day - 1)
        late_y = late_amp * gaussian(t, 0.56, 0.09)
        early_y = early_amp * gaussian(t, 0.34, 0.08)
        ax.plot(t, late_y, color=colors[idx], linewidth=1.8, linestyle=":")
        ax.plot(t, early_y, color=colors[idx], linewidth=1.8)
    ax.set_title("Two-stage connectivity")
    ax.set_xlabel("Time from stimulus onset (s)")
    ax.axvspan(0.40, 0.60, color="0.90", zorder=0)
    ax.grid(alpha=0.25)

    handles = []
    labels = []
    for idx, day in enumerate(DAYS):
        handle = plt.Line2D([0], [0], color=colors[idx], linewidth=1.8)
        handles.append(handle)
        labels.append(f"D{day}")
    stage_handles = [
        plt.Line2D([0], [0], color="0.25", linewidth=1.8),
        plt.Line2D([0], [0], color="0.25", linewidth=1.8, linestyle=":"),
    ]
    stage_labels = ["earlier-onset pathway", "later-onset pathway"]
    for handle in stage_handles:
        handles.append(handle)
    for label in stage_labels:
        labels.append(label)

    fig.legend(
        handles,
        labels,
        loc="center right",
        frameon=False,
        fontsize=8,
        bbox_to_anchor=(0.99, 0.50),
    )
    fig.suptitle("A Priori Connectivity Predictions")
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.08, right=0.78, wspace=0.24)
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Connectivity prediction] Wrote {fig_path}")
    return {"prediction_models": fig_path}


if __name__ == "__main__":
    save_fig_connect_prediction()
