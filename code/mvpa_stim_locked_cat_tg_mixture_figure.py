#!/usr/bin/env python3
"""Plot stim-locked TG mixture model overlay figures from saved outputs."""

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

from mvpa_stim_locked_cat_tg_mixture_analysis import FIGURES_DIR, OUTPUT_DIR

DAY_GRID = [1, 2, 3, 4, 5]
MODEL_NAMES = ["model_1", "model_2", "model_3"]


def save_fig_mvpa_stim_locked_cat_tg_mixture(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    tg_day_mean_csv = output_dir / "mvpa_stim_locked_cat_tg_timegen_day_mean.csv"
    surface_csv = output_dir / "mvpa_stim_locked_cat_tg_mixture_fitted_surface.csv"

    required = [tg_day_mean_csv, surface_csv]
    missing = []
    for path in required:
        if not path.exists():
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Missing mixture model outputs in "
            f"{output_dir}. Run mvpa_stim_locked_cat_tg_analysis.py and "
            "mvpa_stim_locked_cat_tg_mixture_analysis.py first.\n"
            + "\n".join(missing)
        )

    d_tg = pd.read_csv(tg_day_mean_csv)
    if d_tg.empty:
        raise ValueError(f"Empty TG day-mean table: {tg_day_mean_csv}")

    d_surf = pd.read_csv(surface_csv)
    if d_surf.empty:
        raise ValueError(f"Empty mixture fitted surface table: {surface_csv}")

    vmin = float(d_tg["auc_mean"].min())
    vmax = float(d_tg["auc_mean"].max())

    figure_paths = {}
    for model_name in MODEL_NAMES:
        d_model = d_surf[d_surf["model"] == model_name]
        if d_model.empty:
            raise ValueError(
                f"No fitted surface rows for model={model_name} in {surface_csv}"
            )

        fig, axes = plt.subplots(5, 5, figsize=(18, 16), squeeze=False)
        im = None
        for i, train_day in enumerate(DAY_GRID):
            for j, test_day in enumerate(DAY_GRID):
                ax = axes[i, j]

                g_tg = d_tg[
                    (d_tg["train_day"] == train_day) & (d_tg["test_day"] == test_day)
                ]
                if g_tg.empty:
                    raise ValueError(
                        f"Missing TG matrix day pair in {tg_day_mean_csv}: "
                        f"train_day={train_day}, test_day={test_day}"
                    )
                pivot_tg = g_tg.pivot(
                    index="train_time_sec", columns="test_time_sec", values="auc_mean"
                )
                extent = [
                    float(pivot_tg.columns.min()),
                    float(pivot_tg.columns.max()),
                    float(pivot_tg.index.min()),
                    float(pivot_tg.index.max()),
                ]
                im = ax.imshow(
                    pivot_tg.to_numpy(),
                    origin="lower",
                    aspect="auto",
                    extent=extent,
                    vmin=vmin,
                    vmax=vmax,
                    cmap="viridis",
                )

                g_surf = d_model[
                    (d_model["train_day"] == train_day)
                    & (d_model["test_day"] == test_day)
                ]
                if not g_surf.empty:
                    pivot_surf = g_surf.pivot(
                        index="train_time_sec",
                        columns="test_time_sec",
                        values="mu_mean",
                    )
                    surf_arr = pivot_surf.to_numpy()
                    train_t = pivot_surf.index.to_numpy(dtype=float)
                    test_t = pivot_surf.columns.to_numpy(dtype=float)
                    ax.contour(
                        test_t,
                        train_t,
                        surf_arr,
                        levels=5,
                        colors="white",
                        linewidths=0.7,
                        alpha=0.7,
                    )

                ax.axvline(0.0, color="white", linestyle=":", linewidth=0.8)
                ax.axhline(0.0, color="white", linestyle=":", linewidth=0.8)
                ax.set_title(f"Train D{train_day} -> Test D{test_day}", fontsize=9)
                if i == len(DAY_GRID) - 1:
                    ax.set_xlabel("Test Time (s)")
                if j == 0:
                    ax.set_ylabel("Train Time (s)")

        fig.suptitle(
            f"Cross-Day TG with Mixture Overlay ({model_name})"
        )
        fig.subplots_adjust(
            top=0.94,
            bottom=0.05,
            left=0.05,
            right=0.90,
            wspace=0.30,
            hspace=0.35,
        )
        cax = fig.add_axes([0.92, 0.12, 0.015, 0.74])
        fig.colorbar(im, cax=cax, label="AUC")
        fig_path = figures_dir / f"mvpa_stim_locked_cat_tg_mixture_overlay_{model_name}_5x5.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths[model_name] = fig_path

    return {"figure_paths": figure_paths}


if __name__ == "__main__":
    save_fig_mvpa_stim_locked_cat_tg_mixture()
