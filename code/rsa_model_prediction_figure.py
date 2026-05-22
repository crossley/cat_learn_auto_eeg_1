#!/usr/bin/env python3
"""Plot RSA model-prediction figures from saved outputs."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rsa_model_prediction_analysis import (
    FIGURES_DIR,
    MIN_RETAINED_BIN_PROP,
    MIN_RETAINED_BINS,
    MIN_TRIALS_PER_BIN_SESSION,
    OUTPUT_DIR,
)


def plot_grid_diagnostics(diagnostics, selected_n, fig_path):
    if diagnostics.empty:
        raise ValueError("Empty RSA model grid diagnostics table")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), squeeze=False)
    axes = axes.ravel()
    x = diagnostics["n_bins"].to_numpy()
    for ax, col, ylabel in [
        (axes[0], "n_retained_bins", "Retained bins"),
        (axes[1], "retained_bin_prop", "Retained bin proportion"),
        (axes[2], "median_trials_nonempty", "Median trials / subject-day bin"),
    ]:
        ax.plot(x, diagnostics[col], marker="o", color="black")
        ax.axvline(selected_n, color="tab:red", linestyle="--", linewidth=1)
        ax.set_xlabel("Grid dimension")
        ax.set_ylabel(ylabel)
        ax.invert_xaxis()
    axes[1].axhline(MIN_RETAINED_BIN_PROP, color="0.5", linestyle=":", linewidth=1)
    axes[2].axhline(MIN_TRIALS_PER_BIN_SESSION, color="0.5", linestyle=":", linewidth=1)
    fig.suptitle(f"RSA stimulus-bin grid selection: selected {selected_n}x{selected_n}")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_model_predictions(all_bins, retained_bins, rdms, fig_path):
    if all_bins.empty:
        raise ValueError("Empty RSA model stimulus-bin table")
    if retained_bins.empty:
        raise ValueError("No retained RSA model stimulus bins")
    if len(rdms) == 0:
        raise ValueError("No RSA model RDMs available for plotting")
    fig = plt.figure(figsize=(15.5, 9.0))
    gs = fig.add_gridspec(
        2,
        3,
        left=0.06,
        right=0.90,
        top=0.88,
        bottom=0.09,
        wspace=0.36,
        hspace=0.42,
    )

    ax_cat = fig.add_subplot(gs[0, 0])
    nonempty = all_bins[all_bins["n_trials"] > 0]
    for label, color in [("A", "tab:blue"), ("B", "tab:orange")]:
        d = nonempty[nonempty["category_label"] == label]
        ax_cat.scatter(
            d["x_center"], d["y_center"],
            s=25, alpha=0.7, color=color, edgecolor="none", label=label,
        )
    ax_cat.set_xlabel("x")
    ax_cat.set_ylabel("y")
    ax_cat.set_title("Category distribution")
    ax_cat.set_aspect("equal", adjustable="box")
    ax_cat.legend(title="Category", fontsize=8)

    ax_space = fig.add_subplot(gs[1, 0])
    bin_colors = plt.cm.tab20(
        np.linspace(0, 1, max(len(retained_bins), 2), endpoint=False)
    )
    for _, row in all_bins.iterrows():
        ax_space.scatter(
            row["x_center"],
            row["y_center"],
            s=25,
            alpha=0.18,
            color="0.55",
            edgecolor="none",
        )
    for _, row in retained_bins.iterrows():
        idx = int(row["rdm_order"])
        ax_space.scatter(
            row["x_center"],
            row["y_center"],
            s=70,
            alpha=0.95,
            color=bin_colors[idx],
            edgecolor="black",
            linewidth=0.4,
        )
        ax_space.annotate(
            str(idx),
            (row["x_center"], row["y_center"]),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=8,
            color="white",
            bbox={"boxstyle": "round,pad=0.12", "facecolor": "black", "edgecolor": "none"},
        )
    ax_space.set_xlabel("x")
    ax_space.set_ylabel("y")
    ax_space.set_title("Retained stimulus bins")
    ax_space.set_aspect("equal", adjustable="box")

    axes = [
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 1]),
        fig.add_subplot(gs[1, 2]),
    ]
    for ax, (name, mat) in zip(axes, rdms.items()):
        im = ax.imshow(mat, origin="lower", cmap="viridis", vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_xlabel("Stimulus bin")
        ax.set_ylabel("Stimulus bin")
    cax = fig.add_axes([0.93, 0.22, 0.012, 0.56])
    fig.colorbar(im, cax=cax, label="Model dissimilarity")
    fig.suptitle("RSA model RDM predictions")
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_fig_rsa_model_predictions(
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_csv = output_dir / "rsa_model_grid_diagnostics.csv"
    bins_csv = output_dir / "rsa_model_stimulus_bins.csv"
    rdm_csv = output_dir / "rsa_model_rdms.csv"
    if not diagnostics_csv.exists() or not bins_csv.exists() or not rdm_csv.exists():
        raise FileNotFoundError(
            "Missing RSA model outputs. Run rsa_model_prediction_analysis.py first."
        )
    diagnostics = pd.read_csv(diagnostics_csv)
    all_bins = pd.read_csv(bins_csv)
    retained_bins = all_bins[all_bins["retained"]].copy()
    retained_bins = retained_bins.sort_values(
        ["category_label", "boundary_signed_center", "sf_bin", "ori_bin"]
    ).reset_index(drop=True)
    retained_bins["rdm_order"] = np.arange(len(retained_bins))
    rdm_df = pd.read_csv(rdm_csv)
    if diagnostics.empty:
        raise ValueError(f"Empty RSA model diagnostics table: {diagnostics_csv}")
    if all_bins.empty:
        raise ValueError(f"Empty RSA model bins table: {bins_csv}")
    if rdm_df.empty:
        raise ValueError(f"Empty RSA model RDM table: {rdm_csv}")
    rdms = {}
    for model, g in rdm_df.groupby("model", sort=False):
        n_bins = int(max(g["bin_i"].max(), g["bin_j"].max()) + 1)
        mat = np.full((n_bins, n_bins), np.nan)
        for row in g.itertuples():
            mat[int(row.bin_i), int(row.bin_j)] = float(row.dissimilarity)
        rdms[model] = mat
    selected_n = int(
        diagnostics.sort_values(
            ["n_retained_bins", "n_bins"], ascending=[False, False]
        ).iloc[0]["n_bins"]
    )
    selected_rows = diagnostics[
        (diagnostics["n_retained_bins"] >= MIN_RETAINED_BINS)
        & (diagnostics["retained_bin_prop"] >= MIN_RETAINED_BIN_PROP)
    ]
    if not selected_rows.empty:
        selected_n = int(selected_rows.sort_values("n_bins", ascending=False).iloc[0]["n_bins"])
    diagnostics_fig = figures_dir / "rsa_model_grid_diagnostics.png"
    prediction_fig = figures_dir / "rsa_model_prediction_rdms.png"
    plot_grid_diagnostics(diagnostics, selected_n, diagnostics_fig)
    plot_model_predictions(all_bins, retained_bins, rdms, prediction_fig)
    return {
        "diagnostics_figure": diagnostics_fig,
        "prediction_figure": prediction_fig,
    }




if __name__ == "__main__":
    save_fig_rsa_model_predictions()
