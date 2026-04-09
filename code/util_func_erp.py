#!/usr/bin/env python3
"""ERP utilities to create evoked responses and write figure outputs."""

from pathlib import Path
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
from util_func_wrangle import util_wrangle_load_sessions

os.environ["NUMBA_DISABLE_JIT"] = "1"


def util_erp_make_figures():
    """
    Build grand-average ERPs and save figures.
    """

    output_dir = Path("../output/erp")
    figures_dir = Path("../figures/erp")

    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    event_map = {
        "stim_a": ["Stim/A"],
        "stim_b": ["Stim/B"],
        "stim_all": ["Stim/A", "Stim/B"],
        "fb_cor": ["FB/Cor"],
        "fb_inc": ["FB/Inc"],
        "fb_all": ["FB/Cor", "FB/Inc"],
    }

    evoked_stim_all_rec = []
    sessions = util_wrangle_load_sessions()
    d = pd.DataFrame(sessions)

    for sbj in d["subject"].unique():
        ds = d[d["subject"] == sbj]
        for day in ds["day"].unique():
            dsd = ds[ds["day"] == day]
            beh_df = dsd["beh_df"].copy()
            epochs = dsd["epochs"].iloc[0]
            evoked_stim_all = epochs[event_map["stim_all"]].average()
            evoked_stim_all_rec.append(
                {
                    "subject": sbj,
                    "day": day,
                    "evoked_stim_all": evoked_stim_all,
                }
            )

    evoked_stim_all_rec = pd.DataFrame(evoked_stim_all_rec)

    for s in evoked_stim_all_rec["subject"].unique():
        ds = evoked_stim_all_rec[evoked_stim_all_rec["subject"] == s]
        fig, axes = plt.subplots(1, 5, figsize=(25, 4), squeeze=False)
        for i, day in enumerate(ds["day"].unique()):
            ax = axes[0, i]
            evoked = ds.loc[ds["day"] == day, "evoked_stim_all"].iloc[0]
            evoked.plot(
                axes=ax,
                show=False,
                spatial_colors=True,
                titles=f"Day {day}",
            )
            ax.set_title(f"Day {day}")
        fig.suptitle(f"ERP: stim_all -- subject {s}")
        fig.savefig(figures_dir / f"erp_stim_all_sub_{s}.png")
        plt.close(fig)

    evoked_stim_all_mean = {}
    for day, g in evoked_stim_all_rec.groupby("day"):
        evoked_stim_all_mean[day] = mne.grand_average(g["evoked_stim_all"].tolist())

    fig, axes = plt.subplots(1, 5, figsize=(25, 4), squeeze=False)
    for i, day in enumerate(evoked_stim_all_rec["day"].unique()):
        ax = axes[0, i]
        evoked_stim_all_mean[day].plot(
            axes=ax,
            show=False,
            spatial_colors=True,
            titles=f"Day {day}",
        )
        ax.set_title(f"Day {day}")

    fig.suptitle("Grand Average ERP: stim_all")
    fig.savefig(figures_dir / "erp_stim_all.png")
    plt.close(fig)
