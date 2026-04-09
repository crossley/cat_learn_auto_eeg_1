#!/usr/bin/env python3
"""Compute visual-motor functional connectivity across days."""

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import mne
from util_func_wrangle import util_wrangle_load_sessions

os.environ["NUMBA_DISABLE_JIT"] = "1"

_CODE_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CODE_DIR.parent
_OUTPUT_ROOT = _PROJECT_DIR / "output"
_FIGURES_ROOT = _PROJECT_DIR / "figures"


def _resolve_connectivity_func():
    try:
        from mne_connectivity import spectral_connectivity_epochs  # type: ignore

        return spectral_connectivity_epochs, "mne_connectivity"
    except Exception:
        pass

    try:
        from mne.connectivity import spectral_connectivity_epochs  # type: ignore

        return spectral_connectivity_epochs, "mne.connectivity"
    except Exception:
        return None, None


def util_connect_compute_visual_motor(
    epo_dir: Path | str = Path("../task_eeg_preprocessed"),
    output_dir: Path | str = _OUTPUT_ROOT / "connectivity",
    figures_dir: Path | str = _FIGURES_ROOT / "connectivity",
    min_epochs: int = 20,
):
    """Compute dWPLI/ImCoh visual-motor connectivity and day-level statistics."""
    epo_dir = Path(epo_dir)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")

    session_csv = output_dir / "connectivity_session_level.csv"
    subject_day_csv = output_dir / "connectivity_day_subject_means.csv"
    model_csv = output_dir / "connectivity_mixed_model_results.csv"
    qc_csv = output_dir / "connectivity_qc_log.csv"

    qc_columns = ["session_file", "subject", "day", "stage", "reason", "detail"]
    qc_rows = []
    conn_func, conn_source = _resolve_connectivity_func()
    if conn_func is None:
        msg = (
            "spectral_connectivity_epochs unavailable. Install mne-connectivity "
            "or use an MNE version exposing mne.connectivity.spectral_connectivity_epochs."
        )
        qc_rows.append(
            {
                "session_file": "",
                "subject": np.nan,
                "day": np.nan,
                "stage": "dependency",
                "reason": msg,
                "detail": "",
            }
        )
        pd.DataFrame(qc_rows, columns=qc_columns).to_csv(qc_csv, index=False)
        raise RuntimeError(msg)

    roi_visual = ["O1", "Oz", "O2"]
    roi_motor = ["C3", "Cz", "C4"]

    bands = {
        "alpha": (8.0, 12.0),
        "beta": (13.0, 30.0),
    }
    metrics = {
        "dwpli": "wpli2_debiased",
        "imcoh": "imcoh",
    }

    session_rows = []

    sessions = util_wrangle_load_sessions(epo_dir=epo_dir, preload=False)
    for item in sessions:
        session_file = item["epo_file"]
        subject = item["subject"]
        day = item["day"]
        epochs = item["epochs"]

        missing_channels = [
            ch for ch in (roi_visual + roi_motor) if ch not in epochs.ch_names
        ]
        if missing_channels:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "roi_channels",
                    "reason": "missing_roi_channels",
                    "detail": ",".join(missing_channels),
                }
            )
            continue

        stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
        if not stim_events:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "event_select",
                    "reason": "no_stim_events",
                    "detail": "",
                }
            )
            continue

        stim_epochs = epochs[stim_events]
        if len(stim_epochs) < min_epochs:
            qc_rows.append(
                {
                    "session_file": session_file,
                    "subject": subject,
                    "day": day,
                    "stage": "epoch_count",
                    "reason": "insufficient_epochs",
                    "detail": f"n_stim_epochs={len(stim_epochs)} < min_epochs={min_epochs}",
                }
            )
            continue

        tmax_use = min(0.6, float(stim_epochs.tmax))
        work_epochs = (
            stim_epochs.copy()
            .load_data()
            .crop(tmin=0.1, tmax=tmax_use)
        )
        work_epochs.pick(roi_visual + roi_motor)

        vis_idx = [work_epochs.ch_names.index(ch) for ch in roi_visual]
        mot_idx = [work_epochs.ch_names.index(ch) for ch in roi_motor]
        indices = (
            np.repeat(vis_idx, len(mot_idx)),
            np.tile(mot_idx, len(vis_idx)),
        )

        for band_name, (fmin, fmax) in bands.items():
            for metric_name, method_name in metrics.items():
                try:
                    conn = conn_func(
                        work_epochs,
                        method=method_name,
                        mode="multitaper",
                        indices=indices,
                        fmin=fmin,
                        fmax=fmax,
                        faverage=True,
                        verbose="ERROR",
                    )
                    conn_data = np.asarray(conn.get_data()).squeeze()
                    conn_value = float(np.nanmean(np.real(conn_data)))
                except Exception as exc:
                    qc_rows.append(
                        {
                            "session_file": session_file,
                            "subject": subject,
                            "day": day,
                            "stage": "connectivity",
                            "reason": "compute_error",
                            "detail": (
                                f"metric={metric_name}, band={band_name}, error={exc}"
                            ),
                        }
                    )
                    continue

                session_rows.append(
                    {
                        "session_file": session_file,
                        "subject": subject,
                        "day": day,
                        "metric": metric_name,
                        "method": method_name,
                        "band": band_name,
                        "fmin": fmin,
                        "fmax": fmax,
                        "n_stim_epochs_used": int(len(work_epochs)),
                        "roi_visual": ",".join(roi_visual),
                        "roi_motor": ",".join(roi_motor),
                        "n_channel_pairs": int(len(indices[0])),
                        "connectivity_value": conn_value,
                        "connectivity_backend": conn_source,
                    }
                )

    session_df = pd.DataFrame(session_rows)
    qc_df = pd.DataFrame(qc_rows, columns=qc_columns)

    if session_df.empty:
        session_df.to_csv(session_csv, index=False)
        qc_df.to_csv(qc_csv, index=False)
        pd.DataFrame().to_csv(subject_day_csv, index=False)
        pd.DataFrame().to_csv(model_csv, index=False)
        raise RuntimeError("Connectivity stage produced no valid session rows.")

    subject_day_df = (
        session_df.groupby(["subject", "day", "metric", "band"], as_index=False)[
            "connectivity_value"
        ]
        .mean()
        .sort_values(["metric", "band", "subject", "day"])
    )

    model_rows = []
    for (metric_name, band_name), g in subject_day_df.groupby(["metric", "band"]):
        if g["subject"].nunique() < 2 or g["day"].nunique() < 2:
            model_rows.append(
                {
                    "metric": metric_name,
                    "band": band_name,
                    "n_subjects": int(g["subject"].nunique()),
                    "n_rows": int(len(g)),
                    "day_coef": np.nan,
                    "day_se": np.nan,
                    "day_pvalue": np.nan,
                    "day_ci_low": np.nan,
                    "day_ci_high": np.nan,
                    "status": "insufficient_data",
                    "detail": "Need >=2 subjects and >=2 day values",
                }
            )
            continue

        try:
            model = smf.mixedlm(
                "connectivity_value ~ day",
                data=g,
                groups=g["subject"],
            ).fit(reml=False, method="lbfgs", disp=False)

            ci = model.conf_int().loc["day"]
            model_rows.append(
                {
                    "metric": metric_name,
                    "band": band_name,
                    "n_subjects": int(g["subject"].nunique()),
                    "n_rows": int(len(g)),
                    "day_coef": float(model.params["day"]),
                    "day_se": float(model.bse["day"]),
                    "day_pvalue": float(model.pvalues["day"]),
                    "day_ci_low": float(ci[0]),
                    "day_ci_high": float(ci[1]),
                    "status": "ok",
                    "detail": "",
                }
            )
        except Exception as exc:
            try:
                ols = smf.ols("connectivity_value ~ day", data=g).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": g["subject"]},
                )
                ci = ols.conf_int().loc["day"]
                model_rows.append(
                    {
                        "metric": metric_name,
                        "band": band_name,
                        "n_subjects": int(g["subject"].nunique()),
                        "n_rows": int(len(g)),
                        "day_coef": float(ols.params["day"]),
                        "day_se": float(ols.bse["day"]),
                        "day_pvalue": float(ols.pvalues["day"]),
                        "day_ci_low": float(ci[0]),
                        "day_ci_high": float(ci[1]),
                        "status": "ols_fallback",
                        "detail": f"mixedlm_error={exc}",
                    }
                )
            except Exception as exc2:
                model_rows.append(
                    {
                        "metric": metric_name,
                        "band": band_name,
                        "n_subjects": int(g["subject"].nunique()),
                        "n_rows": int(len(g)),
                        "day_coef": np.nan,
                        "day_se": np.nan,
                        "day_pvalue": np.nan,
                        "day_ci_low": np.nan,
                        "day_ci_high": np.nan,
                        "status": "model_error",
                        "detail": f"mixedlm_error={exc}; ols_error={exc2}",
                    }
                )

    model_df = pd.DataFrame(model_rows).sort_values(["metric", "band"])

    figure_paths = {}
    for (metric_name, band_name), g in subject_day_df.groupby(["metric", "band"]):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for subject, g_sub in g.groupby("subject"):
            g_sub = g_sub.sort_values("day")
            ax.plot(
                g_sub["day"],
                g_sub["connectivity_value"],
                color="gray",
                alpha=0.35,
                linewidth=1.2,
            )

        g_mean = g.groupby("day", as_index=False)["connectivity_value"].mean()
        ax.plot(
            g_mean["day"],
            g_mean["connectivity_value"],
            color="tab:blue",
            marker="o",
            linewidth=2.4,
            label="Group mean",
        )
        ax.set_title(f"Visual-Motor Connectivity Across Days ({metric_name}, {band_name})")
        ax.set_xlabel("Day")
        ax.set_ylabel("Connectivity")
        ax.legend(loc="best")
        ax.grid(alpha=0.25)

        fig_path = figures_dir / f"connectivity_{metric_name}_{band_name}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        figure_paths[f"{metric_name}_{band_name}"] = fig_path

    session_df.to_csv(session_csv, index=False)
    subject_day_df.to_csv(subject_day_csv, index=False)
    model_df.to_csv(model_csv, index=False)
    qc_df.to_csv(qc_csv, index=False)

    print(f"Wrote connectivity session table: {session_csv}")
    print(f"Wrote connectivity subject-day table: {subject_day_csv}")
    print(f"Wrote connectivity mixed-model results: {model_csv}")
    print(f"Wrote connectivity QC log: {qc_csv}")
    print(f"Saved connectivity figures: {len(figure_paths)}")
    for key, path in sorted(figure_paths.items()):
        print(f"- {key}: {path}")

    return {
        "session_df": session_df,
        "subject_day_df": subject_day_df,
        "model_df": model_df,
        "qc_df": qc_df,
        "session_csv": session_csv,
        "subject_day_csv": subject_day_csv,
        "model_csv": model_csv,
        "qc_csv": qc_csv,
        "figure_paths": figure_paths,
    }
