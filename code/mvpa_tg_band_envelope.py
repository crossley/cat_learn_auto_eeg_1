#!/usr/bin/env python3
"""Cross-day TG on band-limited amplitude envelopes (Hilbert)."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from analysis_utils import parallel_collect
from mvpa_tg_cross_day import process_cross_day_pair, write_cross_day_outputs
from mvpa_tg_window_structure import extract_tg_window_auc, fit_tg_window_gradients
from mvpa_tg_within_day import pick_eeg_interpolate_bads, session_cache_key

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
FIGURES_DIR = PROJECT_DIR / "figures"
N_JOBS = 8

BANDS: dict[str, tuple[float, float]] = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def band_tg_matrix_glob(band_name: str) -> str:
    return f"band_tg_{band_name}_matrix_sub_*_trainD*_testD*.npz"


def _sem(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(len(x)))


def load_sessions_for_band_tg(project_dir: Path | str = PROJECT_DIR):
    import re
    project_dir = Path(project_dir)
    beh_dir = project_dir / "Behavioural"
    epo_dir = project_dir / "EEG_epo"
    beh_re = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")
    epo_re = re.compile(r"^P(\d+)_D([\d_]+)-epo\.fif$")

    beh_map = {}
    for beh_path in sorted(beh_dir.glob("*.csv")):
        m = beh_re.match(beh_path.name)
        if m is None:
            continue
        subject = int(m.group(1))
        day = int(m.group(2)) // 100
        beh_map[(subject, day)] = beh_path

    epo_map = {}
    for epo_path in sorted(epo_dir.glob("*-epo.fif")):
        m = epo_re.match(epo_path.name)
        if m is None:
            continue
        subject = int(m.group(1))
        day = int(m.group(2).split("_")[0])
        epo_map[(subject, day)] = epo_path

    sessions = []
    for key in sorted(set(beh_map) & set(epo_map)):
        subject, day = key
        sessions.append(
            {
                "subject": subject,
                "day": day,
                "beh_file": beh_map[key].name,
                "epo_file": epo_map[key].name,
                "epo_path": str(epo_map[key]),
            }
        )
    return sessions


def prepare_band_envelope_cache(
    session_item, cache_dir, band_name, fmin, fmax, min_epochs, random_state
):
    subject = int(session_item["subject"])
    day = int(session_item["day"])
    session_file = session_item["epo_file"]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        cache_dir
        / f"band_tg_{band_name}_envelope_cache_interp_bads_{session_cache_key(session_item)}.npz"
    )

    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as z:
            y = z["y"]
            t = z["t"]
            ch_names = z["ch_names"]
        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "session_file": session_file,
            "cache_path": str(cache_path),
            "n_trials": int(len(y)),
            "n_a": int(np.sum(y == 0)),
            "n_b": int(np.sum(y == 1)),
            "n_times": int(len(t)),
            "ch_names": ch_names.tolist(),
        }

    try:
        epochs = mne.read_epochs(session_item["epo_path"], preload=False, verbose="ERROR")
        stim_events = [x for x in ["Stim/A", "Stim/B"] if x in epochs.event_id]
        if len(stim_events) < 2:
            raise ValueError(f"missing_stim_labels:{','.join(stim_events)}")
        epochs = epochs[stim_events].copy().load_data()
        pick_eeg_interpolate_bads(epochs)

        analysis_tmin, analysis_tmax = -0.2, 0.8
        if band_name == "delta":
            left_margin = analysis_tmin - float(epochs.times[0])
            right_margin = float(epochs.times[-1]) - analysis_tmax
            if left_margin < 1.0 or right_margin < 1.0:
                return {
                    "ok": False,
                    "qc": {
                        "session_file": session_file,
                        "subject": subject,
                        "day": day,
                        "stage": "filter_margin",
                        "reason": "delta_skipped_insufficient_epoch_margin",
                        "detail": (
                            f"Need >=1 s outside analysis window; "
                            f"left={left_margin:.3f}, right={right_margin:.3f}"
                        ),
                    },
                }

        epochs.filter(
            l_freq=fmin,
            h_freq=fmax,
            method="fir",
            fir_design="firwin",
            phase="zero-double",
            pad="reflect_limited",
            verbose="ERROR",
        )
        epochs.apply_hilbert(envelope=True, verbose="ERROR")
        epochs.resample(128, npad="auto")

        codes = epochs.events[:, 2]
        y = np.full(len(codes), -1, dtype=int)
        y[codes == epochs.event_id["Stim/A"]] = 0
        y[codes == epochs.event_id["Stim/B"]] = 1
        keep = y >= 0
        y = y[keep]
        X = epochs.get_data()[keep]
        t = epochs.times.copy()
        ch_names = np.array(epochs.ch_names, dtype=str)
        if len(y) < min_epochs:
            raise ValueError(f"insufficient_epochs:n_trials={len(y)} < min_epochs={min_epochs}")
        if min(np.sum(y == 0), np.sum(y == 1)) < 5:
            raise ValueError(
                f"insufficient_class_trials:n_a={int(np.sum(y==0))}, n_b={int(np.sum(y==1))}"
            )
        np.savez_compressed(cache_path, X=X, y=y, t=t, ch_names=ch_names)
        return {
            "ok": True,
            "subject": subject,
            "day": day,
            "session_file": session_file,
            "cache_path": str(cache_path),
            "n_trials": int(len(y)),
            "n_a": int(np.sum(y == 0)),
            "n_b": int(np.sum(y == 1)),
            "n_times": int(len(t)),
            "ch_names": ch_names.tolist(),
        }
    except Exception as exc:
        msg = str(exc)
        reason = msg.split(":")[0] if ":" in msg else "prep_error"
        return {
            "ok": False,
            "qc": {
                "session_file": session_file,
                "subject": subject,
                "day": day,
                "stage": "prepare_band_envelope",
                "reason": reason,
                "detail": msg,
            },
        }


def prepare_band_envelope_cache_from_task(task):
    return prepare_band_envelope_cache(
        task["session_item"],
        task["cache_dir"],
        task["band_name"],
        task["fmin"],
        task["fmax"],
        task["min_epochs"],
        task["random_state"],
    )


def process_cross_day_pair_from_task(task):
    return process_cross_day_pair(task["pair_item"], random_state=task["random_state"])


def summarize_band_tg_outputs(
    output_root: Path | str = PROJECT_DIR / "output",
    bands: dict[str, tuple[float, float]] = BANDS,
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_root = Path(output_root)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for band_name in bands:
        matrix_dir = output_root
        for path in sorted(matrix_dir.glob(band_tg_matrix_glob(band_name))):
            from mvpa_tg_window_structure import parse_matrix_path
            parsed = parse_matrix_path(path)
            if parsed is None:
                continue
            subject, train_day, test_day = parsed
            with np.load(path, allow_pickle=False) as z:
                mat = np.asarray(z["auc"], dtype=float)
                t = np.asarray(z["time_sec"], dtype=float)
            diag = np.diag(mat)
            for time_sec, auc in zip(t, diag):
                rows.append(
                    {
                        "band": band_name,
                        "subject": subject,
                        "train_day": train_day,
                        "test_day": test_day,
                        "time_sec": float(time_sec),
                        "auc": float(auc),
                    }
                )
    diag_df = pd.DataFrame(rows)
    diag_csv = output_dir / "band_tg_diagonal_timecourse_subject_pairs.csv"
    mean_csv = output_dir / "band_tg_diagonal_timecourse_mean.csv"
    diag_df.to_csv(diag_csv, index=False)
    if diag_df.empty:
        pd.DataFrame().to_csv(mean_csv, index=False)
        return {"diag_csv": diag_csv, "mean_csv": mean_csv, "figure": None}
    subject_df = (
        diag_df.groupby(["band", "subject", "time_sec"], as_index=False)["auc"]
        .mean()
        .sort_values(["band", "subject", "time_sec"])
    )
    mean_df = (
        subject_df.groupby(["band", "time_sec"], as_index=False)
        .agg(auc_mean=("auc", "mean"), auc_sem=("auc", _sem), n_subjects=("subject", "nunique"))
        .sort_values(["band", "time_sec"])
    )
    mean_df.to_csv(mean_csv, index=False)

    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_path = figures_dir / "band_tg_diagonal_timecourses.png"
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for band_name, g in mean_df.groupby("band"):
        g = g.sort_values("time_sec")
        ax.plot(g["time_sec"], g["auc_mean"], linewidth=1.8, label=band_name)
        ax.fill_between(
            g["time_sec"],
            g["auc_mean"] - g["auc_sem"],
            g["auc_mean"] + g["auc_sem"],
            alpha=0.15,
        )
    ax.axhline(0.5, color="0.3", linestyle=":", linewidth=1)
    ax.axvline(0.0, color="0.5", linestyle=":", linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Diagonal AUC")
    ax.set_title("Band-Envelope Cross-Day TG Diagonal Timecourse")
    ax.legend(title="Band")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"diag_csv": diag_csv, "mean_csv": mean_csv, "figure": fig_path}


def run_band_tg_window_gradients(
    output_root: Path | str = PROJECT_DIR / "output",
    bands: dict[str, tuple[float, float]] = BANDS,
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_root = Path(output_root)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    slope_rows = []
    interaction_rows = []
    for band_name in bands:
        matrix_dir = output_root
        if not list(matrix_dir.glob(band_tg_matrix_glob(band_name))):
            continue
        d = extract_tg_window_auc(
            matrix_dir=matrix_dir,
            matrix_glob=band_tg_matrix_glob(band_name),
        )
        if d.empty:
            continue
        d["band"] = band_name
        rows.append(d)
        slope_df, interaction_summary, _, _ = fit_tg_window_gradients(d)
        slope_df["band"] = band_name
        slope_rows.append(slope_df)
        interaction_summary["band"] = band_name
        interaction_rows.append(interaction_summary)
    window_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    slope_df = pd.concat(slope_rows, ignore_index=True) if slope_rows else pd.DataFrame()
    interaction_df = pd.DataFrame(interaction_rows)
    window_csv = output_dir / "band_tg_window_auc_subject_pairs.csv"
    slope_csv = output_dir / "band_tg_window_gradient_slopes.csv"
    interaction_csv = output_dir / "band_tg_window_slope_differences.csv"
    window_df.to_csv(window_csv, index=False)
    slope_df.to_csv(slope_csv, index=False)
    interaction_df.to_csv(interaction_csv, index=False)
    fig_path = None
    if not slope_df.empty:
        figures_dir.mkdir(parents=True, exist_ok=True)
        fig_path = figures_dir / "band_tg_window_gradient_slopes.png"
        fig, ax = plt.subplots(figsize=(8.4, 4.4))
        band_order = [band for band in bands if band in set(slope_df["band"])]
        offsets = {"early": -0.16, "late": 0.16}
        colors = {"early": "tab:blue", "late": "tab:orange"}
        x_base = np.arange(len(band_order), dtype=float)
        for window_name in ["early", "late"]:
            g = slope_df[slope_df["window"] == window_name].set_index("band").reindex(band_order)
            x = x_base + offsets[window_name]
            y = g["estimate"].to_numpy(dtype=float)
            yerr = np.vstack(
                [y - g["ci_low"].to_numpy(dtype=float), g["ci_high"].to_numpy(dtype=float) - y]
            )
            ax.errorbar(x, y, yerr=yerr, fmt="o", capsize=3, color=colors[window_name], label=window_name)
        ax.axhline(0.0, color="0.35", linestyle=":", linewidth=1)
        ax.set_xticks(x_base)
        ax.set_xticklabels(band_order)
        ax.set_xlabel("Band")
        ax.set_ylabel("AUC slope per day distance")
        ax.set_title("Band-Envelope TG Window Gradients")
        ax.legend(title="Window")
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    return {
        "window_csv": window_csv,
        "slope_csv": slope_csv,
        "interaction_csv": interaction_csv,
        "figure": fig_path,
        "window_df": window_df,
        "slope_df": slope_df,
        "interaction_df": interaction_df,
    }


def run_band_envelope_cross_day_tg(
    bands: dict[str, tuple[float, float]] = BANDS,
    min_epochs: int = 20,
    random_state: int = 42,
    n_workers: int | None = None,
    output_root: Path | str = PROJECT_DIR / "output",
    output_dir: Path | str = OUTPUT_DIR,
    figures_dir: Path | str = FIGURES_DIR,
):
    output_root = Path(output_root)
    output_dir = Path(output_dir)
    figures_dir = Path(figures_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    mne.set_log_level("ERROR")
    sessions = load_sessions_for_band_tg()
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    band_results = {}
    progress = {}
    for band_name, (fmin, fmax) in bands.items():
        t0 = time.time()
        cache_dir = output_root
        print(
            f"[Band TG] Preparing {band_name} envelope caches for {len(sessions)} sessions "
            f"(n_workers={n_workers})...",
            flush=True,
        )
        prep_tasks = [
            {
                "session_item": item,
                "cache_dir": cache_dir,
                "band_name": band_name,
                "fmin": fmin,
                "fmax": fmax,
                "min_epochs": min_epochs,
                "random_state": random_state,
            }
            for item in sessions
        ]
        prep_results = parallel_collect(prepare_band_envelope_cache_from_task, prep_tasks, n_workers)
        prepared = [r for r in prep_results if r["ok"]]
        qc_rows = [r["qc"] for r in prep_results if not r["ok"]]
        day_data = {
            (int(r["subject"]), int(r["day"])): {
                "cache_path": r["cache_path"],
                "session_file": r["session_file"],
            }
            for r in prepared
        }
        pair_items = []
        rng = np.random.default_rng(random_state)
        for subject in sorted({k[0] for k in day_data}):
            days = sorted(k[1] for k in day_data if k[0] == subject)
            for train_day in days:
                for test_day in days:
                    if train_day == test_day:
                        continue
                    pair_items.append(
                        {
                            "subject": subject,
                            "train_day": train_day,
                            "test_day": test_day,
                            "train_cache_path": day_data[(subject, train_day)]["cache_path"],
                            "test_cache_path": day_data[(subject, test_day)]["cache_path"],
                            "train_session_file": day_data[(subject, train_day)]["session_file"],
                            "test_session_file": day_data[(subject, test_day)]["session_file"],
                            "pair_seed": int(rng.integers(0, 2**31 - 1)),
                        }
                    )

        print(
            f"[Band TG] Running {band_name} cross-day TG on {len(pair_items)} directed pairs "
            f"(prepared_sessions={len(prepared)}, n_workers={n_workers})...",
            flush=True,
        )

        matrix_dir = output_root
        matrix_dir.mkdir(parents=True, exist_ok=True)
        cross_subject_csv = output_root / f"band_tg_{band_name}_cross_day_subject_level.csv"
        cross_day_mean_csv = output_root / f"band_tg_{band_name}_cross_day_day_mean.csv"
        cross_matrix_day_mean_csv = (
            output_root / f"band_tg_{band_name}_cross_day_timegen_day_mean.csv"
        )

        if len(pair_items) == 0:
            cross_rows = [{"ok": False, "qc": r} for r in qc_rows]
        else:
            pair_tasks = [{"pair_item": item, "random_state": random_state} for item in pair_items]
            cross_rows = parallel_collect(process_cross_day_pair_from_task, pair_tasks, n_workers)
            cross_rows.extend({"ok": False, "qc": r} for r in qc_rows)

        ok_results = [r for r in cross_rows if r.get("ok")]
        fail_qc = [r["qc"] for r in cross_rows if not r.get("ok") and "qc" in r]

        cross_accum: dict = {}
        cross_time_template = None
        cross_row_dicts = []
        for result in ok_results:
            row = result["row"]
            mat = np.asarray(result["mat"], dtype=float)
            t_vec = np.asarray(result["t"], dtype=float)
            cross_row_dicts.append(row)
            np.savez_compressed(
                matrix_dir / (
                    f"band_tg_{band_name}_matrix_sub_{int(row['subject']):03d}"
                    f"_trainD{int(row['train_day'])}"
                    f"_testD{int(row['test_day'])}.npz"
                ),
                auc=mat,
                time_sec=t_vec,
            )
            if cross_time_template is None:
                cross_time_template = t_vec
            key = (int(row["train_day"]), int(row["test_day"]))
            if key not in cross_accum:
                cross_accum[key] = {
                    "sum": np.zeros_like(mat, dtype=float),
                    "count": np.zeros_like(mat, dtype=float),
                }
            valid = np.isfinite(mat)
            cross_accum[key]["sum"][valid] += mat[valid]
            cross_accum[key]["count"][valid] += 1.0

        out = write_cross_day_outputs(
            cross_rows=cross_row_dicts,
            cross_matrix_accum=cross_accum,
            cross_time_template=cross_time_template,
            cross_matrix_dir=matrix_dir,
            cross_subject_csv=cross_subject_csv,
            cross_day_mean_csv=cross_day_mean_csv,
            cross_matrix_day_mean_csv=cross_matrix_day_mean_csv,
        )
        if fail_qc:
            pd.DataFrame(fail_qc).to_csv(
                output_root / f"band_tg_{band_name}_qc_log.csv", index=False
            )

        band_results[band_name] = out
        progress[band_name] = {
            "prepared_sessions": len(prepared),
            "cross_day_pairs": len(pair_items),
            "elapsed_sec": time.time() - t0,
            "output_dir": str(output_root),
            "n_workers": n_workers,
        }
        (output_root / f"band_tg_{band_name}_progress.json").write_text(
            json.dumps(progress[band_name], indent=2)
        )

    summary = summarize_band_tg_outputs(
        output_root=output_root, bands=bands, output_dir=output_dir, figures_dir=figures_dir
    )
    band_gradient = run_band_tg_window_gradients(
        output_root=output_root, bands=bands, output_dir=output_dir, figures_dir=figures_dir
    )
    return {
        "band_results": band_results,
        "progress": progress,
        "summary": summary,
        "band_gradient": band_gradient,
    }


if __name__ == "__main__":
    run_band_envelope_cross_day_tg()
