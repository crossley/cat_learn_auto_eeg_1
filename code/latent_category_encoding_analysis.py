#!/usr/bin/env python3
"""Category-linked latent trajectory analyses."""

from __future__ import annotations

from pathlib import Path
import json
import os
import time
import warnings

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import mne
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning as SklearnConvergenceWarning
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from figure_style import OUTPUT_DIR
from latent_dynamics_utils import (
    group_distance_summary,
    score_trajectory_geometry,
    trajectory_metrics,
)
from load_project_data import load_sessions
from util_mvpa import build_clf, pick_eeg_interpolate_bads

N_JOBS = 8
RANDOM_STATE = 42
N_SPLITS = 5
N_COMPONENTS = 6
N_EVIDENCE_COMPONENTS = 3
TMIN = 0.0
TMAX = 0.8
MIN_CLASS_TRIALS = 5

ANALYSES = {
    "category_centroid": "A-minus-B ERP category contrast",
    "classifier_axis": "A-minus-B classifier decision axis",
    "category_evidence": "cross-validated category evidence",
}

EVIDENCE_FEATURES = [
    "a_evidence",
    "b_evidence",
    "signed_evidence",
    "decision_separation",
    "mean_abs_decision",
]


def prepare_stimulus_epochs(item):
    epochs = mne.read_epochs(item["epo_path"], preload=False, verbose="ERROR")
    stim_events = []
    for event_name in ["Stim/A", "Stim/B"]:
        if event_name in epochs.event_id:
            stim_events.append(event_name)
    if len(stim_events) < 2:
        raise ValueError(f"missing_stim_labels:{','.join(stim_events)}")
    stim_epochs = epochs[stim_events].copy()
    stim_epochs.load_data()
    pick_eeg_interpolate_bads(stim_epochs)
    stim_epochs.resample(128, npad="auto")

    codes = stim_epochs.events[:, 2]
    y = np.full(len(codes), -1, dtype=int)
    y[codes == stim_epochs.event_id["Stim/A"]] = 0
    y[codes == stim_epochs.event_id["Stim/B"]] = 1
    keep_trials = y >= 0
    y = y[keep_trials]
    x = stim_epochs.get_data()[keep_trials]
    times = stim_epochs.times.copy()
    keep_times = (times >= TMIN) & (times <= TMAX)
    x = x[:, :, keep_times]
    times = times[keep_times]
    if int(np.sum(y == 0)) < MIN_CLASS_TRIALS or int(np.sum(y == 1)) < MIN_CLASS_TRIALS:
        raise ValueError(
            "insufficient_class_trials:"
            f"n_a={int(np.sum(y == 0))},n_b={int(np.sum(y == 1))}"
        )
    return x, y, times, list(stim_epochs.ch_names)


def classifier_weights_and_evidence(x, y, times, ch_names, random_state):
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=random_state)
    axis = np.full((len(times), len(ch_names)), np.nan, dtype=float)
    evidence_rows = []
    for ti, time_sec in enumerate(times):
        xt = x[:, :, ti]
        clf = build_clf(random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            clf.fit(xt, y)
        scaler = clf.named_steps["scaler"]
        logreg = clf.named_steps["logreg"]
        weights = logreg.coef_.ravel().astype(float)
        scale = np.asarray(scaler.scale_, dtype=float)
        scale[scale == 0] = 1.0
        axis[ti, :] = -(weights / scale)

        clf_cv = build_clf(random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SklearnConvergenceWarning)
            decision = cross_val_predict(
                clf_cv,
                xt,
                y,
                cv=cv,
                method="decision_function",
            )
        decision = np.asarray(decision, dtype=float)
        a_decision = decision[y == 0]
        b_decision = decision[y == 1]
        signed = np.where(y == 1, decision, -decision)
        evidence_rows.append(
            {
                "time_sec": float(time_sec),
                "a_evidence": float(-np.mean(a_decision)),
                "b_evidence": float(np.mean(b_decision)),
                "signed_evidence": float(np.mean(signed)),
                "decision_separation": float(np.mean(b_decision) - np.mean(a_decision)),
                "mean_abs_decision": float(np.mean(np.abs(decision))),
            }
        )
    return axis, pd.DataFrame(evidence_rows)


def process_session(item):
    subject = int(item["subject"])
    day = int(item["day"])
    x, y, times, ch_names = prepare_stimulus_epochs(item)
    centroid = np.mean(x[y == 0], axis=0) - np.mean(x[y == 1], axis=0)
    centroid = centroid.T
    axis, evidence = classifier_weights_and_evidence(
        x,
        y,
        times,
        ch_names,
        RANDOM_STATE,
    )
    evidence["subject"] = subject
    evidence["day"] = day
    evidence["session_file"] = item["epo_file"]
    return {
        "ok": True,
        "subject": subject,
        "day": day,
        "session_file": item["epo_file"],
        "times": times,
        "ch_names": ch_names,
        "centroid": centroid,
        "classifier_axis": axis,
        "evidence": evidence,
        "n_trials": int(len(y)),
        "n_a": int(np.sum(y == 0)),
        "n_b": int(np.sum(y == 1)),
    }


def process_session_safe(item):
    try:
        if threadpool_limits is None:
            return process_session(item), None
        with threadpool_limits(limits=1):
            return process_session(item), None
    except Exception as exc:
        return None, {
            "session_file": item["epo_file"],
            "subject": int(item["subject"]),
            "day": int(item["day"]),
            "stage": "category_encoding",
            "reason": "error",
            "detail": str(exc),
        }


def validate_channel_order(results):
    ch_names = results[0]["ch_names"]
    for result in results[1:]:
        if result["ch_names"] != ch_names:
            raise ValueError("Stimulus channel order differs across sessions")
    return ch_names


def make_sensor_points(results, matrix_key, analysis, output_dir):
    ch_names = validate_channel_order(results)
    rows = []
    x_rows = []
    for result in results:
        mat = result[matrix_key]
        if mat.shape[1] != len(ch_names):
            raise ValueError(f"{analysis} matrix has unexpected channel count")
        for ti, time_sec in enumerate(result["times"]):
            rows.append(
                {
                    "analysis": analysis,
                    "subject": int(result["subject"]),
                    "day": int(result["day"]),
                    "session_file": result["session_file"],
                    "time_sec": float(time_sec),
                }
            )
            x_rows.append(mat[ti, :])
    x = np.vstack(x_rows)
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{analysis} matrix contains non-finite values")
    scaler = StandardScaler()
    xz = scaler.fit_transform(x)
    pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    scores = pca.fit_transform(xz)
    component_cols = [f"latent_{idx}" for idx in range(1, N_COMPONENTS + 1)]
    points = pd.DataFrame(rows)
    for idx, col in enumerate(component_cols):
        points[col] = scores[:, idx]

    component_rows = []
    for comp_i, var in enumerate(pca.explained_variance_ratio_, start=1):
        for channel, weight in zip(ch_names, pca.components_[comp_i - 1]):
            component_rows.append(
                {
                    "analysis": analysis,
                    "component": int(comp_i),
                    "feature": channel,
                    "weight": float(weight),
                    "explained_variance_ratio": float(var),
                }
            )
    components = pd.DataFrame(component_rows)
    write_analysis_outputs(points, components, component_cols, analysis, output_dir)


def make_evidence_points(results, output_dir):
    evidence = pd.concat([result["evidence"] for result in results], ignore_index=True)
    evidence = evidence.sort_values(["subject", "day", "time_sec"])
    x = evidence[EVIDENCE_FEATURES].to_numpy(float)
    if not np.all(np.isfinite(x)):
        raise ValueError("Category evidence matrix contains non-finite values")
    scaler = StandardScaler()
    xz = scaler.fit_transform(x)
    n_components = min(N_EVIDENCE_COMPONENTS, len(EVIDENCE_FEATURES))
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    scores = pca.fit_transform(xz)
    component_cols = [f"latent_{idx}" for idx in range(1, n_components + 1)]
    points = evidence[["subject", "day", "session_file", "time_sec"]].copy()
    points["analysis"] = "category_evidence"
    for idx, col in enumerate(component_cols):
        points[col] = scores[:, idx]

    component_rows = []
    for comp_i, var in enumerate(pca.explained_variance_ratio_, start=1):
        for feature, weight in zip(EVIDENCE_FEATURES, pca.components_[comp_i - 1]):
            component_rows.append(
                {
                    "analysis": "category_evidence",
                    "component": int(comp_i),
                    "feature": feature,
                    "weight": float(weight),
                    "explained_variance_ratio": float(var),
                }
            )
    components = pd.DataFrame(component_rows)

    evidence_path = output_dir / "latent_category_evidence_timecourse_subject.csv"
    evidence.to_csv(evidence_path, index=False)
    summary = (
        evidence.groupby(["day", "time_sec"], as_index=False)
        .agg(
            signed_evidence_mean=("signed_evidence", "mean"),
            signed_evidence_sem=(
                "signed_evidence",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
                if len(x) > 1
                else np.nan,
            ),
            decision_separation_mean=("decision_separation", "mean"),
            decision_separation_sem=(
                "decision_separation",
                lambda x: float(np.std(x, ddof=1) / np.sqrt(len(x)))
                if len(x) > 1
                else np.nan,
            ),
            n_subjects=("subject", "nunique"),
        )
        .sort_values(["day", "time_sec"])
    )
    summary.to_csv(output_dir / "latent_category_evidence_timecourse_day.csv", index=False)
    write_analysis_outputs(points, components, component_cols, "category_evidence", output_dir)


def write_analysis_outputs(points, components, component_cols, analysis, output_dir):
    distance_df, score_df, summary_df = score_trajectory_geometry(
        points,
        component_cols,
        "time_sec",
        analysis,
    )
    group_distance_df = group_distance_summary(distance_df)
    metric_df = trajectory_metrics(points, component_cols, "time_sec", analysis)

    points.to_csv(output_dir / f"latent_{analysis}_points.csv", index=False)
    components.to_csv(output_dir / f"latent_{analysis}_components.csv", index=False)
    metric_df.to_csv(output_dir / f"latent_{analysis}_metrics.csv", index=False)
    distance_df.to_csv(output_dir / f"latent_{analysis}_subject_distances.csv", index=False)
    group_distance_df.to_csv(
        output_dir / f"latent_{analysis}_group_distances.csv",
        index=False,
    )
    score_df.to_csv(output_dir / f"latent_{analysis}_model_subject.csv", index=False)
    summary_df.to_csv(output_dir / f"latent_{analysis}_model_summary.csv", index=False)


def run_latent_category_encoding(output_dir: Path | str = OUTPUT_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "latent_category_encoding_progress.json"
    t0 = time.time()
    sessions = load_sessions(load_epochs=False)
    progress_path.write_text(
        json.dumps({"stage": "sessions", "done": 0, "total": len(sessions)}, indent=2)
    )
    print(
        f"[latent category] processing {len(sessions)} sessions "
        f"with n_jobs={N_JOBS}",
        flush=True,
    )
    results_raw = Parallel(n_jobs=N_JOBS)(
        delayed(process_session_safe)(item) for item in sessions
    )

    results = []
    qc_rows = []
    for idx, (result, qc) in enumerate(results_raw, start=1):
        if result is not None:
            results.append(result)
        if qc is not None:
            qc_rows.append(qc)
        progress_path.write_text(
            json.dumps(
                {
                    "stage": "sessions",
                    "done": idx,
                    "total": len(sessions),
                    "elapsed_sec": time.time() - t0,
                },
                indent=2,
            )
        )
    pd.DataFrame(qc_rows).to_csv(
        output_dir / "latent_category_encoding_qc.csv",
        index=False,
    )
    if not results:
        raise ValueError("No category-encoding sessions were computed")

    progress_path.write_text(json.dumps({"stage": "category_centroid"}, indent=2))
    make_sensor_points(results, "centroid", "category_centroid", output_dir)

    progress_path.write_text(json.dumps({"stage": "classifier_axis"}, indent=2))
    make_sensor_points(results, "classifier_axis", "classifier_axis", output_dir)

    progress_path.write_text(json.dumps({"stage": "category_evidence"}, indent=2))
    make_evidence_points(results, output_dir)

    session_rows = []
    for result in results:
        session_rows.append(
            {
                "subject": int(result["subject"]),
                "day": int(result["day"]),
                "session_file": result["session_file"],
                "n_trials": int(result["n_trials"]),
                "n_a": int(result["n_a"]),
                "n_b": int(result["n_b"]),
            }
        )
    pd.DataFrame(session_rows).to_csv(
        output_dir / "latent_category_encoding_sessions.csv",
        index=False,
    )

    progress_path.write_text(
        json.dumps(
            {
                "stage": "complete",
                "elapsed_sec": time.time() - t0,
                "n_sessions": int(len(results)),
                "n_subjects": int(len({r["subject"] for r in results})),
            },
            indent=2,
        )
    )
    print("[latent category] complete", flush=True)


if __name__ == "__main__":
    run_latent_category_encoding()
