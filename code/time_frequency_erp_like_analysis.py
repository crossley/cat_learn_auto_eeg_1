#!/usr/bin/env python3
"""ERP-like condition averages over time-frequency sequence features."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from analysis_utils import parallel_collect
from sensor_rois import STRICT_SENSOR_ROIS
from sequence_feature_interface import (
    OUTPUT_DIR,
    load_feature_sequence,
    load_sequence_sessions,
)

PREFIX = "time_frequency_erp_like"
N_JOBS = 4
DEFAULT_FREQS = np.array([4, 6, 8, 10, 12, 16, 20, 30], dtype=float)
DEFAULT_BASELINE = (None, 0.0)
DEFAULT_BASELINE_MODE = "logratio"
DEFAULT_RESAMPLE_HZ = 128.0

FREQUENCY_BANDS = {
    "theta": (4.0, 7.0),
    "alpha": (8.0, 12.0),
    "beta": (16.0, 30.0),
}

TIME_WINDOWS = {
    "early": (0.0, 0.2),
    "middle": (0.2, 0.5),
    "late": (0.5, 0.8),
}


def _stage_for_day(day: int) -> str:
    if int(day) <= 1:
        return "early_practice"
    if int(day) >= 4:
        return "late_practice"
    return "mid_practice"


def _parse_feature_names(feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for index, name in enumerate(feature_names):
        if "_" not in name or not name.endswith("Hz"):
            raise ValueError(f"Unexpected time-frequency feature name: {name}")
        channel, freq_text = name.rsplit("_", 1)
        rows.append(
            {
                "feature_index": int(index),
                "channel": channel,
                "freq_hz": float(freq_text[:-2]),
            }
        )
    return pd.DataFrame(rows)


def _sem(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) <= 1:
        return np.nan
    return float(np.nanstd(values, ddof=1) / np.sqrt(len(values)))


def _ttest_1samp_summary(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    out = {"n": int(len(values)), "mean": np.nan, "sem": np.nan, "t": np.nan, "p_value": np.nan}
    if len(values) == 0:
        return out
    out["mean"] = float(np.nanmean(values))
    out["sem"] = _sem(values)
    if len(values) > 1:
        sd = float(np.nanstd(values, ddof=1))
        if np.isfinite(sd) and sd > np.finfo(float).eps:
            out["t"] = float(out["mean"] / (sd / np.sqrt(len(values))))
            try:
                from scipy import stats

                out["p_value"] = float(stats.ttest_1samp(values, 0.0, nan_policy="omit").pvalue)
            except Exception:
                out["p_value"] = np.nan
    return out


def _roi_specs(channels: list[str]) -> dict[str, list[str]]:
    channel_set = set(channels)
    specs = {"sensorwide": sorted(channel_set)}
    for roi, roi_channels in STRICT_SENSOR_ROIS.items():
        present = sorted(channel_set & set(roi_channels))
        if present:
            specs[roi] = present
    return specs


def _feature_cube(dataset) -> tuple[np.ndarray, pd.DataFrame, list[str], np.ndarray]:
    feature_df = _parse_feature_names(dataset.feature_names)
    channels = feature_df["channel"].drop_duplicates().tolist()
    freqs = np.sort(feature_df["freq_hz"].unique().astype(float))
    expected = pd.MultiIndex.from_product([channels, freqs], names=["channel", "freq_hz"])
    observed = pd.MultiIndex.from_frame(feature_df[["channel", "freq_hz"]])
    if not observed.equals(expected):
        raise ValueError("Time-frequency features are not ordered as channel-frequency blocks")
    cube = dataset.X.reshape(dataset.X.shape[0], dataset.X.shape[1], len(channels), len(freqs))
    return cube, feature_df, channels, freqs


def _condition_rows(dataset, cube: np.ndarray, channels: list[str], freqs: np.ndarray) -> pd.DataFrame:
    rows = []
    for label, code in [("cat_a", 0), ("cat_b", 1)]:
        idx = np.where(dataset.y == code)[0]
        if len(idx) == 0:
            continue
        mean_cube = np.nanmean(cube[idx], axis=0)
        for ci, channel in enumerate(channels):
            for fi, freq in enumerate(freqs):
                rows.append(
                    pd.DataFrame(
                        {
                            "subject": int(dataset.subject),
                            "day": int(dataset.session),
                            "practice_stage": _stage_for_day(dataset.session),
                            "condition": label,
                            "channel": channel,
                            "freq_hz": float(freq),
                            "time_s": dataset.time.astype(float),
                            "power": mean_cube[:, ci, fi],
                            "n_trials": int(len(idx)),
                            "session_file": dataset.session_file,
                        }
                    )
                )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _window_summary_rows(
    dataset,
    cube: np.ndarray,
    channels: list[str],
    freqs: np.ndarray,
) -> pd.DataFrame:
    roi_map = _roi_specs(channels)
    channel_index = {ch: i for i, ch in enumerate(channels)}
    rows = []
    for condition, code in [("cat_a", 0), ("cat_b", 1)]:
        trial_idx = np.where(dataset.y == code)[0]
        if len(trial_idx) == 0:
            continue
        for band, (f0, f1) in FREQUENCY_BANDS.items():
            freq_idx = np.where((freqs >= f0) & (freqs <= f1))[0]
            if len(freq_idx) == 0:
                continue
            for window, (t0, t1) in TIME_WINDOWS.items():
                time_idx = np.where((dataset.time >= t0) & (dataset.time <= t1))[0]
                if len(time_idx) == 0:
                    continue
                for electrode in channels:
                    ch_idx = np.array([channel_index[electrode]], dtype=int)
                    values = cube[np.ix_(trial_idx, time_idx, ch_idx, freq_idx)]
                    rows.append(
                        {
                            "subject": int(dataset.subject),
                            "day": int(dataset.session),
                            "practice_stage": _stage_for_day(dataset.session),
                            "summary_level": "electrode",
                            "roi": electrode,
                            "condition": condition,
                            "band": band,
                            "window": window,
                            "power_mean": float(np.nanmean(values)),
                            "n_trials": int(len(trial_idx)),
                            "n_channels": 1,
                            "n_freqs": int(len(freq_idx)),
                            "time_start_s": float(t0),
                            "time_end_s": float(t1),
                        }
                    )
                for roi, roi_channels in roi_map.items():
                    ch_idx = np.array([channel_index[ch] for ch in roi_channels], dtype=int)
                    values = cube[np.ix_(trial_idx, time_idx, ch_idx, freq_idx)]
                    rows.append(
                        {
                            "subject": int(dataset.subject),
                            "day": int(dataset.session),
                            "practice_stage": _stage_for_day(dataset.session),
                            "summary_level": "strict_roi",
                            "roi": roi,
                            "condition": condition,
                            "band": band,
                            "window": window,
                            "power_mean": float(np.nanmean(values)),
                            "n_trials": int(len(trial_idx)),
                            "n_channels": int(len(ch_idx)),
                            "n_freqs": int(len(freq_idx)),
                            "time_start_s": float(t0),
                            "time_end_s": float(t1),
                        }
                    )
    return pd.DataFrame(rows)


def _timecourse_rows(dataset, cube: np.ndarray, channels: list[str], freqs: np.ndarray) -> pd.DataFrame:
    roi_map = _roi_specs(channels)
    channel_index = {ch: i for i, ch in enumerate(channels)}
    rows = []
    for condition, code in [("cat_a", 0), ("cat_b", 1)]:
        trial_idx = np.where(dataset.y == code)[0]
        if len(trial_idx) == 0:
            continue
        for band, (f0, f1) in FREQUENCY_BANDS.items():
            freq_idx = np.where((freqs >= f0) & (freqs <= f1))[0]
            if len(freq_idx) == 0:
                continue
            for roi, roi_channels in roi_map.items():
                ch_idx = np.array([channel_index[ch] for ch in roi_channels], dtype=int)
                values = np.nanmean(cube[np.ix_(trial_idx, np.arange(len(dataset.time)), ch_idx, freq_idx)], axis=(0, 2, 3))
                rows.append(
                    pd.DataFrame(
                        {
                            "subject": int(dataset.subject),
                            "day": int(dataset.session),
                            "practice_stage": _stage_for_day(dataset.session),
                            "condition": condition,
                            "roi": roi,
                            "band": band,
                            "time_s": dataset.time.astype(float),
                            "power_mean": values,
                            "n_trials": int(len(trial_idx)),
                            "n_channels": int(len(ch_idx)),
                            "n_freqs": int(len(freq_idx)),
                        }
                    )
                )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def compute_session_tables(dataset) -> dict[str, pd.DataFrame]:
    cube, _feature_df, channels, freqs = _feature_cube(dataset)
    condition_df = _condition_rows(dataset, cube, channels, freqs)
    window_df = _window_summary_rows(dataset, cube, channels, freqs)
    timecourse_df = _timecourse_rows(dataset, cube, channels, freqs)
    return {
        "condition": condition_df,
        "window": window_df,
        "timecourse": timecourse_df,
    }


def _condition_contrasts(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    keys = [
        "subject",
        "day",
        "practice_stage",
        "summary_level",
        "roi",
        "band",
        "window",
        "time_start_s",
        "time_end_s",
    ]
    wide = summary_df.pivot_table(index=keys, columns="condition", values="power_mean").reset_index()
    if "cat_a" not in wide or "cat_b" not in wide:
        return pd.DataFrame()
    wide["contrast"] = "cat_a_minus_cat_b"
    wide["power_diff"] = wide["cat_a"] - wide["cat_b"]
    return wide.rename_axis(None, axis=1)


def _day_mean(df: pd.DataFrame, value_cols: list[str], group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows = []
    for key, group in df.groupby(group_cols, dropna=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(group_cols, key))
        row["n_subjects"] = int(group["subject"].nunique()) if "subject" in group else int(len(group))
        for col in value_cols:
            vals = group[col].to_numpy(dtype=float)
            row[f"{col}_mean"] = float(np.nanmean(vals))
            row[f"{col}_sem"] = _sem(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def _day_effects(contrast_df: pd.DataFrame) -> pd.DataFrame:
    if contrast_df.empty:
        return pd.DataFrame()
    keys = ["summary_level", "roi", "band", "window"]
    rows = []
    strict = contrast_df[contrast_df["summary_level"] == "strict_roi"].copy()
    for key, group in strict.groupby(keys):
        wide = group.pivot_table(index="subject", columns="day", values="power_diff")
        if wide.empty:
            continue
        day_values = sorted([int(c) for c in wide.columns])
        for day in day_values:
            summary = _ttest_1samp_summary(wide[day].to_numpy(dtype=float))
            rows.append(
                dict(zip(keys, key))
                | {
                    "effect": "cat_a_minus_cat_b",
                    "day": int(day),
                    "comparison_day": np.nan,
                    "n_subjects": summary["n"],
                    "mean": summary["mean"],
                    "sem": summary["sem"],
                    "t": summary["t"],
                    "p_value": summary["p_value"],
                }
            )
        if 1 in wide.columns:
            for day in day_values:
                if day == 1:
                    continue
                delta = (wide[day] - wide[1]).to_numpy(dtype=float)
                summary = _ttest_1samp_summary(delta)
                rows.append(
                    dict(zip(keys, key))
                    | {
                        "effect": "day_change_from_day_1",
                        "day": int(day),
                        "comparison_day": 1,
                        "n_subjects": summary["n"],
                        "mean": summary["mean"],
                        "sem": summary["sem"],
                        "t": summary["t"],
                        "p_value": summary["p_value"],
                    }
                )
    return pd.DataFrame(rows)


def _session_worker(payload: dict) -> dict:
    subject = payload["subject"]
    day = payload["day"]
    try:
        dataset = load_feature_sequence(
            payload,
            feature_kind="time_frequency",
            use_cache=not payload["no_feature_cache"],
            force_recompute=payload["force_feature_recompute"],
            verbose=payload["feature_cache_verbose"],
            freqs=np.asarray(payload["freqs"], dtype=float),
            resample_hz=payload["resample_hz"],
            baseline=payload["baseline"],
            mode=payload["baseline_mode"],
            picks=payload["picks"],
        )
        tables = compute_session_tables(dataset)
        qc = {
            "subject": int(subject),
            "day": int(day),
            "stage": "time_frequency",
            "ok": True,
            "reason": "",
            "detail": "",
            "n_trials": int(dataset.X.shape[0]),
            "n_times": int(dataset.X.shape[1]),
            "n_features": int(dataset.X.shape[2]),
        }
        return {"ok": True, "subject": subject, "day": day, "tables": tables, "qc": qc}
    except Exception as exc:
        return {
            "ok": False,
            "subject": subject,
            "day": day,
            "tables": {},
            "qc": {
                "subject": int(subject),
                "day": int(day),
                "stage": "time_frequency",
                "ok": False,
                "reason": "extract_error",
                "detail": str(exc),
                "n_trials": 0,
                "n_times": 0,
                "n_features": 0,
            },
        }


def _concat_results(results: list[dict], name: str) -> pd.DataFrame:
    frames = [r["tables"][name] for r in results if r.get("ok") and name in r["tables"] and not r["tables"][name].empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def run_time_frequency_erp_like(
    output_dir: Path | str = OUTPUT_DIR,
    n_workers: int | None = None,
    max_sessions: int | None = None,
    smoke: bool = False,
    no_feature_cache: bool = False,
    force_feature_recompute: bool = False,
    feature_cache_verbose: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    freqs = DEFAULT_FREQS.copy()
    resample_hz = DEFAULT_RESAMPLE_HZ
    picks = None
    if smoke:
        freqs = np.array([8.0, 12.0], dtype=float)
        resample_hz = 64.0
        picks = sorted(set(STRICT_SENSOR_ROIS["visual"] + STRICT_SENSOR_ROIS["central"]))
        if max_sessions is None:
            max_sessions = 1
        n_workers = 1

    sessions = load_sequence_sessions(load_epochs=False)
    if max_sessions is not None:
        sessions = sessions[: int(max_sessions)]
    if not sessions:
        raise RuntimeError("No sessions available for time-frequency ERP-like analysis")

    payloads = []
    for item in sessions:
        payload = dict(item)
        payload["freqs"] = freqs
        payload["resample_hz"] = resample_hz
        payload["baseline"] = DEFAULT_BASELINE
        payload["baseline_mode"] = DEFAULT_BASELINE_MODE
        payload["picks"] = picks
        payload["no_feature_cache"] = bool(no_feature_cache)
        payload["force_feature_recompute"] = bool(force_feature_recompute)
        payload["feature_cache_verbose"] = bool(feature_cache_verbose)
        payloads.append(payload)

    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    print(f"[{PREFIX}] running {len(payloads)} sessions with {n_workers} worker(s)", flush=True)
    if n_workers == 1:
        results = []
        for done, payload in enumerate(payloads, start=1):
            results.append(_session_worker(payload))
            if done == len(payloads) or done % 5 == 0:
                print(f"[{PREFIX}] completed {done}/{len(payloads)} sessions", flush=True)
    else:
        print(f"[{PREFIX}] parallel batch running; cache is shared via output/", flush=True)
        results = parallel_collect(_session_worker, payloads, n_workers=n_workers)
        print(f"[{PREFIX}] completed {len(payloads)}/{len(payloads)} sessions", flush=True)

    qc_df = pd.DataFrame([r["qc"] for r in results])
    condition_df = _concat_results(results, "condition")
    window_df = _concat_results(results, "window")
    timecourse_df = _concat_results(results, "timecourse")
    if condition_df.empty or window_df.empty:
        raise RuntimeError("Time-frequency ERP-like analysis produced no valid rows")

    contrast_df = _condition_contrasts(window_df)
    day_mean_df = _day_mean(
        window_df,
        ["power_mean"],
        ["day", "practice_stage", "summary_level", "roi", "condition", "band", "window"],
    )
    contrast_day_mean_df = _day_mean(
        contrast_df,
        ["power_diff"],
        ["day", "practice_stage", "summary_level", "roi", "band", "window"],
    )
    stage_mean_df = _day_mean(
        window_df,
        ["power_mean"],
        ["practice_stage", "summary_level", "roi", "condition", "band", "window"],
    )
    contrast_stage_mean_df = _day_mean(
        contrast_df,
        ["power_diff"],
        ["practice_stage", "summary_level", "roi", "band", "window"],
    )
    timecourse_day_mean_df = _day_mean(
        timecourse_df,
        ["power_mean"],
        ["day", "practice_stage", "condition", "roi", "band", "time_s"],
    )
    condition_day_mean_df = _day_mean(
        condition_df,
        ["power"],
        ["day", "practice_stage", "condition", "channel", "freq_hz", "time_s"],
    )
    day_effect_df = _day_effects(contrast_df)

    paths = {
        "session_condition_csv": output_dir / f"{PREFIX}_session_condition_channel_freq_time.csv",
        "day_condition_csv": output_dir / f"{PREFIX}_day_condition_channel_freq_time.csv",
        "session_window_csv": output_dir / f"{PREFIX}_subject_day_window_band_roi_condition.csv",
        "session_contrast_csv": output_dir / f"{PREFIX}_subject_day_window_band_roi_contrast.csv",
        "day_window_csv": output_dir / f"{PREFIX}_day_window_band_roi_condition.csv",
        "day_contrast_csv": output_dir / f"{PREFIX}_day_window_band_roi_contrast.csv",
        "stage_window_csv": output_dir / f"{PREFIX}_practice_stage_window_band_roi_condition.csv",
        "stage_contrast_csv": output_dir / f"{PREFIX}_practice_stage_window_band_roi_contrast.csv",
        "timecourse_csv": output_dir / f"{PREFIX}_subject_day_band_roi_timecourse.csv",
        "timecourse_day_csv": output_dir / f"{PREFIX}_day_band_roi_timecourse.csv",
        "day_effect_csv": output_dir / f"{PREFIX}_day_effect_summary.csv",
        "qc_csv": output_dir / f"{PREFIX}_qc_log.csv",
        "metadata_json": output_dir / f"{PREFIX}_metadata.json",
    }
    condition_df.to_csv(paths["session_condition_csv"], index=False)
    condition_day_mean_df.to_csv(paths["day_condition_csv"], index=False)
    window_df.to_csv(paths["session_window_csv"], index=False)
    contrast_df.to_csv(paths["session_contrast_csv"], index=False)
    day_mean_df.to_csv(paths["day_window_csv"], index=False)
    contrast_day_mean_df.to_csv(paths["day_contrast_csv"], index=False)
    stage_mean_df.to_csv(paths["stage_window_csv"], index=False)
    contrast_stage_mean_df.to_csv(paths["stage_contrast_csv"], index=False)
    timecourse_df.to_csv(paths["timecourse_csv"], index=False)
    timecourse_day_mean_df.to_csv(paths["timecourse_day_csv"], index=False)
    day_effect_df.to_csv(paths["day_effect_csv"], index=False)
    qc_df.to_csv(paths["qc_csv"], index=False)

    metadata = {
        "analysis": PREFIX,
        "feature_interface": "sequence_feature_interface.load_time_frequency_sequence",
        "feature_shape": "trials x time x channel-frequency features",
        "baseline": {
            "window_s": [DEFAULT_BASELINE[0], DEFAULT_BASELINE[1]],
            "mode": DEFAULT_BASELINE_MODE,
            "note": "Power was baseline-corrected inside the shared sequence feature interface before aggregation.",
        },
        "freqs_hz": [float(f) for f in freqs],
        "frequency_bands_hz": {k: [float(v[0]), float(v[1])] for k, v in FREQUENCY_BANDS.items()},
        "time_windows_s": {k: [float(v[0]), float(v[1])] for k, v in TIME_WINDOWS.items()},
        "practice_stage_definition": {
            "early_practice": "day <= 1",
            "mid_practice": "2 <= day <= 3",
            "late_practice": "day >= 4",
        },
        "strict_sensor_rois": STRICT_SENSOR_ROIS,
        "smoke": bool(smoke),
        "max_sessions": None if max_sessions is None else int(max_sessions),
        "n_sessions_requested": int(len(payloads)),
        "n_sessions_ok": int(qc_df["ok"].sum()) if "ok" in qc_df else 0,
        "elapsed_sec": float(time.time() - t0),
    }
    paths["metadata_json"].write_text(json.dumps(metadata, indent=2))
    for path in paths.values():
        print(f"[{PREFIX}] wrote {path}", flush=True)
    return paths


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Run a small single-worker validation path.")
    parser.add_argument("--max-sessions", type=int, default=None, help="Limit the number of matched sessions.")
    parser.add_argument("--n-workers", type=int, default=None, help="Parallel workers for full analysis.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-feature-cache", action="store_true")
    parser.add_argument("--force-feature-recompute", action="store_true")
    parser.add_argument("--feature-cache-verbose", action="store_true")
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_time_frequency_erp_like(
        output_dir=args.output_dir,
        n_workers=args.n_workers,
        max_sessions=args.max_sessions,
        smoke=args.smoke,
        no_feature_cache=args.no_feature_cache,
        force_feature_recompute=args.force_feature_recompute,
        feature_cache_verbose=args.feature_cache_verbose,
    )
