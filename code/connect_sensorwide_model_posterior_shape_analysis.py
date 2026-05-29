#!/usr/bin/env python3
"""Bayesian temporal-shape comparison for connectivity model preferences."""

from __future__ import annotations

import math
import os
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from connect_sensorwide_analysis import OUTPUT_DIR

ACTIVE_PCT = float(os.environ.get("ACTIVE_PCT", "0.20"))
TAU_EFFECT = float(os.environ.get("TAU_EFFECT", "0.15"))
MIN_WIDTH_SEC = float(os.environ.get("MIN_WIDTH_SEC", "0.05"))
SHAPE_GRID_STEP_SEC = float(os.environ.get("SHAPE_GRID_STEP_SEC", "0.025"))
ONE_LB = (0.08, 0.58)
ONE_UB_MAX = 0.68
EARLY_LB = (0.10, 0.30)
EARLY_UB_MAX = 0.40
LATE_LB = (0.35, 0.58)
LATE_UB_MAX = 0.68


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing posterior shape input: {path}. "
            "Run connect_sensorwide_model_posterior_pairwise_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty posterior shape input: {path}")
    return d


def active_pct_suffix():
    if np.isclose(ACTIVE_PCT, 0.20):
        return ""
    pct_int = int(round(ACTIVE_PCT * 100.0))
    return f"_top{pct_int}"


def posterior_pairwise_subject_path(output_dir):
    suffix = active_pct_suffix()
    active_path = output_dir / (
        f"connect_sensorwide_model_posterior_pairwise_subject{suffix}.csv"
    )
    if active_path.exists():
        return active_path
    return output_dir / "connect_sensorwide_model_posterior_pairwise_subject.csv"


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def logsumexp(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return np.nan
    max_val = float(np.max(vals))
    total = 0.0
    for val in vals:
        total += math.exp(float(val) - max_val)
    return float(max_val + math.log(total))


def normalize_template(vals):
    vals = np.asarray(vals, dtype=float)
    norm = float(np.sqrt(np.sum(vals**2)))
    if norm <= np.finfo(float).eps:
        raise ValueError("Cannot normalize empty temporal template")
    return vals / norm


def candidate_boundary_times(times):
    rows = []
    last_time = None
    for time_val in times:
        time_val = float(time_val)
        if last_time is None:
            rows.append(time_val)
            last_time = time_val
            continue
        if time_val >= last_time + SHAPE_GRID_STEP_SEC:
            rows.append(time_val)
            last_time = time_val
    return rows


def candidate_intervals(times, lb_bounds, ub_max):
    rows = []
    boundary_times = candidate_boundary_times(times)
    for lb in boundary_times:
        if lb < lb_bounds[0] or lb > lb_bounds[1]:
            continue
        for ub in boundary_times:
            if ub < lb + MIN_WIDTH_SEC:
                continue
            if ub > ub_max:
                continue
            rows.append({"lb": float(lb), "ub": float(ub)})
    if len(rows) == 0:
        raise ValueError(f"No intervals for lb={lb_bounds}, ub_max={ub_max}")
    return rows


def interval_template(times, lb, ub):
    vals = []
    for time_val in times:
        if float(time_val) >= lb and float(time_val) <= ub:
            vals.append(1.0)
        else:
            vals.append(0.0)
    return normalize_template(vals)


def global_template(times):
    vals = []
    for _time_val in times:
        vals.append(1.0)
    return normalize_template(vals)


def log_bf_positive_mean(scores):
    vals = np.asarray(scores, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2:
        return np.nan, np.nan, np.nan, np.nan
    sigma = float(np.std(vals, ddof=1))
    if sigma <= np.finfo(float).eps:
        sigma = 1e-6
    n_obs = float(len(vals))
    ybar = float(np.mean(vals))
    var_post = 1.0 / (n_obs / sigma**2 + 1.0 / TAU_EFFECT**2)
    mean_post = var_post * n_obs * ybar / sigma**2
    sd_post = float(math.sqrt(var_post))
    p_gt0 = normal_cdf(mean_post / sd_post)
    log_bf = 0.5 * math.log(var_post / TAU_EFFECT**2)
    log_bf += 0.5 * mean_post**2 / var_post
    log_bf_positive = log_bf + math.log(max(p_gt0, 1e-300)) - math.log(0.5)
    return float(log_bf_positive), mean_post, sd_post, p_gt0


def contrast_matrix(subject_df):
    d = subject_df[np.isclose(subject_df["active_pct"].astype(float), ACTIVE_PCT)]
    if d.empty:
        raise ValueError(f"Missing subject pairwise rows for active_pct={ACTIVE_PCT}")
    contrasts = []
    for contrast in d["contrast"].drop_duplicates():
        g = d[d["contrast"] == contrast]
        if "plot_default" in g.columns and not bool(g["plot_default"].iloc[0]):
            continue
        contrasts.append(str(contrast))
    if len(contrasts) == 0:
        raise ValueError("No default contrasts found for posterior shape analysis")
    return d, contrasts


def subject_time_matrix(d, contrast):
    g = d[d["contrast"] == contrast].copy()
    wide = g.pivot_table(
        index="subject",
        columns="time_center_sec",
        values="diff",
        aggfunc="mean",
    )
    wide = wide.dropna(axis=0, how="any")
    if wide.empty:
        raise ValueError(f"No complete subject timecourses for contrast={contrast}")
    times = np.asarray(wide.columns, dtype=float)
    vals = wide.to_numpy(dtype=float)
    return times, vals, wide.index.to_numpy(dtype=int)


def projection_scores(Y, template):
    scores = []
    for row in Y:
        scores.append(float(np.dot(row, template)))
    return np.asarray(scores, dtype=float)


def score_template(Y, template):
    scores = projection_scores(Y, template)
    log_bf, mean_post, sd_post, p_gt0 = log_bf_positive_mean(scores)
    return {
        "log_bf": log_bf,
        "effect_mean": mean_post,
        "effect_sd": sd_post,
        "effect_p_gt0": p_gt0,
    }


def global_result(Y, times):
    template = global_template(times)
    result = score_template(Y, template)
    result["shape_model"] = "global"
    result["lb_one"] = np.nan
    result["ub_one"] = np.nan
    result["lb_early"] = np.nan
    result["ub_early"] = np.nan
    result["lb_late"] = np.nan
    result["ub_late"] = np.nan
    return result


def one_window_results(Y, times):
    rows = []
    intervals = candidate_intervals(times, ONE_LB, ONE_UB_MAX)
    for interval in intervals:
        template = interval_template(times, interval["lb"], interval["ub"])
        row = score_template(Y, template)
        row["shape_model"] = "one_window"
        row["lb_one"] = interval["lb"]
        row["ub_one"] = interval["ub"]
        row["lb_early"] = np.nan
        row["ub_early"] = np.nan
        row["lb_late"] = np.nan
        row["ub_late"] = np.nan
        rows.append(row)
    return rows


def interval_score_rows(Y, times, intervals, prefix):
    rows = []
    for interval in intervals:
        template = interval_template(times, interval["lb"], interval["ub"])
        score = score_template(Y, template)
        row = {
            f"lb_{prefix}": interval["lb"],
            f"ub_{prefix}": interval["ub"],
            "log_bf": score["log_bf"],
            "effect_mean": score["effect_mean"],
            "effect_p_gt0": score["effect_p_gt0"],
        }
        rows.append(row)
    return rows


def two_window_results(Y, times):
    rows = []
    early_intervals = candidate_intervals(times, EARLY_LB, EARLY_UB_MAX)
    late_intervals = candidate_intervals(times, LATE_LB, LATE_UB_MAX)
    early_rows = interval_score_rows(Y, times, early_intervals, "early")
    late_rows = interval_score_rows(Y, times, late_intervals, "late")
    for early in early_rows:
        for late in late_rows:
            row = {
                "shape_model": "two_window",
                "lb_one": np.nan,
                "ub_one": np.nan,
                "lb_early": early["lb_early"],
                "ub_early": early["ub_early"],
                "lb_late": late["lb_late"],
                "ub_late": late["ub_late"],
                "log_bf": early["log_bf"] + late["log_bf"],
                "effect_mean": np.nan,
                "effect_sd": np.nan,
                "effect_p_gt0": np.nan,
                "effect_early_mean": early["effect_mean"],
                "effect_early_p_gt0": early["effect_p_gt0"],
                "effect_late_mean": late["effect_mean"],
                "effect_late_p_gt0": late["effect_p_gt0"],
            }
            rows.append(row)
    return rows


def add_model_posterior(rows):
    grouped = {}
    for row in rows:
        model = row["shape_model"]
        if model not in grouped:
            grouped[model] = []
        grouped[model].append(float(row["log_bf"]))
    model_log = {"none": 0.0}
    for model, vals in grouped.items():
        model_log[model] = logsumexp(vals) - math.log(float(len(vals)))
    denom_vals = []
    for val in model_log.values():
        denom_vals.append(float(val))
    denom = logsumexp(denom_vals)
    for row in rows:
        model = row["shape_model"]
        row["model_log_bf"] = model_log[model]
        row["posterior_model_prob"] = math.exp(model_log[model] - denom)
        row["posterior_candidate_prob"] = math.exp(
            float(row["log_bf"]) - logsumexp(grouped[model])
        )
    none_row = {
        "shape_model": "none",
        "log_bf": 0.0,
        "model_log_bf": 0.0,
        "posterior_model_prob": math.exp(0.0 - denom),
        "posterior_candidate_prob": 1.0,
        "lb_one": np.nan,
        "ub_one": np.nan,
        "lb_early": np.nan,
        "ub_early": np.nan,
        "lb_late": np.nan,
        "ub_late": np.nan,
    }
    rows.append(none_row)
    return rows


def summarize_models(rows, contrast, n_subjects):
    summary_rows = []
    seen = []
    for row in rows:
        model = row["shape_model"]
        if model in seen:
            continue
        seen.append(model)
        d_model = []
        for candidate in rows:
            if candidate["shape_model"] == model:
                d_model.append(candidate)
        best = d_model[0]
        for candidate in d_model:
            if float(candidate["posterior_candidate_prob"]) > float(
                best["posterior_candidate_prob"]
            ):
                best = candidate
        summary_rows.append(
            {
                "contrast": contrast,
                "active_pct": ACTIVE_PCT,
                "shape_model": model,
                "posterior_model_prob": float(best["posterior_model_prob"]),
                "model_log_bf": float(best["model_log_bf"]),
                "best_candidate_prob": float(best["posterior_candidate_prob"]),
                "lb_one": best.get("lb_one", np.nan),
                "ub_one": best.get("ub_one", np.nan),
                "lb_early": best.get("lb_early", np.nan),
                "ub_early": best.get("ub_early", np.nan),
                "lb_late": best.get("lb_late", np.nan),
                "ub_late": best.get("ub_late", np.nan),
                "n_subjects": int(n_subjects),
            }
        )
    return summary_rows


def run_connect_sensorwide_model_posterior_shape(
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    subject_df = require_csv(posterior_pairwise_subject_path(output_dir))
    d, contrasts = contrast_matrix(subject_df)
    candidate_rows = []
    summary_rows = []
    for contrast_i, contrast in enumerate(contrasts, start=1):
        print(
            f"[connect posterior shape] contrast {contrast_i}/{len(contrasts)}: "
            f"{contrast}",
            flush=True,
        )
        times, Y, subjects = subject_time_matrix(d, contrast)
        one_count = len(candidate_intervals(times, ONE_LB, ONE_UB_MAX))
        early_count = len(candidate_intervals(times, EARLY_LB, EARLY_UB_MAX))
        late_count = len(candidate_intervals(times, LATE_LB, LATE_UB_MAX))
        two_count = early_count * late_count
        print(
            "[connect posterior shape] "
            f"subjects={len(subjects)}, times={len(times)}, "
            f"one={one_count}, early={early_count}, late={late_count}, "
            f"two={two_count}",
            flush=True,
        )
        rows = []
        rows.append(global_result(Y, times))
        print("[connect posterior shape] scored global", flush=True)
        for row in one_window_results(Y, times):
            rows.append(row)
        print("[connect posterior shape] scored one-window candidates", flush=True)
        for row in two_window_results(Y, times):
            rows.append(row)
        print("[connect posterior shape] scored two-window candidates", flush=True)
        rows = add_model_posterior(rows)
        print("[connect posterior shape] computed model posterior", flush=True)
        for row in rows:
            row["contrast"] = contrast
            row["active_pct"] = ACTIVE_PCT
            row["n_subjects"] = int(len(subjects))
            candidate_rows.append(row)
        for row in summarize_models(rows, contrast, len(subjects)):
            summary_rows.append(row)

    suffix = active_pct_suffix()
    candidate_path = output_dir / (
        f"connect_sensorwide_model_posterior_shape_candidates{suffix}.csv"
    )
    summary_path = output_dir / (
        f"connect_sensorwide_model_posterior_shape_summary{suffix}.csv"
    )
    pd.DataFrame(candidate_rows).to_csv(candidate_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"[connect posterior shape] wrote {candidate_path}", flush=True)
    print(f"[connect posterior shape] wrote {summary_path}", flush=True)
    return {"candidates": candidate_path, "summary": summary_path}


if __name__ == "__main__":
    run_connect_sensorwide_model_posterior_shape()
