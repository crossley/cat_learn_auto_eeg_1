#!/usr/bin/env python3
"""Bayesian interval models for D1-split connectivity evidence."""

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
EARLY_LB = (0.10, 0.25)
EARLY_UB_MAX = 0.40
LATE_LB = (0.35, 0.55)
LATE_UB_MAX = 0.70


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing D1-split Bayesian interval input: {path}. "
            "Run connect_sensorwide_model_timecourse_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty D1-split Bayesian interval input: {path}")
    return d


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


def filter_active_pct(d):
    if "active_pct" not in d.columns:
        return d.copy()
    g = d[np.isclose(d["active_pct"].astype(float), ACTIVE_PCT)].copy()
    if g.empty:
        raise ValueError(f"Missing rows for active_pct={ACTIVE_PCT}")
    return g


def compute_d1_contrast(score_df):
    d = filter_active_pct(score_df)
    rows = []
    group_cols = ["subject", "lock_time", "time_center_sec"]
    for key, g in d.groupby(group_cols):
        subject, lock_time, time_center = key
        d1_vals = []
        competitor_vals = []
        for row in g.itertuples(index=False):
            model = str(row.model)
            split_day = row.split_day
            is_d1 = False
            if model in ["two_stage_binary", "two_stage_hybrid"]:
                if np.isfinite(split_day) and int(split_day) == 1:
                    is_d1 = True
            if is_d1:
                d1_vals.append(float(row.rho))
            else:
                competitor_vals.append(float(row.rho))
        if len(d1_vals) == 0 or len(competitor_vals) == 0:
            continue
        d1_best = float(np.nanmax(np.asarray(d1_vals, dtype=float)))
        competitor_best = float(np.nanmax(np.asarray(competitor_vals, dtype=float)))
        rows.append(
            {
                "subject": int(subject),
                "lock_time": float(lock_time),
                "time_center_sec": float(time_center),
                "active_pct": ACTIVE_PCT,
                "d1_best_rho": d1_best,
                "competitor_best_rho": competitor_best,
                "contrast": d1_best - competitor_best,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No D1-split contrast rows were produced")
    return out


def candidate_intervals(times, lb_bounds, ub_max):
    times = np.asarray(times, dtype=float)
    rows = []
    for lb in times:
        if lb < lb_bounds[0] or lb > lb_bounds[1]:
            continue
        for ub in times:
            if ub < lb + MIN_WIDTH_SEC:
                continue
            if ub > ub_max:
                continue
            rows.append({"lb": float(lb), "ub": float(ub)})
    if len(rows) == 0:
        raise ValueError(
            "No candidate intervals found: "
            f"lb={lb_bounds}, ub_max={ub_max}, min_width={MIN_WIDTH_SEC}"
        )
    return rows


def design_column(times, lb, ub):
    vals = []
    for time_val in times:
        if float(time_val) >= lb and float(time_val) <= ub:
            vals.append(1.0)
        else:
            vals.append(0.0)
    return np.asarray(vals, dtype=float)


def estimate_sigma(y):
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 3:
        raise ValueError("Too few observations to estimate sigma")
    sigma = float(np.std(y, ddof=1))
    if sigma <= np.finfo(float).eps:
        raise ValueError("Cannot estimate sigma from flat contrast data")
    return sigma


def log_marginal_likelihood(y, X, sigma, tau):
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    n_obs = len(y)
    if X.ndim == 1:
        X = X.reshape(n_obs, 1)
    n_coef = X.shape[1]
    base = -0.5 * n_obs * math.log(2.0 * math.pi * sigma**2)
    base -= float(np.dot(y, y)) / (2.0 * sigma**2)
    if n_coef == 0:
        return base, np.zeros(0), np.zeros((0, 0))
    xtx = X.T @ X
    xty = X.T @ y
    precision = xtx / sigma**2
    for idx in range(n_coef):
        precision[idx, idx] += 1.0 / tau**2
    sign, logdet = np.linalg.slogdet(precision)
    if sign <= 0:
        raise ValueError("Non-positive Bayesian precision determinant")
    b = xty / sigma**2
    cov = np.linalg.inv(precision)
    mean = cov @ b
    log_ev = base
    log_ev -= n_coef * math.log(tau)
    log_ev -= 0.5 * float(logdet)
    log_ev += 0.5 * float(b.T @ cov @ b)
    return float(log_ev), mean, cov


def model0_result(y, sigma):
    log_ev, _mean, _cov = log_marginal_likelihood(
        y,
        np.zeros((len(y), 0), dtype=float),
        sigma,
        TAU_EFFECT,
    )
    return {
        "model": "none",
        "log_evidence": log_ev,
        "n_interval_candidates": 1,
    }


def one_interval_results(y, times, sigma, intervals, model_name):
    rows = []
    for interval in intervals:
        x = design_column(times, interval["lb"], interval["ub"])
        log_ev, mean, cov = log_marginal_likelihood(y, x, sigma, TAU_EFFECT)
        sd = float(math.sqrt(cov[0, 0]))
        p_pos = normal_cdf(float(mean[0]) / sd)
        rows.append(
            {
                "model": model_name,
                "lb_early": interval["lb"] if model_name == "early" else np.nan,
                "ub_early": interval["ub"] if model_name == "early" else np.nan,
                "lb_late": interval["lb"] if model_name == "late" else np.nan,
                "ub_late": interval["ub"] if model_name == "late" else np.nan,
                "log_evidence": log_ev,
                "effect_early_mean": float(mean[0])
                if model_name == "early"
                else np.nan,
                "effect_early_sd": sd if model_name == "early" else np.nan,
                "effect_early_p_gt0": p_pos if model_name == "early" else np.nan,
                "effect_late_mean": float(mean[0])
                if model_name == "late"
                else np.nan,
                "effect_late_sd": sd if model_name == "late" else np.nan,
                "effect_late_p_gt0": p_pos if model_name == "late" else np.nan,
            }
        )
    return rows


def two_interval_results(y, times, sigma, early_intervals, late_intervals):
    rows = []
    for early in early_intervals:
        x_early = design_column(times, early["lb"], early["ub"])
        for late in late_intervals:
            x_late = design_column(times, late["lb"], late["ub"])
            X = np.column_stack([x_early, x_late])
            log_ev, mean, cov = log_marginal_likelihood(y, X, sigma, TAU_EFFECT)
            sd_early = float(math.sqrt(cov[0, 0]))
            sd_late = float(math.sqrt(cov[1, 1]))
            rows.append(
                {
                    "model": "early_late",
                    "lb_early": early["lb"],
                    "ub_early": early["ub"],
                    "lb_late": late["lb"],
                    "ub_late": late["ub"],
                    "log_evidence": log_ev,
                    "effect_early_mean": float(mean[0]),
                    "effect_early_sd": sd_early,
                    "effect_early_p_gt0": normal_cdf(float(mean[0]) / sd_early),
                    "effect_late_mean": float(mean[1]),
                    "effect_late_sd": sd_late,
                    "effect_late_p_gt0": normal_cdf(float(mean[1]) / sd_late),
                }
            )
    return rows


def posterior_weights(rows):
    log_vals = []
    for row in rows:
        log_vals.append(float(row["log_evidence"]))
    denom = logsumexp(log_vals)
    out = []
    for row in rows:
        new_row = dict(row)
        new_row["posterior_interval_prob"] = math.exp(
            float(row["log_evidence"]) - denom
        )
        out.append(new_row)
    return out, denom


def weighted_mean(rows, col):
    total = 0.0
    weight_total = 0.0
    for row in rows:
        val = row.get(col, np.nan)
        weight = row.get("posterior_interval_prob", np.nan)
        if np.isfinite(val) and np.isfinite(weight):
            total += float(weight) * float(val)
            weight_total += float(weight)
    if weight_total <= np.finfo(float).eps:
        return np.nan
    return float(total / weight_total)


def summarize_model(model_rows, model_log_evidence):
    if len(model_rows) == 0:
        return {}
    first_model = model_rows[0]["model"]
    return {
        "model": first_model,
        "log_evidence": model_log_evidence,
        "n_interval_candidates": int(len(model_rows)),
        "lb_early_mean": weighted_mean(model_rows, "lb_early"),
        "ub_early_mean": weighted_mean(model_rows, "ub_early"),
        "lb_late_mean": weighted_mean(model_rows, "lb_late"),
        "ub_late_mean": weighted_mean(model_rows, "ub_late"),
        "effect_early_mean": weighted_mean(model_rows, "effect_early_mean"),
        "effect_early_sd": weighted_mean(model_rows, "effect_early_sd"),
        "effect_early_p_gt0": weighted_mean(model_rows, "effect_early_p_gt0"),
        "effect_late_mean": weighted_mean(model_rows, "effect_late_mean"),
        "effect_late_sd": weighted_mean(model_rows, "effect_late_sd"),
        "effect_late_p_gt0": weighted_mean(model_rows, "effect_late_p_gt0"),
    }


def group_contrast_summary(contrast_df):
    rows = []
    for time_center, g in contrast_df.groupby("time_center_sec"):
        vals = g["contrast"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        sem = np.nan
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / math.sqrt(len(vals)))
        rows.append(
            {
                "time_center_sec": float(time_center),
                "contrast_mean": float(np.mean(vals)),
                "contrast_sem": sem,
                "n_subjects": int(len(vals)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        raise ValueError("No D1 contrast summary rows were produced")
    return out.sort_values("time_center_sec")


def run_connect_sensorwide_d1_split_bayes_interval(
    output_dir: Path | str = OUTPUT_DIR,
):
    output_dir = Path(output_dir)
    score_df = require_csv(
        output_dir / "connect_sensorwide_model_timecourse_subject_scores.csv"
    )
    contrast_df = compute_d1_contrast(score_df)
    times = contrast_df["time_center_sec"].to_numpy(dtype=float)
    y = contrast_df["contrast"].to_numpy(dtype=float)
    sigma = estimate_sigma(y)
    unique_times = np.sort(contrast_df["time_center_sec"].unique())
    early_intervals = candidate_intervals(unique_times, EARLY_LB, EARLY_UB_MAX)
    late_intervals = candidate_intervals(unique_times, LATE_LB, LATE_UB_MAX)

    all_interval_rows = []
    model_rows = []
    none_row = model0_result(y, sigma)
    model_rows.append(none_row)

    early_raw = one_interval_results(y, times, sigma, early_intervals, "early")
    early_rows, early_log_ev = posterior_weights(early_raw)
    for row in early_rows:
        all_interval_rows.append(row)
    model_rows.append(summarize_model(early_rows, early_log_ev))

    late_raw = one_interval_results(y, times, sigma, late_intervals, "late")
    late_rows, late_log_ev = posterior_weights(late_raw)
    for row in late_rows:
        all_interval_rows.append(row)
    model_rows.append(summarize_model(late_rows, late_log_ev))

    early_late_raw = two_interval_results(
        y,
        times,
        sigma,
        early_intervals,
        late_intervals,
    )
    early_late_rows, early_late_log_ev = posterior_weights(early_late_raw)
    for row in early_late_rows:
        all_interval_rows.append(row)
    model_rows.append(summarize_model(early_late_rows, early_late_log_ev))

    model_log_vals = []
    for row in model_rows:
        model_log_vals.append(float(row["log_evidence"]))
    model_denom = logsumexp(model_log_vals)
    for row in model_rows:
        row["posterior_model_prob"] = math.exp(
            float(row["log_evidence"]) - model_denom
        )
        row["active_pct"] = ACTIVE_PCT
        row["tau_effect"] = TAU_EFFECT
        row["sigma_observation"] = sigma

    contrast_path = output_dir / "connect_sensorwide_d1_split_contrast.csv"
    contrast_summary_path = (
        output_dir / "connect_sensorwide_d1_split_contrast_summary.csv"
    )
    model_path = output_dir / "connect_sensorwide_d1_split_bayes_models.csv"
    interval_path = output_dir / "connect_sensorwide_d1_split_bayes_intervals.csv"

    contrast_df.to_csv(contrast_path, index=False)
    group_contrast_summary(contrast_df).to_csv(contrast_summary_path, index=False)
    pd.DataFrame(model_rows).to_csv(model_path, index=False)
    pd.DataFrame(all_interval_rows).to_csv(interval_path, index=False)

    print(f"[connect D1 Bayes] wrote {contrast_path}", flush=True)
    print(f"[connect D1 Bayes] wrote {contrast_summary_path}", flush=True)
    print(f"[connect D1 Bayes] wrote {model_path}", flush=True)
    print(f"[connect D1 Bayes] wrote {interval_path}", flush=True)
    return {
        "contrast": contrast_path,
        "contrast_summary": contrast_summary_path,
        "models": model_path,
        "intervals": interval_path,
    }


if __name__ == "__main__":
    run_connect_sensorwide_d1_split_bayes_interval()
