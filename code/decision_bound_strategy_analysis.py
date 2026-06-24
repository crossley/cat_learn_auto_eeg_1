#!/usr/bin/env python3
"""GRT decision-bound strategy fits and links to MVPA block restructuring."""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from scipy.special import ndtr

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None

from block_model_utils import assign_accumulated_blocks, sem
from load_project_data import load_sessions

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
N_JOBS = 8
EPS = 1e-9


def _response_vector(resp) -> np.ndarray:
    labels = pd.Series(resp).astype(str).str.upper().to_numpy()
    y = np.full(len(labels), -1, dtype=int)
    y[labels == "A"] = 0
    y[labels == "B"] = 1
    if np.any(y < 0):
        bad = sorted(set(labels[y < 0].tolist()))
        raise ValueError(f"unexpected response labels: {bad}")
    return y


def _line_probability_b(xy, a1, a2, b, sigma, polarity):
    xy = np.asarray(xy, dtype=float)
    norm = float(np.sqrt(a1 * a1 + a2 * a2))
    if norm <= np.finfo(float).eps:
        raise ValueError("zero decision-bound normal vector")
    z = (a1 * xy[:, 0] + a2 * xy[:, 1] + b) / (float(sigma) * norm)
    p_positive = ndtr(z)
    p_b = p_positive if int(polarity) == 1 else 1.0 - p_positive
    return np.clip(p_b, EPS, 1.0 - EPS)


def _neg_loglike(y_resp, p_b):
    y_resp = np.asarray(y_resp, dtype=int)
    return -float(np.sum(y_resp * np.log(p_b) + (1 - y_resp) * np.log(1.0 - p_b)))


def _criteria(log_likelihood, n_obs, n_params):
    aic = 2.0 * n_params - 2.0 * log_likelihood
    bic = np.log(float(n_obs)) * n_params - 2.0 * log_likelihood
    return float(aic), float(bic)


def _fit_unix(xy, y_resp, polarity, x_bounds, sigma_bounds):
    def objective(params):
        c, log_sigma = params
        sigma = float(np.exp(log_sigma))
        p_b = _line_probability_b(xy, 1.0, 0.0, -float(c), sigma, polarity)
        return _neg_loglike(y_resp, p_b)

    starts = [
        [np.nanmedian(xy[:, 0]), np.log(8.0)],
        [np.nanpercentile(xy[:, 0], 35), np.log(12.0)],
        [np.nanpercentile(xy[:, 0], 65), np.log(12.0)],
    ]
    bounds = [x_bounds, (np.log(sigma_bounds[0]), np.log(sigma_bounds[1]))]
    return _best_minimize(objective, starts, bounds)


def _fit_uniy(xy, y_resp, polarity, y_bounds, sigma_bounds):
    def objective(params):
        c, log_sigma = params
        sigma = float(np.exp(log_sigma))
        p_b = _line_probability_b(xy, 0.0, 1.0, -float(c), sigma, polarity)
        return _neg_loglike(y_resp, p_b)

    starts = [
        [np.nanmedian(xy[:, 1]), np.log(8.0)],
        [np.nanpercentile(xy[:, 1], 35), np.log(12.0)],
        [np.nanpercentile(xy[:, 1], 65), np.log(12.0)],
    ]
    bounds = [y_bounds, (np.log(sigma_bounds[0]), np.log(sigma_bounds[1]))]
    return _best_minimize(objective, starts, bounds)


def _fit_glc(xy, y_resp, polarity, b_bounds, sigma_bounds):
    center = np.nanmean(xy, axis=0)
    starts = []
    for theta in np.linspace(-np.pi, np.pi, 8, endpoint=False):
        a = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        starts.append([theta, -float(center @ a), np.log(10.0)])
    bounds = [
        (-np.pi, np.pi),
        b_bounds,
        (np.log(sigma_bounds[0]), np.log(sigma_bounds[1])),
    ]

    def objective(params):
        theta, b, log_sigma = params
        sigma = float(np.exp(log_sigma))
        a1 = float(np.cos(theta))
        a2 = float(np.sin(theta))
        p_b = _line_probability_b(xy, a1, a2, float(b), sigma, polarity)
        return _neg_loglike(y_resp, p_b)

    return _best_minimize(objective, starts, bounds)


def _best_minimize(objective, starts, bounds):
    best = None
    for start in starts:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = minimize(
                objective,
                np.asarray(start, dtype=float),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 1000, "ftol": 1e-9},
            )
        if best is None or float(result.fun) < float(best.fun):
            best = result
    return best


def _fit_model_family(xy, y_resp, model, bounds):
    fits = []
    for polarity in [1, -1]:
        if model == "unix":
            result = _fit_unix(xy, y_resp, polarity, bounds["x"], bounds["sigma"])
            c, log_sigma = result.x
            params = {
                "a1": 1.0,
                "a2": 0.0,
                "b": -float(c),
                "criterion": float(c),
                "sigma": float(np.exp(log_sigma)),
            }
            n_params = 2
        elif model == "uniy":
            result = _fit_uniy(xy, y_resp, polarity, bounds["y"], bounds["sigma"])
            c, log_sigma = result.x
            params = {
                "a1": 0.0,
                "a2": 1.0,
                "b": -float(c),
                "criterion": float(c),
                "sigma": float(np.exp(log_sigma)),
            }
            n_params = 2
        elif model == "glc":
            result = _fit_glc(xy, y_resp, polarity, bounds["b"], bounds["sigma"])
            theta, b, log_sigma = result.x
            params = {
                "a1": float(np.cos(theta)),
                "a2": float(np.sin(theta)),
                "b": float(b),
                "criterion": np.nan,
                "theta": float(theta),
                "sigma": float(np.exp(log_sigma)),
            }
            n_params = 3
        else:
            raise ValueError(f"unknown decision-bound model: {model}")
        log_likelihood = -float(result.fun)
        aic, bic = _criteria(log_likelihood, len(y_resp), n_params)
        fits.append(
            {
                "model": model,
                "polarity": int(polarity),
                "log_likelihood": log_likelihood,
                "aic": aic,
                "bic": bic,
                "n_params": int(n_params),
                "fit_success": bool(result.success),
                "fit_message": str(result.message),
                **params,
            }
        )
    return min(fits, key=lambda row: row["bic"])


def _bounds_for_xy(xy):
    x_min, y_min = np.nanmin(xy, axis=0)
    x_max, y_max = np.nanmax(xy, axis=0)
    pad = 20.0
    return {
        "x": (float(x_min - pad), float(x_max + pad)),
        "y": (float(y_min - pad), float(y_max + pad)),
        "b": (-160.0, 160.0),
        "sigma": (0.5, 80.0),
    }


def fit_block_models(block_df):
    xy = block_df[["x", "y"]].to_numpy(dtype=float)
    y_resp = _response_vector(block_df["resp"])
    bounds = _bounds_for_xy(xy)
    rows = []
    for model in ["unix", "uniy", "glc"]:
        rows.append(_fit_model_family(xy, y_resp, model, bounds))
    return rows


def load_block_behaviour(max_subjects=None):
    sessions = load_sessions(load_epochs=False)
    rows = []
    for item in sessions:
        d = item["beh"].copy()
        d["subject"] = int(item["subject"])
        d["day"] = int(item["day"])
        d = assign_accumulated_blocks(d)
        rows.append(d)
    beh = pd.concat(rows, ignore_index=True)
    if max_subjects is not None:
        keep = sorted(beh["subject"].unique())[: int(max_subjects)]
        beh = beh[beh["subject"].isin(keep)].copy()
    return beh


def process_subject(task):
    subject = int(task["subject"])
    d_sub = task["beh"]
    rows = []
    qc_rows = []
    for block, g in d_sub.groupby("block"):
        try:
            model_rows = fit_block_models(g)
            for row in model_rows:
                row.update(
                    {
                        "subject": subject,
                        "day": int(g["day"].iloc[0]),
                        "day_block": int(g["day_block"].iloc[0]),
                        "block": int(block),
                        "n_trials": int(len(g)),
                        "n_resp_a": int((g["resp"].astype(str).str.upper() == "A").sum()),
                        "n_resp_b": int((g["resp"].astype(str).str.upper() == "B").sum()),
                        "accuracy": float((g["fb"].astype(str).str.lower() == "correct").mean()),
                    }
                )
                rows.append(row)
        except Exception as exc:
            qc_rows.append(
                {
                    "subject": subject,
                    "block": int(block),
                    "stage": "fit_block",
                    "reason": "fit_error",
                    "detail": str(exc),
                }
            )
    print(f"[decision bound] subject {subject}: {len(rows)} fit rows", flush=True)
    return {"rows": rows, "qc": qc_rows}


def add_model_weights(fits_df):
    frames = []
    for _key, g in fits_df.groupby(["subject", "block"], dropna=False):
        g = g.copy()
        min_bic = float(g["bic"].min())
        g["delta_bic_best"] = g["bic"] - min_bic
        raw = np.exp(-0.5 * g["delta_bic_best"].to_numpy(dtype=float))
        denom = float(np.sum(raw))
        g["bic_weight"] = raw / denom if denom > 0 else np.nan
        best_1d = float(g[g["model"].isin(["unix", "uniy"])]["bic"].min())
        glc_bic = float(g[g["model"] == "glc"]["bic"].iloc[0])
        g["glc_bic_advantage_vs_best_1d"] = best_1d - glc_bic
        g["is_best_model"] = g["delta_bic_best"] <= 1e-9
        frames.append(g)
    return pd.concat(frames, ignore_index=True).sort_values(["subject", "block", "model"])


def infer_strategy_switch(weights_df, min_bic_advantage=0.0):
    rows = []
    glc = weights_df[weights_df["model"] == "glc"].copy()
    for subject, g in glc.groupby("subject"):
        g = g.sort_values("block").reset_index(drop=True)
        switch_block = np.nan
        status = "no_persistent_glc_switch"
        for i in range(len(g) - 1):
            now = g.iloc[i]
            nxt = g.iloc[i + 1]
            if int(nxt["block"]) != int(now["block"]) + 1:
                continue
            now_ok = bool(now["is_best_model"]) and float(now["glc_bic_advantage_vs_best_1d"]) > min_bic_advantage
            next_ok = bool(nxt["is_best_model"]) and float(nxt["glc_bic_advantage_vs_best_1d"]) > min_bic_advantage
            if now_ok and next_ok:
                switch_block = int(now["block"])
                status = "persistent_glc_switch"
                break
        if np.isfinite(switch_block):
            day = ((int(switch_block) - 1) // 5) + 1
            day_block = ((int(switch_block) - 1) % 5) + 1
        else:
            day = np.nan
            day_block = np.nan
        rows.append(
            {
                "subject": int(subject),
                "strategy_switch_block": switch_block,
                "strategy_switch_day": day,
                "strategy_switch_day_block": day_block,
                "switch_status": status,
                "min_bic_advantage": float(min_bic_advantage),
            }
        )
    return pd.DataFrame(rows).sort_values("subject")


def extract_mvpa_split_blocks(output_dir, tmin=0.3, tmax=0.6):
    path = Path(output_dir) / "mvpa_block_model_timecourse_subject.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MVPA block output: {path}")
    d = pd.read_csv(path)
    d = d[
        (d["model"] == "discrete")
        & (d["time_sec"] >= float(tmin))
        & (d["time_sec"] <= float(tmax))
        & np.isfinite(d["split_block"])
    ].copy()
    if d.empty:
        raise ValueError(f"No finite MVPA discrete rows in {path}")
    d["evidence_above_baseline"] = -d["delta_bic_baseline"].astype(float)
    d["day1_split"] = d["split_block"].astype(float) <= 5.0
    split_summary = (
        d.groupby(["subject", "split_block"], as_index=False)
        .agg(
            mean_delta_bic_best=("delta_bic_best", "mean"),
            mean_evidence_above_baseline=("evidence_above_baseline", "mean"),
            n_times=("time_sec", "nunique"),
        )
        .sort_values(["subject", "mean_delta_bic_best", "split_block"])
    )
    best = split_summary.groupby("subject", as_index=False).first()
    best = best.rename(
        columns={
            "split_block": "mvpa_best_split_block",
            "mean_delta_bic_best": "mvpa_split_mean_delta_bic_best",
            "mean_evidence_above_baseline": "mvpa_split_mean_evidence_above_baseline",
        }
    )
    split_summary["mvpa_tmin"] = float(tmin)
    split_summary["mvpa_tmax"] = float(tmax)
    best["mvpa_tmin"] = float(tmin)
    best["mvpa_tmax"] = float(tmax)
    best["mvpa_best_split_day"] = ((best["mvpa_best_split_block"].astype(int) - 1) // 5) + 1
    best["mvpa_best_split_day_block"] = ((best["mvpa_best_split_block"].astype(int) - 1) % 5) + 1
    return split_summary, best.sort_values("subject")


def make_mvpa_link(switch_df, mvpa_best_df):
    d = switch_df.merge(mvpa_best_df, on="subject", how="inner")
    d["block_distance"] = d["mvpa_best_split_block"] - d["strategy_switch_block"]
    d["abs_block_distance"] = np.abs(d["block_distance"])
    finite_switch = np.isfinite(d["strategy_switch_block"])
    d["switch_group"] = np.where(
        finite_switch & (d["strategy_switch_block"] <= 5),
        "day1_or_early",
        np.where(finite_switch, "later", "no_switch"),
    )
    return d.sort_values("subject")


def summarize_block_weights(weights_df):
    rows = []
    for key, g in weights_df.groupby(["block", "day", "day_block", "model"], dropna=False):
        block, day, day_block, model = key
        rows.append(
            {
                "block": int(block),
                "day": int(day),
                "day_block": int(day_block),
                "model": model,
                "bic_weight_mean": float(np.nanmean(g["bic_weight"])),
                "bic_weight_sem": sem(g["bic_weight"]),
                "delta_bic_best_mean": float(np.nanmean(g["delta_bic_best"])),
                "glc_advantage_mean": float(np.nanmean(g["glc_bic_advantage_vs_best_1d"])),
                "glc_advantage_sem": sem(g["glc_bic_advantage_vs_best_1d"]),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["block", "model"])


def run_decision_bound_strategy_analysis(
    output_dir=OUTPUT_DIR,
    n_workers=None,
    max_subjects=None,
    min_bic_advantage=0.0,
    mvpa_tmin=0.3,
    mvpa_tmax=0.6,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    beh = load_block_behaviour(max_subjects=max_subjects)
    tasks = [
        {"subject": int(subject), "beh": g.copy()}
        for subject, g in sorted(beh.groupby("subject"))
    ]
    if n_workers is None:
        n_workers = N_JOBS
    n_workers = max(1, int(n_workers))
    print(f"[decision bound] Running {len(tasks)} subjects (n_workers={n_workers})", flush=True)

    def iter_jobs():
        for task in tasks:
            yield delayed(process_subject)(task)

    if n_workers == 1:
        results = [process_subject(task) for task in tasks]
    else:
        print(
            "[decision bound] using threading backend for parallel subject fits",
            flush=True,
        )
        if threadpool_limits is None:
            results = Parallel(n_jobs=n_workers, backend="threading", verbose=0)(iter_jobs())
        else:
            with threadpool_limits(limits=1):
                results = Parallel(n_jobs=n_workers, backend="threading", verbose=0)(iter_jobs())

    fit_rows = []
    qc_rows = []
    for result in results:
        fit_rows.extend(result["rows"])
        qc_rows.extend(result["qc"])
    fits_df = pd.DataFrame(fit_rows)
    qc_df = pd.DataFrame(qc_rows, columns=["subject", "block", "stage", "reason", "detail"])
    if fits_df.empty:
        qc_df.to_csv(output_dir / "decision_bound_qc_log.csv", index=False)
        raise RuntimeError("Decision-bound fitting produced no valid rows")
    weights_df = add_model_weights(fits_df)
    switch_df = infer_strategy_switch(weights_df, min_bic_advantage=min_bic_advantage)
    mvpa_split_summary, mvpa_best = extract_mvpa_split_blocks(
        output_dir,
        tmin=mvpa_tmin,
        tmax=mvpa_tmax,
    )
    link_df = make_mvpa_link(switch_df, mvpa_best)
    block_summary = summarize_block_weights(weights_df)

    paths = {
        "fits": output_dir / "decision_bound_block_model_fits.csv",
        "weights": output_dir / "decision_bound_block_model_weights.csv",
        "block_summary": output_dir / "decision_bound_block_model_summary.csv",
        "switch": output_dir / "decision_bound_strategy_switch_subject.csv",
        "mvpa_split_summary": output_dir / "decision_bound_mvpa_split_summary.csv",
        "mvpa_link": output_dir / "decision_bound_mvpa_switch_link.csv",
        "qc": output_dir / "decision_bound_qc_log.csv",
    }
    fits_df.to_csv(paths["fits"], index=False)
    weights_df.to_csv(paths["weights"], index=False)
    block_summary.to_csv(paths["block_summary"], index=False)
    switch_df.to_csv(paths["switch"], index=False)
    mvpa_split_summary.to_csv(paths["mvpa_split_summary"], index=False)
    link_df.to_csv(paths["mvpa_link"], index=False)
    qc_df.to_csv(paths["qc"], index=False)
    for path in paths.values():
        print(f"[decision bound] wrote {path}", flush=True)
    return paths


def build_arg_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--n-workers", type=int, default=N_JOBS)
    parser.add_argument("--max-subjects", type=int, default=None)
    parser.add_argument("--min-bic-advantage", type=float, default=0.0)
    parser.add_argument("--mvpa-tmin", type=float, default=0.3)
    parser.add_argument("--mvpa-tmax", type=float, default=0.6)
    return parser


if __name__ == "__main__":
    args = build_arg_parser().parse_args()
    run_decision_bound_strategy_analysis(
        output_dir=args.output_dir,
        n_workers=args.n_workers,
        max_subjects=args.max_subjects,
        min_bic_advantage=args.min_bic_advantage,
        mvpa_tmin=args.mvpa_tmin,
        mvpa_tmax=args.mvpa_tmax,
    )
