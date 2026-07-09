#!/usr/bin/env python3
"""Shared 25-block model-evidence helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

N_BLOCKS_PER_DAY = 5
DAYS = [1, 2, 3, 4, 5]
BLOCKS = list(range(1, 26))


def sem(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) <= 1:
        return np.nan
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals)))


def assign_accumulated_blocks(beh, n_blocks_per_day=N_BLOCKS_PER_DAY):
    d = beh.sort_values("trial").reset_index(drop=True).copy()
    day = int(d["day"].iloc[0])
    d["day_block"] = np.nan
    for block_i, idx in enumerate(np.array_split(np.arange(len(d)), n_blocks_per_day), start=1):
        d.loc[idx, "day_block"] = int(block_i)
    d["day_block"] = d["day_block"].astype(int)
    d["block"] = (day - 1) * n_blocks_per_day + d["day_block"]
    return d


def template_value(model, block_i, block_j, split_block=None):
    if model == "baseline":
        return 0.0
    if block_i == block_j:
        return 0.0
    if model == "continuous":
        return float(0.65 * min(block_i, block_j) / float(max(BLOCKS)))
    if model == "discrete":
        if split_block is None:
            raise ValueError("discrete model requires split_block")
        i_late = block_i > split_block
        j_late = block_j > split_block
        if i_late != j_late:
            return 0.0
        return float(0.65 * min(block_i, block_j) / float(max(BLOCKS)))
    raise ValueError(f"Unknown model: {model}")


def model_specs():
    rows = [
        {"model_label": "Baseline", "model": "baseline", "split_block": np.nan},
        {
            "model_label": "Continuous Restructuring",
            "model": "continuous",
            "split_block": np.nan,
        },
    ]
    for split_block in range(1, max(BLOCKS)):
        rows.append(
            {
                "model_label": f"Discrete Restructuring B{split_block}",
                "model": "discrete",
                "split_block": float(split_block),
            }
        )
    return rows


def design_matrix(pair_rows, spec, include_same_block=False):
    intercept = np.ones(len(pair_rows), dtype=float)
    cols = [intercept]
    if include_same_block:
        cols.append(
            np.asarray(
                [
                    1.0 if int(row["block_i"]) == int(row["block_j"]) else 0.0
                    for row in pair_rows
                ],
                dtype=float,
            )
        )
    if spec["model"] != "baseline":
        split_block = (
            int(spec["split_block"]) if np.isfinite(spec["split_block"]) else None
        )
        cols.append(
            np.asarray(
                [
                    template_value(
                        str(spec["model"]),
                        int(row["block_i"]),
                        int(row["block_j"]),
                        split_block,
                    )
                    for row in pair_rows
                ],
                dtype=float,
            )
        )
    return np.column_stack(cols)


def fit_bic(y, x):
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    good = np.isfinite(y)
    for col_i in range(x.shape[1]):
        good &= np.isfinite(x[:, col_i])
    y = y[good]
    x = x[good]
    n_obs = int(len(y))
    if n_obs < 4:
        return {"bic": np.nan, "r2": np.nan, "n_obs": n_obs, "n_params": np.nan}
    keep_cols = [0]
    for col_i in range(1, x.shape[1]):
        col = x[:, col_i]
        if float(np.nanmax(col) - np.nanmin(col)) > np.finfo(float).eps:
            keep_cols.append(col_i)
    x = x[:, keep_cols]
    n_params = int(x.shape[1])
    beta, _resid, rank, _singular = np.linalg.lstsq(x, y, rcond=None)
    if int(rank) < n_params:
        return {"bic": np.nan, "r2": np.nan, "n_obs": n_obs, "n_params": n_params}
    pred = x @ beta
    resid = y - pred
    rss = max(float(np.sum(resid**2)), np.finfo(float).eps)
    tss = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = np.nan
    if tss > np.finfo(float).eps:
        r2 = 1.0 - rss / tss
    bic = float(n_obs * np.log(rss / float(n_obs)) + n_params * np.log(float(n_obs)))
    return {"bic": bic, "r2": float(r2), "n_obs": n_obs, "n_params": n_params}


def score_payload(subject, time_sec, pair_rows, y_vals, include_same_block=False):
    rows = []
    for spec in model_specs():
        fit = fit_bic(
            y_vals,
            design_matrix(pair_rows, spec, include_same_block=include_same_block),
        )
        rows.append(
            {
                "subject": int(subject),
                "time_sec": float(time_sec),
                "model_label": spec["model_label"],
                "model": spec["model"],
                "split_block": spec["split_block"],
                "bic": fit["bic"],
                "r2": fit["r2"],
                "n_obs": fit["n_obs"],
                "n_params": fit["n_params"],
            }
        )
    return rows


def add_delta_bic(score_df):
    frames = []
    for _key, g in score_df.groupby(["subject", "time_sec"], dropna=False):
        g = g.copy()
        finite = g["bic"].to_numpy(float)
        finite = finite[np.isfinite(finite)]
        g["delta_bic_best"] = np.nan if len(finite) == 0 else g["bic"] - float(np.min(finite))
        base = g[g["model_label"] == "Baseline"]
        if base.empty or not np.isfinite(float(base["bic"].iloc[0])):
            g["delta_bic_baseline"] = np.nan
        else:
            g["delta_bic_baseline"] = g["bic"] - float(base["bic"].iloc[0])
        frames.append(g)
    return pd.concat(frames, ignore_index=True)


def summarize_scores(score_df):
    rows = []
    group_cols = ["time_sec", "model_label", "model", "split_block"]
    for key, g in score_df.groupby(group_cols, dropna=False):
        key_vals = dict(zip(group_cols, key))
        delta_base = g["delta_bic_baseline"].to_numpy(float)
        delta_best = g["delta_bic_best"].to_numpy(float)
        r2_vals = g["r2"].to_numpy(float)
        delta_base = delta_base[np.isfinite(delta_base)]
        delta_best = delta_best[np.isfinite(delta_best)]
        r2_vals = r2_vals[np.isfinite(r2_vals)]
        rows.append(
            {
                **key_vals,
                "delta_bic_baseline_mean": float(np.mean(delta_base)) if len(delta_base) else np.nan,
                "delta_bic_baseline_sem": sem(delta_base),
                "delta_bic_best_mean": float(np.mean(delta_best)) if len(delta_best) else np.nan,
                "r2_mean": float(np.mean(r2_vals)) if len(r2_vals) else np.nan,
                "r2_sem": sem(r2_vals),
                "n_subjects": int(g["subject"].nunique()),
            }
        )
    return pd.DataFrame(rows).sort_values(["time_sec", "delta_bic_best_mean"])


def vector_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 3:
        return np.nan
    x = x[good] - float(np.mean(x[good]))
    y = y[good] - float(np.mean(y[good]))
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x * y) / denom)


def write_scores(score_rows, subject_path, summary_path):
    score_df = add_delta_bic(pd.DataFrame(score_rows))
    summary_df = summarize_scores(score_df)
    score_df.to_csv(subject_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    return score_df, summary_df


def write_grouped_scores(score_rows, group_cols, subject_path, summary_path):
    """
    Write block-model scores separately within extra analysis groups.

    The model comparison itself is unchanged.  This wrapper applies the same
    add_delta_bic/summarize_scores logic independently within each measure,
    band, or signal-estimate group so unrelated connectivity variants are not
    pooled before the BIC deltas are computed.
    """
    if not group_cols:
        return write_scores(score_rows, subject_path, summary_path)

    score_df = pd.DataFrame(score_rows)
    if score_df.empty:
        raise ValueError("No block-model score rows were produced")

    score_frames = []
    summary_frames = []
    for key, group in score_df.groupby(group_cols, dropna=False):
        key_vals = key if isinstance(key, tuple) else (key,)
        scored = add_delta_bic(group.copy())
        summary = summarize_scores(scored)
        for col, val in zip(group_cols, key_vals):
            summary[col] = val
        score_frames.append(scored)
        summary_frames.append(summary)

    score_out = pd.concat(score_frames, ignore_index=True)
    summary_out = pd.concat(summary_frames, ignore_index=True)
    summary_out = summary_out.sort_values(
        list(group_cols) + ["time_sec", "delta_bic_best_mean"]
    )
    score_out.to_csv(subject_path, index=False)
    summary_out.to_csv(summary_path, index=False)
    return score_out, summary_out
