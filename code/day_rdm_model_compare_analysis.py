#!/usr/bin/env python3
"""Compare empirical day RDMs with standard gradual and stage model RDMs."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DAYS = [1, 2, 3, 4, 5]
N_PERMUTATIONS = int(os.environ.get("N_PERMUTATIONS", "5000"))
RANDOM_STATE = 42


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing day-RDM input: {path}. "
            "Run model_compare_5x5_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty day-RDM input: {path}")
    return d


def empirical_distance(value, value_kind):
    if value_kind == "distance":
        return float(value)
    if value_kind == "similarity":
        return float(1.0 - value)
    raise ValueError(f"Unknown empirical value kind: {value_kind}")


def rank_vector(vals):
    vals = np.asarray(vals, dtype=float)
    good = np.isfinite(vals)
    out = np.full(vals.shape, np.nan, dtype=float)
    if int(np.sum(good)) == 0:
        return out
    finite = vals[good]
    order = np.argsort(finite, kind="mergesort")
    ranks = np.empty(len(finite), dtype=float)
    sorted_vals = finite[order]
    start = 0
    while start < len(sorted_vals):
        stop = start + 1
        while stop < len(sorted_vals) and sorted_vals[stop] == sorted_vals[start]:
            stop += 1
        rank_val = (start + stop - 1) / 2.0
        for idx in range(start, stop):
            ranks[order[idx]] = rank_val
        start = stop
    out[good] = ranks
    return out


def finite_corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if int(np.sum(good)) < 3:
        return np.nan
    x = x[good]
    y = y[good]
    x = x - float(np.mean(x))
    y = y - float(np.mean(y))
    denom = float(np.sqrt(np.sum(x**2) * np.sum(y**2)))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(x * y) / denom)


def spearman_corr(x, y):
    return finite_corr(rank_vector(x), rank_vector(y))


def pair_list():
    rows = []
    for i, day_i in enumerate(DAYS):
        for day_j in DAYS[(i + 1) :]:
            rows.append({"day_i": day_i, "day_j": day_j})
    return rows


def symmetrised_subject_day_rdm(d_subject):
    value_kind = str(d_subject["value_kind"].iloc[0])
    rows = []
    for pair in pair_list():
        day_i = int(pair["day_i"])
        day_j = int(pair["day_j"])
        vals = []
        d_forward = d_subject[
            (d_subject["train_day"] == day_i) & (d_subject["test_day"] == day_j)
        ]
        d_reverse = d_subject[
            (d_subject["train_day"] == day_j) & (d_subject["test_day"] == day_i)
        ]
        for _, row in d_forward.iterrows():
            vals.append(empirical_distance(row["value"], value_kind))
        for _, row in d_reverse.iterrows():
            vals.append(empirical_distance(row["value"], value_kind))
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        rows.append(
            {
                "day_i": day_i,
                "day_j": day_j,
                "distance": float(np.mean(vals)),
            }
        )
    if len(rows) < 4:
        raise ValueError(
            "Too few empirical day-RDM distances for subject: "
            f"n={len(rows)}"
        )
    return rows


def remap_day(day, day_map):
    if day_map is None:
        return int(day)
    return int(day_map[int(day)])


def model_distance(model, day_i, day_j, split_day=None, day_map=None):
    day_i = remap_day(day_i, day_map)
    day_j = remap_day(day_j, day_map)
    if model == "gradual":
        return float(abs(day_i - day_j))
    if model == "two_stage":
        if split_day is None:
            raise ValueError("two_stage model requires split_day")
        i_late = day_i > split_day
        j_late = day_j > split_day
        if i_late == j_late:
            return 0.0
        return 1.0
    raise ValueError(f"Unknown day-RDM model: {model}")


def model_vector(day_rows, model, split_day=None, day_map=None):
    vals = []
    for row in day_rows:
        vals.append(
            model_distance(
                model,
                int(row["day_i"]),
                int(row["day_j"]),
                split_day=split_day,
                day_map=day_map,
            )
        )
    return np.asarray(vals, dtype=float)


def empirical_vector(day_rows):
    vals = []
    for row in day_rows:
        vals.append(float(row["distance"]))
    return np.asarray(vals, dtype=float)


def model_specs():
    rows = []
    rows.append({"model": "gradual", "split_day": np.nan, "label": "gradual"})
    for split_day in [1, 2, 3, 4]:
        rows.append(
            {
                "model": "two_stage",
                "split_day": split_day,
                "label": f"two_stage_{split_day}",
            }
        )
    return rows


def best_two_stage(day_rows, emp_vec, day_map=None):
    best_rho = np.nan
    best_split = np.nan
    for split_day in [1, 2, 3, 4]:
        pred = model_vector(
            day_rows,
            "two_stage",
            split_day=split_day,
            day_map=day_map,
        )
        rho = spearman_corr(emp_vec, pred)
        if not np.isfinite(rho):
            continue
        if not np.isfinite(best_rho) or rho > best_rho:
            best_rho = float(rho)
            best_split = int(split_day)
    return best_rho, best_split


def score_subject(day_rows):
    emp_vec = empirical_vector(day_rows)
    rows = []
    gradual_pred = model_vector(day_rows, "gradual")
    rows.append(
        {
            "model": "gradual",
            "split_day": np.nan,
            "rho": spearman_corr(emp_vec, gradual_pred),
        }
    )
    for split_day in [1, 2, 3, 4]:
        pred = model_vector(day_rows, "two_stage", split_day=split_day)
        rows.append(
            {
                "model": "two_stage",
                "split_day": split_day,
                "rho": spearman_corr(emp_vec, pred),
            }
        )
    return rows


def permuted_day_map(rng):
    shuffled = np.asarray(DAYS, dtype=int).copy()
    rng.shuffle(shuffled)
    out = {}
    for idx, day in enumerate(DAYS):
        out[day] = int(shuffled[idx])
    return out


def permutation_score(spec, day_rows, emp_vec, rng):
    day_map = permuted_day_map(rng)
    split_arg = None
    if np.isfinite(spec["split_day"]):
        split_arg = int(spec["split_day"])
    pred = model_vector(
        day_rows,
        spec["model"],
        split_day=split_arg,
        day_map=day_map,
    )
    return spearman_corr(emp_vec, pred)


def split_mean_from_subject_scores(payloads, split_day):
    vals = []
    for payload in payloads:
        for score in payload["scores"]:
            if score["model"] != "two_stage":
                continue
            if int(score["split_day"]) != int(split_day):
                continue
            if np.isfinite(score["rho"]):
                vals.append(float(score["rho"]))
    if len(vals) == 0:
        return np.nan, np.nan, 0
    arr = np.asarray(vals, dtype=float)
    sem = np.nan
    if len(arr) > 1:
        sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return float(np.mean(arr)), sem, int(len(arr))


def gradual_mean_from_subject_scores(payloads):
    vals = []
    for payload in payloads:
        for score in payload["scores"]:
            if score["model"] != "gradual":
                continue
            if np.isfinite(score["rho"]):
                vals.append(float(score["rho"]))
    if len(vals) == 0:
        return np.nan, np.nan, 0
    arr = np.asarray(vals, dtype=float)
    sem = np.nan
    if len(arr) > 1:
        sem = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return float(np.mean(arr)), sem, int(len(arr))


def shared_best_split(payloads):
    best_split = np.nan
    best_mean = np.nan
    best_sem = np.nan
    best_n = 0
    for split_day in [1, 2, 3, 4]:
        mean_val, sem_val, n_val = split_mean_from_subject_scores(
            payloads,
            split_day,
        )
        if not np.isfinite(mean_val):
            continue
        if not np.isfinite(best_mean) or mean_val > best_mean:
            best_split = int(split_day)
            best_mean = float(mean_val)
            best_sem = sem_val
            best_n = int(n_val)
    return best_split, best_mean, best_sem, best_n


def permuted_shared_scores(payloads, specs, rng):
    split_accum = {}
    split_counts = {}
    for split_day in [1, 2, 3, 4]:
        split_accum[split_day] = 0.0
        split_counts[split_day] = 0
    gradual_accum = 0.0
    gradual_count = 0
    for payload in payloads:
        day_map = permuted_day_map(rng)
        for spec in specs:
            split_arg = None
            if np.isfinite(spec["split_day"]):
                split_arg = int(spec["split_day"])
            pred = model_vector(
                payload["day_rows"],
                spec["model"],
                split_day=split_arg,
                day_map=day_map,
            )
            rho = spearman_corr(payload["emp_vec"], pred)
            if not np.isfinite(rho):
                continue
            if spec["model"] == "gradual":
                gradual_accum += float(rho)
                gradual_count += 1
            elif spec["model"] == "two_stage":
                split_day = int(spec["split_day"])
                split_accum[split_day] += float(rho)
                split_counts[split_day] += 1
    gradual_mean = np.nan
    if gradual_count > 0:
        gradual_mean = gradual_accum / float(gradual_count)
    best_mean = np.nan
    best_split = np.nan
    for split_day in [1, 2, 3, 4]:
        if split_counts[split_day] == 0:
            continue
        mean_val = split_accum[split_day] / float(split_counts[split_day])
        if not np.isfinite(best_mean) or mean_val > best_mean:
            best_mean = float(mean_val)
            best_split = int(split_day)
    return gradual_mean, best_mean, best_split


def condition_results(condition_key, d_condition, rng):
    modality, measure, window, value_kind = condition_key
    specs = model_specs()
    score_rows = []
    payloads = []
    for subject, g_subject in d_condition.groupby("subject"):
        day_rows = symmetrised_subject_day_rdm(g_subject)
        emp_vec = empirical_vector(day_rows)
        subject_scores = score_subject(day_rows)
        payloads.append(
            {
                "subject": int(subject),
                "day_rows": day_rows,
                "emp_vec": emp_vec,
                "scores": subject_scores,
            }
        )
        for score in subject_scores:
            row = {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "subject": int(subject),
            }
            for col, val in score.items():
                row[col] = val
            score_rows.append(row)

    observed = {}
    for spec in specs:
        observed[spec["label"]] = []
    for payload in payloads:
        for score in payload["scores"]:
            label = str(score["model"])
            if label == "two_stage":
                label = f"two_stage_{int(score['split_day'])}"
            if label in observed and np.isfinite(score["rho"]):
                observed[label].append(float(score["rho"]))

    shared_split, shared_mean, shared_sem, shared_n = shared_best_split(payloads)
    gradual_mean, _gradual_sem, _gradual_n = gradual_mean_from_subject_scores(
        payloads
    )

    perm_scores = {}
    for spec in specs:
        perm_scores[spec["label"]] = []
    perm_shared_scores = []
    perm_shared_diffs = []
    for _perm_i in range(N_PERMUTATIONS):
        accum = {}
        counts = {}
        for spec in specs:
            accum[spec["label"]] = 0.0
            counts[spec["label"]] = 0
        for payload in payloads:
            day_map = permuted_day_map(rng)
            for spec in specs:
                split_arg = None
                if np.isfinite(spec["split_day"]):
                    split_arg = int(spec["split_day"])
                pred = model_vector(
                    payload["day_rows"],
                    spec["model"],
                    split_day=split_arg,
                    day_map=day_map,
                )
                rho = spearman_corr(payload["emp_vec"], pred)
                if np.isfinite(rho):
                    accum[spec["label"]] += float(rho)
                    counts[spec["label"]] += 1
        for spec in specs:
            label = spec["label"]
            if counts[label] > 0:
                perm_scores[label].append(accum[label] / float(counts[label]))
        perm_gradual, perm_shared, _perm_split = permuted_shared_scores(
            payloads,
            specs,
            rng,
        )
        if np.isfinite(perm_shared):
            perm_shared_scores.append(float(perm_shared))
        if np.isfinite(perm_shared) and np.isfinite(perm_gradual):
            perm_shared_diffs.append(float(perm_shared) - float(perm_gradual))

    summary_rows = []
    for spec in specs:
        label = spec["label"]
        vals = np.asarray(observed[label], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        observed_mean = float(np.mean(vals))
        sem = np.nan
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        p_greater = np.nan
        if len(perm_scores[label]) > 0:
            count = 0
            for val in perm_scores[label]:
                if val >= observed_mean:
                    count += 1
            p_greater = float((count + 1.0) / (len(perm_scores[label]) + 1.0))
        split_day = spec["split_day"]
        if spec["model"] == "two_stage_best":
            split_vals = []
            for payload in payloads:
                for score in payload["scores"]:
                    if score["model"] == "two_stage_best":
                        if np.isfinite(score["split_day"]):
                            split_vals.append(float(score["split_day"]))
            if len(split_vals) > 0:
                split_day = float(pd.Series(split_vals).mode().iloc[0])
        summary_rows.append(
            {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "model": spec["model"],
                "split_day": split_day,
                "mean_rho": observed_mean,
                "sem_rho": sem,
                "n_subjects": int(len(vals)),
                "p_perm_greater": p_greater,
                "n_permutations": int(len(perm_scores[label])),
            }
        )

    p_shared = np.nan
    if len(perm_shared_scores) > 0 and np.isfinite(shared_mean):
        count = 0
        for val in perm_shared_scores:
            if val >= shared_mean:
                count += 1
        p_shared = float((count + 1.0) / (len(perm_shared_scores) + 1.0))
    summary_rows.append(
        {
            "modality": modality,
            "measure": measure,
            "window": window,
            "value_kind": value_kind,
            "model": "two_stage_shared_best",
            "split_day": shared_split,
            "mean_rho": shared_mean,
            "sem_rho": shared_sem,
            "n_subjects": shared_n,
            "p_perm_greater": p_shared,
            "n_permutations": int(len(perm_shared_scores)),
        }
    )

    pairwise_rows = []
    observed_delta = np.nan
    if np.isfinite(shared_mean) and np.isfinite(gradual_mean):
        observed_delta = float(shared_mean) - float(gradual_mean)
    p_greater = np.nan
    if len(perm_shared_diffs) > 0 and np.isfinite(observed_delta):
        count = 0
        for val in perm_shared_diffs:
            if val >= observed_delta:
                count += 1
        p_greater = float((count + 1.0) / (len(perm_shared_diffs) + 1.0))
    pairwise_rows.append(
        {
            "modality": modality,
            "measure": measure,
            "window": window,
            "value_kind": value_kind,
            "shared_best_split_day": shared_split,
            "mean_diff_shared_two_minus_gradual": observed_delta,
            "n_subjects": int(min(shared_n, _gradual_n)),
            "p_perm_shared_two_greater_gradual": p_greater,
            "n_permutations": int(len(perm_shared_diffs)),
        }
    )
    return score_rows, summary_rows, pairwise_rows


def collect_results(empirical_df):
    rng = np.random.default_rng(RANDOM_STATE)
    score_rows = []
    summary_rows = []
    pairwise_rows = []
    group_cols = ["modality", "measure", "window", "value_kind"]
    for key, g in empirical_df.groupby(group_cols):
        print(f"[day RDM] {key}", flush=True)
        scores, summary, pairwise = condition_results(key, g, rng)
        for row in scores:
            score_rows.append(row)
        for row in summary:
            summary_rows.append(row)
        for row in pairwise:
            pairwise_rows.append(row)
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(summary_rows),
        pd.DataFrame(pairwise_rows),
    )


def write_group_day_rdms(empirical_df, output_dir):
    rows = []
    group_cols = ["modality", "measure", "window", "value_kind"]
    for key, g in empirical_df.groupby(group_cols):
        modality, measure, window, value_kind = key
        for pair in pair_list():
            day_i = int(pair["day_i"])
            day_j = int(pair["day_j"])
            vals = []
            d_forward = g[(g["train_day"] == day_i) & (g["test_day"] == day_j)]
            d_reverse = g[(g["train_day"] == day_j) & (g["test_day"] == day_i)]
            for _, row in d_forward.iterrows():
                vals.append(empirical_distance(row["value"], value_kind))
            for _, row in d_reverse.iterrows():
                vals.append(empirical_distance(row["value"], value_kind))
            vals = np.asarray(vals, dtype=float)
            vals = vals[np.isfinite(vals)]
            val = np.nan
            if len(vals) > 0:
                val = float(np.mean(vals))
            rows.append(
                {
                    "modality": modality,
                    "measure": measure,
                    "window": window,
                    "value_kind": value_kind,
                    "day_i": day_i,
                    "day_j": day_j,
                    "distance_mean": val,
                    "n_values": int(len(vals)),
                }
            )
    path = output_dir / "day_rdm_model_compare_group_rdms.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_model_rdm_correlations(output_dir):
    rows = []
    pair_rows = pair_list()
    models = [
        {"model": "gradual", "split_day": np.nan, "label": "gradual"},
        {"model": "two_stage", "split_day": 1, "label": "two_stage_D1"},
        {"model": "two_stage", "split_day": 2, "label": "two_stage_D2"},
        {"model": "two_stage", "split_day": 3, "label": "two_stage_D3"},
        {"model": "two_stage", "split_day": 4, "label": "two_stage_D4"},
    ]
    vectors = {}
    for spec in models:
        split_arg = None
        if np.isfinite(spec["split_day"]):
            split_arg = int(spec["split_day"])
        vectors[spec["label"]] = model_vector(
            pair_rows,
            spec["model"],
            split_day=split_arg,
        )
    for spec_i in models:
        for spec_j in models:
            label_i = spec_i["label"]
            label_j = spec_j["label"]
            rows.append(
                {
                    "model_i": label_i,
                    "model_j": label_j,
                    "spearman_rho": spearman_corr(
                        vectors[label_i],
                        vectors[label_j],
                    ),
                }
            )
    path = output_dir / "day_rdm_model_compare_model_correlations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_day_rdm_model_compare(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    empirical_path = output_dir / "model_compare_5x5_empirical_values.csv"
    empirical_df = require_csv(empirical_path)
    print("[day RDM] comparing empirical day RDMs to model RDMs", flush=True)
    scores_df, summary_df, pairwise_df = collect_results(empirical_df)
    scores_path = output_dir / "day_rdm_model_compare_subject_scores.csv"
    summary_path = output_dir / "day_rdm_model_compare_summary.csv"
    pairwise_path = output_dir / "day_rdm_model_compare_pairwise.csv"
    scores_df.to_csv(scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    group_path = write_group_day_rdms(empirical_df, output_dir)
    model_corr_path = write_model_rdm_correlations(output_dir)
    print(f"[day RDM] wrote {scores_path}", flush=True)
    print(f"[day RDM] wrote {summary_path}", flush=True)
    print(f"[day RDM] wrote {pairwise_path}", flush=True)
    print(f"[day RDM] wrote {group_path}", flush=True)
    print(f"[day RDM] wrote {model_corr_path}", flush=True)
    return {
        "subject_scores": scores_path,
        "summary": summary_path,
        "pairwise": pairwise_path,
        "group_rdms": group_path,
        "model_correlations": model_corr_path,
    }


if __name__ == "__main__":
    run_day_rdm_model_compare()
