#!/usr/bin/env python3
"""Compare MVPA transfer matrices with native cross-generalization templates."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd

from mvpa_stim_locked_cat_late_window_analysis import CLASSIFIERS

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"
DAYS = [1, 2, 3, 4, 5]
N_PERMUTATIONS = int(os.environ.get("N_PERMUTATIONS", "5000"))
RANDOM_STATE = 42


def require_csv(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing MVPA transfer input: {path}. "
            "Run the matching mvpa_stim_locked_cat_*_window_transfer_analysis.py "
            "script first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty MVPA transfer input: {path}")
    return d


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


def pair_rows(include_diagonal):
    rows = []
    for train_day in DAYS:
        for test_day in DAYS:
            if not include_diagonal and train_day == test_day:
                continue
            rows.append({"train_day": train_day, "test_day": test_day})
    return rows


def mvpa_subject_values(d_subject, include_diagonal):
    rows = []
    vals = []
    for pair in pair_rows(include_diagonal):
        train_day = int(pair["train_day"])
        test_day = int(pair["test_day"])
        d_pair = d_subject[
            (d_subject["train_day"] == train_day)
            & (d_subject["test_day"] == test_day)
        ]
        if d_pair.empty:
            continue
        pair_vals = d_pair["auc"].to_numpy(dtype=float)
        pair_vals = pair_vals[np.isfinite(pair_vals)]
        if len(pair_vals) == 0:
            continue
        rows.append({"train_day": train_day, "test_day": test_day})
        vals.append(float(np.mean(pair_vals)))
    min_n = 10
    if not include_diagonal:
        min_n = 8
    if len(vals) < min_n:
        raise ValueError(
            "Too few MVPA transfer matrix cells for model comparison: "
            f"n={len(vals)}"
        )
    return rows, np.asarray(vals, dtype=float)


def remap_day(day, day_map):
    if day_map is None:
        return int(day)
    return int(day_map[int(day)])


def template_value(template, train_day, test_day, split_day=None, day_map=None):
    train_day = remap_day(train_day, day_map)
    test_day = remap_day(test_day, day_map)
    if template == "one_stage_bottleneck":
        return float((min(train_day, test_day) - 1.0) / 4.0)
    if template == "one_stage_closeness":
        return float(1.0 - abs(train_day - test_day) / 4.0)
    if template == "two_stage_binary":
        if split_day is None:
            raise ValueError("two_stage_binary template requires split_day")
        train_late = train_day > split_day
        test_late = test_day > split_day
        if train_late == test_late:
            return 1.0
        return 0.0
    if template == "two_stage_bottleneck":
        if split_day is None:
            raise ValueError("two_stage_bottleneck template requires split_day")
        train_late = train_day > split_day
        test_late = test_day > split_day
        if train_late != test_late:
            return 0.0
        return float((min(train_day, test_day) - 1.0) / 4.0)
    raise ValueError(f"Unknown MVPA transfer template: {template}")


def template_vector(pair_rows_this, template, split_day=None, day_map=None):
    vals = []
    for row in pair_rows_this:
        vals.append(
            template_value(
                template,
                int(row["train_day"]),
                int(row["test_day"]),
                split_day=split_day,
                day_map=day_map,
            )
        )
    return np.asarray(vals, dtype=float)


def template_specs():
    rows = []
    rows.append(
        {
            "template": "one_stage_bottleneck",
            "split_day": np.nan,
            "label": "one_stage_bottleneck",
        }
    )
    rows.append(
        {
            "template": "one_stage_closeness",
            "split_day": np.nan,
            "label": "one_stage_closeness",
        }
    )
    for template in ["two_stage_binary", "two_stage_bottleneck"]:
        for split_day in [1, 2, 3, 4]:
            rows.append(
                {
                    "template": template,
                    "split_day": split_day,
                    "label": f"{template}_{split_day}",
                }
            )
    return rows


def score_subject(pair_rows_this, values):
    rows = []
    for spec in template_specs():
        split_arg = None
        if np.isfinite(spec["split_day"]):
            split_arg = int(spec["split_day"])
        pred = template_vector(
            pair_rows_this,
            spec["template"],
            split_day=split_arg,
        )
        rows.append(
            {
                "template": spec["template"],
                "split_day": spec["split_day"],
                "rho": spearman_corr(values, pred),
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


def shared_best_family(payloads, template_family):
    best_split = np.nan
    best_mean = np.nan
    best_sem = np.nan
    best_n = 0
    for split_day in [1, 2, 3, 4]:
        vals = []
        for payload in payloads:
            for score in payload["scores"]:
                if score["template"] != template_family:
                    continue
                if int(score["split_day"]) != int(split_day):
                    continue
                if np.isfinite(score["rho"]):
                    vals.append(float(score["rho"]))
        if len(vals) == 0:
            continue
        arr = np.asarray(vals, dtype=float)
        mean_val = float(np.mean(arr))
        sem_val = np.nan
        if len(arr) > 1:
            sem_val = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        if not np.isfinite(best_mean) or mean_val > best_mean:
            best_split = int(split_day)
            best_mean = mean_val
            best_sem = sem_val
            best_n = int(len(arr))
    return {
        "split_day": best_split,
        "mean": best_mean,
        "sem": best_sem,
        "n": best_n,
    }


def mean_template_score(payloads, template):
    vals = []
    for payload in payloads:
        for score in payload["scores"]:
            if score["template"] != template:
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


def permuted_scores(payloads, rng):
    accum = {}
    counts = {}
    for spec in template_specs():
        accum[spec["label"]] = 0.0
        counts[spec["label"]] = 0
    for payload in payloads:
        day_map = permuted_day_map(rng)
        for spec in template_specs():
            split_arg = None
            if np.isfinite(spec["split_day"]):
                split_arg = int(spec["split_day"])
            pred = template_vector(
                payload["pair_rows"],
                spec["template"],
                split_day=split_arg,
                day_map=day_map,
            )
            rho = spearman_corr(payload["values"], pred)
            if np.isfinite(rho):
                accum[spec["label"]] += float(rho)
                counts[spec["label"]] += 1
    out = {}
    for label in accum:
        out[label] = np.nan
        if counts[label] > 0:
            out[label] = accum[label] / float(counts[label])
    return out


def best_permuted_family(perm_scores, template_family):
    best_val = np.nan
    best_split = np.nan
    for split_day in [1, 2, 3, 4]:
        label = f"{template_family}_{split_day}"
        val = perm_scores[label]
        if not np.isfinite(val):
            continue
        if not np.isfinite(best_val) or val > best_val:
            best_val = float(val)
            best_split = int(split_day)
    return best_val, best_split


def condition_results(condition_key, d_condition, rng, include_diagonal):
    classifier, window = condition_key
    score_rows = []
    payloads = []
    for subject, g_subject in d_condition.groupby("subject"):
        pair_rows_this, values = mvpa_subject_values(g_subject, include_diagonal)
        subject_scores = score_subject(pair_rows_this, values)
        payloads.append(
            {
                "subject": int(subject),
                "pair_rows": pair_rows_this,
                "values": values,
                "scores": subject_scores,
            }
        )
        for score in subject_scores:
            row = {
                "classifier": classifier,
                "window": window,
                "include_diagonal": bool(include_diagonal),
                "subject": int(subject),
            }
            for col, val in score.items():
                row[col] = val
            score_rows.append(row)

    summary_rows = []
    observed_scores = {}
    for spec in template_specs():
        vals = []
        for payload in payloads:
            for score in payload["scores"]:
                if score["template"] != spec["template"]:
                    continue
                if np.isfinite(spec["split_day"]):
                    if int(score["split_day"]) != int(spec["split_day"]):
                        continue
                if np.isfinite(score["rho"]):
                    vals.append(float(score["rho"]))
        arr = np.asarray(vals, dtype=float)
        mean_val = np.nan
        sem_val = np.nan
        if len(arr) > 0:
            mean_val = float(np.mean(arr))
            if len(arr) > 1:
                sem_val = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
        observed_scores[spec["label"]] = mean_val
        summary_rows.append(
            {
                "classifier": classifier,
                "window": window,
                "include_diagonal": bool(include_diagonal),
                "template": spec["template"],
                "split_day": spec["split_day"],
                "mean_rho": mean_val,
                "sem_rho": sem_val,
                "n_subjects": int(len(arr)),
                "p_perm_greater": np.nan,
                "n_permutations": 0,
            }
        )

    shared = {}
    for family in ["two_stage_binary", "two_stage_bottleneck"]:
        shared[family] = shared_best_family(payloads, family)
        summary_rows.append(
            {
                "classifier": classifier,
                "window": window,
                "include_diagonal": bool(include_diagonal),
                "template": f"{family}_shared_best",
                "split_day": shared[family]["split_day"],
                "mean_rho": shared[family]["mean"],
                "sem_rho": shared[family]["sem"],
                "n_subjects": shared[family]["n"],
                "p_perm_greater": np.nan,
                "n_permutations": 0,
            }
        )

    bottleneck_mean, _bottleneck_sem, bottleneck_n = mean_template_score(
        payloads,
        "one_stage_bottleneck",
    )
    closeness_mean, _closeness_sem, closeness_n = mean_template_score(
        payloads,
        "one_stage_closeness",
    )

    perm_template_vals = {}
    for spec in template_specs():
        perm_template_vals[spec["label"]] = []
    perm_shared_vals = {}
    perm_diffs = {}
    for family in ["two_stage_binary", "two_stage_bottleneck"]:
        perm_shared_vals[family] = []
        perm_diffs[(family, "one_stage_bottleneck")] = []
        perm_diffs[(family, "one_stage_closeness")] = []
    for _perm_i in range(N_PERMUTATIONS):
        p_scores = permuted_scores(payloads, rng)
        for label, val in p_scores.items():
            if np.isfinite(val):
                perm_template_vals[label].append(float(val))
        for family in ["two_stage_binary", "two_stage_bottleneck"]:
            best_val, _best_split = best_permuted_family(p_scores, family)
            if np.isfinite(best_val):
                perm_shared_vals[family].append(float(best_val))
            for one_template in ["one_stage_bottleneck", "one_stage_closeness"]:
                one_val = p_scores[one_template]
                if np.isfinite(best_val) and np.isfinite(one_val):
                    perm_diffs[(family, one_template)].append(
                        float(best_val) - float(one_val)
                    )

    for row in summary_rows:
        template = row["template"]
        perm_vals = None
        if template.endswith("_shared_best"):
            family = template.replace("_shared_best", "")
            perm_vals = perm_shared_vals[family]
        else:
            label = template
            if np.isfinite(row["split_day"]):
                label = f"{template}_{int(row['split_day'])}"
            perm_vals = perm_template_vals[label]
        p_perm = np.nan
        if perm_vals is not None and len(perm_vals) > 0:
            count = 0
            for val in perm_vals:
                if val >= row["mean_rho"]:
                    count += 1
            p_perm = float((count + 1.0) / (len(perm_vals) + 1.0))
        row["p_perm_greater"] = p_perm
        row["n_permutations"] = int(len(perm_vals))

    pairwise_rows = []
    for family in ["two_stage_binary", "two_stage_bottleneck"]:
        for one_template, one_mean, one_n in [
            ("one_stage_bottleneck", bottleneck_mean, bottleneck_n),
            ("one_stage_closeness", closeness_mean, closeness_n),
        ]:
            observed_delta = np.nan
            if np.isfinite(shared[family]["mean"]) and np.isfinite(one_mean):
                observed_delta = float(shared[family]["mean"]) - float(one_mean)
            perm_vals = perm_diffs[(family, one_template)]
            p_perm = np.nan
            if len(perm_vals) > 0 and np.isfinite(observed_delta):
                count = 0
                for val in perm_vals:
                    if val >= observed_delta:
                        count += 1
                p_perm = float((count + 1.0) / (len(perm_vals) + 1.0))
            pairwise_rows.append(
                {
                    "classifier": classifier,
                    "window": window,
                    "include_diagonal": bool(include_diagonal),
                    "stage_family": family,
                    "one_stage_template": one_template,
                    "shared_best_split_day": shared[family]["split_day"],
                    "mean_diff_stage_minus_one_stage": observed_delta,
                    "n_subjects": int(min(shared[family]["n"], one_n)),
                    "p_perm_stage_greater_one_stage": p_perm,
                    "n_permutations": int(len(perm_vals)),
                }
            )
    return score_rows, summary_rows, pairwise_rows


def mvpa_rows(output_dir):
    frames = []
    for window in ["early", "late"]:
        path = (
            output_dir
            / f"mvpa_stim_locked_cat_{window}_window_transfer_subject_pairs.csv"
        )
        frames.append(require_csv(path))
    d = pd.concat(frames, ignore_index=True)
    d = d[d["classifier"].isin(CLASSIFIERS)].copy()
    d = d[d["fit_status"].isin(["transfer", "cv"])].copy()
    if d.empty:
        raise ValueError("No completed MVPA transfer rows found")
    return d


def collect_results(d):
    rng = np.random.default_rng(RANDOM_STATE)
    score_rows = []
    summary_rows = []
    pairwise_rows = []
    for include_diagonal in [True, False]:
        for key, g in d.groupby(["classifier", "window"]):
            print(
                f"[MVPA transfer models] {key}, diagonal={include_diagonal}",
                flush=True,
            )
            scores, summary, pairwise = condition_results(
                key,
                g,
                rng,
                include_diagonal,
            )
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


def group_matrices(d, output_dir):
    rows = []
    for key, g in d.groupby(["classifier", "window"]):
        classifier, window = key
        for train_day in DAYS:
            for test_day in DAYS:
                d_pair = g[
                    (g["train_day"] == train_day) & (g["test_day"] == test_day)
                ]
                vals = d_pair["auc"].to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                val = np.nan
                if len(vals) > 0:
                    val = float(np.mean(vals))
                rows.append(
                    {
                        "classifier": classifier,
                        "window": window,
                        "train_day": train_day,
                        "test_day": test_day,
                        "auc_mean": val,
                        "n_values": int(len(vals)),
                    }
                )
    path = output_dir / "mvpa_transfer_model_compare_group_matrices.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def write_model_correlations(output_dir):
    rows = []
    pairs = pair_rows(True)
    models = [
        {"template": "one_stage_bottleneck", "split_day": np.nan},
        {"template": "one_stage_closeness", "split_day": np.nan},
    ]
    for template in ["two_stage_binary", "two_stage_bottleneck"]:
        for split_day in [1, 2, 3, 4]:
            models.append({"template": template, "split_day": split_day})
    vectors = {}
    labels = []
    for spec in models:
        label = spec["template"]
        split_arg = None
        if np.isfinite(spec["split_day"]):
            split_arg = int(spec["split_day"])
            label = f"{label}_D{split_arg}"
        labels.append(label)
        vectors[label] = template_vector(
            pairs,
            spec["template"],
            split_day=split_arg,
        )
    for label_i in labels:
        for label_j in labels:
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
    path = output_dir / "mvpa_transfer_model_compare_model_correlations.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_mvpa_transfer_model_compare(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    d = mvpa_rows(output_dir)
    scores, summary, pairwise = collect_results(d)
    scores_path = output_dir / "mvpa_transfer_model_compare_subject_scores.csv"
    summary_path = output_dir / "mvpa_transfer_model_compare_summary.csv"
    pairwise_path = output_dir / "mvpa_transfer_model_compare_pairwise.csv"
    scores.to_csv(scores_path, index=False)
    summary.to_csv(summary_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)
    group_path = group_matrices(d, output_dir)
    corr_path = write_model_correlations(output_dir)
    print(f"[MVPA transfer models] wrote {scores_path}", flush=True)
    print(f"[MVPA transfer models] wrote {summary_path}", flush=True)
    print(f"[MVPA transfer models] wrote {pairwise_path}", flush=True)
    print(f"[MVPA transfer models] wrote {group_path}", flush=True)
    print(f"[MVPA transfer models] wrote {corr_path}", flush=True)
    return {
        "subject_scores": scores_path,
        "summary": summary_path,
        "pairwise": pairwise_path,
        "group_matrices": group_path,
        "model_correlations": corr_path,
    }


if __name__ == "__main__":
    run_mvpa_transfer_model_compare()
