#!/usr/bin/env python3
"""Template-similarity tests for 5x5 day-structure matrices."""

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
            f"Missing template-similarity input: {path}. "
            "Run model_compare_5x5_analysis.py first."
        )
    d = pd.read_csv(path)
    if d.empty:
        raise ValueError(f"Empty template-similarity input: {path}")
    return d


def zscore_vector(vals):
    vals = np.asarray(vals, dtype=float)
    good = np.isfinite(vals)
    if int(np.sum(good)) < 3:
        return None
    out = np.full(vals.shape, np.nan, dtype=float)
    finite = vals[good]
    sd = float(np.std(finite))
    if sd <= np.finfo(float).eps:
        return None
    out[good] = (finite - float(np.mean(finite))) / sd
    return out


def vector_similarity(observed, template):
    obs_z = zscore_vector(observed)
    template_z = zscore_vector(template)
    if obs_z is None or template_z is None:
        return np.nan
    good = np.isfinite(obs_z) & np.isfinite(template_z)
    if int(np.sum(good)) < 3:
        return np.nan
    obs = obs_z[good]
    pred = template_z[good]
    denom = float(np.sqrt(np.sum(obs**2) * np.sum(pred**2)))
    if denom <= np.finfo(float).eps:
        return np.nan
    return float(np.sum(obs * pred) / denom)


def pair_rows_from_subject(d_subject):
    rows = []
    values = []
    for train_day in DAYS:
        for test_day in DAYS:
            if train_day == test_day:
                continue
            d_pair = d_subject[
                (d_subject["train_day"] == train_day)
                & (d_subject["test_day"] == test_day)
            ]
            if d_pair.empty:
                continue
            vals = d_pair["value"].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            rows.append({"train_day": train_day, "test_day": test_day})
            values.append(float(np.mean(vals)))
    if len(values) < 6:
        raise ValueError(
            "Too few off-diagonal values for template similarity: "
            f"n={len(values)}"
        )
    return rows, np.asarray(values, dtype=float)


def remap_day(day, day_map):
    if day_map is None:
        return day
    return int(day_map[int(day)])


def template_value(value_kind, template, train_day, test_day, split_day, day_map):
    train_day = remap_day(train_day, day_map)
    test_day = remap_day(test_day, day_map)
    if template == "one_stage":
        if value_kind == "similarity":
            return float((min(train_day, test_day) - 1.0) / 4.0)
        if value_kind == "distance":
            return float(abs(train_day - test_day) / 4.0)
    if template == "two_stage":
        if split_day is None:
            raise ValueError("two_stage template requires split_day")
        train_late = train_day > split_day
        test_late = test_day > split_day
        cross_stage = 0.0
        if train_late != test_late:
            cross_stage = 1.0
        if value_kind == "similarity":
            return float(1.0 - cross_stage)
        if value_kind == "distance":
            return float(cross_stage)
    raise ValueError(f"Unknown template/value_kind: {template}, {value_kind}")


def template_vector(pair_rows, value_kind, template, split_day=None, day_map=None):
    vals = []
    for row in pair_rows:
        vals.append(
            template_value(
                value_kind,
                template,
                int(row["train_day"]),
                int(row["test_day"]),
                split_day,
                day_map,
            )
        )
    return np.asarray(vals, dtype=float)


def permuted_day_map(rng):
    shuffled = np.asarray(DAYS, dtype=int).copy()
    rng.shuffle(shuffled)
    out = {}
    for idx, day in enumerate(DAYS):
        out[day] = int(shuffled[idx])
    return out


def best_two_stage_similarity(values, pair_rows, value_kind, day_map=None):
    best_similarity = np.nan
    best_split = np.nan
    for split_day in [1, 2, 3, 4]:
        template = template_vector(
            pair_rows,
            value_kind,
            "two_stage",
            split_day=split_day,
            day_map=day_map,
        )
        sim = vector_similarity(values, template)
        if not np.isfinite(sim):
            continue
        if not np.isfinite(best_similarity) or sim > best_similarity:
            best_similarity = float(sim)
            best_split = int(split_day)
    return best_similarity, best_split


def subject_template_scores(d_subject):
    pair_rows, values = pair_rows_from_subject(d_subject)
    value_kind = str(d_subject["value_kind"].iloc[0])
    rows = []
    one_template = template_vector(pair_rows, value_kind, "one_stage")
    rows.append(
        {
            "template": "one_stage",
            "split_day": np.nan,
            "similarity": vector_similarity(values, one_template),
        }
    )
    for split_day in [1, 2, 3, 4]:
        two_template = template_vector(
            pair_rows,
            value_kind,
            "two_stage",
            split_day=split_day,
        )
        rows.append(
            {
                "template": "two_stage",
                "split_day": split_day,
                "similarity": vector_similarity(values, two_template),
            }
        )
    best_similarity, best_split = best_two_stage_similarity(
        values,
        pair_rows,
        value_kind,
    )
    rows.append(
        {
            "template": "two_stage_best",
            "split_day": best_split,
            "similarity": best_similarity,
        }
    )
    return rows, pair_rows, values, value_kind


def permutation_score(template, split_day, values, pair_rows, value_kind, rng):
    day_map = permuted_day_map(rng)
    if template == "two_stage_best":
        sim, _split = best_two_stage_similarity(
            values,
            pair_rows,
            value_kind,
            day_map=day_map,
        )
        return sim
    template_vec = template_vector(
        pair_rows,
        value_kind,
        template,
        split_day=split_day,
        day_map=day_map,
    )
    return vector_similarity(values, template_vec)


def permutation_subject_scores(score_rows, d_subject, rng):
    _subject_scores, pair_rows, values, value_kind = subject_template_scores(d_subject)
    rows = []
    for perm_i in range(N_PERMUTATIONS):
        for score_row in score_rows:
            template = score_row["template"]
            split_day = score_row["split_day"]
            split_arg = None
            if np.isfinite(split_day):
                split_arg = int(split_day)
            sim = permutation_score(
                template,
                split_arg,
                values,
                pair_rows,
                value_kind,
                rng,
            )
            rows.append(
                {
                    "perm_i": perm_i,
                    "template": template,
                    "split_day": split_day,
                    "similarity": sim,
                }
            )
    return rows


def collect_subject_scores(empirical_df):
    rng = np.random.default_rng(RANDOM_STATE)
    score_rows = []
    perm_rows = []
    group_cols = ["modality", "measure", "window", "value_kind", "subject"]
    for key, g in empirical_df.groupby(group_cols):
        modality, measure, window, value_kind, subject = key
        subject_scores, _pair_rows, _values, _kind = subject_template_scores(g)
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
        subject_perm = permutation_subject_scores(subject_scores, g, rng)
        for perm in subject_perm:
            row = {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "subject": int(subject),
            }
            for col, val in perm.items():
                row[col] = val
            perm_rows.append(row)
    return pd.DataFrame(score_rows), pd.DataFrame(perm_rows)


def summarize_template_scores(scores_df, perm_df):
    rows = []
    scores_df = scores_df.copy()
    score_split_ids = []
    for _, row in scores_df.iterrows():
        split_day = row["split_day"]
        split_id = "none"
        if row["template"] == "two_stage_best":
            split_id = "best"
        elif np.isfinite(split_day):
            split_id = str(int(split_day))
        score_split_ids.append(split_id)
    scores_df["split_id"] = score_split_ids
    perm_df = perm_df.copy()
    perm_split_ids = []
    for _, row in perm_df.iterrows():
        split_day = row["split_day"]
        split_id = "none"
        if row["template"] == "two_stage_best":
            split_id = "best"
        elif np.isfinite(split_day):
            split_id = str(int(split_day))
        perm_split_ids.append(split_id)
    perm_df["split_id"] = perm_split_ids
    group_cols = [
        "modality",
        "measure",
        "window",
        "value_kind",
        "template",
        "split_id",
    ]
    for key, g in scores_df.groupby(group_cols):
        modality, measure, window, value_kind, template, split_id = key
        vals = g["similarity"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        observed_mean = float(np.mean(vals))
        sem = np.nan
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        d_perm = perm_df[
            (perm_df["modality"] == modality)
            & (perm_df["measure"] == measure)
            & (perm_df["window"] == window)
            & (perm_df["value_kind"] == value_kind)
            & (perm_df["template"] == template)
            & (perm_df["split_id"] == split_id)
        ]
        perm_means = []
        for _perm_i, gp in d_perm.groupby("perm_i"):
            perm_vals = gp["similarity"].to_numpy(dtype=float)
            perm_vals = perm_vals[np.isfinite(perm_vals)]
            if len(perm_vals) > 0:
                perm_means.append(float(np.mean(perm_vals)))
        p_perm = np.nan
        if len(perm_means) > 0:
            count = 0
            for val in perm_means:
                if val >= observed_mean:
                    count += 1
            p_perm = float((count + 1.0) / (len(perm_means) + 1.0))
        split_day = np.nan
        split_vals = g["split_day"].to_numpy(dtype=float)
        split_vals = split_vals[np.isfinite(split_vals)]
        if len(split_vals) > 0:
            split_day = float(pd.Series(split_vals).mode().iloc[0])
        rows.append(
            {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "template": template,
                "split_day": split_day,
                "mean_similarity": observed_mean,
                "sem_similarity": sem,
                "n_subjects": int(len(vals)),
                "p_perm_greater": p_perm,
                "n_permutations": int(len(perm_means)),
            }
        )
    return pd.DataFrame(rows)


def summarize_pairwise(scores_df, perm_df):
    rows = []
    group_cols = ["modality", "measure", "window", "value_kind", "subject"]
    observed_rows = []
    for key, g in scores_df.groupby(group_cols):
        modality, measure, window, value_kind, subject = key
        d_one = g[g["template"] == "one_stage"]
        d_two = g[g["template"] == "two_stage_best"]
        if d_one.empty or d_two.empty:
            continue
        observed_rows.append(
            {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "subject": int(subject),
                "diff_one_minus_two_best": (
                    float(d_one["similarity"].iloc[0])
                    - float(d_two["similarity"].iloc[0])
                ),
            }
        )
    observed_df = pd.DataFrame(observed_rows)
    condition_cols = ["modality", "measure", "window", "value_kind"]
    for key, g in observed_df.groupby(condition_cols):
        modality, measure, window, value_kind = key
        vals = g["diff_one_minus_two_best"].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        observed_mean = float(np.mean(vals))
        sem = np.nan
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        d_perm = perm_df[
            (perm_df["modality"] == modality)
            & (perm_df["measure"] == measure)
            & (perm_df["window"] == window)
            & (perm_df["value_kind"] == value_kind)
        ]
        perm_means = []
        for _perm_i, gp in d_perm.groupby("perm_i"):
            diffs = []
            for subject, gs in gp.groupby("subject"):
                d_one = gs[gs["template"] == "one_stage"]
                d_two = gs[gs["template"] == "two_stage_best"]
                if d_one.empty or d_two.empty:
                    continue
                diffs.append(
                    float(d_one["similarity"].iloc[0])
                    - float(d_two["similarity"].iloc[0])
                )
            if len(diffs) > 0:
                perm_means.append(float(np.mean(diffs)))
        p_two_sided = np.nan
        if len(perm_means) > 0:
            count = 0
            observed_abs = abs(observed_mean)
            for val in perm_means:
                if abs(val) >= observed_abs:
                    count += 1
            p_two_sided = float((count + 1.0) / (len(perm_means) + 1.0))
        rows.append(
            {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "mean_diff_one_minus_two_best": observed_mean,
                "sem_diff": sem,
                "n_subjects": int(len(vals)),
                "p_perm_two_sided": p_two_sided,
                "n_permutations": int(len(perm_means)),
            }
        )
    return pd.DataFrame(rows), observed_df


def template_specs():
    rows = []
    rows.append({"template": "one_stage", "split_day": np.nan, "label": "one_stage"})
    for split_day in [1, 2, 3, 4]:
        rows.append(
            {
                "template": "two_stage",
                "split_day": split_day,
                "label": f"two_stage_{split_day}",
            }
        )
    rows.append(
        {
            "template": "two_stage_best",
            "split_day": np.nan,
            "label": "two_stage_best",
        }
    )
    return rows


def subject_payloads(d_condition):
    payloads = []
    for subject, g_subject in d_condition.groupby("subject"):
        subject_scores, pair_rows, values, value_kind = subject_template_scores(
            g_subject
        )
        payloads.append(
            {
                "subject": int(subject),
                "scores": subject_scores,
                "pair_rows": pair_rows,
                "values": values,
                "value_kind": value_kind,
            }
        )
    if len(payloads) == 0:
        raise ValueError("No subject payloads for template comparison")
    return payloads


def score_lookup(subject_scores):
    out = {}
    for row in subject_scores:
        template = row["template"]
        split_day = row["split_day"]
        split_id = "none"
        if template == "two_stage_best":
            split_id = "best"
        elif np.isfinite(split_day):
            split_id = str(int(split_day))
        key = (template, split_id)
        out[key] = float(row["similarity"])
    return out


def split_mode(subject_scores):
    vals = []
    for row in subject_scores:
        if row["template"] != "two_stage_best":
            continue
        split_day = row["split_day"]
        if np.isfinite(split_day):
            vals.append(float(split_day))
    if len(vals) == 0:
        return np.nan
    return float(pd.Series(vals).mode().iloc[0])


def condition_template_results(condition_key, d_condition, rng):
    modality, measure, window, value_kind = condition_key
    payloads = subject_payloads(d_condition)
    specs = template_specs()
    score_rows = []
    observed_by_label = {}
    for spec in specs:
        observed_by_label[spec["label"]] = []

    for payload in payloads:
        lookup = score_lookup(payload["scores"])
        for score in payload["scores"]:
            row = {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "subject": payload["subject"],
            }
            for col, val in score.items():
                row[col] = val
            score_rows.append(row)
        for spec in specs:
            template = spec["template"]
            split_id = "none"
            if template == "two_stage_best":
                split_id = "best"
            elif np.isfinite(spec["split_day"]):
                split_id = str(int(spec["split_day"]))
            key = (template, split_id)
            if key in lookup and np.isfinite(lookup[key]):
                observed_by_label[spec["label"]].append(float(lookup[key]))

    perm_values = {}
    for spec in specs:
        perm_values[spec["label"]] = []
    perm_pairwise = []
    for _perm_i in range(N_PERMUTATIONS):
        accum = {}
        counts = {}
        for spec in specs:
            accum[spec["label"]] = 0.0
            counts[spec["label"]] = 0
        pair_diffs = []
        for payload in payloads:
            one_sim = np.nan
            best_sim = np.nan
            for spec in specs:
                split_arg = None
                if np.isfinite(spec["split_day"]):
                    split_arg = int(spec["split_day"])
                sim = permutation_score(
                    spec["template"],
                    split_arg,
                    payload["values"],
                    payload["pair_rows"],
                    payload["value_kind"],
                    rng,
                )
                if np.isfinite(sim):
                    accum[spec["label"]] += float(sim)
                    counts[spec["label"]] += 1
                if spec["label"] == "one_stage":
                    one_sim = sim
                if spec["label"] == "two_stage_best":
                    best_sim = sim
            if np.isfinite(one_sim) and np.isfinite(best_sim):
                pair_diffs.append(float(one_sim) - float(best_sim))
        for spec in specs:
            label = spec["label"]
            if counts[label] > 0:
                perm_values[label].append(accum[label] / float(counts[label]))
        if len(pair_diffs) > 0:
            perm_pairwise.append(float(np.mean(pair_diffs)))

    summary_rows = []
    for spec in specs:
        label = spec["label"]
        vals = np.asarray(observed_by_label[label], dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        observed_mean = float(np.mean(vals))
        sem = np.nan
        if len(vals) > 1:
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals)))
        perm_means = perm_values[label]
        p_perm = np.nan
        if len(perm_means) > 0:
            count = 0
            for val in perm_means:
                if val >= observed_mean:
                    count += 1
            p_perm = float((count + 1.0) / (len(perm_means) + 1.0))
        split_day = spec["split_day"]
        if spec["template"] == "two_stage_best":
            all_scores = []
            for payload in payloads:
                for score in payload["scores"]:
                    all_scores.append(score)
            split_day = split_mode(all_scores)
        summary_rows.append(
            {
                "modality": modality,
                "measure": measure,
                "window": window,
                "value_kind": value_kind,
                "template": spec["template"],
                "split_day": split_day,
                "mean_similarity": observed_mean,
                "sem_similarity": sem,
                "n_subjects": int(len(vals)),
                "p_perm_greater": p_perm,
                "n_permutations": int(len(perm_means)),
            }
        )

    pairwise_rows = []
    one_vals = np.asarray(observed_by_label["one_stage"], dtype=float)
    two_vals = np.asarray(observed_by_label["two_stage_best"], dtype=float)
    n_pair = min(len(one_vals), len(two_vals))
    if n_pair > 0:
        diffs = one_vals[:n_pair] - two_vals[:n_pair]
        diffs = diffs[np.isfinite(diffs)]
        if len(diffs) > 0:
            observed_diff = float(np.mean(diffs))
            sem = np.nan
            if len(diffs) > 1:
                sem = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
            p_two = np.nan
            if len(perm_pairwise) > 0:
                count = 0
                for val in perm_pairwise:
                    if abs(val) >= abs(observed_diff):
                        count += 1
                p_two = float((count + 1.0) / (len(perm_pairwise) + 1.0))
            pairwise_rows.append(
                {
                    "modality": modality,
                    "measure": measure,
                    "window": window,
                    "value_kind": value_kind,
                    "mean_diff_one_minus_two_best": observed_diff,
                    "sem_diff": sem,
                    "n_subjects": int(len(diffs)),
                    "p_perm_two_sided": p_two,
                    "n_permutations": int(len(perm_pairwise)),
                }
            )
    return score_rows, summary_rows, pairwise_rows


def collect_template_results(empirical_df):
    rng = np.random.default_rng(RANDOM_STATE)
    score_rows = []
    summary_rows = []
    pairwise_rows = []
    group_cols = ["modality", "measure", "window", "value_kind"]
    for key, g in empirical_df.groupby(group_cols):
        print(f"[5x5 template] {key}", flush=True)
        scores, summary, pairwise = condition_template_results(key, g, rng)
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


def run_model_compare_5x5_template(output_dir=OUTPUT_DIR):
    output_dir = Path(output_dir)
    empirical_path = output_dir / "model_compare_5x5_empirical_values.csv"
    empirical_df = require_csv(empirical_path)
    print("[5x5 template] scoring subject templates", flush=True)
    scores_df, summary_df, pairwise_df = collect_template_results(empirical_df)

    scores_path = output_dir / "model_compare_5x5_template_subject_scores.csv"
    summary_path = output_dir / "model_compare_5x5_template_summary.csv"
    pairwise_path = output_dir / "model_compare_5x5_template_pairwise.csv"
    scores_df.to_csv(scores_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    pairwise_df.to_csv(pairwise_path, index=False)
    print(f"[5x5 template] wrote {scores_path}", flush=True)
    print(f"[5x5 template] wrote {summary_path}", flush=True)
    print(f"[5x5 template] wrote {pairwise_path}", flush=True)
    return {
        "subject_scores": scores_path,
        "summary": summary_path,
        "pairwise": pairwise_path,
    }


if __name__ == "__main__":
    run_model_compare_5x5_template()
