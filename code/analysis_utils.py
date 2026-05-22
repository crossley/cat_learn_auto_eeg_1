#!/usr/bin/env python3
"""Shared statistical and parallel-execution utilities."""

from __future__ import annotations

import os

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")

from joblib import Parallel, delayed

try:
    from threadpoolctl import threadpool_limits
except Exception:
    threadpool_limits = None


def parallel_collect(func, items, n_workers):
    def iter_delayed_items():
        for item in items:
            yield delayed(func)(item)

    if n_workers == 1:
        results = []
        for item in items:
            results.append(func(item))
        return results
    if threadpool_limits is None:
        return Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            iter_delayed_items()
        )
    with threadpool_limits(limits=1):
        return Parallel(n_jobs=n_workers, backend="loky", verbose=0)(
            iter_delayed_items()
        )


def model_term_summary(model, term):
    names = list(model.model.exog_names)
    idx = names.index(term)
    ci = model.conf_int()
    if hasattr(ci, "iloc"):
        lo, hi = ci.iloc[idx]
    else:
        lo, hi = ci[idx]
    return {
        "term": term,
        "estimate": float(
            model.params.iloc[idx] if hasattr(model.params, "iloc") else model.params[idx]
        ),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p_value": float(
            model.pvalues.iloc[idx] if hasattr(model.pvalues, "iloc") else model.pvalues[idx]
        ),
    }
