#!/usr/bin/env python3
"""Shared boundary-distance helpers."""

from __future__ import annotations

import os
import re
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/xdg-cache")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "output"


def load_behaviour_with_boundary(project_dir: Path | str = PROJECT_DIR):
    project_dir = Path(project_dir)
    beh_dir = project_dir / "Behavioural"
    beh_re = re.compile(r"^sub_(\d+)_day_(\d+)_data\.csv$")
    rows = []
    for path in sorted(beh_dir.glob("*.csv")):
        m = beh_re.match(path.name)
        if m is None:
            continue
        d = pd.read_csv(path)
        subject = int(m.group(1))
        day_code = int(m.group(2))
        day = day_code // 100
        d["subject"] = subject
        d["day_code"] = day_code
        d["day"] = day
        d["beh_file"] = path.name
        rows.append(d)
    if not rows:
        raise FileNotFoundError(f"No behavioural CSV files found in {beh_dir}")
    beh = pd.concat(rows, ignore_index=True)
    beh["cat_binary"] = (beh["cat"].astype(str) == "B").astype(int)
    clf = LogisticRegression(solver="lbfgs", C=1e6, max_iter=1000)
    X = beh[["xt", "yt"]].to_numpy(dtype=float)
    y = beh["cat_binary"].to_numpy(dtype=int)
    clf.fit(X, y)
    w = clf.coef_[0].astype(float)
    b = float(clf.intercept_[0])
    norm = float(np.linalg.norm(w))
    decision_distance = (X @ w + b) / norm
    correct_side_distance = np.where(y == 1, decision_distance, -decision_distance)
    beh["boundary_distance"] = correct_side_distance
    beh["boundary_distance_abs"] = np.abs(correct_side_distance)
    beh["boundary_decision_distance"] = decision_distance
    beh["accuracy"] = (beh["fb"].astype(str).str.lower() == "correct").astype(float)
    beh["rt_sec"] = pd.to_numeric(beh["rt"], errors="coerce") / 1000.0
    boundary = {
        "coef_xt": float(w[0]),
        "coef_yt": float(w[1]),
        "intercept": b,
        "norm": norm,
        "classes": "A=0,B=1",
    }
    return beh, boundary
