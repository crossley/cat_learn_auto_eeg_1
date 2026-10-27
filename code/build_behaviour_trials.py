#!/usr/bin/env python3
"""One row per trial of the category task, lab and home, in training order.

The lab files (Behavioural/) are the five weekly EEG sessions. The home
files (at_home_data/) are the sessions in between, four a week, plus a
dual-task day (22) and two button-switch days (23, 24) at the end. The
home files carry no dates, so sessions are ordered by the home day count,
and lab day k + 1 is placed after the 4k-th home training session, as
the protocol had it. Subject and day always come from the filename: the
columns inside are wrong in several files (see DATA_QUESTIONS.md). A
trial is its row position in the file, which is what the EEG epoch
metadata calls trial_index, so the two tables join on it.

The quirks found in the home files are handled by the three tables
below, taken from the project lead's curation and checked against the
files: EXCLUDED files are not read, REASSIGNED files were saved under
the wrong subject, and SKIP_BAD_LINES files have one malformed line.

Writes output/behaviour_trials.csv, one row per trial, with the column
names used by the later experiments, and output/behaviour_sessions.csv,
one row per session, for checking the order.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
LAB_DIR = PROJECT_DIR / "Behavioural"  # sub_<sub>_day_<D><sub>_data.csv, D = lab day 1-5
HOME_DIR = PROJECT_DIR / "at_home_data"  # <sub>/sub_<sub>_day_<n>_data.csv, n = home day count
TRIALS_PATH = PROJECT_DIR / "output" / "behaviour_trials.csv"
SESSIONS_PATH = PROJECT_DIR / "output" / "behaviour_sessions.csv"

LAB_TRIALS = 550
HOME_SESSIONS_PER_WEEK = 4  # lab day k + 1 follows the 4k-th home training session
DUAL_TASK_DAY, BUTTON_SWITCH_DAYS = 22, (23, 24)

EXCLUDED = {  # home file: reason
    "002/sub_002_day_23_data.csv": "extra home day",
    "008/sub_008_day_18_data.csv": "extra home day (EEG days not counted)",
    "008/sub_008_day_19_data.csv": "extra home day",
    "008/sub_008_day_20_data.csv": "extra home day",
    "008/sub_008_day_21_data.csv": "extra home day",
    "008/sub_008_day_22_data.csv": "dual task excluded (extra training days before it)",
    "008/sub_008_day_23_data.csv": "button switch excluded",
    "008/sub_008_day_24_data.csv": "button switch excluded",
    "015/sub_015_day_22_data.csv": "dual task excluded (two home days missed)",
    "015/sub_015_day_23_data.csv": "button switch excluded",
    "015/sub_015_day_24_data.csv": "button switch excluded",
    "019/sub_019_day_13_data.csv": "computer froze after 84 trials",
    "019/sub_019_day_18_data.csv": "extra home day",
    "019/sub_019_day_22_data.csv": "dual task excluded (extra training days before it)",
    "019/sub_019_day_23_data.csv": "button switch excluded",
    "019/sub_019_day_24_data.csv": "button switch excluded",
    "_unresolved/sub_001_day_22_data_copy2.csv": "found in participant 2's folder under subject 1: owner unknown",
}
REASSIGNED = {  # home file saved under the program's default subject 1: (subject, home day)
    "_unresolved/sub_001_day_22_data_copy1.csv": (6, 22),
    "_unresolved/sub_001_day_23_data_copy1.csv": (6, 23),
    "_unresolved/sub_001_day_24_data_copy1.csv": (6, 24),
    "_unresolved/sub_001_day_22_data_copy3.csv": (1, 22),
    "_unresolved/sub_001_day_23_data_copy2.csv": (1, 23),
    "_unresolved/sub_001_day_24_data_copy2.csv": (1, 24),
}
SKIP_BAD_LINES = {"at_home_data/015/sub_015_day_07_data.csv"}  # line 62 has an extra field; that trial is lost
# A trial is its row position, as in the EEG metadata (trial_index); the file's own
# trial column agrees everywhere except in these files.
TRIAL_COLUMN_GLITCHES = {"Behavioural/sub_003_day_403_data.csv": "row 401 is numbered 403",
                         "at_home_data/015/sub_015_day_07_data.csv": "trial 61 lost to the malformed line"}

COLUMNS = ["subject", "session_order", "session_kind", "lab_day", "home_day", "session_start", "trial",
           "training_trial", "phase", "probe_condition", "cat", "x", "y", "xt", "yt", "resp", "resp_key", "fb",
           "rt", "file"]  # session_start is empty here: the Exp1 files carry no timestamps


def read_session(path, subject, kind, lab_day, home_day):
    """One file as trial rows; subject and day from the caller, never from the columns."""
    rel = str(path.relative_to(PROJECT_DIR))
    trials = pd.read_csv(path, on_bad_lines="skip" if rel in SKIP_BAD_LINES else "error")
    trials = trials.dropna(subset=["trial"])  # sub_019_day_24 ends with blank comma lines
    assert (trials.trial.to_numpy() == np.arange(len(trials))).all() or rel in TRIAL_COLUMN_GLITCHES, \
        f"{rel}: trial numbers do not run 0..{len(trials) - 1}"
    if kind == "lab":
        assert len(trials) == LAB_TRIALS, f"{rel}: {len(trials)} trials"
    rows = pd.DataFrame({"subject": subject, "session_kind": kind, "lab_day": lab_day, "home_day": home_day,
                         "session_start": np.nan, "trial": np.arange(len(trials)), "phase": "train",
                         "probe_condition": np.nan,
                         "cat": trials.cat, "x": trials.x, "y": trials.y, "xt": trials.xt, "yt": trials.yt,
                         "resp": trials.resp, "resp_key": trials.resp_key if "resp_key" in trials else np.nan,
                         "fb": trials.fb, "rt": trials.rt, "file": rel})
    return rows


def home_files(subject):
    """(path, home day) of the subject's usable home files, in day order."""
    files = []
    for path in sorted(HOME_DIR.glob(f"{subject:03d}/sub_{subject:03d}_day_*_data.csv")):
        if str(path.relative_to(HOME_DIR)) not in EXCLUDED:
            files.append((path, int(re.search(r"_day_(\d+)_", path.name).group(1))))
    for rel, (owner, day) in REASSIGNED.items():
        if owner == subject:
            files.append((HOME_DIR / rel, day))
    return sorted(files, key=lambda f: f[1])


def session_order(lab_days, home_days):
    """Interleave lab day k + 1 after the 4k-th home training session; dual task and switch days last."""
    training = [("home", d) for d in home_days if d < DUAL_TASK_DAY]
    order = [("lab", 1)] if 1 in lab_days else []
    taken = 0
    for k, lab_day in enumerate(sorted(d for d in lab_days if d > 1), start=1):
        order += training[taken:k * HOME_SESSIONS_PER_WEEK]
        taken = min(k * HOME_SESSIONS_PER_WEEK, len(training))
        order.append(("lab", lab_day))
    order += training[taken:]
    order += [("home", d) for d in home_days if d >= DUAL_TASK_DAY]
    return order


if __name__ == "__main__":
    lab = {}  # (subject, lab day) -> path
    for path in sorted(LAB_DIR.glob("sub_*_day_*_data.csv")):
        subject, code = map(int, re.match(r"sub_(\d+)_day_(\d+)_data", path.name).groups())
        assert code % 100 == subject, f"{path.name}: day code does not end in the subject number"
        lab[(subject, code // 100)] = path
    assert sum(str(p.relative_to(HOME_DIR)) in EXCLUDED or str(p.relative_to(HOME_DIR)) in REASSIGNED
               for p in HOME_DIR.rglob("*.csv")) == len(EXCLUDED) + len(REASSIGNED), "a listed file is missing"

    trials = []
    for subject in sorted({s for s, _ in lab}):
        lab_days = sorted(d for s, d in lab if s == subject)
        homes = dict(home_files(subject))
        home_days = sorted(homes.values())
        homes = {d: p for p, d in homes.items()}
        for order, (where, day) in enumerate(session_order(lab_days, home_days), start=1):
            if where == "lab":
                rows = read_session(lab[(subject, day)], subject, "lab", day, np.nan)
            else:
                kind = "dual_task" if day == DUAL_TASK_DAY else "button_switch" if day in BUTTON_SWITCH_DAYS else "home"
                rows = read_session(homes[day], subject, kind, np.nan, day)
            trials.append(rows.assign(session_order=order))
    trials = pd.concat(trials, ignore_index=True)
    is_training = trials.session_kind.isin(["lab", "home"])
    trials["training_trial"] = np.nan
    trials.loc[is_training, "training_trial"] = trials[is_training].groupby("subject").cumcount()
    trials = trials[COLUMNS]
    TRIALS_PATH.parent.mkdir(exist_ok=True)
    trials.to_csv(TRIALS_PATH, index=False)

    sessions = trials.groupby(["subject", "session_order", "session_kind", "file"], dropna=False).agg(
        lab_day=("lab_day", "first"), home_day=("home_day", "first"), n_trials=("trial", "size"),
        accuracy=("fb", lambda fb: round((fb == "Correct").mean(), 3)),
        median_rt=("rt", "median")).reset_index().sort_values(["subject", "session_order"])
    sessions.to_csv(SESSIONS_PATH, index=False)
    print(f"{len(trials)} trials, {len(sessions)} sessions, {trials.subject.nunique()} subjects")
    print(sessions.groupby("session_kind").size().to_string())
    print(sessions.groupby("subject").agg(sessions=("session_order", "size"), training=("session_kind", lambda k: k.isin(["lab", "home"]).sum()),
                                          order=("session_kind", lambda k: "".join(x[0] for x in k))).to_string())
