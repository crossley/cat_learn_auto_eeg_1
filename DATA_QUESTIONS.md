# Open questions about the raw data

Questions that only the people who ran the sessions can answer.
Each entry says what the files show and what decision is waiting
on the answer. Answer by editing the entry; the preprocessing
script then gets a one-line fact with this file as its source.

## P14, day 5: recording without a behaviour file (opened 2026-10-23)

`EEG/P14_D5.bdf` exists (recorded 2025-10-21, 39 minutes, a full
550-trial run of triggers) but `Behavioural/` holds P14 files for
days 1 to 4 only. The old pipeline put its epochs in
`EEG_epo_unmatched/`.

Question: was the day-5 behaviour file for P14 saved somewhere
else, or lost?

Decision waiting: `P14_D5.bdf` is skipped by the trigger audit
(status "unusable") until a behaviour file is found. Without one
the trials cannot be labelled and the recording cannot be used.

## Six recordings with a numeric suffix in the name (opened 2026-10-23)

`P11_D2_5.bdf`, `P14_D4_19.bdf`, `P15_D3_10.bdf`, `P1_D3_9.bdf`,
`P1_D4_14.bdf` and `P4_D2_9.bdf` are the only recordings for those
sessions; no un-suffixed file exists beside them. Each contains one
START trigger and a full run whose trigger codes match the
behaviour file, so they are processed as ordinary sessions.

Question: what does the suffix mean (a BioSemi file counter, a
re-saved file, something else)? Nothing in the data depends on the
answer; it is recorded so nobody wonders again.

## P3, day 4: duplicated trial number (opened 2026-10-23)

`Behavioural/sub_003_day_403_data.csv` has 550 rows, but the row in
position 401 carries trial number 403, so 403 appears twice and
401 never. The EEG triggers match the rows in file order (category,
response and feedback codes all agree for all 550 trials), so the
row position, not the `trial` column, is the trial identity. The
preprocessing metadata carries the position as `trial_index`.

Question: was this a logging glitch in the task, or was the file
edited?

Decision taken 2026-10-23: use rows in recorded order; never sort
by `trial`. Analyses that key on `trial` for this session should
use `trial_index` instead.

## Dropped triggers (no question, for the record)

The trigger audit (`code/find_missing_triggers.py`, output in
`output/trigger_audit.csv` and `output/missing_triggers.csv`) finds
stimulus triggers missing in 23 sessions (80 trials, 33 of them in
P17 and 27 in P18), feedback triggers missing in 7 trials (P1 d5,
P6 d1, P8 d1, P8 d3, P10 d2, P11 d3, P13 d5), and one ITI trigger
missing in each of P2 day 2, P12 day 3, P14 day 3 and P15 day 3. Trials without a stimulus or
feedback trigger are dropped from the corresponding epochs; the
missing ITI only affects trial counting and is handled in the audit.

## Sessions whose EOG electrodes recorded nothing usable (opened 2026-10-24)

A survey of EXG1 to EXG4 in all 95 recordings (1 Hz high-passed
standard deviation and the correlation between EXG1 and EXG2;
normal sessions give 10 to 40 uV and r around 0.4) finds six
sessions where the eye electrodes were not recording:

| session | what the channels show |
|---|---|
| P4 D3 | EXG1 to EXG4 identical, 2 mV |
| P4 D5 | EXG1 to EXG4 identical, 0.1 mV |
| P12 D1 | EXG1 to EXG4 identical, 3.4 mV |
| P17 D5 | EXG1 to EXG4 identical, 1 to 2 mV |
| P18 D3 | EXG2 at 11 mV (EXG1 normal) |
| P18 D4 | EXG1 at 11 mV, EXG2 to EXG4 identical, 3 mV |

Identical signals on four electrodes at mV amplitude means they
were not connected to the head; a single channel at 11 mV means a
dangling or dry electrode. No Exp2 recording shows either pattern.

Question: do the session notes say anything about the external
electrodes on these days?

Decision taken 2026-10-24: the preprocessing asserts that EXG1 and
EXG2 are live (each below 250 uV above 1 Hz, correlation below
0.9) and skips the session otherwise, so these six sessions have no
`EEG_epo_v2` file and appear as "skipped" in the QC table with the
reason. The 2026-10-23 files for P4 D3, P4 D5, P12 D1 and P17 D5,
in which the selection had removed non-ocular components and left
the blinks in, were deleted. Rescuing these sessions with an
EOG-free component classifier (ICLabel) is possible and was shown
to work on P4 D3, P4 D5, P12 D1 and P17 D5; it is left for later
if the sessions turn out to matter.

Correction 2026-10-24: the survey above was computed over whole
files. Restricted to the task interval (first to last trigger),
P12 D1's eye electrodes are normal (EXG1 28 uV, EXG2 38 uV, r
0.66); the millivolt garbage is a 40-second tail after the task.
The other five are dead during the task as well (P4 D5's four
channels sit at 110 to 120 uV with r above 0.9, so they were not on
the head). See the next entry.

## Recordings run 5 to 10 minutes before the task and up to 5 after (answered 2026-10-27)

Every Exp1 file starts well before the first task trigger (median
6.2 minutes, range 5.4 to 9.6) and continues after the last one
(median 55 s, up to 4.7 minutes; P12 D1 and P18 D4 end with tens of
millivolts, the cap coming off). Exp2 to 4 recordings start and end
within seconds of the task. PyPREP's bad-channel search and the ICA
fit currently see the whole file, so in Exp1 a sixth of what they
learn from is set-up time.

Question: was something recorded on purpose in those first minutes
(eyes-open/closed rest?), or is it set-up?

Answer 2026-10-27 (project lead): it is the resting-state recording,
made before each task session: 5 minutes, eyes closed, the first
5 minutes of the BDF file. So a rest pipeline takes raw.crop(0, 300)
of each task recording, not the old scripts' 300 s ending 5 s before
the first task event. Nothing in the files marks it: the task
program's trigger table has no rest code, 85 of 95 recordings carry
no pulse at all before the task's START, and the other 10 carry one
or two stray pulses (codes 0, 4 or 255) at arbitrary times.

Decision waiting: crop each recording to the trigger interval (with
a margin) before referencing and ICA, in all four experiments. In
Exp3 and Exp4 this rescues three sessions that PyPREP rejected or
the EOG check skipped because of a dead tail; here it rescues P12
D1 and makes the Exp1 reference and ICA comparable to the other
experiments.

## External electrode (EXG) placements (answered)

Same layout as the other experiments, confirmed by the lab head on
2026-10-22 and 2026-10-23 for all four experiments: EXG1 beside the
left eye (horizontal EOG), EXG2 below the left eye (vertical EOG),
EXG3 and EXG4 mastoids, EXG5 to EXG8 not connected. Preprocessing
removes ICA components correlating with VEOG (EXG2 minus Fp1) or
HEOG (EXG1 minus F8).

## P10 day 4: AutoReject dropped 48% of the response-locked epochs (deferred 2026-10-27)

`output/preprocess_resp_epochs_v2_qc.csv` shows P10 day 4 keeping 286
of 550 response epochs, while the task run of the same recording kept
99.9% of its stimulus and feedback epochs. The data are not bad: the
surviving epochs have a median peak-to-peak of 64 uV, well below a
normal session, so the rejection thresholds came out far too strict.
AutoReject is fitted on every tenth epoch (`clean()` in
`code/preprocess_task_epochs_v2.py`, shared by the three pipelines),
and on this session's 55-epoch subsample the Bayesian threshold search
landed badly. The same weakness exists in all three pipelines; this is
the one session where it bit.

Fixable, but computationally expensive: fitting on all epochs costs
roughly five times the AutoReject time per session, and because the
fit lives in the shared pipeline it means rerunning the task, RSA and
response-locked epochs of all four experiments to keep them
consistent. Decision 2026-10-27: accept the loss for now and treat
P10 day 4 as unavailable for response-locked analyses. To be revisited.

## Home-training behaviour files (imported from SharePoint 2026-10-27, curation rules answered)

Participants trained at home between the weekly lab sessions. The
files were not in this repo; they live on the lab SharePoint under
`projects/PACE/2025/semester_2/at_home_days/`, one folder per
participant named by the participant. They were copied into
`at_home_data/<subject>/` here, grouped by the subject number in the
filename, and the named folders were not kept. 347 files (340 in the
subject folders, 7 in `_unresolved/`), 19 participants, `sub_<sub>_day_<n>_data.csv` with `day` the home-session
count (1 to 24), not the lab day code (101 to 519). Columns are the
lab file's (subject, day, trial, cat, x, y, xt, yt, resp, rt, fb); 32
files from days 23 and 24 add `resp_key`, and 16 files from day 22 add
`ns_*` columns (a dual-task variant: digit size and side, cue,
congruency) and run 283 to 384 trials where the other home sessions
run 300. There are no timestamps in any home file, and SharePoint's
modification dates are a single upload batch, so a home session's date
is unknown; its position relative to the lab sessions has to come
from the day count and the lab schedule (`use_this_pace_day_tracker.xlsx`
on SharePoint, which names participants and was not copied).

Loaders must take the subject from the filename, not the `subject`
column: three files renamed by hand still carry subject 1 inside
(sub_015_day_01, sub_016_day_22, sub_019_day_01), and sub_003's day 19
and 20 files carry day 18 and 19. sub_015_day_07 has one malformed
line (12 fields instead of 11).

Four filenames occurred in more than one participant folder.
`sub_007_day_17` was identical in participant 7's and participant
17's folders and was kept once, under 007. The others differ and sit
in `at_home_data/_unresolved/` as `_copyN` files, numbered by the
folders' alphabetical order:

- `sub_001_day_22_data_copy1`, `sub_001_day_23_data_copy1`,
  `sub_001_day_24_data_copy1` came from participant 6's folder, whose
  own files stop at day 20. They are most likely participant 6's days
  22 to 24, run with the program's default subject number 1.
- `sub_001_day_22_data_copy3`, `sub_001_day_23_data_copy2`,
  `sub_001_day_24_data_copy2` came from participant 1's own folder.
- `sub_001_day_22_data_copy2` came from participant 2's folder, which
  already has a `sub_002_day_22` file, so its owner is unclear.

Answered 2026-10-27 from the project lead's earlier curation (an old
loader, `load_data_home`, whose rules were checked against these
files and hold):

- Day numbering: the first session was the EEG lab day. Most
  participants counted it and began home sessions at day 02; a few
  (5, 13, 15, 17, 19) began at day 01. Day 22 was a dual-task day
  (the `ns_*` columns), days 23 and 24 were button-switch days (the
  `resp_key` column, reversed response mapping). Those three are
  analysed apart from the training days.
- The copy1 files in `_unresolved/` are participant 6's days 22 to
  24, run under the default subject number 1. The copy3 (day 22) and
  copy2 (days 23, 24) files are participant 1's own. The day-22 copy2
  file from participant 2's folder is ambiguous and excluded.
- `sub_007_day_17` as found in participant 17's folder was attributed
  to participant 17 in the old curation (renamed sub_017_day_17,
  subject set to 17); participant 17 has no other file for days 16 to
  21. Its content says subject 7 and is byte-identical to the file in
  participant 7's folder, so one of the two copies is a stray. Kept
  here once, under 007, pending the loader's rule.
- Excluded sessions: sub_002_day_23 (extra day); sub_008 days 18 to
  21 (extra days, EEG days not counted) and 22 to 24; sub_015 days 22
  to 24 (two home days missed); sub_019_day_13 (computer froze after
  84 trials), day 18 (extra) and days 22 to 24.
- Fixes: sub_003 days 19 and 20 carry day 18 and 19 inside (use the
  filename); sub_015_day_07 line 62 has an extra field (trial 61);
  sub_019_day_24 switches to day 25 from trial 276 and ends with
  blank comma lines; subject-column fixes for sub_015_day_01,
  sub_016_day_22, sub_019_day_01.

These become the exclusion and fix tables of the home-data loader; the
source files stay as copied. Home sessions are ordered by their day
count, not by date; if real dates are ever wanted, someone with access
to the day tracker would need to export a dates-by-subject-number table
(decision 2026-10-27: counting by session is fine for now). Still open: what participant 2's day-22
file from participant 1's number is (excluded meanwhile).

For the record: SharePoint's `eeg_data` for this experiment holds only
9 of the 95 BDFs; this repo's `EEG/` is the complete copy. Two lab
files are named differently on SharePoint (`sub_003_day_401`,
`sub_015_day_15`) from the corrected names here (`sub_003_day_403`,
`sub_015_day_215`).
