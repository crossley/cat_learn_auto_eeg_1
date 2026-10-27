# Code review: Experiment 1 (cat_learn_auto_eeg)

Review date: 2026-10-21. Scope: preprocessing and the
loader/alignment layer. No code was changed. Severity: A invalidates
or corrupts results; B biases or limits them; C style/consistency.

Audit scripts and their outputs live in the session scratchpad
(`check_alignment_exp12.py`, `check_rt_paired.py`,
`check_exp1_loader_paired.py`, `check_raw_triggers_exp1.py`). They
are read-only and can be copied into `tests/` on request.

## A1. Task epochs are CMS-referenced, not average-referenced

`code/preprocess_epochs.py:90-98` fits PyPREP but keeps the input
`raw` and only copies `prep.still_noisy_channels` into
`raw.info["bads"]`. PyPREP's referenced and interpolated output
(`prep.raw`) is discarded. Nothing downstream calls
`set_eeg_reference`.

Evidence: all 94 `EEG_epo/*.fif` have `custom_ref_applied = 0`; the
mean across channels per sample is 12 to 64 microvolts (would be
about 0 if average referenced).

Consequence: every EEG result rests on an unreferenced montage, and
channels PyPREP interpolated are still raw noise. Experiments 3 and 4
fixed this on 2026-10-05 (`finalized_prep_raw` in their
`preprocess_epochs.py`). Fix: port that function and the post-
AutoReject re-reference, add the zero-mean validation, reprocess.

## A2. Epoch files carry no trial metadata; alignment is heuristic

None of the 94 epoch files have `beh_trial_index`. They were made
on 2026-04-10 by `code/util_func_epo.py`, deleted in commit fcc4019.
The current `preprocess_epochs.py` (which writes metadata) has never
produced `EEG_epo/`. So every analysis uses the fallback in
`code/load_project_data.py:219-273`: a stride guess over MNE's
`selection` indices, then a dynamic-programming re-alignment that
chooses whichever monotonic trial mapping best matches A/B labels.

That fallback is doing real work. Raw BDF triggers show dropped
stimulus triggers (ITI, response and feedback triggers intact) in at
least these sessions:

| session | missing stim triggers | first affected trial |
|---|---|---|
| P17 D1 | 32 (+1 spurious code 0) | 174 |
| P18 D1 | 12 | 131 |
| P18 D2 | 7 (+2 spurious code 4) | 21 |
| P18 D3 | 4 | 156 |
| P18 D5 | 3 | 188 |
| P14 D1 | 3 | 134 |
| P13 D5 | 1 stim, 1 feedback | 406 |
| P6 D1 | 1 feedback (+1 code 255) | 13 |

Independent timing test (stimulus-to-feedback latency minus
recorded RT must fall in 200 to 440 ms; it does for 87 sessions with
0 to 2 exceptions each). Under the loader's own alignment:

| session | pairs | out of window | first bad trial |
|---|---|---|---|
| P17 D1 | 313 | 76 | 234 |
| P18 D2 | 421 | 37 | 4 |
| P18 D1 | 453 | 29 | 418 |
| P18 D3 | 456 | 5 | 538 |
| P6 D1 | 546 | 4 | 10 |
| P14 D1 | 547 | 3 | 468 |
| P3 D4 | 544 | 3 | 402 |

So the heuristic recovers most sessions but leaves three (P17 D1,
P18 D1, P18 D2) with tens of misaligned trials in current outputs.
A label-matching aligner can also silently succeed on a wrong
mapping, which is why the check above uses timing, not labels.

Fix: replace counting and the fallback with the exact-sequence
contract used in Experiments 3 and 4 (`_task_event_contract`,
`decode_task_events`), with a per-session omitted-trigger table
(expected codes are deterministic from `cat`, `resp`, `fb`), write
`beh_trial_index` metadata, and delete the fallback. Validate every
output with the timing test.

## A3. Behaviour file with duplicated trial number

`Behavioural/sub_003_day_403_data.csv` has two rows numbered 403
and is not monotonic. `load_project_data.py:122` sorts by `trial`,
which reorders rows relative to the recorded sequence; the EEG
stream shows trials 401 and 402 swapped relative to the sorted file.
Fix: load in recorded order and assert contiguity; treat this file
as a documented exception.

## B1. No event-count or label validation at epoching

`preprocess_epochs.py:115-127` assigns trial indices by counting
stimulus events and never checks that the count equals 550 or that
the EEG category code matches the behavioural `cat`. Any dropped
trigger shifts all later trials silently. Covered by the A2 fix.

## B2. Status channel read without masking system bits

`preprocess_epochs.py:76` uses `find_events` on the raw Status
value. Spurious codes 0, 4 and 255 appear in P17 D1, P18 D2 and
P6 D1. Experiments 3 and 4 mask to the low byte and drop zeros
(`canonicalize_status_events`). Port it.

## B3. Silent handling of non-64-channel recordings

`preprocess_epochs.py:64-65` renames channels only if exactly 64 cap
channels are found, otherwise continues with unmapped names and
`on_missing="ignore"`. Should raise, as Experiments 3 and 4 do.

## B4. Heavy AutoReject rejection in six sessions

Fewer than 60 percent of epochs retained: P14 D2, P14 D4, P1 D3,
P3 D5, P4 D4, P7 D4 (retention 0.50 to 0.55). Median retention is
0.91. These sessions should be flagged in QC tables and considered
for exclusion rules before reprocessing.

## C1. Provenance

`EEG_epo/` cannot be regenerated from committed code; the producing
script was deleted. `task_eeg_preprocessed/` is a first-generation
derivative (different filter, RANSAC, no resampling) that no current
code uses. README lists several deleted scripts. Reprocessing under
A1 and A2 resolves all three.

## C2. Same-day temporal generalization leakage (already known)

`code/mvpa_stim_locked_cat_tg_analysis.py` scored same-day pairs
without cross-validation (pre-stimulus diagonal AUC about 0.70). The
fix is uncommitted and outputs were not re-run. Commit and re-run
after reprocessing.

## Verified as sound

- Band-pass, resample-with-events order, AutoReject settings and
  epoch windows match the other repositories.
- Epochs keep bad channels (MNE `picks="eeg"` retains bads), and
  `util_mvpa.pick_eeg_interpolate_bads` interpolates them before
  decoding. 40 of 94 files have 1 to 5 bads.
- `util_mvpa.py` is byte-identical across all four repositories;
  scaler and classifier are fitted inside each fold.

# Stage 3 and 4: MVPA core and inference (added 2026-10-21)

## A4. Which analyses the alignment problem (A2) actually touches

Category labels in `mvpa_stim_locked_cat_time_resolved_analysis.py`,
`mvpa_stim_locked_cat_roi_time_resolved_analysis.py`,
`mvpa_stim_locked_cat_tg_analysis.py` and `util_mvpa.py` come from
EEG event codes, so those decoding results are unaffected by trial
misalignment. Everything that joins behaviour through
`align_behaviour_to_epochs` is affected in the misaligned sessions:
both block-model scripts, both difficulty scripts, the feedback-locked
category TG (labels from behaviour `cat`), the peak-behaviour summary,
all connectivity scripts, `erp_grand_average_analysis.py` and the RSA
engine (stimulus coordinates). Fifteen scripts in total.

## B5. The 25-block model set does not operationalize abrupt versus gradual

`block_model_utils.py:33-48`. The "continuous" template is
0.65 * min(i, j) / 25 for every off-diagonal block pair, so it predicts
that transfer grows with the earlier block's index. The "discrete"
template is the same quantity zeroed when the two blocks straddle the
split. Both are monotone growth models; neither encodes a change of
representation (which predicts transfer as a function of block
distance and regime membership, not of min(i, j)). A step-plus-ramp
and a distance-based template are needed before the comparison can
answer the question paper 1 asks.

Two smaller points. The 25 x 25 pair AUCs entered into each BIC fit
share blocks and are not independent, so per-subject BIC differences
are overconfident; the across-subject mean and SEM of delta BIC are
the only defensible summary. Block membership
(`assign_accumulated_blocks`) splits the retained trials of a day
into quintiles, so block boundaries move with rejection; use the
recorded trial number instead.

## B6. Day-effect inference falls back to OLS silently

`mvpa_stim_locked_cat_time_resolved_analysis.py:407-453`. The mixed
model fails at 110 of 128 time points and the fallback OLS ignores
subject clustering, then FDR is applied across time. Replace with
per-subject least-squares slopes tested by sign flip with a maximum
statistic, or a cluster-mass test, as the other repositories do.

## B7. Peak picking before correlating with accuracy

`mvpa_stim_locked_cat_peak_behavior_analysis.py:56-66` picks the
maximum AUC sample per subject and day inside each window and then
correlates it with accuracy. Selecting the maximum inflates the value
and its coupling to anything that scales noise. Use `window_auc`
(already written) or pick the peak time on held-out folds.

## C3. Deleted scripts behind retained results

`mvpa_stim_locked_cat_tg_window_structure_analysis.py`,
`..._tg_day1_distinctiveness_analysis.py` and `..._tg_mixture_*` were
deleted in commit be1c777 (2026-06-26). Their outputs (for example
`mvpa_stim_locked_cat_tg_day1_distinctiveness_model_terms.csv`, the
source of the 0.023 day-1 transfer deficit) remain in `output/` and
the README still lists the scripts. Restore or re-implement before
the day-1 result is used.

## Verified in stage 3 and 4

- Within-day decoding is cross-validated at every time sample with
  the scaler inside the pipeline; Haufe patterns are computed per fold.
- Cross-day TG fits on the training day only and tests on a balanced
  draw of the test day; same-day cells need the pending CV fix.
- `util_mvpa.balanced_day_subset` samples without replacement.

# Stage 5 to 7: RSA, connectivity, behaviour, ERP (added 2026-10-21)

## A5. The RSA family cannot run

`util_rsa_time_resolved.py:26` and `rsa_model_prediction_analysis.py`
import `util_boundary_distance`, deleted in June. Every `rsa_*`
output in `output/` (dated 2026-05-20) is unreproducible from the
tree. Restore the module or fold the boundary fit into the RSA
engine before any RSA result is used.

## B8. RSA dissimilarities are biased by trial counts

The engine computes 1 minus the Pearson correlation between unit-mean
topographies (`util_rsa_time_resolved.py:176-179`). That distance is
inflated by noise, and noise falls with the number of trials in a
bin. Bins at the edge of the stimulus distribution have few trials
and are far from every other bin, so the physical-distance model is
collinear with noise. Measured on the saved windowed RDMs:

| quantity | 0.05 to 0.2 s | 0.4 to 0.7 s |
|---|---|---|
| corr(dissimilarity, 1/n_i + 1/n_j) | 0.17 | 0.15 |
| corr(physical model, 1/n_i + 1/n_j) | 0.34 | 0.34 |
| corr(dissimilarity, physical model) | 0.11 | 0.07 |

Bin counts range from 6 to 53. The physical-distance fit, and any
day trend in it, is partly a trial-count artefact. Use cross-
validated distances (the crossnobis design in Experiment 2) or equal
trial counts per bin. Model fits are also Pearson, not Spearman as
the README implies, and are run one model at a time although the
four model RDMs are correlated.

## B9. Connectivity estimates at low frequencies are not valid

`connect_multimeasure_utils.py:64-65, 152-164` uses Morlet wavelets
with 2 cycles from 1 to 40 Hz on epochs cropped to 0 to 0.8 s. Below
about 5 Hz the wavelet is longer than the epoch, so the delta band
and the low end of "broadband" are edge artefacts. Restrict to 4 Hz
and above, or compute connectivity on longer epochs once response-
locked re-epoching exists. Two further points: the visual-central
and visual-frontal ROI pairs differ systematically in inter-
electrode distance, so their contrast is confounded with signal-to-
noise ratio; and the connectivity block model
(`connect_multimeasure_block_model_timecourse_analysis.py`) inherits
the template problem in B5. `erp_model_timecourse_analysis.py` uses
the same templates on 10 day pairs per subject, too few for BIC to
mean anything.

## B10. Decision-bound model set

`decision_bound_strategy_analysis.py` fits both polarities (good) but
has no conjunctive or guessing model, so early day-1 blocks that are
really guessing or conjunctive will be assigned to a rule or to the
GLC. The Experiment 2 `claude/analyses/a02` set is more complete;
combine the two. The MVPA link (`extract_mvpa_split_blocks`) depends
on the block-model templates (B5).

## Verified in stage 5 to 7

- Decision-bound likelihoods, BIC, and two-block persistence rule
  for the strategy switch.
- ERP grand averages are correct per-session means (subject to A2
  alignment for the behaviour-conditioned splits).
