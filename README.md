# Category Learning Automaticity EEG

This repository contains the experiment code and analysis scripts for a
multi-day category-learning EEG project. The scientific goal is to track how
behavioural performance and EEG signatures change as participants develop
automaticity in the task.

The code assumes a fixed project layout. If files are missing, misnamed, or
stored in the wrong place, scripts fail clearly rather than searching for
alternatives.

## Project Layout

Required local data folders:

```text
Behavioural/sub_<subject>_day_<daycode>_data.csv
EEG/P<subject>_D<daycode>.bdf
EEG_epo/P<subject>_D<daycode>-epo.fif
```

Generated outputs:

```text
output/
figures/
```

`Behavioural/`, `EEG/`, `EEG_epo/`, and `output/` are not tracked in git.
Generated figures are tracked so the repository shows the current state of
the analyses.

## Environment

```bash
conda env create -f environment.yml
conda activate cat-learn-auto-eeg
```

## Analysis Scripts

Each script in `code/` owns one analysis. Run from the repository root:

```bash
python code/<script>.py
```

There is deliberately no all-in-one runner.

### Shared infrastructure

| Script | Role |
|---|---|
| `experiment.py` | PsychoPy task and EEG trigger logic |
| `stimuli.py` | Category stimulus generation |
| `preprocess_epochs.py` | Raw BDF → cleaned epoch FIF |
| `load_project_data.py` | Behavioural/epoch loading and alignment |
| `analysis_utils.py` | `parallel_collect`, `model_term_summary` |
| `boundary_distance.py` | Perpendicular boundary-distance computation |

### ERP analyses (`erp_`)

| Script | Analysis |
|---|---|
| `erp_grand_average.py` | Grand-average ERPs, stim- and response-locked |
| `erp_latency_trajectories.py` | ERP peak-latency trajectories across days |
| `erp_n2_boundary.py` | Frontal N2 × boundary distance |
| `erp_frn_rpe_day1.py` | FRN and RPE analysis (Day 1) |
| `erp_feedback_window_predictors.py` | Feedback-window amplitudes × feedback, boundary distance, and RPE |
| `erp_p3_boundary.py` | Parietal P3 × boundary distance |
| `erp_rt_bridge.py` | ERP–RT bridge analysis |

### MVPA time-resolved (`mvpa_time_resolved_`)

| Script | Analysis |
|---|---|
| `mvpa_time_resolved_stim_locked.py` | Stimulus-locked decoding timecourses |
| `mvpa_time_resolved_response_locked.py` | Response-locked decoding timecourses |

### MVPA temporal generalization (`mvpa_tg_`)

| Script | Analysis |
|---|---|
| `mvpa_tg_within_day.py` | Within-day TG matrices (session cache infrastructure) |
| `mvpa_tg_cross_day.py` | Cross-day TG matrices |
| `mvpa_tg_stim_feedback_cross_epoch.py` | Cross-epoch TG between stimulus and feedback A/B codes |
| `mvpa_tg_feedback_locked.py` | Feedback-locked category-label cross-day TG |
| `mvpa_tg_window_structure.py` | Early/late AUC windows and day-distance gradients |
| `mvpa_tg_day1_distinctiveness.py` | Day-1 distinctiveness and anchored trajectories |
| `mvpa_tg_band_envelope.py` | Cross-day TG on band-limited amplitude envelopes |
| `mvpa_tg_band_signed.py` | Cross-day TG on band-limited signed voltages |
| `mvpa_tg_broadband_vs_band.py` | Broadband vs band diagnostic (diagonal timecourses) |
| `mvpa_classifier_confidence.py` | Classifier confidence time-resolved |

### Connectivity (`connect_`)

| Script | Analysis |
|---|---|
| `connect_stim_locked.py` | Stim-locked visual-motor abs(ImCoh) |
| `connect_response_locked.py` | Response-locked visual-motor abs(ImCoh) |
| `connect_sensorwide.py` | 16-channel sensor-wide connectivity dynamics |

## Parallelism

Scripts that do heavy computation expose an `N_JOBS` constant near the top of
the file. Edit it directly before running. No automatic worker count is inferred
and no silent fallback to serial execution occurs.

## Style

Flat one-analysis-per-file layout. Shared operations live in `analysis_utils.py`
(`parallel_collect`, `model_term_summary`) and `load_project_data.py`. Small
helpers are duplicated in each file rather than extracted into a shared module.
Cross-file imports are used only for non-trivial shared functions (`build_clf`,
`prepare_session_cache`, `write_cross_day_outputs`, etc.).
