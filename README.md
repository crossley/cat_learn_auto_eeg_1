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
| `util_boundary_distance.py` | Shared boundary-distance helper functions |
| `util_mvpa.py` | Shared TG classifier/cache helpers |
| `util_rsa_time_resolved.py` | Shared RSA compute engine used by stim/feedback entry points |
| `util_rsa_figure.py` | Shared RSA plotting engine used by stim/feedback figure entry points |

### Boundary distance (`boundary_`)

| Script | Analysis |
|---|---|
| `boundary_distance_analysis.py` | Compute trial-level category-boundary distances |

### ERP analyses (`erp_`)

| Script | Analysis |
|---|---|
| `erp_grand_average_analysis.py` | Compute core grand-average ERP outputs |
| `erp_grand_average_figure.py` | Plot core grand-average ERP figures |

### MVPA (`mvpa_`)

| Script | Analysis |
|---|---|
| `mvpa_stim_locked_cat_time_resolved_analysis.py` | Compute stimulus-locked category decoding outputs |
| `mvpa_stim_locked_cat_time_resolved_figure.py` | Plot stimulus-locked category decoding figures |
| `mvpa_feedback_locked_cat_time_resolved_analysis.py` | Extract feedback-locked category decoding diagonals |
| `mvpa_feedback_locked_cat_time_resolved_figure.py` | Plot feedback-locked category decoding figures |
| `mvpa_stim_locked_cat_tg_analysis.py` | Compute cross-day TG outputs |
| `mvpa_stim_locked_cat_tg_figure.py` | Plot cross-day TG figures |
| `mvpa_feedback_locked_cat_tg_analysis.py` | Compute feedback-locked category TG outputs |
| `mvpa_feedback_locked_cat_tg_figure.py` | Plot feedback-locked category TG figures |
| `mvpa_stim_feedback_cat_tg_analysis.py` | Compute cross-epoch stimulus/feedback TG outputs |
| `mvpa_stim_feedback_cat_tg_figure.py` | Plot cross-epoch stimulus/feedback TG figures |
| `mvpa_stim_locked_cat_tg_window_structure_analysis.py` | Compute early/late AUC windows and day-distance gradients |
| `mvpa_stim_locked_cat_tg_day1_distinctiveness_analysis.py` | Compute Day-1 distinctiveness summaries |
| `mvpa_stim_locked_cat_tg_day1_distinctiveness_figure.py` | Plot Day-1 distinctiveness figures |

### RSA (`rsa_`)

| Script | Analysis |
|---|---|
| `rsa_model_prediction_analysis.py` | Build model RDM prediction inputs |
| `rsa_model_prediction_figure.py` | Plot model RDM prediction figures |
| `rsa_stim_time_resolved_analysis.py` | Stimulus-locked time-resolved RSA outputs |
| `rsa_stim_time_resolved_figure.py` | Stimulus-locked time-resolved RSA figures |
| `rsa_stim_windowed_analysis.py` | Stimulus-locked short-window RSA outputs |
| `rsa_stim_windowed_figure.py` | Stimulus-locked short-window RSA figures |
| `rsa_feedback_time_resolved_analysis.py` | Feedback-locked time-resolved RSA outputs |
| `rsa_feedback_time_resolved_figure.py` | Feedback-locked time-resolved RSA figures |
| `rsa_feedback_windowed_analysis.py` | Feedback-locked short-window RSA outputs |
| `rsa_feedback_windowed_figure.py` | Feedback-locked short-window RSA figures |

### Connectivity (`connect_`)

| Script | Analysis |
|---|---|
| `connect_sensorwide_analysis.py` | Compute sensor-wide stim/feedback connectivity outputs |
| `connect_sensorwide_figure.py` | Plot sensor-wide stim/feedback carpet figures |

## Parallelism

Scripts that do heavy computation expose an `N_JOBS` constant near the top of
the file. Edit it directly before running. No automatic worker count is inferred
and no silent fallback to serial execution occurs.

## Style

Flat one-purpose entry-point layout. Analysis scripts end in `_analysis.py` and
write outputs only. Figure scripts end in `_figure.py` and read saved outputs
only. There is deliberately no all-in-one runner, and orchestration scripts such
as `inspect_results_eeg.py` or `run_new_analyses.py` should not be used.

Shared helper modules are allowed only when duplication would make the analysis
files harder to audit. Current helper modules are listed in the shared
infrastructure table above.

## Expected Outputs

The active analyses write these output families:

| Script | Output files |
|---|---|
| `boundary_distance_analysis.py` | `boundary_distance_model_params.csv`, `boundary_distance_behaviour_trial_level.csv` |
| `erp_grand_average_analysis.py` | `erp_grand_average_by_day_lock_condition.csv`, `erp_grand_average_subject_day_all.csv`, `erp_grand_average_qc.csv`, `erp_grand_average_progress.json` |
| `mvpa_stim_locked_cat_time_resolved_analysis.py` | `mvpa_stim_locked_cat_session_timecourse.csv`, `mvpa_stim_locked_cat_subject_day_timecourse.csv`, `mvpa_stim_locked_cat_day_means_timecourse.csv`, `mvpa_stim_locked_cat_day_effect_per_time.csv`, `mvpa_stim_locked_cat_qc_log.csv`, `mvpa_stim_locked_cat_progress.json`, `mvpa_stim_locked_cat_haufe_*` |
| `mvpa_feedback_locked_cat_time_resolved_analysis.py` | `mvpa_feedback_locked_cat_time_resolved_day_means_timecourse.csv` |
| `mvpa_stim_locked_cat_tg_analysis.py` | `mvpa_stim_locked_cat_tg_subject_level.csv`, `mvpa_stim_locked_cat_tg_day_mean.csv`, `mvpa_stim_locked_cat_tg_timegen_day_mean.csv`, `mvpa_stim_locked_cat_tg_qc_log.csv`, `mvpa_stim_locked_cat_tg_matrix_sub_*.npz` |
| `mvpa_feedback_locked_cat_tg_analysis.py` | `mvpa_feedback_locked_cat_tg_subject_level.csv`, `mvpa_feedback_locked_cat_tg_day_mean.csv`, `mvpa_feedback_locked_cat_tg_timegen_day_mean.csv`, `mvpa_feedback_locked_cat_tg_qc_log.csv`, `mvpa_feedback_locked_cat_tg_matrix_sub_*.npz`, `mvpa_feedback_locked_cat_tg_haufe_*` |
| `mvpa_stim_feedback_cat_tg_analysis.py` | `mvpa_stim_feedback_cat_tg_subject_level.csv`, `mvpa_stim_feedback_cat_tg_day_mean.csv`, `mvpa_stim_feedback_cat_tg_day_pair_mean.csv`, `mvpa_stim_feedback_cat_tg_timegen_day_mean.csv`, `mvpa_stim_feedback_cat_tg_timegen_day_pair_mean.csv`, `mvpa_stim_feedback_cat_tg_qc_log.csv` |
| `mvpa_stim_locked_cat_tg_window_structure_analysis.py` | `mvpa_stim_locked_cat_tg_window_auc_subject_pairs.csv`, `mvpa_stim_locked_cat_tg_window_gradient_slopes.csv`, `mvpa_stim_locked_cat_tg_window_slope_difference.json` |
| `mvpa_stim_locked_cat_tg_day1_distinctiveness_analysis.py` | `mvpa_stim_locked_cat_tg_day1_window_auc_subject_pairs.csv`, `mvpa_stim_locked_cat_tg_day1_distinctiveness_model_terms.csv`, `mvpa_stim_locked_cat_tg_day1_pair_type_summary.csv`, `mvpa_stim_locked_cat_tg_day_pair_window_auc_matrix*.csv` |
| `rsa_model_prediction_analysis.py` | `rsa_model_grid_diagnostics.csv`, `rsa_model_stimulus_bins.csv`, `rsa_model_rdms.csv` |
| `rsa_*_time_resolved_analysis.py` | `rsa_*_time_resolved_rdms.csv`, `rsa_*_bin_epoch_counts.csv`, `rsa_*_model_fit_timecourses.csv`, `rsa_*_cross_day_geometry_similarity.csv`, `rsa_*_model_vectors.csv`, `rsa_*_time_resolved_qc_log.csv` |
| `rsa_*_windowed_analysis.py` | `rsa_*_windowed_rdms.csv`, `rsa_*_windowed_bin_epoch_counts.csv`, `rsa_*_windowed_model_fit_timecourses.csv`, `rsa_*_windowed_cross_day_geometry_similarity.csv`, `rsa_*_windowed_model_vectors.csv`, `rsa_*_windowed_qc_log.csv` |
| `connect_sensorwide_analysis.py` | `sensorwide_carpet_timeseries.csv`, `sensorwide_carpet_timeseries_checkpoint.csv`, `sensorwide_channel_layout.csv`, `connect_sensorwide_progress.json`, optional `connect_sensorwide_qc_skipped.csv` |

## Expected Figures

The active figure scripts generate these retained figure families:

| Script | Figure files |
|---|---|
| `erp_grand_average_figure.py` | `erp_grand_average_stim_all.png`, `erp_grand_average_stim_correct_vs_incorrect.png`, `erp_grand_average_stim_cat_a_vs_cat_b.png`, `erp_grand_average_feedback_all.png`, `erp_grand_average_feedback_correct_vs_incorrect.png`, `erp_grand_average_feedback_cat_a_vs_cat_b.png` |
| `mvpa_stim_locked_cat_time_resolved_figure.py` | `mvpa_stim_locked_cat_auc_by_day_panels.png`, `mvpa_stim_locked_cat_haufe_similarity_timegen_matrices_5x5.png` |
| `mvpa_feedback_locked_cat_time_resolved_figure.py` | `mvpa_feedback_locked_cat_auc_by_day_panels.png` |
| `mvpa_stim_locked_cat_tg_figure.py` | `mvpa_stim_locked_cat_tg_transfer_5x4.png`, `mvpa_stim_locked_cat_tg_timegen_matrices_5x5.png` |
| `mvpa_feedback_locked_cat_tg_figure.py` | `mvpa_feedback_locked_cat_tg_transfer_5x4.png`, `mvpa_feedback_locked_cat_tg_timegen_matrices_5x5.png`, `mvpa_feedback_locked_cat_tg_haufe_similarity_timegen_matrices_5x5.png` |
| `mvpa_stim_feedback_cat_tg_figure.py` | `mvpa_stim_feedback_cat_tg_timegen_matrices_stim_to_feedback_5x5.png`, `mvpa_stim_feedback_cat_tg_timegen_matrices_feedback_to_stim_5x5.png` |
| `mvpa_stim_locked_cat_tg_day1_distinctiveness_figure.py` | `mvpa_stim_locked_cat_tg_day_pair_window_matrices_by_summary.png` |
| `rsa_model_prediction_figure.py` | `rsa_model_grid_diagnostics.png`, `rsa_model_prediction_rdms.png` |
| `rsa_*_time_resolved_figure.py` | `rsa_*_model_fit_timecourses.png`, `rsa_*_neural_rdm_snapshots.png`, `rsa_*_cross_day_geometry_similarity.png`, `rsa_*_cross_day_geometry_timecourse_pairs.png` |
| `rsa_stim_time_resolved_figure.py` | Also `rsa_stim_cross_day_geometry_timecourse.png`, `rsa_stim_cross_day_geometry_timecourse_5x5.png` |
| `rsa_*_windowed_figure.py` | `rsa_*_windowed_model_fit_timecourses.png`, `rsa_*_windowed_neural_rdm_snapshots.png`, `rsa_*_windowed_cross_day_geometry_similarity.png`, `rsa_*_windowed_cross_day_geometry_timecourse_pairs.png` |
| `connect_sensorwide_figure.py` | `sensorwide_carpet_stim_*.png`, `sensorwide_carpet_feedback_*.png` for broadband, delta, theta, and alpha when sensorwide outputs exist |
