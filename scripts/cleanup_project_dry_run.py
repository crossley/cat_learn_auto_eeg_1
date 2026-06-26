#!/usr/bin/env python3
"""Write dry-run cleanup manifests for generated project files.

This script does not delete anything. It writes flat CSV manifests into
`logs/` so cleanup decisions can be reviewed before removal.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
FIGURES_DIR = PROJECT_DIR / "figures"
OUTPUT_DIR = PROJECT_DIR / "output"
CODE_DIR = PROJECT_DIR / "code"
LOGS_DIR = PROJECT_DIR / "logs"

KEEP_FIGURES = {
    "connect_block_model_timecourse.png",
    "connect_roi_timecourse_by_day_visual_frontal_central.png",
    "connect_roi_timecourse_visual_central_minus_frontal.png",
    "connect_roi_timecourse_visual_frontal_central.png",
    "decision_bound_best_model_heatmap.png",
    "decision_bound_block_model_evidence.png",
    "decision_bound_glc_advantage_heatmap.png",
    "decision_bound_optimal_fixed_best_model_heatmap.png",
    "decision_bound_optimal_fixed_block_model_evidence.png",
    "decision_bound_optimal_fixed_diagnostic_glc_evidence_heatmap.png",
    "erp_grand_average_feedback_all.png",
    "erp_grand_average_stim_all.png",
    "erp_model_timecourse.png",
    "mvpa_block_model_timecourse.png",
    "mvpa_feedback_locked_cat_tg_timegen_matrices_5x5.png",
    "mvpa_stim_feedback_cat_tg_timegen_matrices_feedback_to_stim_5x5.png",
    "mvpa_stim_feedback_cat_tg_timegen_matrices_stim_to_feedback_5x5.png",
    "mvpa_stim_locked_cat_roi_time_resolved_auc.png",
    "mvpa_stim_locked_cat_tg_timegen_matrices_5x5.png",
    "presentation_connectivity_model_predictions.png",
    "presentation_connectivity_model_timecourse_top10.png",
    "presentation_connectivity_top10_decomposition_overlay.png",
    "presentation_erp_gfp.png",
    "presentation_erp_stim_all.png",
    "presentation_mvpa_model_timecourse.png",
    "presentation_mvpa_peak_and_behavior.png",
    "presentation_mvpa_time_resolved_auc.png",
    "presentation_mvpa_window_transfer_model_predictions.png",
    "presentation_rsa_model_fit_timecourses.png",
    "presentation_rsa_model_prediction_rdms.png",
    "rsa_feedback_model_fit_timecourses.png",
    "rsa_feedback_windowed_model_fit_timecourses.png",
    "rsa_model_grid_diagnostics.png",
    "rsa_model_prediction_rdms.png",
    "rsa_stim_model_fit_timecourses.png",
    "rsa_stim_windowed_model_fit_timecourses.png",
    "sequence_hmm_voltage_dominance_entropy_timecourse_4states.png",
    "sequence_hmm_voltage_model_selection.png",
    "sequence_hmm_voltage_state_occupancy_timecourse_4states.png",
    "sequence_hmm_voltage_stim_0_800_dominance_entropy_timecourse_4states.png",
    "sequence_hmm_voltage_stim_0_800_model_selection.png",
    "sequence_hmm_voltage_stim_0_800_state_occupancy_timecourse_4states.png",
}

KEEP_OUTPUT_PREFIXES = (
    "connect_block_model_timecourse_",
    "connect_roi_timecourse_",
    "decision_bound_block_model_",
    "decision_bound_glc_",
    "decision_bound_mvpa_",
    "decision_bound_optimal_fixed_",
    "decision_bound_qc_",
    "decision_bound_strategy_",
    "erp_grand_average_",
    "erp_model_timecourse_",
    "mvpa_block_model_timecourse_",
    "mvpa_feedback_locked_cat_tg_",
    "mvpa_stim_feedback_cat_tg_",
    "mvpa_stim_locked_cat_roi_time_resolved_",
    "mvpa_stim_locked_cat_tg_",
    "mvpa_stim_locked_cat_",
    "mvpa_tg_diagonal_presentation_model_bic_",
    "mvpa_late_subwindow_presentation_model_bic_",
    "presentation_mvpa_model_timecourse",
    "rsa_feedback_time_resolved_",
    "rsa_feedback_windowed_",
    "rsa_model_",
    "rsa_stim_time_resolved_",
    "rsa_stim_windowed_",
    "sequence_hmm_voltage",
    "sequence_features_",
    "sensorwide_",
    "connect_sensorwide_",
)

KEEP_CODE_FILES = {
    "analysis_utils.py",
    "block_model_timecourse_figure.py",
    "block_model_utils.py",
    "connect_block_model_timecourse_analysis.py",
    "connect_roi_timecourse_analysis.py",
    "connect_roi_timecourse_figure.py",
    "connect_sensorwide_analysis.py",
    "connect_sensorwide_model_bic_timecourse_analysis.py",
    "connect_sensorwide_model_posterior_pairwise_analysis.py",
    "connect_sensorwide_model_posterior_shape_analysis.py",
    "connect_sensorwide_model_timecourse_analysis.py",
    "connect_sensorwide_window_average_analysis.py",
    "decision_bound_mvpa_diagnostic_figure.py",
    "decision_bound_strategy_analysis.py",
    "decision_bound_strategy_figure.py",
    "erp_grand_average_analysis.py",
    "erp_grand_average_figure.py",
    "erp_model_timecourse_analysis.py",
    "erp_model_timecourse_figure.py",
    "experiment.py",
    "load_project_data.py",
    "mvpa_block_model_timecourse_analysis.py",
    "mvpa_feedback_locked_cat_tg_analysis.py",
    "mvpa_feedback_locked_cat_tg_figure.py",
    "mvpa_presentation_model_bic_analysis.py",
    "mvpa_stim_feedback_cat_tg_analysis.py",
    "mvpa_stim_feedback_cat_tg_figure.py",
    "mvpa_stim_locked_cat_roi_time_resolved_analysis.py",
    "mvpa_stim_locked_cat_roi_time_resolved_figure.py",
    "mvpa_stim_locked_cat_tg_analysis.py",
    "mvpa_stim_locked_cat_tg_figure.py",
    "mvpa_stim_locked_cat_time_resolved_analysis.py",
    "preprocess_epochs.py",
    "presentation_figure.py",
    "presentation_mvpa_model_timecourse_analysis.py",
    "rsa_feedback_time_resolved_analysis.py",
    "rsa_feedback_time_resolved_figure.py",
    "rsa_feedback_windowed_analysis.py",
    "rsa_feedback_windowed_figure.py",
    "rsa_model_prediction_analysis.py",
    "rsa_model_prediction_figure.py",
    "rsa_stim_time_resolved_analysis.py",
    "rsa_stim_time_resolved_figure.py",
    "rsa_stim_windowed_analysis.py",
    "rsa_stim_windowed_figure.py",
    "sensor_rois.py",
    "sequence_feature_interface.py",
    "sequence_hmm_analysis.py",
    "sequence_hmm_figure.py",
    "stimuli.py",
    "util_mvpa.py",
    "util_rsa_figure.py",
    "util_rsa_time_resolved.py",
}


def csv_escape(value):
    text = str(value)
    if any(ch in text for ch in [",", '"', "\n"]):
        return '"' + text.replace('"', '""') + '"'
    return text


def write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("path,size_bytes,action,reason\n")
        for row in rows:
            f.write(",".join(csv_escape(x) for x in row) + "\n")


def file_size(path):
    return path.stat().st_size if path.exists() else 0


def figure_rows():
    rows = []
    for path in sorted(FIGURES_DIR.glob("*")):
        if not path.is_file():
            continue
        if path.name in KEEP_FIGURES:
            continue
        rows.append((path.relative_to(PROJECT_DIR), file_size(path), "remove", "not in retained figure list"))
    return rows


def output_rows():
    rows = []
    for path in sorted(OUTPUT_DIR.glob("*")):
        if not path.is_file():
            continue
        if path.name.startswith(KEEP_OUTPUT_PREFIXES):
            continue
        rows.append((path.relative_to(PROJECT_DIR), file_size(path), "remove_candidate", "output prefix not retained"))
    return rows


def code_rows():
    rows = []
    for path in sorted(CODE_DIR.glob("*.py")):
        if path.name in KEEP_CODE_FILES:
            continue
        rows.append((path.relative_to(PROJECT_DIR), file_size(path), "review_remove_candidate", "not in retained code set"))
    return rows


def log_rows():
    rows = []
    for path in sorted(LOGS_DIR.glob("*")):
        if not path.is_file():
            continue
        if path.name.startswith("cleanup_"):
            continue
        rows.append((path.relative_to(PROJECT_DIR), file_size(path), "remove_candidate", "old run log or pid file"))
    return rows


def main():
    manifests = {
        "logs/cleanup_figures_remove.csv": figure_rows(),
        "logs/cleanup_output_remove_candidates.csv": output_rows(),
        "logs/cleanup_code_review_remove_candidates.csv": code_rows(),
        "logs/cleanup_logs_remove_candidates.csv": log_rows(),
    }
    for rel_path, rows in manifests.items():
        write_manifest(PROJECT_DIR / rel_path, rows)

    summary_path = PROJECT_DIR / "logs/cleanup_summary.txt"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Cleanup dry-run summary\n")
        f.write("=======================\n")
        f.write(f"retained figures requested: {len(KEEP_FIGURES)}\n")
        for rel_path, rows in manifests.items():
            total = sum(int(row[1]) for row in rows)
            f.write(f"{rel_path}: {len(rows)} files, {total} bytes\n")

    print(f"wrote {summary_path.relative_to(PROJECT_DIR)}")
    for rel_path, rows in manifests.items():
        print(f"wrote {rel_path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
