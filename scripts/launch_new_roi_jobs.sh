#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-python}"
FORCE="${FORCE:-0}"

if [[ "${NEW_ROI_JOBS_CHILD:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  stamp="$(date +%Y%m%d_%H%M%S)"
  run_log="${LOG_DIR}/new_roi_jobs_${stamp}.log"
  pid_file="${LOG_DIR}/new_roi_jobs_${stamp}.pid"

  NEW_ROI_JOBS_CHILD=1 PYTHON_BIN="${PYTHON_BIN}" FORCE="${FORCE}" nohup bash "$0" > "${run_log}" 2>&1 &
  pid="$!"

  printf '%s\n' "${pid}" > "${pid_file}"
  printf '%s\n' "${pid}" > "${LOG_DIR}/new_roi_jobs_latest.pid"
  printf '%s\n' "${run_log}" > "${LOG_DIR}/new_roi_jobs_latest.logpath"
  printf '%s\n' "${pid_file}" > "${LOG_DIR}/new_roi_jobs_latest.pidpath"

  echo "Started new ROI analysis batch"
  echo "PID: ${pid}"
  echo "Log: ${run_log}"
  echo "PID file: ${pid_file}"
  exit 0
fi

cd "${ROOT}"

run_step() {
  local label="$1"
  shift
  echo
  echo "==== ${label} ===="
  date
  "$@"
}

run_step_if_needed() {
  local label="$1"
  local sentinel="$2"
  shift 2
  if [[ "${FORCE}" != "1" && -s "${sentinel}" ]]; then
    echo
    echo "==== ${label} ===="
    date
    echo "Skipping; found ${sentinel}"
    return 0
  fi
  run_step "${label}" "$@"
}

echo "New ROI analysis batch started"
echo "Repo: ${ROOT}"
echo "Python: ${PYTHON_BIN}"
echo "Force: ${FORCE}"
date

run_step_if_needed \
  "ERP model-timecourse analysis" \
  "${ROOT}/output/erp_model_timecourse_summary.csv" \
  "${PYTHON_BIN}" code/erp_model_timecourse_analysis.py
run_step_if_needed \
  "ERP model-timecourse figure" \
  "${ROOT}/figures/erp_model_timecourse.png" \
  "${PYTHON_BIN}" code/erp_model_timecourse_figure.py

run_step_if_needed \
  "Connectivity ROI time-course analysis" \
  "${ROOT}/output/connect_roi_timecourse_day_mean.csv" \
  "${PYTHON_BIN}" code/connect_roi_timecourse_analysis.py
run_step_if_needed \
  "Connectivity ROI time-course figure" \
  "${ROOT}/figures/connect_roi_timecourse_visual_frontal_central.png" \
  "${PYTHON_BIN}" code/connect_roi_timecourse_figure.py

run_step_if_needed \
  "MVPA ROI time-resolved analysis" \
  "${ROOT}/output/mvpa_stim_locked_cat_roi_day_means_timecourse.csv" \
  "${PYTHON_BIN}" code/mvpa_stim_locked_cat_roi_time_resolved_analysis.py
run_step_if_needed \
  "MVPA ROI time-resolved figure" \
  "${ROOT}/figures/mvpa_stim_locked_cat_roi_time_resolved_auc.png" \
  "${PYTHON_BIN}" code/mvpa_stim_locked_cat_roi_time_resolved_figure.py

echo
echo "New ROI analysis batch completed"
date
