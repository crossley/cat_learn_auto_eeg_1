#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT}/logs"
TAIL_LINES="${TAIL_LINES:-60}"

pid=""
log_path=""

if [[ -f "${LOG_DIR}/new_roi_jobs_latest.pid" ]]; then
  pid="$(cat "${LOG_DIR}/new_roi_jobs_latest.pid")"
fi

if [[ -f "${LOG_DIR}/new_roi_jobs_latest.logpath" ]]; then
  log_path="$(cat "${LOG_DIR}/new_roi_jobs_latest.logpath")"
fi

if [[ -n "${pid}" ]]; then
  echo "PID: ${pid}"
  if ps -p "${pid}" > /dev/null 2>&1; then
    echo "Status: running"
  else
    echo "Status: not running"
  fi
else
  echo "PID: none recorded"
fi

progress="${ROOT}/output/mvpa_stim_locked_cat_roi_progress.json"
if [[ -f "${progress}" ]]; then
  echo
  echo "MVPA ROI progress:"
  cat "${progress}"
  echo
fi

if [[ -n "${log_path}" && -f "${log_path}" ]]; then
  echo
  echo "Log: ${log_path}"
  echo "Last ${TAIL_LINES} lines:"
  tail -n "${TAIL_LINES}" "${log_path}"
else
  echo
  echo "Log: none recorded"
fi
