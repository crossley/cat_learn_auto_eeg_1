#!/usr/bin/env python3

import os

os.environ["NUMBA_DISABLE_JIT"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

from util_func_erp import run_erp, save_fig_erp
from util_func_connect import (
    run_connect_visual_motor,
    save_fig_connect_visual_motor,
    run_connect_sensorwide_dynamics,
    save_fig_connect_sensorwide_dynamics,
)
from util_func_mvpa import (
    run_mvpa_time_resolved,
    save_fig_mvpa_time_resolved,
    run_mvpa_temporal_generalization_within_day,
    run_mvpa_temporal_generalization_cross_day,
    save_fig_mvpa_temporal_generalization_within_day,
    save_fig_mvpa_temporal_generalization_cross_day,
)


if __name__ == "__main__":
    # Re-epoching is intentionally excluded; analyses start from ../EEG_epo.

    # ERP
    # run_erp()
    # save_fig_erp()

    # Connectivity: visual-motor
    # run_connect_visual_motor()
    # save_fig_connect_visual_motor()

    # Connectivity: sensor-wide dynamics
    # run_connect_sensorwide_dynamics()
    # save_fig_connect_sensorwide_dynamics()

    # MVPA time-resolved
    run_mvpa_time_resolved()
    save_fig_mvpa_time_resolved()

    # MVPA temporal generalization: time x time
    # run_mvpa_temporal_generalization_within_day()
    # save_fig_mvpa_temporal_generalization_within_day()

    # MVPA temporal generalization: day x day
    # run_mvpa_temporal_generalization_cross_day()
    # save_fig_mvpa_temporal_generalization_cross_day()
