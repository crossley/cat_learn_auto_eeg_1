#!/usr/bin/env python3

import os

os.environ["NUMBA_DISABLE_JIT"] = "1"

from util_func_epo import util_epo_make_from_bdf
from util_func_wrangle import util_wrangle_load_sessions
from util_func_erp import util_erp_make_figures
from util_func_connect import util_connect_compute_visual_motor
from util_func_mvpa import (
    util_mvpa_time_resolved,
    util_mvpa_temporal_generalization,
)


if __name__ == "__main__":

    util_epo_make_from_bdf()

    # print("\nBuilding ERP figures from preprocessed epochs...")
    # erp_outputs = util_erp_make_figures()

    # print("\nComputing visual-motor functional connectivity across days...")
    # connectivity_outputs = util_connect_compute_visual_motor()

    # print("\nComputing time-resolved MVPA category decoding across days...")
    # mvpa_outputs = util_mvpa_time_resolved()

    # print("\nComputing temporal-generalization MVPA (within-day and cross-day)...")
    # tg_outputs = util_mvpa_temporal_generalization(n_workers=4)

    # pass
