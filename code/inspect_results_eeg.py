#!/usr/bin/env python3

import os

# os.environ["NUMBA_DISABLE_JIT"] = "1"
# os.environ["OMP_NUM_THREADS"] = "1"
# os.environ["MKL_NUM_THREADS"] = "1"
# os.environ["OPENBLAS_NUM_THREADS"] = "1"
# os.environ["NUMEXPR_NUM_THREADS"] = "1"
# os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
# os.environ["BLIS_NUM_THREADS"] = "1"

from util_func_epo import util_epo_make_from_bdf
from util_func_wrangle import util_wrangle_load_sessions
from util_func_erp import util_erp_make_figures
from util_func_connect import (
    util_connect_compute_visual_motor,
    util_connect_explore_sensorwide_dynamics,
)
from util_func_mvpa import (
    util_mvpa_time_resolved,
    util_mvpa_temporal_generalization,
)


if __name__ == "__main__":

    # util_epo_make_from_bdf()
    # util_erp_make_figures()
    # util_connect_compute_visual_motor()
    util_connect_explore_sensorwide_dynamics()
    # util_mvpa_time_resolved()
    # util_mvpa_temporal_generalization(n_workers=4)

