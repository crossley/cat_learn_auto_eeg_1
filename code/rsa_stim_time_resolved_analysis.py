#!/usr/bin/env python3
"""Write single-timepoint stimulus-locked RSA outputs."""

from rsa_time_resolved_stim_locked import run_rsa_time_resolved


if __name__ == "__main__":
    run_rsa_time_resolved(save_figures=False)
