#!/usr/bin/env python3
"""Write short-window feedback-locked RSA outputs."""

from rsa_time_resolved_stim_locked import run_rsa_windowed


if __name__ == "__main__":
    run_rsa_windowed(
        save_figures=False,
        output_prefix="rsa_feedback_windowed",
        event_names=("FB/Cor", "FB/Inc"),
        log_label="RSA feedback windowed",
    )
