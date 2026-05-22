#!/usr/bin/env python3
"""Write single-timepoint feedback-locked RSA outputs."""

from util_rsa_time_resolved import run_rsa_time_resolved


if __name__ == "__main__":
    run_rsa_time_resolved(
        output_prefix="rsa_feedback",
        event_names=("FB/Cor", "FB/Inc"),
        log_label="RSA feedback",
    )
