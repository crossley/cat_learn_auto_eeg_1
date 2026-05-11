#!/usr/bin/env python3
"""Run the added TG window, band-envelope TG, and ERP latency analyses."""

import argparse

from util_func_new_analyses import (
    run_all_new_analyses,
    run_band_envelope_cross_day_tg,
    run_cross_day_tg_window_structure,
    run_erp_peak_latency_trajectories,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tg-windows", action="store_true", help="Run cross-day TG window-gradient analysis.")
    parser.add_argument("--erp-latencies", action="store_true", help="Run ERP peak-latency trajectory analysis.")
    parser.add_argument("--band-tg", action="store_true", help="Run band-envelope cross-day TG. This can be slow.")
    parser.add_argument("--all-light", action="store_true", help="Run TG windows and ERP latencies.")
    parser.add_argument("--n-workers", type=int, default=None, help="Workers for band-envelope cross-day TG.")
    args = parser.parse_args()

    if args.all_light or (not args.tg_windows and not args.erp_latencies and not args.band_tg):
        run_all_new_analyses(run_band_tg=False)
    else:
        if args.tg_windows:
            run_cross_day_tg_window_structure()
        if args.erp_latencies:
            run_erp_peak_latency_trajectories()
    if args.band_tg:
        run_band_envelope_cross_day_tg(n_workers=args.n_workers)


if __name__ == "__main__":
    main()
