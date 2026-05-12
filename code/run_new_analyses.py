#!/usr/bin/env python3
"""Run the added TG window, band-envelope TG, and ERP latency analyses."""

import argparse

from util_func_new_analyses import (
    run_all_new_analyses,
    run_boundary_behaviour_analysis,
    run_boundary_tg_individual_difference_analysis,
    run_band_envelope_cross_day_tg,
    run_band_signed_voltage_cross_day_tg,
    run_cross_day_tg_window_structure,
    run_day1_rw_frn_analysis,
    run_erp_peak_latency_trajectories,
    run_n2_boundary_distance_analysis,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tg-windows", action="store_true", help="Run cross-day TG window-gradient analysis.")
    parser.add_argument("--erp-latencies", action="store_true", help="Run ERP peak-latency trajectory analysis.")
    parser.add_argument("--band-tg", action="store_true", help="Run band-envelope cross-day TG. This can be slow.")
    parser.add_argument("--band-signed-tg", action="store_true", help="Run signed band-limited cross-day TG. This can be slow.")
    parser.add_argument("--boundary-behaviour", action="store_true", help="Run boundary-distance RT/accuracy analysis.")
    parser.add_argument("--n2-boundary", action="store_true", help="Run N2 boundary-distance analysis.")
    parser.add_argument("--day1-rw-frn", action="store_true", help="Run Day 1 RW/RPE FRN analysis.")
    parser.add_argument("--boundary-tg-individual", action="store_true", help="Run boundary-slope by late-TG individual-differences analysis.")
    parser.add_argument("--all-light", action="store_true", help="Run TG windows and ERP latencies.")
    parser.add_argument("--n-workers", type=int, default=None, help="Workers for band-envelope cross-day TG.")
    args = parser.parse_args()

    if args.all_light or (
        not args.tg_windows
        and not args.erp_latencies
        and not args.band_tg
        and not args.band_signed_tg
        and not args.boundary_behaviour
        and not args.n2_boundary
        and not args.day1_rw_frn
        and not args.boundary_tg_individual
    ):
        run_all_new_analyses(run_band_tg=False)
    else:
        if args.tg_windows:
            run_cross_day_tg_window_structure()
        if args.erp_latencies:
            run_erp_peak_latency_trajectories()
        if args.boundary_behaviour:
            run_boundary_behaviour_analysis()
        if args.n2_boundary:
            run_n2_boundary_distance_analysis(n_workers=args.n_workers)
        if args.day1_rw_frn:
            run_day1_rw_frn_analysis(n_workers=args.n_workers)
        if args.boundary_tg_individual:
            run_boundary_tg_individual_difference_analysis()
    if args.band_tg:
        run_band_envelope_cross_day_tg(n_workers=args.n_workers)
    if args.band_signed_tg:
        run_band_signed_voltage_cross_day_tg(n_workers=args.n_workers)


if __name__ == "__main__":
    main()
