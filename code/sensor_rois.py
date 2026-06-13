#!/usr/bin/env python3
"""Strict sensor-region definitions shared by ROI analyses."""

STRICT_SENSOR_ROIS = {
    "visual": ["PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2", "Iz"],
    "frontal": ["Fp1", "Fpz", "Fp2", "AF3", "AFz", "AF4", "F3", "F1", "Fz", "F2", "F4"],
    "central": ["FC1", "FCz", "FC2", "C3", "C1", "Cz", "C2", "C4", "CP1", "CPz", "CP2"],
}

ROI_LABELS = {
    "visual": "visual sensors",
    "frontal": "frontal sensors",
    "central": "central sensors",
    "visual_frontal": "visual to frontal",
    "visual_central": "visual to central",
}


def roi_channels(name):
    return list(STRICT_SENSOR_ROIS[name])


def cross_roi_pairs(source, target):
    pairs = []
    for ch_i in STRICT_SENSOR_ROIS[source]:
        for ch_j in STRICT_SENSOR_ROIS[target]:
            if ch_i != ch_j:
                pairs.append(tuple(sorted((ch_i, ch_j))))
    return sorted(set(pairs))
