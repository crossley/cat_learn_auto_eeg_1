from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))


def test_time_frequency_erp_like_synthetic_tables():
    from sequence_feature_interface import SequenceDataset
    from time_frequency_erp_like_analysis import (
        _condition_contrasts,
        compute_session_tables,
    )

    times = np.array([-0.1, 0.0, 0.1, 0.3, 0.6], dtype=float)
    channels = ["POz", "Oz", "Cz"]
    freqs = [4.0, 8.0, 12.0]
    feature_names = [f"{ch}_{freq:g}Hz" for ch in channels for freq in freqs]
    X = np.ones((4, len(times), len(feature_names)), dtype=float)
    X[0:2] += 0.5
    y = np.array([0, 0, 1, 1], dtype=int)
    dataset = SequenceDataset(
        X=X,
        y=y,
        time=times,
        feature_names=feature_names,
        metadata=pd.DataFrame({"cat": ["A", "A", "B", "B"]}),
        subject=1,
        session=1,
        session_file="synthetic-epo.fif",
        feature_kind="time_frequency",
    )

    tables = compute_session_tables(dataset)
    assert set(tables) == {"condition", "window", "timecourse"}
    assert {"cat_a", "cat_b"} == set(tables["condition"]["condition"])
    assert {"electrode", "strict_roi"} <= set(tables["window"]["summary_level"])
    assert "visual" in set(tables["window"]["roi"])

    contrast = _condition_contrasts(tables["window"])
    visual = contrast[
        (contrast["summary_level"] == "strict_roi")
        & (contrast["roi"] == "visual")
        & (contrast["band"] == "alpha")
        & (contrast["window"] == "early")
    ]
    assert not visual.empty
    assert np.allclose(visual["power_diff"].to_numpy(dtype=float), 0.5)
