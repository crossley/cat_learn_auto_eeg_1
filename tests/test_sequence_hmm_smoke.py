from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))


def test_sequence_hmm_synthetic_smoke():
    from sequence_feature_interface import SequenceDataset
    from sequence_hmm_analysis import fit_state_models_for_dataset

    rng = np.random.default_rng(7)
    n_trials = 12
    n_times = 8
    X = rng.normal(size=(n_trials, n_times, 3))
    X[:, 4:, 0] += 2.0
    y = np.array([0, 1] * 6)
    dataset = SequenceDataset(
        X=X,
        y=y,
        time=np.linspace(-0.1, 0.6, n_times),
        feature_names=["f0", "f1", "f2"],
        metadata=pd.DataFrame({"trial_index_aligned": np.arange(n_trials)}),
        subject=1,
        session=1,
        session_file="synthetic-epo.fif",
        feature_kind="voltage",
    )

    result = fit_state_models_for_dataset(
        dataset,
        state_grid=range(2, 4),
        max_components=3,
        random_state=3,
    )

    assert len(result["model_rows"]) == 2
    assert any(row["selected_by_bic"] for row in result["model_rows"])
    assert len(result["time_rows"]) > 0
    assert len(result["trial_rows"]) > 0
    assert len(result["dwell_rows"]) > 0
    assert len(result["transition_rows"]) > 0
