from pathlib import Path
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))


def _synthetic_dataset(subject, day, feature_kind):
    from sequence_feature_interface import SequenceDataset

    rng = np.random.default_rng(subject * 100 + day)
    n_trials = 24
    n_times = 8
    n_features = 4
    y = np.array([0, 1] * (n_trials // 2), dtype=int)
    X = rng.normal(0.0, 0.5, size=(n_trials, n_times, n_features)).astype(np.float32)
    X[y == 1, 3:, 0] += 0.8
    X[y == 0, 3:, 0] -= 0.8
    metadata = pd.DataFrame(
        {
            "subject": subject,
            "session": day,
            "day": day,
            "trial_index_aligned": np.arange(n_trials),
        }
    )
    return SequenceDataset(
        X=X,
        y=y,
        time=np.linspace(-0.1, 0.4, n_times),
        feature_names=[f"f{i}" for i in range(n_features)],
        metadata=metadata,
        subject=subject,
        session=day,
        session_file=f"sub_{subject:03d}_day_{day}.fif",
        feature_kind=feature_kind,
    )


def test_sequence_rnn_smoke_outputs(monkeypatch, tmp_path):
    import sequence_rnn_analysis as sra

    sessions = [
        {"subject": 1, "day": 1, "epo_file": "s1d1.fif"},
        {"subject": 1, "day": 2, "epo_file": "s1d2.fif"},
    ]

    def fake_load_sessions(load_epochs=False):
        assert load_epochs is False
        return sessions

    def fake_load_feature_sequence(item, feature_kind, **kwargs):
        return _synthetic_dataset(int(item["subject"]), int(item["day"]), feature_kind)

    monkeypatch.setattr(sra, "load_sequence_sessions", fake_load_sessions)
    monkeypatch.setattr(sra, "load_feature_sequence", fake_load_feature_sequence)

    result = sra.run_sequence_rnn_analysis(
        output_dir=tmp_path,
        feature_kinds=["voltage"],
        smoke=True,
        models="bag_logreg",
        prefix_fractions="0.5,1.0",
        min_trials=10,
        min_class_trials=4,
        cv_splits=3,
        max_sessions=2,
    )

    assert result["subject_csv"].exists()
    assert result["group_csv"].exists()
    assert result["qc_csv"].exists()
    subject_df = pd.read_csv(result["subject_csv"])
    group_df = pd.read_csv(result["group_csv"])
    assert {"within_session_cv", "cross_session_transfer"} <= set(subject_df["evaluation"])
    assert set(subject_df["model"]) == {"bag_logreg"}
    assert {0.5, 1.0} <= set(group_df["prefix_fraction"].round(3))
