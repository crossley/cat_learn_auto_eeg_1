from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))


def test_stimulus_generation_columns_and_labels():
    from stimuli import make_stim_cats

    stimuli = make_stim_cats(n_stimuli_per_category=5)

    assert list(stimuli.columns) == ["x", "y", "cat", "condition", "xt", "yt"]
    assert set(stimuli["cat"]) == {"A", "B"}
    assert len(stimuli) == 10


def test_behavioural_filename_contract():
    from load_project_data import BEHAVIOURAL_FILE_RE

    match = BEHAVIOURAL_FILE_RE.match("sub_001_day_100_data.csv")

    assert match is not None
    assert int(match.group(1)) == 1
    assert int(match.group(2)) == 100


def test_align_behaviour_to_epochs_uses_metadata_trial_index():
    from load_project_data import align_behaviour_to_epochs

    behaviour = pd.DataFrame(
        {
            "trial": [0, 1, 2],
            "cat": ["A", "B", "A"],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "xt": [0.1, 0.2, 0.3],
            "yt": [0.4, 0.5, 0.6],
            "resp": ["A", "B", "A"],
            "rt": [500, 600, 700],
            "fb": ["Correct", "Correct", "Incorrect"],
        }
    )

    class FakeSelectedEpochs:
        metadata = pd.DataFrame({"beh_trial_index": [0, 2]})
        selection = np.array([0, 2])

        def __len__(self):
            return 2

    class FakeEpochs:
        event_id = {"Stim/A": 20, "Stim/B": 21}

        def __getitem__(self, event_names):
            assert event_names == ["Stim/A", "Stim/B"]
            return FakeSelectedEpochs()

    selected_epochs, aligned = align_behaviour_to_epochs(behaviour, FakeEpochs())

    assert len(selected_epochs) == 2
    assert aligned["trial"].tolist() == [0, 2]


def test_analysis_output_roots_are_flat():
    nested_constant_re = re.compile(
        r"^\s*(OUTPUT_DIR|FIGURES_DIR)\s*=\s*PROJECT_DIR\s*/\s*"
        r'"(output|figures)"\s*/'
    )
    offenders = []
    for path in sorted((PROJECT_DIR / "code").glob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), start=1):
            if nested_constant_re.search(line):
                offenders.append(f"{path.relative_to(PROJECT_DIR)}:{line_no}:{line.strip()}")

    assert offenders == []


def test_generated_artifacts_do_not_use_known_output_subdirectories():
    banned_fragments = [
        '/ "cache_stim_arrays"',
        '/ "cache_band_envelope_arrays"',
        '/ "cache_band_signed_arrays"',
        '/ "tg_cross_day_subject_matrices"',
        '/ "progress.json"',
        '/ "qc_skipped.csv"',
        '/ "tg_qc_log.csv"',
    ]
    offenders = []
    for path in sorted((PROJECT_DIR / "code").glob("*.py")):
        text = path.read_text()
        for fragment in banned_fragments:
            if fragment in text:
                offenders.append(f"{path.relative_to(PROJECT_DIR)} contains {fragment}")

    assert offenders == []
