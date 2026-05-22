#!/usr/bin/env python3
"""Write trial-level boundary-distance outputs."""

import pandas as pd

from util_boundary_distance import OUTPUT_DIR, load_behaviour_with_boundary


if __name__ == "__main__":
    beh, boundary = load_behaviour_with_boundary()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([boundary]).to_csv(
        OUTPUT_DIR / "boundary_distance_model_params.csv", index=False
    )
    beh.to_csv(OUTPUT_DIR / "boundary_distance_behaviour_trial_level.csv", index=False)
