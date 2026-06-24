from pathlib import Path
import sys

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "code"))


def test_line_probability_matches_monte_carlo_seeded():
    from decision_bound_strategy_analysis import _line_probability_b

    rng = np.random.default_rng(11)
    xy = np.array([[55.0, 45.0]], dtype=float)
    a1, a2, b = 1.0, -1.0, 0.0
    sigma = 10.0
    closed = float(_line_probability_b(xy, a1, a2, b, sigma, polarity=1)[0])

    draws = rng.normal(loc=xy[0], scale=sigma, size=(250000, 2))
    decision = a1 * draws[:, 0] + a2 * draws[:, 1] + b
    monte_carlo = float(np.mean(decision >= 0.0))

    assert abs(closed - monte_carlo) < 0.005


def test_line_probability_polarity_complements():
    from decision_bound_strategy_analysis import _line_probability_b

    xy = np.array([[40.0, 60.0], [60.0, 40.0]], dtype=float)
    p_pos = _line_probability_b(xy, 1.0, -1.0, 0.0, 8.0, polarity=1)
    p_neg = _line_probability_b(xy, 1.0, -1.0, 0.0, 8.0, polarity=-1)

    assert np.allclose(p_pos + p_neg, 1.0)
