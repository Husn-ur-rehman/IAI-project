"""Calibration must only ever look at players from clean matches."""
import pandas as pd

from cheatdetect.calibrate import compute_thresholds


def test_thresholds_ignore_cheater_matches():
    clean = [("n1_P%d" % i, "accuracy", i / 100, "n1") for i in range(100)]
    cheat = [("c1_P%d" % i, "accuracy", 0.99, "c1") for i in range(1000)]
    values = pd.DataFrame(clean + cheat, columns=["pid", "metric", "value", "match_id"])

    t = compute_thresholds(values)

    assert t.loc["accuracy", "n_players"] == 100
    assert abs(t.loc["accuracy", "p95"] - 0.9405) < 1e-6   # unaffected by the 1000 cheaters at 0.99
