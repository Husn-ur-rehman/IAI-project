"""Calibration must only ever look at players from clean matches."""
import pandas as pd

from cheatdetect.calibrate import compute_thresholds


def test_thresholds_use_only_calibration_half_of_clean_matches():
    clean = [("n2_P%d" % i, "accuracy", i / 100, "n2") for i in range(100)]
    held_out = [("n3_P%d" % i, "accuracy", 0.01, "n3") for i in range(500)]
    cheat = [("c1_P%d" % i, "accuracy", 0.99, "c1") for i in range(1000)]
    values = pd.DataFrame(clean + held_out + cheat, columns=["pid", "metric", "value", "match_id"])

    t = compute_thresholds(values)

    assert t.loc["accuracy", "n_players"] == 100
    assert abs(t.loc["accuracy", "p95"] - 0.9405) < 1e-6   # unaffected by cheaters and by the held-out half
