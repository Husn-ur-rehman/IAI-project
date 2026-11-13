"""Evaluation maths and the calibration / evaluation split."""
import numpy as np
import pytest

from cheatdetect.calibrate import is_calibration_match
from cheatdetect.evaluate import pr_curve, scores


def test_scores():
    y = np.array([True, True, False, False, True])
    p = np.array([True, False, True, False, True])
    s = scores(y, p)
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (2, 1, 1, 1)
    assert s["precision"] == pytest.approx(2 / 3) and s["recall"] == pytest.approx(2 / 3)
    assert s["fpr"] == pytest.approx(0.5)


def test_pr_curve_handles_ties():
    y = [True, False, True, False]
    score = [0.9, 0.5, 0.5, 0.1]          # one tie at 0.5
    prec, rec = pr_curve(y, score)
    assert len(prec) == 3                  # one point per distinct score
    assert prec[0] == 1.0 and rec[0] == 0.5
    assert prec[1] == pytest.approx(2 / 3) and rec[1] == 1.0


def test_calibration_split():
    assert is_calibration_match("n2") and is_calibration_match("n478")
    assert not is_calibration_match("n3")
    assert not is_calibration_match("c2")   # cheater matches never calibrate
