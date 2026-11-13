"""Bridge: one row per player, everything comes from the KB, the dial works."""
import pytest

from cheatdetect import bridge
from conftest import PLAYERS

PIDS = list(PLAYERS)


def test_one_row_per_player_without_labels(kb):
    v = bridge.assess_all(PIDS)
    assert v.pid.tolist() == PIDS
    assert "is_cheater" not in v.columns                       # labels only via attach_labels
    assert {"action", "decision_mode", "cheating_cf", "diagnosis", "cf_aimbot",
            "cf_wallhack", "cf_rage", "cf_legit", "symptoms", "explanation"} <= set(v.columns)


def test_rows_carry_kb_answers(kb):
    v = bridge.assess_all(PIDS).set_index("pid")
    assert v.loc["x_aim", "action"] == "review_for_ban"
    assert v.loc["x_aim", "decision_mode"] == "human"
    assert "abnormal(headshot_rate)@p99" in v.loc["x_aim", "symptoms"]
    assert v.loc["x_aim", "explanation"][0].startswith("Action: review_for_ban")
    assert v.loc["x_norm", "explanation"][0].startswith("cheating: CF")  # why-not for cleared
    assert v.loc["x_pro", "cf_legit"] > 0.5


def test_strictness_dial(kb):
    bridge.assess_all(PIDS)
    try:
        bridge.set_action_thresholds(review_for_ban=0.99, hide_from_leaderboard=0.5, monitor=0.2)
        assert bridge.action_thresholds()["review_for_ban"] == 0.99
        after = bridge.redecide(PIDS).set_index("pid")
        assert after.loc["x_aim", "action"] == "hide_from_leaderboard"   # no longer reaches review
        assert after.loc["x_aim", "decision_mode"] == "automatic"
    finally:
        bridge.reset_action_thresholds()
    assert bridge.redecide(["x_aim"]).action[0] == "review_for_ban"


def test_dial_rejects_inconsistent_thresholds(kb):
    with pytest.raises(ValueError):
        bridge.set_action_thresholds(review_for_ban=0.3, hide_from_leaderboard=0.5, monitor=0.2)
