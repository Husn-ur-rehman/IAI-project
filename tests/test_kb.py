"""Knowledge-base behaviour on synthetic players with known expected verdicts
(fixture `kb` in conftest.py)."""
import pytest

from cheatdetect import prolog_io


def assess(e, pid):
    r = list(e.query(f"assess({pid}, A, M, CF, D, DCF)"))[0]
    return r["A"], r["M"], float(r["CF"]), r["D"]


def test_mycin_combination(kb):
    assert float(list(kb.query("combine(0.6, 0.5, Z)"))[0]["Z"]) == pytest.approx(0.8)
    assert float(list(kb.query("combine(0.6, -0.4, Z)"))[0]["Z"]) == pytest.approx(0.2 / 0.6)


def test_blatant_aimbot_goes_to_human_review(kb):
    action, mode, cf, diag = assess(kb, "x_aim")
    assert (action, mode, diag) == ("review_for_ban", "human", "aimbot")
    assert cf >= 0.75


def test_normal_player_cleared(kb):
    action, mode, cf, diag = assess(kb, "x_norm")
    assert (action, mode, diag) == ("clear", "automatic", "none")


def test_insufficient_evidence(kb):
    assert assess(kb, "x_low")[0] == "no_action"


def test_legit_outlier_is_not_punished(kb):
    action, _, cf, _ = assess(kb, "x_pro")
    assert action == "clear" and cf == 0.0
    legit = float(list(kb.query("prove(x_pro, legit_outlier, CF, _)"))[0]["CF"])
    assert legit > 0.5


def test_wallhack_diagnosed(kb):
    action, mode, _, diag = assess(kb, "x_wall")
    assert diag == "wallhack" and mode == "automatic" and action != "clear"


def test_explanations(kb):
    assess(kb, "x_aim")
    why = prolog_io.lines("why(x_aim, L)")
    assert why[0].startswith("Action: review_for_ban (human)")
    assert any("rule a1" in l for l in why)
    assess(kb, "x_norm")
    why_not = prolog_io.lines("why_not(x_norm, aimbot, L)")
    assert any("not fired" in l for l in why_not)


def test_labels_never_reach_the_kb(tmp_path):
    import pandas as pd

    players = pd.DataFrame([dict(pid="c1_P1", rounds_played=1, kills=0, deaths=0, assists=0,
                                 shots_fired=0, gun_hits=0, head_hits=0, damage=0,
                                 flash_assists=0, enemies_flashed=0, is_cheater=True)])
    kills = pd.DataFrame(columns=["pid", "weapon", "headshot", "thrusmoke", "penetrated",
                                  "attacker_blind", "noscope", "in_air", "distance"])
    text = prolog_io.write_facts(players, kills, tmp_path / "f.pl").read_text()
    assert "cheat" not in text.lower()
