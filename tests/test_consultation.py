"""Consultation: a user-described player is reasoned over and then forgotten."""
import pytest

from cheatdetect import prolog_io
from cheatdetect.consultation import PlayerInput, run_consult


def leftover(kb):
    return list(kb.query("aggregate_all(count, (stat(P,_,_), sub_atom(P,0,_,_,check_)), N)"))[0]["N"]


def test_aimbot_like_player_goes_to_a_human(kb):
    r = run_consult(PlayerInput(rounds_played=20, kills=40, deaths=5, shots_fired=300, gun_hits=200,
                                head_hits=160, damage=4000, headshot_kills=40, wallbang_kills=0))
    assert (r["action"], r["decision_mode"], r["diagnosis"]) == ("review_for_ban", "human", "aimbot")
    assert r["verdict"] == "Very likely cheating"
    assert any("s_abnormal" in l and "headshot_rate" in l for l in r["inference"])   # forward chaining shown
    assert r["proof"][0].startswith("cheating: CF")                                  # backward proof shown
    assert leftover(kb) == 0


def test_typical_player_is_cleared_with_why_not(kb):
    r = run_consult(PlayerInput(rounds_played=20, kills=15, deaths=15, shots_fired=300, gun_hits=60,
                                head_hits=12, damage=1500, headshot_kills=6, wallbang_kills=0))
    assert (r["action"], r["verdict"]) == ("clear", "Looks legit")
    assert any("not fired" in l for l in r["why_not"]["aimbot"])
    assert leftover(kb) == 0


def test_strong_legit_player_counts_as_evidence_against(kb):
    r = run_consult(PlayerInput(rounds_played=20, kills=38, deaths=10, shots_fired=400, gun_hits=100,
                                head_hits=30, damage=3800, headshot_kills=20, wallbang_kills=0))
    assert r["action"] == "clear" and r["cfs"]["legit_outlier"] > 0.5


def test_impossible_input_is_rejected(kb):
    with pytest.raises(ValueError, match="Hits cannot be more than shots"):
        run_consult(PlayerInput(shots_fired=10, gun_hits=50))
    assert PlayerInput(kills=3, headshot_kills=5).problems() == ["Headshot kills cannot be more than kills."]
    assert leftover(kb) == 0


def test_game_limits_are_enforced():
    # the input from the user's screenshot: 1 round, 30 kills, 4 deaths, 20 headshot kills, 19 headshot hits
    p = PlayerInput(rounds_played=1, kills=30, deaths=4, assists=5, shots_fired=394, gun_hits=68,
                    head_hits=19, damage=300, headshot_kills=20, wallbang_kills=1, smoke_kills=1).problems()
    assert any(s.startswith("Kills plus assists cannot exceed 5") for s in p)
    assert any(s.startswith("Deaths cannot exceed rounds played") for s in p)
    assert any(s.startswith("Headshot kills cannot be more than headshot hits") for s in p)
    assert len(p) == 3


def test_app_examples_pass_the_checks():
    import ast

    from cheatdetect import config

    src = (config.ROOT / "app" / "streamlit_app.py").read_text(encoding="utf-8")
    block = src[src.index("EXAMPLES = {"):src.index("def load_example")]
    ns = {"PlayerInput": PlayerInput}
    exec(compile(ast.parse(block), "examples", "exec"), ns)
    assert PlayerInput().problems() == []
    for name, example in ns["EXAMPLES"].items():
        assert example.problems() == [], name


def test_facts_cover_every_kill_flag():
    facts = PlayerInput(kills=3, headshot_kills=2, wallbang_kills=1).facts("p")
    kills = [f for f in facts if f.startswith("kill(")]
    assert len(kills) == 3
    assert sum("ak47, true" in k for k in kills) == 2          # two headshot kills
