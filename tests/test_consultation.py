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


VALID = dict(rounds_played=20, kills=12, deaths=14, assists=4, shots_fired=300, gun_hits=60, head_hits=15,
             damage=1500, headshot_kills=8, wallbang_kills=4)


def only_problem(**change):
    """The single problem raised when one field of a valid player is changed."""
    p = PlayerInput(**{**VALID, **change}).problems()
    assert len(p) == 1, p
    return p[0]


def test_base_player_is_valid():
    assert PlayerInput(**VALID).problems() == []


def test_rule_1_rounds_cap_allows_overtime():
    assert PlayerInput(**{**VALID, "rounds_played": 32}).problems() == []      # real overtime match length
    assert only_problem(rounds_played=37).startswith("Rounds played cannot exceed 36")


def test_rule_2_each_kill_type_on_its_own_not_the_sum():
    # 9 headshot kills + 4 through walls = 13 > 12 kills, but kills can be both: valid
    assert PlayerInput(**{**VALID, "headshot_kills": 9, "head_hits": 15}).problems() == []
    assert only_problem(blind_kills=13) == "Kills while flashed cannot be more than kills."
    assert only_problem(wallbang_kills=13) == "Kills through walls cannot be more than kills."


def test_rule_3_deaths_at_most_one_per_round():
    assert only_problem(deaths=21).startswith("Deaths cannot exceed rounds played (20)")


def test_rule_4_kills_plus_assists_at_most_five_per_round():
    assert PlayerInput(**{**VALID, "assists": 88}).problems() == []            # 12 + 88 = 100 = 5 x 20
    assert only_problem(assists=89).startswith("Kills plus assists cannot exceed 100")


def test_rule_5_damage_at_most_500_per_round():
    assert only_problem(damage=10_001).startswith("Total damage cannot exceed 10000")


def test_rule_6_headshot_kills_need_headshot_hits():
    assert only_problem(head_hits=7).startswith("Headshot kills cannot be more than headshot hits")


def test_damage_at_least_hits():
    assert only_problem(damage=59).startswith("Total damage must be at least the number of hits")


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
