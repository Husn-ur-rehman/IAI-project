"""Knowledge-base behaviour on synthetic players with known expected verdicts.

The real calibrated thresholds are swapped for fixed test thresholds, so these tests
do not depend on the dataset.
"""
import pytest

from cheatdetect import prolog_io

TEST_THRESHOLDS = {  # metric: (p95, p99)
    "headshot_rate": (0.4, 0.7), "accuracy": (0.3, 0.4), "adr": (135, 180),
    "kpr": (1.35, 1.9), "kd": (2.8, 7), "hs_kill_rate": (0.8, 0.92),
    "wallbang_rate": (0.2, 0.78), "smoke_kill_rate": (0.15, 0.25),
    "blind_kill_rate": (0.09, 0.14), "air_kill_rate": (0.05, 0.13),
}


def player(pid, rounds, kills, deaths, shots, hits, heads, damage, hs_kills=0, wallbangs=0):
    stats = dict(rounds_played=rounds, kills=kills, deaths=deaths, assists=0, shots_fired=shots,
                 gun_hits=hits, head_hits=heads, damage=damage, flash_assists=0, enemies_flashed=0)
    lines = [f"stat({pid},{k},{v})." for k, v in stats.items()]
    for i in range(kills):
        hs = "true" if i < hs_kills else "false"
        wb = "true" if i < wallbangs else "false"
        lines.append(f"kill({pid},ak47,{hs},false,{wb},false,false,false,10.0).")
    return lines


PLAYERS = {
    # blatant aimbot: extreme headshots and accuracy, huge impact
    "x_aim": player("x_aim", 20, 40, 5, 300, 200, 160, 4000, hs_kills=40),
    # ordinary player
    "x_norm": player("x_norm", 20, 15, 15, 300, 60, 12, 1500, hs_kills=6),
    # too little data to judge
    "x_low": player("x_low", 3, 1, 3, 10, 2, 0, 50),
    # strong but legit: high impact, normal aim
    "x_pro": player("x_pro", 20, 38, 10, 400, 100, 30, 3800, hs_kills=20),
    # wallbang-heavy player with normal aim
    "x_wall": player("x_wall", 20, 30, 8, 300, 80, 25, 2900, hs_kills=12, wallbangs=24),
}


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    path = tmp_path_factory.mktemp("kb") / "facts.pl"
    path.write_text("\n".join(l for ls in PLAYERS.values() for l in ls) + "\n")
    e = prolog_io.engine()
    prolog_io.reload_kb()
    list(e.query("retractall(threshold(_,_,_))"))
    for m, (p95, p99) in TEST_THRESHOLDS.items():
        e.assertz(f"threshold({m}, p95, {p95})")
        e.assertz(f"threshold({m}, p99, {p99})")
    prolog_io.load_facts(path)
    yield e
    prolog_io.reload_kb()  # restore the real thresholds


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
