"""Shared fixture: the KB loaded with synthetic players and fixed test thresholds,
so tests do not depend on the dataset or on the calibrated thresholds."""
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
    list(e.query("reset_action_thresholds"))
    prolog_io.reload_kb()  # restore the real thresholds
