"""Fact-extraction tests on a tiny hand-made match."""
import json

from cheatdetect.facts import extract_match, is_gun


def test_is_gun():
    assert is_gun("weapon_ak47") and is_gun("awp")
    assert not is_gun("weapon_knife_t") and not is_gun("hegrenade") and not is_gun("nova")


def test_extract_match(tmp_path):
    folder = tmp_path / "with_cheater_present"
    folder.mkdir()
    match = {
        "round_announce_match_start": [{"tick": 10}],
        "player_team": [{"user_steamid": "Player_9", "isbot": True}],
        "cheaters": [{"steamid": "Player_1"}],
        "player_spawn": [{"tick": 5, "user_steamid": "Player_1"},  # warmup, ignored
                         {"tick": 20, "user_steamid": "Player_1"},
                         {"tick": 20, "user_steamid": "Player_2"},
                         {"tick": 20, "user_steamid": "Player_9"}],
        "weapon_fire": [{"tick": 30, "user_steamid": "Player_1", "weapon": "weapon_ak47"},
                        {"tick": 31, "user_steamid": "Player_1", "weapon": "weapon_ak47"},
                        {"tick": 32, "user_steamid": "Player_1", "weapon": "weapon_hegrenade"}],
        "player_hurt": [{"tick": 25, "attacker_steamid": "Player_1", "user_steamid": "Player_2",
                         "weapon": "ak47", "hitgroup": "chest", "dmg_health": 30, "health": 70},
                        # lethal headshot reports overkill (140) but only 70 HP was left
                        {"tick": 30, "attacker_steamid": "Player_1", "user_steamid": "Player_2",
                         "weapon": "ak47", "hitgroup": "head", "dmg_health": 140, "health": 0}],
        "player_death": [{"tick": 30, "attacker_steamid": "Player_1", "user_steamid": "Player_2",
                          "assister_steamid": "", "weapon": "ak47", "headshot": True,
                          "penetrated": 1, "distance": 12.5}],
    }
    path = folder / "7.json"
    path.write_text(json.dumps(match))

    players, kills = extract_match(path)
    by_id = {p["steamid"]: p for p in players}

    assert set(by_id) == {"Player_1", "Player_2"}          # bot dropped
    p1 = by_id["Player_1"]
    assert p1["pid"] == "c7_Player_1" and p1["is_cheater"]
    assert p1["rounds_played"] == 1                          # warmup spawn ignored
    assert p1["shots_fired"] == 2                            # grenade not counted
    assert (p1["gun_hits"], p1["head_hits"], p1["damage"], p1["kills"]) == (2, 1, 100, 1)  # overkill capped
    assert by_id["Player_2"]["deaths"] == 1 and not by_id["Player_2"]["is_cheater"]
    assert len(kills) == 1 and kills[0]["headshot"] and kills[0]["penetrated"]
