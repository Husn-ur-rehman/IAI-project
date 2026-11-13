"""Fact extraction (data abstraction) layer.

Turns raw CS2CD event logs into the base facts the Prolog knowledge base reasons over.
Python only parses and COUNTS here. It derives no rates and makes no judgements;
the KB does that (see kb/).

Outputs (data/processed/):
  players.parquet  one row per player per match: raw counts + context + ground-truth label
  kills.parquet    one row per kill, asserted into Prolog as kill/9 facts

The `is_cheater` label is ground truth for EVALUATION ONLY and is never asserted into the KB.

Usage: python -m cheatdetect.facts [--limit N]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from cheatdetect import config

# Weapons that fire bullets. Grenades, knives, zeus and fire are excluded from shot and hit counts.
# Shotguns are excluded because one trigger pull fires many pellets, which breaks hits/shots.
SHOTGUNS = {"nova", "xm1014", "mag7", "sawedoff"}
NON_GUNS = {
    "hegrenade", "flashbang", "smokegrenade", "molotov", "incgrenade", "decoy",
    "inferno", "taser", "world", "c4", "planted_c4",
}


def _weapon(name: str) -> str:
    return name.removeprefix("weapon_")


def is_gun(weapon: str) -> bool:
    w = _weapon(weapon)
    return not (w in NON_GUNS or w.startswith("knife") or w == "bayonet" or w in SHOTGUNS)


def match_id_for(path: Path) -> str:
    prefix = "c" if path.parent.name == "with_cheater_present" else "n"
    return f"{prefix}{path.stem}"


def extract_match(path: Path) -> tuple[list[dict], list[dict]]:
    """Return (player rows, kill rows) for one match file."""
    d = json.loads(path.read_text(encoding="utf-8"))
    mid = match_id_for(path)

    start = min((e["tick"] for e in d.get("round_announce_match_start", [])), default=0)
    after = lambda events: [e for e in events if e.get("tick", 0) >= start]  # noqa: E731

    bots = {e["user_steamid"] for e in d.get("player_team", []) if e.get("isbot")}
    cheaters = {c["steamid"] for c in d.get("cheaters", [])}
    info = (d.get("CSstats_info") or [{}])[0]
    ranks = {e["user_steamid"]: e for e in d.get("rank_update", [])}

    players: dict[str, dict] = {}

    def row(sid: str) -> dict:
        if sid not in players:
            players[sid] = dict(
                kills=0, deaths=0, assists=0, rounds_played=0,
                shots_fired=0, gun_hits=0, head_hits=0, damage=0,
                flash_assists=0, enemies_flashed=0,
            )
        return players[sid]

    def valid(sid: str) -> bool:
        return bool(sid) and sid not in bots

    for e in after(d.get("player_spawn", [])):
        if valid(e["user_steamid"]):
            row(e["user_steamid"])["rounds_played"] += 1

    for e in after(d.get("weapon_fire", [])):
        if valid(e["user_steamid"]) and is_gun(e["weapon"]):
            row(e["user_steamid"])["shots_fired"] += 1

    for e in after(d.get("player_hurt", [])):
        att, vic = e.get("attacker_steamid"), e.get("user_steamid")
        if not valid(att) or att == vic:
            continue
        r = row(att)
        r["damage"] += e.get("dmg_health", 0)
        if is_gun(e["weapon"]):
            r["gun_hits"] += 1
            r["head_hits"] += e.get("hitgroup") == "head"

    kills = []
    for e in after(d.get("player_death", [])):
        att, vic = e.get("attacker_steamid"), e.get("user_steamid")
        if valid(vic):
            row(vic)["deaths"] += 1
        if valid(e.get("assister_steamid")):
            r = row(e["assister_steamid"])
            r["assists"] += 1
            r["flash_assists"] += bool(e.get("assistedflash"))
        if not valid(att) or att == vic:
            continue
        row(att)["kills"] += 1
        kills.append(dict(
            pid=f"{mid}_{att}", match_id=mid, tick=e["tick"], weapon=_weapon(e["weapon"]),
            gun=is_gun(e["weapon"]), headshot=bool(e.get("headshot")),
            thrusmoke=bool(e.get("thrusmoke")), penetrated=int(e.get("penetrated", 0)) > 0,
            attacker_blind=bool(e.get("attackerblind")), noscope=bool(e.get("noscope")),
            in_air=bool(e.get("attackerinair")), distance=float(e.get("distance", 0.0)),
        ))

    for e in after(d.get("player_blind", [])):
        att, vic = e.get("attacker_steamid"), e.get("user_steamid")
        if valid(att) and att != vic and e.get("blind_duration", 0) > 1.0:
            row(att)["enemies_flashed"] += 1  # includes teammates; the dataset has no reliable team info

    rows = []
    for sid, r in players.items():
        if r["rounds_played"] == 0:
            continue  # never actually played (spectator or disconnected before start)
        rk = ranks.get(sid, {})
        rows.append(dict(
            pid=f"{mid}_{sid}", match_id=mid, steamid=sid,
            map=info.get("map"), mode=info.get("match_making_type"),
            rating=rk.get("rank_old"), rating_change=rk.get("rank_change"), wins_total=rk.get("num_wins"),
            **r,
            is_cheater=sid in cheaters,
        ))
    return rows, kills


def match_files(limit: int | None = None) -> list[Path]:
    files = sorted(config.CHEATER_DIR.glob("*.json")) + sorted(config.CLEAN_DIR.glob("*.json"))
    if limit:
        files = sorted(config.CHEATER_DIR.glob("*.json"))[:limit] + sorted(config.CLEAN_DIR.glob("*.json"))[:limit]
    return files


def build(limit: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_players, all_kills, errors = [], [], []
    for path in match_files(limit):
        try:
            p, k = extract_match(path)
        except Exception as exc:  # keep going; report at the end
            errors.append((path.name, repr(exc)))
            continue
        all_players += p
        all_kills += k

    players = pd.DataFrame(all_players)
    kills = pd.DataFrame(all_kills)

    int_cols = ["kills", "deaths", "assists", "rounds_played", "shots_fired", "gun_hits",
                "head_hits", "damage", "flash_assists", "enemies_flashed"]
    players[int_cols] = players[int_cols].astype("int32")
    players[["rating", "rating_change", "wins_total"]] = players[["rating", "rating_change", "wins_total"]].astype("float32")
    for c in ["match_id", "map", "mode"]:
        players[c] = players[c].astype("category")
    kills["distance"] = kills["distance"].astype("float32")
    kills["tick"] = kills["tick"].astype("int32")
    kills["weapon"] = kills["weapon"].astype("category")

    if errors:
        print(f"{len(errors)} files failed:")
        for name, err in errors:
            print(" ", name, err)
    return players, kills


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="only N matches from each folder (dev mode)")
    args = ap.parse_args()

    players, kills = build(args.limit)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    players.to_parquet(config.PLAYERS_FILE, index=False)
    kills.to_parquet(config.KILLS_FILE, index=False)
    print(f"players: {len(players)} rows ({players.is_cheater.sum()} cheaters) -> {config.PLAYERS_FILE}")
    print(f"kills:   {len(kills)} rows -> {config.KILLS_FILE}")


if __name__ == "__main__":
    main()
