"""Python <-> Prolog plumbing: write base facts to a .pl file and load the KB."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from cheatdetect import config

STAT_KEYS = ["rounds_played", "kills", "deaths", "assists", "shots_fired",
             "gun_hits", "head_hits", "damage", "flash_assists", "enemies_flashed"]


def atom(s: str) -> str:
    return "'" + str(s).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _b(x) -> str:
    return "true" if x else "false"


def write_facts(players: pd.DataFrame, kills: pd.DataFrame, path: Path = config.FACTS_FILE) -> Path:
    """Write stat/3 and kill/9 facts. The ground-truth label is never written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in players[["pid", *STAT_KEYS]].itertuples(index=False):
            pid = atom(row.pid)
            for key in STAT_KEYS:
                f.write(f"stat({pid},{key},{int(getattr(row, key))}).\n")
        for k in kills.itertuples(index=False):
            f.write(
                f"kill({atom(k.pid)},{atom(k.weapon)},{_b(k.headshot)},{_b(k.thrusmoke)},"
                f"{_b(k.penetrated)},{_b(k.attacker_blind)},{_b(k.noscope)},{_b(k.in_air)},"
                f"{float(k.distance):.2f}).\n"
            )
    return path


@lru_cache(maxsize=1)
def engine():
    """The (process-wide) Prolog engine with the KB consulted."""
    from pyswip import Prolog

    prolog = Prolog()
    prolog.consult(config.KB_MAIN.as_posix())
    return prolog


def reload_kb():
    engine().consult(config.KB_MAIN.as_posix())


def load_facts(path: Path = config.FACTS_FILE):
    list(engine().query(f"load_facts({atom(path.as_posix())})"))
