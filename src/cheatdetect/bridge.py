"""Bridge: ask the knowledge base about every player and collect the answers into one table.

Python makes no decisions here; every action, CF, diagnosis and explanation comes from Prolog.
The ground-truth label is attached only AFTER the KB has answered (attach_labels).

Usage: python -m cheatdetect.bridge     -> data/processed/verdicts.parquet
"""
from __future__ import annotations

import time

import pandas as pd

from cheatdetect import config, prolog_io
from cheatdetect.prolog_io import atom, text

DIAGNOSES = {"cf_aimbot": "aimbot", "cf_wallhack": "wallhack",
             "cf_rage": "rage_cheat", "cf_legit": "legit_outlier"}

ACTIONS = ["review_for_ban", "hide_from_leaderboard", "monitor"]  # strictest first


def prepare(write_facts: bool = True):
    """Load the KB, calibrated thresholds and base facts into the Prolog engine."""
    if write_facts or not config.FACTS_FILE.exists():
        players = pd.read_parquet(config.PLAYERS_FILE)
        kills = pd.read_parquet(config.KILLS_FILE)
        prolog_io.write_facts(players, kills)
    prolog_io.reload_kb()
    prolog_io.load_facts()


def assess_one(pid: str) -> dict:
    e = prolog_io.engine()
    p = atom(pid)
    cf_goals = ", ".join(f"prove({p}, {g}, {v.upper()}, _)" for v, g in DIAGNOSES.items())
    goal = (f"assess({p}, A, M, CF, D, DCF), {cf_goals}, "
            f"symptom_labels({p}, S), explanation({p}, A, L)")
    r = list(e.query(goal, maxresult=1))[0]
    return dict(
        pid=pid,
        action=text(r["A"]),
        decision_mode=text(r["M"]),
        cheating_cf=float(r["CF"]),
        diagnosis=text(r["D"]),
        diagnosis_cf=float(r["DCF"]),
        **{v: float(r[v.upper()]) for v in DIAGNOSES},
        symptoms=[text(s) for s in r["S"]],
        explanation=[text(l) for l in r["L"]],
    )


def assess_all(pids: list[str]) -> pd.DataFrame:
    return pd.DataFrame([assess_one(p) for p in pids])


# ---------- strictness dial ---------------------------------------------------

def set_action_thresholds(review_for_ban: float, hide_from_leaderboard: float, monitor: float):
    """Change the action thresholds inside the KB. They must not increase from review to monitor."""
    if not (review_for_ban >= hide_from_leaderboard >= monitor >= 0):
        raise ValueError("need review_for_ban >= hide_from_leaderboard >= monitor >= 0")
    e = prolog_io.engine()
    for action, t in zip(ACTIONS, (review_for_ban, hide_from_leaderboard, monitor)):
        list(e.query(f"set_action_threshold({action}, {float(t)})"))


def reset_action_thresholds():
    list(prolog_io.engine().query("reset_action_thresholds"))


def action_thresholds() -> dict[str, float]:
    return {text(r["A"]): float(r["T"]) for r in prolog_io.engine().query("action_threshold(A, T)")}


def redecide(pids: list[str]) -> pd.DataFrame:
    """Ask the KB to decide again under the current thresholds.
    Symptoms from the last assess are reused (thresholds only affect layer 3)."""
    e = prolog_io.engine()
    rows = []
    for pid in pids:
        r = list(e.query(f"decide({atom(pid)}, A, _, _), action_mode(A, M)", maxresult=1))[0]
        rows.append(dict(pid=pid, action=text(r["A"]), decision_mode=text(r["M"])))
    return pd.DataFrame(rows)


# ---------- labels (evaluation only) -------------------------------------------

def attach_labels(verdicts: pd.DataFrame) -> pd.DataFrame:
    players = pd.read_parquet(config.PLAYERS_FILE)
    ctx = players[["pid", "match_id", "steamid", "mode", "map", "rating", "is_cheater"]]
    ctx = ctx.rename(columns={"mode": "match_mode"})
    return verdicts.merge(ctx, on="pid", how="left", validate="one_to_one")


def main():
    t0 = time.perf_counter()
    prepare()
    pids = pd.read_parquet(config.PLAYERS_FILE, columns=["pid"]).pid.tolist()
    verdicts = attach_labels(assess_all(pids))
    verdicts.to_parquet(config.VERDICTS_FILE, index=False)
    print(f"{len(verdicts)} verdicts in {time.perf_counter() - t0:.1f}s -> {config.VERDICTS_FILE}")
    print(verdicts.action.value_counts().to_string())


if __name__ == "__main__":
    main()
