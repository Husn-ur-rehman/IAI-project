"""Consultation: ask the expert system about a player the user describes.

The user's numbers become temporary Prolog facts (stat/3, kill/9) for a fresh player id.
The same knowledge base used for the real players reasons over them, and the facts are
removed again afterwards. The result is split by capability:

  inference         what forward chaining derived (metric -> symptom) and the backward proof
  diagnosis         certainty factor per cheat type and the primary diagnosis
  decision making   the action and whether the system takes it alone
  decision support  for a human decision: the recommendation, the evidence, and what
                    would change the verdict

Functions prefixed `run_` must be called on the Prolog thread (KBService does this).
"""
from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass

from cheatdetect import bridge, prolog_io
from cheatdetect.prolog_io import atom, text


MAX_ROUNDS = 36   # 24 regulation rounds + overtime; the dataset's longest match is 32

# Field -> label as the form shows it, so error messages use the same words.
KILL_TYPES = {
    "headshot_kills": "Headshot kills",
    "wallbang_kills": "Kills through walls",
    "smoke_kills": "Kills through smoke",
    "blind_kills": "Kills while flashed",
    "airborne_kills": "Kills while airborne",
}


@dataclass
class PlayerInput:
    rounds_played: int = 20
    kills: int = 15
    deaths: int = 14
    assists: int = 4
    shots_fired: int = 320
    gun_hits: int = 62
    head_hits: int = 11
    damage: int = 1500
    headshot_kills: int = 7
    wallbang_kills: int = 1
    smoke_kills: int = 0
    blind_kills: int = 0
    airborne_kills: int = 0

    def problems(self) -> list[str]:
        """Plain-language reasons the input cannot describe a real match (empty if fine)."""
        p = []
        if min(asdict(self).values()) < 0:
            p.append("Numbers cannot be negative.")
        if self.rounds_played < 1:
            p.append("Rounds played must be at least 1.")
            return p
        if self.rounds_played > MAX_ROUNDS:
            p.append(f"Rounds played cannot exceed {MAX_ROUNDS}: a match is at most 24 rounds plus overtime.")
        r = self.rounds_played
        # Limits of the game: 5 enemies per round, one life per round, 100 HP per enemy.
        if self.kills + self.assists > 5 * r:
            p.append(f"Kills plus assists cannot exceed {5 * r}: there are only 5 enemies per round, "
                     f"and each enemy death gives you a kill or an assist, not both.")
        if self.deaths > r:
            p.append(f"Deaths cannot exceed rounds played ({r}): a player dies at most once per round.")
        if self.damage > 500 * r:
            p.append(f"Total damage cannot exceed {500 * r}: 5 enemies with 100 HP each per round.")
        if self.kills > self.damage:
            p.append("Total damage must be at least the number of kills: every kill takes damage.")
        if self.gun_hits > self.damage:
            p.append("Total damage must be at least the number of hits: every hit removes at least 1 HP.")
        # Parts cannot be bigger than their whole.
        if self.gun_hits > self.shots_fired:
            p.append("Hits cannot be more than shots fired.")
        if self.head_hits > self.gun_hits:
            p.append("Headshot hits cannot be more than hits.")
        if self.headshot_kills > self.head_hits:
            p.append("Headshot kills cannot be more than headshot hits: each headshot kill needs a hit to the head.")
        # Each kill type is checked on its own, never as a sum: one kill can be a headshot,
        # through a wall and while flashed at the same time.
        for name, label in KILL_TYPES.items():
            if getattr(self, name) > self.kills:
                p.append(f"{label} cannot be more than kills.")
        return p

    def facts(self, pid: str) -> list[str]:
        """stat/3 and kill/9 facts. Kills get their flags from the counts the user gave."""
        a = atom(pid)
        stats = dict(rounds_played=self.rounds_played, kills=self.kills, deaths=self.deaths,
                     assists=self.assists, shots_fired=self.shots_fired, gun_hits=self.gun_hits,
                     head_hits=self.head_hits, damage=self.damage, flash_assists=0, enemies_flashed=0)
        out = [f"stat({a}, {k}, {int(v)})" for k, v in stats.items()]
        b = lambda i, n: "true" if i < n else "false"  # noqa: E731
        for i in range(self.kills):
            out.append(f"kill({a}, ak47, {b(i, self.headshot_kills)}, {b(i, self.smoke_kills)}, "
                       f"{b(i, self.wallbang_kills)}, {b(i, self.blind_kills)}, false, "
                       f"{b(i, self.airborne_kills)}, 15.0)")
        return out


VERDICT = {
    "no_action": "Not enough data to judge",
    "clear": "Looks legit",
    "monitor": "Slightly suspicious",
    "hide_from_leaderboard": "Likely cheating",
    "review_for_ban": "Very likely cheating",
}


def run_consult(inp: PlayerInput) -> dict:
    problems = inp.problems()
    if problems:
        raise ValueError(" ".join(problems))
    pid = f"check_{uuid.uuid4().hex[:10]}"
    e = prolog_io.engine()
    try:
        for f in inp.facts(pid):
            e.assertz(f)
        a = bridge.assess_one(pid)                      # forward + backward chaining, decision, explanation
        p = atom(pid)
        metrics = {text(r["M"]): float(r["V"]) for r in e.query(f"value({p}, M, V)")}
        thresholds = {}
        for r in e.query("threshold(M, L, V)"):
            thresholds.setdefault(text(r["M"]), {})[text(r["L"])] = float(r["V"])
        inference = prolog_io.lines(f"inference_lines({p}, L)")
        proof = prolog_io.lines(f"how({p}, cheating, L)")
        why_not = {g: prolog_io.lines(f"why_not({p}, {g}, L)")
                   for g in ("aimbot", "wallhack", "rage_cheat", "legit_outlier")}
        action_thr = bridge.action_thresholds()
    finally:
        list(e.query(f"forget_player({atom(pid)})"))

    return dict(
        verdict=VERDICT[a["action"]], action=a["action"], decision_mode=a["decision_mode"],
        cheating_cf=a["cheating_cf"], diagnosis=a["diagnosis"], diagnosis_cf=a["diagnosis_cf"],
        cfs={"aimbot": a["cf_aimbot"], "wallhack": a["cf_wallhack"], "rage_cheat": a["cf_rage"],
             "legit_outlier": a["cf_legit"]},
        symptoms=a["symptoms"], fired_rules=a["fired_rules"], explanation=a["explanation"],
        metrics=metrics, thresholds=thresholds, inference=inference, proof=proof, why_not=why_not,
        action_thresholds=action_thr,
    )
