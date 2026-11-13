"""Moderator decisions: the human half of decision support.

The KB never bans anyone. It queues players for review; a moderator's decision is recorded
here and overrides the KB's automatic action on the leaderboard.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from cheatdetect import config

DECISIONS_FILE = config.PROCESSED_DIR / "moderator_decisions.csv"
COLUMNS = ["time_utc", "pid", "decision", "kb_action", "cheating_cf", "note"]

DECISIONS = {
    "ban": "Ban confirmed",
    "keep_hidden": "Kept hidden, monitoring",
    "restore": "Restored to the leaderboard",
}


def load() -> pd.DataFrame:
    if DECISIONS_FILE.exists():
        return pd.read_csv(DECISIONS_FILE)
    return pd.DataFrame(columns=COLUMNS)


def record(pid: str, decision: str, kb_action: str, cheating_cf: float, note: str = "") -> None:
    if decision not in DECISIONS:
        raise ValueError(f"unknown decision {decision!r}")
    row = pd.DataFrame([[datetime.now(timezone.utc).isoformat(timespec="seconds"), pid, decision,
                         kb_action, round(float(cheating_cf), 4), note]], columns=COLUMNS)
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    row.to_csv(DECISIONS_FILE, mode="a", header=not DECISIONS_FILE.exists(), index=False)


def latest() -> dict[str, str]:
    """pid -> most recent decision."""
    d = load()
    return {} if d.empty else d.groupby("pid").decision.last().to_dict()
