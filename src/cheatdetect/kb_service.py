"""One long-lived owner of the Prolog engine for the UI.

SWI-Prolog is not safe to query from several threads at once, and Streamlit runs every
rerun on a different thread. Every KB call therefore goes through one dedicated worker thread.
Everything the UI shows about a player comes from these calls, or from the verdict table
they produced.
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import Future

import pandas as pd

from cheatdetect import bridge, config, prolog_io
from cheatdetect.prolog_io import atom, text


class KBService:
    """Runs every KB call on one daemon worker thread, one call at a time.

    Known, harmless: pyswip attaches a Prolog engine to the worker thread and never releases
    it, so SWI-Prolog prints "Failed to stop Prolog threads. Not reclaming memory." at exit.
    The OS frees that memory when the process ends. Releasing the engine at exit
    (PL_thread_destroy_engine) silenced the message but made pyswip's own cleanup crash with
    an access violation after a failed in-process app test, so it is deliberately not done.
    """

    def __init__(self):
        self._jobs: queue.Queue = queue.Queue()
        self._thread = threading.Thread(target=self._loop, name="prolog", daemon=True)
        self._thread.start()
        self._verdicts: pd.DataFrame | None = None
        self._run(self._boot)

    def _loop(self):
        while True:
            fn, args, kwargs, fut = self._jobs.get()
            try:
                fut.set_result(fn(*args, **kwargs))
            except BaseException as exc:  # hand the error to the caller
                fut.set_exception(exc)

    def _run(self, fn, *args, **kwargs):
        fut: Future = Future()
        self._jobs.put((fn, args, kwargs, fut))
        return fut.result()

    # ---------- runs on the Prolog thread ----------
    def _boot(self):
        bridge.prepare(write_facts=not config.FACTS_FILE.exists())
        pids = pd.read_parquet(config.PLAYERS_FILE, columns=["pid"]).pid.tolist()
        self._verdicts = bridge.attach_labels(bridge.assess_all(pids))   # also runs forward chaining for all

    def _apply(self, review, hide, monitor):
        bridge.set_action_thresholds(review, hide, monitor)
        new = bridge.redecide(self._verdicts.pid.tolist()).set_index("pid")
        v = self._verdicts.set_index("pid")
        v["action"] = new["action"]
        v["decision_mode"] = new["decision_mode"]
        self._verdicts = v.reset_index()

    def _reset(self):
        bridge.reset_action_thresholds()
        defaults = bridge.action_thresholds()
        self._apply(defaults["review_for_ban"], defaults["hide_from_leaderboard"], defaults["monitor"])

    def _rules(self) -> pd.DataFrame:
        rows = []
        for r in prolog_io.engine().query("br(Id, Concl, Prems, CF)"):
            prems = [text(p) for p in r["Prems"]]
            rows.append(dict(rule=text(r["Id"]), concludes=text(r["Concl"]),
                             if_all_of=", ".join(prems), rule_cf=float(r["CF"])))
        return pd.DataFrame(rows)

    def _metric_thresholds(self) -> pd.DataFrame:
        rows = [dict(metric=text(r["M"]), level=text(r["L"]), value=float(r["V"]))
                for r in prolog_io.engine().query("threshold(M, L, V)")]
        t = pd.DataFrame(rows).pivot(index="metric", columns="level", values="value")
        desc = {text(r["M"]): text(r["D"]) for r in prolog_io.engine().query("metric(M, D)")}
        t.insert(0, "meaning", t.index.map(desc))
        return t.reset_index()

    def _why_not(self, pid, goal):
        return prolog_io.lines(f"why_not({atom(pid)}, {goal}, L)")

    # ---------- public API (any thread) ----------
    def verdicts(self) -> pd.DataFrame:
        return self._verdicts.copy()

    def player(self, pid: str) -> dict:
        """Live assessment of one player, straight from the KB."""
        return self._run(bridge.assess_one, pid)

    def why_not(self, pid: str, goal: str) -> list[str]:
        return self._run(self._why_not, pid, goal)

    def consult(self, inp) -> dict:
        """Assess a player described by the user (temporary facts, removed afterwards)."""
        from cheatdetect.consultation import run_consult
        return self._run(run_consult, inp)

    def apply_thresholds(self, review: float, hide: float, monitor: float):
        self._run(self._apply, review, hide, monitor)

    def reset_thresholds(self):
        self._run(self._reset)

    def action_thresholds(self) -> dict[str, float]:
        return self._run(bridge.action_thresholds)

    def rules(self) -> pd.DataFrame:
        return self._run(self._rules)

    def metric_thresholds(self) -> pd.DataFrame:
        return self._run(self._metric_thresholds)
