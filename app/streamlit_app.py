"""CS2 Integrity Console: the decision-support UI.

Run:  streamlit run app/streamlit_app.py

Every verdict, CF and explanation on screen comes from the Prolog knowledge base (via
KBService). Python here only lays it out and records the moderator's decisions.
"""
from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from cheatdetect import config, moderation
from cheatdetect.calibrate import is_calibration_match
from cheatdetect.evaluate import scores
from cheatdetect.kb_service import KBService

st.set_page_config(page_title="CS2 Integrity Console", layout="wide", initial_sidebar_state="collapsed")

ACTION_LABEL = {"review_for_ban": "Ban review", "hide_from_leaderboard": "Hidden", "monitor": "Monitor",
                "clear": "Clear", "no_action": "Too little data"}
ACTION_CLASS = {"review_for_ban": "review", "hide_from_leaderboard": "hide", "monitor": "monitor",
                "clear": "clear", "no_action": "none"}
ICON = {
    "review_for_ban": '<path d="M5 1 9 9H1Z" fill="none" stroke="currentColor" stroke-width="1.4" stroke-linejoin="round"/><path d="M5 4v2.2" stroke="currentColor" stroke-width="1.3"/>',
    "hide_from_leaderboard": '<circle cx="5" cy="5" r="3.6" fill="none" stroke="currentColor" stroke-width="1.4"/><path d="M2.4 7.6 7.6 2.4" stroke="currentColor" stroke-width="1.4"/>',
    "monitor": '<circle cx="5" cy="5" r="3.6" fill="none" stroke="currentColor" stroke-width="1.4"/><circle cx="5" cy="5" r="1.3" fill="currentColor"/>',
    "clear": '<path d="M1.8 5.2 4 7.4 8.3 2.6" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>',
    "no_action": '<path d="M2 5h6" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>',
}
DIAG = {"aimbot": "Aimbot", "wallhack": "Wallhack", "rage_cheat": "Rage cheat", "legit_outlier": "Legit strong player",
        "none": "No cheat found"}
METRIC = {"headshot_rate": "Headshot rate", "accuracy": "Accuracy", "adr": "Damage per round", "kpr": "Kills per round",
          "kd": "Kill/death", "hs_kill_rate": "Headshot kills", "wallbang_rate": "Wallbang kills",
          "smoke_kill_rate": "Smoke kills", "blind_kill_rate": "Kills while flashed", "air_kill_rate": "Airborne kills"}

CSS = """
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Lexend:wght@400;500;600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono&display=swap">
<style>
:root { --brand:#7F011F; --on-brand:#F5EBD0; --ink:#24070E; --ink-2:#5E3C43; --panel:#FBF6E8; --sunk:#EFE3C2; --rule:#DFCFA6;
        --tint-1:#E3C8B5; --tint-2:#CC9992; }
html, body, .stApp, .stMarkdown, p, li, label, input, textarea { font-family: "Source Sans 3", "Segoe UI", sans-serif; }
h1, h2, h3, h4, .stTabs button p { font-family: "Lexend", "Segoe UI", sans-serif !important; letter-spacing: -.01em; }
.block-container { padding-top: 1.2rem; max-width: 1280px; }
.band { background: var(--brand); color: var(--on-brand); border-radius: 10px; padding: 14px 18px; display: flex;
        justify-content: space-between; align-items: baseline; flex-wrap: wrap; gap: 8px; margin-bottom: .6rem; }
.band h1 { font-family: "Lexend", sans-serif; font-size: 1.35rem; font-weight: 600; margin: 0; padding: 0; color: var(--on-brand); }
.band p { margin: 0; opacity: .85; font-size: .9rem; }
.strip { display: flex; flex-wrap: wrap; gap: 6px 24px; color: var(--ink-2); margin: .2rem 0 .4rem; }
.strip b { font-family: "Lexend", sans-serif; font-weight: 600; font-size: 1.3rem; color: var(--ink); margin-right: 4px; font-variant-numeric: tabular-nums; }
.pill { display: inline-flex; align-items: center; gap: 5px; padding: 1px 9px 1px 7px; border-radius: 999px; font-size: .8rem; font-weight: 600; white-space: nowrap; border: 1px solid transparent; }
.pill svg { width: 11px; height: 11px; }
.s-none { border: 1px dashed var(--ink-2); color: var(--ink-2); }
.s-clear { border-color: var(--ink-2); color: var(--ink-2); }
.s-monitor { background: var(--tint-1); color: var(--brand); }
.s-hide { background: var(--tint-2); color: var(--ink); }
.s-review { background: var(--brand); color: var(--on-brand); }
.stats { display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 1px; background: var(--rule); border: 1px solid var(--rule); border-radius: 8px; overflow: hidden; margin: .4rem 0 .8rem; }
.stats div { background: var(--panel); padding: 8px 10px; }
.stats b { display: block; font-family: "Lexend", sans-serif; font-weight: 600; font-size: 1.15rem; font-variant-numeric: tabular-nums; }
.stats small { color: var(--ink-2); display: block; }
.cfrow { display: grid; grid-template-columns: 9rem 1fr 2.8rem; gap: 10px; align-items: center; margin: 4px 0; font-size: .92rem; }
.track { height: 8px; border-radius: 4px; background: var(--sunk); overflow: hidden; }
.track i { display: block; height: 100%; border-radius: 4px; background: var(--brand); }
.cfrow .v { text-align: right; color: var(--ink-2); font-variant-numeric: tabular-nums; }
.cfrow.top { font-weight: 600; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: .3rem 0 .6rem; }
.chip { font-size: .85rem; padding: 2px 8px; border-radius: 6px; background: var(--sunk); }
.chip em { font-style: normal; color: var(--brand); font-weight: 600; margin-left: 4px; }
.trace { background: var(--sunk); border-radius: 8px; padding: 12px; font-family: "JetBrains Mono", Consolas, monospace; font-size: .76rem; line-height: 1.6; white-space: pre; overflow-x: auto; }
.muted { color: var(--ink-2); font-size: .9rem; }
.head-row { display: flex; justify-content: space-between; align-items: baseline; gap: 10px; flex-wrap: wrap; }
.head-row h3 { margin: 0; padding: 0; }
</style>
"""


# ---------- data ----------------------------------------------------------------

@st.cache_resource(show_spinner="Loading the knowledge base and assessing every player...")
def kb() -> KBService:
    return KBService()


@st.cache_data
def player_stats() -> pd.DataFrame:
    p = pd.read_parquet(config.PLAYERS_FILE)
    p["adr"] = p.damage / p.rounds_played.clip(lower=1)
    return p[["pid", "rounds_played", "kills", "deaths", "adr", "map", "mode"]]


def table() -> pd.DataFrame:
    """Current KB verdicts + stats + the moderator's latest decision."""
    stats = player_stats()[["pid", "rounds_played", "kills", "deaths", "adr"]]
    v = kb().verdicts().merge(stats, on="pid", how="left")       # map / match_mode already come with the verdicts
    v["mode"] = v.match_mode
    decided = moderation.latest()
    v["moderator"] = [decided.get(p, "") for p in v.pid]
    return v


# ---------- small renderers ----------------------------------------------------------

def pill(action: str) -> str:
    return (f'<span class="pill s-{ACTION_CLASS[action]}"><svg viewBox="0 0 10 10" aria-hidden="true">{ICON[action]}</svg>'
            f'{ACTION_LABEL[action]}</span>')


def symptom_chip(s: str) -> str:
    if s.startswith("abnormal(") and "@" in s:
        metric, level = s[len("abnormal("):].split(")@")
        return f'<span class="chip">{METRIC.get(metric, metric)}<em>above {level}</em></span>'
    return f'<span class="chip">{html.escape(s.replace("_", " ").capitalize())}</span>'


def render_player(pid: str, key: str):
    a = kb().player(pid)                           # live from Prolog
    s = player_stats().set_index("pid").loc[pid]
    decided = moderation.latest().get(pid)
    st.markdown(
        f'<div class="head-row"><h3>{html.escape(pid)}</h3>{pill(a["action"])}</div>'
        f'<p class="muted">{s["mode"]} match on {s["map"]}. '
        f'{"Decided by a moderator: " + moderation.DECISIONS[decided] + "." if decided else ""}</p>'
        f'<div class="stats"><div><b>{a["cheating_cf"]:.2f}</b><small>Cheating CF</small></div>'
        f'<div><b>{s.adr:.0f}</b><small>Damage per round</small></div>'
        f'<div><b>{s.kills}/{s.deaths}</b><small>Kills/deaths</small></div>'
        f'<div><b>{s.rounds_played}</b><small>Rounds</small></div></div>', unsafe_allow_html=True)

    cfs = sorted([("aimbot", a["cf_aimbot"]), ("wallhack", a["cf_wallhack"]), ("rage_cheat", a["cf_rage"]),
                  ("legit_outlier", a["cf_legit"])], key=lambda x: -x[1])
    st.markdown("".join(
        f'<div class="cfrow {"top" if i == 0 and v > 0 else ""}"><span>{DIAG[k]}</span>'
        f'<span class="track"><i style="width:{max(v, 0) * 100:.0f}%"></i></span><span class="v">{v:.2f}</span></div>'
        for i, (k, v) in enumerate(cfs)), unsafe_allow_html=True)

    if a["symptoms"]:
        st.markdown("**What the rules found**<div class='chips'>" + "".join(symptom_chip(x) for x in a["symptoms"]) + "</div>",
                    unsafe_allow_html=True)

    title = "Why the system cleared this player" if a["action"] == "clear" else "Why the system took this action"
    with st.expander(title, expanded=a["action"] == "review_for_ban"):
        st.markdown(f'<div class="trace">{html.escape(chr(10).join(a["explanation"]))}</div>', unsafe_allow_html=True)
        if a["action"] != "no_action":
            goal = st.selectbox("Ask why a diagnosis did not fire", ["aimbot", "wallhack", "rage_cheat", "legit_outlier"],
                                format_func=DIAG.get, key=f"whynot-{key}")
            st.markdown(f'<div class="trace">{html.escape(chr(10).join(kb().why_not(pid, goal)))}</div>',
                        unsafe_allow_html=True)

    if a["action"] in ("review_for_ban", "hide_from_leaderboard"):
        c1, c2, c3 = st.columns(3)
        for col, (code, label) in zip((c1, c2, c3), [("ban", "Confirm ban"), ("keep_hidden", "Keep hidden, monitor"),
                                                      ("restore", "Restore to leaderboard")]):
            if col.button(label, key=f"{code}-{key}", type="primary" if code == "ban" else "secondary",
                          use_container_width=True):
                moderation.record(pid, code, a["action"], a["cheating_cf"])
                st.toast(f"{moderation.DECISIONS[code]} for {pid}.")
                st.rerun()


# ---------- views ----------------------------------------------------------------

def view_queue(v: pd.DataFrame):
    q = v[(v.action == "review_for_ban") & (v.moderator == "")].sort_values("cheating_cf", ascending=False)
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown(f"#### Waiting for a ban decision ({len(q)})")
        if q.empty:
            st.info("The queue is empty. Every player sent for review has a moderator decision.")
            return
        shown = q[["pid", "cheating_cf", "diagnosis", "kills", "deaths"]].rename(columns={
            "pid": "Player", "cheating_cf": "Cheating CF", "diagnosis": "Diagnosis", "kills": "Kills",
            "deaths": "Deaths"})
        shown["Diagnosis"] = shown.Diagnosis.map(DIAG)
        event = st.dataframe(shown, hide_index=True, use_container_width=True, height=520, on_select="rerun",
                             selection_mode="single-row", key="queue",
                             column_config={"Cheating CF": st.column_config.ProgressColumn(
                                 "Cheating CF", min_value=0.0, max_value=1.0, format="%.2f", width="medium"),
                                 "Kills": st.column_config.NumberColumn(width="small"),
                                 "Deaths": st.column_config.NumberColumn(width="small")})
        rows = event.selection.rows if event and event.selection else []
        pid = shown.iloc[rows[0]].Player if rows else shown.iloc[0].Player
    with right:
        render_player(pid, "queue")


def view_leaderboard(v: pd.DataFrame):
    st.markdown("#### Top damage per round")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.8])
    maps = c1.multiselect("Map", sorted(v["map"].dropna().unique()), placeholder="All maps")
    modes = c2.multiselect("Match type", sorted(v["mode"].dropna().unique()), placeholder="All match types")
    min_rounds = c3.slider("Minimum rounds played", 1, 30, 12)
    show_hidden = c4.toggle("Show hidden players", value=False)
    b = v[v.rounds_played >= min_rounds]
    if maps:
        b = b[b["map"].isin(maps)]
    if modes:
        b = b[b["mode"].isin(modes)]
    b = b.sort_values("adr", ascending=False).copy()

    # The KB hides flagged players automatically; a moderator decision overrides it.
    hidden = b.action.isin(["review_for_ban", "hide_from_leaderboard"])
    hidden = (hidden & (b.moderator != "restore")) | (b.moderator.isin(["ban", "keep_hidden"]))
    if not show_hidden:
        b, hidden = b[~hidden], hidden[~hidden]
    b, hidden = b.head(50), hidden.head(50)
    b["Status"] = np.where(b.moderator == "ban", "Banned",
                  np.where(b.moderator == "restore", "Restored by moderator", b.action.map(ACTION_LABEL)))
    b["Public rank"] = np.where(hidden, "Hidden", "")
    b.loc[~hidden, "Public rank"] = [str(i) for i in range(1, int((~hidden).sum()) + 1)]
    st.dataframe(b[["Public rank", "pid", "adr", "kills", "deaths", "rounds_played", "Status", "map"]].rename(columns={
        "pid": "Player", "adr": "Damage per round", "kills": "Kills", "deaths": "Deaths", "rounds_played": "Rounds",
        "map": "Map"}), hide_index=True, use_container_width=True, height=560,
        column_config={"Damage per round": st.column_config.NumberColumn(format="%.0f")})
    if show_hidden:
        st.caption(f"{int(hidden.sum())} of these {len(b)} players are hidden from the public ranking. "
                   "They keep their row here so a moderator can see what the system removed.")
    else:
        st.caption("This is the public ranking. Turn on 'Show hidden players' to see who the system removed.")


def view_lookup(v: pd.DataFrame):
    st.markdown("#### Look up any player")
    pid = st.selectbox("Player", v.pid.sort_values().tolist(), index=None, placeholder="Type a player id, for example c1_Player_10")
    if pid:
        render_player(pid, "lookup")
    else:
        st.markdown('<p class="muted">Pick a player to see the knowledge base\'s verdict and its full reasoning, '
                    'including why a cheat was not diagnosed.</p>', unsafe_allow_html=True)


@st.cache_data
def dial_curve(cf: pd.Series, y: pd.Series, assessed: pd.Series) -> pd.DataFrame:
    rows = []
    for t in np.round(np.arange(0.2, 0.99, 0.025), 3):
        s = scores(y.to_numpy(), (cf >= t).to_numpy() & assessed.to_numpy())
        rows += [dict(threshold=t, measure="Precision", value=s["precision"]),
                 dict(threshold=t, measure="Recall", value=s["recall"])]
    return pd.DataFrame(rows)


def view_dial(v: pd.DataFrame):
    cur = kb().action_thresholds()
    st.markdown("#### How certain must the system be before it acts?")
    st.markdown('<p class="muted">Applying new thresholds rewrites the <code>action_threshold/2</code> facts in Prolog. '
                'The knowledge base then decides again for every player.</p>', unsafe_allow_html=True)
    with st.form("dial"):
        c1, c2, c3 = st.columns(3)
        monitor = c1.slider("Monitor at CF", 0.05, 0.95, cur["monitor"], 0.05)
        hide = c2.slider("Hide from leaderboard at CF", 0.05, 0.95, cur["hide_from_leaderboard"], 0.05)
        review = c3.slider("Send for ban review at CF", 0.05, 0.95, cur["review_for_ban"], 0.05)
        a, b = st.columns([1, 5])
        apply = a.form_submit_button("Apply", type="primary")
        reset = b.form_submit_button("Reset to defaults")
    if apply:
        if not (review >= hide >= monitor):
            st.error("Each step must be at least as strict as the one before: monitor ≤ hide ≤ ban review.")
        else:
            with st.spinner("Prolog is deciding again for every player..."):
                kb().apply_thresholds(review, hide, monitor)
            st.rerun()
    if reset:
        with st.spinner("Restoring the default thresholds..."):
            kb().reset_thresholds()
        st.rerun()

    ev = v[~v.match_id.map(is_calibration_match).astype(bool)]
    y = ev.is_cheater.to_numpy()
    levels = [("Monitor or stricter", ["monitor", "hide_from_leaderboard", "review_for_ban"]),
              ("Hidden or stricter", ["hide_from_leaderboard", "review_for_ban"]), ("Ban review", ["review_for_ban"])]
    st.markdown("##### Against the real labels (held-out players)")
    tiles = []
    for name, acts in levels:
        s = scores(y, ev.action.isin(acts).to_numpy())
        tiles.append(f'<div><b>{s["flagged"]}</b><small>{name}: flagged</small>'
                     f'<small>{s["precision"]:.0%} are labelled cheaters, {s["recall"]:.0%} of cheaters caught</small></div>')
    st.markdown(f'<div class="stats" style="grid-template-columns:repeat(3,minmax(0,1fr))">{"".join(tiles)}</div>',
                unsafe_allow_html=True)

    curve = dial_curve(ev.cheating_cf, ev.is_cheater, ev.action != "no_action")
    base = alt.Chart(curve).encode(
        x=alt.X("threshold:Q", title="Ban-review threshold (cheating CF)", scale=alt.Scale(domain=[0.2, 0.975])),
        y=alt.Y("value:Q", title=None, axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1])),
        color=alt.Color("measure:N", scale=alt.Scale(domain=["Precision", "Recall"], range=["#7F011F", "#5E3C43"]),
                        legend=alt.Legend(title=None, orient="top")),
        strokeDash=alt.StrokeDash("measure:N", scale=alt.Scale(domain=["Precision", "Recall"], range=[[1, 0], [5, 4]]),
                                  legend=None),
        tooltip=[alt.Tooltip("threshold:Q", format=".3f"), "measure:N", alt.Tooltip("value:Q", format=".1%")])
    rule = alt.Chart(pd.DataFrame({"t": [review]})).mark_rule(color="#24070E", strokeDash=[3, 3]).encode(x="t:Q")
    chart = (base.mark_line(strokeWidth=2) + rule).properties(height=280).configure_axisX(grid=False, tickCount=8) \
        .configure_axisY(gridColor="#DFCFA6").configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True)


def view_knowledge(_v: pd.DataFrame):
    st.markdown("#### What the knowledge base knows")
    st.markdown('<p class="muted">Read live from Prolog. Layer 1 (forward chaining) turns metrics into symptoms using '
                'these thresholds. Layer 2 (backward chaining) combines symptoms into diagnoses with MYCIN certainty '
                'factors. Layer 3 turns the cheating CF into an action.</p>', unsafe_allow_html=True)
    c1, c2 = st.columns([1.3, 1], gap="large")
    with c1:
        st.markdown("##### Diagnosis rules")
        r = kb().rules()
        r["concludes"] = r.concludes.map(lambda c: DIAG.get(c, c.capitalize()))
        st.dataframe(r.rename(columns={"rule": "Rule", "concludes": "Concludes", "if_all_of": "If all of",
                                       "rule_cf": "Rule CF"}), hide_index=True, use_container_width=True, height=560)
    with c2:
        st.markdown("##### What counts as abnormal")
        t = kb().metric_thresholds()
        t["metric"] = t.metric.map(lambda m: METRIC.get(m, m))
        t = t.drop(columns="meaning")
        st.dataframe(t.rename(columns={"metric": "Metric", "p95": "Above p95 (CF 0.4)",
                                       "p99": "Above p99 (CF 0.8)"}), hide_index=True, use_container_width=True,
                     column_config={"Above p95 (CF 0.4)": st.column_config.NumberColumn(format="%.3f"),
                                    "Above p99 (CF 0.8)": st.column_config.NumberColumn(format="%.3f")})
        st.markdown("##### When the system acts")
        a = kb().action_thresholds()
        st.dataframe(pd.DataFrame([(ACTION_LABEL[k], v, "Human decides" if k == "review_for_ban" else "Automatic")
                                   for k, v in sorted(a.items(), key=lambda x: x[1])],
                                  columns=["Action", "At cheating CF", "Who decides"]), hide_index=True,
                     use_container_width=True)


# ---------- page -----------------------------------------------------------------------

def main():
    st.markdown(CSS, unsafe_allow_html=True)
    v = table()
    counts = v.action.value_counts()
    waiting = int(((v.action == "review_for_ban") & (v.moderator == "")).sum())
    st.markdown(
        f'<div class="band"><h1>CS2 Integrity Console</h1><p>Verdicts from the Prolog knowledge base, '
        f'{len(v):,} player-matches</p></div>'
        f'<div class="strip"><span><b>{waiting}</b>waiting for a moderator</span>'
        f'<span><b>{counts.get("hide_from_leaderboard", 0)}</b>hidden automatically</span>'
        f'<span><b>{counts.get("monitor", 0)}</b>being monitored</span>'
        f'<span><b>{counts.get("clear", 0):,}</b>cleared</span>'
        f'<span><b>{counts.get("no_action", 0):,}</b>too little data to judge</span></div>', unsafe_allow_html=True)

    tabs = st.tabs(["Review queue", "Leaderboard", "Player lookup", "Strictness dial", "Knowledge base"])
    for tab, view in zip(tabs, [view_queue, view_leaderboard, view_lookup, view_dial, view_knowledge]):
        with tab:
            view(v)


main()
