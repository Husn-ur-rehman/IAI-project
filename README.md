# CS2 Leaderboard Cheat-Detection Expert System

An expert system that flags suspected cheaters on a CS2 leaderboard. It demonstrates **inference**, **diagnostics**, **decision making** (automatic low-stakes actions) and **decision support** (a human review queue for bans).

- **Python:** data pipeline, calibration, evaluation, Streamlit UI.
- **SWI-Prolog (via pyswip):** knowledge base, certainty factors, why/how explanations.

**Live app:** https://cs2-integrity-console.streamlit.app/

## Layout

```
data/raw/cs2cd/      CS2CD JSON event files (not committed)
data/processed/      players.parquet, kills.parquet (committed); facts, verdicts (generated)
kb/                  Prolog knowledge base
src/cheatdetect/     Python package (facts, calibration, bridge, evaluation)
app/                 Streamlit decision-support UI
tests/               pytest
```

## Setup

Requires Python 3.11+ and 64-bit SWI-Prolog on PATH.

The console runs from the committed `data/processed/` files. To rerun the full pipeline, download the
JSON event files of [CS2CD](https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection)
(CC BY 4.0) into `data/raw/cs2cd/`.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
pytest
```

## Pipeline

```
python -m cheatdetect.facts       # JSON -> data/processed/players.parquet, kills.parquet
python -m cheatdetect.calibrate   # calibration half of clean matches -> kb/thresholds.pl
python -m cheatdetect.bridge      # KB verdict per player -> data/processed/verdicts.parquet
python -m cheatdetect.evaluate    # held-out evaluation -> reports/ (figures, tables; not committed)
streamlit run app/streamlit_app.py   # moderator console (decision support)
```

## Moderator console

| Tab | What it does |
|---|---|
| Check a player | Enter one player's match numbers. The KB reasons over them as temporary facts and shows inference (metrics → symptoms, proof tree), diagnosis (CF per cheat type), decision making (action, automatic or human) and decision support (recommendation, evidence, why-not). The facts are removed afterwards. |
| Review queue | Players the KB sent for a ban decision, most certain first, with the live Prolog explanation. The moderator confirms, keeps hidden or restores. Decisions go to `data/processed/moderator_decisions.csv`. |
| Leaderboard | Public ranking by damage per round. Players the KB hid lose their rank, and a moderator decision overrides the KB. |
| Player lookup | Any player's verdict, why it was reached, and why-not for any diagnosis |
| Strictness dial | Rewrites the `action_threshold/2` facts in Prolog and re-decides every player; shows precision and recall on held-out players |
| Knowledge base | The rules, thresholds and action levels, read live from Prolog |

## Data

- **Dataset:** CS2CD (Loo, Lužkov, Burelli, IT University of Copenhagen, 2025), CC BY 4.0, DOI 10.57967/hf/5315. https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection
- **What's used:** only the JSON event files, not the tick-level parquet.
- **Labels:** cheater labels are real and per player.
- **Anonymisation:** player IDs are anonymised per match, so leaderboard "accounts" are synthetic groupings.
