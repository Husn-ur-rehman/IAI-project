# CS2 Leaderboard Cheat-Detection Expert System

An expert system that flags suspected cheaters on a CS2 leaderboard. It demonstrates **inference**, **diagnostics**, **decision making** (automatic low-stakes actions) and **decision support** (a human review queue for bans).

- **Python:** data pipeline, calibration, evaluation, Streamlit UI.
- **SWI-Prolog (via pyswip):** knowledge base, certainty factors, why/how explanations.

## Layout

```
data/raw/cs2cd/      CS2CD JSON event files (not committed)
data/processed/      aggregated per-player-match features
kb/                  Prolog knowledge base
src/cheatdetect/     Python package (features, bridge, evaluation)
app/                 Streamlit decision-support UI
scripts/             one-off scripts (dataset download)
tests/               pytest
reports/             figures and metrics for the report
```

## Setup

Requires Python 3.11+ and 64-bit SWI-Prolog on PATH.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_cs2cd.py     # ~1.8 GB, JSON files only
pytest
```

## Data

- **Dataset:** CS2CD (Loo, Lužkov, Burelli, IT University of Copenhagen, 2025), CC BY 4.0, DOI 10.57967/hf/5315. https://huggingface.co/datasets/CS2CD/CS2CD.Counter-Strike_2_Cheat_Detection
- **What's used:** only the JSON event files, not the tick-level parquet.
- **Labels:** cheater labels are real and per player.
- **Anonymisation:** player IDs are anonymised per match, so leaderboard "accounts" are synthetic groupings.
