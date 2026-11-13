"""Project paths. Everything else imports locations from here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = ROOT / "data" / "raw" / "cs2cd"
CHEATER_DIR = RAW_DIR / "with_cheater_present"
CLEAN_DIR = RAW_DIR / "no_cheater_present"

PROCESSED_DIR = ROOT / "data" / "processed"
FEATURES_FILE = PROCESSED_DIR / "player_match_features.parquet"

KB_DIR = ROOT / "kb"
KB_MAIN = KB_DIR / "main.pl"

REPORTS_DIR = ROOT / "reports"
