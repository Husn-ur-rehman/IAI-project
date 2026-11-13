"""Environment smoke tests: dataset present, Prolog bridge works."""
import json

from cheatdetect import config


def test_dataset_present():
    assert len(list(config.CHEATER_DIR.glob("*.json"))) == 317
    assert len(list(config.CLEAN_DIR.glob("*.json"))) == 478


def test_cheater_labels_are_per_player():
    sample = next(config.CHEATER_DIR.glob("*.json"))
    data = json.loads(sample.read_text())
    assert data["cheaters"] and "steamid" in data["cheaters"][0]


def test_prolog_bridge():
    from pyswip import Prolog

    prolog = Prolog()
    prolog.consult(config.KB_MAIN.as_posix())
    assert list(prolog.query("kb_ready"))
