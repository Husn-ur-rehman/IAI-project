"""Environment smoke tests: dataset present, Prolog bridge works."""
import json

import pytest

from cheatdetect import config

needs_raw = pytest.mark.skipif(not config.CHEATER_DIR.exists(),
                               reason="raw CS2CD files not downloaded (not committed)")


@needs_raw
def test_dataset_present():
    assert len(list(config.CHEATER_DIR.glob("*.json"))) == 317
    assert len(list(config.CLEAN_DIR.glob("*.json"))) == 478


@needs_raw
def test_cheater_labels_are_per_player():
    sample = next(config.CHEATER_DIR.glob("*.json"))
    data = json.loads(sample.read_text())
    assert data["cheaters"] and "steamid" in data["cheaters"][0]


def test_prolog_bridge():
    from pyswip import Prolog

    prolog = Prolog()
    prolog.consult(config.KB_MAIN.as_posix())
    assert list(prolog.query("kb_ready"))
