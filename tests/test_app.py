"""End-to-end smoke test of the Streamlit console (needs the processed data)."""
import pytest

from cheatdetect import config, moderation

pytestmark = pytest.mark.skipif(not config.VERDICTS_FILE.exists(), reason="run the pipeline first")


@pytest.fixture(scope="module")
def app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(config.ROOT / "app" / "streamlit_app.py"), default_timeout=180)
    at.run()
    return at


def test_app_renders_every_view(app):
    assert not app.exception
    assert [t.label for t in app.tabs] == ["Review queue", "Leaderboard", "Player lookup", "Strictness dial",
                                           "Knowledge base"]
    assert any("CS2 Integrity Console" in m.value for m in app.markdown)


def test_queue_shows_kb_explanation(app):
    text = "\n".join(m.value for m in app.markdown)
    assert "Action: review_for_ban (human)" in text      # live why/2 output from Prolog


def test_moderator_decision_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setattr(moderation, "DECISIONS_FILE", tmp_path / "d.csv")
    moderation.record("c1_Player_10", "ban", "review_for_ban", 0.9)
    moderation.record("c1_Player_10", "restore", "review_for_ban", 0.9)
    assert moderation.latest() == {"c1_Player_10": "restore"}
    with pytest.raises(ValueError):
        moderation.record("x", "delete_account", "clear", 0.0)
