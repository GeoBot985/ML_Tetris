from __future__ import annotations

import json

from tetris.high_score import HighScore, load_high_score, record_high_score


def test_load_high_score_defaults_when_missing(tmp_path):
    assert load_high_score(tmp_path / "high_score.json") == HighScore()


def test_load_high_score_defaults_when_corrupt(tmp_path):
    path = tmp_path / "high_score.json"
    path.write_text("not json", encoding="utf-8")

    assert load_high_score(path) == HighScore()


def test_record_high_score_writes_new_record(tmp_path):
    path = tmp_path / "high_score.json"

    high_score = record_high_score(path, score=8598, lines=30, difficulty="Normal")

    assert high_score == HighScore(score=8598, lines=30, difficulty="Normal")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["score"] == 8598
    assert payload["lines"] == 30
    assert payload["difficulty"] == "Normal"


def test_record_high_score_keeps_existing_record_when_lower(tmp_path):
    path = tmp_path / "high_score.json"
    record_high_score(path, score=8598, lines=30, difficulty="Normal")

    high_score = record_high_score(path, score=1200, lines=4, difficulty="Easy")

    assert high_score == HighScore(score=8598, lines=30, difficulty="Normal")
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["score"] == 8598
