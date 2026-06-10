from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class HighScore:
    score: int = 0
    lines: int = 0
    difficulty: str | None = None


def load_high_score(path: Path) -> HighScore:
    if not path.exists():
        return HighScore()
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return HighScore()
    if not isinstance(payload, dict):
        return HighScore()
    return _high_score_from_payload(payload)


def record_high_score(path: Path, *, score: int, lines: int, difficulty: str | None) -> HighScore:
    current = load_high_score(path)
    candidate = HighScore(score=max(0, int(score)), lines=max(0, int(lines)), difficulty=difficulty)
    if candidate.score <= current.score:
        return current
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "score": candidate.score,
                "lines": candidate.lines,
                "difficulty": candidate.difficulty,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return candidate


def _high_score_from_payload(payload: dict[str, Any]) -> HighScore:
    try:
        score = max(0, int(payload.get("score", 0)))
    except (TypeError, ValueError):
        score = 0
    try:
        lines = max(0, int(payload.get("lines", 0)))
    except (TypeError, ValueError):
        lines = 0
    difficulty = payload.get("difficulty")
    if difficulty is not None:
        difficulty = str(difficulty)
    return HighScore(score=score, lines=lines, difficulty=difficulty)
