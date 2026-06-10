from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Difficulty:
    name: str
    start_ms: int
    per_level_ms: int
    minimum_ms: int

    def gravity_for_level(self, level: int) -> int:
        gravity = self.start_ms + (level - 1) * self.per_level_ms
        return max(self.minimum_ms, gravity)


EASY = Difficulty("Easy", start_ms=850, per_level_ms=-30, minimum_ms=300)
NORMAL = Difficulty("Normal", start_ms=700, per_level_ms=-35, minimum_ms=220)
HARD = Difficulty("Hard", start_ms=550, per_level_ms=-40, minimum_ms=160)


def difficulty_by_name(name: str) -> Difficulty:
    lookup = {
        EASY.name.lower(): EASY,
        NORMAL.name.lower(): NORMAL,
        HARD.name.lower(): HARD,
    }
    try:
        return lookup[name.strip().lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown difficulty: {name}") from exc
