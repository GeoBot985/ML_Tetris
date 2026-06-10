from __future__ import annotations


LINE_CLEAR_SCORES = {1: 100, 2: 300, 3: 500, 4: 800}


def line_clear_score(lines: int, level: int) -> int:
    return LINE_CLEAR_SCORES.get(lines, 0) * level


def soft_drop_score(cells: int) -> int:
    return cells


def hard_drop_score(cells: int) -> int:
    return cells * 2
