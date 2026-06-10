from __future__ import annotations

from dataclasses import dataclass
from typing import Any


LINE_CLEAR_REWARDS = {
    1: 1.0,
    2: 2.75,
    3: 5.5,
    4: 10.0,
}

LINE_CLEAR_COMPLEXITY_REWARDS = {
    1: 0.25,
    2: 0.85,
    3: 1.75,
    4: 4.0,
}

# Per-step penalties applied to every board state
STACK_HEIGHT_WEIGHT = 0.04       # penalise max stack height
STACK_RISK_THRESHOLD = 5         # start discouraging runaway stacks at this height
STACK_RISK_WEIGHT = 0.12         # small extra penalty per risky stack row
STACK_REDUCTION_REWARD_WEIGHT = 0.18  # reward lowering the highest column
STACK_RISK_REDUCTION_BONUS_WEIGHT = 0.12  # extra reward for escaping risky height
AGGREGATE_HEIGHT_WEIGHT = 0.002  # penalise total height across all columns
HOLE_PENALTY_WEIGHT = 5.0        # strong penalty; holes undo multiple line-clear rewards
ROW_HOLE_PENALTY_WEIGHT = 0.35   # penalise rows damaged by one or more covered holes
BUMPINESS_WEIGHT = 0.18          # reward smooth surface
WELL_WEIGHT = 0.12               # penalise very deep single-column wells
GAME_OVER_PENALTY = 15.0
PAUSE_PENALTY = 0.05


@dataclass(frozen=True)
class BoardProfile:
    width: int
    height: int
    column_heights: tuple[int, ...]
    stack_height: int
    aggregate_height: int
    hole_count: int
    row_hole_count: int
    hole_density: float
    bumpiness: int
    max_well_depth: int


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    line_clear_reward: float
    line_clear_complexity_reward: float
    stack_reduction_reward: float
    stack_height_penalty: float
    stack_risk_penalty: float
    aggregate_height_penalty: float
    hole_penalty: float
    row_hole_penalty: float
    bumpiness_penalty: float
    well_penalty: float
    pause_penalty: float
    game_over_penalty: float
    lines_cleared_delta: int
    profile: BoardProfile


def _board_rows(snapshot: dict[str, Any]) -> list[list[str | None]]:
    board = snapshot.get("locked_board") or snapshot.get("board") or []
    return [list(row) for row in board]


def profile_board(snapshot: dict[str, Any]) -> BoardProfile:
    rows = _board_rows(snapshot)
    height = len(rows)
    width = len(rows[0]) if rows else 0
    column_heights: list[int] = []
    hole_count = 0
    hole_rows: set[int] = set()

    for x in range(width):
        seen_block = False
        col_height = 0
        for y in range(height):
            cell = rows[y][x]
            if cell is not None:
                if not seen_block:
                    col_height = height - y
                seen_block = True
            elif seen_block:
                hole_count += 1
                hole_rows.add(y)
        column_heights.append(col_height)

    stack_height = max(column_heights, default=0)
    aggregate_height = sum(column_heights)
    hole_density = float(hole_count) / float(width * height) if width and height else 0.0

    bumpiness = 0
    for i in range(len(column_heights) - 1):
        bumpiness += abs(column_heights[i] - column_heights[i + 1])

    # Max well depth: how deep any column is relative to both neighbours
    max_well_depth = 0
    for i, h in enumerate(column_heights):
        left = column_heights[i - 1] if i > 0 else h
        right = column_heights[i + 1] if i < len(column_heights) - 1 else h
        well_depth = min(left, right) - h
        if well_depth > 0:
            max_well_depth = max(max_well_depth, well_depth)

    return BoardProfile(
        width=width,
        height=height,
        column_heights=tuple(column_heights),
        stack_height=stack_height,
        aggregate_height=aggregate_height,
        hole_count=hole_count,
        row_hole_count=len(hole_rows),
        hole_density=hole_density,
        bumpiness=bumpiness,
        max_well_depth=max_well_depth,
    )


def _line_clear_reward(previous_snapshot: dict[str, Any], current_snapshot: dict[str, Any]) -> tuple[int, float, float]:
    previous_lines = int(previous_snapshot.get("lines_cleared", 0))
    current_lines = int(current_snapshot.get("lines_cleared", 0))
    cleared = max(0, current_lines - previous_lines)
    if cleared == 0:
        return 0, 0.0, 0.0
    reward = LINE_CLEAR_REWARDS.get(cleared, LINE_CLEAR_REWARDS[4] * (cleared / 4.0))
    complexity_reward = LINE_CLEAR_COMPLEXITY_REWARDS.get(
        cleared,
        LINE_CLEAR_COMPLEXITY_REWARDS[4] * (cleared / 4.0),
    )
    return cleared, float(reward), float(complexity_reward)


def calculate_reward(previous_snapshot: dict[str, Any], current_snapshot: dict[str, Any]) -> RewardBreakdown:
    previous_profile = profile_board(previous_snapshot)
    profile = profile_board(current_snapshot)
    lines_cleared_delta, line_clear_reward, line_clear_complexity_reward = _line_clear_reward(previous_snapshot, current_snapshot)

    stack_reduction = max(0, previous_profile.stack_height - profile.stack_height)
    previous_risk_height = max(0, previous_profile.stack_height - STACK_RISK_THRESHOLD + 1)
    current_risk_height = max(0, profile.stack_height - STACK_RISK_THRESHOLD + 1)
    stack_risk_reduction = max(0, previous_risk_height - current_risk_height)
    stack_reduction_reward = (
        stack_reduction * STACK_REDUCTION_REWARD_WEIGHT
        + stack_risk_reduction * STACK_RISK_REDUCTION_BONUS_WEIGHT
    )
    stack_height_penalty = profile.stack_height * STACK_HEIGHT_WEIGHT
    stack_risk_penalty = max(0, profile.stack_height - STACK_RISK_THRESHOLD + 1) * STACK_RISK_WEIGHT
    aggregate_height_penalty = profile.aggregate_height * AGGREGATE_HEIGHT_WEIGHT
    hole_penalty = profile.hole_count * HOLE_PENALTY_WEIGHT * (1.0 / max(1, profile.width * profile.height))
    row_hole_penalty = profile.row_hole_count * ROW_HOLE_PENALTY_WEIGHT
    bumpiness_penalty = profile.bumpiness * BUMPINESS_WEIGHT
    well_penalty = profile.max_well_depth * WELL_WEIGHT
    pause_penalty = PAUSE_PENALTY if current_snapshot.get("paused") else 0.0
    game_over_penalty = GAME_OVER_PENALTY if current_snapshot.get("game_over") else 0.0

    total = (
        line_clear_reward
        + line_clear_complexity_reward
        + stack_reduction_reward
        - stack_height_penalty
        - stack_risk_penalty
        - aggregate_height_penalty
        - hole_penalty
        - row_hole_penalty
        - bumpiness_penalty
        - well_penalty
        - pause_penalty
        - game_over_penalty
    )
    return RewardBreakdown(
        total=float(total),
        line_clear_reward=float(line_clear_reward),
        line_clear_complexity_reward=float(line_clear_complexity_reward),
        stack_reduction_reward=float(stack_reduction_reward),
        stack_height_penalty=float(stack_height_penalty),
        stack_risk_penalty=float(stack_risk_penalty),
        aggregate_height_penalty=float(aggregate_height_penalty),
        hole_penalty=float(hole_penalty),
        row_hole_penalty=float(row_hole_penalty),
        bumpiness_penalty=float(bumpiness_penalty),
        well_penalty=float(well_penalty),
        pause_penalty=float(pause_penalty),
        game_over_penalty=float(game_over_penalty),
        lines_cleared_delta=lines_cleared_delta,
        profile=profile,
    )
